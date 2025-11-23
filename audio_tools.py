"""
Audio processing utilities for ASR dataset preparation.
Based on omnilingual-asr pipeline.
"""

import io
import logging
import numpy as np
import pyarrow as pa
import soundfile as sf
from typing import Optional, Dict, Any


def binary_to_list_int8(binary_array: pa.BinaryArray) -> pa.ListArray:
    """
    Efficiently converts PyArrow BinaryArray to ListArray of int8.
    This avoids extra copying when converting from pyarrow to pandas.
    
    Args:
        binary_array: PyArrow binary array
        
    Returns:
        ListArray of int8 values
    """
    # Get the buffers from the binary array
    buffers = binary_array.buffers()
    offsets = buffers[1]
    data = buffers[2]
    
    # Create int8 array from the data buffer
    int8_array = pa.Array.from_buffers(
        pa.int8(),
        len(data),
        [None, data]
    )
    
    # Create list array using the offsets
    list_array = pa.ListArray.from_arrays(
        pa.Array.from_buffers(pa.int32(), len(binary_array) + 1, [None, offsets]),
        int8_array
    )
    
    return list_array


def bytes_to_tensor(
    audio_bytes: bytes,
    target_sample_rate: int = 16_000
) -> np.ndarray:
    """
    Converts audio bytes to waveform array.
    
    Args:
        audio_bytes: Compressed audio bytes (FLAC, OGG, etc.)
        target_sample_rate: Target sample rate (default 16kHz)
        
    Returns:
        Numpy array of waveform data
    """
    audio_io = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_io, dtype='float32')
    
    # Convert stereo to mono if needed
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        # Simple resampling (for production, use librosa or torchaudio)
        duration = len(waveform) / sample_rate
        target_length = int(duration * target_sample_rate)
        waveform = np.interp(
            np.linspace(0, len(waveform) - 1, target_length),
            np.arange(len(waveform)),
            waveform
        )
    
    return waveform


class AudioTableProcessor:
    """
    Main class for processing audio data in PyArrow tables.
    Handles resampling, format conversion, and schema transformation.
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16_000,
        target_format: str = "FLAC",
        compression_level: int = 5
    ):
        """
        Args:
            target_sample_rate: Target sample rate for audio (default 16kHz)
            target_format: Target audio format (FLAC or OGG)
            compression_level: Compression level (0-8 for FLAC)
        """
        self.target_sample_rate = target_sample_rate
        self.target_format = target_format.upper()
        self.compression_level = compression_level
        
    def process_audio(self, audio_dict: Dict[str, Any]) -> tuple[bytes, int]:
        """
        Process a single audio sample: resample and compress.
        
        Args:
            audio_dict: Dict with 'array' and 'sampling_rate' keys
            
        Returns:
            Tuple of (compressed_bytes, audio_size)
        """
        # Handle multiple input formats that HuggingFace / Ray may provide
        # Typical shapes:
        #  - {'array': [...], 'sampling_rate': 16000}
        #  - {'path': '/abs/path/to/file.wav'}
        #  - {'bytes': b'...'} or raw bytes
        #  - numpy array (assume sample rate provided separately is missing)

        # Case: raw bytes (compressed audio)
        if isinstance(audio_dict, (bytes, bytearray)):
            audio_io = io.BytesIO(audio_dict)
            waveform, sample_rate = sf.read(audio_io, dtype='float32')
        # Case: dict-like
        elif isinstance(audio_dict, dict):
            if 'array' in audio_dict and 'sampling_rate' in audio_dict:
                waveform = np.array(audio_dict['array'], dtype=np.float32)
                sample_rate = int(audio_dict['sampling_rate'])
            elif 'bytes' in audio_dict:
                audio_io = io.BytesIO(audio_dict['bytes'])
                waveform, sample_rate = sf.read(audio_io, dtype='float32')
            elif 'path' in audio_dict and audio_dict['path']:
                waveform, sample_rate = sf.read(str(audio_dict['path']), dtype='float32')
            else:
                # Try to be permissive: maybe the dict contains numpy types
                try:
                    waveform = np.array(audio_dict)  # will fail for mapping-types
                    sample_rate = self.target_sample_rate
                except Exception as e:
                    raise ValueError(f"Unrecognized audio input format: keys={list(audio_dict.keys())}") from e
        # Case: numpy array given directly - assume target sample rate
        elif isinstance(audio_dict, np.ndarray):
            waveform = np.array(audio_dict, dtype=np.float32)
            sample_rate = self.target_sample_rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_dict)}")
        
        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            duration = len(waveform) / sample_rate
            target_length = int(duration * self.target_sample_rate)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, target_length),
                np.arange(len(waveform)),
                waveform
            )
        
        audio_size = len(waveform)
        
        # Compress to target format
        audio_io = io.BytesIO()
        sf.write(
            audio_io,
            waveform,
            self.target_sample_rate,
            format=self.target_format,
            subtype='PCM_16' if self.target_format == 'FLAC' else None
        )
        audio_bytes = audio_io.getvalue()
        
        return audio_bytes, audio_size
    
    def process_batch(
        self,
        batch: Dict[str, list],
        audio_column: str = "audio"
    ) -> Dict[str, list]:
        """
        Process a batch of audio samples.
        
        Args:
            batch: Dictionary of lists (batch format)
            audio_column: Name of the audio column
            
        Returns:
            Batch with processed audio_bytes and audio_size columns
        """
        audio_bytes_list = []
        audio_size_list = []
        
        for audio_dict in batch[audio_column]:
            try:
                audio_bytes, audio_size = self.process_audio(audio_dict)
                audio_bytes_list.append(audio_bytes)
                audio_size_list.append(audio_size)
            except Exception as e:
                # Use logging to ensure messages are properly handled by the
                # environment; keep fallback behavior of empty audio when a
                # sample fails to process.
                logging.exception("Error processing audio: %s", e)
                # Use empty audio as fallback
                audio_bytes_list.append(b"")
                audio_size_list.append(0)
        
        batch['audio_bytes'] = audio_bytes_list
        batch['audio_size'] = audio_size_list
        
        return batch


def map_to_target_schema(
    batch: Dict[str, list],
    split: str,
    corpus: str
) -> Dict[str, list]:
    """
    Transforms batches to the target schema format.
    
    Args:
        batch: Input batch dictionary
        split: Dataset split (train, dev, test)
        corpus: Corpus name
        
    Returns:
        Batch with target schema columns
    """
    # Add metadata columns
    batch_size = len(next(iter(batch.values())))
    batch['split'] = [split] * batch_size
    batch['corpus'] = [corpus] * batch_size
    
    return batch

