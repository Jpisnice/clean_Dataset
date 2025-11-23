"""
Small utility to read an `audio_bytes` field from a Parquet file and play it.

Usage examples:
  python play_parquet_audio.py output_data/version=0/corpus=vaani/split=train/language=gom_Deva/part-0.parquet --row 0

Dependencies:
  pip install soundfile sounddevice pyarrow numpy

This script will attempt to use `sounddevice` to play back the decoded audio
waveform (float32). If `sounddevice` is not available, it falls back to
`simpleaudio` which plays 16-bit PCM.
"""

from __future__ import annotations

import io
import argparse
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    import soundfile as sf
except Exception as e:
    print("soundfile is required: pip install soundfile", file=sys.stderr)
    raise


def get_audio_bytes_from_row(table, row_index: int = 0, column_name: str = "audio_bytes") -> bytes:
    """Extract raw bytes from a parquet table row field.

    The parquet `audio_bytes` column in this project is stored as either:
      - raw bytes/binary
      - list<int8>

    This helper tries to handle both forms and returns a bytes object.
    """
    if column_name not in table.column_names:
        raise KeyError(f"Column '{column_name}' not found in parquet file; available: {table.column_names}")

    col = table[column_name]
    # If pyarrow.Table then extract as Python value
    # We'll pull the single row as a Python object
    val = col[row_index].as_py()

    # If it's already bytes/bytearray
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)

    # If it's a list of ints (list[int] or similar) -> convert
    if isinstance(val, (list, tuple)):
        # ensure ints are in 0..255 (or signed -128..127) — use python's bytes
        return bytes(val)

    # Some pyarrow blobs may be returned differently; try a permissive conversion
    try:
        return bytes(val)
    except Exception as e:
        raise ValueError("Could not convert parquet field to bytes") from e


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode compressed audio bytes (FLAC/ogg/wav) to a numpy waveform and sample-rate.

    Uses soundfile under the hood which supports many compressed formats if
    libsndfile supports them.
    """
    bio = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(bio, dtype="float32")
    # If stereo, convert to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sr


def play_waveform(waveform: np.ndarray, samplerate: int):
    """Play waveform numpy array using sounddevice when available, otherwise simpleaudio fallback."""
    try:
        import sounddevice as sd

        print(f"Playing via sounddevice @ {samplerate} Hz — duration {len(waveform)/samplerate:.2f}s")
        sd.play(waveform, samplerate)
        sd.wait()
        return
    except Exception:
        pass

    # Fallback -> simpleaudio (requires int16 PCM)
    try:
        import simpleaudio as sa

        # Convert float32 [-1, 1] to PCM16
        pcm16 = np.clip(waveform, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        print(f"Playing via simpleaudio @ {samplerate} Hz — duration {len(waveform)/samplerate:.2f}s")
        play_obj = sa.play_buffer(pcm16.tobytes(), 1, 2, samplerate)
        play_obj.wait_done()
        return
    except Exception:
        pass

    raise RuntimeError("No audio playback backend available. Install sounddevice or simpleaudio.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Play `audio_bytes` stored in a Parquet row.")
    parser.add_argument("parquet", help="Path to a parquet file (part-0.parquet, etc)")
    parser.add_argument("--row", type=int, default=0, help="Which row to play (default 0)")
    parser.add_argument("--column", default="audio_bytes", help="Column name containing audio bytes (default 'audio_bytes')")
    args = parser.parse_args()

    # Read parquet table (pyarrow) so we can inspect values cleanly
    # Use ParquetFile to read a single file directly without dataset discovery
    parquet_file = pq.ParquetFile(args.parquet)
    table = parquet_file.read()

    try:
        audio_bytes = get_audio_bytes_from_row(table, args.row, args.column)
    except Exception as e:
        print("Error extracting audio bytes:", e, file=sys.stderr)
        return 2

    try:
        waveform, sr = decode_audio(audio_bytes)
    except Exception as e:
        print("Error decoding audio data (not a supported compressed format?):", e, file=sys.stderr)
        return 3

    try:
        play_waveform(waveform, sr)
    except Exception as e:
        print("Playback failed:", e, file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
