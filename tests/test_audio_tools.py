import io
import numpy as np
import soundfile as sf

from audio_tools import AudioTableProcessor


def make_sine(duration_secs=0.25, sr=16000, freq=440.0):
    t = np.linspace(0, duration_secs, int(sr * duration_secs), endpoint=False)
    return 0.1 * np.sin(2 * np.pi * freq * t).astype('float32')


def test_process_audio_various_inputs():
    proc = AudioTableProcessor(target_sample_rate=16000)

    waveform = make_sine(duration_secs=0.2, sr=16000)

    # Create compressed bytes (FLAC)
    audio_io = io.BytesIO()
    sf.write(audio_io, waveform, 16000, format='FLAC', subtype='PCM_16')
    compressed = audio_io.getvalue()

    # 1) dict with array and sampling_rate
    obj1 = {'array': waveform.tolist(), 'sampling_rate': 16000}
    b1, s1 = proc.process_audio(obj1)
    assert isinstance(b1, (bytes, bytearray)) and len(b1) > 0
    assert s1 > 0

    # 2) dict with bytes
    obj2 = {'bytes': compressed}
    b2, s2 = proc.process_audio(obj2)
    assert isinstance(b2, (bytes, bytearray)) and len(b2) > 0
    assert s2 > 0

    # 3) raw bytes
    b3, s3 = proc.process_audio(compressed)
    assert isinstance(b3, (bytes, bytearray)) and len(b3) > 0
    assert s3 > 0

    # 4) numpy array directly
    b4, s4 = proc.process_audio(waveform)
    assert isinstance(b4, (bytes, bytearray)) and len(b4) > 0
    assert s4 > 0


def test_process_batch_handles_mixed_inputs():
    proc = AudioTableProcessor(target_sample_rate=16000)
    waveform = make_sine(duration_secs=0.1)

    audio_io = io.BytesIO()
    sf.write(audio_io, waveform, 16000, format='FLAC', subtype='PCM_16')
    compressed = audio_io.getvalue()

    batch_in = {
        'audio': [
            {'array': waveform.tolist(), 'sampling_rate': 16000},
            {'bytes': compressed},
            compressed,
            waveform
        ]
    }

    out = proc.process_batch(batch_in, audio_column='audio')
    assert 'audio_bytes' in out and 'audio_size' in out
    assert len(out['audio_bytes']) == 4
    assert all(isinstance(x, (bytes, bytearray)) for x in out['audio_bytes'])
    assert all(isinstance(x, int) for x in out['audio_size'])