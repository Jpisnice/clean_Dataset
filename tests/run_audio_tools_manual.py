import sys
from pathlib import Path

# Ensure repo root is on sys.path when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_audio_tools import test_process_audio_various_inputs, test_process_batch_handles_mixed_inputs


if __name__ == '__main__':
    print('Running audio_tools manual tests...')
    test_process_audio_various_inputs()
    test_process_batch_handles_mixed_inputs()
    print('All manual tests passed')
