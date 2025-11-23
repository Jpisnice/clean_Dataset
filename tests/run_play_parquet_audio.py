"""Tiny runner to demonstrate usage of play_parquet_audio.py

This script won't automatically play audio unless `--play` is passed
so it is safe in CI environments. It reads the provided parquet file and
prints details about the selected row.
"""

import argparse
import os
import sys

from play_parquet_audio import get_audio_bytes_from_row, decode_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="?", default=os.path.join("output_data", "version=0", "corpus=vaani", "split=train", "language=gom_Deva", "part-0.parquet"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--play", action="store_true", help="Actually play the audio (requires sounddevice or simpleaudio)")
    parser.add_argument("--column", default="audio_bytes")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"Parquet file not found: {args.parquet}")
        return 2

    print("Reading:", args.parquet)
    try:
        raw = get_audio_bytes_from_row(__import__('pyarrow').parquet.read_table(args.parquet), args.row, args.column)
    except Exception as e:
        print("ERROR reading audio_bytes:", e, file=sys.stderr)
        return 3

    print(f"Extracted bytes: {len(raw)} bytes")

    try:
        waveform, sr = decode_audio(raw)
    except Exception as e:
        print("ERROR decoding audio (maybe not a supported compressed format):", e, file=sys.stderr)
        return 4

    print(f"Decoded waveform: length={len(waveform)} samples, samplerate={sr}, duration={len(waveform)/sr:.3f}s")

    if args.play:
        from play_parquet_audio import play_waveform
        play_waveform(waveform, sr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
