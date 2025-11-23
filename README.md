# Vaani Konkani ASR Dataset Preparation

A data preparation pipeline for processing the **Vaani Konkani** dataset from HuggingFace into a standardized Parquet format optimized for ASR training.

## Overview

This pipeline processes the [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) dataset, specifically the Konkani language subset filtered by state (default: Goa). It converts raw audio and transcripts into a standardized format with:

- **Text Cleaning**: Removes English translations (`{watchman}`) and special tokens (`<pause>`)
- **Audio Processing**: Resamples to 16kHz mono and compresses to FLAC
- **State Filtering**: Filters samples by state (default: Goa)
- **Partitioned Storage**: Organized by `corpus/split/language` for efficient loading

## Installation

Install dependencies using `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Quick Start

### 1. Get HuggingFace Token

This is a **gated dataset** that requires authentication:

1. Visit https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part
2. Click "Access repository" and accept the terms
3. Get your token from https://huggingface.co/settings/tokens
4. Copy the token (starts with `hf_`)
cmd to run: uv run process_vaani.py ./output_data --hf-token hf_xxxxxxxxxxxxxxxxxxxxxxxxx --state Goa --max-samples 10 --use-ray

### 2. Test Text Cleaning

First, verify the text cleaning works correctly:

```bash
uv run process_vaani.py ./output_data --test-cleaning
```

This will show examples of how transcripts are cleaned (removing `{english}` translations and `<pause>` tokens).

### 3. Process Dataset

**Quick test (10 samples per split):**
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
uv run process_vaani.py ./output_data --max-samples 10 --use-ray
```

**Process full dataset:**
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
uv run process_vaani.py ./output_data --use-ray
```

**Alternative: Pass token directly**
```bash
uv run process_vaani.py ./output_data --hf-token hf_YOUR_TOKEN_HERE --use-ray
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `output_dir` | Yes | - | Output directory for parquet files |
| `--hf-token` | No* | - | HuggingFace token (or set `HF_TOKEN` env var) |
| `--state` | No | `Goa` | State to filter for (e.g., `Goa`, `Karnataka`) |
| `--max-samples` | No | - | Limit samples per split (for testing) |
| `--use-ray` | No | - | Use Ray for faster distributed processing |
| `--test-cleaning` | No | - | Test text cleaning only (no token needed) |

\* Required for actual processing (not needed for `--test-cleaning`)

## Text Cleaning

The pipeline automatically cleans transcripts by removing:

1. **English translations in curly braces**: `{watchman}`, `{side}`, `{blue color shirt}`
2. **Special tokens**: `<pause>`, `</pause>`, `<noise>`, etc.
3. **Extra punctuation**: `--` sequences

### Examples

```
Before: एक वॉचमन {watchman} असा तचा साइडीन {side} एक
After:  एक वॉचमन असा तचा साइडीन एक

Before: दोन बुरगे असा एक बुर्ग्यान <pause> ब्लू कलरचो शर्ट {blue color shirt} घा --</pause>
After:  दोन बुरगे असा एक बुर्ग्यान ब्लू कलरचो शर्ट घा
```

## Output Format

### Directory Structure

```
output_data/version=0/
└── corpus=vaani/
    ├── split=train/
    │   └── language=kok_Deva/
    │       └── part-0.parquet
    ├── split=validation/
    │   └── language=kok_Deva/
    │       └── part-0.parquet
    └── split=test/
        └── language=kok_Deva/
            └── part-0.parquet
```

**Language Code**: `kok_Deva` = Konkani in Devanagari script

### Schema

Each Parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Cleaned Devanagari transcription |
| `audio_bytes` | list\<int8\> | Compressed FLAC audio as bytes |
| `audio_size` | int64 | Decoded waveform size (samples at 16kHz) |
| `corpus` | dictionary | Dataset name: `"vaani"` |
| `split` | dictionary | Split: `"train"`, `"validation"`, or `"test"` |
| `language` | dictionary | Language code: `"kok_Deva"` |

**Duration calculation**: `duration_seconds = audio_size / 16_000`

## Usage Examples

### Load Processed Data

```python
import polars as pl

# Load training data
df = pl.read_parquet(
    "output_data/version=0/corpus=vaani/split=train/language=kok_Deva/"
)

print(f"Samples: {len(df):,}")
print(f"Duration: {df['audio_size'].sum() / 16_000 / 3600:.2f} hours")
print(df.head())
```

### Decode Audio

```python
from audio_tools import bytes_to_tensor

# Get audio bytes from parquet
audio_bytes = df['audio_bytes'][0]
text = df['text'][0]

# Convert to numpy waveform
waveform = bytes_to_tensor(bytes(audio_bytes))

print(f"Text: {text}")
print(f"Audio shape: {waveform.shape}")
print(f"Duration: {len(waveform) / 16_000:.2f} seconds")
```

### Load All Splits

```python
import polars as pl

# Load all splits
df = pl.scan_parquet("output_data/version=0/**/*.parquet").collect()

# Filter by split
train_df = df.filter(pl.col("split") == "train")
val_df = df.filter(pl.col("split") == "validation")
test_df = df.filter(pl.col("split") == "test")

print(f"Train: {len(train_df):,} samples")
print(f"Validation: {len(val_df):,} samples")
print(f"Test: {len(test_df):,} samples")
```

## View Statistics

After processing, view dataset statistics:

```bash
uv run dataset_ingestion.py stats ./output_data/version=0
```

This shows:
- Number of samples per split
- Total duration in hours
- Breakdown by corpus and language

## Test Audio Playback

To verify the processed audio data, you can play individual samples:

```bash
uv run .\play_parquet_audio.py "output_data\version=0\corpus=vaani\split=train\language=gom_Deva\part-0.parquet" --row 0
```

This will:
- Decode the audio from the specified row
- Play it through your default audio output
- Show the duration and sample rate

**Note**: The script requires `sounddevice` for audio playback, which is automatically installed with the project dependencies.

## Project Structure

```
.
├── process_vaani.py          # Main processing script
├── vaani_text_cleaner.py    # Vaani-specific text cleaning
├── audio_tools.py           # Audio processing utilities
├── text_tools.py            # General text normalization
├── dataset_ingestion.py     # Core pipeline implementation
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Troubleshooting

### "No HuggingFace token provided"

Make sure to set the token:
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Or pass it directly:
```bash
uv run process_vaani.py ./data --hf-token hf_YOUR_TOKEN_HERE
```

### "No samples found for state 'Goa'"

Check available states in the dataset:
```python
from datasets import load_dataset

ds = load_dataset("ARTPARK-IISc/Vaani-transcription-part", "Konkani", 
                  split="train", token="YOUR_TOKEN")
print("Available states:", set(ds['state']))
```

Then use a different state:
```bash
uv run process_vaani.py ./data --state Karnataka --hf-token hf_YOUR_TOKEN
```

### Ray Memory Issues

If Ray runs out of memory, process without it:
```bash
uv run process_vaani.py ./data --hf-token hf_YOUR_TOKEN
# (omit --use-ray flag)
```

### Slow Processing

- Use `--use-ray` for faster processing (3-5x speedup on multi-core)
- Process fewer samples with `--max-samples` for testing
- Check network speed (dataset downloads from HuggingFace)

## Performance

Expected processing times:

| Dataset Size | Samples | Time (with Ray) |
|-------------|---------|-----------------|
| Test (10/split) | ~30 | 1-2 minutes |
| Small (100/split) | ~300 | 5-10 minutes |
| Full Goa subset | ~5,000 | 10-30 minutes |

Times depend on:
- Number of CPU cores
- Network speed (for downloading)
- Whether Ray is used

## References

- **Dataset**: [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part)
- **Dataset Viewer**: [Konkani Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part/viewer/Konkani)
- **Based on**: [Facebook Research's omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)

## License

This implementation follows the same spirit as the original omnilingual-asr project.

---

**Ready to process your Vaani Konkani dataset?** Start with `--test-cleaning` to verify everything works! 🚀
# clean_Dataset
