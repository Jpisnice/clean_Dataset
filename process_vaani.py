"""
Process Vaani Konkani dataset from HuggingFace.

Dataset: ARTPARK-IISc/Vaani-transcription-part
Language: Konkani (filtered for Goa state)
Structure:
  - audio: STRUCT(bytes, path)
  - language: string
  - gender: string
  - state: string (filter for 'Goa')
  - district: string
  - transcript: string
  - referenceImage: string
"""

import ray
from datasets import load_dataset
from dataset_ingestion import DatasetIngestionPipeline, compute_dataset_statistics
from vaani_text_cleaner import normalize_vaani_transcript


def process_vaani_konkani(
    output_dir: str,
    filter_state: str = "Goa",
    max_samples: int = None,
    use_ray: bool = True,
    hf_token: str = None
):
    """
    Process Vaani Konkani dataset with Goa state filter.
    
    Args:
        output_dir: Output directory for parquet files
        filter_state: State to filter for (default: 'Goa')
        max_samples: Maximum samples per split (for testing)
        use_ray: Whether to use Ray for processing
        hf_token: HuggingFace token for gated dataset access
    """
    print("="*80)
    print("VAANI KONKANI DATASET PROCESSING")
    print("="*80)
    print(f"Dataset: ARTPARK-IISc/Vaani-transcription-part")
    print(f"Language: Konkani")
    print(f"Filter: state = '{filter_state}'")
    print(f"Splits: train, validation, test")
    print("="*80 + "\n")
    
    # Check for HuggingFace token
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token provided!")
        print("This is a GATED dataset and requires authentication.")
        print("\nTo get access:")
        print("  1. Go to: https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part")
        print("  2. Click 'Access repository' and accept the terms")
        print("  3. Get your token from: https://huggingface.co/settings/tokens")
        print("  4. Run with: HF_TOKEN=your_token uv run process_vaani.py ...")
        print("     OR use: --hf-token your_token")
        print()
        return
    
    if use_ray:
        ray.init(ignore_reinit_error=True)
    
    # Create custom pipeline
    pipeline = VaaniDatasetPipeline(
        output_dir=output_dir,
        name="vaani_konkani",
        version=0,
        hf_token=hf_token
    )
    
    results = []
    splits_map = {
        "train": "train",
        "validation": "validation", 
        "test": "test"
    }
    
    for hf_split, our_split in splits_map.items():
        print(f"\n{'='*80}")
        print(f"Processing split: {hf_split}")
        print(f"{'='*80}")
        
        result = pipeline.process_vaani_dataset(
            dataset_name="ARTPARK-IISc/Vaani-transcription-part",
            dataset_config="Konkani",
            corpus_name="vaani",
            language_code="gom_Deva",  # Konkani in Devanagari script
            split=hf_split,
            filter_state=filter_state,
            max_samples=max_samples,
            use_ray=use_ray
        )
        
        if result:
            results.append(result)
    
    if use_ray:
        ray.shutdown()
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"{'Corpus':<15} | {'Language':<12} | {'Split':<12} | {'Samples':>10} | {'Hours':>8}")
        print("-"*80)
        
        for r in results:
            print(f"{r['corpus']:<15} | {r['language']:<12} | {r['split']:<12} | "
                  f"{r['num_samples']:>10,} | {r['duration_hours']:>8.2f}")
        
        total_samples = sum(r['num_samples'] for r in results)
        total_hours = sum(r['duration_hours'] for r in results)
        
        print("="*80)
        print(f"TOTAL: {total_samples:,} samples | {total_hours:.2f} hours")
        print("="*80)
        
        # Compute statistics
        print("\nComputing final statistics...")
        compute_dataset_statistics(f"{output_dir}/version=0")
    else:
        print("\n⚠️  No data was processed. Check filters and dataset availability.")


class VaaniDatasetPipeline(DatasetIngestionPipeline):
    """Extended pipeline for Vaani dataset with custom filtering and cleaning."""
    
    def __init__(self, hf_token: str = None, **kwargs):
        """
        Initialize with HuggingFace token support.
        
        Args:
            hf_token: HuggingFace authentication token
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.hf_token = hf_token
    
    def process_vaani_dataset(
        self,
        dataset_name: str,
        dataset_config: str,
        corpus_name: str,
        language_code: str,
        split: str,
        filter_state: str = "Goa",
        max_samples: int = None,
        use_ray: bool = True
    ):
        """
        Process Vaani dataset with state filtering and text cleaning.
        
        Args:
            dataset_name: HuggingFace dataset ID
            dataset_config: Dataset config (language name)
            corpus_name: Corpus name for output
            language_code: Standardized language code
            split: Dataset split
            filter_state: State to filter for
            max_samples: Max samples to process
            use_ray: Use Ray for processing
        """
        from datasets import Audio
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        print(f"Loading dataset: {dataset_name} (config: {dataset_config}, split: {split})")
        
        # Load dataset
        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                token=self.hf_token  # Use token for gated dataset access
            )
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("\nIf this is an authentication error, make sure you:")
            print("  1. Have accepted the dataset terms on HuggingFace")
            print("  2. Provided a valid token with --hf-token or HF_TOKEN env variable")
            return None
        
        print(f"Loaded {len(dataset)} samples")
        
        # Filter by state (use batched filtering to avoid audio decoding)
        print(f"Filtering for state = '{filter_state}'...")
        
        def filter_by_state(states):
            """Filter function that works on batches and avoids audio decoding."""
            return [state == filter_state for state in states]
        
        dataset = dataset.filter(
            filter_by_state, 
            batched=True,
            input_columns=['state']  # Only access 'state' column, not audio
        )
        print(f"After filtering: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print(f"⚠️  No samples found for state '{filter_state}'")
            return None
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples for testing")
        
        # Cast audio column
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sample_rate))
        
        # Process batches
        def process_batch(batch):
            # Clean transcripts first
            cleaned_texts = []
            for text in batch['transcript']:
                cleaned = normalize_vaani_transcript(text, remove_punctuation=False)
                cleaned_texts.append(cleaned if cleaned else "")
            
            # Process audio
            batch_audio = {
                'audio': batch['audio']
            }
            batch_audio = self.audio_processor.process_batch(batch_audio, audio_column='audio')
            
            # Create target schema
            result = {
                'text': cleaned_texts,
                'audio_bytes': batch_audio['audio_bytes'],
                'audio_size': batch_audio['audio_size'],
                'corpus': [corpus_name] * len(cleaned_texts),
                'split': [split] * len(cleaned_texts),
                'language': [language_code] * len(cleaned_texts),
            }
            
            return result
        
        # Process with Ray or standard mapping
        if use_ray and ray.is_initialized():
            print("Processing with Ray...")
            ray_dataset = ray.data.from_huggingface(dataset)
            ray_dataset = ray_dataset.map_batches(process_batch, batch_size=100)
            # Ray's Dataset API differs between versions. Some versions provide
            # `to_arrow()` while others expose `to_pandas()` or different methods.
            # Prefer `to_arrow()` when available; fall back to conversion via
            # pandas -> pyarrow to ensure we consistently produce a pyarrow.Table
            # which the downstream code expects.
            if hasattr(ray_dataset, "to_arrow"):
                processed_data = ray_dataset.to_arrow()
            else:
                # Convert to pandas DataFrame first, then to pyarrow Table.
                # Keep preserve_index=False to avoid adding the pandas index.
                try:
                    import pandas as _pd

                    df = ray_dataset.to_pandas()
                    processed_data = pa.Table.from_pandas(df, preserve_index=False)
                except Exception:
                    # As a very last resort, attempt to collect as a list of
                    # dictionaries and build an arrow table directly.
                    collected = ray_dataset.take_all()
                    processed_data = pa.Table.from_pylist(collected)
        else:
            print("Processing without Ray...")
            processed_data = dataset.map(
                process_batch,
                batched=True,
                batch_size=100,
                remove_columns=dataset.column_names,
                desc="Processing batches"
            )
            processed_data = processed_data.data.table
        
        # Convert audio_bytes if needed
        if isinstance(processed_data['audio_bytes'][0], bytes):
            audio_bytes_list = pa.array(
                [list(b) for b in processed_data['audio_bytes']],
                type=pa.list_(pa.int8())
            )
            processed_data = processed_data.set_column(
                processed_data.schema.get_field_index('audio_bytes'),
                'audio_bytes',
                audio_bytes_list
            )
        
        # Filter out empty texts
        mask = pa.compute.greater(pa.compute.utf8_length(processed_data['text']), 0)
        processed_data = processed_data.filter(mask)
        
        print(f"After text cleaning: {len(processed_data)} samples")
        
        if len(processed_data) == 0:
            print("⚠️  No valid samples after processing")
            return None
        
        # Ensure stable string types for merging across splits
        # Some backends may create dictionary-typed columns while others use plain
        # string columns, which later causes pyarrow.Dataset merges to fail.
        # Cast the key text columns to string explicitly so all parquet parts
        # have a consistent schema.
        for col_name in ("corpus", "split", "language", "text"):
            if col_name in processed_data.column_names:
                try:
                    # Build a plain utf8 array from python values to avoid
                    # dictionary-encoded types across different produced files.
                    values = processed_data[col_name].to_pylist()
                    utf8_arr = pa.array([None if v is None else str(v) for v in values], type=pa.string())
                    col_idx = processed_data.schema.get_field_index(col_name)
                    processed_data = processed_data.set_column(col_idx, col_name, utf8_arr)
                except Exception:
                    # If casting fails, attempt a fallback cast (best-effort)
                    try:
                        col_idx = processed_data.schema.get_field_index(col_name)
                        processed_data = processed_data.set_column(
                            col_idx,
                            col_name,
                            pa.compute.cast(processed_data[col_name], pa.string())
                        )
                    except Exception:
                        # Give up silently — compute_dataset_statistics will
                        # surface any remaining incompatibilities.
                        pass

        # Create output directory
        from pathlib import Path
        output_path = (
            Path(self.output_dir) / f"version={self.version}" /
            f"corpus={corpus_name}" /
            f"split={split}" /
            f"language={language_code}"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write to parquet
        output_file = output_path / "part-0.parquet"
        print(f"Writing to: {output_file}")
        
        # Avoid dictionary encoding at write-time to keep schema consistent
        # across all produced parquet files (dictionary columns can differ
        # per-file and break merges in pyarrow when combining parts).
        pq.write_table(
            processed_data,
            output_file,
            row_group_size=self.row_group_size,
            compression='zstd',
            use_dictionary=False
        )
        
        # Compute statistics
        total_duration_hours = sum(processed_data['audio_size'].to_pylist()) / self.target_sample_rate / 3600
        print(f"✓ Written {len(processed_data)} samples ({total_duration_hours:.2f} hours)")
        
        return {
            'corpus': corpus_name,
            'language': language_code,
            'split': split,
            'num_samples': len(processed_data),
            'duration_hours': total_duration_hours
        }


def test_text_cleaning():
    """Test the text cleaning on sample transcripts."""
    print("\n" + "="*80)
    print("TESTING TEXT CLEANING")
    print("="*80 + "\n")
    
    test_examples = [
        "एक वॉचमन {watchman} असा तचा साइडीन {side} एक",
        "दोन बुरगे असा एक बुर्ग्यान <pause> ब्लू कलरचो शर्ट {blue color shirt} घा --</pause>",
        "सामान्य वाक्य बिना कोई विशेष टोकन",
    ]
    
    for i, example in enumerate(test_examples, 1):
        cleaned = normalize_vaani_transcript(example)
        print(f"Example {i}:")
        print(f"  Original: {example}")
        print(f"  Cleaned:  {cleaned}")
        print()


if __name__ == "__main__":
    import sys
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Process Vaani Konkani dataset",
        epilog="""
Examples:
  # Test text cleaning
  uv run process_vaani.py ./data --test-cleaning
  
  # Process with environment variable token
  HF_TOKEN=hf_xxxxxxxxxxxx uv run process_vaani.py ./data --use-ray
  
  # Process with token argument
  uv run process_vaani.py ./data --hf-token hf_xxxxxxxxxxxx --use-ray
  
  # Quick test with 10 samples
  HF_TOKEN=hf_xxxxxxxxxxxx uv run process_vaani.py ./data --max-samples 10 --use-ray
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('output_dir', help='Output directory for parquet files')
    parser.add_argument('--test-cleaning', action='store_true', 
                       help='Test text cleaning only (no HF token needed)')
    parser.add_argument('--state', default='Goa', 
                       help='State to filter for (default: Goa)')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum samples per split (for testing)')
    parser.add_argument('--use-ray', action='store_true',
                       help='Use Ray for faster processing')
    parser.add_argument('--hf-token', type=str,
                       help='HuggingFace token (or set HF_TOKEN env variable)')
    
    args = parser.parse_args()
    
    if args.test_cleaning:
        test_text_cleaning()
    else:
        # Get token from argument or environment variable
        hf_token = args.hf_token or os.environ.get('HF_TOKEN')
        
        process_vaani_konkani(
            output_dir=args.output_dir,
            filter_state=args.state,
            max_samples=args.max_samples,
            use_ray=args.use_ray,
            hf_token=hf_token
        )

