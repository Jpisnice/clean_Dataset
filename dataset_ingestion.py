"""
Dataset ingestion pipeline for ASR data preparation.
Based on omnilingual-asr pipeline from Facebook Research.

This script converts HuggingFace audio datasets into a standardized parquet format
optimized for multilingual ASR training.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from datasets import load_dataset, Audio
import ray
from audio_tools import AudioTableProcessor, binary_to_list_int8
from text_tools import text_normalize


# Target schema for ASR datasets
TARGET_SCHEMA = pa.schema([
    pa.field('text', pa.string()),
    pa.field('audio_bytes', pa.list_(pa.int8())),
    pa.field('audio_size', pa.int64()),
    pa.field('corpus', pa.dictionary(pa.int32(), pa.string())),
    pa.field('split', pa.dictionary(pa.int32(), pa.string())),
    pa.field('language', pa.dictionary(pa.int32(), pa.string())),
])


class DatasetIngestionPipeline:
    """
    Pipeline for ingesting and processing audio datasets.
    """
    
    def __init__(
        self,
        output_dir: str,
        name: str = "asr_dataset",
        version: int = 0,
        target_sample_rate: int = 16_000,
        shuffle_buffer_size: int = 1000,
        row_group_size: int = 100
    ):
        """
        Args:
            output_dir: Root directory for output parquet files
            name: Dataset name
            version: Dataset version
            target_sample_rate: Target audio sample rate (default 16kHz)
            shuffle_buffer_size: Buffer size for shuffling
            row_group_size: Rows per parquet row group
        """
        self.output_dir = Path(output_dir)
        self.name = name
        self.version = version
        self.target_sample_rate = target_sample_rate
        self.shuffle_buffer_size = shuffle_buffer_size
        self.row_group_size = row_group_size
        
        self.audio_processor = AudioTableProcessor(target_sample_rate=target_sample_rate)
        
    def process_dataset(
        self,
        dataset_name: str,
        corpus_name: str,
        language_code: str,
        split: str = "train",
        text_column: str = "transcription",
        audio_column: str = "audio",
        dataset_config: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_ray: bool = True
    ):
        """
        Process a single dataset split.
        
        Args:
            dataset_name: HuggingFace dataset ID
            corpus_name: Name to use for corpus (e.g., 'fleurs', 'mls')
            language_code: Language code (e.g., 'eng_Latn', 'fra_Latn')
            split: Dataset split ('train', 'validation', 'test')
            text_column: Name of the text/transcription column
            audio_column: Name of the audio column
            dataset_config: HuggingFace dataset config/subset name
            max_samples: Maximum samples to process (for testing)
            use_ray: Whether to use Ray for distributed processing
        """
        print(f"\n{'='*60}")
        print(f"Processing: {corpus_name} | {language_code} | {split}")
        print(f"{'='*60}")
        
        # Load dataset
        print(f"Loading dataset from HuggingFace: {dataset_name}")
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Loaded {len(dataset)} samples")
        
        # Cast audio to standard format
        dataset = dataset.cast_column(audio_column, Audio(sampling_rate=self.target_sample_rate))
        
        # Define processing function
        def process_batch(batch):
            # Process audio
            batch = self.audio_processor.process_batch(batch, audio_column=audio_column)
            
            # Normalize text
            normalized_texts = []
            for text in batch[text_column]:
                normalized = text_normalize(
                    text,
                    language_code.split('_')[0],  # Extract base language code
                    lower_case=True,
                    remove_numbers=True,
                    remove_brackets=False
                )
                normalized_texts.append(normalized if normalized else "")
            
            # Create target schema batch
            result = {
                'text': normalized_texts,
                'audio_bytes': batch['audio_bytes'],
                'audio_size': batch['audio_size'],
                'corpus': [corpus_name] * len(normalized_texts),
                'split': [split] * len(normalized_texts),
                'language': [language_code] * len(normalized_texts),
            }
            
            return result
        
        # Process with Ray or standard mapping
        if use_ray and ray.is_initialized():
            print("Processing with Ray...")
            ray_dataset = ray.data.from_huggingface(dataset)
            ray_dataset = ray_dataset.map_batches(process_batch, batch_size=100)
            processed_data = ray_dataset.to_arrow()
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
        
        # Convert audio_bytes from binary to list<int8>
        if isinstance(processed_data['audio_bytes'][0], bytes):
            audio_bytes_binary = pa.array([list(b) for b in processed_data['audio_bytes']], type=pa.list_(pa.int8()))
            processed_data = processed_data.set_column(
                processed_data.schema.get_field_index('audio_bytes'),
                'audio_bytes',
                audio_bytes_binary
            )
        
        # Filter out empty texts
        mask = pa.compute.greater(pa.compute.utf8_length(processed_data['text']), 0)
        processed_data = processed_data.filter(mask)
        
        print(f"After filtering: {len(processed_data)} samples")
        
        # Create output directory
        output_path = (
            self.output_dir / f"version={self.version}" /
            f"corpus={corpus_name}" /
            f"split={split}" /
            f"language={language_code}"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write to parquet
        output_file = output_path / "part-0.parquet"
        print(f"Writing to: {output_file}")
        
        pq.write_table(
            processed_data,
            output_file,
            row_group_size=self.row_group_size,
            compression='zstd',
            use_dictionary=True
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


def compute_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """
    Compute statistics across the entire dataset.
    
    Args:
        dataset_path: Path to the parquet dataset
        
    Returns:
        Dictionary of statistics
    """
    print("\n" + "="*60)
    print("Computing Dataset Statistics")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    stats = []
    
    # Find all parquet files
    parquet_files = list(dataset_path.rglob("*.parquet"))

    for parquet_file in parquet_files:
        # Use Polars for robust reading and automatic dtype handling.
        try:
            df = pl.read_parquet(parquet_file)
        except Exception as e:
            print(f"❌ Failed to read {parquet_file}: {e}")
            continue

        # Extract metadata from path
        parts = parquet_file.parts
        corpus = next(p.split('=')[1] for p in parts if p.startswith('corpus='))
        split = next(p.split('=')[1] for p in parts if p.startswith('split='))
        language = next(p.split('=')[1] for p in parts if p.startswith('language='))

        # Ensure required columns exist and coerce types to avoid inconsistencies
        if 'audio_size' not in df.columns:
            print(f"⚠️  Skipping file with missing audio_size: {parquet_file}")
            continue

        # Cast text-like columns to Utf8 to avoid dictionary/encoding mismatches
        for c in ('corpus', 'split', 'language', 'text'):
            if c in df.columns:
                try:
                    df = df.with_columns(pl.col(c).cast(pl.Utf8))
                except Exception:
                    # ignore and continue; we only need audio_size for duration stats
                    pass

        num_samples = df.shape[0]
        total_audio_size = int(df['audio_size'].sum())
        duration_hours = total_audio_size / 16_000 / 3600

        stats.append({
            'corpus': corpus,
            'language': language,
            'split': split,
            'num_samples': num_samples,
            'duration_hours': duration_hours
        })

        print(f"{corpus:15} | {language:15} | {split:10} | {num_samples:8,} samples | {duration_hours:8.2f} hours")
    
    return stats


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(description="ASR Dataset Ingestion Pipeline")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a dataset')
    process_parser.add_argument('dataset_name', help='HuggingFace dataset ID')
    process_parser.add_argument('output_dir', help='Output directory')
    process_parser.add_argument('--corpus', required=True, help='Corpus name')
    process_parser.add_argument('--language', required=True, help='Language code')
    process_parser.add_argument('--split', default='train', help='Dataset split')
    process_parser.add_argument('--config', help='Dataset config/subset')
    process_parser.add_argument('--text-column', default='transcription', help='Text column name')
    process_parser.add_argument('--audio-column', default='audio', help='Audio column name')
    process_parser.add_argument('--max-samples', type=int, help='Max samples to process')
    process_parser.add_argument('--name', default='asr_dataset', help='Dataset name')
    process_parser.add_argument('--version', type=int, default=0, help='Dataset version')
    process_parser.add_argument('--use-ray', action='store_true', help='Use Ray for processing')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Compute dataset statistics')
    stats_parser.add_argument('dataset_path', help='Path to parquet dataset')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        # Initialize Ray if requested
        if args.use_ray:
            if not ray.is_initialized():
                ray.init()
        
        # Create pipeline
        pipeline = DatasetIngestionPipeline(
            output_dir=args.output_dir,
            name=args.name,
            version=args.version
        )
        
        # Process dataset
        result = pipeline.process_dataset(
            dataset_name=args.dataset_name,
            corpus_name=args.corpus,
            language_code=args.language,
            split=args.split,
            text_column=args.text_column,
            audio_column=args.audio_column,
            dataset_config=args.config,
            max_samples=args.max_samples,
            use_ray=args.use_ray
        )
        
        print("\n✓ Processing complete!")
        print(f"  Samples: {result['num_samples']:,}")
        print(f"  Duration: {result['duration_hours']:.2f} hours")
        
        if args.use_ray:
            ray.shutdown()
    
    elif args.command == 'stats':
        stats = compute_dataset_statistics(args.dataset_path)
        
        total_samples = sum(s['num_samples'] for s in stats)
        total_hours = sum(s['duration_hours'] for s in stats)
        
        print("\n" + "="*60)
        print(f"Total: {total_samples:,} samples | {total_hours:.2f} hours")
        print("="*60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

