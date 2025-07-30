#!/usr/bin/env python3
"""
Dataset Downloader Script

This script allows you to download specific parquet datasets from the UFGVC collection.
Usage:
    python download_dataset.py --dataset cotton80 --output ./data
    python download_dataset.py --list  # List all available datasets
"""

import os
import argparse
import requests
from pathlib import Path
from typing import Dict, Any
import sys

class DatasetDownloader:
    """A utility class for downloading UFGVC datasets in parquet format"""
    
    # Available datasets configuration (same as in ufgvc.py)
    DATASETS = {
        'cotton80': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/cotton80_dataset.parquet?download=true',
            'filename': 'cotton80_dataset.parquet',
            'description': 'Cotton classification dataset with 80 classes',
            'size': '~50MB'
        },
        'soybean': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soybean_dataset.parquet?download=true',
            'filename': 'soybean_dataset.parquet',
            'description': 'Soybean classification dataset',
            'size': '~150MB'
        },
        'soy_ageing_r1': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R1_dataset.parquet?download=true',
            'filename': 'soy_ageing_R1_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 1',
            'size': '~100MB'
        },
        'soy_ageing_r3': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R3_dataset.parquet?download=true',
            'filename': 'soy_ageing_R3_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 3',
            'size': '~100MB'
        },
        'soy_ageing_r4': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R4_dataset.parquet?download=true',
            'filename': 'soy_ageing_R4_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 4',
            'size': '~100MB'
        },
        'soy_ageing_r5': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R5_dataset.parquet?download=true',
            'filename': 'soy_ageing_R5_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 5',
            'size': '~100MB'
        },
        'soy_ageing_r6': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R6_dataset.parquet?download=true',
            'filename': 'soy_ageing_R6_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 6',
            'size': '~100MB'
        }
    }
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize the downloader
        
        Args:
            output_dir (str): Directory where datasets will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> None:
        """List all available datasets"""
        print("Available datasets:")
        print("-" * 80)
        for name, config in self.DATASETS.items():
            print(f"Dataset: {name}")
            print(f"  Description: {config['description']}")
            print(f"  File: {config['filename']}")
            print(f"  Size: {config.get('size', 'Unknown')}")
            print()
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> str:
        """
        Download a specific dataset
        
        Args:
            dataset_name (str): Name of the dataset to download
            force (bool): Force download even if file exists
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            ValueError: If dataset name is not found
            RuntimeError: If download fails
        """
        if dataset_name not in self.DATASETS:
            available = list(self.DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
        
        config = self.DATASETS[dataset_name]
        filepath = self.output_dir / config['filename']
        
        # Check if file already exists
        if filepath.exists() and not force:
            print(f"Dataset '{dataset_name}' already exists at: {filepath}")
            print("Use --force to re-download")
            return str(filepath)
        
        print(f"Downloading dataset: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"URL: {config['url']}")
        print(f"Saving to: {filepath}")
        print("-" * 50)
        
        try:
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Show progress
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            progress_bar = "█" * int(progress // 2) + "░" * (50 - int(progress // 2))
                            print(f"\rProgress: [{progress_bar}] {progress:.1f}% ({downloaded_size // 1024 // 1024}MB)", end="")
                        else:
                            print(f"\rDownloaded: {downloaded_size // 1024 // 1024}MB", end="")
            
            print(f"\n✅ Download completed successfully!")
            print(f"File saved to: {filepath}")
            print(f"File size: {filepath.stat().st_size / 1024 / 1024:.2f}MB")
            
            return str(filepath)
            
        except requests.RequestException as e:
            if filepath.exists():
                filepath.unlink()  # Remove incomplete file
            raise RuntimeError(f"Failed to download {dataset_name}: {e}")
        except Exception as e:
            if filepath.exists():
                filepath.unlink()  # Remove incomplete file
            raise RuntimeError(f"Unexpected error while downloading {dataset_name}: {e}")
    
    def download_multiple_datasets(self, dataset_names: list, force: bool = False) -> list:
        """
        Download multiple datasets
        
        Args:
            dataset_names (list): List of dataset names to download
            force (bool): Force download even if files exist
            
        Returns:
            list: List of paths to downloaded files
        """
        downloaded_files = []
        
        for dataset_name in dataset_names:
            try:
                filepath = self.download_dataset(dataset_name, force=force)
                downloaded_files.append(filepath)
                print()  # Add spacing between downloads
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")
                print()
        
        return downloaded_files
    
    def download_all_datasets(self, force: bool = False) -> list:
        """
        Download all available datasets
        
        Args:
            force (bool): Force download even if files exist
            
        Returns:
            list: List of paths to downloaded files
        """
        print(f"Downloading all {len(self.DATASETS)} datasets...")
        return self.download_multiple_datasets(list(self.DATASETS.keys()), force=force)


def main():
    """Main function to handle command line interface"""
    parser = argparse.ArgumentParser(
        description="Download UFGVC datasets in parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # List all available datasets
  %(prog)s --dataset cotton80               # Download cotton80 dataset
  %(prog)s --dataset cotton80 --output ./datasets  # Download to specific directory
  %(prog)s --dataset cotton80 --force       # Force re-download
  %(prog)s --dataset cotton80,soybean       # Download multiple datasets
  %(prog)s --all                           # Download all datasets
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset name(s) to download (comma-separated for multiple)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data',
        help='Output directory for downloaded datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available datasets'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all available datasets'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force download even if file already exists'
    )
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = DatasetDownloader(output_dir=args.output)
    
    try:
        # List datasets
        if args.list:
            downloader.list_datasets()
            return
        
        # Download all datasets
        if args.all:
            print("Starting download of all datasets...")
            downloaded_files = downloader.download_all_datasets(force=args.force)
            print(f"\n✅ Successfully downloaded {len(downloaded_files)} datasets")
            return
        
        # Download specific dataset(s)
        if args.dataset:
            dataset_names = [name.strip() for name in args.dataset.split(',')]
            
            if len(dataset_names) == 1:
                filepath = downloader.download_dataset(dataset_names[0], force=args.force)
                print(f"\n✅ Dataset downloaded to: {filepath}")
            else:
                print(f"Starting download of {len(dataset_names)} datasets...")
                downloaded_files = downloader.download_multiple_datasets(dataset_names, force=args.force)
                print(f"\n✅ Successfully downloaded {len(downloaded_files)} datasets")
            return
        
        # No action specified
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
