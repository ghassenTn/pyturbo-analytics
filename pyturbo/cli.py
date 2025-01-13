"""Command-line interface for PyTurbo."""

import argparse
import sys
import time
from typing import List, Optional

from . import __version__
from .core import TurboFrame
from .config import get_device_info, set_num_threads

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the PyTurbo CLI."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="PyTurbo - High-performance data analysis tool"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"PyTurbo {__version__}"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Input file to process (CSV, Parquet, etc.)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available"
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.info:
        device_info = get_device_info()
        print("\nSystem Information:")
        print(f"CPUs available: {device_info['num_cpus']}")
        print(f"GPUs available: {device_info['num_gpus']}")
        if device_info['gpu_info']:
            for gpu_id, info in device_info['gpu_info'].items():
                print(f"\nGPU {gpu_id}:")
                print(f"  Total memory: {info['total_memory'] / 1e9:.2f} GB")
                print(f"  Free memory: {info['free_memory'] / 1e9:.2f} GB")
        return 0

    if parsed_args.threads:
        set_num_threads(parsed_args.threads)

    if parsed_args.file:
        # Simple file processing example
        start_time = time.time()
        df = TurboFrame.from_csv(parsed_args.file, use_gpu=parsed_args.use_gpu)
        print(f"\nLoaded {len(df)} rows in {time.time() - start_time:.2f} seconds")
        print("\nDataset Overview:")
        print(df.data.describe())

    return 0

def profile(args: Optional[List[str]] = None) -> int:
    """Profile data processing operations."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Profile PyTurbo operations"
    )
    parser.add_argument(
        "file",
        help="Input file to profile"
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["groupby", "sort", "merge"],
        help="Operations to profile"
    )

    parsed_args = parser.parse_args(args)
    
    print(f"Profiling {parsed_args.file} with operations: {parsed_args.operations}")
    # Add profiling logic here
    return 0

def benchmark(args: Optional[List[str]] = None) -> int:
    """Benchmark PyTurbo against other libraries."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Benchmark PyTurbo against other libraries"
    )
    parser.add_argument(
        "file",
        help="Input file to benchmark"
    )
    parser.add_argument(
        "--compare-with",
        nargs="+",
        default=["pandas"],
        help="Libraries to compare with"
    )

    parsed_args = parser.parse_args(args)
    
    print(f"Benchmarking {parsed_args.file} against: {parsed_args.compare_with}")
    # Add benchmarking logic here
    return 0

if __name__ == "__main__":
    sys.exit(main())
