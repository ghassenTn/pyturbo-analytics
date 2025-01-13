# PyTurbo Analytics: High-Performance Data Analysis Library 

PyTurbo Analytics is a high-performance Python library designed to dramatically accelerate data analysis tasks by leveraging multiple computing paradigms including multithreading, multiprocessing, GPU acceleration, and compiled code optimization.

## Features

- **Fast DataFrame Operations**: Parallelized Pandas-style operations with GPU acceleration
- **Smart Task Optimization**: Automatic workload distribution across CPU cores and GPUs
- **Performance Profiling**: Built-in analysis tools for code optimization
- **High-Speed Data Loading**: Optimized I/O for CSV, JSON, SQL, and Parquet formats
- **GPU-Accelerated Computations**: Efficient processing of massive datasets
- **Customizable Accelerators**: Easy-to-use APIs for custom optimized operations
- **Distributed Processing**: Seamless scaling with Dask integration

## Installation

Basic installation:
```bash
pip install pyturbo-analytics
```

With GPU support:
```bash
pip install pyturbo-analytics[gpu]
```

For development installation:
```bash
git clone https://github.com/ghassenTn/pyturbo-analytics.git
cd pyturbo-analytics
pip install -e ".[dev]"
```

## Quick Start

```python
import pyturbo as pt

# Create a TurboFrame (high-performance DataFrame)
tf = pt.TurboFrame.from_csv("large_dataset.csv")

# Perform accelerated operations
result = tf.groupby("category").agg({
    "value": ["mean", "sum", "count"]
}).compute()

# Use GPU acceleration (if installed with [gpu] option)
with pt.use_gpu():
    result = tf.merge(other_tf, on="key")
```

## Comparison with Pandas

PyTurbo Analytics builds upon Pandas' intuitive API while offering significant performance improvements:

### Key Advantages

1. **Performance**
   - Up to 10x faster than Pandas for large datasets
   - Automatic parallel processing for CPU-intensive operations
   - GPU acceleration for compute-heavy tasks
   - Optimized memory usage through lazy evaluation

2. **Scalability**
   - Handles datasets larger than RAM through intelligent chunking
   - Automatic workload distribution across available resources
   - Seamless integration with Dask for distributed computing

3. **Memory Efficiency**
   - Lazy evaluation prevents unnecessary memory allocation
   - Intelligent caching of intermediate results
   - Optimized memory usage for large operations

4. **Ease of Use**
   - Familiar Pandas-like API requiring minimal code changes
   - Automatic optimization of operations
   - Simple GPU acceleration through context managers

### Performance Comparison

| Operation          | Pandas    | PyTurbo Analytics | Speedup |
|-------------------|-----------|-------------------|---------|
| GroupBy + Agg     | 10.2s    | 1.1s             | 9.3x    |
| Merge (Large)     | 15.5s    | 2.3s             | 6.7x    |
| Complex Query     | 8.7s     | 1.4s             | 6.2x    |
| CSV Load (10GB)   | 45.3s    | 5.8s             | 7.8x    |

*Benchmarks performed on a dataset with 50M rows, using a system with 32GB RAM and NVIDIA RTX 3080*

### Example: Customer Analytics

Here's a real-world example comparing Pandas and PyTurbo Analytics for customer transaction analysis:

```python
import pandas as pd
import pyturbo as pt

# Pandas version
def analyze_with_pandas(customers_df, transactions_df):
    # Merge customers and transactions
    merged = customers_df.merge(
        transactions_df,
        on='customer_id',
        how='left'
    )
    
    # Complex analysis
    result = (merged
             .groupby(['region', 'category'])
             .agg({
                 'amount': ['count', 'sum', 'mean'],
                 'age': 'mean'
             })
             .sort_values(('amount', 'sum'), ascending=False))
    return result

# PyTurbo Analytics version
def analyze_with_pyturbo(customers_df, transactions_df):
    # Convert to TurboFrames
    tf_customers = pt.TurboFrame(customers_df)
    tf_transactions = pt.TurboFrame(transactions_df)
    
    # Same analysis, but with GPU acceleration
    with pt.use_gpu():
        result = (tf_customers.merge(tf_transactions, on='customer_id', how='left')
                 .groupby(['region', 'category'])
                 .agg({
                     'amount': ['count', 'sum', 'mean'],
                     'age': 'mean'
                 })
                 .sort_values(('amount', 'sum'), ascending=False)
                 .compute())
    return result
```

For a complete performance comparison example with synthetic data, check out [examples/performance_comparison.py](examples/performance_comparison.py).

### When to Use PyTurbo Analytics

- Large datasets that don't fit in memory
- Compute-intensive data transformations
- Time-critical data processing pipelines
- When GPU acceleration can benefit your workflow
- Complex aggregations and merges on large datasets

### When to Stick with Pandas

- Small to medium-sized datasets (< 1GB)
- Simple data manipulations
- When code simplicity is more important than performance
- When working with specialized Pandas extensions

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for GPU acceleration)
- CUDA Toolkit 11.x (for GPU features)

## Documentation

Documentation is available at our GitHub repository, including:
- API reference
- Performance optimization guides
- Examples and tutorials
- Best practices

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyTurbo Analytics in your research, please cite:

```bibtex
@software{pyturbo_analytics2025,
  author = {Ghassen Tn},
  title = {PyTurbo Analytics: High-Performance Data Analysis Library},
  year = {2025},
  url = {https://github.com/ghassenTn/pyturbo-analytics}
}
