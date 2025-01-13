# PyTurbo Analytics: High-Performance Data Analysis Library 🚀

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
