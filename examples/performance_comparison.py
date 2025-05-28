"""
Performance Comparison Example: PyTurbo Analytics vs Pandas

This example demonstrates the performance benefits of PyTurbo Analytics
compared to pandas using synthetic data for common data analysis tasks.
"""

import time
import numpy as np
import pandas as pd
import pyturbo as pt
from typing import Tuple, Dict

def generate_synthetic_data(num_rows: int = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data for testing."""
    # Generate customer data
    np.random.seed(42)
    customers = pd.DataFrame({
        'customer_id': range(num_rows),
        'age': np.random.randint(18, 90, num_rows),
        'income': np.random.normal(60000, 20000, num_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_rows),
        'signup_date': pd.date_range(start='2020-01-01', periods=num_rows),
    })
    
    # Generate transaction data (multiple transactions per customer)
    transactions = pd.DataFrame({
        'transaction_id': range(num_rows * 5),
        'customer_id': np.random.choice(customers['customer_id'], num_rows * 5),
        'amount': np.random.normal(100, 50, num_rows * 5),
        'category': np.random.choice(['Food', 'Electronics', 'Clothing', 'Books', 'Other'], num_rows * 5),
        'transaction_date': pd.date_range(start='2020-01-01', periods=num_rows * 5),
    })
    
    return customers, transactions

def benchmark_pandas(customers: pd.DataFrame, transactions: pd.DataFrame) -> Dict[str, float]:
    """Benchmark pandas operations."""
    timings = {}
    
    # Benchmark GroupBy operation
    start = time.time()
    avg_by_region = customers.groupby('region').agg({
        'age': 'mean',
        'income': ['mean', 'std']
    })
    timings['groupby'] = time.time() - start
    
    # Benchmark Merge operation
    start = time.time()
    customer_transactions = customers.merge(
        transactions,
        on='customer_id',
        how='left'
    )
    timings['merge'] = time.time() - start
    
    # Benchmark Complex Query
    start = time.time()
    result = (customer_transactions
             .groupby(['region', 'category'])
             .agg({
                 'amount': ['count', 'sum', 'mean'],
                 'age': 'mean'
             })
             .reset_index()
             .sort_values(('amount', 'sum'), ascending=False)
             )
    timings['complex_query'] = time.time() - start
    
    return timings

def benchmark_pyturbo(customers: pd.DataFrame, transactions: pd.DataFrame) -> Dict[str, float]:
    """Benchmark PyTurbo Analytics operations."""
    timings = {}
    
    # Convert to TurboFrames
    tf_customers = pt.TurboFrame(customers)
    tf_transactions = pt.TurboFrame(transactions)
    
    # Benchmark GroupBy operation
    start = time.time()
    avg_by_region = tf_customers.groupby('region').agg({
        'age': 'mean',
        'income': ['mean', 'std']
    }).compute()
    timings['groupby'] = time.time() - start
    
    # Benchmark Merge operation
    start = time.time()
    customer_transactions = tf_customers.merge(
        tf_transactions,
        on='customer_id',
        how='left'
    ).compute()
    timings['merge'] = time.time() - start
    
    # Benchmark Complex Query on CPU
    # Uses tf_customers and tf_transactions initialized at the start of the function.
    # These are expected to be CPU-backed as no use_gpu() context was active for them.
    start_cpu = time.time()
    result_cpu = (tf_customers.merge(tf_transactions, on='customer_id', how='left')
                 .groupby(['region', 'category'])
                 .agg({
                     'amount': ['count', 'sum', 'mean'],
                     'age': 'mean'
                 })
                 .sort_values(('amount', 'sum'), ascending=False)
                 .compute())
    timings['complex_query_cpu'] = time.time() - start_cpu

    # Benchmark Complex Query with GPU acceleration
    # Re-initialize TurboFrames from pandas DataFrames inside use_gpu context
    # to include potential data transfer to GPU in this benchmark.
    start_gpu = time.time()
    with pt.use_gpu():
        # TurboFrame.__init__ should convert to cuDF automatically here
        gpu_tf_customers = pt.TurboFrame(customers) 
        gpu_tf_transactions = pt.TurboFrame(transactions)
        
        result_gpu = (gpu_tf_customers.merge(gpu_tf_transactions, on='customer_id', how='left')
                     .groupby(['region', 'category'])
                     .agg({
                         'amount': ['count', 'sum', 'mean'],
                         'age': 'mean'
                     })
                     .sort_values(('amount', 'sum'), ascending=False)
                     .compute())
    timings['complex_query_gpu'] = time.time() - start_gpu
    
    return timings

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    customers, transactions = generate_synthetic_data(num_rows=1_000_000)
    print(f"Generated {len(customers):,} customer records and {len(transactions):,} transactions")
    
    # Run pandas benchmarks
    print("\nRunning pandas benchmarks...")
    pandas_timings = benchmark_pandas(customers, transactions)
    
    # Run PyTurbo Analytics benchmarks
    print("\nRunning PyTurbo Analytics benchmarks...")
    pyturbo_timings = benchmark_pyturbo(customers, transactions)
    
    # Print results
    print("\nPerformance Comparison (seconds):")
    print("-" * 60)
    # Adjusted header to be more generic as PyTurbo will have CPU/GPU variants
    print(f"{'Operation':<30} {'Pandas':>10} {'PyTurbo':>10} {'Speedup':>10}")
    print("-" * 60)
    
    # Define the order of operations for printing and handle special complex query case
    operations_to_print = ['groupby', 'merge', 'complex_query']

    for operation_key in operations_to_print:
        pandas_time = pandas_timings.get(operation_key)
        
        if operation_key == 'complex_query':
            # PyTurbo CPU
            pyturbo_cpu_time = pyturbo_timings.get('complex_query_cpu')
            if pandas_time is not None and pyturbo_cpu_time is not None:
                speedup_cpu = pandas_time / pyturbo_cpu_time
                print(f"{'Complex Query (CPU)':<30} {pandas_time:>10.3f} {pyturbo_cpu_time:>10.3f} {speedup_cpu:>10.1f}x")
            else:
                print(f"{'Complex Query (CPU)':<30} {pandas_time:>10.3f} {'N/A':>10} {'N/A':>10}")

            # PyTurbo GPU
            pyturbo_gpu_time = pyturbo_timings.get('complex_query_gpu')
            if pandas_time is not None and pyturbo_gpu_time is not None:
                speedup_gpu = pandas_time / pyturbo_gpu_time
                print(f"{'Complex Query (GPU)':<30} {pandas_time:>10.3f} {pyturbo_gpu_time:>10.3f} {speedup_gpu:>10.1f}x")
            else:
                # Pandas time might be printed again if only GPU fails, but that's okay for this structure
                print(f"{'Complex Query (GPU)':<30} {pandas_time if pandas_time is not None else 'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            # For simple operations like groupby and merge
            pyturbo_time = pyturbo_timings.get(operation_key)
            if pandas_time is not None and pyturbo_time is not None:
                speedup = pandas_time / pyturbo_time
                print(f"{operation_key:<30} {pandas_time:>10.3f} {pyturbo_time:>10.3f} {speedup:>10.1f}x")
            else:
                # Handle cases where a timing might be missing
                print(f"{operation_key:<30} {pandas_time if pandas_time is not None else 'N/A':>10} {pyturbo_time if pyturbo_time is not None else 'N/A':>10} {'N/A':>10}")
    print("-" * 60)

if __name__ == "__main__":
    main()
