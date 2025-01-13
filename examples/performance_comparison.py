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
    
    # Benchmark Complex Query with GPU acceleration
    start = time.time()
    with pt.use_gpu():
        result = (tf_customers.merge(tf_transactions, on='customer_id', how='left')
                 .groupby(['region', 'category'])
                 .agg({
                     'amount': ['count', 'sum', 'mean'],
                     'age': 'mean'
                 })
                 .sort_values(('amount', 'sum'), ascending=False)
                 .compute())
    timings['complex_query'] = time.time() - start
    
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
    print(f"{'Operation':<20} {'Pandas':>10} {'PyTurbo':>10} {'Speedup':>10}")
    print("-" * 60)
    
    for operation in pandas_timings:
        pandas_time = pandas_timings[operation]
        pyturbo_time = pyturbo_timings[operation]
        speedup = pandas_time / pyturbo_time
        print(f"{operation:<20} {pandas_time:>10.3f} {pyturbo_time:>10.3f} {speedup:>10.1f}x")

if __name__ == "__main__":
    main()
