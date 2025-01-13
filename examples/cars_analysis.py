"""
Example script demonstrating PyTurbo's capabilities with vehicle metrics dataset.
"""

import time
import pandas as pd
import numpy as np
import pyturbo as pt

def benchmark(func):
    """Simple benchmark decorator"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@benchmark
def pandas_analysis(df):
    """Perform analysis using regular pandas"""
    results = {}
    
    # Basic statistics for numerical columns
    results['stats'] = df[['DAILY_DISTANCE', 'AVERAGE_SPEED', 'FUEL_COST', 'AVERAGE_ECO_SCORE']].describe()
    
    # Daily metrics
    results['daily_metrics'] = df.groupby('CVEH').agg({
        'DAILY_DISTANCE': ['mean', 'max', 'sum'],
        'FUEL_COST': ['mean', 'sum'],
        'AVERAGE_SPEED': 'mean',
        'AVERAGE_ECO_SCORE': 'mean',
        'NB_HARD_BRAKING': 'sum',
        'NB_HARD_ACCELERATION': 'sum',
        'NB_FATIGUE_DRIVINH': 'sum'
    })
    
    # Efficiency analysis
    df['fuel_efficiency'] = df['DAILY_DISTANCE'] / df['DAILY_FUEL']
    results['efficiency'] = df.groupby('CVEH')['fuel_efficiency'].mean()
    
    # Find days with highest fuel consumption
    results['high_fuel_days'] = df.nlargest(5, 'FUEL_COST')[
        ['THE_DAY', 'CVEH', 'DAILY_DISTANCE', 'FUEL_COST', 'AVERAGE_SPEED']
    ]
    
    # Safety metrics
    results['safety_metrics'] = df.groupby('CVEH').agg({
        'NB_HARD_BRAKING': 'sum',
        'NB_HARD_ACCELERATION': 'sum',
        'NB_FATIGUE_DRIVINH': 'sum',
        'AVERAGE_ECO_SCORE': 'mean'
    })
    
    return results

@benchmark
def turbo_analysis(tf):
    """Perform the same analysis using PyTurbo"""
    results = {}
    
    # Basic statistics for numerical columns
    results['stats'] = tf.data[['DAILY_DISTANCE', 'AVERAGE_SPEED', 'FUEL_COST', 'AVERAGE_ECO_SCORE']].describe()
    
    # Daily metrics
    results['daily_metrics'] = tf.groupby('CVEH').agg({
        'DAILY_DISTANCE': ['mean', 'max', 'sum'],
        'FUEL_COST': ['mean', 'sum'],
        'AVERAGE_SPEED': 'mean',
        'AVERAGE_ECO_SCORE': 'mean',
        'NB_HARD_BRAKING': 'sum',
        'NB_HARD_ACCELERATION': 'sum',
        'NB_FATIGUE_DRIVINH': 'sum'
    })
    
    # Efficiency analysis
    tf_data = tf.data.copy()
    tf_data['fuel_efficiency'] = tf_data['DAILY_DISTANCE'] / tf_data['DAILY_FUEL']
    results['efficiency'] = tf_data.groupby('CVEH')['fuel_efficiency'].mean()
    
    # Find days with highest fuel consumption
    results['high_fuel_days'] = tf_data.nlargest(5, 'FUEL_COST')[
        ['THE_DAY', 'CVEH', 'DAILY_DISTANCE', 'FUEL_COST', 'AVERAGE_SPEED']
    ]
    
    # Safety metrics
    results['safety_metrics'] = tf_data.groupby('CVEH').agg({
        'NB_HARD_BRAKING': 'sum',
        'NB_HARD_ACCELERATION': 'sum',
        'NB_FATIGUE_DRIVINH': 'sum',
        'AVERAGE_ECO_SCORE': 'mean'
    })
    
    return results

def print_analysis_results(results):
    """Print formatted analysis results"""
    print("\n=== Vehicle Fleet Analysis ===")
    
    print("\n1. Overall Statistics:")
    print(results['stats'])
    
    print("\n2. Daily Performance Metrics (per vehicle):")
    print(results['daily_metrics'])
    
    print("\n3. Top 5 Days with Highest Fuel Cost:")
    print(results['high_fuel_days'])
    
    print("\n4. Vehicle Safety Metrics:")
    print(results['safety_metrics'])
    
    print("\n5. Fuel Efficiency (km/L):")
    print(results['efficiency'])

def main():
    # Load data using pandas for comparison
    print("\nLoading with Pandas...")
    pandas_df = pd.read_csv('2024-11-09.csv')
    
    # Load data using PyTurbo
    print("\nLoading with PyTurbo...")
    turbo_df = pt.TurboFrame.from_csv('2024-11-09.csv')
    
    print("\nPandas Analysis:")
    pandas_results = pandas_analysis(pandas_df)
    
    print("\nPyTurbo Analysis:")
    # Try GPU if available
    try:
        with pt.use_gpu():
            turbo_results = turbo_analysis(turbo_df)
    except RuntimeError:
        # Fall back to CPU if GPU not available
        turbo_results = turbo_analysis(turbo_df)
    
    # Print detailed analysis
    print_analysis_results(pandas_results)
    
    # Verify results match
    print("\nVerifying results match between Pandas and PyTurbo...")
    pd.testing.assert_frame_equal(
        pandas_results['daily_metrics'].sort_index(), 
        turbo_results['daily_metrics'].sort_index(),
        check_dtype=False
    )
    print("✓ Results verified!")

if __name__ == "__main__":
    main()
