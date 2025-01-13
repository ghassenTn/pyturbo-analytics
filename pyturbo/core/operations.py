"""Core computation engine for PyTurbo operations."""

import numpy as np
from typing import Any, Union, Optional
import pandas as pd
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ..config import get_config

class ComputeEngine:
    """
    Handles computation strategy and execution for TurboFrame operations.
    
    This engine automatically selects the optimal execution strategy based on:
    - Operation type (CPU vs GPU intensive)
    - Data size
    - Available hardware
    - Current workload
    """
    
    def __init__(self):
        """Initialize the compute engine."""
        self.config = get_config()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config['num_threads'])
        self._process_pool = ProcessPoolExecutor(max_workers=self.config['num_threads'])
        
    def execute(self, 
                data: Union[pd.DataFrame, 'cudf.DataFrame'],
                strategy: Optional[str] = None) -> Any:
        """
        Execute computation on the data using the optimal strategy.
        
        Args:
            data: Input data to process
            strategy: Optional strategy override ('thread', 'process', 'gpu', or 'auto')
            
        Returns:
            Computed result
        """
        if strategy is None:
            strategy = self._select_strategy(data)
            
        if strategy == 'gpu':
            return self._execute_gpu(data)
        elif strategy == 'thread':
            return self._execute_threaded(data)
        elif strategy == 'process':
            return self._execute_multiprocess(data)
        else:
            return data  # Return as-is for small operations
            
    def _select_strategy(self, data: Any) -> str:
        """
        Select the optimal execution strategy based on data and operation characteristics.
        
        Args:
            data: Input data
            
        Returns:
            Selected strategy name
        """
        # Simple heuristic - can be expanded based on benchmarking
        if hasattr(data, 'shape'):
            size = data.shape[0] * data.shape[1]
            if size < 1000:  # Small data
                return 'direct'
            elif size < 100000:  # Medium data
                return 'thread'
            else:  # Large data
                return 'process'
        return 'direct'
        
    def _execute_gpu(self, data: 'cudf.DataFrame') -> Any:
        """Execute on GPU."""
        # GPU execution is handled by cuDF directly
        return data
        
    def _execute_threaded(self, data: pd.DataFrame) -> Any:
        """Execute using thread pool."""
        if isinstance(data, pd.DataFrame):
            # Convert to dask for parallel execution
            ddf = dd.from_pandas(data, npartitions=self.config['num_threads'])
            return ddf.compute(scheduler='threads')
        return data
        
    def _execute_multiprocess(self, data: pd.DataFrame) -> Any:
        """Execute using process pool."""
        if isinstance(data, pd.DataFrame):
            ddf = dd.from_pandas(data, npartitions=self.config['num_threads'])
            return ddf.compute(scheduler='processes')
        return data
        
    def __del__(self):
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=False)
        self._process_pool.shutdown(wait=False)

# Global compute engine instance
compute_engine = ComputeEngine()
