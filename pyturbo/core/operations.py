"""Core computation engine for PyTurbo operations."""

import numpy as np
from typing import Any, Union, Optional, Tuple, Dict
import pandas as pd
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing # Ensure multiprocessing is imported at the top

# Attempt to import GPU-specific libraries and check availability
try:
    import cudf
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cudf = None # Define cudf as None if not available for type hinting and checks

from ..config import get_config


class ComputeEngine:
    """
    Handles computation strategy and execution for TurboFrame operations.
    
    This engine automatically selects the optimal execution strategy based on:
    - Operation type
    - Data size
    - Available hardware (CPU/GPU)
    - Configuration settings (e.g., preferred device)
    """
    
    def __init__(self):
        """Initialize the compute engine."""
        self.config = get_config()
        # Check for 'num_threads' key, provide default if not present
        num_threads = self.config.get('num_threads', multiprocessing.cpu_count())
        self._thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        self._process_pool = ProcessPoolExecutor(max_workers=num_threads)
        
    def execute(self, 
                data: Union[pd.DataFrame, 'cudf.DataFrame'],
                operation_details: Tuple[str, Dict],
                strategy: Optional[str] = None) -> Any:
        """
        Execute computation on the data using the optimal strategy.
        
        Args:
            data: Input data to process (Pandas DataFrame or cuDF DataFrame)
            operation_details: Tuple containing operation type (str) and parameters (dict)
                               e.g., ('groupby_agg', {'by': ['col_A'], 'agg_funcs': {'col_B': 'sum'}})
            strategy: Optional strategy override ('direct', 'thread', 'process', 'gpu')
            
        Returns:
            Computed result (typically a DataFrame)
        """
        op_type, op_params = operation_details

        if strategy is None:
            strategy = self._select_strategy(data, op_type, op_params)
            
        if strategy == 'gpu':
            return self._execute_gpu(data, op_type, op_params)
        elif strategy == 'thread':
            return self._execute_threaded(data, op_type, op_params)
        elif strategy == 'process':
            return self._execute_multiprocess(data, op_type, op_params)
        elif strategy == 'direct':
            return self._execute_direct(data, op_type, op_params)
        else: # Should not happen with proper strategy selection
            raise ValueError(f"Unknown or unsupported strategy: {strategy}")

    def _select_strategy(self, data: Any, op_type: str, op_params: Dict) -> str:
        """
        Select the optimal execution strategy.
        
        Args:
            data: Input data
            op_type: Type of operation (e.g., 'groupby_agg', 'merge', 'sort_values')
            op_params: Parameters for the operation
            
        Returns:
            Selected strategy name ('direct', 'thread', 'process', 'gpu')
        """
        preferred_device = self.config.get('device', 'cpu')
        # Added 'sort_values' to gpu_suitable_ops
        gpu_suitable_ops = ['groupby_agg', 'merge', 'getitem', 'sort_values'] 

        if HAS_GPU and preferred_device.startswith('gpu') and op_type in gpu_suitable_ops:
            return 'gpu'

        # If data is already on GPU (cuDF instance), and we have GPU, prefer GPU execution
        if HAS_GPU and isinstance(data, cudf.DataFrame): 
            return 'gpu'

        # Fallback to CPU strategies for Pandas DataFrames if GPU is not selected
        if isinstance(data, pd.DataFrame):
            size = data.shape[0] * (data.shape[1] if data.ndim > 1 else 1)
            if size < self.config.get('direct_threshold', 1000):  # Small data
                return 'direct'
            # For groupby, merge, and sort_values, Dask is preferred for larger data
            # Added 'sort_values' for Dask consideration
            if op_type in ['groupby_agg', 'merge', 'sort_values']: 
                if size < self.config.get('thread_threshold', 100000): # Medium data
                    return 'thread'
                else: # Large data
                    return 'process'
            # For other ops like getitem on larger data, can use threads or direct.
            # Defaulting to 'thread' for other medium/large pandas ops if not explicitly 'direct'.
            return 'thread' 

        # Default for other types or unforeseen cases (e.g. non-cuDF/non-Pandas, though not expected)
        return 'direct'

    def _execute_direct(self, data: pd.DataFrame, op_type: str, op_params: Dict) -> Any:
        """Execute directly using Pandas."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Direct execution expects a Pandas DataFrame.")

        if op_type == 'groupby_agg':
            return data.groupby(op_params['by']).agg(op_params['agg_funcs'])
        elif op_type == 'merge':
            right_df = op_params['right']
            if HAS_GPU and isinstance(right_df, cudf.DataFrame): # Check instance type
                right_df = right_df.to_pandas()
            return data.merge(right_df, 
                              on=op_params['on'], 
                              how=op_params['how'], 
                              **op_params.get('kwargs', {}))
        elif op_type == 'getitem':
            return data[op_params['key']]
        elif op_type == 'sort_values':
            by_arg = op_params.get('by')
            # Create a copy of op_params to avoid modifying the original dict
            remaining_kwargs = {k: v for k, v in op_params.items() if k != 'by'}
            return data.sort_values(by=by_arg, **remaining_kwargs)
        else:
            raise ValueError(f"Unsupported operation type for direct execution: {op_type}")
        
    def _execute_gpu(self, data: Union[pd.DataFrame, 'cudf.DataFrame'], op_type: str, op_params: Dict) -> Any:
        """Execute on GPU using cuDF."""
        if not HAS_GPU or cudf is None: # Check cudf module
            raise EnvironmentError("GPU (cuDF) not available for _execute_gpu.")
        
        if not isinstance(data, cudf.DataFrame): # Check instance type
            if isinstance(data, pd.DataFrame):
                data = cudf.DataFrame.from_pandas(data) # Convert to cuDF DataFrame
            else:
                raise TypeError("GPU execution expects a cuDF DataFrame or Pandas DataFrame to convert.")

        if op_type == 'groupby_agg':
            return data.groupby(op_params['by']).agg(op_params['agg_funcs'])
        elif op_type == 'merge':
            right_df = op_params['right']
            if isinstance(right_df, pd.DataFrame):
                right_df = cudf.DataFrame.from_pandas(right_df) # Convert to cuDF DataFrame
            elif not isinstance(right_df, cudf.DataFrame): # Check instance type
                 raise TypeError("Right operand for GPU merge must be cuDF or Pandas DataFrame.")
            return data.merge(right_df, 
                              on=op_params['on'], 
                              how=op_params['how'], 
                              **op_params.get('kwargs', {}))
        elif op_type == 'getitem':
            return data[op_params['key']]
        elif op_type == 'sort_values':
            by_arg = op_params.get('by')
            remaining_kwargs = {k: v for k, v in op_params.items() if k != 'by'}
            return data.sort_values(by=by_arg, **remaining_kwargs)
        else:
            raise ValueError(f"Unsupported operation type for GPU execution: {op_type}")
        
    def _execute_threaded(self, data: pd.DataFrame, op_type: str, op_params: Dict) -> Any:
        """Execute using Dask with thread pool."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Threaded execution with Dask expects a Pandas DataFrame.")

        npartitions = self.config.get('num_threads', multiprocessing.cpu_count())
        ddf = dd.from_pandas(data, npartitions=npartitions)

        if op_type == 'groupby_agg':
            result_ddf = ddf.groupby(op_params['by']).agg(op_params['agg_funcs'])
        elif op_type == 'merge':
            right_df = op_params['right']
            if not isinstance(right_df, pd.DataFrame):
                 if HAS_GPU and isinstance(right_df, cudf.DataFrame): # Check instance type
                    right_df = right_df.to_pandas()
                 else:
                    raise TypeError("Right operand for Dask merge must be Pandas DataFrame.")
            right_ddf = dd.from_pandas(right_df, npartitions=npartitions)
            result_ddf = dd.merge(ddf, right_ddf, 
                                  on=op_params['on'], 
                                  how=op_params['how'], 
                                  **op_params.get('kwargs', {}))
        elif op_type == 'getitem':
            result_ddf = ddf[op_params['key']]
        elif op_type == 'sort_values':
            by_arg = op_params.get('by')
            remaining_kwargs = {k: v for k, v in op_params.items() if k != 'by'}
            result_ddf = ddf.sort_values(by=by_arg, **remaining_kwargs)
        else:
            raise ValueError(f"Unsupported operation type for threaded Dask execution: {op_type}")
            
        return result_ddf.compute(scheduler='threads')
        
    def _execute_multiprocess(self, data: pd.DataFrame, op_type: str, op_params: Dict) -> Any:
        """Execute using Dask with process pool."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Multiprocess execution with Dask expects a Pandas DataFrame.")

        npartitions = self.config.get('num_threads', multiprocessing.cpu_count())
        ddf = dd.from_pandas(data, npartitions=npartitions)

        if op_type == 'groupby_agg':
            result_ddf = ddf.groupby(op_params['by']).agg(op_params['agg_funcs'])
        elif op_type == 'merge':
            right_df = op_params['right']
            if not isinstance(right_df, pd.DataFrame):
                 if HAS_GPU and isinstance(right_df, cudf.DataFrame): # Check instance type
                    right_df = right_df.to_pandas()
                 else:
                    raise TypeError("Right operand for Dask merge must be Pandas DataFrame.")
            right_ddf = dd.from_pandas(right_df, npartitions=npartitions)
            result_ddf = dd.merge(ddf, right_ddf, 
                                  on=op_params['on'], 
                                  how=op_params['how'],
                                  **op_params.get('kwargs', {}))
        elif op_type == 'getitem':
            result_ddf = ddf[op_params['key']]
        elif op_type == 'sort_values':
            by_arg = op_params.get('by')
            remaining_kwargs = {k: v for k, v in op_params.items() if k != 'by'}
            result_ddf = ddf.sort_values(by=by_arg, **remaining_kwargs)
        else:
            raise ValueError(f"Unsupported operation type for multiprocess Dask execution: {op_type}")

        return result_ddf.compute(scheduler='processes')
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_thread_pool') and self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        if hasattr(self, '_process_pool') and self._process_pool:
            self._process_pool.shutdown(wait=False)

# Global compute engine instance
compute_engine = ComputeEngine()
