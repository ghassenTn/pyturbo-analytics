"""TurboFrame: High-performance DataFrame implementation."""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from contextlib import contextmanager

from .operations import compute_engine
from ..config import get_device_info, get_config # Added get_config

# Optional GPU support
try:
    import cudf
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cudf = None # Define cudf as None if not available for type checks


class TurboGroupBy:
    """
    Represents a pending groupby operation on a TurboFrame.
    This object is returned by TurboFrame.groupby() and provides methods like .agg().
    """
    def __init__(self, parent_frame: 'TurboFrame', by: Union[str, List[str]], **groupby_kwargs):
        """
        Initialize a TurboGroupBy object.

        Args:
            parent_frame: The TurboFrame instance on which groupby() was called.
            by: The column(s) to group by.
            **groupby_kwargs: Additional keyword arguments for the groupby operation itself.
        """
        self._parent_frame: 'TurboFrame' = parent_frame
        self._by: Union[str, List[str]] = by
        self._groupby_kwargs: Dict[str, Any] = groupby_kwargs

    def agg(self, agg_funcs: Dict[str, Any], **agg_op_kwargs) -> 'TurboFrame':
        """
        Perform a lazy aggregation on the grouped data.

        Args:
            agg_funcs: A dictionary specifying the aggregation functions for columns.
                       e.g., {'col_A': 'mean', 'col_B': ['sum', 'count']}
            **agg_op_kwargs: Additional keyword arguments for the aggregation operation (rarely used directly).

        Returns:
            A new TurboFrame instance with the 'groupby_agg' operation added to its pipeline.
        """
        op_params = {
            'by': self._by,
            'agg_funcs': agg_funcs,
            **self._groupby_kwargs, # Include groupby specific args like sort=True/False
            **agg_op_kwargs # Include any explicit agg operation args
        }
        
        # The operation type is 'groupby_agg' for ComputeEngine
        op = ('groupby_agg', op_params)
        
        # Create a new TurboFrame, chaining from the parent frame's initial data and prior operations
        return TurboFrame(
            self._parent_frame._initial_data, 
            operations=self._parent_frame._ops_pipeline + [op]
        )

    # Potentially other methods like .sum(), .mean(), .count() could be added here,
    # which would call self.agg(...) with predefined agg_funcs. For example:
    # def mean(self, numeric_only=None, **kwargs):
    #     # This would require knowing which columns are numeric to apply mean,
    #     # or passing specific columns. Pandas GroupBy.mean behavior is complex.
    #     # For now, focusing on .agg() as it's more explicit.
    #     pass


class TurboFrame:
    """
    A high-performance DataFrame that automatically leverages available hardware acceleration.
    
    The TurboFrame provides a Pandas-like interface but automatically distributes computations
    across available CPU cores and GPU devices for maximum performance.
    """
    
    def __init__(self, data: Union[pd.DataFrame, 'cudf.DataFrame', np.ndarray, Dict, 'TurboFrame'], operations: Optional[List[tuple]] = None):
        """
        Initialize a TurboFrame.
        
        Args:
            data: Input data (Pandas DataFrame, cuDF DataFrame, NumPy array, dict)
                  or another TurboFrame instance (for internal chaining).
            operations: A list of pending operations (for internal use).
        """
        self._ops_pipeline: List[tuple] = operations or []
        self._initial_data: Union[pd.DataFrame, 'cudf.DataFrame', None] = None
        self._computed_data: Union[pd.DataFrame, 'cudf.DataFrame', None] = None
        self._device: str = "cpu" # Default target device for this frame's data

        global_config = get_config()
        prefer_gpu_globally = HAS_GPU and global_config.get('device', 'cpu').startswith('gpu')

        if isinstance(data, TurboFrame):
            self._initial_data = data._initial_data
            # Correctly combine pipelines: existing frame's pipeline + new operations for this frame
            self._ops_pipeline = data._ops_pipeline + (operations or []) 
            self._device = data._device # Inherit device from the source TurboFrame's current state/preference
            # However, if global config mandates GPU, and inherited _initial_data is CPU, consider conversion
            if prefer_gpu_globally and self._device == "cpu" and isinstance(self._initial_data, pd.DataFrame):
                try:
                    self._initial_data = cudf.DataFrame.from_pandas(self._initial_data)
                    self._device = "gpu" # Update device of this new frame
                except Exception as e:
                    print(f"Failed to convert inherited Pandas DataFrame to cuDF in __init__ (use_gpu context): {e}")
        elif HAS_GPU and isinstance(data, cudf.DataFrame):
            self._initial_data = data
            self._device = "gpu"
        elif isinstance(data, pd.DataFrame):
            if prefer_gpu_globally:
                try:
                    self._initial_data = cudf.DataFrame.from_pandas(data)
                    self._device = "gpu"
                except Exception as e:
                    print(f"Failed to convert Pandas DataFrame to cuDF in __init__ (use_gpu context): {e}. Falling back to CPU.")
                    self._initial_data = data
                    self._device = "cpu"
            else:
                self._initial_data = data
                self._device = "cpu"
        elif isinstance(data, (np.ndarray, dict)):
            df = pd.DataFrame(data)
            if prefer_gpu_globally:
                try:
                    self._initial_data = cudf.DataFrame.from_pandas(df)
                    self._device = "gpu"
                except Exception as e:
                    print(f"Failed to convert dict/ndarray to cuDF in __init__ (use_gpu context): {e}. Falling back to CPU.")
                    self._initial_data = df
                    self._device = "cpu"
            else:
                self._initial_data = df
                self._device = "cpu"
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @classmethod
    def from_csv(cls, 
                 filepath: str, 
                 use_gpu: bool = None, 
                 **kwargs) -> 'TurboFrame':
        """
        Create a TurboFrame from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            use_gpu: Whether to load directly to GPU memory
            **kwargs: Additional arguments passed to pd.read_csv or cudf.read_csv
            
        Returns:
            TurboFrame instance
        """
        # This method creates a new TurboFrame, so it doesn't involve lazy loading directly for itself,
        # but the TurboFrame it creates will be lazy.
        # The device preference from use_gpu should set the target device of the new frame.
        target_device = "cpu"
        if use_gpu is None:
            use_gpu = get_device_info()['has_gpu'] # Check if system has GPU
        
        if use_gpu and HAS_GPU:
            try:
                data = cudf.read_csv(filepath, **kwargs)
                instance = cls(data)
                instance._device = "gpu"
                return instance
            except Exception as e:
                print(f"GPU loading failed: {e}. Falling back to CPU.")
        
        data = pd.read_csv(filepath, **kwargs)
        instance = cls(data)
        instance._device = "cpu" # Explicitly set device based on loading
        return instance

    def to_gpu(self) -> 'TurboFrame':
        """Set the target device for computation to GPU and transfer initial data if eager."""
        if not HAS_GPU:
            print("GPU support not available. Target device remains CPU.")
            return self
        
        # If there's a pipeline, just set the target device for future compute
        # If no pipeline, and data is on CPU, transfer it now.
        if not self._ops_pipeline and self._device == "cpu" and self._initial_data is not None:
             if isinstance(self._initial_data, pd.DataFrame):
                self._initial_data = cudf.DataFrame.from_pandas(self._initial_data)
        
        self._device = "gpu" # Set target device
        self._computed_data = None # Invalidate cache
        return self
        
    def to_cpu(self) -> 'TurboFrame':
        """Set the target device for computation to CPU and transfer initial data if eager."""
        # If there's a pipeline, just set the target device for future compute
        # If no pipeline, and data is on GPU, transfer it now.
        if not self._ops_pipeline and self._device == "gpu" and self._initial_data is not None:
            if HAS_GPU and isinstance(self._initial_data, cudf.DataFrame):
                self._initial_data = self._initial_data.to_pandas()

        global_config = get_config()
        if global_config.get('device', 'cpu').startswith('gpu'):
            print("Warning: to_cpu() called while use_gpu() context is active. Data transferred to CPU, but subsequent operations might still prefer GPU based on global config.")

        self._device = "cpu" # Set target device
        self._computed_data = None # Invalidate cache
        return self
        
    @property
    def data(self) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Get the underlying DataFrame. Triggers computation if pending operations exist.
        """
        if self._ops_pipeline:
            # If there are pending operations, compute them.
            # The result of compute() will be stored in self._computed_data
            if self._computed_data is None:
                 self._computed_data = self.compute()
            return self._computed_data
        else:
            # No operations, return initial data, ensuring it's on the correct device.
            if self._device == "gpu" and HAS_GPU and isinstance(self._initial_data, pd.DataFrame):
                return cudf.DataFrame.from_pandas(self._initial_data)
            elif self._device == "cpu" and HAS_GPU and isinstance(self._initial_data, cudf.DataFrame):
                return self._initial_data.to_pandas()
            return self._initial_data

    def compute(self) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Execute all pending computations and return the result.
        
        Returns:
            DataFrame with computed results
        """
        if not self._ops_pipeline:
            # If no operations, return a copy of initial data respecting the target device
            current_data = self._initial_data
            if self._device == "gpu" and HAS_GPU:
                if isinstance(current_data, pd.DataFrame):
                    current_data = cudf.DataFrame.from_pandas(current_data)
            elif self._device == "cpu":
                if HAS_GPU and isinstance(current_data, cudf.DataFrame):
                    current_data = current_data.to_pandas()
            # Return a copy to prevent modification of _initial_data if it's returned directly
            return current_data.copy() if current_data is not None else None

        # Make a copy of initial_data to start the pipeline
        # Ensure it's on the target device.
        current_data = self._initial_data.copy() # Start with a copy

        if self._device == "gpu" and HAS_GPU:
            if not isinstance(current_data, cudf.DataFrame):
                current_data = cudf.DataFrame.from_pandas(current_data)
        elif self._device == "cpu":
            if HAS_GPU and isinstance(current_data, cudf.DataFrame):
                current_data = current_data.to_pandas()
        
        # Placeholder for compute_engine logic
        # In the future, compute_engine will handle sequences of operations.
        # For now, we simulate by applying pandas/cudf methods directly in a loop.
        for op_type, op_params in self._ops_pipeline:
            if op_type == 'groupby':
                # Groupby in pandas/cudf returns a GroupBy object.
                # The actual aggregation (e.g., .agg(), .sum()) is a subsequent step.
                # The 'groupby' operation is passed to compute_engine.
                # compute_engine will be responsible for performing the group by
                # and handling the result (e.g., returning a GroupBy object or an aggregated DataFrame
                # if aggregation details are part of op_params or a subsequent operation).
        # For 'groupby_agg', ComputeEngine will handle it.
        # The 'groupby' op_type is no longer directly added by TurboFrame.groupby()
        # It's now encapsulated within 'groupby_agg' created by TurboGroupBy.agg().
        # If a raw 'groupby' op were to exist (e.g., for iteration over groups),
        # its handling in compute() would need to be defined.
        # For now, 'groupby_agg' is the primary path.
        if op_type == 'groupby_agg': # Changed from 'groupby' to 'groupby_agg'
            current_data = compute_engine.execute(current_data, operation_details=(op_type, op_params.copy()))
        elif op_type == 'merge':
            right_df_source = op_params.get('right')
                # Resolve right_df_source if it's a TurboFrame
                if isinstance(right_df_source, TurboFrame):
                    right_df = right_df_source.compute() # Compute the right frame if it's also lazy
                else: # Should be a DataFrame if not TurboFrame (as per current merge op prep)
                    right_df = right_df_source

                
                global_config = get_config()
                prefer_gpu_globally = HAS_GPU and global_config.get('device', 'cpu').startswith('gpu')

                # Device alignment for merge operation based on global preference
                if prefer_gpu_globally:
                    if isinstance(current_data, pd.DataFrame):
                        current_data = cudf.DataFrame.from_pandas(current_data)
                    if isinstance(right_df, pd.DataFrame):
                        right_df = cudf.DataFrame.from_pandas(right_df)
                else: # Prefer CPU
                    if HAS_GPU and isinstance(current_data, cudf.DataFrame):
                        current_data = current_data.to_pandas()
                    if HAS_GPU and isinstance(right_df, cudf.DataFrame):
                        right_df = right_df.to_pandas()
                
                # After alignment, both current_data and right_df should be of the same type (either both Pandas or both cuDF)
                # Or one might be None if an error occurred or data was empty. Add checks if necessary.

                params_for_merge = {k: v for k, v in op_params.items() if k != 'right'}
                
                # The actual merge should now be on consistent-type data.
                # This could still be passed to compute_engine, which would get data already on the "correct" device type.
                # For now, let's assume TurboFrame.compute does the final merge call if it's simple.
                # However, the task was to make TurboFrame proactive before ComputeEngine.
                # The ComputeEngine will also have its own strategy selection.
                # Let's ensure this data is then passed to compute_engine.
                
                # Update op_params['right'] to the potentially converted right_df for compute_engine
                op_params_updated = op_params.copy()
                op_params_updated['right'] = right_df
                current_data = compute_engine.execute(current_data, operation_details=(op_type, op_params_updated))

            elif op_type == 'select':
                current_data = current_data[op_params['key']]
                # Ensure it's still a DataFrame (e.g. if a single column is selected as Series)
                if not isinstance(current_data, (pd.DataFrame, cudf.DataFrame if HAS_GPU else pd.DataFrame)):
                    current_data = current_data.to_frame()


        # Clear the pipeline and store the computed result
        # self._ops_pipeline = [] # Option 1: Clear pipeline after compute
        self._computed_data = current_data # Cache the result
        return current_data
        
    def groupby(self, by: Union[str, List[str]], **kwargs) -> 'TurboGroupBy':
        """
        Group DataFrame by specified columns.

        This method initiates a lazy groupby operation. Aggregation methods like .agg()
        must be called on the returned TurboGroupBy object to complete the operation
        and add it to the computation pipeline.

        Args:
            by: Column(s) to group by.
            **kwargs: Additional arguments for the groupby operation (e.g., sort=True).

        Returns:
            A TurboGroupBy object representing the pending groupby operation.
        """
        return TurboGroupBy(self, by=by, **kwargs)
        
    def merge(self, 
             right: 'TurboFrame', 
             how: str = 'inner', 
             on: Optional[Union[str, List[str]]] = None,
             **kwargs) -> 'TurboFrame':
        """
        Lazily merge with another TurboFrame. The operation is added to the pipeline.
        """
        # Store the right TurboFrame itself, or its data if it's not a TurboFrame (though API implies TurboFrame)
        # If right is a TurboFrame, its computation will be handled during self.compute()
        op_params = {'right': right, 'how': how, 'on': on, **kwargs}
        op = ('merge', op_params)
        return TurboFrame(self._initial_data, operations=self._ops_pipeline + [op])
        
    def __getitem__(self, key):
        """Lazily select columns. The operation is added to the pipeline."""
        op = ('select', {'key': key})
        return TurboFrame(self._initial_data, operations=self._ops_pipeline + [op])

    def sort_values(self, by, **kwargs) -> 'TurboFrame':
        """
        Lazily sort by the values along either axis. The operation is added to the pipeline.
        
        Args:
            by: Name or list of names to sort by.
            **kwargs: Additional arguments passed to underlying sort_values 
                      (e.g., ascending=True, inplace=False, kind='quicksort', na_position='last')
                      
        Returns:
            A new TurboFrame with the sort operation added to its pipeline.
        """
        op_params = {'by': by, **kwargs}
        op = ('sort_values', op_params)
        return TurboFrame(self._initial_data, operations=self._ops_pipeline + [op])
        
    def __len__(self):
        """Return number of rows. Triggers computation if needed."""
        return len(self.data) # self.data will trigger compute()
        
    def __str__(self):
        """String representation. Triggers computation if needed for row count."""
        # Avoid printing the full data if it requires computation, just show status
        if self._ops_pipeline and self._computed_data is None:
            return f"TurboFrame(pending_ops={len(self._ops_pipeline)}, target_device={self._device})"
        # If data is computed or no ops, show normal string repr
        computed_now = self.data # This will trigger compute if needed
        return f"TurboFrame(rows={len(computed_now)}, device={self._device})\n{str(computed_now)}"
        
    def __repr__(self):
        """Detailed string representation."""
        return self.__str__()
