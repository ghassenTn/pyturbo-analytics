"""TurboFrame: High-performance DataFrame implementation."""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from contextlib import contextmanager

from .operations import compute_engine
from ..config import get_device_info

# Optional GPU support
try:
    import cudf
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class TurboFrame:
    """
    A high-performance DataFrame that automatically leverages available hardware acceleration.
    
    The TurboFrame provides a Pandas-like interface but automatically distributes computations
    across available CPU cores and GPU devices for maximum performance.
    """
    
    def __init__(self, data: Union[pd.DataFrame, 'cudf.DataFrame', np.ndarray, Dict]):
        """
        Initialize a TurboFrame with input data.
        
        Args:
            data: Input data in various formats (Pandas DataFrame, cuDF DataFrame, NumPy array, or dict)
        """
        self._data = None
        self._gpu_data = None
        self._device = "cpu"
        
        if HAS_GPU and isinstance(data, cudf.DataFrame):
            self._gpu_data = data
            self._device = "gpu"
        elif isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, (np.ndarray, dict)):
            self._data = pd.DataFrame(data)
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
        if use_gpu is None:
            use_gpu = get_device_info()['has_gpu']
            
        if use_gpu and HAS_GPU:
            try:
                data = cudf.read_csv(filepath, **kwargs)
                return cls(data)
            except Exception as e:
                print(f"GPU loading failed: {e}. Falling back to CPU.")
                
        data = pd.read_csv(filepath, **kwargs)
        return cls(data)
        
    def to_gpu(self) -> 'TurboFrame':
        """Transfer data to GPU memory."""
        if not HAS_GPU:
            print("GPU support not available. Operation will be no-op.")
            return self
            
        if self._device == "gpu":
            return self
            
        if self._data is not None:
            self._gpu_data = cudf.DataFrame.from_pandas(self._data)
            self._data = None
            self._device = "gpu"
        return self
        
    def to_cpu(self) -> 'TurboFrame':
        """Transfer data to CPU memory."""
        if self._device == "cpu":
            return self
            
        if self._gpu_data is not None:
            self._data = self._gpu_data.to_pandas()
            self._gpu_data = None
            self._device = "cpu"
        return self
        
    @property
    def data(self) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """Get the underlying DataFrame."""
        return self._gpu_data if self._device == "gpu" else self._data
        
    def compute(self) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Execute all pending computations and return the result.
        
        Returns:
            DataFrame with computed results
        """
        return compute_engine.execute(self.data)
        
    def groupby(self, by, **kwargs):
        """
        Group DataFrame using a mapper or by a list of columns.
        
        Args:
            by: Names of columns to group by
            **kwargs: Additional arguments passed to underlying groupby
            
        Returns:
            GroupBy object
        """
        return self.data.groupby(by, **kwargs)
        
    def merge(self, 
             right: 'TurboFrame', 
             how: str = 'inner', 
             on: Optional[Union[str, List[str]]] = None,
             **kwargs) -> 'TurboFrame':
        """
        Merge with another TurboFrame.
        
        Args:
            right: Right TurboFrame to merge with
            how: Type of merge to perform
            on: Column(s) to merge on
            **kwargs: Additional arguments passed to underlying merge
            
        Returns:
            Merged TurboFrame
        """
        # Ensure both frames are on same device
        if self._device != right._device:
            if self._device == "gpu" and HAS_GPU:
                right = right.to_gpu()
            else:
                right = right.to_cpu()
                
        result = self.data.merge(right.data, how=how, on=on, **kwargs)
        return TurboFrame(result)
        
    def __getitem__(self, key):
        """Enable Pandas-style column access."""
        return TurboFrame(self.data[key])
        
    def __len__(self):
        """Return number of rows."""
        return len(self.data)
        
    def __str__(self):
        """String representation."""
        return f"TurboFrame(rows={len(self)}, device={self._device})\n{str(self.data)}"
        
    def __repr__(self):
        """Detailed string representation."""
        return self.__str__()
