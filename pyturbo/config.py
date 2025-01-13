"""Configuration and device management for PyTurbo."""

import os
import contextlib
from typing import Optional, Dict, Any
import multiprocessing
import threading

import numpy as np

# Global configuration
_CONFIG = {
    'device': 'cpu',
    'num_threads': multiprocessing.cpu_count(),
    'num_gpus': 0,
    'gpu_memory': {},
}

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dict containing device information
    """
    num_gpus = 0
    has_gpu = False
    gpu_info = {}
    
    try:
        import cupy as cp
        num_gpus = cp.cuda.runtime.getDeviceCount()
        has_gpu = num_gpus > 0
        
        if has_gpu:
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    mem_info = cp.cuda.runtime.memGetInfo()
                    gpu_info[i] = {
                        'total_memory': mem_info[1],
                        'free_memory': mem_info[0]
                    }
    except ImportError:
        pass  # GPU support not available
        
    return {
        'num_cpus': multiprocessing.cpu_count(),
        'num_gpus': num_gpus,
        'has_gpu': has_gpu,
        'gpu_info': gpu_info
    }

@contextlib.contextmanager
def use_gpu(device_id: Optional[int] = 0):
    """
    Context manager for GPU operations.
    
    Args:
        device_id: GPU device ID to use
    """
    prev_device = _CONFIG['device']
    try:
        if not get_device_info()['has_gpu']:
            print("Warning: GPU requested but not available. Using CPU instead.")
            yield
            return
            
        _CONFIG['device'] = f'gpu:{device_id}'
        yield
    finally:
        _CONFIG['device'] = prev_device

def set_num_threads(n: Optional[int] = None):
    """
    Set the number of threads for CPU operations.
    
    Args:
        n: Number of threads to use. If None, uses number of CPU cores.
    """
    if n is None:
        n = multiprocessing.cpu_count()
        
    _CONFIG['num_threads'] = n
    
    # Set thread count for various libraries
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)
    
    try:
        import mkl
        mkl.set_num_threads(n)
    except ImportError:
        pass
        
def get_config() -> Dict[str, Any]:
    """
    Get current configuration.
    
    Returns:
        Dict containing current configuration
    """
    return _CONFIG.copy()

# Initialize configuration
device_info = get_device_info()
_CONFIG['num_gpus'] = device_info['num_gpus']
_CONFIG['gpu_memory'] = device_info['gpu_info']
