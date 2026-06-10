"""
Device utilities optimized for MacBook Air M4 with Metal Performance Shaders support.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def get_optimal_device():
    """
    Get optimal device for MacBook Air M4.
    Prioritizes MPS (Metal Performance Shaders) for M4 chip.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) for M4 chip acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU - consider enabling MPS for M4 acceleration")
    
    return device

def setup_torch_for_m4():
    """
    Configure PyTorch settings optimized for M4 chip.
    """
    if torch.backends.mps.is_available():
        # Enable MPS fallback for operations not supported on MPS
        torch.backends.mps.allow_tf32 = True
        logger.info("M4 MPS acceleration configured")
    
    # Set number of threads for CPU operations (useful for data loading)
    torch.set_num_threads(8)  # M4 has 8 cores
    
    return get_optimal_device()

def move_to_device(obj, device):
    """
    Safely move tensor/model to device with MPS compatibility checks.
    """
    if device.type == "mps":
        # Some operations might not be supported on MPS yet
        try:
            return obj.to(device)
        except RuntimeError as e:
            logger.warning(f"MPS operation failed, falling back to CPU: {e}")
            return obj.to("cpu")
    else:
        return obj.to(device)