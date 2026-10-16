"""Process-lifetime shared allocation for KV cache backing storage."""

import torch

from torch_mcpu import _C


def xmalloc(size: int, device: int | None = None) -> torch.Tensor:
    """Allocate uninitialized shared bytes as a non-resizable MCPU Tensor.

    Set TORCH_MCPU_SHARED_SIZE and either TORCH_MCPU_SHARED_SIM_FILE (x86) or
    TORCH_MCPU_SHARED_BASE (pre-mapped build) before the first call. Ranks sharing
    a region must allocate the same sizes in the same order. Each allocation
    reserves a multiple of 256 bytes until process exit. Callers synchronize
    reads and writes. See docs/shared_xmalloc.md for setup.
    """
    if device is None:
        device = torch.mcpu.current_device()
    return _C._xmalloc(size, device)
