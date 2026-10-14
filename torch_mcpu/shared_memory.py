"""Process-lifetime shared allocation for KV cache backing storage."""

import torch
from torch_mcpu import _C


def xmalloc(size: int, device: int | None = None) -> torch.Tensor:
    """Allocate uninitialized shared bytes as a non-resizable MCPU Tensor.

    Configure TORCH_MCPU_SHARED_SIZE before the first call. The pre-mapped build also
    requires TORCH_MCPU_SHARED_BASE, mapped by hardware startup. The default Linux
    simulator instead uses TORCH_MCPU_SHARED_SIM_FILE (one unique file per TP group);
    TORCH_MCPU_SHARED_BASE optionally requests a fixed virtual address in the simulator.

    Both ranks must call with the same sizes in the same order. Each call
    reserves size rounded up to 256 bytes. Tensor deletion does not reclaim
    space. Initialization and cross-rank write/read synchronization belong to
    the caller; this function never clears the region.
    """
    if device is None:
        device = torch.mcpu.current_device()
    return _C._xmalloc(size, device)
