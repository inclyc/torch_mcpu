# Shared KV arena: `xmalloc`

`c10::mcpu::xmalloc(bytes)` returns a pointer to process-lifetime shared memory.
The Python wrapper returns a non-resizable MCPU `uint8` tensor. Allocations are
aligned to 256 bytes and are never individually freed or reused.

```python
from torch_mcpu.shared_memory import xmalloc
cache_storage = xmalloc(4096)
```

Each TP rank has its own allocation cursor. Ranks in one TP group must allocate
the same sizes in the same order so their offsets refer to the same physical
region. Allocation provides no synchronization; callers publish writes and
finish reads before reusing blocks.

## x86 simulation

```sh
export TORCH_MCPU_SHARED_SIM_FILE=/dev/shm/run-tp0.kv
export TORCH_MCPU_SHARED_SIZE=1073741824
# Optional: request a page-aligned virtual address.
export TORCH_MCPU_SHARED_BASE=0x100000000
```

Both ranks use the same fresh file and size. Different DP/TP groups use different
files. `mmap(MAP_SHARED)` supplies the backing; an existing file must have the
requested size and is never cleared on attach. Remove the file after all workers
exit. A fixed base uses `MAP_FIXED_NOREPLACE` and never overwrites an
existing mapping.

## Pre-mapped device region

Build with `-DTORCH_MCPU_SHARED_USE_PREMAPPED=ON` (or the equivalent build
option), then provide `TORCH_MCPU_SHARED_BASE` and `TORCH_MCPU_SHARED_SIZE`.
The device is responsible for mapping the same physical region into both ranks;
the common alignment and capacity checks remain in this module. Device coherence
and hardware mapping are outside the x86 simulation tests.

## Test

```sh
cd /tmp
/workspace/.venv/bin/python -m pytest \
  /workspace/torch_mcpu/tests/test_memory.py -k TestSharedXmalloc -q
```

The tests cover alignment, capacity and configuration errors, process sharing,
DP-group isolation, late attachment, fixed-address protection, tensor lifetime,
and the pre-mapped C++ branch.
