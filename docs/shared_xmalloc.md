# Shared KV arena: xmalloc

`c10::mcpu::xmalloc(bytes)` returns a local pointer into a process-lifetime
shared region. The Python adapter returns an uninitialized, non-resizable
`torch.uint8` MCPU Tensor over that pointer:

```python
from torch_mcpu.shared_memory import xmalloc

cache_storage = xmalloc(4096)
```

The ordinary MCPU allocator is unchanged. This interface supplies persistent KV
backing; the companion vLLM XCPU plugin change manages cache layout, ownership,
and cross-rank synchronization.

## Common contract

- Set environment variables before the first call. They are read once when the
  region is initialized. Start fresh worker processes, not a fork of an already
  initialized allocator.
- One region per process; one shared region per TP pair. Each rank has a **local**
  allocation cursor starting at zero. Both ranks must allocate identical sizes
  in identical order during startup, so their offsets alias. A shared cursor
  would incorrectly allocate two separate copies. Cross-rank allocation order
  and size agreement are the caller's responsibility and are not checked here.
- Align the region's starting address upward to 256 bytes; round each requested
  size upward to 256 bytes. `TORCH_MCPU_SHARED_SIZE` must cover these padding bytes as well
  as the requested cache bytes. Zero, overflow, and out-of-bounds allocations fail.
- Never clear on attach. Only the designated writer initializes/updates cache.
  No synchronization is implied by allocation: finish device writes before
  notifying the reader, and finish reads before allowing block reuse.
- There is no individual free/reset. Tensor deletion does not recycle memory;
  all allocations remain valid until process exit. Do not call `free`/`orFree`.
  Shared bytes are not counted in the ordinary allocator's memory statistics.

## Linux/x86 simulation (default build)

Give both ranks the same **unique per-run, per-group** file path and size:

```sh
export TORCH_MCPU_SHARED_SIM_FILE=/dev/shm/my-run-tp0.kv
export TORCH_MCPU_SHARED_SIZE=1073741824
# Optional: TORCH_MCPU_SHARED_BASE=0x... requests a page-aligned local virtual address.
```

`mmap(MAP_SHARED)` provides the backing. Initialization is file-locked, and an
existing nonempty file must match the requested size; it is never truncated or
cleared on attach. Different TP pairs use different files. The launcher must
remove its files **after all workers exit**; workers do not unlink them. A fresh
filename per run avoids attaching stale data. Size reserves addressable file
space, not guaranteed physical RAM: budget the tmpfs capacity accordingly.

Without `TORCH_MCPU_SHARED_BASE`, the OS selects each process's virtual address. With it,
the simulator uses `MAP_FIXED_NOREPLACE` and fails if that address is occupied.
No existing mapping is overwritten. Simulation requires Linux and does not
register the region with the ordinary OpenReg memory-protection tracker.

## Pre-mapped device region

```sh
TORCH_MCPU_SHARED_USE_PREMAPPED=ON /workspace/.venv/bin/python -m pip install --no-build-isolation --no-deps .
```

Equivalent CMake switch: `-DTORCH_MCPU_SHARED_USE_PREMAPPED=ON`.

```sh
export TORCH_MCPU_SHARED_BASE=0x100000000
export TORCH_MCPU_SHARED_SIZE=1073741824
```

The pre-mapped branch uses that region directly, with no `mmap` or simulation file.
**Assumption:** hardware startup has already made the address accessible to this
process/device and mapped the same physical region for both ranks. An unmapped
physical address is not sufficient. Different groups receive separate regions.
Future device mapping/registration calls belong in `open_region()`; alignment and
capacity logic stay common. Actual hardware kernels and coherence are not validated by
the x86 tests.

## Verification

From outside the source directory, after rebuilding/installing:

```sh
cd /tmp
/workspace/.venv/bin/python -m pytest /workspace/torch_mcpu/tests/test_memory.py -k TestSharedXmalloc -q
```

Tests cover two concurrent TP pairs, repeated block writes/read acknowledgments,
late attachment without clearing, MCPU Tensor copies/views, alignment, lifetime,
capacity/overflow/configuration errors, and fixed-address collision handling.
The pre-mapped compile branch is separately compiled and run against a supplied local
buffer to test its base/size logic without claiming real hardware validation.
