# MCPU kernel launch migration guide

This document describes the migration rule for `torch_mcpu`.

For allocator lifetime and stream-reuse rules behind pointer-only kernels, see
[`mcpu_stream_memory_lifetime.md`](mcpu_stream_memory_lifetime.md).

## Contents

- [Runtime model](#runtime-model)
- [Operator structure](#operator-structure)
- [Migration checkpoints](#migration-checkpoints)
- [Launch macro](#launch-macro)
- [Pointer-capture pattern](#pointer-capture-pattern)
- [Tensor-capture pattern](#tensor-capture-pattern)
- [External torch extensions](#external-torch-extensions)
  - [`torch_xcpu` migration example](#torch_xcpu-migration-example)

## Runtime model

MCPU device memory is page protected by default. A tensor allocation is readable
and writable only while a kernel task is running on an MCPU stream. Direct CPU
access to MCPU storage is intentionally invalid because it bypasses the launch
queue.

Build-time mode:

- `-DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`: device allocations use ordinary
  aligned allocation, `orMemoryProtect/Unprotect` are no-ops, and
  `KernelMemoryGuard` / `KernelPointerMemoryGuard` compile to empty RAII
  objects.

`MCPU_LAUNCH_TIMED_KERNEL` is the default launch interface for migrated kernel
work. It enqueues memory access on the current MCPU stream, reserves a timing
event, and opens a timing scope with the same name.

## Operator structure

Split every operator into metadata work and memory work.

- Metadata work stays outside the launch: input checks, dtype checks, dtype
  dispatch, output tensor allocation, resizing, sizes/strides, scalar argument
  normalization, and raw `data_ptr` extraction.
- Memory work goes inside the launch: dereferencing pointers, reads, writes,
  CPU fallback execution over CPU views, and copies.
- If Python needs a scalar or CPU tensor value immediately, synchronize the
  relevant stream before reading.

## Migration checkpoints

When migrating an operator or external extension to the asynchronous launch
path, check the following before considering the migration complete:

- Python-side tests that inspect results immediately must add the necessary
  `torch.accelerator.synchronize()` calls after launching MCPU work. The
  operator implementation should stay asynchronous; the test owns the
  synchronization needed for deterministic reads.
- For every launched function, inspect its callers up the call stack and verify
  that pointer-like or array-like arguments are not backed by stack-allocated
  storage that can go out of scope before the worker thread runs the task. This
  includes raw pointers, pointer arrays, temporary `std::array`/C arrays, and
  small metadata buffers passed through wrapper functions.
- Prefer pointer capture for hand-written kernels. Use tensor capture when the
  launched task must call ATen operators or helper APIs that need tensor views.
- Prefer `KernelPointerMemoryGuard` whenever the pointer set is known before
  launch. Use `KernelAllMemoryGuard` only when the kernel discovers indirect
  MCPU allocation addresses from metadata inside the launched task.
- If the captured launch arguments exceed the stream inline slot limit, put
  copied scalar/pointer metadata in a heap args struct and move-capture a
  `std::unique_ptr` to it. Borrowed tensor `data_ptr`s inside that struct must
  not be freed.

## Launch Macro

Use `MCPU_LAUNCH_TIMED_KERNEL` from `runtime/McpuKernelLaunch.h`:

```cpp
MCPU_LAUNCH_TIMED_KERNEL("mcpu::my_kernel", ([captures]), {
  // memory work
});
```

The single name is used both as the launch record name and as the timing scope
name. Wrap the capture list in parentheses, for example `([out_ptr, n])`, so
commas in the capture list are parsed as part of one macro argument. Wrap the
kernel body in `{ ... }` for readability.

For oversized argument lists, capture one owning args object instead of many
values: `([args = std::move(args)])`.

The queue provides 480 bytes of inline task storage. On the current 64-bit ABI,
the complete slot occupies 512 bytes including metadata and padding. The limit
applies to the complete queued task, including launch wrappers and arguments.
Changing this layout requires rebuilding the runtime, external extensions, and
cached compiled artifacts together; do not mix the previous 256-byte layout
with the new one.

## Pointer-Capture Pattern

Use this by default for hand-written kernels, including kernels under
`csrc/aten/vllm_kernels`. Extract raw pointers and scalar launch parameters
before enqueueing, then capture only those values in the task.

```cpp
int64_t n = out.numel();
const float* in_ptr = input.data_ptr<float>();

VLLM_MCPU_DISPATCH_FLOAT(out, "my_pointer_kernel", {
  scalar_t* out_ptr = out.data_ptr<scalar_t>();
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::my_pointer_kernel",
      ([out_ptr, in_ptr, n]),
      {
        at::mcpu::KernelPointerMemoryGuard guard({out_ptr, in_ptr});
        my_pointer_kernel<scalar_t>(out_ptr, in_ptr, n);
      });
});
```

The raw pointers are passed to the implementation just like CUDA kernels receive
device pointers. Dtype dispatch stays outside the launched task. The launched
task should not perform pointer lookup unless it must call an ATen operator.

If the full pointer set is not known until after reading MCPU metadata inside
the task, keep the pointer-capture shape but use `KernelAllMemoryGuard` instead
of `KernelPointerMemoryGuard`:

```cpp
MCPU_LAUNCH_TIMED_KERNEL("mcpu::my_indirect_kernel", ([metadata_ptr, n]), {
  at::mcpu::KernelAllMemoryGuard guard;
  my_indirect_kernel(metadata_ptr, n);
});
```

Prefer `KernelPointerMemoryGuard` when the pointer set is statically available.
`KernelAllMemoryGuard` is intended for kernels that need indirect pointers such
as address tables.

## Tensor-Capture Pattern

Use tensor capture when the task calls ATen operators or helpers that construct
CPU views from MCPU tensors. Capturing tensors keeps their storage alive and
`KernelMemoryGuard` unlocks the backing allocations for the duration of the
task.

```cpp
MCPU_LAUNCH_TIMED_KERNEL(
    "mcpu::aten::some_op.out",
    ([out, input, alpha]),
    {
      at::mcpu::KernelMemoryGuard guard(out, input);
      auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
      auto cpu_input = ops::get_cpu_tensor_view_if_needed(input);
      at::some_cpu_op_out(cpu_out, cpu_input, alpha);
    });
```

`KernelMemoryGuard` also accepts nested `std::optional` arguments. Pass optional
tensors directly instead of creating undefined tensor sentinels or assembling a
conditional pointer array:

```cpp
MCPU_LAUNCH_TIMED_KERNEL(
    "mcpu::my_optional_tensor_kernel",
    ([out, input, bias]),
    {
      at::mcpu::KernelMemoryGuard guard(out, input, bias);
      my_optional_tensor_kernel(out, input, bias);
    });
```

An empty optional is ignored. An engaged optional is processed recursively, so
the same rule also applies to optional values already supported by
`KernelMemoryGuard`, such as `c10::IValue`.

This pattern is appropriate for fallback-style ATen operators. If the kernel is
pure hand-written pointer code, prefer the pointer-capture pattern instead.

## External Torch Extensions

External projects registering MCPU kernels should follow the same rules:

- include `runtime/McpuKernelLaunch.h`;
- link against `libtorch_mcpu.so` so stream, timing, and memory-guard symbols
  resolve normally;
- extract pointers, shape, and stride metadata before enqueueing;
- enqueue memory work with `MCPU_LAUNCH_TIMED_KERNEL`;
- use pointer capture by default with `KernelPointerMemoryGuard`;
- use tensor capture only when the task must call ATen operators or tensor-view
  helpers;
- keep Python tests from reading protected storage directly. If a test needs
  results, synchronize explicitly and copy with `tensor.to("cpu")`.

### `torch_xcpu` Migration Example

`torch_xcpu` is an important external-project example for the `torch_mcpu`
launch and memory-protection model. Its custom ops are registered outside
`torch_mcpu`, but their kernels still run against MCPU allocations and must obey
the same launch boundary.

For hand-written pointer kernels in `torch_xcpu_impl`, use the pointer-capture
pattern shown above:

- compile with `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=1` so
  `KernelPointerMemoryGuard` is active. External projects should prefer reading
  `torch_mcpu.get_compile_flags()` from the installed package and appending
  those flags to their extension build, so memory-protection and timing settings
  stay aligned with the installed runtime;
- pass only raw pointers and scalar metadata into the launched task;
- guard all dereferenced MCPU pointers with `KernelPointerMemoryGuard`;
- keep tensor capture for code paths that call ATen operators.

This keeps external kernel code close to the in-tree
`torch_mcpu/csrc/aten/vllm_kernels` style while avoiding direct CPU access to
MCPU storage outside the stream task.


## Stream-produced launch conditions

Use `at::mcpu::KernelLaunchConditionScope` when a fixed batch of kernels should
execute only if a flag produced earlier on the same stream is true. The scope
attaches a borrowed `const bool*` to both ordinary and timed kernel launches.
The stream worker checks it before memory guards or kernel work. Timed launches
reserve slots on the submitter, as before; skipped tasks leave zero timestamps
and are excluded from completed profiler events.

The caller must never read the flag to decide what to submit. Update it in a
preceding unconditional stream task, outside the condition scope. Keep all flag
reads and writes on that stream; the runtime rejects launches to another stream
and rejects nonempty nested conditions. An empty scope preserves an existing
condition. Scope destruction restores ordinary submission, including on exceptions.

Use one `KernelLaunchLifetimeScope` to retain an invocation's state across all
of its submissions. It enqueues an unconditional owner-retaining tail task on
the stream captured at construction, including when host submission unwinds.
Individual conditional tasks do not copy or release the owner:

```cpp
auto state = std::make_shared<InvocationState>();
at::mcpu::KernelLaunchLifetimeScope lifetime(state);
auto* raw = state.get();
for (int round = 0; round < max_rounds; ++round) {
  at::mcpu::launch_kernel([raw, round] {
    raw->active = round < raw->actual_rounds;
  });
  at::mcpu::KernelLaunchConditionScope condition(lifetime, &raw->active);
  existing_kernel_wrapper_a(...);
  existing_kernel_wrapper_b(...);
}
```

The lifetime scope must be created before any submission borrowing its state,
and outlive every such submission and condition scope. The condition pointer
must refer to storage kept alive by that owner. Keep the stream alive through
scope destruction. Do not read/write the predicate from the host after publishing
tasks, or borrow its storage on another stream. This changes ownership only;
existing wrappers and their host-side metadata work remain in place.

The convenience `KernelLaunchConditionScope(shared_ptr<const bool>)` overload
retains once per condition scope and also uses an unconditional tail task. For
multiple regions in one invocation, prefer the explicit lifetime overload above
to avoid an extra cleanup task per region. An empty owner/condition is a no-op.

If cleanup publication fails, the runtime retains its owner while synchronizing
the captured stream. This is an exceptional fallback, not part of normal
submission. If safe completion cannot be established (including a self-wait on
the worker), it terminates rather than release storage still in use.
Synchronous-launch builds also wait for the cleanup task before scope destruction
returns. Waiting releases the Python GIL when held. Asynchronous builds retain
their normal enqueue-only path.

Skipped tasks still destroy their captures.
Conditional closures that exceed the inline queue slot size use an owning heap
closure, while fitting tasks retain inline submission. Ordinary launches keep
the existing inline path. This mechanism neither captures nor replays kernels,
and does not predicate arbitrary raw `orLaunchKernel` calls or host-side work.

For collectives, every rank must agree the condition before entering guarded
communication; local emptiness is insufficient. V5/V6 EP exchanges round counts
in dispatch record headers on the stream and uses fixed configuration-derived
submission bounds for this purpose.

V5/V6 still use process-global MPI window contexts. Calls sharing a context must
be ordered through their final reduction, using a single communication stream or
explicit cross-stream dependencies with matching collective order on every rank.
Invocation-local predicates do not make overlapping EP calls on those windows
safe. Stream priorities select worker idle policies, not collective ordering or
preemptive scheduling.

Profiler reservations support concurrent host submitters. Skipped tasks retain
zero timestamps and still consume reservation capacity (262,144 slots globally);
they do not become completed trace events. Stop host submissions and drain streams
before resetting or reading timing payloads. Atomic reservation does not make
live reset/snapshot operations safe.

V5/V6 EP retains its chunk plan once around the complete invocation and borrows
`active_round` across all rounds. Rebuild external extensions after updating this
header/runtime pair; the condition and scope layouts changed. Previously compiled
JIT/Inductor artifacts using the old header must also be regenerated.

For measured execution-sequence overhead with a supplied stream queue, ownership
costs, and reproducible benchmarks, see
[条件 kernel launch 的执行序列性能评估](kernel_launch_condition_performance.md).
