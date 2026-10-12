#include <runtime/McpuKernelLaunch.h>

#include "DeviceCachingAllocator.h"
#include "OpenRegException.h"
#include "OpenRegStream.h"

#include <include/openreg.h>

#include <exception>
#include <thread>

namespace at::mcpu {
namespace {
thread_local KernelLaunchCondition current_condition;
}

KernelLaunchLifetimeScope::KernelLaunchLifetimeScope(
    std::shared_ptr<const void> owner)
    : stream_(
          owner ? static_cast<orStream_t>(c10::mcpu::getCurrentMcpuStream())
                : nullptr),
      owner_(std::move(owner)) {}

KernelLaunchLifetimeScope::~KernelLaunchLifetimeScope() noexcept {
  if (!owner_)
    return;
  try {
    // Bypass conditional/timed launch wrappers: cleanup must always execute.
    // Keep our copy until publication succeeds, including on an enqueue error.
    if (orLaunchKernel(stream_, [owner = owner_]() noexcept { (void)owner; }) ==
        orSuccess) {
#if !TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH
      return;
#endif
    }
  } catch (...) {
    // No-throw shared_ptr capture normally makes this path unreachable. Still
    // retain the owner while draining any successfully submitted prefix.
  }
  try {
    // Honor synchronous-launch builds for the cleanup task too. In asynchronous
    // builds this is only an enqueue-failure fallback. Never self-synchronize.
    c10::mcpu::detail::OptionalPythonGilRelease release_gil;
    if (stream_->worker.get_id() != std::this_thread::get_id() &&
        orStreamSynchronize(stream_) == orSuccess) {
      return;
    }
  } catch (...) {
  }
  // Releasing borrowed storage when completion cannot be established is unsafe.
  std::terminate();
}

KernelLaunchConditionScope::KernelLaunchConditionScope(
    std::shared_ptr<const bool> enabled)
    : owned_lifetime_(enabled) {
  if (!enabled)
    return;
  TORCH_CHECK(
      !current_condition.enabled,
      "nested kernel launch conditions are unsupported");
  current_condition = {owned_lifetime_.stream_, enabled.get()};
  owns_condition_ = true;
}

KernelLaunchConditionScope::KernelLaunchConditionScope(
    const KernelLaunchLifetimeScope& lifetime,
    const bool* enabled) {
  if (!enabled)
    return;
  TORCH_CHECK(lifetime.owner_, "kernel launch condition requires a live owner");
  TORCH_CHECK(
      !current_condition.enabled,
      "nested kernel launch conditions are unsupported");
  TORCH_CHECK(
      lifetime.stream_ ==
          static_cast<orStream_t>(c10::mcpu::getCurrentMcpuStream()),
      "kernel launch condition must use its lifetime's stream");
  current_condition = {lifetime.stream_, enabled};
  owns_condition_ = true;
}

KernelLaunchConditionScope::~KernelLaunchConditionScope() {
  if (owns_condition_)
    current_condition = {};
}

namespace detail {
const KernelLaunchCondition& kernel_launch_condition() noexcept {
  return current_condition;
}
} // namespace detail
} // namespace at::mcpu

namespace at::mcpu::detail {

bool enter_kernel_task() {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  bool previous = openreg::isInKernelTask();
  openreg::setInKernelTask(true);
  return previous;
#else
  return true;
#endif
}

void exit_kernel_task(bool previous) noexcept {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  openreg::setInKernelTask(previous);
#else
  (void)previous;
#endif
}

void protect_memory(void* ptr) noexcept {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  (void)orMemoryProtect(ptr);
#else
  (void)ptr;
#endif
}

void unprotect_memory(
    const void* ptr,
    std::unordered_set<void*>& unprotected_pointers) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  TORCH_CHECK(
      openreg::isInKernelTask(),
      "mcpu memory may only be accessed from a launched kernel task");
  if (ptr == nullptr) {
    return;
  }

  orPointerAttributes attr;
  if (orPointerGetAttributes(&attr, ptr) != orSuccess ||
      attr.type != orMemoryTypeDevice) {
    return;
  }

  auto [_, inserted] = unprotected_pointers.insert(attr.pointer);
  if (inserted) {
    MCPU_CHECK(orMemoryUnprotect(attr.pointer));
  }
#else
  (void)ptr;
  (void)unprotected_pointers;
#endif
}

void unprotect_tensor_memory(
    const at::TensorBase& tensor,
    std::unordered_set<void*>& unprotected_pointers) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  if (!tensor.defined() || !tensor.has_storage() || tensor.numel() == 0) {
    return;
  }

  unprotect_memory(tensor.data_ptr(), unprotected_pointers);
#else
  (void)tensor;
  (void)unprotected_pointers;
#endif
}

void unprotect_all_device_memory(
    std::unordered_set<void*>& unprotected_pointers) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  TORCH_CHECK(
      openreg::isInKernelTask(),
      "mcpu memory may only be accessed from a launched kernel task");
  c10::mcpu::unprotectAllAllocatedMemory(unprotected_pointers);
#else
  (void)unprotected_pointers;
#endif
}

orStream_t get_kernel_launch_stream(const at::Tensor& stream_tensor) {
  TORCH_CHECK(
      stream_tensor.defined() &&
          stream_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "launch_kernel expects an mcpu tensor to select the stream");
  return c10::mcpu::getCurrentMcpuStream(stream_tensor.device().index());
}

} // namespace at::mcpu::detail
