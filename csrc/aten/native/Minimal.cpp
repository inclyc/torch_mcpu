#include "Minimal.h"
#if TORCH_MCPU_ENABLE_CPU_FALLBACK
#include "MCPUFallback.h"
#endif
#include <runtime/DeviceCachingAllocator.h>
#include "runtime/McpuKernelLaunch.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/SmallVector.h>

#include <cstring>
#include <memory>
#include <vector>

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
#include <unordered_set>
#endif

namespace at::native::mcpu {

namespace {

bool is_mcpu_tensor(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::PrivateUse1;
}

bool is_host_device_copy(const at::Tensor& self, const at::Tensor& dst) {
  return (self.is_cpu() && is_mcpu_tensor(dst)) ||
      (is_mcpu_tensor(self) && dst.is_cpu());
}

at::Tensor make_mcpu_tensor_cpu_view(const at::Tensor& tensor) {
  return at::from_blob(
      tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      tensor.options().device(at::kCPU));
}

bool is_last_dim_contiguous(const at::Tensor& tensor) {
  const int64_t dim = tensor.dim();
  return dim == 0 || tensor.size(dim - 1) <= 1 || tensor.stride(dim - 1) == 1;
}

struct MemcpyCopyPlan {
  bool valid = false;
  bool noop = false;
  size_t copy_bytes = 0;
  int64_t outer_count = 0;
  c10::SmallVector<int64_t, 8> outer_sizes;
  c10::SmallVector<int64_t, 8> src_byte_strides;
  c10::SmallVector<int64_t, 8> dst_byte_strides;
};

void record_stream_if_cross_stream(
    const at::Tensor& tensor,
    c10::mcpu::McpuStream stream);

MemcpyCopyPlan make_memcpy_copy_plan(
    const at::Tensor& self,
    const at::Tensor& dst) {
  MemcpyCopyPlan plan;
  if (self.layout() != c10::Layout::Strided ||
      dst.layout() != c10::Layout::Strided ||
      self.scalar_type() != dst.scalar_type() || self.sizes() != dst.sizes() ||
      !is_last_dim_contiguous(self) || !is_last_dim_contiguous(dst)) {
    return plan;
  }

  if (self.is_same(dst)) {
    plan.valid = true;
    plan.noop = true;
    return plan;
  }

  const int64_t numel = self.numel();
  if (numel == 0) {
    plan.valid = true;
    plan.noop = true;
    return plan;
  }

  if (self.device() == dst.device() && self.is_alias_of(dst)) {
    return plan;
  }
  if (self.device() != dst.device() && self.data_ptr() == dst.data_ptr()) {
    return plan;
  }

  const int64_t ndim = self.dim();
  const size_t elem_size = self.element_size();

  if (ndim == 0) {
    plan.valid = true;
    plan.copy_bytes = elem_size;
    plan.outer_count = 1;
    return plan;
  }

  // Singleton axes have arbitrary strides. Establish the physical contiguous
  // block from element stride one, rather than from a singleton's stride.
  // A broadcast source axis must stay outside the memcpy block as well.
  int64_t inner_start = ndim;
  int64_t inner_numel = 1;
  for (int64_t dim = ndim - 1; dim >= 0; --dim) {
    if (self.size(dim) > 1) {
      if (self.stride(dim) != inner_numel || dst.stride(dim) != inner_numel) {
        break;
      }
      inner_numel *= self.size(dim);
    }
    inner_start = dim;
  }
  if (inner_numel == 1 && numel > 1) {
    return plan;
  }
  plan.valid = true;
  plan.copy_bytes = static_cast<size_t>(inner_numel) * elem_size;
  plan.outer_count = 1;
  for (int64_t dim = 0; dim < inner_start; ++dim) {
    plan.outer_count *= self.size(dim);
    plan.outer_sizes.push_back(self.size(dim));
    plan.src_byte_strides.push_back(
        self.stride(dim) * static_cast<int64_t>(elem_size));
    plan.dst_byte_strides.push_back(
        dst.stride(dim) * static_cast<int64_t>(elem_size));
  }

  return plan;
}

void execute_memcpy_copy(
    const char* src_base,
    char* dst_base,
    const MemcpyCopyPlan& plan) {
  const int64_t outer_ndim = plan.outer_sizes.size();
  if (outer_ndim == 0) {
    std::memcpy(dst_base, src_base, plan.copy_bytes);
  } else if (outer_ndim == 1) {
    const int64_t size0 = plan.outer_sizes[0];
    const int64_t src_s0 = plan.src_byte_strides[0];
    const int64_t dst_s0 = plan.dst_byte_strides[0];
    for (int64_t i = 0; i < size0; ++i) {
      std::memcpy(
          dst_base + i * dst_s0, src_base + i * src_s0, plan.copy_bytes);
    }
  } else if (outer_ndim == 2) {
    const int64_t size0 = plan.outer_sizes[0];
    const int64_t size1 = plan.outer_sizes[1];
    const int64_t src_s0 = plan.src_byte_strides[0];
    const int64_t src_s1 = plan.src_byte_strides[1];
    const int64_t dst_s0 = plan.dst_byte_strides[0];
    const int64_t dst_s1 = plan.dst_byte_strides[1];
    for (int64_t i = 0; i < size0; ++i) {
      for (int64_t j = 0; j < size1; ++j) {
        std::memcpy(
            dst_base + i * dst_s0 + j * dst_s1,
            src_base + i * src_s0 + j * src_s1,
            plan.copy_bytes);
      }
    }
  } else {
    c10::SmallVector<int64_t, 8> indices(outer_ndim, 0);
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    for (int64_t i = 0; i < plan.outer_count; ++i) {
      std::memcpy(
          dst_base + dst_offset, src_base + src_offset, plan.copy_bytes);

      for (int64_t dim = outer_ndim - 1; dim >= 0; --dim) {
        ++indices[dim];
        src_offset += plan.src_byte_strides[dim];
        dst_offset += plan.dst_byte_strides[dim];
        if (indices[dim] < plan.outer_sizes[dim]) {
          break;
        }
        src_offset -= indices[dim] * plan.src_byte_strides[dim];
        dst_offset -= indices[dim] * plan.dst_byte_strides[dim];
        indices[dim] = 0;
      }
    }
  }
}

void launch_async_memcpy_copy(
    c10::mcpu::McpuStream stream,
    const char* record_name,
    const void* src_ptr,
    void* dst_ptr,
    MemcpyCopyPlan plan,
    std::vector<at::Tensor> keep_alive = {}) {
  auto plan_ptr = std::make_shared<MemcpyCopyPlan>(std::move(plan));
  at::mcpu::launch_timed_kernel_on_stream(
      stream,
      record_name,
      [src_ptr,
       dst_ptr,
       plan_ptr,
       record_name,
       keep_alive = std::move(keep_alive)](
          at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
        // Host allocations are not tracked by the mcpu caching allocator.
        // Retain their Tensor owners until the queued memcpy has completed.
        (void)keep_alive;
        at::mcpu::KernelPointerMemoryGuard guard({src_ptr, dst_ptr});
        execute_memcpy_copy(
            static_cast<const char*>(src_ptr),
            static_cast<char*>(dst_ptr),
            *plan_ptr);
      });
}

void launch_async_host_device_copy(
    const at::Tensor& self,
    const at::Tensor& dst) {
  const at::Tensor& mcpu_tensor = self.is_cpu() ? dst : self;
  auto stream = c10::mcpu::getCurrentMcpuStream(mcpu_tensor.device().index());

  at::mcpu::launch_timed_kernel_on_stream(
      stream,
      "mcpu::_copy_from.host_device",
      [self, dst](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::_copy_from.host_device", timing_event);

        if (self.is_cpu()) {
          at::mcpu::KernelMemoryGuard guard(dst);
          at::Tensor dst_as_cpu = make_mcpu_tensor_cpu_view(dst);
          at::native::copy_(const_cast<at::Tensor&>(dst_as_cpu), self, true);
        } else {
          at::mcpu::KernelMemoryGuard guard(self);
          at::Tensor self_as_cpu = make_mcpu_tensor_cpu_view(self);
          at::native::copy_(const_cast<at::Tensor&>(dst), self_as_cpu, true);
        }
      });
}

void launch_async_same_device_copy_fallback(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  auto stream = c10::mcpu::getCurrentMcpuStream(self.device().index());
  record_stream_if_cross_stream(self, stream);
  record_stream_if_cross_stream(dst, stream);

  const void* src_ptr = self.data_ptr();
  void* dst_ptr = dst.data_ptr();
  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  auto dst_sizes = dst.sizes().vec();
  auto dst_strides = dst.strides().vec();
  auto self_options = self.options().device(at::kCPU);
  auto dst_options = dst.options().device(at::kCPU);

  at::mcpu::launch_timed_kernel_on_stream(
      stream,
      "mcpu::_copy_from.same_device",
      [src_ptr,
       dst_ptr,
       self_sizes = std::move(self_sizes),
       self_strides = std::move(self_strides),
       dst_sizes = std::move(dst_sizes),
       dst_strides = std::move(dst_strides),
       self_options,
       dst_options,
       non_blocking](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::_copy_from.same_device", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard({src_ptr, dst_ptr});
        at::Tensor dst_as_cpu =
            at::from_blob(dst_ptr, dst_sizes, dst_strides, dst_options);
        const at::Tensor self_as_cpu = at::from_blob(
            const_cast<void*>(src_ptr), self_sizes, self_strides, self_options);
        at::native::copy_(
            const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);
      });
}

bool try_launch_async_host_device_memcpy(
    const at::Tensor& self,
    const at::Tensor& dst) {
  MemcpyCopyPlan plan = make_memcpy_copy_plan(self, dst);
  if (!plan.valid) {
    return false;
  }
  if (plan.noop) {
    return true;
  }

  const at::Tensor& mcpu_tensor = self.is_cpu() ? dst : self;
  auto stream = c10::mcpu::getCurrentMcpuStream(mcpu_tensor.device().index());
  record_stream_if_cross_stream(mcpu_tensor, stream);
  launch_async_memcpy_copy(
      stream,
      "mcpu::_copy_from.host_device.memcpy",
      self.data_ptr(),
      dst.data_ptr(),
      std::move(plan),
      {self, dst});
  return true;
}

void record_stream_if_cross_stream(
    const at::Tensor& tensor,
    c10::mcpu::McpuStream stream) {
  const c10::DataPtr& data_ptr = tensor.storage().data_ptr();
  if (!c10::mcpu::isAllocationStream(data_ptr, stream)) {
    c10::mcpu::recordStream(data_ptr, stream);
  }
}

} // namespace

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT IMPL
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT IMPL

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  MemoryGuard guard(self);

  return at::cpu::as_strided_symint(self, size, stride, storage_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  if (self.is_same(dst)) {
    return dst;
  }
  // Match copy_'s same-view no-op before validating writable destinations.
  // Keep cross-device shared-pointer facades on their synchronization path.
  if (self.device() == dst.device() && self.layout() == c10::Layout::Strided &&
      dst.layout() == c10::Layout::Strided && self.is_alias_of(dst) &&
      self.storage_offset() == dst.storage_offset() &&
      self.strides() == dst.strides() && self.sizes() == dst.sizes() &&
      self.scalar_type() == dst.scalar_type() &&
      self.is_conj() == dst.is_conj() && self.is_neg() == dst.is_neg()) {
    return dst;
  }
  // Fail on the host before an asynchronous worker can write overlapping rows
  // or throw from the CPU-view fallback. Empty destinations perform no writes.
  if (dst.numel() != 0) {
    TORCH_CHECK(
        !self.is_neg() && !self.is_conj() && !dst.is_neg() && !dst.is_conj(),
        "mcpu::_copy_from: lazy negative/conjugate views are unsupported");
    at::assert_no_internal_overlap(dst);
  }

  if (is_mcpu_tensor(self) && is_mcpu_tensor(dst) &&
      self.device() == dst.device()) {
    at::assert_no_partial_overlap(dst, self);
    MemcpyCopyPlan plan = make_memcpy_copy_plan(self, dst);
    if (plan.valid) {
      if (plan.noop) {
        return dst;
      }
      const auto* src_ptr = static_cast<const char*>(self.data_ptr());
      auto* dst_ptr = static_cast<char*>(dst.data_ptr());
      auto stream = c10::mcpu::getCurrentMcpuStream(self.device().index());
      record_stream_if_cross_stream(self, stream);
      record_stream_if_cross_stream(dst, stream);
      launch_async_memcpy_copy(
          stream,
          "mcpu::_copy_from.same_device.memcpy",
          src_ptr,
          dst_ptr,
          std::move(plan));
      return dst;
    }
    launch_async_same_device_copy_fallback(self, dst, non_blocking);
    return dst;
  }

  if (non_blocking && is_host_device_copy(self, dst)) {
    if (try_launch_async_host_device_memcpy(self, dst)) {
      return dst;
    }
    launch_async_host_device_copy(self, dst);
    return dst;
  }

  synchronize_if_mcpu(self);
  synchronize_if_mcpu(dst);
  MemcpyCopyPlan plan;
  if ((is_mcpu_tensor(self) && is_mcpu_tensor(dst)) ||
      is_host_device_copy(self, dst)) {
    plan = make_memcpy_copy_plan(self, dst);
    if (plan.valid && plan.noop) {
      return dst;
    }
  }
  MemoryGuard guard(self, dst);

  if (plan.valid) {
    execute_memcpy_copy(
        static_cast<const char*>(self.data_ptr()),
        static_cast<char*>(dst.data_ptr()),
        plan);
    return dst;
  }

  if (self.device() == dst.device()) {
    at::Tensor dst_as_cpu = at::from_blob(
        dst.data_ptr(),
        dst.sizes(),
        dst.strides(),
        dst.options().device(at::kCPU));
    const at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));

    at::native::copy_(
        const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);

  } else {
    if (self.is_cpu()) {
      at::Tensor dst_as_cpu = make_mcpu_tensor_cpu_view(dst);

      at::native::copy_(
          const_cast<at::Tensor&>(dst_as_cpu), self, non_blocking);

    } else {
      at::Tensor self_as_cpu = make_mcpu_tensor_cpu_view(self);

      at::native::copy_(
          const_cast<at::Tensor&>(dst), self_as_cpu, non_blocking);
    }
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  synchronize_if_mcpu(self);
  MemoryGuard guard(self);
  return at::native::_local_scalar_dense_cpu(self);
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  MemoryGuard guard(self);
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor unfold(
    const at::Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  const auto ndim = self.dim();
  const auto wrapped_dim =
      at::maybe_wrap_dim(dimension, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  const int64_t max_size = ndim == 0 ? 1 : sizes[wrapped_dim];
  TORCH_CHECK(size >= 0, "size is ", size, " but must be >= 0");
  TORCH_CHECK(
      size <= max_size,
      "maximum size for tensor at dimension ",
      wrapped_dim,
      " is ",
      max_size,
      " but size is ",
      size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");

  sizes.push_back(size);
  strides.push_back(ndim == 0 ? 1 : strides[wrapped_dim]);
  if (wrapped_dim < ndim) {
    sizes[wrapped_dim] = (sizes[wrapped_dim] - size) / step + 1;
    strides[wrapped_dim] *= step;
  }

  return at::native::mcpu::as_strided(
      self,
      c10::fromIntArrayRefSlow(sizes),
      c10::fromIntArrayRefSlow(strides),
      std::nullopt);
}

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
// LITERALINCLUDE START: FALLBACK IMPL
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static const std::unordered_set<c10::OperatorName> cpu_fallback_blocklist = {
      c10::OperatorName("aten::abs", ""),
      c10::OperatorName("aten::abs", "out"),
  };

  const auto& op_name = op.schema().operator_name();
  if (cpu_fallback_blocklist.count(op_name)) {
    TORCH_CHECK(
        false, "Operator '", op_name, "' is not implemented for device mcpu.");
  } else {
    // Call our custom CPU fallback implementation instead of PyTorch's
    at::native::mcpu::custom::cpu_fallback(op, stack, /*error_on_views=*/true);
  }
}
// LITERALINCLUDE END: FALLBACK IMPL
#endif

} // namespace at::native::mcpu
