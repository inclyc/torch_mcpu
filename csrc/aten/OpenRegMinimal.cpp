#include "native/Minimal.h"

#include <runtime/DeviceCachingAllocator.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/ops/copy_native.h>

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#endif

#include <torch/library.h>

namespace at::mcpu {

namespace {

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT WRAPPER
at::Tensor wrapper_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return at::native::mcpu::empty_memory_format(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT WRAPPER

at::Tensor wrapper_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return at::native::mcpu::empty_strided(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor wrapper_as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  return at::native::mcpu::as_strided(self, size, stride, storage_offset);
}

const at::Tensor& wrapper_resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::mcpu::resize_(self, size, memory_format);
}

at::Tensor wrapper__reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::mcpu::_reshape_alias(self, size, stride);
}

at::Tensor& wrapper_copy_(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  // Native copy_ handles identity before dispatching _copy_from, so a lazy
  // view copied to itself cannot enter the math-bit fallback recursively.
  if (dst.is_same(src)) {
    return at::native::copy_(dst, src, non_blocking);
  }
  // copy_ falls through Negative/Conjugate dispatch. Reject unsupported lazy
  // views before _copy_from's math-bit fallback clones them and re-enters
  // copy_. CPU sources can be cloned safely by that fallback; PrivateUse1
  // dispatch also covers CPU-to-MCPU copies and must keep accepting them.
  TORCH_CHECK(
      (src.is_cpu() || (!src.is_neg() && !src.is_conj())) && !dst.is_neg() &&
          !dst.is_conj(),
      "mcpu::copy_: lazy negative/conjugate views are unsupported");
  // Call the native function directly; redispatching copy_ would recurse.
  return at::native::copy_(dst, src, non_blocking);
}

at::Tensor wrapper__copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  return at::native::mcpu::_copy_from(self, dst, non_blocking);
}

at::Tensor wrapper__copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  return at::native::mcpu::_copy_from_and_resize(self, dst);
}

at::Scalar wrapper__local_scalar_densor(const at::Tensor& self) {
  return at::native::mcpu::_local_scalar_dense(self);
}

at::Tensor& wrapper_set_source_Tensor_(
    at::Tensor& self,
    const at::Tensor& source) {
  return at::native::mcpu::set_source_Tensor_(self, source);
}

at::Tensor& wrapper_set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::mcpu::set_source_Storage_(self, source);
}

at::Tensor& wrapper_set_source_Storage_storage_offsetset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::native::mcpu::set_source_Storage_storage_offset_(
      result, storage, storage_offset, size, stride);
}

at::Tensor wrapper_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::mcpu::view(self, size);
}

at::Tensor wrapper_unfold(
    const at::Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  return at::native::mcpu::unfold(self, dimension, size, step);
}

void wrapper_record_stream(at::Tensor& self, c10::Stream stream) {
  c10::mcpu::recordStream(
      self.storage().data_ptr(), c10::mcpu::McpuStream{stream});
}

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
// LITERALINCLUDE START: FALLBACK WRAPPER
void wrapper_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::native::mcpu::cpu_fallback(op, stack);
}
// LITERALINCLUDE END: FALLBACK WRAPPER
#else
void wrapper_missing_mcpu_kernel(
    const c10::OperatorHandle& op,
    torch::jit::Stack*) {
  const auto& op_name = op.schema().operator_name();
  TORCH_CHECK(
      false,
      "Operator '",
      op_name,
      "' is not implemented for torch_mcpu. CPU fallback is disabled because ",
      "it is incompatible with mcpu stream execution. Please migrate this ATen ",
      "operator into torch_mcpu following docs/how_to_impl_aten_ops.md.");
}
#endif

} // namespace

// LITERALINCLUDE START: TORCH_LIBRARY_IMPL DEFAULT
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", wrapper_empty_memory_format);
  m.impl("empty_strided", wrapper_empty_strided);
  m.impl("as_strided", wrapper_as_strided);
  m.impl("resize_", wrapper_resize_);
  m.impl("_reshape_alias", wrapper__reshape_alias);
  m.impl("copy_", wrapper_copy_);
  m.impl("_copy_from", wrapper__copy_from);
  m.impl("_copy_from_and_resize", wrapper__copy_from_and_resize);
  m.impl("_local_scalar_dense", wrapper__local_scalar_densor);
  m.impl("set_.source_Tensor", wrapper_set_source_Tensor_);
  m.impl("set_.source_Storage", wrapper_set_source_Storage_);
  m.impl(
      "set_.source_Storage_storage_offset",
      wrapper_set_source_Storage_storage_offsetset_);
  m.impl("view", wrapper_view);
  m.impl("unfold", wrapper_unfold);
  m.impl("record_stream", wrapper_record_stream);
}
// LITERALINCLUDE END: TORCH_LIBRARY_IMPL DEFAULT

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
// LITERALINCLUDE START: FALLBACK GLOBAL
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&wrapper_cpu_fallback>());
}
// LITERALINCLUDE END: FALLBACK GLOBAL

// LITERALINCLUDE START: FALLBACK SINGLE
// TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//   m.impl(
//       "sub.Tensor",
//       torch::CppFunction::makeFromBoxedFunction<&wrapper_cpu_fallback>());
// }
// LITERALINCLUDE END: FALLBACK SINGLE
#else
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &wrapper_missing_mcpu_kernel>());
}
#endif

} // namespace at::mcpu
