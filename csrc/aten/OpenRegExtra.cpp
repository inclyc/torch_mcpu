#include "native/Extra.h"

#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>

#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/library.h>

namespace at::mcpu {

namespace {
at::Tensor wrapper_quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::mcpu::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t wrapper__fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  return at::native::mcpu::_fused_sdp_choice(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

void wrapper_quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  at::native::mcpu::quantize_tensor_per_tensor_affine_stub(
      rtensor, qtensor, scale, zero_point);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
wrapper__scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  return at::native::mcpu::_scaled_dot_product_fused_attention_overrideable(
      query,
      key,
      value,
      attn_bias,
      dropout_p,
      is_causal,
      return_debug_mask,
      scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
wrapper_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return at::native::mcpu::
      _scaled_dot_product_fused_attention_overrideable_backward(
          grad_out,
          query,
          key,
          value,
          attn_bias,
          grad_input_mask,
          out,
          logsumexp,
          cum_seq_q,
          cum_seq_k,
          max_q,
          max_k,
          dropout_p,
          is_causal,
          philox_seed,
          philox_offset,
          scale);
}

at::Tensor wrapper_custom_autograd_fn_returns_self(at::Tensor x) {
  return at::native::mcpu::custom_autograd_fn_returns_self(x);
}

at::Tensor wrapper_custom_autograd_fn_aliasing(at::Tensor x) {
  return at::native::mcpu::custom_autograd_fn_aliasing(x);
}

at::Tensor& wrapper_abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::mcpu::abs_out(self, out);
}

void wrapper_abs_stub(at::TensorIteratorBase& iter) {
  at::native::mcpu::abs_kernel(iter);
}

at::Tensor wrapper_custom_abs(at::Tensor x) {
  return at::native::mcpu::custom_abs(x);
}
} // namespace

using namespace at::native;
// Registration via STUB
// LITERALINCLUDE START: STUB DEFAULT
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &wrapper_abs_stub);
REGISTER_PRIVATEUSE1_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &wrapper_quantize_tensor_per_tensor_affine_stub);
REGISTER_PRIVATEUSE1_DISPATCH(
    _fused_sdp_choice_stub,
    &wrapper__fused_sdp_choice);
// LITERALINCLUDE END: STUB DEFAULT

// Registration of custom operators
// LITERALINCLUDE START: CUSTOM OPERATOR SCHEMA
TORCH_LIBRARY(mcpu, m) {
  m.def("custom_abs(Tensor input)-> Tensor");
  m.def(
      "stream_sleep_fill_(Tensor(a!) input, int value, int sleep_ms) -> Tensor(a!)");
  m.def(
      "stream_sleep_copy_(Tensor(a!) dst, Tensor src, int sleep_ms) -> Tensor(a!)");
  m.def("first_element_int(Tensor input) -> int");
  m.def(
      "kernel_launch_condition_counts(Tensor stream_selector) -> (int, int, int)");
  m.def(
      "kernel_launch_condition_lifetime_counts(Tensor stream_selector, bool enabled, bool fail_submission) -> (int, int, int)");
  m.def(
      "kernel_launch_parallel_condition_counts(Tensor stream_selector, int launches) -> (int, int)");
  m.def(
      "kernel_launch_lifetime_counts("
      "Tensor stream_selector, int launches"
      ") -> (int, int, int)");
  m.def(
      "kernel_launch_failed_lifetime_counts("
      "Tensor stream_selector"
      ") -> (int, int, int)");
}
// LITERALINCLUDE END: CUSTOM OPERATOR SCHEMA

// LITERALINCLUDE START: CUSTOM OPERATOR DEFAULT
TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("custom_abs", &wrapper_custom_abs);
  m.impl("stream_sleep_fill_", &at::native::mcpu::stream_sleep_fill_);
  m.impl("stream_sleep_copy_", &at::native::mcpu::stream_sleep_copy_);
  m.impl("first_element_int", &at::native::mcpu::first_element_int);
  m.impl(
      "kernel_launch_condition_counts",
      &at::native::mcpu::kernel_launch_condition_counts);
  m.impl(
      "kernel_launch_condition_lifetime_counts",
      &at::native::mcpu::kernel_launch_condition_lifetime_counts);
  m.impl(
      "kernel_launch_parallel_condition_counts",
      &at::native::mcpu::kernel_launch_parallel_condition_counts);
  m.impl(
      "kernel_launch_lifetime_counts",
      &at::native::mcpu::kernel_launch_lifetime_counts);
  m.impl(
      "kernel_launch_failed_lifetime_counts",
      &at::native::mcpu::kernel_launch_failed_lifetime_counts);
}
// LITERALINCLUDE END: CUSTOM OPERATOR DEFAULT

// The rest is for testing purposes
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("quantize_per_tensor", &wrapper_quantize_per_tensor);
  m.impl("_fused_sdp_choice", &wrapper__fused_sdp_choice);
  m.impl(
      "_scaled_dot_product_fused_attention_overrideable",
      &wrapper__scaled_dot_product_fused_attention_overrideable);
  m.impl(
      "_scaled_dot_product_fused_attention_overrideable_backward",
      &wrapper_scaled_dot_product_fused_attention_overrideable_backward);
}

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def("custom_autograd_fn_returns_self(Tensor input)-> Tensor");
  m.def("custom_autograd_fn_aliasing(Tensor(a) input)-> Tensor(a)");
}

TORCH_LIBRARY_IMPL(mcpu, AutogradPrivateUse1, m) {
  m.impl(
      "custom_autograd_fn_returns_self",
      &wrapper_custom_autograd_fn_returns_self);
  m.impl("custom_autograd_fn_aliasing", &wrapper_custom_autograd_fn_aliasing);
}

} // namespace at::mcpu
