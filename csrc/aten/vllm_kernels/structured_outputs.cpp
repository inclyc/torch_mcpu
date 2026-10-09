// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for
// vllm/v1/worker/gpu/structured_outputs.py::_apply_grammar_bitmask_kernel

#include "common.h"

namespace {

template <typename scalar_t>
static void vllm_apply_grammar_bitmask_typed(
    scalar_t* logits_ptr,
    int64_t num_masks,
    int64_t logits_stride,
    const int32_t* idx_ptr,
    const int32_t* cu_num_logits_ptr,
    const int32_t* bitmask_ptr,
    int64_t bitmask_stride,
    int64_t vocab_size,
    int64_t mask_stride) {
  const scalar_t neg_inf =
      static_cast<scalar_t>(-std::numeric_limits<float>::infinity());
  int64_t num_words = (vocab_size + 31) / 32;

  for (int64_t mask_idx = 0; mask_idx < num_masks; mask_idx++) {
    const int64_t mapping = idx_ptr[mask_idx];
    const int64_t req_idx = mapping / mask_stride;
    const int64_t position = mapping % mask_stride;
    const int64_t start = cu_num_logits_ptr[req_idx];
    const int64_t end = cu_num_logits_ptr[req_idx + 1];
    // Adaptive speculative verification can remove scheduled positions. Do
    // not form an output pointer until we know that this position is active.
    if (position >= end - start)
      continue;
    const int64_t row = start + position;
    scalar_t* row_ptr = logits_ptr + row * logits_stride;
    const int32_t* mask_row = bitmask_ptr + mask_idx * bitmask_stride;

    for (int64_t w = 0; w < num_words; w++) {
      uint32_t packed = (uint32_t)mask_row[w];
      int64_t base = w * 32;
      int lim = (int)((base + 32 <= vocab_size) ? 32 : vocab_size - base);
      for (int b = 0; b < lim; b++) {
        if (((packed >> b) & 1u) == 0)
          row_ptr[base + b] = neg_inf;
      }
    }
  }
}

void vllm_apply_grammar_bitmask_impl(
    at::Tensor& logits,
    const at::Tensor& logits_indices,
    const at::Tensor& cu_num_logits,
    const at::Tensor& bitmask,
    int64_t vocab_size,
    int64_t mask_stride) {
  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(logits_indices, 1, "logits_indices");
  VLLM_MCPU_CHECK_DTYPE(logits_indices, at::kInt, "logits_indices");
  VLLM_MCPU_CHECK_DIM(bitmask, 2, "bitmask");
  VLLM_MCPU_CHECK_DTYPE(bitmask, at::kInt, "bitmask");

  int64_t num_masks = logits_indices.size(0);
  if (num_masks == 0 || vocab_size == 0)
    return;

  int64_t logits_stride = logits.stride(0);
  const int32_t* idx_ptr = logits_indices.data_ptr<int32_t>();
  const int32_t* cu_num_logits_ptr = cu_num_logits.data_ptr<int32_t>();
  const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
  int64_t bitmask_stride = bitmask.stride(0);

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_apply_grammar_bitmask", {
    scalar_t* logits_ptr = logits.data_ptr<scalar_t>();
    at::mcpu::launch_timed_kernel(
        "mcpu::vllm_apply_grammar_bitmask",
        [logits_ptr,
         num_masks,
         logits_stride,
         idx_ptr,
         cu_num_logits_ptr,
         bitmask_ptr,
         bitmask_stride,
         vocab_size,
         mask_stride](at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::vllm_apply_grammar_bitmask", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {logits_ptr, idx_ptr, cu_num_logits_ptr, bitmask_ptr});
          vllm_apply_grammar_bitmask_typed<scalar_t>(
              logits_ptr,
              num_masks,
              logits_stride,
              idx_ptr,
              cu_num_logits_ptr,
              bitmask_ptr,
              bitmask_stride,
              vocab_size,
              mask_stride);
        });
  });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_apply_grammar_bitmask("
      "Tensor(a!) logits, Tensor logits_indices, Tensor cu_num_logits, "
      "Tensor bitmask, int vocab_size, int mask_stride"
      ") -> ()");
}
TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_apply_grammar_bitmask", &vllm_apply_grammar_bitmask_impl);
}
