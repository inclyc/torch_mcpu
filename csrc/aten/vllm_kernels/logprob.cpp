// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/sample/logprob.py:
//   _ranks_kernel             — rank of sampled token in the logit distribution
//   _fill_logprob_token_ids_kernel — select sampled/top-k/custom token IDs

#include <algorithm>
#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// _ranks_kernel
// output[req] = number of logits >= logits[req, token_ids[req]]
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_ranks_kernel_typed(
    int64_t* out_ptr,
    const scalar_t* logits_ptr,
    int64_t batch,
    int64_t logits_stride,
    const int64_t* ids_ptr,
    int64_t vocab_size) {
  for (int64_t req = 0; req < batch; req++) {
    const scalar_t* row = logits_ptr + req * logits_stride;
    int64_t tid = ids_ptr[req];
    int64_t rank = 0;
    // Triton assumes a valid sampled token.  Keep malformed/padded warmup
    // inputs from turning that undefined access into a host-side crash.
    if (tid < 0 || tid >= vocab_size) {
      out_ptr[req] = 0;
      continue;
    }
    if constexpr (std::is_same_v<scalar_t, float>) {
      float x = row[tid];
      for (int64_t i = 0; i < vocab_size; i++) {
        if (row[i] >= x)
          rank++;
      }
    } else {
      float x = static_cast<float>(row[tid]);
      for (int64_t i = 0; i < vocab_size; i++) {
        if (static_cast<float>(row[i]) >= x)
          rank++;
      }
    }
    out_ptr[req] = rank;
  }
}

void vllm_ranks_kernel_impl(
    at::Tensor& output, // [batch], int64
    const at::Tensor& logits, // [batch, vocab_size], any float
    const at::Tensor& token_ids, // [batch], int64
    int64_t vocab_size) {
  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(output, 1, "output");
  VLLM_MCPU_CHECK_DTYPE(output, at::kLong, "output");
  VLLM_MCPU_CHECK_DIM(token_ids, 1, "token_ids");
  VLLM_MCPU_CHECK_DTYPE(token_ids, at::kLong, "token_ids");

  int64_t batch = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  const int64_t* ids_ptr = token_ids.data_ptr<int64_t>();
  int64_t* out_ptr = output.data_ptr<int64_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_ranks_kernel", {
    const scalar_t* logits_ptr = logits.data_ptr<scalar_t>();
    at::mcpu::launch_timed_kernel(
        "mcpu::vllm_ranks_kernel",
        [out_ptr, logits_ptr, batch, logits_stride, ids_ptr, vocab_size](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::vllm_ranks_kernel", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {out_ptr, logits_ptr, ids_ptr});
          vllm_ranks_kernel_typed<scalar_t>(
              out_ptr, logits_ptr, batch, logits_stride, ids_ptr, vocab_size);
        });
  });
}

// ---------------------------------------------------------------------------
// _fill_logprob_token_ids_kernel
// ---------------------------------------------------------------------------
void vllm_fill_logprob_token_ids_kernel_impl(
    at::Tensor& out_token_ids,
    int64_t out_token_ids_stride,
    at::Tensor& out_valid_mask,
    int64_t out_valid_mask_stride,
    const at::Tensor& sampled_token_ids,
    const at::Tensor& topk_indices,
    int64_t topk_indices_stride,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& num_per_req_token_ids,
    const at::Tensor& per_req_token_ids,
    int64_t per_req_token_ids_stride,
    int64_t num_topk,
    int64_t padded_cols) {
  VLLM_MCPU_CHECK_DIM(out_token_ids, 2, "out_token_ids");
  VLLM_MCPU_CHECK_DTYPE(out_token_ids, at::kLong, "out_token_ids");
  VLLM_MCPU_CHECK_DIM(out_valid_mask, 2, "out_valid_mask");
  VLLM_MCPU_CHECK_DTYPE(out_valid_mask, at::kBool, "out_valid_mask");
  VLLM_MCPU_CHECK_DTYPE(sampled_token_ids, at::kLong, "sampled_token_ids");
  VLLM_MCPU_CHECK_DTYPE(topk_indices, at::kInt, "topk_indices");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(
      num_per_req_token_ids, at::kInt, "num_per_req_token_ids");
  VLLM_MCPU_CHECK_DTYPE(per_req_token_ids, at::kInt, "per_req_token_ids");
  VLLM_MCPU_CHECK(
      num_topk >= 0 && num_topk <= topk_indices.size(1),
      "num_topk out of range");
  VLLM_MCPU_CHECK(padded_cols > 0, "padded_cols must be positive");

  int64_t batch = sampled_token_ids.size(0);
  VLLM_MCPU_CHECK(
      out_token_ids.size(0) == batch && out_valid_mask.size(0) == batch,
      "output batch size mismatch");

  int64_t* out_ids_ptr = out_token_ids.data_ptr<int64_t>();
  bool* out_mask_ptr = out_valid_mask.data_ptr<bool>();
  const int64_t* sampled_ptr = sampled_token_ids.data_ptr<int64_t>();
  const int32_t* topk_ptr = topk_indices.data_ptr<int32_t>();
  const int32_t* mapping_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const int32_t* num_custom_ptr = num_per_req_token_ids.data_ptr<int32_t>();
  const int32_t* custom_ptr = per_req_token_ids.data_ptr<int32_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_fill_logprob_token_ids_kernel",
      [batch,
       num_topk,
       out_ids_ptr,
       out_mask_ptr,
       sampled_ptr,
       topk_ptr,
       mapping_ptr,
       num_custom_ptr,
       custom_ptr,
       out_token_ids_stride,
       out_valid_mask_stride,
       topk_indices_stride,
       per_req_token_ids_stride,
       padded_cols](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_fill_logprob_token_ids_kernel", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {out_ids_ptr,
             out_mask_ptr,
             sampled_ptr,
             topk_ptr,
             mapping_ptr,
             num_custom_ptr,
             custom_ptr});
        for (int64_t batch_idx = 0; batch_idx < batch; batch_idx++) {
          int64_t* out_ids_row = out_ids_ptr + batch_idx * out_token_ids_stride;
          bool* out_mask_row = out_mask_ptr + batch_idx * out_valid_mask_stride;
          out_ids_row[0] = sampled_ptr[batch_idx];
          out_mask_row[0] = true;

          int32_t req_idx = mapping_ptr[batch_idx];
          int32_t num_custom = num_custom_ptr[req_idx];
          int64_t count = num_custom > 0 ? num_custom : num_topk;
          count = std::min(count, padded_cols);
          const int32_t* source = num_custom > 0
              ? custom_ptr + (int64_t)req_idx * per_req_token_ids_stride
              : topk_ptr + batch_idx * topk_indices_stride;
          for (int64_t col = 0; col < count; col++) {
            out_ids_row[col + 1] = source[col];
            out_mask_row[col + 1] = true;
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_ranks_kernel("
      "Tensor(a!) output, "
      "Tensor logits, "
      "Tensor token_ids, "
      "int vocab_size"
      ") -> ()");
  m.def(
      "vllm_fill_logprob_token_ids_kernel("
      "Tensor(a!) out_token_ids, "
      "int out_token_ids_stride, "
      "Tensor(b!) out_valid_mask, "
      "int out_valid_mask_stride, "
      "Tensor sampled_token_ids, "
      "Tensor topk_indices, "
      "int topk_indices_stride, "
      "Tensor expanded_idx_mapping, "
      "Tensor num_per_req_token_ids, "
      "Tensor per_req_token_ids, "
      "int per_req_token_ids_stride, "
      "int num_topk, "
      "int padded_cols"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_ranks_kernel", &vllm_ranks_kernel_impl);
  m.impl(
      "vllm_fill_logprob_token_ids_kernel",
      &vllm_fill_logprob_token_ids_kernel_impl);
}
