// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/model_states/mamba_hybrid.py.

#include "common.h"

#include <algorithm>
#include <cstring>
#include <optional>

namespace {

void check_int32_vector(const at::Tensor& tensor, const char* name) {
  VLLM_MCPU_CHECK(tensor.dim() == 1, name, " must be 1D");
  VLLM_MCPU_CHECK(
      tensor.scalar_type() == at::kInt, name, " must have int32 dtype");
  VLLM_MCPU_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int64_vector(const at::Tensor& tensor, const char* name) {
  VLLM_MCPU_CHECK(tensor.dim() == 1, name, " must be 1D");
  VLLM_MCPU_CHECK(
      tensor.scalar_type() == at::kLong, name, " must have int64 dtype");
  VLLM_MCPU_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_mamba_copy_metadata(
    const at::Tensor& block_table_ptrs,
    const at::Tensor& state_base_addrs,
    const at::Tensor& state_block_strides,
    const at::Tensor& state_elem_sizes,
    const at::Tensor& state_inner_sizes,
    const at::Tensor& state_conv_widths,
    const at::Tensor& state_group_indices,
    const at::Tensor& state_dim_row_count,
    const at::Tensor& state_dim_row_stride,
    int64_t block_table_stride,
    int64_t total_states) {
  check_int64_vector(block_table_ptrs, "block_table_ptrs");
  check_int64_vector(state_base_addrs, "state_base_addrs");
  check_int64_vector(state_block_strides, "state_block_strides");
  check_int32_vector(state_elem_sizes, "state_elem_sizes");
  check_int64_vector(state_inner_sizes, "state_inner_sizes");
  check_int32_vector(state_conv_widths, "state_conv_widths");
  check_int32_vector(state_group_indices, "state_group_indices");
  check_int32_vector(state_dim_row_count, "state_dim_row_count");
  check_int64_vector(state_dim_row_stride, "state_dim_row_stride");
  VLLM_MCPU_CHECK(
      block_table_stride > 0, "block_table_stride must be positive");
  VLLM_MCPU_CHECK(total_states >= 0, "total_states must be non-negative");
  for (const at::Tensor* tensor :
       {&state_base_addrs,
        &state_block_strides,
        &state_elem_sizes,
        &state_inner_sizes,
        &state_conv_widths,
        &state_group_indices,
        &state_dim_row_count,
        &state_dim_row_stride}) {
    VLLM_MCPU_CHECK(
        tensor->numel() == total_states,
        "all state metadata tensors must match the state grid size");
  }
}

void copy_mamba_state_block(
    int64_t state_index,
    int64_t block_table_row,
    int32_t source_column,
    int32_t destination_column,
    int32_t token_bias,
    const int64_t* block_table_ptrs,
    int64_t num_groups,
    int64_t block_table_stride,
    const int64_t* state_base_addrs,
    const int64_t* state_block_strides,
    const int32_t* state_elem_sizes,
    const int64_t* state_inner_sizes,
    const int32_t* state_conv_widths,
    const int32_t* state_group_indices,
    const int32_t* state_dim_row_count,
    const int64_t* state_dim_row_stride,
    bool conv_state_dim_first) {
  const int32_t group_index = state_group_indices[state_index];
  VLLM_MCPU_CHECK(
      0 <= group_index && group_index < num_groups,
      "state_group_indices contains an out-of-range group");
  VLLM_MCPU_CHECK(
      source_column >= 0 && source_column < block_table_stride,
      "source mamba block column is out of range");
  VLLM_MCPU_CHECK(
      destination_column >= 0 && destination_column < block_table_stride,
      "destination mamba block column is out of range");
  VLLM_MCPU_CHECK(
      block_table_ptrs[group_index] != 0,
      "block_table_ptrs contains a null pointer");
  VLLM_MCPU_CHECK(
      state_base_addrs[state_index] != 0,
      "state_base_addrs contains a null pointer");

  const int32_t* block_table =
      reinterpret_cast<const int32_t*>(block_table_ptrs[group_index]) +
      block_table_row * block_table_stride;
  const int64_t source_block = block_table[source_column];
  const int64_t destination_block = block_table[destination_column];
  const int64_t block_stride = state_block_strides[state_index];
  const int64_t element_size = state_elem_sizes[state_index];
  const int64_t inner_size = state_inner_sizes[state_index];
  const int32_t conv_width = state_conv_widths[state_index];
  VLLM_MCPU_CHECK(source_block >= 0, "source block id must be non-negative");
  VLLM_MCPU_CHECK(
      destination_block >= 0, "destination block id must be non-negative");
  VLLM_MCPU_CHECK(block_stride > 0, "state block stride must be positive");
  VLLM_MCPU_CHECK(element_size > 0, "state element size must be positive");
  VLLM_MCPU_CHECK(inner_size > 0, "state inner size must be positive");
  VLLM_MCPU_CHECK(token_bias >= 0, "mamba token bias must be non-negative");

  auto* state_base =
      reinterpret_cast<unsigned char*>(state_base_addrs[state_index]);
  auto* destination = state_base + destination_block * block_stride;

  if (conv_width > 0) {
    VLLM_MCPU_CHECK(
        token_bias <= conv_width, "mamba token bias exceeds conv width");
    auto* source = state_base + source_block * block_stride;
    if (conv_state_dim_first) {
      const int32_t row_count = state_dim_row_count[state_index];
      const int64_t row_stride = state_dim_row_stride[state_index];
      VLLM_MCPU_CHECK(row_count > 0, "DS conv row count must be positive");
      VLLM_MCPU_CHECK(row_stride > 0, "DS conv row stride must be positive");
      const int64_t copy_bytes =
          (static_cast<int64_t>(conv_width) - token_bias) * element_size;
      const int64_t bias_bytes =
          static_cast<int64_t>(token_bias) * element_size;
      for (int32_t row = 0; row < row_count; ++row) {
        std::memmove(
            destination + static_cast<int64_t>(row) * row_stride,
            source + static_cast<int64_t>(row) * row_stride + bias_bytes,
            copy_bytes);
      }
      return;
    }

    const int64_t bias_bytes =
        static_cast<int64_t>(token_bias) * inner_size * element_size;
    const int64_t copy_bytes = (static_cast<int64_t>(conv_width) - token_bias) *
        inner_size * element_size;
    std::memmove(destination, source + bias_bytes, copy_bytes);
    return;
  }

  const int64_t temporal_source_column =
      static_cast<int64_t>(source_column) + token_bias;
  VLLM_MCPU_CHECK(
      temporal_source_column < block_table_stride,
      "temporal source mamba block column is out of range");
  const int64_t temporal_source_block = block_table[temporal_source_column];
  VLLM_MCPU_CHECK(
      temporal_source_block >= 0,
      "temporal source block id must be non-negative");
  const auto* source = state_base + temporal_source_block * block_stride;
  std::memmove(destination, source, inner_size * element_size);
}

void vllm_preprocess_mamba_align_impl(
    const at::Tensor& idx_mapping,
    at::Tensor& state_idx,
    const at::Tensor& num_computed_tokens,
    const at::Tensor& query_start_loc,
    at::Tensor& num_accepted_tokens,
    at::Tensor& src_col,
    at::Tensor& src_off,
    int64_t num_reqs,
    int64_t mamba_block_size) {
  check_int32_vector(idx_mapping, "idx_mapping");
  check_int32_vector(state_idx, "state_idx");
  check_int32_vector(num_computed_tokens, "num_computed_tokens");
  check_int32_vector(query_start_loc, "query_start_loc");
  check_int32_vector(num_accepted_tokens, "num_accepted_tokens");
  check_int32_vector(src_col, "src_col");
  check_int32_vector(src_off, "src_off");
  VLLM_MCPU_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  VLLM_MCPU_CHECK(
      num_reqs <= idx_mapping.numel(), "idx_mapping must cover num_reqs");
  VLLM_MCPU_CHECK(
      query_start_loc.numel() >= num_reqs + 1,
      "query_start_loc must contain num_reqs + 1 entries");
  VLLM_MCPU_CHECK(mamba_block_size > 0, "mamba_block_size must be positive");
  VLLM_MCPU_CHECK(
      state_idx.numel() == num_computed_tokens.numel() &&
          state_idx.numel() == num_accepted_tokens.numel() &&
          state_idx.numel() == src_col.numel() &&
          state_idx.numel() == src_off.numel(),
      "per-request-slot mamba tensors must have equal lengths");

  const auto* idx_ptr = idx_mapping.data_ptr<int32_t>();
  auto* state_ptr = state_idx.data_ptr<int32_t>();
  const auto* computed_ptr = num_computed_tokens.data_ptr<int32_t>();
  const auto* query_ptr = query_start_loc.data_ptr<int32_t>();
  auto* accepted_ptr = num_accepted_tokens.data_ptr<int32_t>();
  auto* src_col_ptr = src_col.data_ptr<int32_t>();
  auto* src_off_ptr = src_off.data_ptr<int32_t>();
  const int64_t max_num_reqs = state_idx.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_preprocess_mamba_align",
      [idx_ptr,
       state_ptr,
       computed_ptr,
       query_ptr,
       accepted_ptr,
       src_col_ptr,
       src_off_ptr,
       num_reqs,
       max_num_reqs,
       mamba_block_size](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_preprocess_mamba_align", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {idx_ptr,
             state_ptr,
             computed_ptr,
             query_ptr,
             accepted_ptr,
             src_col_ptr,
             src_off_ptr});
#pragma omp parallel for schedule(static)
        for (int64_t batch_index = 0; batch_index < num_reqs; ++batch_index) {
          const int32_t request_index = idx_ptr[batch_index];
          VLLM_MCPU_CHECK(
              0 <= request_index && request_index < max_num_reqs,
              "idx_mapping contains an out-of-range request index");
          const int32_t old_state_index = state_ptr[request_index];
          const int32_t accepted = accepted_ptr[request_index];
          src_col_ptr[request_index] = old_state_index;
          src_off_ptr[request_index] = std::max(accepted - 1, 0);
          const int64_t query_length =
              static_cast<int64_t>(query_ptr[batch_index + 1]) -
              query_ptr[batch_index];
          const int64_t computed_after =
              static_cast<int64_t>(computed_ptr[request_index]) + query_length;
          const int64_t new_state_index =
              (computed_after + mamba_block_size - 1) / mamba_block_size - 1;
          VLLM_MCPU_CHECK(
              new_state_index >= std::numeric_limits<int32_t>::min() &&
                  new_state_index <= std::numeric_limits<int32_t>::max(),
              "computed mamba state index exceeds int32");
          state_ptr[request_index] = static_cast<int32_t>(new_state_index);
          if (old_state_index >= 0 && old_state_index != new_state_index) {
            accepted_ptr[request_index] = 1;
          }
        }
      });
}

void vllm_precopy_mamba_align_impl(
    const at::Tensor& state_idx,
    const at::Tensor& src_col,
    const at::Tensor& token_bias,
    const at::Tensor& block_table_ptrs,
    int64_t block_table_stride,
    const at::Tensor& state_base_addrs,
    const at::Tensor& state_block_strides,
    const at::Tensor& state_elem_sizes,
    const at::Tensor& state_inner_sizes,
    const at::Tensor& state_conv_widths,
    const at::Tensor& state_group_indices,
    const at::Tensor& state_dim_row_count,
    const at::Tensor& state_dim_row_stride,
    const at::Tensor& idx_mapping,
    int64_t num_reqs,
    int64_t total_states,
    bool conv_state_dim_first) {
  check_int32_vector(state_idx, "state_idx");
  check_int32_vector(src_col, "src_col");
  check_int32_vector(token_bias, "token_bias");
  check_int32_vector(idx_mapping, "idx_mapping");
  check_mamba_copy_metadata(
      block_table_ptrs,
      state_base_addrs,
      state_block_strides,
      state_elem_sizes,
      state_inner_sizes,
      state_conv_widths,
      state_group_indices,
      state_dim_row_count,
      state_dim_row_stride,
      block_table_stride,
      total_states);
  VLLM_MCPU_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  VLLM_MCPU_CHECK(
      num_reqs <= idx_mapping.numel(), "idx_mapping must cover num_reqs");
  VLLM_MCPU_CHECK(
      state_idx.numel() == src_col.numel() &&
          state_idx.numel() == token_bias.numel(),
      "per-request-slot mamba tensors must have equal lengths");

  const auto* state_ptr = state_idx.data_ptr<int32_t>();
  const auto* src_col_ptr = src_col.data_ptr<int32_t>();
  const auto* token_bias_ptr = token_bias.data_ptr<int32_t>();
  const auto* block_table_ptr = block_table_ptrs.data_ptr<int64_t>();
  const auto* base_ptr = state_base_addrs.data_ptr<int64_t>();
  const auto* block_stride_ptr = state_block_strides.data_ptr<int64_t>();
  const auto* elem_size_ptr = state_elem_sizes.data_ptr<int32_t>();
  const auto* inner_size_ptr = state_inner_sizes.data_ptr<int64_t>();
  const auto* conv_width_ptr = state_conv_widths.data_ptr<int32_t>();
  const auto* group_ptr = state_group_indices.data_ptr<int32_t>();
  const auto* row_count_ptr = state_dim_row_count.data_ptr<int32_t>();
  const auto* row_stride_ptr = state_dim_row_stride.data_ptr<int64_t>();
  const auto* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int64_t max_num_reqs = state_idx.numel();
  const int64_t num_groups = block_table_ptrs.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_precopy_mamba_align",
      [=](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_precopy_mamba_align", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        for (int64_t batch_index = 0; batch_index < num_reqs; ++batch_index) {
          for (int64_t state_index = 0; state_index < total_states;
               ++state_index) {
            const int32_t request_index = idx_ptr[batch_index];
            if (request_index < 0) {
              continue;
            }
            VLLM_MCPU_CHECK(
                request_index < max_num_reqs,
                "idx_mapping contains an out-of-range request index");
            const int32_t source_column = src_col_ptr[request_index];
            const int32_t destination_column = state_ptr[request_index];
            if (source_column < 0 || source_column == destination_column) {
              continue;
            }
            copy_mamba_state_block(
                state_index,
                batch_index,
                source_column,
                destination_column,
                token_bias_ptr[request_index],
                block_table_ptr,
                num_groups,
                block_table_stride,
                base_ptr,
                block_stride_ptr,
                elem_size_ptr,
                inner_size_ptr,
                conv_width_ptr,
                group_ptr,
                row_count_ptr,
                row_stride_ptr,
                conv_state_dim_first);
          }
        }
      });
}

void vllm_postprocess_mamba_impl(
    at::Tensor& num_accepted_tokens,
    const at::Tensor& state_idx,
    const std::optional<at::Tensor>& num_scheduled_tokens,
    const at::Tensor& num_computed_tokens,
    const std::optional<at::Tensor>& num_draft_tokens,
    const at::Tensor& block_table_ptrs,
    int64_t block_table_stride,
    const at::Tensor& state_base_addrs,
    const at::Tensor& state_block_strides,
    const at::Tensor& state_elem_sizes,
    const at::Tensor& state_inner_sizes,
    const at::Tensor& state_conv_widths,
    const at::Tensor& state_group_indices,
    const at::Tensor& state_dim_row_count,
    const at::Tensor& state_dim_row_stride,
    const std::optional<at::Tensor>& num_accepted_tokens_out,
    const std::optional<at::Tensor>& idx_mapping,
    int64_t num_reqs,
    int64_t total_states,
    int64_t block_size,
    bool conv_state_dim_first,
    bool has_idx_mapping,
    bool precomputed_new_computed) {
  check_int32_vector(num_accepted_tokens, "num_accepted_tokens");
  check_int32_vector(state_idx, "state_idx");
  check_int32_vector(num_computed_tokens, "num_computed_tokens");
  if (num_scheduled_tokens) {
    check_int32_vector(*num_scheduled_tokens, "num_scheduled_tokens");
  }
  if (num_draft_tokens) {
    check_int32_vector(*num_draft_tokens, "num_draft_tokens");
  }
  if (num_accepted_tokens_out) {
    check_int32_vector(*num_accepted_tokens_out, "num_accepted_tokens_out");
  }
  if (idx_mapping) {
    check_int32_vector(*idx_mapping, "idx_mapping");
  }
  check_mamba_copy_metadata(
      block_table_ptrs,
      state_base_addrs,
      state_block_strides,
      state_elem_sizes,
      state_inner_sizes,
      state_conv_widths,
      state_group_indices,
      state_dim_row_count,
      state_dim_row_stride,
      block_table_stride,
      total_states);
  VLLM_MCPU_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  VLLM_MCPU_CHECK(block_size > 0, "block_size must be positive");
  VLLM_MCPU_CHECK(
      has_idx_mapping == idx_mapping.has_value(),
      "HAS_IDX_MAPPING must match idx_mapping presence");
  VLLM_MCPU_CHECK(
      precomputed_new_computed ==
          (!num_scheduled_tokens.has_value() && !num_draft_tokens.has_value()),
      "precomputed mode must match scheduled/draft tensor presence");
  VLLM_MCPU_CHECK(
      has_idx_mapping || num_accepted_tokens_out.has_value(),
      "V1 postprocess requires num_accepted_tokens_out");
  VLLM_MCPU_CHECK(
      !idx_mapping || idx_mapping->numel() >= num_reqs,
      "idx_mapping must cover num_reqs");

  auto* accepted_ptr = num_accepted_tokens.data_ptr<int32_t>();
  const auto* state_ptr = state_idx.data_ptr<int32_t>();
  const auto* scheduled_ptr = num_scheduled_tokens
      ? num_scheduled_tokens->data_ptr<int32_t>()
      : nullptr;
  const auto* computed_ptr = num_computed_tokens.data_ptr<int32_t>();
  const auto* draft_ptr =
      num_draft_tokens ? num_draft_tokens->data_ptr<int32_t>() : nullptr;
  const auto* block_table_ptr = block_table_ptrs.data_ptr<int64_t>();
  const auto* base_ptr = state_base_addrs.data_ptr<int64_t>();
  const auto* block_stride_ptr = state_block_strides.data_ptr<int64_t>();
  const auto* elem_size_ptr = state_elem_sizes.data_ptr<int32_t>();
  const auto* inner_size_ptr = state_inner_sizes.data_ptr<int64_t>();
  const auto* conv_width_ptr = state_conv_widths.data_ptr<int32_t>();
  const auto* group_ptr = state_group_indices.data_ptr<int32_t>();
  const auto* row_count_ptr = state_dim_row_count.data_ptr<int32_t>();
  const auto* row_stride_ptr = state_dim_row_stride.data_ptr<int64_t>();
  auto* accepted_out_ptr = num_accepted_tokens_out
      ? const_cast<int32_t*>(num_accepted_tokens_out->data_ptr<int32_t>())
      : nullptr;
  const auto* idx_ptr =
      idx_mapping ? idx_mapping->data_ptr<int32_t>() : nullptr;
  const int64_t max_num_reqs = num_accepted_tokens.numel();
  const int64_t num_groups = block_table_ptrs.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_postprocess_mamba",
      [=](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_postprocess_mamba", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        for (int64_t batch_index = 0; batch_index < num_reqs; ++batch_index) {
          const int32_t request_index =
              has_idx_mapping ? idx_ptr[batch_index] : batch_index;
          if (request_index < 0) {
            continue;
          }
          VLLM_MCPU_CHECK(
              request_index < max_num_reqs,
              "idx_mapping contains an out-of-range request index");
          // Legacy in-place callers must also preserve the decision across
          // all layer states when state zero resets the acceptance count.
          const int64_t num_accepted = accepted_ptr[request_index];
          for (int64_t state_index = 0; state_index < total_states;
               ++state_index) {
            const int32_t source_block_index = state_ptr[request_index];
            int64_t new_num_computed;
            int64_t num_tokens_running_state;
            if (precomputed_new_computed) {
              new_num_computed = computed_ptr[request_index];
              num_tokens_running_state = new_num_computed - num_accepted + 1;
            } else {
              num_tokens_running_state =
                  static_cast<int64_t>(computed_ptr[request_index]) +
                  scheduled_ptr[request_index] - draft_ptr[request_index];
              new_num_computed = num_tokens_running_state + num_accepted - 1;
            }
            const int64_t aligned_new_computed =
                (new_num_computed / block_size) * block_size;
            if (aligned_new_computed < num_tokens_running_state) {
              continue;
            }
            const int64_t accept_token_bias =
                aligned_new_computed - num_tokens_running_state;
            const int64_t destination_block_index =
                aligned_new_computed / block_size - 1;
            VLLM_MCPU_CHECK(
                accept_token_bias <= std::numeric_limits<int32_t>::max() &&
                    destination_block_index >=
                        std::numeric_limits<int32_t>::min() &&
                    destination_block_index <=
                        std::numeric_limits<int32_t>::max(),
                "postprocess mamba indices exceed int32");
            if (source_block_index == destination_block_index &&
                state_index == 0) {
              if (accepted_out_ptr != nullptr) {
                accepted_out_ptr[request_index] = 1;
              } else {
                accepted_ptr[request_index] = 1;
              }
            }
            if (source_block_index == destination_block_index &&
                accept_token_bias == 0) {
              continue;
            }
            copy_mamba_state_block(
                state_index,
                has_idx_mapping ? batch_index : request_index,
                source_block_index,
                static_cast<int32_t>(destination_block_index),
                static_cast<int32_t>(accept_token_bias),
                block_table_ptr,
                num_groups,
                block_table_stride,
                base_ptr,
                block_stride_ptr,
                elem_size_ptr,
                inner_size_ptr,
                conv_width_ptr,
                group_ptr,
                row_count_ptr,
                row_stride_ptr,
                conv_state_dim_first);
          }
        }
      });
}

void vllm_scatter_num_accepted_impl(
    const at::Tensor& idx_mapping,
    const at::Tensor& num_sampled,
    at::Tensor& num_accepted) {
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(num_sampled, 1, "num_sampled");
  VLLM_MCPU_CHECK_DIM(num_accepted, 1, "num_accepted");
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(num_sampled, at::kInt, "num_sampled");
  VLLM_MCPU_CHECK_DTYPE(num_accepted, at::kInt, "num_accepted");
  VLLM_MCPU_CHECK(
      idx_mapping.numel() == num_sampled.numel(),
      "idx_mapping and num_sampled must have equal lengths");

  const int64_t num_reqs = idx_mapping.numel();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* sampled_ptr = num_sampled.data_ptr<int32_t>();
  int32_t* accepted_ptr = num_accepted.data_ptr<int32_t>();
  const int64_t max_num_reqs = num_accepted.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_scatter_num_accepted",
      [idx_ptr, sampled_ptr, accepted_ptr, num_reqs, max_num_reqs](
          at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_scatter_num_accepted", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {idx_ptr, sampled_ptr, accepted_ptr});
        for (int64_t row = 0; row < num_reqs; ++row) {
          const int32_t request_index = idx_ptr[row];
          // Match Triton: -1 is a filtered pipeline-parallel row and must
          // not write the destination.
          if (request_index < 0) {
            continue;
          }
          if (request_index >= max_num_reqs) {
            continue;
          }
          accepted_ptr[request_index] = std::max(sampled_ptr[row], int32_t{1});
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_preprocess_mamba_align("
      "Tensor idx_mapping, "
      "Tensor(a!) state_idx, "
      "Tensor num_computed_tokens, "
      "Tensor query_start_loc, "
      "Tensor(b!) num_accepted_tokens, "
      "Tensor(c!) src_col, "
      "Tensor(d!) src_off, "
      "int num_reqs, "
      "int mamba_block_size"
      ") -> ()");
  m.def(
      "vllm_precopy_mamba_align("
      "Tensor state_idx, "
      "Tensor src_col, "
      "Tensor token_bias, "
      "Tensor block_table_ptrs, "
      "int block_table_stride, "
      "Tensor state_base_addrs, "
      "Tensor state_block_strides, "
      "Tensor state_elem_sizes, "
      "Tensor state_inner_sizes, "
      "Tensor state_conv_widths, "
      "Tensor state_group_indices, "
      "Tensor state_dim_row_count, "
      "Tensor state_dim_row_stride, "
      "Tensor idx_mapping, "
      "int num_reqs, "
      "int total_states, "
      "bool conv_state_dim_first"
      ") -> ()");
  m.def(
      "vllm_postprocess_mamba("
      "Tensor(a!) num_accepted_tokens, "
      "Tensor state_idx, "
      "Tensor? num_scheduled_tokens, "
      "Tensor num_computed_tokens, "
      "Tensor? num_draft_tokens, "
      "Tensor block_table_ptrs, "
      "int block_table_stride, "
      "Tensor state_base_addrs, "
      "Tensor state_block_strides, "
      "Tensor state_elem_sizes, "
      "Tensor state_inner_sizes, "
      "Tensor state_conv_widths, "
      "Tensor state_group_indices, "
      "Tensor state_dim_row_count, "
      "Tensor state_dim_row_stride, "
      "Tensor(b!)? num_accepted_tokens_out, "
      "Tensor? idx_mapping, "
      "int num_reqs, "
      "int total_states, "
      "int block_size, "
      "bool conv_state_dim_first, "
      "bool has_idx_mapping, "
      "bool precomputed_new_computed"
      ") -> ()");
  m.def(
      "vllm_scatter_num_accepted("
      "Tensor idx_mapping, "
      "Tensor num_sampled, "
      "Tensor(a!) num_accepted"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_preprocess_mamba_align", &vllm_preprocess_mamba_align_impl);
  m.impl("vllm_precopy_mamba_align", &vllm_precopy_mamba_align_impl);
  m.impl("vllm_postprocess_mamba", &vllm_postprocess_mamba_impl);
  m.impl("vllm_scatter_num_accepted", &vllm_scatter_num_accepted_impl);
}
