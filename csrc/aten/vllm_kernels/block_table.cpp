// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/{block_table,gpu/block_table}.py:
//   gather_block_tables
//   compute_slot_mappings

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstdint>
#include <cstring>
#include "common.h"

namespace {

void vllm_gather_block_tables_kernel_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& src_block_table_ptrs, // [num_groups], uint64
    const at::Tensor& dst_block_table_ptrs, // [num_groups], uint64
    const at::Tensor& block_table_strides, // [num_groups], int64
    const at::Tensor& num_blocks, // [num_groups, max_num_reqs], int32
    int64_t num_blocks_stride,
    int64_t num_reqs,
    int64_t num_groups,
    int64_t num_reqs_padded) {
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(
      src_block_table_ptrs, at::kUInt64, "src_block_table_ptrs");
  VLLM_MCPU_CHECK_DTYPE(
      dst_block_table_ptrs, at::kUInt64, "dst_block_table_ptrs");
  VLLM_MCPU_CHECK_DTYPE(block_table_strides, at::kLong, "block_table_strides");
  VLLM_MCPU_CHECK_DTYPE(num_blocks, at::kInt, "num_blocks");
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(src_block_table_ptrs, 1, "src_block_table_ptrs");
  VLLM_MCPU_CHECK_DIM(dst_block_table_ptrs, 1, "dst_block_table_ptrs");
  VLLM_MCPU_CHECK_DIM(block_table_strides, 1, "block_table_strides");
  VLLM_MCPU_CHECK_DIM(num_blocks, 2, "num_blocks");
  VLLM_MCPU_CHECK(
      idx_mapping.is_contiguous(), "idx_mapping must be contiguous");
  VLLM_MCPU_CHECK(
      src_block_table_ptrs.is_contiguous(),
      "src_block_table_ptrs must be contiguous");
  VLLM_MCPU_CHECK(
      dst_block_table_ptrs.is_contiguous(),
      "dst_block_table_ptrs must be contiguous");
  VLLM_MCPU_CHECK(
      block_table_strides.is_contiguous(),
      "block_table_strides must be contiguous");
  VLLM_MCPU_CHECK(num_blocks.is_contiguous(), "num_blocks must be contiguous");
  VLLM_MCPU_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  VLLM_MCPU_CHECK(num_groups >= 0, "num_groups must be non-negative");
  VLLM_MCPU_CHECK(
      num_reqs_padded >= num_reqs, "num_reqs_padded must be at least num_reqs");
  VLLM_MCPU_CHECK(
      idx_mapping.numel() == num_reqs, "idx_mapping size must match num_reqs");
  VLLM_MCPU_CHECK(
      src_block_table_ptrs.numel() == num_groups,
      "src_block_table_ptrs size must match num_groups");
  VLLM_MCPU_CHECK(
      dst_block_table_ptrs.numel() == num_groups,
      "dst_block_table_ptrs size must match num_groups");
  VLLM_MCPU_CHECK(
      block_table_strides.numel() == num_groups,
      "block_table_strides size must match num_groups");
  VLLM_MCPU_CHECK(
      num_blocks.size(0) == num_groups,
      "num_blocks rows must match num_groups");
  VLLM_MCPU_CHECK(
      num_blocks_stride == num_blocks.stride(0),
      "num_blocks_stride must match num_blocks.stride(0)");

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const uint64_t* src_ptrs = src_block_table_ptrs.data_ptr<uint64_t>();
  const uint64_t* dst_ptrs = dst_block_table_ptrs.data_ptr<uint64_t>();
  const int64_t* strides_ptr = block_table_strides.data_ptr<int64_t>();
  const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();
  const int64_t max_num_reqs = num_blocks.size(1);

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_gather_block_tables_kernel",
      [idx_ptr,
       src_ptrs,
       dst_ptrs,
       strides_ptr,
       num_blocks_ptr,
       num_blocks_stride,
       max_num_reqs,
       num_reqs,
       num_groups,
       num_reqs_padded](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_gather_block_tables_kernel", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
          VLLM_MCPU_CHECK(
              src_ptrs[group_id] != 0,
              "src_block_table_ptrs contains a null pointer");
          VLLM_MCPU_CHECK(
              dst_ptrs[group_id] != 0,
              "dst_block_table_ptrs contains a null pointer");
          const int64_t stride = strides_ptr[group_id];
          VLLM_MCPU_CHECK(stride > 0, "block table strides must be positive");
          const int32_t* src =
              reinterpret_cast<const int32_t*>(src_ptrs[group_id]);
          int32_t* dst = reinterpret_cast<int32_t*>(dst_ptrs[group_id]);
          for (int64_t batch_idx = 0; batch_idx < num_reqs_padded;
               ++batch_idx) {
            int32_t* dst_row = dst + batch_idx * stride;
            if (batch_idx >= num_reqs) {
              for (int64_t col = 0; col < stride; ++col)
                dst_row[col] = 0;
              continue;
            }
            const int32_t req_idx = idx_ptr[batch_idx];
            VLLM_MCPU_CHECK(
                0 <= req_idx && req_idx < max_num_reqs,
                "idx_mapping contains an out-of-range request index");
            const int32_t count =
                num_blocks_ptr[group_id * num_blocks_stride + req_idx];
            VLLM_MCPU_CHECK(
                0 <= count && count <= stride,
                "num_blocks contains a count outside the block table stride");
            const int32_t* src_row = src + (int64_t)req_idx * stride;
            for (int32_t col = 0; col < count; ++col)
              dst_row[col] = src_row[col];
          }
        }
      });
}

void vllm_compute_slot_mappings_kernel_impl(
    int64_t max_num_tokens,
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs + 1], int32
    const at::Tensor& positions, // [num_tokens], int64
    const at::Tensor& block_table_ptrs, // [num_groups], uint64
    const at::Tensor& block_table_strides, // [num_groups], int64
    const at::Tensor& block_sizes, // [num_groups], int32
    at::Tensor& slot_mappings, // [num_groups, max_num_tokens], int64
    int64_t slot_mappings_stride,
    int64_t cp_rank,
    int64_t cp_size,
    int64_t cp_interleave,
    int64_t pad_id,
    int64_t num_groups,
    int64_t num_reqs,
    const std::optional<at::Tensor>& kernel_block_sizes,
    const std::optional<at::Tensor>& slot_mapping_enabled,
    const std::optional<at::Tensor>& dcp_sharded) {
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(query_start_loc, at::kInt, "query_start_loc");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(block_table_ptrs, at::kUInt64, "block_table_ptrs");
  VLLM_MCPU_CHECK_DTYPE(block_table_strides, at::kLong, "block_table_strides");
  VLLM_MCPU_CHECK_DTYPE(block_sizes, at::kInt, "block_sizes");
  VLLM_MCPU_CHECK_DTYPE(slot_mappings, at::kLong, "slot_mappings");
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(query_start_loc, 1, "query_start_loc");
  VLLM_MCPU_CHECK_DIM(positions, 1, "positions");
  VLLM_MCPU_CHECK_DIM(block_table_ptrs, 1, "block_table_ptrs");
  VLLM_MCPU_CHECK_DIM(block_table_strides, 1, "block_table_strides");
  VLLM_MCPU_CHECK_DIM(block_sizes, 1, "block_sizes");
  VLLM_MCPU_CHECK_DIM(slot_mappings, 2, "slot_mappings");
  VLLM_MCPU_CHECK(
      idx_mapping.is_contiguous(), "idx_mapping must be contiguous");
  VLLM_MCPU_CHECK(
      query_start_loc.is_contiguous(), "query_start_loc must be contiguous");
  VLLM_MCPU_CHECK(positions.is_contiguous(), "positions must be contiguous");
  VLLM_MCPU_CHECK(
      block_table_ptrs.is_contiguous(), "block_table_ptrs must be contiguous");
  VLLM_MCPU_CHECK(
      block_table_strides.is_contiguous(),
      "block_table_strides must be contiguous");
  VLLM_MCPU_CHECK(
      block_sizes.is_contiguous(), "block_sizes must be contiguous");
  VLLM_MCPU_CHECK(
      slot_mappings.is_contiguous(), "slot_mappings must be contiguous");
  VLLM_MCPU_CHECK(max_num_tokens >= 0, "max_num_tokens must be non-negative");
  VLLM_MCPU_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  VLLM_MCPU_CHECK(num_groups >= 0, "num_groups must be non-negative");
  VLLM_MCPU_CHECK(cp_size > 0, "cp_size must be positive");
  VLLM_MCPU_CHECK(cp_interleave > 0, "cp_interleave must be positive");
  VLLM_MCPU_CHECK(
      0 <= cp_rank && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
  VLLM_MCPU_CHECK(
      idx_mapping.numel() == num_reqs, "idx_mapping size must match num_reqs");
  VLLM_MCPU_CHECK(
      query_start_loc.numel() == num_reqs + 1,
      "query_start_loc size must be num_reqs + 1");
  VLLM_MCPU_CHECK(
      block_table_ptrs.numel() == num_groups,
      "block_table_ptrs size must match num_groups");
  VLLM_MCPU_CHECK(
      block_table_strides.numel() == num_groups,
      "block_table_strides size must match num_groups");
  VLLM_MCPU_CHECK(
      block_sizes.numel() == num_groups,
      "block_sizes size must match num_groups");
  VLLM_MCPU_CHECK(
      slot_mappings.size(0) == num_groups,
      "slot_mappings rows must match num_groups");
  VLLM_MCPU_CHECK(
      max_num_tokens <= slot_mappings.size(1),
      "slot_mappings must cover max_num_tokens");
  VLLM_MCPU_CHECK(
      slot_mappings_stride == slot_mappings.stride(0),
      "slot_mappings_stride must match slot_mappings.stride(0)");

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* query_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();
  const uint64_t* table_ptrs = block_table_ptrs.data_ptr<uint64_t>();
  const int64_t* table_strides = block_table_strides.data_ptr<int64_t>();
  const int32_t* sizes_ptr = block_sizes.data_ptr<int32_t>();
  const int32_t* kernel_sizes_ptr = kernel_block_sizes.has_value()
      ? kernel_block_sizes->data_ptr<int32_t>()
      : sizes_ptr;
  const bool* enabled_ptr = slot_mapping_enabled.has_value()
      ? slot_mapping_enabled->data_ptr<bool>()
      : nullptr;
  const bool* sharded_ptr =
      dcp_sharded.has_value() ? dcp_sharded->data_ptr<bool>() : nullptr;
  int64_t* slots_ptr = slot_mappings.data_ptr<int64_t>();
  const int64_t positions_numel = positions.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_compute_slot_mappings_kernel",
      [max_num_tokens,
       idx_ptr,
       query_ptr,
       positions_ptr,
       table_ptrs,
       table_strides,
       sizes_ptr,
       kernel_sizes_ptr,
       enabled_ptr,
       sharded_ptr,
       slots_ptr,
       slot_mappings_stride,
       cp_rank,
       cp_size,
       cp_interleave,
       pad_id,
       num_groups,
       num_reqs,
       positions_numel](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_compute_slot_mappings_kernel", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        VLLM_MCPU_CHECK(
            query_ptr[0] == 0, "query_start_loc must start at zero");
        const int32_t actual_num_tokens = query_ptr[num_reqs];
        VLLM_MCPU_CHECK(
            0 <= actual_num_tokens && actual_num_tokens <= max_num_tokens,
            "query_start_loc[-1] must be within max_num_tokens");
        VLLM_MCPU_CHECK(
            actual_num_tokens <= positions_numel,
            "positions must cover all tokens in query_start_loc");
        for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
          VLLM_MCPU_CHECK(
              table_ptrs[group_id] != 0,
              "block_table_ptrs contains a null pointer");
          const int32_t* table =
              reinterpret_cast<const int32_t*>(table_ptrs[group_id]);
          const int64_t table_stride = table_strides[group_id];
          const int64_t block_size = sizes_ptr[group_id];
          const int64_t kernel_block_size = kernel_sizes_ptr[group_id];
          const bool enabled = enabled_ptr == nullptr || enabled_ptr[group_id];
          const bool sharded = sharded_ptr == nullptr || sharded_ptr[group_id];
          VLLM_MCPU_CHECK(
              table_stride > 0, "block table strides must be positive");
          VLLM_MCPU_CHECK(block_size > 0, "block sizes must be positive");
          VLLM_MCPU_CHECK(
              block_size <= std::numeric_limits<int64_t>::max() / cp_size,
              "block_size * cp_size overflows int64");
          int64_t* slots = slots_ptr + group_id * slot_mappings_stride;

          for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
            const int32_t req_idx = idx_ptr[batch_idx];
            const int32_t start = query_ptr[batch_idx];
            const int32_t end = query_ptr[batch_idx + 1];
            VLLM_MCPU_CHECK(
                0 <= start && start <= end && end <= actual_num_tokens,
                "query_start_loc must be non-decreasing and within bounds");
            if (req_idx < 0 || !enabled) {
              for (int32_t token_idx = start; token_idx < end; ++token_idx)
                slots[token_idx] = pad_id;
              continue;
            }
            const int32_t* req_table = table + req_idx * table_stride;
            for (int32_t token_idx = start; token_idx < end; ++token_idx) {
              const int64_t position = positions_ptr[token_idx];
              VLLM_MCPU_CHECK(position >= 0, "positions must be non-negative");
              int64_t local_position = position;
              if (cp_size != 1 && sharded) {
                const int64_t global_block_size = block_size * cp_size;
                const int64_t block_offset = position % global_block_size;
                if ((block_offset / cp_interleave) % cp_size != cp_rank) {
                  slots[token_idx] = pad_id;
                  continue;
                }
                const int64_t rounds = block_offset / (cp_interleave * cp_size);
                const int64_t remainder = block_offset % cp_interleave;
                local_position = (position / global_block_size) * block_size +
                    rounds * cp_interleave + remainder;
              }
              const int64_t block_idx = local_position / kernel_block_size;
              VLLM_MCPU_CHECK(
                  block_idx < table_stride,
                  "position maps outside the block table stride");
              const int64_t block_number = req_table[block_idx];
              slots[token_idx] = block_number * kernel_block_size +
                  local_position % kernel_block_size;
            }
          }
          for (int64_t token_idx = actual_num_tokens;
               token_idx < max_num_tokens;
               ++token_idx)
            slots[token_idx] = pad_id;
        }
      });
}

void vllm_compute_slot_mapping_kernel_impl(
    int64_t num_tokens,
    int64_t max_num_tokens,
    const at::Tensor& query_start_loc, // [num_reqs + 1], int32
    const at::Tensor& positions, // [num_tokens], int64
    const at::Tensor& block_table, // [max_num_reqs, max_num_blocks], int32
    int64_t block_table_stride,
    int64_t block_size,
    at::Tensor& slot_mapping, // [max_num_tokens], int64
    int64_t cp_size,
    int64_t cp_rank,
    int64_t cp_interleave,
    int64_t pad_id) {
  VLLM_MCPU_CHECK_DTYPE(query_start_loc, at::kInt, "query_start_loc");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(block_table, at::kInt, "block_table");
  VLLM_MCPU_CHECK_DTYPE(slot_mapping, at::kLong, "slot_mapping");
  VLLM_MCPU_CHECK_DIM(query_start_loc, 1, "query_start_loc");
  VLLM_MCPU_CHECK_DIM(positions, 1, "positions");
  VLLM_MCPU_CHECK_DIM(block_table, 2, "block_table");
  VLLM_MCPU_CHECK_DIM(slot_mapping, 1, "slot_mapping");
  VLLM_MCPU_CHECK(
      query_start_loc.is_contiguous(), "query_start_loc must be contiguous");
  VLLM_MCPU_CHECK(positions.is_contiguous(), "positions must be contiguous");
  VLLM_MCPU_CHECK(
      block_table.is_contiguous(), "block_table must be contiguous");
  VLLM_MCPU_CHECK(
      slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  VLLM_MCPU_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  VLLM_MCPU_CHECK(
      max_num_tokens >= num_tokens,
      "max_num_tokens must be at least num_tokens");
  VLLM_MCPU_CHECK(
      num_tokens <= positions.numel(), "positions must cover num_tokens");
  VLLM_MCPU_CHECK(
      max_num_tokens <= slot_mapping.numel(),
      "slot_mapping must cover max_num_tokens");
  VLLM_MCPU_CHECK(
      block_table_stride == block_table.stride(0),
      "block_table_stride must match block_table.stride(0)");
  VLLM_MCPU_CHECK(block_size > 0, "block_size must be positive");
  VLLM_MCPU_CHECK(cp_size > 0, "cp_size must be positive");
  VLLM_MCPU_CHECK(cp_interleave > 0, "cp_interleave must be positive");
  VLLM_MCPU_CHECK(
      0 <= cp_rank && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
  VLLM_MCPU_CHECK(
      block_size <= std::numeric_limits<int64_t>::max() / cp_size,
      "block_size * cp_size overflows int64");

  const int32_t* query_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* table_ptr = block_table.data_ptr<int32_t>();
  int64_t* slots_ptr = slot_mapping.data_ptr<int64_t>();
  const int64_t num_reqs = query_start_loc.numel() - 1;

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_compute_slot_mapping_kernel",
      [num_tokens,
       max_num_tokens,
       query_ptr,
       positions_ptr,
       table_ptr,
       block_table_stride,
       block_size,
       slots_ptr,
       cp_size,
       cp_rank,
       cp_interleave,
       pad_id,
       num_reqs](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_compute_slot_mapping_kernel", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {query_ptr, positions_ptr, table_ptr, slots_ptr});
        VLLM_MCPU_CHECK(
            query_ptr[0] == 0, "query_start_loc must start at zero");
        VLLM_MCPU_CHECK(
            query_ptr[num_reqs] == num_tokens,
            "query_start_loc[-1] must equal num_tokens");
        const int64_t global_block_size = block_size * cp_size;
        for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
          const int32_t start = query_ptr[req_idx];
          const int32_t end = query_ptr[req_idx + 1];
          VLLM_MCPU_CHECK(
              0 <= start && start <= end && end <= num_tokens,
              "query_start_loc must be non-decreasing and within bounds");
          const int32_t* req_table = table_ptr + req_idx * block_table_stride;
          for (int32_t token_idx = start; token_idx < end; ++token_idx) {
            const int64_t position = positions_ptr[token_idx];
            VLLM_MCPU_CHECK(position >= 0, "positions must be non-negative");
            const int64_t block_idx = position / global_block_size;
            VLLM_MCPU_CHECK(
                block_idx < block_table_stride,
                "position maps outside the block table stride");
            const int64_t block_offset = position % global_block_size;
            const int64_t block_number = req_table[block_idx];
            if (cp_size == 1) {
              slots_ptr[token_idx] = block_number * block_size + block_offset;
              continue;
            }
            const bool is_local =
                (block_offset / cp_interleave) % cp_size == cp_rank;
            const int64_t rounds = block_offset / (cp_interleave * cp_size);
            const int64_t remainder = block_offset % cp_interleave;
            const int64_t local_offset = rounds * cp_interleave + remainder;
            slots_ptr[token_idx] =
                is_local ? block_number * block_size + local_offset : pad_id;
          }
        }
        for (int64_t token_idx = num_tokens; token_idx < max_num_tokens;
             ++token_idx)
          slots_ptr[token_idx] = pad_id;
      });
}

void zero_kv_blocks_kernel_impl(
    const at::Tensor& seg_addrs, // [n_segs], uint64 byte addresses
    const at::Tensor& block_ids, // [n_blocks], int64
    int64_t n_blocks,
    int64_t n_segs,
    int64_t page_size_el) {
  VLLM_MCPU_CHECK_DTYPE(seg_addrs, at::kUInt64, "seg_addrs");
  VLLM_MCPU_CHECK_DTYPE(block_ids, at::kLong, "block_ids");
  VLLM_MCPU_CHECK_DIM(seg_addrs, 1, "seg_addrs");
  VLLM_MCPU_CHECK_DIM(block_ids, 1, "block_ids");
  VLLM_MCPU_CHECK(seg_addrs.is_contiguous(), "seg_addrs must be contiguous");
  VLLM_MCPU_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  VLLM_MCPU_CHECK(n_blocks >= 0, "n_blocks must be non-negative");
  VLLM_MCPU_CHECK(n_segs >= 0, "n_segs must be non-negative");
  VLLM_MCPU_CHECK(page_size_el > 0, "page_size_el must be positive");
  VLLM_MCPU_CHECK(seg_addrs.size(0) >= n_segs, "seg_addrs must cover n_segs");
  VLLM_MCPU_CHECK(
      block_ids.size(0) >= n_blocks, "block_ids must cover n_blocks");

  const uint64_t* __restrict__ seg_addrs_ptr = seg_addrs.data_ptr<uint64_t>();
  const int64_t* __restrict__ block_ids_ptr = block_ids.data_ptr<int64_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::zero_kv_blocks_kernel_impl",
      [seg_addrs_ptr, block_ids_ptr, n_blocks, n_segs, page_size_el](
          at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::zero_kv_blocks_kernel_impl", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
#pragma omp parallel for collapse(2)
        for (int64_t block_index = 0; block_index < n_blocks; ++block_index) {
          for (int64_t seg_index = 0; seg_index < n_segs; ++seg_index) {
            const int64_t block_id = block_ids_ptr[block_index];
            int32_t* __restrict__ ptr =
                reinterpret_cast<int32_t*>(seg_addrs_ptr[seg_index]);
            std::memset(
                ptr + block_id * page_size_el,
                0,
                page_size_el * sizeof(int32_t));
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_gather_block_tables_kernel("
      "Tensor idx_mapping, "
      "Tensor src_block_table_ptrs, "
      "Tensor dst_block_table_ptrs, "
      "Tensor block_table_strides, "
      "Tensor num_blocks, "
      "int num_blocks_stride, "
      "int num_reqs, "
      "int num_groups, "
      "int num_reqs_padded"
      ") -> ()");
  // vllm/vllm/v1/worker/gpu/block_table.py
  m.def(
      "vllm_compute_slot_mappings_kernel("
      "int max_num_tokens, "
      "Tensor idx_mapping, "
      "Tensor query_start_loc, "
      "Tensor positions, "
      "Tensor block_table_ptrs, "
      "Tensor block_table_strides, "
      "Tensor block_sizes, "
      "Tensor(a!) slot_mappings, "
      "int slot_mappings_stride, "
      "int cp_rank, "
      "int cp_size, "
      "int cp_interleave, "
      "int pad_id, "
      "int num_groups, "
      "int num_reqs, "
      "Tensor? kernel_block_sizes=None, "
      "Tensor? slot_mapping_enabled=None, "
      "Tensor? dcp_sharded=None"
      ") -> ()");
  // vllm/vllm/v1/worker/block_table.py
  m.def(
      "vllm_compute_slot_mapping_kernel("
      "int num_tokens, "
      "int max_num_tokens, "
      "Tensor query_start_loc, "
      "Tensor positions, "
      "Tensor block_table, "
      "int block_table_stride, "
      "int block_size, "
      "Tensor(a!) slot_mapping, "
      "int cp_size, "
      "int cp_rank, "
      "int cp_interleave, "
      "int pad_id"
      ") -> ()");
  m.def(
      "zero_kv_blocks_kernel_impl("
      "Tensor seg_addrs, "
      "Tensor block_ids, "
      "SymInt n_blocks, "
      "SymInt n_segs, "
      "SymInt page_size_el"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_gather_block_tables_kernel", &vllm_gather_block_tables_kernel_impl);
  m.impl(
      "vllm_compute_slot_mappings_kernel",
      &vllm_compute_slot_mappings_kernel_impl);
  m.impl(
      "vllm_compute_slot_mapping_kernel",
      &vllm_compute_slot_mapping_kernel_impl);
  m.impl("zero_kv_blocks_kernel_impl", &zero_kv_blocks_kernel_impl);
}
