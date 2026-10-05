# Owner(s): ["module: PrivateUse1"]

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestV2ModelStateKernels(TestCase):
    def test_preprocess_mamba_align_advances_and_resets_crossed_slots(self):
        idx_mapping = torch.tensor([2, 0, 3], dtype=torch.int32, device="mcpu")
        state_idx = torch.tensor([0, 88, -1, 1], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor(
            [4, 0, 0, 7], dtype=torch.int32, device="mcpu"
        )
        query_start = torch.tensor([0, 1, 5, 7], dtype=torch.int32, device="mcpu")
        num_accepted = torch.tensor(
            [3, 77, 2, 4], dtype=torch.int32, device="mcpu"
        )
        src_col = torch.full((4,), 99, dtype=torch.int32, device="mcpu")
        src_off = torch.full((4,), 99, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_preprocess_mamba_align(
            idx_mapping,
            state_idx,
            num_computed,
            query_start,
            num_accepted,
            src_col,
            src_off,
            3,
            4,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            state_idx.cpu(), torch.tensor([1, 88, 0, 2], dtype=torch.int32)
        )
        self.assertEqual(
            num_accepted.cpu(), torch.tensor([1, 77, 2, 1], dtype=torch.int32)
        )
        self.assertEqual(
            src_col.cpu(), torch.tensor([0, 99, -1, 1], dtype=torch.int32)
        )
        self.assertEqual(
            src_off.cpu(), torch.tensor([2, 99, 1, 3], dtype=torch.int32)
        )

    def test_precopy_mamba_align_matches_conv_and_temporal_semantics(self):
        num_blocks = 13
        block_table = torch.arange(
            1, num_blocks, dtype=torch.int32, device="mcpu"
        ).reshape(3, 4)
        conv = torch.arange(
            num_blocks * 4 * 3, dtype=torch.float32, device="mcpu"
        ).reshape(num_blocks, 4, 3)
        temporal = torch.arange(
            num_blocks * 4, dtype=torch.float32, device="mcpu"
        ).reshape(num_blocks, 2, 2)
        conv_before = conv.cpu().clone()
        temporal_before = temporal.cpu().clone()

        state_idx = torch.tensor([0, 1, 0], dtype=torch.int32, device="mcpu")
        src_col = torch.tensor([-1, 1, 1], dtype=torch.int32, device="mcpu")
        token_bias = torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu")
        block_table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.int64, device="mcpu"
        )
        state_base_addrs = torch.tensor(
            [conv.data_ptr(), temporal.data_ptr()],
            dtype=torch.int64,
            device="mcpu",
        )
        state_block_strides = torch.tensor(
            [
                conv.stride(0) * conv.element_size(),
                temporal.stride(0) * temporal.element_size(),
            ],
            dtype=torch.int64,
            device="mcpu",
        )
        state_elem_sizes = torch.tensor(
            [conv.element_size(), temporal.element_size()],
            dtype=torch.int32,
            device="mcpu",
        )
        state_inner_sizes = torch.tensor(
            [conv.stride(1), temporal[0].numel()],
            dtype=torch.int64,
            device="mcpu",
        )
        state_conv_widths = torch.tensor([4, 0], dtype=torch.int32, device="mcpu")
        state_group_indices = torch.zeros(2, dtype=torch.int32, device="mcpu")
        state_dim_row_count = torch.zeros(2, dtype=torch.int32, device="mcpu")
        state_dim_row_stride = torch.zeros(2, dtype=torch.int64, device="mcpu")
        idx_mapping = torch.arange(3, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_precopy_mamba_align(
            state_idx,
            src_col,
            token_bias,
            block_table_ptrs,
            block_table.stride(0),
            state_base_addrs,
            state_block_strides,
            state_elem_sizes,
            state_inner_sizes,
            state_conv_widths,
            state_group_indices,
            state_dim_row_count,
            state_dim_row_stride,
            idx_mapping,
            3,
            2,
            False,
        )
        torch.mcpu.synchronize()

        conv_expected = conv_before.clone()
        temporal_expected = temporal_before.clone()
        source_block = int(block_table.cpu()[2, 1])
        destination_block = int(block_table.cpu()[2, 0])
        temporal_source_block = int(block_table.cpu()[2, 3])
        conv_expected[destination_block, :2] = conv_before[source_block, 2:]
        temporal_expected[destination_block] = temporal_before[
            temporal_source_block
        ]
        self.assertEqual(conv.cpu(), conv_expected)
        self.assertEqual(temporal.cpu(), temporal_expected)

    def test_postprocess_mamba_align_copies_temporal_state(self):
        block_table = torch.tensor(
            [[1, 2, 3, 4]], dtype=torch.int32, device="mcpu"
        )
        temporal = torch.arange(20, dtype=torch.float32, device="mcpu").reshape(
            5, 4
        )
        temporal_before = temporal.cpu().clone()
        num_accepted = torch.tensor([3], dtype=torch.int32, device="mcpu")
        state_idx = torch.tensor([0], dtype=torch.int32, device="mcpu")
        new_num_computed = torch.tensor([8], dtype=torch.int32, device="mcpu")
        block_table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.int64, device="mcpu"
        )
        state_base_addrs = torch.tensor(
            [temporal.data_ptr()], dtype=torch.int64, device="mcpu"
        )
        state_block_strides = torch.tensor(
            [temporal.stride(0) * temporal.element_size()],
            dtype=torch.int64,
            device="mcpu",
        )
        state_elem_sizes = torch.tensor(
            [temporal.element_size()], dtype=torch.int32, device="mcpu"
        )
        state_inner_sizes = torch.tensor([4], dtype=torch.int64, device="mcpu")
        state_conv_widths = torch.zeros(1, dtype=torch.int32, device="mcpu")
        state_group_indices = torch.zeros(1, dtype=torch.int32, device="mcpu")
        state_dim_row_count = torch.zeros(1, dtype=torch.int32, device="mcpu")
        state_dim_row_stride = torch.zeros(1, dtype=torch.int64, device="mcpu")
        idx_mapping = torch.zeros(1, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_postprocess_mamba(
            num_accepted,
            state_idx,
            None,
            new_num_computed,
            None,
            block_table_ptrs,
            block_table.stride(0),
            state_base_addrs,
            state_block_strides,
            state_elem_sizes,
            state_inner_sizes,
            state_conv_widths,
            state_group_indices,
            state_dim_row_count,
            state_dim_row_stride,
            None,
            idx_mapping,
            1,
            1,
            4,
            False,
            True,
            True,
        )
        torch.mcpu.synchronize()

        expected = temporal_before.clone()
        expected[2] = temporal_before[3]
        self.assertEqual(temporal.cpu(), expected)
        self.assertEqual(num_accepted.cpu(), torch.tensor([3], dtype=torch.int32))

    def test_scatter_num_accepted_skips_filtered_rows_and_clamps_to_one(self):
        idx_mapping = torch.tensor([2, -1, 0, 3], dtype=torch.int32, device="mcpu")
        num_sampled = torch.tensor([0, 5, -3, 6], dtype=torch.int32, device="mcpu")
        num_accepted = torch.full((4,), 77, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_scatter_num_accepted(
            idx_mapping,
            num_sampled,
            num_accepted,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            num_accepted.cpu(),
            torch.tensor([1, 77, 1, 6], dtype=torch.int32),
        )

    def test_postprocess_align_preserves_acceptance_for_all_layer_states(self):
        # A same-block checkpoint resets acceptance to one, but every layer
        # must still save the state for the originally accepted token.
        def device_tensor(values, dtype=torch.int32):
            return torch.tensor(values, dtype=dtype, device="mcpu")

        block_table = device_tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
        states = [
            (torch.arange(20, dtype=torch.float32).reshape(5, 4) + i * 100)
            .to("mcpu") for i in range(3)
        ]
        before = [state.cpu().clone() for state in states]
        accepted = device_tensor([77, 88, 3])
        torch.ops.mcpu.vllm_postprocess_mamba(
            accepted,
            device_tensor([0, 0, 0]),
            None,
            device_tensor([0, 0, 4]),
            None,
            device_tensor([block_table.data_ptr()], torch.int64),
            block_table.stride(0),
            device_tensor([state.data_ptr() for state in states], torch.int64),
            device_tensor([16] * 3, torch.int64),
            device_tensor([4] * 3),
            device_tensor([4] * 3, torch.int64),
            device_tensor([0] * 3),
            device_tensor([0] * 3),
            device_tensor([0] * 3),
            device_tensor([0] * 3, torch.int64),
            None,
            device_tensor([2, -1]),
            2, 3, 4, False, True, True,
        )
        torch.mcpu.synchronize()
        for state, original in zip(states, before):
            expected = original.clone()
            expected[1] = original[3]
            self.assertEqual(state.cpu(), expected)
        self.assertEqual(
            accepted.cpu(), torch.tensor([77, 88, 1], dtype=torch.int32)
        )


if __name__ == "__main__":
    run_tests()
