# Owner(s): ["module: PrivateUse1"]

import math

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestVllmKernelLaunch(TestCase):
    def test_prepare_prefill_inputs_lookahead_and_legacy_layout(self):
        def tensor(values):
            return torch.tensor(values, dtype=torch.int32, device="mcpu")

        for legacy in (False, True):
            inputs = tensor([-99] * 5)
            # Request 2 is decoding; request 1 finishes prefill; request 0
            # has one prompt token left. The 2D output has a padded row stride.
            next_tokens = (
                tensor([-99] * 3) if legacy else tensor([[-99] * 5] * 3)[:, :3]
            )
            torch.ops.mcpu.vllm_prepare_prefill_inputs(
                inputs, next_tokens, tensor([2, 0, 1]), tensor([0, 1, 3, 5]),
                tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24],
                        [30, 31, 32, 33, 34]]),
                tensor([4, 3, 2]), tensor([1, 1, 2]),
            )
            torch.mcpu.synchronize()
            self.assertEqual(
                inputs.cpu(), torch.tensor([-99, 11, 12, 21, 22], dtype=torch.int32)
            )
            expected = (
                [13, -99, -99] if legacy else
                [[13, 0, -99], [0, 0, -99], [0, 0, -99]]
            )
            self.assertEqual(
                next_tokens.cpu(), torch.tensor(expected, dtype=torch.int32)
            )

    def test_dflash2_selector_walk_greedy_and_invalid_request(self):
        scores = torch.full(
            (2, 3, 3, 3), -10.0, dtype=torch.float32, device="mcpu"
        )
        scores[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0], device="mcpu")
        scores[0, 1, 2] = torch.tensor([5.0, 7.0, 6.0], device="mcpu")
        scores[0, 2, 1] = torch.tensor([9.0, 8.0, 7.0], device="mcpu")
        candidates = torch.tensor(
            [
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
            ],
            dtype=torch.int64,
            device="mcpu",
        )
        tokens = torch.full((2, 3), -1, dtype=torch.int64, device="mcpu")
        realized = torch.full((2, 3, 3), -99.0, device="mcpu")
        req_state = torch.tensor(
            [0, 0, 0, -1, -1, -1], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_dflash2_selector_walk(
            scores,
            candidates,
            torch.arange(6, dtype=torch.int64, device="mcpu"),
            req_state,
            torch.zeros(2, device="mcpu"),
            torch.zeros(2, dtype=torch.int64, device="mcpu"),
            tokens,
            realized,
            2,
            3,
            3,
            False,
            False,
        )
        torch.mcpu.synchronize()

        self.assertEqual(tokens.cpu()[0], torch.tensor([12, 21, 30]))
        self.assertEqual(tokens.cpu()[1], torch.tensor([-1, -1, -1]))
        self.assertEqual(realized.cpu()[0, 0], torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(realized.cpu()[0, 1], torch.tensor([5.0, 7.0, 6.0]))
        self.assertEqual(realized.cpu()[0, 2], torch.tensor([9.0, 8.0, 7.0]))
        self.assertEqual(realized.cpu()[1], torch.full((3, 3), -99.0))

    def test_dflash2_cache_draft_logits_updates_only_valid_rows(self):
        draft_logits = torch.full((2, 2, 8), 42.0, device="mcpu")
        cached = torch.tensor(
            [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
            dtype=torch.int64,
            device="mcpu",
        )
        candidates = torch.tensor(
            [[[2, 3], [4, 5]], [[6, 7], [2, 4]]],
            dtype=torch.int64,
            device="mcpu",
        )
        scores = torch.tensor(
            [[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]],
            device="mcpu",
        )
        req_state = torch.tensor(
            [1, 1, -1, -1], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_dflash2_cache_draft_logits(
            draft_logits, cached, candidates, scores, req_state, 4, 2, 2
        )
        torch.mcpu.synchronize()

        actual_logits = draft_logits.cpu()
        actual_cached = cached.cpu()
        self.assertEqual(actual_logits[0], torch.full((2, 8), 42.0))
        self.assertTrue(torch.isneginf(actual_logits[1, :, :2]).all())
        self.assertEqual(actual_logits[1, 0, 2:4], torch.tensor([0.2, 0.3]))
        self.assertEqual(actual_logits[1, 1, 4:6], torch.tensor([0.4, 0.5]))
        self.assertEqual(
            actual_cached[1], torch.tensor([[2, 3], [4, 5]], dtype=torch.int64)
        )

    def test_autoregressive_prepare_prefill_inputs_matches_triton(self):
        last_indices = torch.full((4,), -1, dtype=torch.int64, device="mcpu")
        current_step = torch.tensor(3, dtype=torch.int64, device="mcpu")
        draft_ids = torch.full((8,), -9, dtype=torch.int32, device="mcpu")
        draft_positions = torch.full(
            (8,), -9, dtype=torch.int64, device="mcpu"
        )
        draft_query = torch.full((5,), -9, dtype=torch.int32, device="mcpu")
        draft_seq_lens = torch.full((4,), -9, dtype=torch.int32, device="mcpu")
        target_ids = torch.tensor(
            [10, 11, 12, 99, 20, 21, 22, 88],
            dtype=torch.int32,
            device="mcpu",
        )
        target_positions = torch.arange(8, dtype=torch.int64, device="mcpu")
        idx_mapping = torch.tensor([2, 0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor(
            [[30], [31], [42], [33]], dtype=torch.int64, device="mcpu"
        )
        next_prefill = torch.tensor(
            [77, 71, 72, 73], dtype=torch.int32, device="mcpu"
        )
        num_sampled = torch.tensor([2, 0], dtype=torch.int32, device="mcpu")
        num_rejected = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 4, 7], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([100, 200], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_autoregressive_prepare_prefill_inputs(
            last_indices,
            current_step,
            draft_ids,
            draft_positions,
            draft_query,
            draft_seq_lens,
            target_ids,
            target_positions,
            idx_mapping,
            last_sampled,
            next_prefill,
            num_sampled,
            num_rejected,
            query_start,
            seq_lens,
            4,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            last_indices.cpu(), torch.tensor([2, 6, 0, 0], dtype=torch.int64)
        )
        self.assertEqual(current_step.cpu(), torch.tensor(0, dtype=torch.int64))
        self.assertEqual(
            draft_ids.cpu(),
            torch.tensor([11, 12, 42, -9, 21, 22, 77, -9], dtype=torch.int32),
        )
        self.assertEqual(
            draft_positions.cpu(),
            torch.tensor([0, 1, 2, -9, 4, 5, 6, -9], dtype=torch.int64),
        )
        self.assertEqual(
            draft_query.cpu(), torch.tensor([0, 4, 7, 7, 7], dtype=torch.int32)
        )
        self.assertEqual(
            draft_seq_lens.cpu(),
            torch.tensor([100, 200, 0, 0], dtype=torch.int32),
        )

    def test_autoregressive_prepare_decode_inputs_matches_triton(self):
        draft_tokens = torch.tensor(
            [[101, 102], [201, 202]], dtype=torch.int64, device="mcpu"
        )
        target_seq_lens = torch.tensor(
            [9, 10], dtype=torch.int32, device="mcpu"
        )
        num_rejected = torch.tensor([2, 0], dtype=torch.int32, device="mcpu")
        input_ids = torch.full((4,), -9, dtype=torch.int32, device="mcpu")
        positions = torch.tensor([4, 9, -9, -9], device="mcpu")
        query_start = torch.full((5,), -9, dtype=torch.int32, device="mcpu")
        seq_lens = torch.full((4,), -9, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_autoregressive_prepare_decode_inputs(
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_ids,
            positions,
            query_start,
            seq_lens,
            10,
            4,
            True,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            input_ids.cpu(), torch.tensor([101, 201, -9, -9], dtype=torch.int32)
        )
        self.assertEqual(
            positions.cpu(), torch.tensor([5, 9, -9, -9], dtype=torch.int64)
        )
        self.assertEqual(
            query_start.cpu(), torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
        )
        self.assertEqual(
            seq_lens.cpu(), torch.tensor([8, 10, 0, 0], dtype=torch.int32)
        )

        # Gemma4 MTP uses the same token/padding writes without advancing
        # positions or active sequence lengths.
        positions.fill_(7)
        seq_lens.fill_(6)
        torch.ops.mcpu.vllm_autoregressive_prepare_decode_inputs(
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_ids,
            positions,
            query_start,
            seq_lens,
            10,
            4,
            False,
        )
        torch.mcpu.synchronize()
        self.assertEqual(
            positions.cpu(), torch.tensor([7, 7, 7, 7], dtype=torch.int64)
        )
        self.assertEqual(
            seq_lens.cpu(), torch.tensor([6, 6, 0, 0], dtype=torch.int32)
        )

    def test_autoregressive_update_draft_inputs_matches_triton(self):
        output_tokens = torch.full(
            (2, 3), -1, dtype=torch.int64, device="mcpu"
        )
        next_hidden = torch.full(
            (2, 4), -9, dtype=torch.bfloat16, device="mcpu"
        )
        input_ids = torch.full((2,), -9, dtype=torch.int32, device="mcpu")
        positions = torch.tensor([8, 9], dtype=torch.int64, device="mcpu")
        seq_lens = torch.tensor([9, 10], dtype=torch.int32, device="mcpu")
        draft_tokens = torch.tensor(
            [101, 201], dtype=torch.int64, device="mcpu"
        )
        current_step = torch.tensor(1, dtype=torch.int64, device="mcpu")
        hidden = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=torch.bfloat16,
            device="mcpu",
        )

        torch.ops.mcpu.vllm_autoregressive_update_draft_inputs(
            output_tokens,
            output_tokens.stride(0),
            next_hidden,
            next_hidden.stride(0),
            input_ids,
            positions,
            seq_lens,
            draft_tokens,
            current_step,
            hidden,
            hidden.stride(0),
            hidden.shape[1],
            10,
            3,
            True,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output_tokens.cpu(),
            torch.tensor([[-1, 101, -1], [-1, 201, -1]], dtype=torch.int64),
        )
        self.assertEqual(
            input_ids.cpu(), torch.tensor([101, 201], dtype=torch.int32)
        )
        self.assertEqual(next_hidden.cpu(), hidden.cpu())
        self.assertEqual(
            positions.cpu(), torch.tensor([9, 9], dtype=torch.int64)
        )
        self.assertEqual(seq_lens.cpu(), torch.tensor([10, 10], dtype=torch.int32))

        # The final speculative step records its token and returns before
        # changing any next-forward input, exactly like the Triton kernel.
        current_step.fill_(2)
        draft_tokens.add_(1)
        input_ids.fill_(-7)
        next_hidden.fill_(-7)
        positions.fill_(4)
        seq_lens.fill_(5)
        torch.ops.mcpu.vllm_autoregressive_update_draft_inputs(
            output_tokens,
            output_tokens.stride(0),
            next_hidden,
            next_hidden.stride(0),
            input_ids,
            positions,
            seq_lens,
            draft_tokens,
            current_step,
            hidden,
            hidden.stride(0),
            hidden.shape[1],
            10,
            3,
            False,
        )
        torch.mcpu.synchronize()
        self.assertEqual(
            output_tokens.cpu(),
            torch.tensor(
                [[-1, 101, 102], [-1, 201, 202]], dtype=torch.int64
            ),
        )
        self.assertEqual(
            input_ids.cpu(), torch.tensor([-7, -7], dtype=torch.int32)
        )
        self.assertEqual(
            next_hidden.cpu(),
            torch.full((2, 4), -7, dtype=torch.bfloat16),
        )
        self.assertEqual(
            positions.cpu(), torch.tensor([4, 4], dtype=torch.int64)
        )
        self.assertEqual(seq_lens.cpu(), torch.tensor([5, 5], dtype=torch.int32))

    def test_eagle_prepare_next_token_padded_matches_triton_validity_rules(self):
        sampled = torch.tensor(
            [[10, 11, -1, -1], [5, -2, 100, 7], [30, 31, 32, 33]],
            dtype=torch.int32,
            device="mcpu",
        )
        discard = torch.tensor([False, False, True], device="mcpu")
        backup = torch.tensor([90, 91, 92], dtype=torch.int32, device="mcpu")
        next_tokens = torch.empty(3, dtype=torch.int32, device="mcpu")
        valid_counts = torch.empty(3, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_eagle_prepare_next_token_padded(
            sampled,
            discard,
            backup,
            next_tokens,
            valid_counts,
            100,
            4,
            3,
        )
        torch.mcpu.synchronize()

        # Triton treats -1 as the only negative sentinel; -2 remains valid.
        self.assertEqual(
            next_tokens.cpu(), torch.tensor([11, 7, 92], dtype=torch.int32)
        )
        self.assertEqual(
            valid_counts.cpu(), torch.tensor([2, 3, 0], dtype=torch.int32)
        )

    def test_eagle_prepare_inputs_padded_handles_empty_draft_request(self):
        cu_draft = torch.tensor([2, 2, 5], dtype=torch.int32, device="mcpu")
        valid = torch.tensor([2, 0, 3], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 4, 5, 10], dtype=torch.int32, device="mcpu")
        indices = torch.empty(3, dtype=torch.int32, device="mcpu")
        rejected = torch.empty(3, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_eagle_prepare_inputs_padded(
            cu_draft, valid, query_start, indices, rejected, 3
        )
        torch.mcpu.synchronize()

        self.assertEqual(indices.cpu(), torch.tensor([2, 4, 8], dtype=torch.int32))
        self.assertEqual(rejected.cpu(), torch.tensor([1, 0, 1], dtype=torch.int32))

    def test_eagle_step_slot_mapping_metadata_clamps_and_pads_grid(self):
        positions = torch.tensor([2, 9], dtype=torch.int64, device="mcpu")
        block_table = torch.tensor(
            [[10, 11], [20, 21]], dtype=torch.int32, device="mcpu"
        )
        seq_lens = torch.tensor([3, 10], dtype=torch.int32, device="mcpu")
        out_positions = torch.full((2,), 99, dtype=torch.int64, device="mcpu")
        slots = torch.full((4,), 99, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_eagle_step_slot_mapping_metadata(
            positions,
            block_table,
            seq_lens,
            out_positions,
            slots,
            4,
            10,
            2,
            -1,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_positions.cpu(), torch.tensor([3, 0], dtype=torch.int64)
        )
        self.assertEqual(seq_lens.cpu(), torch.tensor([4, 1], dtype=torch.int32))
        self.assertEqual(
            slots.cpu(), torch.tensor([43, -1, -1, -1], dtype=torch.int64)
        )

    def test_rejection_sampler_expand_float_and_zero_length_request(self):
        values = torch.tensor([0.5, 0.0, 0.25], device="mcpu")
        cu_num_tokens = torch.tensor([2, 2, 5], dtype=torch.int32, device="mcpu")
        output = torch.empty(5, device="mcpu")

        torch.ops.mcpu.vllm_rejection_sampler_expand(
            output, values, cu_num_tokens, 0, 1, 128
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output.cpu(), torch.tensor([0.5, 0.5, 0.25, 0.25, 0.25])
        )

    def test_rejection_sampler_expand_int_and_replace(self):
        values = torch.tensor([4, 0, 7], dtype=torch.int32, device="mcpu")
        cu_num_tokens = torch.tensor([1, 3, 4], dtype=torch.int32, device="mcpu")
        output = torch.empty(4, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_rejection_sampler_expand(
            output, values, cu_num_tokens, 0, 9, 128
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output.cpu(), torch.tensor([4, 9, 9, 7], dtype=torch.int32)
        )

    def test_rejection_sampler_expand_honors_runtime_max_num_tokens(self):
        values = torch.tensor([4, 7], dtype=torch.int32, device="mcpu")
        cu_num_tokens = torch.tensor([3, 7], dtype=torch.int32, device="mcpu")
        output = torch.full((7,), -1, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_rejection_sampler_expand(
            output, values, cu_num_tokens, 0, 9, 2
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output.cpu(),
            torch.tensor([4, 4, -1, 7, 7, -1, -1], dtype=torch.int32),
        )

    def test_migrated_rejection_ops_are_not_registered_by_torch_mcpu(self):
        for op_name in (
            "vllm_rejection_compute_local_logits_stats",
            "vllm_rejection_resample",
        ):
            with self.assertRaises(AttributeError):
                getattr(torch.ops.mcpu, op_name)

    def test_rejection_v2_greedy_pipeline_handles_rejection_and_bonus(self):
        target = torch.zeros((6, 5), dtype=torch.float32, device="mcpu")
        for row, token in enumerate((1, 2, 3, 4, 1, 2)):
            target[row, token] = 10.0
        draft_sampled = torch.tensor(
            [-1, 1, 2, -1, 0, 1], dtype=torch.int32, device="mcpu"
        )
        cu_logits = torch.tensor([0, 3, 6], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        expanded_mapping = torch.tensor(
            [0, 0, 0, 1, 1, 1], dtype=torch.int32, device="mcpu"
        )
        temperature = torch.zeros(2, dtype=torch.float32, device="mcpu")
        seed = torch.tensor([11, 22], dtype=torch.int64, device="mcpu")
        pos = torch.arange(6, dtype=torch.int64, device="mcpu")
        block_shape = (6, 1)
        target_argmax = torch.tensor(
            [[1], [2], [3], [4], [1], [2]],
            dtype=torch.int64,
            device="mcpu",
        )
        target_max = torch.full(block_shape, 10.0, device="mcpu")
        target_sumexp = torch.zeros(block_shape, device="mcpu")
        draft_max = torch.zeros(block_shape, device="mcpu")
        draft_sumexp = torch.zeros(block_shape, device="mcpu")

        sampled = torch.full((2, 3), -99, dtype=torch.int64, device="mcpu")
        num_sampled = torch.empty(2, dtype=torch.int32, device="mcpu")
        target_lse = torch.empty(2, device="mcpu")
        draft_lse = torch.empty(2, device="mcpu")
        torch.ops.mcpu.vllm_rejection(
            sampled,
            num_sampled,
            target_lse,
            draft_lse,
            target,
            target_argmax,
            target_max,
            target_sumexp,
            draft_sampled,
            None,
            draft_max,
            draft_sumexp,
            cu_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            None,
            None,
            False,
            1,
        )

        resampled_argmax = torch.tensor(
            [[3], [4]], dtype=torch.int64, device="mcpu"
        )
        resampled_max = torch.full((2, 1), 10.0, device="mcpu")
        torch.ops.mcpu.vllm_rejection_insert(
            sampled,
            num_sampled,
            resampled_argmax,
            resampled_max,
            cu_logits,
            expanded_mapping,
            temperature,
            1,
        )
        flat_sampled = torch.zeros(6, dtype=torch.int64, device="mcpu")
        torch.ops.mcpu.vllm_rejection_flatten(
            flat_sampled, sampled, num_sampled, cu_logits
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_sampled.cpu(), torch.tensor([3, 1], dtype=torch.int32))
        self.assertEqual(sampled[0].cpu(), torch.tensor([1, 2, 3]))
        self.assertEqual(sampled[1, 0].cpu(), torch.tensor(4))
        self.assertEqual(flat_sampled.cpu(), torch.tensor([1, 2, 3, 4, 0, 0]))

    def test_rejection_block_verification_matches_equal_distributions(self):
        target = torch.tensor(
            [[1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [0.0, 0.0, 0.0]],
            device="mcpu",
        )
        draft = target[:2].reshape(1, 2, 3).clone()
        draft_sampled = torch.tensor([-1, 1, 2], dtype=torch.int32, device="mcpu")
        mapping = torch.zeros(3, dtype=torch.int32, device="mcpu")
        local_pos = torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu")
        temperature = torch.ones(1, device="mcpu")
        shape = (3, 1)
        target_argmax = torch.tensor(
            [[1], [2], [0]], dtype=torch.int64, device="mcpu"
        )
        target_max = torch.tensor([[3.0], [3.0], [0.0]], device="mcpu")
        block_sumexp = torch.exp(torch.tensor([-2.0, -1.0, 0.0])).sum().item()
        target_sumexp = torch.tensor(
            [[block_sumexp], [block_sumexp], [3.0]], device="mcpu"
        )
        draft_max = torch.tensor([[3.0], [3.0], [0.0]], device="mcpu")
        draft_sumexp = torch.tensor(
            [[block_sumexp], [block_sumexp], [0.0]], device="mcpu"
        )
        cu_logits = torch.tensor([0, 3], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        cumulative = torch.full((3,), -99.0, device="mcpu")
        torch.ops.mcpu.vllm_rejection_cumulative_log_p(
            cumulative,
            target,
            target_max,
            target_sumexp,
            draft_sampled,
            draft,
            draft_max,
            draft_sumexp,
            cu_logits,
            idx_mapping,
            temperature,
            1,
        )
        residual = torch.full(shape, -99.0, device="mcpu")
        torch.ops.mcpu.vllm_rejection_local_residual_mass(
            residual,
            cumulative,
            target,
            target_max,
            target_sumexp,
            draft,
            draft_max,
            draft_sumexp,
            mapping,
            local_pos,
            temperature,
            3,
            2,
        )
        sampled = torch.full((1, 3), -1, dtype=torch.int64, device="mcpu")
        num_sampled = torch.empty(1, dtype=torch.int32, device="mcpu")
        target_lse = torch.empty(1, device="mcpu")
        draft_lse = torch.empty(1, device="mcpu")
        torch.ops.mcpu.vllm_rejection(
            sampled,
            num_sampled,
            target_lse,
            draft_lse,
            target,
            target_argmax,
            target_max,
            target_sumexp,
            draft_sampled,
            draft,
            draft_max,
            draft_sumexp,
            cu_logits,
            idx_mapping,
            temperature,
            torch.tensor([7], dtype=torch.int64, device="mcpu"),
            torch.tensor([11, 12, 13], dtype=torch.int64, device="mcpu"),
            cumulative,
            residual,
            True,
            1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(cumulative[:2].cpu(), torch.zeros(2))
        self.assertAlmostEqual(residual[1, 0].cpu().item(), 0.0, places=6)
        self.assertEqual(num_sampled.cpu(), torch.tensor([2], dtype=torch.int32))
        self.assertEqual(sampled[0, :2].cpu(), torch.tensor([1, 2]))

    def test_rejection_greedy_matches_mixed_batch_semantics(self):
        output = torch.full((3, 4), -1, dtype=torch.int32, device="mcpu")
        cu_tokens = torch.tensor([2, 4, 4], dtype=torch.int32, device="mcpu")
        draft_ids = torch.tensor([1, 2, 3, 9], dtype=torch.int32, device="mcpu")
        target_argmax = torch.tensor([1, 5, 3, 4], device="mcpu")
        bonus_ids = torch.tensor([[8], [7], [6]], dtype=torch.int32, device="mcpu")
        is_greedy = torch.tensor([True, False, True], device="mcpu")

        torch.ops.mcpu.vllm_rejection_greedy(
            output,
            cu_tokens,
            draft_ids,
            target_argmax,
            bonus_ids,
            is_greedy,
            3,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output.cpu(),
            torch.tensor(
                [[1, 5, -1, -1], [-1, -1, -1, -1], [6, -1, -1, -1]],
                dtype=torch.int32,
            ),
        )

    def test_sample_recovered_matches_adjusted_probability_argmax(self):
        cu_tokens = torch.tensor([1, 2], dtype=torch.int32, device="mcpu")
        draft_ids = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        target_probs = torch.tensor(
            [[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]],
            dtype=torch.float32,
            device="mcpu",
        )
        draft_probs = torch.tensor(
            [[0.05, 0.5, 0.1], [0.4, 0.1, 0.3]],
            dtype=torch.float32,
            device="mcpu",
        )
        inv_q = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 3.0, 1.0]],
            dtype=torch.float64,
            device="mcpu",
        )
        output = torch.empty(2, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_sample_recovered(
            output,
            cu_tokens,
            draft_ids,
            draft_probs,
            target_probs,
            inv_q,
            3,
        )
        torch.mcpu.synchronize()
        self.assertEqual(output.cpu(), torch.tensor([2, 1], dtype=torch.int32))

        # With no draft distribution, Triton masks only the proposed token.
        torch.ops.mcpu.vllm_sample_recovered(
            output,
            cu_tokens,
            draft_ids,
            None,
            target_probs,
            inv_q.float(),
            3,
        )
        torch.mcpu.synchronize()
        self.assertEqual(output.cpu(), torch.tensor([1, 0], dtype=torch.int32))

    def test_rejection_random_handles_padding_rejection_and_bonus(self):
        output = torch.full((2, 3), -1, dtype=torch.int32, device="mcpu")
        cu_tokens = torch.tensor([2, 4], dtype=torch.int32, device="mcpu")
        draft_ids = torch.tensor([1, -1, 2, 0], dtype=torch.int32, device="mcpu")
        target_probs = torch.tensor(
            [
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.05, 0.05, 0.9],
                [0.8, 0.1, 0.1],
            ],
            dtype=torch.float32,
            device="mcpu",
        )
        bonus_ids = torch.tensor([[8], [9]], dtype=torch.int32, device="mcpu")
        recovered = torch.tensor([6, 7, 5, 4], dtype=torch.int32, device="mcpu")
        uniform = torch.tensor(
            [0.2, 0.0, 0.2, 0.2], dtype=torch.float64, device="mcpu"
        )
        is_greedy = torch.tensor([False, False], device="mcpu")

        torch.ops.mcpu.vllm_rejection_random(
            output,
            cu_tokens,
            draft_ids,
            None,
            target_probs,
            bonus_ids,
            recovered,
            uniform,
            is_greedy,
            2,
            3,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            output.cpu(),
            torch.tensor([[1, 7, -1], [2, 0, 9]], dtype=torch.int32),
        )

    def test_min_p_kernel(self):
        logits = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]], device="mcpu"
        )
        idx = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        min_p = torch.tensor([math.exp(-1.5), 0.0], dtype=torch.float32, device="mcpu")

        torch.ops.mcpu.vllm_min_p_kernel(logits, idx, min_p, 3)
        torch.mcpu.synchronize()

        expected = torch.tensor(
            [[float("-inf"), 1.0, 2.0], [3.0, 2.0, 1.0]]
        )
        self.assertEqual(logits.cpu(), expected)

    def test_prompt_logprobs_token_ids(self):
        out = torch.empty(3, dtype=torch.int64, device="mcpu")
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
            device="mcpu",
        )

        torch.ops.mcpu.vllm_prompt_logprobs_token_ids(
            out, query_start_loc, idx_mapping, num_computed, all_token_ids
        )
        torch.mcpu.synchronize()

        self.assertEqual(out.cpu(), torch.tensor([22, 23, 11], dtype=torch.int64))

    def test_get_num_sampled_and_rejected_returns_num_rejected(self):
        num_sampled = torch.tensor([1, 2, 3], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([5, 2, 8], dtype=torch.int32, device="mcpu")
        cu_num_logits = torch.tensor([0, 4, 7, 12], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([4, 6, 8], dtype=torch.int32, device="mcpu")

        num_rejected = torch.empty_like(num_sampled)
        torch.ops.mcpu.vllm_get_num_sampled_and_rejected(
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_sampled.cpu(), torch.tensor([1, 0, 3], dtype=torch.int32))
        self.assertEqual(num_rejected.cpu(), torch.tensor([3, 0, 2], dtype=torch.int32))

    def test_apply_grammar_bitmask_masks_zero_bits(self):
        logits = torch.arange(70, dtype=torch.float32).reshape(2, 35)
        logits_mcpu = logits.to("mcpu")
        logits_indices = torch.tensor([1], dtype=torch.int32, device="mcpu")
        # Token 1 and token 33 are rejected (zero bits). All other valid
        # vocabulary positions are allowed (one bits).
        bitmask = torch.tensor(
            [[-3, 0b101]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits_mcpu, logits_indices,
            torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu"),
            bitmask, 35, 1,
        )
        torch.mcpu.synchronize()

        expected = logits.clone()
        expected[1, 1] = float("-inf")
        expected[1, 33] = float("-inf")
        self.assertEqual(logits_mcpu.cpu(), expected)

    def test_apply_grammar_bitmask_handles_signed_high_bit(self):
        logits = torch.zeros((1, 32), dtype=torch.bfloat16, device="mcpu")
        logits_indices = torch.tensor([0], dtype=torch.int32, device="mcpu")
        # Bits 0-30 are one; signed bit 31 is zero and must be masked.
        bitmask = torch.tensor(
            [[0x7FFFFFFF]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits, logits_indices,
            torch.tensor([0, 1], dtype=torch.int32, device="mcpu"),
            bitmask, 32, 1,
        )
        torch.mcpu.synchronize()

        expected = torch.zeros(32, dtype=torch.bfloat16)
        expected[31] = float("-inf")
        self.assertEqual(logits.cpu()[0], expected)

    def test_apply_grammar_bitmask_compacted_mapping_differential(self):
        # Reordered requests, a subset of positions, two empty requests, and
        # inactive scheduled positions (including one past the last logits row).
        mapping = [8, 0, 1, 3, 4, 5, 12, 14, 15, 16, 19]
        offsets = [0, 2, 2, 3, 6, 6]
        mask_stride = 4
        generator = torch.Generator().manual_seed(1729)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for vocab in (1, 31, 32, 33, 8193):
                with self.subTest(dtype=dtype, vocab=vocab):
                    words = (vocab + 31) // 32
                    original = torch.randn(8, vocab + 6, generator=generator).to(dtype)
                    expected = original.clone()
                    masks = torch.randint(
                        -(2**31), 2**31, (len(mapping) + 2, words + 3),
                        dtype=torch.int32, generator=generator,
                    )
                    # Explicit all-allow/all-deny cases supplement random words.
                    masks[1, 1:words + 1] = -1
                    masks[2, 1:words + 1] = 0
                    for mask_row, encoded in enumerate(mapping):
                        req, position = divmod(encoded, mask_stride)
                        if position >= offsets[req + 1] - offsets[req]:
                            continue
                        for token in range(vocab):
                            if not ((int(masks[mask_row + 1, token // 32 + 1])
                                     >> (token % 32)) & 1):
                                expected[1 + offsets[req] + position, 2 + token] = -math.inf
                    storage = original.to("mcpu")
                    mask_storage = masks.to("mcpu")
                    indices = torch.tensor(mapping, dtype=torch.int32, device="mcpu")
                    cu = torch.tensor(offsets, dtype=torch.int32, device="mcpu")
                    # Nonzero storage offsets and padded strides on both matrices.
                    torch.ops.mcpu.vllm_apply_grammar_bitmask(
                        storage[1:7, 2:vocab + 4], indices, cu,
                        mask_storage[1:-1, 1:words + 1], vocab, mask_stride,
                    )
                    torch.mcpu.synchronize()
                    self.assertEqual(storage.cpu(), expected, atol=0, rtol=0)
                    self.assertEqual(mask_storage.cpu(), masks)
                    self.assertEqual(indices.cpu().tolist(), mapping)
                    self.assertEqual(cu.cpu().tolist(), offsets)

    def test_apply_grammar_bitmask_empty_and_cross_stream_offsets(self):
        def ints(data):
            return torch.tensor(data, dtype=torch.int32, device="mcpu")

        logits = torch.full((2, 35), 7.0, device="mcpu")
        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits, ints([]), ints([0]), ints([]).reshape(0, 1), 0, 1,
        )
        # No vocabulary is also a no-op for a nonempty mapping.
        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits, ints([0]), ints([0, 2]), ints([[0, 0]]), 0, 2,
        )
        torch.mcpu.synchronize()
        self.assertEqual(logits.cpu(), torch.full((2, 35), 7.0))
        copy_stream, compute_stream = torch.mcpu.Stream(), torch.mcpu.Stream()
        cu, indices, masks = ints([0, 0, 0]), ints([2, 3]), ints([[0, 0], [-1, -1]])
        copy_stream.wait_stream(torch.mcpu.current_stream())
        with torch.mcpu.stream(copy_stream):
            cu.copy_(torch.tensor([0, 1, 2], dtype=torch.int32), non_blocking=True)
        compute_stream.wait_stream(copy_stream)
        with torch.mcpu.stream(compute_stream):
            torch.ops.mcpu.vllm_apply_grammar_bitmask(logits, indices, cu, masks, 35, 2)
        compute_stream.synchronize()
        expected = torch.full((2, 35), 7.0)
        expected[1] = -math.inf
        self.assertEqual(logits.cpu(), expected)

    def test_logit_bias_preserves_all_allowed_tokens(self):
        vocab_size = 1100
        for num_allowed in (513, 1024):
            with self.subTest(num_allowed=num_allowed):
                logits_cpu = torch.arange(vocab_size, dtype=torch.float32).reshape(1, -1)
                logits = logits_cpu.to("mcpu")
                expanded_idx_mapping = torch.tensor(
                    [0], dtype=torch.int32, device="mcpu"
                )
                num_allowed_token_ids = torch.tensor(
                    [num_allowed], dtype=torch.int32, device="mcpu"
                )
                allowed_token_ids = torch.arange(
                    1024, dtype=torch.int32, device="mcpu"
                ).reshape(1, -1)
                num_logit_bias = torch.zeros(1, dtype=torch.int32, device="mcpu")
                bias_token_ids = torch.zeros(
                    (1, 1024), dtype=torch.int32, device="mcpu"
                )
                bias = torch.zeros((1, 1024), dtype=torch.float32, device="mcpu")
                pos = torch.zeros(1, dtype=torch.int64, device="mcpu")
                min_lens = torch.zeros(1, dtype=torch.int32, device="mcpu")
                num_stop_token_ids = torch.zeros(
                    1, dtype=torch.int32, device="mcpu"
                )
                stop_token_ids = torch.zeros(
                    (1, 128), dtype=torch.int32, device="mcpu"
                )

                torch.ops.mcpu.vllm_bias_kernel(
                    logits,
                    vocab_size,
                    expanded_idx_mapping,
                    num_allowed_token_ids,
                    allowed_token_ids,
                    num_logit_bias,
                    bias_token_ids,
                    bias,
                    pos,
                    min_lens,
                    num_stop_token_ids,
                    stop_token_ids,
                )
                torch.mcpu.synchronize()

                expected = logits_cpu.clone()
                expected[:, num_allowed:] = float("-inf")
                self.assertEqual(logits.cpu(), expected)

    def test_fill_logprob_token_ids_custom_overrides_topk(self):
        out_ids = torch.full((3, 5), -99, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((3, 5), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([10, 20, 30], dtype=torch.int64, device="mcpu")
        topk = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device="mcpu"
        )
        mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([0, 3, 0], dtype=torch.int32, device="mcpu")
        custom = torch.tensor(
            [[40, 41, 42, 43], [7, 8, 9, 10], [50, 51, 52, 53]],
            dtype=torch.int32,
            device="mcpu",
        )

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk,
            topk.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            2,
            4,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor(
                [[10, 1, 2, -99, -99], [20, 3, 4, -99, -99], [30, 7, 8, 9, -99]],
                dtype=torch.int64,
            ),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                ]
            ),
        )

    def test_fill_logprob_token_ids_num_topk_zero(self):
        out_ids = torch.full((2, 4), -1, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((2, 4), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([11, 22], dtype=torch.int64, device="mcpu")
        topk_unused = torch.zeros((2, 1), dtype=torch.int32, device="mcpu")
        mapping = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        custom = torch.tensor(
            [[0, 0, 0], [70, 71, 72]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk_unused,
            topk_unused.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            0,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor([[11, -1, -1, -1], [22, 70, 71, -1]], dtype=torch.int64),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor(
                [[True, False, False, False], [True, True, True, False]]
            ),
        )

    def test_fill_logprob_token_ids_respects_padded_cols(self):
        out_ids = torch.full((1, 5), -1, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((1, 5), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([11], dtype=torch.int64, device="mcpu")
        topk = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="mcpu")
        mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([3], dtype=torch.int32, device="mcpu")
        custom = torch.tensor([[70, 71, 72]], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk,
            topk.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            3,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor([[11, 70, 71, -1, -1]], dtype=torch.int64),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor([[True, True, True, False, False]]),
        )

    def test_combine_sampled_and_draft_tokens_supports_zero_or_one_sampled(self):
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([100, 200], dtype=torch.int64, device="mcpu")
        query_start = torch.tensor([0, 3, 6], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([5, 5], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([2, 2], dtype=torch.int32, device="mcpu")
        drafts = torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int64, device="mcpu"
        )

        cases = (
            (1, [0, 200, 20, 100, 10, 11]),
            (0, [0, 20, 21, 10, 11, 12]),
        )
        for num_new_sampled, expected_input in cases:
            with self.subTest(num_new_sampled=num_new_sampled):
                input_ids = torch.zeros(6, dtype=torch.int32, device="mcpu")
                cu_num_logits = torch.tensor(
                    [0, 2, 5], dtype=torch.int32, device="mcpu"
                )
                logits_indices = torch.empty(5, dtype=torch.int64, device="mcpu")

                torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
                    input_ids,
                    idx_mapping,
                    last_sampled,
                    query_start,
                    seq_lens,
                    prefill_len,
                    drafts,
                    drafts.stride(0),
                    cu_num_logits,
                    logits_indices,
                    num_new_sampled,
                )
                torch.mcpu.synchronize()

                self.assertEqual(
                    input_ids.cpu(), torch.tensor(expected_input, dtype=torch.int32)
                )
                self.assertEqual(
                    logits_indices.cpu(), torch.tensor([1, 2, 3, 4, 5])
                )

    def test_combine_sampled_and_draft_tokens_skips_prefill_writes(self):
        input_ids = torch.full((3,), 9, dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([100], dtype=torch.int64, device="mcpu")
        query_start = torch.tensor([0, 3], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([2], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([2], dtype=torch.int32, device="mcpu")
        drafts = torch.tensor([[10, 11]], dtype=torch.int64, device="mcpu")
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        logits_indices = torch.empty(2, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start,
            seq_lens,
            prefill_len,
            drafts,
            drafts.stride(0),
            cu_num_logits,
            logits_indices,
            0,
        )
        torch.mcpu.synchronize()

        self.assertEqual(input_ids.cpu(), torch.full((3,), 9, dtype=torch.int32))
        self.assertEqual(logits_indices.cpu(), torch.tensor([1, 2]))

    def test_combine_sampled_and_draft_tokens_preserves_prompt_tail(self):
        input_ids = torch.tensor([41, 42], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([100], dtype=torch.int64, device="mcpu")
        query_start = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([5], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([4], dtype=torch.int32, device="mcpu")
        drafts = torch.tensor([[10]], dtype=torch.int64, device="mcpu")
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        logits_indices = torch.empty(2, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start,
            seq_lens,
            prefill_len,
            drafts,
            drafts.stride(0),
            cu_num_logits,
            logits_indices,
            1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(input_ids.cpu(), torch.tensor([41, 10], dtype=torch.int32))
        self.assertEqual(logits_indices.cpu(), torch.tensor([0, 1]))

    def test_post_update_skips_negative_mapping_and_updates_state(self):
        idx_mapping = torch.tensor([-1, 1, 0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([10, 20], dtype=torch.int32, device="mcpu")
        # vLLM 0.24 stores this pointer-compatible buffer as [max_reqs, 1].
        last_sampled = torch.full((2, 1), -1, dtype=torch.int64, device="mcpu")
        output_counts = torch.zeros((2, 10), dtype=torch.int32, device="mcpu")
        sampled = torch.tensor(
            [[99, 98], [3, 4], [5, 6]], dtype=torch.int64, device="mcpu"
        )
        num_sampled = torch.tensor([1, 2, 1], dtype=torch.int32, device="mcpu")
        num_rejected = torch.tensor([9, 1, 0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.full((2, 8), -1, dtype=torch.int32, device="mcpu")
        total_len = torch.tensor([1, 2], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_post_update(
            idx_mapping,
            num_computed,
            last_sampled,
            output_counts,
            output_counts.stride(0),
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            query_start,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_computed.cpu(), torch.tensor([11, 22], dtype=torch.int32))
        self.assertEqual(
            last_sampled.cpu(), torch.tensor([[5], [4]], dtype=torch.int64)
        )
        self.assertEqual(total_len.cpu(), torch.tensor([2, 4], dtype=torch.int32))
        expected_ids = torch.full((2, 8), -1, dtype=torch.int32)
        expected_ids[0, 1] = 5
        expected_ids[1, 2:4] = torch.tensor([3, 4], dtype=torch.int32)
        self.assertEqual(all_token_ids.cpu(), expected_ids)
        expected_counts = torch.zeros((2, 10), dtype=torch.int32)
        expected_counts[0, 5] = 1
        expected_counts[1, 3] = 1
        expected_counts[1, 4] = 1
        self.assertEqual(output_counts.cpu(), expected_counts)

    def test_post_update_supports_optional_query_and_counts(self):
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([10], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([-1], dtype=torch.int64, device="mcpu")
        sampled = torch.tensor([[7]], dtype=torch.int64, device="mcpu")
        num_sampled = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_rejected = torch.tensor([2], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.full((1, 4), -1, dtype=torch.int32, device="mcpu")
        total_len = torch.tensor([1], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_post_update(
            idx_mapping,
            num_computed,
            last_sampled,
            None,
            0,
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            None,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_computed.cpu(), torch.tensor([8], dtype=torch.int32))
        self.assertEqual(last_sampled.cpu(), torch.tensor([-1], dtype=torch.int64))
        self.assertEqual(total_len.cpu(), torch.tensor([1], dtype=torch.int32))
        self.assertEqual(
            all_token_ids.cpu(), torch.full((1, 4), -1, dtype=torch.int32)
        )

    def test_ranks_matches_triton_comparison_and_handles_padded_ids(self):
        logits = torch.tensor(
            [[1.0, 4.0, 2.0, 4.0], [3.0, 2.0, 1.0, 0.0]], device="mcpu"
        )
        token_ids = torch.tensor([1, -1], dtype=torch.int64, device="mcpu")
        output = torch.empty(2, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_ranks_kernel(output, logits, token_ids, 4)
        torch.mcpu.synchronize()

        # Rank counts ties with >=, matching the v0.24 Triton kernel. Invalid
        # padded ids are outside its contract and are made harmless here.
        self.assertEqual(output.cpu(), torch.tensor([2, 0], dtype=torch.int64))

    def test_apply_write_single_supports_staged_write_dtypes(self):
        for dtype in (torch.int32, torch.int64, torch.float32):
            with self.subTest(dtype=dtype):
                output = torch.zeros((3, 6), dtype=dtype, device="mcpu")
                indices = torch.tensor([2, 0, 2], dtype=torch.int32, device="mcpu")
                starts = torch.tensor([1, 3, 4], dtype=torch.int32, device="mcpu")
                contents = torch.tensor([10, 11, 20, 30, 31], dtype=dtype, device="mcpu")
                cu_lens = torch.tensor([2, 3, 5], dtype=torch.int32, device="mcpu")

                torch.ops.mcpu.vllm_apply_write_single(
                    output,
                    output.stride(0),
                    indices,
                    starts,
                    contents,
                    cu_lens,
                )
                torch.mcpu.synchronize()

                expected = torch.zeros((3, 6), dtype=dtype)
                expected[2, 1:3] = torch.tensor([10, 11], dtype=dtype)
                expected[0, 3] = 20
                expected[2, 4:6] = torch.tensor([30, 31], dtype=dtype)
                self.assertEqual(output.cpu(), expected)

    def test_apply_write_multi_resolves_output_pointer_and_stride(self):
        outputs = [
            torch.zeros((2, 5), dtype=torch.int32, device="mcpu"),
            torch.zeros((3, 4), dtype=torch.int32, device="mcpu"),
        ]
        output_ptrs = torch.tensor(
            [output.data_ptr() for output in outputs],
            dtype=torch.uint64,
            device="mcpu",
        )
        output_strides = torch.tensor(
            [output.stride(0) for output in outputs],
            dtype=torch.int64,
            device="mcpu",
        )
        indices = torch.tensor([2, 1, 0], dtype=torch.int32, device="mcpu")
        starts = torch.tensor([1, 2, 0], dtype=torch.int32, device="mcpu")
        contents = torch.tensor([10, 11, 20, 30, 31], dtype=torch.int32, device="mcpu")
        cu_lens = torch.tensor([2, 3, 5], dtype=torch.int32, device="mcpu")
        group_ids = torch.tensor([1, 0, 1], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_apply_write_multi(
            output_ptrs,
            output_strides,
            indices,
            starts,
            contents,
            cu_lens,
            group_ids,
        )
        torch.mcpu.synchronize()

        expected0 = torch.zeros((2, 5), dtype=torch.int32)
        expected0[1, 2] = 20
        expected1 = torch.zeros((3, 4), dtype=torch.int32)
        expected1[2, 1:3] = torch.tensor([10, 11], dtype=torch.int32)
        expected1[0, 0:2] = torch.tensor([30, 31], dtype=torch.int32)
        self.assertEqual(outputs[0].cpu(), expected0)
        self.assertEqual(outputs[1].cpu(), expected1)

    def test_gather_block_tables_pointer_abi_preserves_active_tail(self):
        sources = [
            torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                dtype=torch.int32,
                device="mcpu",
            ),
            torch.tensor(
                [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
                dtype=torch.int32,
                device="mcpu",
            ),
        ]
        destinations = [
            torch.full((4, 4), 99, dtype=torch.int32, device="mcpu"),
            torch.full((4, 3), 99, dtype=torch.int32, device="mcpu"),
        ]
        src_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in sources],
            dtype=torch.uint64,
            device="mcpu",
        )
        dst_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in destinations],
            dtype=torch.uint64,
            device="mcpu",
        )
        strides = torch.tensor([4, 3], dtype=torch.int64, device="mcpu")
        idx_mapping = torch.tensor([2, 0], dtype=torch.int32, device="mcpu")
        num_blocks = torch.tensor(
            [[1, 4, 2], [2, 3, 1]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_gather_block_tables_kernel(
            idx_mapping,
            src_ptrs,
            dst_ptrs,
            strides,
            num_blocks,
            num_blocks.stride(0),
            2,
            2,
            3,
        )
        torch.mcpu.synchronize()

        expected0 = torch.tensor(
            [[30, 31, 99, 99], [10, 99, 99, 99], [0, 0, 0, 0], [99, 99, 99, 99]],
            dtype=torch.int32,
        )
        expected1 = torch.tensor(
            [[60, 99, 99], [40, 41, 99], [0, 0, 0], [99, 99, 99]],
            dtype=torch.int32,
        )
        self.assertEqual(destinations[0].cpu(), expected0)
        self.assertEqual(destinations[1].cpu(), expected1)


    def test_compute_slot_mappings_pointer_abi_and_tail_padding(self):
        block_tables = [
            torch.tensor(
                [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device="mcpu"
            ),
            torch.tensor(
                [[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]],
                dtype=torch.int32,
                device="mcpu",
            ),
        ]
        table_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in block_tables],
            dtype=torch.uint64,
            device="mcpu",
        )
        table_strides = torch.tensor([3, 5], dtype=torch.int64, device="mcpu")
        block_sizes = torch.tensor([4, 2], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 2, 5], dtype=torch.int32, device="mcpu")
        positions = torch.tensor([0, 5, 2, 3, 8], dtype=torch.int64, device="mcpu")
        slots = torch.full((2, 8), 999, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
            8,
            idx_mapping,
            query_start,
            positions,
            table_ptrs,
            table_strides,
            block_sizes,
            slots,
            slots.stride(0),
            0,
            1,
            1,
            -1,
            2,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu(),
            torch.tensor(
                [[80, 85, 42, 43, 48, -1, -1, -1], [80, 85, 62, 63, 68, -1, -1, -1]],
                dtype=torch.int64,
            ),
        )

    def test_compute_slot_mappings_context_parallel_formula(self):
        block_table = torch.tensor(
            [[10]], dtype=torch.int32, device="mcpu"
        )
        table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.uint64, device="mcpu"
        )
        table_strides = torch.tensor([1], dtype=torch.int64, device="mcpu")
        block_sizes = torch.tensor([4], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 8], dtype=torch.int32, device="mcpu")
        positions = torch.arange(8, dtype=torch.int64, device="mcpu")
        slots = torch.empty((1, 8), dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
            8,
            idx_mapping,
            query_start,
            positions,
            table_ptrs,
            table_strides,
            block_sizes,
            slots,
            slots.stride(0),
            1,
            2,
            1,
            -1,
            1,
            1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu()[0],
            torch.tensor([-1, 40, -1, 41, -1, 42, -1, 43], dtype=torch.int64),
        )

    def test_compute_slot_mappings_group_options_and_dummy_requests(self):
        # KV blocks contain two kernel blocks. Group 0 is sharded, group 1
        # is replicated and group 2 has slot mapping disabled.
        table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device="mcpu")

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device="mcpu")

        for cp_size in (1, 2):
            slots = torch.full((3, 12), 999, dtype=torch.int64, device="mcpu")
            torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
                12,
                tensor([0, -1], torch.int32),
                tensor([0, 8, 10], torch.int32),
                tensor(list(range(8)) + [0, 1000], torch.int64),
                tensor([table.data_ptr()] * 3, torch.uint64),
                tensor([4] * 3, torch.int64),
                tensor([4] * 3, torch.int32),
                slots, slots.stride(0), cp_size - 1, cp_size, 1, -1, 3, 2,
                tensor([2] * 3, torch.int32),
                tensor([True, True, False], torch.bool),
                tensor([True, False, True], torch.bool),
            )
            torch.mcpu.synchronize()
            replicated = list(range(20, 28)) + [-1] * 4
            sharded = (
                [-1, 20, -1, 21, -1, 22, -1, 23] + [-1] * 4
                if cp_size == 2 else replicated
            )
            self.assertEqual(
                slots.cpu(),
                torch.tensor([sharded, replicated, [-1] * 12], dtype=torch.int64),
            )

    def test_compute_slot_mapping_v1_abi_context_parallel_and_tail_padding(self):
        block_table = torch.tensor(
            [[10], [20]], dtype=torch.int32, device="mcpu"
        )
        query_start = torch.tensor([0, 4, 8], dtype=torch.int32, device="mcpu")
        positions = torch.arange(8, dtype=torch.int64, device="mcpu")
        slots = torch.full((10,), 999, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mapping_kernel(
            8,
            10,
            query_start,
            positions,
            block_table,
            block_table.stride(0),
            4,
            slots,
            2,
            1,
            1,
            -1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu(),
            torch.tensor(
                [-1, 40, -1, 41, -1, 82, -1, 83, -1, -1],
                dtype=torch.int64,
            ),
        )


    def test_zero_kv_blocks_uses_segment_addresses(self):
        segments = [
            torch.arange(32, dtype=torch.int32, device="mcpu").reshape(4, 8),
            torch.arange(100, 132, dtype=torch.int32, device="mcpu").reshape(4, 8),
        ]
        expected = [segment.cpu() for segment in segments]
        for tensor in expected:
            tensor[1].zero_()
            tensor[3].zero_()
        seg_addrs = torch.tensor(
            [segment.data_ptr() for segment in segments],
            dtype=torch.uint64,
            device="mcpu",
        )
        block_ids = torch.tensor([1, 3], dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.zero_kv_blocks_kernel_impl(
            seg_addrs, block_ids, 2, 2, 8
        )
        torch.mcpu.synchronize()

        self.assertEqual(segments[0].cpu(), expected[0])
        self.assertEqual(segments[1].cpu(), expected[1])


if __name__ == "__main__":
    run_tests()
