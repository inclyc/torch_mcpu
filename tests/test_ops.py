# Owner(s): ["module: PrivateUse1"]

import collections
import functools
import gc
import subprocess
import sys
import textwrap
import unittest

import torch
import torch_mcpu  # noqa: F401
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    def test_empty(self):
        """Test empty tensor creation"""
        x = torch.empty(3, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3]))

        x = torch.empty([2, 3, 4, 5], device="mcpu", names=["N", "C", "H", "W"])
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([2, 3, 4, 5]))

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="mcpu")
            y = torch.empty(3, 3, device="mcpu:0")
            z = x + y
            self.assertEqual(z.device.type, "mcpu")
            self.assertEqual(z.shape, torch.Size([3, 3]))

    def test_zeros(self):
        """Test zeros tensor creation"""
        y = torch.zeros(3, device="mcpu")
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([3]))

    def test_tensor(self):
        """Test tensor creation from empty tuple"""
        z = torch.tensor((), device="mcpu")
        self.assertEqual(z.device.type, "mcpu")
        self.assertEqual(z.shape, torch.Size([0]))


class TestCopy(TestCase):
    def test_copy_cpu_lazy_source(self):
        original = torch.tensor([1 + 2j, 3 - 4j])
        sources = (
            (torch._neg_view(original), -original),
            (original.conj(), torch.conj_physical(original)),
            (torch._neg_view(original.conj()), -torch.conj_physical(original)),
        )
        for source, expected in sources:
            for non_blocking in (False, True):
                with self.subTest(
                    neg=source.is_neg(),
                    conj=source.is_conj(),
                    non_blocking=non_blocking,
                ):
                    destination = torch.empty_like(original, device="mcpu")
                    self.assertIs(
                        destination.copy_(source, non_blocking=non_blocking),
                        destination,
                    )
                    self.assertEqual(destination.cpu(), expected)
                    self.assertEqual(
                        source.to("mcpu", non_blocking=non_blocking).cpu(), expected
                    )
                    self.assertEqual(original, torch.tensor([1 + 2j, 3 - 4j]))

    def test_copy_mcpu_lazy_identity_is_noop(self):
        original = torch.tensor([1 + 2j, 3 - 4j])
        base = original.to("mcpu")
        for view in (torch._neg_view(base), base.conj(), torch._neg_view(base.conj())):
            for non_blocking in (False, True):
                with self.subTest(
                    neg=view.is_neg(),
                    conj=view.is_conj(),
                    non_blocking=non_blocking,
                ):
                    bits = (view.is_neg(), view.is_conj())
                    self.assertIs(view.copy_(view, non_blocking=non_blocking), view)
                    self.assertEqual((view.is_neg(), view.is_conj()), bits)
                    self.assertEqual(base.cpu(), original)

    def test_copy_mcpu_lazy_views_fail_without_recursion(self):
        # Keep a missing guard's stack overflow or hang out of the test runner.
        code = textwrap.dedent("""
            import torch
            import torch_mcpu

            base = torch.tensor([1 + 2j, 3 - 4j], device="mcpu")
            views = (torch._neg_view(base), base.conj(), torch._neg_view(base.conj()))
            for view in views:
                for device in ("cpu", "mcpu"):
                    plain = torch.empty(2, dtype=base.dtype, device=device)
                    for src, dst in ((view, plain), (plain, view)):
                        for non_blocking in (False, True):
                            for copy in (
                                lambda: dst.copy_(src, non_blocking=non_blocking),
                                lambda: torch.ops.aten._copy_from(src, dst, non_blocking),
                            ):
                                try:
                                    copy()
                                except RuntimeError as error:
                                    assert "lazy negative/conjugate views are unsupported" in str(error), str(error)
                                else:
                                    raise AssertionError("MCPU lazy copy unexpectedly succeeded")
            torch.mcpu.synchronize()
            torch.testing.assert_close(base.cpu(), torch.tensor([1 + 2j, 3 - 4j]))
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_copy_same_device(self):
        """Test copy operation on same device"""
        a = torch.ones(10, device="mcpu").clone()
        self.assertEqual(a.cpu(), torch.ones(10))

    def test_copy_same_device_last_dim_contiguous_views(self):
        """Test same-device copy for views with contiguous innermost dim."""
        src_cpu = torch.randn(3, 7, 5)
        dst_cpu = torch.empty(3, 7, 5)
        src = torch.empty(3, 14, 5, device="mcpu")
        dst = torch.empty(3, 14, 5, device="mcpu")
        src_view = src[:, ::2, :]
        dst_view = dst[:, 1::2, :]
        src_view.copy_(src_cpu.to("mcpu"))
        dst_view.copy_(src_view)
        dst_cpu.copy_(dst_view.cpu())
        self.assertEqual(dst_cpu, src_cpu)

    def test_copy_same_device_coalesces_inner_dims(self):
        """Test copy where multiple trailing dims are contiguous."""
        src_cpu = torch.randn(2, 3, 4)
        src = torch.empty(2, 6, 4, device="mcpu")
        dst = torch.empty(2, 6, 4, device="mcpu")
        src_view = src[:, ::2, :]
        dst_view = dst[:, ::2, :]
        src_view.copy_(src_cpu.to("mcpu"))
        dst_view.copy_(src_view)
        self.assertEqual(dst_view.cpu(), src_cpu)

    def test_copy_same_device_non_contiguous_last_dim_fallback(self):
        """Test copy fallback still handles non-contiguous innermost dim."""
        src_cpu = torch.randn(4, 3)
        dst_cpu = torch.empty(4, 3)
        src = src_cpu.t().contiguous().to("mcpu").t()
        dst_base = torch.empty(3, 4, device="mcpu")
        dst = dst_base.t()
        dst.copy_(src)
        dst_cpu.copy_(dst.cpu())
        self.assertEqual(dst_cpu, src_cpu)

    def test_copy_same_device_overlapping_view_fallback(self):
        """Test same-storage overlapping views keep copy_ overlap semantics."""
        base = torch.arange(6.0, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            base[1:].copy_(base[:-1])

    def test_copy_broadcast_to_singleton_row_gap(self):
        """Broadcast scalars must write every selected row, not adjacent columns."""
        for dtype in (torch.float32, torch.bfloat16):
            for source_device in ("cpu", "mcpu"):
                with self.subTest(dtype=dtype, source_device=source_device):
                    base = torch.full((1, 2, 1, 128), -19, dtype=dtype, device="mcpu")
                    target = base[..., 0]
                    source = torch.tensor(7, dtype=dtype, device=source_device)
                    expanded = source.expand(target.shape)
                    torch.ops.aten._copy_from(expanded, target, non_blocking=True)
                    actual = base.cpu()
                    expected = torch.full(actual.shape, -19, dtype=dtype)
                    expected[..., 0] = 7
                    self.assertEqual(actual, expected)
                    self.assertEqual(target.cpu(), expected[..., 0])

    def test_copy_expanded_identity_and_empty_are_noops(self):
        """Internal overlap is harmless for identical views and empty outputs."""
        base = torch.tensor([[2.0, 7.0]], device="mcpu")
        expanded = base.expand(3, 2)
        identical_view = base.expand(3, 2)
        for non_blocking in (False, True):
            torch.ops.aten._copy_from(expanded, expanded, non_blocking=non_blocking)
            torch.ops.aten._copy_from(
                expanded, identical_view, non_blocking=non_blocking
            )
            empty = torch.empty(1, 0, device="mcpu").expand(3, 0)
            source = torch.empty(3, 0, device="mcpu")
            torch.ops.aten._copy_from(source, empty, non_blocking=non_blocking)
            self.assertEqual(empty.cpu(), torch.empty(3, 0))
        self.assertEqual(base.cpu(), torch.tensor([[2.0, 7.0]]))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_copy_shared_pointer_facade_preserves_synchronization(self):
        """A CPU/MCPU shared pointer still needs its pending writer to finish."""
        host = torch.zeros(3, 5, pin_memory=True)
        facade = torch.mcpu.get_mcpu_view_from_cpu_tensor(host)
        self.assertEqual(host.data_ptr(), facade.data_ptr())
        facade.fill_(13)
        torch.ops.aten._copy_from(facade, host, non_blocking=False)
        self.assertEqual(host, torch.full((3, 5), 13.0))
        host.fill_(29)
        torch.ops.aten._copy_from(host, facade, non_blocking=True)
        torch.mcpu.synchronize()
        self.assertEqual(facade.cpu(), torch.full((3, 5), 29.0))

    def test_copy_singleton_row_gap_preserves_readonly_source(self):
        """A trailing singleton does not make a strided source slice contiguous."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = torch.arange(6 * 16, dtype=torch.float32).reshape(3, 2, 1, 16)
                original = original.to(dtype)
                source_base = original.to("mcpu")
                source = source_base[..., 0]
                destination_base = torch.full(
                    (3, 2, 1, 23), -17, dtype=dtype, device="mcpu"
                )
                destination = destination_base[..., 0]
                torch.ops.aten._copy_from(source, destination, non_blocking=False)
                expected = torch.full(destination_base.shape, -17, dtype=dtype)
                expected[..., 0] = original[..., 0]
                self.assertEqual(destination_base.cpu(), expected)
                self.assertEqual(source_base.cpu(), original)
                self.assertEqual(source.cpu(), original[..., 0])

    def test_copy_trailing_singletons_with_arbitrary_strides(self):
        """Size-one strides may be ignored only after proving dense active axes."""
        original = torch.arange(6.0).reshape(2, 3)
        source_base = original.to("mcpu")
        source = source_base.as_strided((2, 3, 1, 1), (3, 1, 97, 53))
        destination = torch.empty_like(source)
        torch.ops.aten._copy_from(source, destination, non_blocking=False)
        self.assertEqual(destination.cpu(), original.reshape(2, 3, 1, 1))

    def test_copy_memcpy_fastpath_preserves_dense_rows_and_outer_broadcast(self):
        """Keep legal memcpy blocks while rejecting singleton-gap coalescing."""
        cpu_source = torch.arange(24.0).reshape(3, 8)
        source_base = cpu_source.to("mcpu")
        source = source_base[:, :5]
        targets = [
            torch.empty(3, 5, device="mcpu"),
            torch.full((3, 9), -11, dtype=torch.float32, device="mcpu")[:, :5],
            torch.empty(3, 5, device="mcpu"),
        ]
        sources = [source, source, source[:1].expand(3, 5)]
        torch.mcpu.synchronize()
        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            for src, target in zip(sources, targets):
                torch.ops.aten._copy_from(src, target, non_blocking=False)
            torch.ops.aten._copy_from(source_base, source_base, non_blocking=False)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()
        for i, target in enumerate(targets):
            expected = cpu_source[:, :5] if i < 2 else cpu_source[:1, :5].expand(3, 5)
            self.assertEqual(target.cpu(), expected)
        self.assertEqual(source_base.cpu(), cpu_source)
        names = [event.get("name") for thread in timing for event in thread.get("events", [])]
        self.assertEqual(names.count("mcpu::_copy_from.same_device.memcpy"), 3, names)

    def test_cross_device_copy(self):
        """Test copy operation across CPU and mcpu"""
        a = torch.rand(10)
        b = a.to(device="mcpu").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        """Test copy operation across different mcpu devices"""
        a = torch.ones(10, device="mcpu:0").to(device="mcpu:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


class TestOps(TestCase):
    def test_masked_select_uses_cpu_fallback(self):
        tensor_cpu = torch.randn(10)
        tensor_mcpu = tensor_cpu.to(device="mcpu")
        mask = tensor_mcpu.gt(0)
        result = torch.masked_select(tensor_mcpu, mask)

        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(
            result.cpu(), torch.masked_select(tensor_cpu, tensor_cpu.gt(0))
        )

    def test_expand(self):
        """Test tensor expand operation"""
        x = torch.tensor([[1], [2], [3]], device="mcpu")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        """Test tensor resize operation"""
        tensor_cpu = torch.randn([4, 4])

        tensor_mcpu = tensor_cpu.mcpu()
        self.assertTrue(tensor_mcpu.size() == torch.Size([4, 4]))

        storage_mcpu = tensor_mcpu.storage()
        self.assertTrue(storage_mcpu.size() == 16)

        tensor_mcpu.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_mcpu.size() == torch.Size([2, 2, 2, 2]))

        storage_mcpu = tensor_mcpu.storage()
        self.assertTrue(storage_mcpu.size() == 16)

    def test_printing(self):
        """Test tensor printing"""
        a = torch.ones(20, device="mcpu")
        print(a.cpu())

    def test_fill_scalar_last_dim_contiguous_view(self):
        base = torch.zeros(2, 6, 4, dtype=torch.float32, device="mcpu")
        view = base[:, ::2, :]

        view.fill_(3.5)

        expected = torch.zeros(2, 6, 4, dtype=torch.float32)
        expected[:, ::2, :] = 3.5
        self.assertEqual(base.cpu(), expected)

    def test_index_fill_inplace_scalar(self):
        base_cpu = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
        index_cpu = torch.tensor([2, 0], dtype=torch.long)
        expected = base_cpu.clone()
        expected[:, ::2, :].index_fill_(-2, index_cpu, -2.5)

        base = base_cpu.to("mcpu")
        x = base[:, ::2, :]
        index = index_cpu.to("mcpu")
        result = torch.ops.aten.index_fill_.int_Scalar(x, -2, index, -2.5)

        self.assertIs(result, x)
        self.assertEqual(base.cpu(), expected)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        """Test backend dispatch stub for abs operation"""
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_mcpu = x_cpu.to("mcpu")

        y_cpu = torch.abs(x_cpu)
        y_mcpu = torch.abs(x_mcpu)
        self.assertEqual(y_cpu, y_mcpu.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_mcpu = o_cpu.to("mcpu")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_mcpu, out=o_mcpu[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_mcpu.cpu())

        # Explicit mcpu out kernels require callers to provide the exact shape;
        # resizing a temporary CPU view would not update the mcpu metadata.
        with self.assertRaisesRegex(RuntimeError, "aten::abs.out"):
            torch.abs(x_mcpu, out=o_mcpu[:, :, 0:6:3])


class TestQuantization(TestCase):
    def test_quantize(self):
        """Test quantization per tensor"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="mcpu")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("mcpu:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        """Test compiled autograd function that returns self"""
        in_ref = torch.randn(4, device="mcpu", requires_grad=True)
        out_ref = torch.ops.mcpu.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.mcpu.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref.cpu(), out_test.cpu())
        self.assertEqual(in_ref.grad.cpu(), in_test.grad.cpu())

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        """Test compiled autograd function with aliasing"""
        in_ref = torch.randn(4, device="mcpu", requires_grad=True)
        out_ref = torch.ops.mcpu.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.mcpu.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref.cpu(), out_test.cpu())
        self.assertEqual(in_ref.grad.cpu(), in_test.grad.cpu())


class TestFallback(TestCase):
    def test_binary_fallback_does_not_capture_tensor_owners(self):
        torch.mcpu.synchronize()
        gc.collect()
        baseline = torch.mcpu.memory_allocated()

        blocker = torch.zeros(1, dtype=torch.int64, device="mcpu")
        blocker_allocation = torch.mcpu.memory_allocated() - baseline
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 500)
        hidden_states = torch.ones(
            (2048, 6144), dtype=torch.bfloat16, device="mcpu"
        )
        scale = torch.ones((), dtype=torch.bfloat16, device="mcpu")
        hidden_states.mul_(scale)

        del hidden_states, scale
        gc.collect()
        self.assertEqual(
            torch.mcpu.memory_allocated(), baseline + blocker_allocation
        )

        torch.mcpu.synchronize()

    def test_unimplemented_factory_falls_back_to_cpu(self):
        expected = torch.triu_indices(4, 5, offset=1)

        # Repeat to cover the lifetime of the temporary CPU result retained by
        # the asynchronous CPU-to-mcpu copy.
        for _ in range(10):
            result = torch.triu_indices(4, 5, offset=1, device="mcpu")
            self.assertEqual(result.device.type, "mcpu")
            self.assertEqual(result.cpu(), expected)

    def test_functional_fallback_preserves_output_stride(self):
        input_cpu = torch.linspace(-1, 1, 12).reshape(3, 4).t()
        input_mcpu = input_cpu.to("mcpu")

        expected = torch.erf(input_cpu)
        result = torch.erf(input_mcpu)

        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.cpu(), expected)

    def test_out_fallback_writes_strided_tensor_without_touching_gaps(self):
        input_cpu = torch.linspace(-1, 1, 12).reshape(3, 4)
        input_mcpu = input_cpu.to("mcpu")
        base = torch.full((3, 8), -99.0, device="mcpu")
        out = base[:, 1::2]
        original_stride = out.stride()

        result = torch.erf(input_mcpu, out=out)

        expected_base = torch.full((3, 8), -99.0)
        expected_base[:, 1::2] = torch.erf(input_cpu)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), original_stride)
        self.assertEqual(base.cpu(), expected_base)

    def test_tensorlist_fallback_returns_and_mutates_mcpu_tensors(self):
        inputs_cpu = [torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0])]
        inputs = [tensor.to("mcpu") for tensor in inputs_cpu]

        outputs = torch._foreach_erf(inputs)
        for output, expected in zip(outputs, torch._foreach_erf(inputs_cpu)):
            self.assertEqual(output.device.type, "mcpu")
            self.assertEqual(output.cpu(), expected)

        data_ptrs = [tensor.data_ptr() for tensor in inputs]
        mutated = torch._foreach_erf_(inputs)
        for result, original, data_ptr, expected in zip(
            mutated, inputs, data_ptrs, torch._foreach_erf_(inputs_cpu)
        ):
            self.assertIs(result, original)
            self.assertEqual(result.data_ptr(), data_ptr)
            self.assertEqual(result.cpu(), expected)

    def test_tensor_ops_use_explicit_mcpu_kernels(self):
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("mcpu")
        y = torch.Tensor([1, 0, 2]).to("mcpu")
        self.assertTrue(x.device.type, "mcpu")
        self.assertFalse(x.is_cpu)

        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z.cpu())

        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("mcpu")
        z = x[y, y]
        self.assertEqual(z_cpu, z.cpu())

    def test_explicit_forward_ops_do_not_fallback(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], device="mcpu", dtype=torch.float32)
        idx = torch.tensor([1, 0], device="mcpu", dtype=torch.long)

        arange_res = torch.arange(1, 7, 2, device="mcpu", dtype=torch.int64)
        torch.mcpu.synchronize()
        self.assertEqual(arange_res.cpu(), torch.tensor([1, 3, 5], dtype=torch.int64))
        self.assertEqual(arange_res.device.type, "mcpu")

        cumsum_res = torch.cumsum(x, dim=1)
        self.assertEqual(
            cumsum_res.cpu(),
            torch.tensor([[1, 3, 6], [4, 9, 15]], dtype=torch.float32),
        )
        self.assertEqual(cumsum_res.device.type, "mcpu")

        int_cumsum = torch.cumsum(
            torch.tensor([1, 1, 1], device="mcpu", dtype=torch.int8),
            dim=0,
        )
        self.assertEqual(int_cumsum.dtype, torch.int64)
        self.assertEqual(int_cumsum.cpu(), torch.tensor([1, 2, 3], dtype=torch.int64))

        bool_cumsum = torch.cumsum(
            torch.tensor([True, True, False], device="mcpu"),
            dim=0,
        )
        self.assertEqual(bool_cumsum.dtype, torch.int64)
        self.assertEqual(bool_cumsum.cpu(), torch.tensor([1, 2, 2], dtype=torch.int64))

        uniform_res = torch.empty(8, device="mcpu")
        self.assertIs(uniform_res.uniform_(0.0, 1.0), uniform_res)
        uniform_cpu = uniform_res.cpu()
        self.assertTrue((uniform_cpu >= 0.0).all())
        self.assertTrue((uniform_cpu < 1.0).all())

        uniform_func = torch.ops.aten.uniform.default(uniform_res, 0.0, 1.0)
        uniform_func_cpu = uniform_func.cpu()
        self.assertEqual(uniform_func.device.type, "mcpu")
        self.assertTrue((uniform_func_cpu >= 0.0).all())
        self.assertTrue((uniform_func_cpu < 1.0).all())

        uniform_out = torch.empty_like(uniform_res)
        self.assertIs(
            torch.ops.aten.uniform.out(uniform_res, 0.0, 1.0, out=uniform_out),
            uniform_out,
        )
        uniform_out_cpu = uniform_out.cpu()
        self.assertTrue((uniform_out_cpu >= 0.0).all())
        self.assertTrue((uniform_out_cpu < 1.0).all())

        rand_res = torch.rand(8, device="mcpu")
        rand_cpu = rand_res.cpu()
        self.assertEqual(rand_res.device.type, "mcpu")
        self.assertTrue((rand_cpu >= 0.0).all())
        self.assertTrue((rand_cpu < 1.0).all())

        rand_out = torch.empty(8, device="mcpu")
        self.assertIs(torch.rand(8, out=rand_out), rand_out)
        rand_out_cpu = rand_out.cpu()
        self.assertTrue((rand_out_cpu >= 0.0).all())
        self.assertTrue((rand_out_cpu < 1.0).all())

        rand_like = torch.rand_like(uniform_res)
        rand_like_cpu = rand_like.cpu()
        self.assertEqual(rand_like.device.type, "mcpu")
        self.assertTrue((rand_like_cpu >= 0.0).all())
        self.assertTrue((rand_like_cpu < 1.0).all())

        sigmoid_res = torch.sigmoid(x)
        self.assertEqual(sigmoid_res.cpu(), torch.sigmoid(x.cpu()))
        self.assertEqual(sigmoid_res.device.type, "mcpu")

        int_sigmoid = torch.sigmoid(
            torch.tensor([0, 1], device="mcpu", dtype=torch.int32)
        )
        self.assertEqual(int_sigmoid.dtype, torch.float32)
        self.assertEqual(
            int_sigmoid.cpu(),
            torch.sigmoid(torch.tensor([0, 1], dtype=torch.int32)),
        )

        silu_res = torch.nn.functional.silu(x)
        self.assertEqual(silu_res.cpu(), torch.nn.functional.silu(x.cpu()))
        self.assertEqual(silu_res.device.type, "mcpu")

        filled = torch.empty(2, 3, device="mcpu", dtype=torch.float32)
        filled.fill_(7)
        torch.mcpu.synchronize()
        self.assertEqual(filled.cpu(), torch.full((2, 3), 7, dtype=torch.float32))
        self.assertEqual(filled.device.type, "mcpu")

        indexed = x[idx, idx]
        self.assertEqual(indexed.cpu(), torch.tensor([5, 1], dtype=torch.float32))
        self.assertEqual(indexed.device.type, "mcpu")

        bad_out = torch.empty(1, device="mcpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "aten::cumsum.out"):
            torch.cumsum(x, dim=1, out=bad_out)

        bad_index_out = torch.empty(1, device="mcpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "aten::index.Tensor_out"):
            torch.ops.aten.index.Tensor_out(x, [idx, idx], out=bad_index_out)

    def test_explicit_forward_ops_batch_2(self):
        x = torch.tensor([[4.0, 6.0], [7.0, 5.0]], device="mcpu")
        y = torch.tensor([[10.0, 20.0], [31.0, 41.0]], device="mcpu")

        add_out = torch.empty_like(x)
        torch.add(x, y, out=add_out)
        self.assertEqual(add_out.cpu(), torch.tensor([[14.0, 26.0], [38.0, 46.0]]))

        div_out = torch.empty_like(x)
        torch.div(y, x, out=div_out)
        self.assertEqual(
            div_out.cpu(),
            torch.tensor([[2.5, 20.0 / 6.0], [31.0 / 7.0, 8.2]]),
        )

        self.assertEqual(torch.div(y, x).cpu(), y.cpu() / x.cpu())
        div_inplace = y.clone()
        div_inplace.div_(x)
        div_inplace_cpu = y.cpu().clone()
        div_inplace_cpu.div_(x.cpu())
        self.assertEqual(div_inplace.cpu(), div_inplace_cpu)

        int_x = torch.tensor([4, 7], device="mcpu", dtype=torch.int64)
        int_y = torch.tensor([2, 2], device="mcpu", dtype=torch.int64)
        self.assertEqual(
            torch.div(int_x, int_y).cpu(),
            torch.div(int_x.cpu(), int_y.cpu()),
        )

        floor_x = torch.tensor([5, -5, 5, -5], device="mcpu", dtype=torch.int64)
        floor_y = torch.tensor([2, 2, -2, -2], device="mcpu", dtype=torch.int64)
        floor_out = torch.empty_like(floor_x)
        torch.floor_divide(floor_x, floor_y, out=floor_out)
        self.assertEqual(
            floor_out.cpu(), torch.floor_divide(floor_x.cpu(), floor_y.cpu())
        )
        self.assertEqual(
            torch.floor_divide(floor_x, floor_y).cpu(),
            torch.floor_divide(floor_x.cpu(), floor_y.cpu()),
        )
        self.assertEqual(
            torch.floor_divide(floor_x, 2).cpu(),
            torch.floor_divide(floor_x.cpu(), 2),
        )
        floor_inplace = floor_x.clone()
        floor_inplace.floor_divide_(floor_y)
        floor_inplace_cpu = floor_x.cpu().clone()
        floor_inplace_cpu.floor_divide_(floor_y.cpu())
        self.assertEqual(floor_inplace.cpu(), floor_inplace_cpu)

        mul_out = torch.empty_like(x)
        torch.mul(x, y, out=mul_out)
        self.assertEqual(mul_out.cpu(), torch.tensor([[40.0, 120.0], [217.0, 205.0]]))

        self.assertEqual(torch.mul(x, y).cpu(), x.cpu() * y.cpu())
        mul_inplace = x.clone()
        mul_inplace.mul_(y)
        mul_inplace_cpu = x.cpu().clone()
        mul_inplace_cpu.mul_(y.cpu())
        self.assertEqual(mul_inplace.cpu(), mul_inplace_cpu)

        remainder_out = torch.empty_like(y)
        torch.remainder(y, x, out=remainder_out)
        self.assertEqual(
            remainder_out.cpu(),
            torch.tensor([[2.0, 2.0], [3.0, 1.0]]),
        )

        sub_res = torch.sub(y, x)
        self.assertEqual(sub_res.cpu(), torch.tensor([[6.0, 14.0], [24.0, 36.0]]))

        cpu_scalar_tensor = torch.tensor(2.0)
        for name, op in (
            ("add", torch.add),
            ("sub", torch.sub),
            ("mul", torch.mul),
            ("div", torch.div),
        ):
            out = torch.empty_like(x)
            op(x, cpu_scalar_tensor, out=out)
            self.assertEqual(out.cpu(), op(x.cpu(), cpu_scalar_tensor))

            result = op(x, cpu_scalar_tensor)
            self.assertEqual(result.cpu(), op(x.cpu(), cpu_scalar_tensor))

            inplace = x.clone()
            getattr(inplace, name + "_")(cpu_scalar_tensor)
            inplace_cpu = x.cpu().clone()
            getattr(inplace_cpu, name + "_")(cpu_scalar_tensor)
            self.assertEqual(inplace.cpu(), inplace_cpu)

        remainder_cpu_scalar_out = torch.empty_like(y)
        torch.remainder(y, cpu_scalar_tensor, out=remainder_cpu_scalar_out)
        self.assertEqual(
            remainder_cpu_scalar_out.cpu(),
            torch.remainder(y.cpu(), cpu_scalar_tensor),
        )

        cos_out = torch.empty_like(x)
        torch.cos(x, out=cos_out)
        self.assertEqual(cos_out.cpu(), torch.cos(x.cpu()))

        sin_out = torch.empty_like(x)
        torch.sin(x, out=sin_out)
        self.assertEqual(sin_out.cpu(), torch.sin(x.cpu()))

        reciprocal_out = torch.empty_like(x)
        torch.reciprocal(x, out=reciprocal_out)
        self.assertEqual(reciprocal_out.cpu(), torch.reciprocal(x.cpu()))

        rsqrt = torch.rsqrt(x)
        self.assertEqual(rsqrt.cpu(), torch.rsqrt(x.cpu()))

        rsqrt_out = torch.empty_like(x)
        torch.rsqrt(x, out=rsqrt_out)
        self.assertEqual(rsqrt_out.cpu(), torch.rsqrt(x.cpu()))

        integer_abs_input = torch.tensor([-3, 0, 4], device="mcpu")
        integer_abs = torch.abs(integer_abs_input)
        self.assertEqual(integer_abs.cpu(), torch.abs(integer_abs_input.cpu()))

        bool_abs_input = torch.tensor([True, False], device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "not implemented for bool"):
            torch.abs(bool_abs_input)

        complex_abs_input = torch.tensor(
            [3 + 4j, 5 + 12j], dtype=torch.complex64, device="mcpu"
        )
        complex_abs = torch.abs(complex_abs_input)
        self.assertEqual(complex_abs.dtype, torch.float32)
        self.assertEqual(complex_abs.cpu(), torch.abs(complex_abs_input.cpu()))

        integer_rsqrt_input = torch.tensor([1, 4, 16], device="mcpu")
        integer_rsqrt = torch.rsqrt(integer_rsqrt_input)
        self.assertEqual(integer_rsqrt.dtype, torch.get_default_dtype())
        self.assertEqual(
            integer_rsqrt.cpu(), torch.rsqrt(integer_rsqrt_input.cpu())
        )

        neg_out = torch.empty_like(x)
        torch.neg(x, out=neg_out)
        self.assertEqual(neg_out.cpu(), torch.neg(x.cpu()))

        sigmoid_out = torch.empty_like(x)
        torch.sigmoid(x, out=sigmoid_out)
        self.assertEqual(sigmoid_out.cpu(), torch.sigmoid(x.cpu()))

        sigmoid_inplace = x.clone()
        sigmoid_inplace.sigmoid_()
        self.assertEqual(sigmoid_inplace.cpu(), x.cpu().sigmoid_())

        silu_out = torch.empty_like(x)
        torch.ops.aten.silu.out(x, out=silu_out)
        self.assertEqual(silu_out.cpu(), torch.nn.functional.silu(x.cpu()))

        silu_inplace = x.clone()
        torch.ops.aten.silu_.default(silu_inplace)
        self.assertEqual(
            silu_inplace.cpu(),
            torch.ops.aten.silu_.default(x.cpu()),
        )

        clamp_out = torch.empty_like(x)
        torch.clamp(x, min=5.0, max=6.0, out=clamp_out)
        self.assertEqual(clamp_out.cpu(), torch.clamp(x.cpu(), min=5.0, max=6.0))

        clamp_tensor_min = torch.tensor([[5.0, 6.0], [4.0, 6.0]], device="mcpu")
        clamp_tensor_out = torch.empty_like(x)
        torch.clamp(x, min=clamp_tensor_min, out=clamp_tensor_out)
        self.assertEqual(
            clamp_tensor_out.cpu(),
            torch.clamp(x.cpu(), min=clamp_tensor_min.cpu()),
        )

        bad_unary_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::neg.out"):
            torch.neg(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::rsqrt.out"):
            torch.rsqrt(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::abs.out"):
            torch.abs(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::sigmoid.out"):
            torch.sigmoid(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::silu.out"):
            torch.ops.aten.silu.out(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::clamp.out"):
            torch.clamp(x, min=5.0, max=6.0, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::clamp.Tensor_out"):
            torch.clamp(x, min=clamp_tensor_min, out=bad_unary_out)

        scalar_pow_out = torch.empty_like(x)
        torch.pow(2.0, x, out=scalar_pow_out)
        self.assertEqual(scalar_pow_out.cpu(), torch.pow(2.0, x.cpu()))

        tensor_scalar_pow = torch.pow(x, 2)
        self.assertEqual(tensor_scalar_pow.cpu(), torch.pow(x.cpu(), 2))

        tensor_scalar_pow_out = torch.empty_like(x)
        torch.pow(x, 2, out=tensor_scalar_pow_out)
        self.assertEqual(tensor_scalar_pow_out.cpu(), torch.pow(x.cpu(), 2))

        bitwise_lhs = torch.tensor([[True], [False]], device="mcpu")
        bitwise_rhs = torch.tensor([[True, False]], device="mcpu")
        bitwise_result = torch.bitwise_and(bitwise_lhs, bitwise_rhs)
        self.assertEqual(
            bitwise_result.cpu(),
            torch.bitwise_and(bitwise_lhs.cpu(), bitwise_rhs.cpu()),
        )

        bitwise_out = torch.empty((2, 2), dtype=torch.bool, device="mcpu")
        torch.bitwise_and(bitwise_lhs, bitwise_rhs, out=bitwise_out)
        self.assertEqual(
            bitwise_out.cpu(),
            torch.bitwise_and(bitwise_lhs.cpu(), bitwise_rhs.cpu()),
        )

        masked_select_input = torch.arange(6, device="mcpu").reshape(2, 3)
        masked_select_mask = torch.tensor([[True], [False]], device="mcpu")
        masked_select_result = torch.masked_select(
            masked_select_input, masked_select_mask
        )
        self.assertEqual(
            masked_select_result.cpu(),
            torch.masked_select(
                masked_select_input.cpu(), masked_select_mask.cpu()
            ),
        )

        masked_select_out = torch.empty(
            3, dtype=masked_select_input.dtype, device="mcpu"
        )
        torch.masked_select(
            masked_select_input, masked_select_mask, out=masked_select_out
        )
        self.assertEqual(masked_select_out.cpu(), masked_select_result.cpu())

        noncontiguous_pow_input = x.t()
        noncontiguous_pow = torch.pow(noncontiguous_pow_input, 2)
        self.assertEqual(
            noncontiguous_pow.cpu(), torch.pow(noncontiguous_pow_input.cpu(), 2)
        )

        bool_pow_input = torch.tensor([True, False], device="mcpu")
        bool_pow = torch.pow(bool_pow_input, 2)
        self.assertEqual(bool_pow.dtype, torch.int64)
        self.assertEqual(bool_pow.cpu(), torch.pow(bool_pow_input.cpu(), 2))

        with self.assertRaisesRegex(RuntimeError, "aten::pow.Tensor_Scalar_out"):
            torch.pow(x, 2, out=torch.empty(1, device="mcpu"))

        with self.assertRaisesRegex(RuntimeError, "aten::bitwise_and.Tensor_out"):
            torch.bitwise_and(
                bitwise_lhs,
                bitwise_rhs,
                out=torch.empty(1, dtype=torch.bool, device="mcpu"),
            )

        with self.assertRaisesRegex(RuntimeError, "aten::masked_select.out"):
            torch.masked_select(
                masked_select_input,
                masked_select_mask,
                out=torch.empty(
                    1, dtype=masked_select_input.dtype, device="mcpu"
                ),
            )

        cat_out = torch.empty(4, 2, device="mcpu")
        torch.cat([x, y], dim=0, out=cat_out)
        self.assertEqual(cat_out.cpu(), torch.cat([x.cpu(), y.cpu()], dim=0))

        empty_1d = torch.empty(0, device="mcpu")
        cat_empty_out = torch.empty(2, 2, device="mcpu")
        torch.cat([empty_1d, x], dim=0, out=cat_empty_out)
        self.assertEqual(cat_empty_out.cpu(), x.cpu())

        vals = torch.empty(2, 1, device="mcpu")
        inds = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.topk(x, 1, dim=1, largest=True, sorted=True, out=(vals, inds))
        ref_vals, ref_inds = torch.topk(x.cpu(), 1, dim=1, largest=True, sorted=True)
        self.assertEqual(vals.cpu(), ref_vals)
        self.assertEqual(inds.cpu(), ref_inds)

        target = torch.zeros(3, 3, device="mcpu")
        row = torch.tensor([0, 2], device="mcpu", dtype=torch.long)
        col = torch.tensor([1, 0], device="mcpu", dtype=torch.long)
        vals = torch.tensor([5.0, 7.0], device="mcpu")
        torch.ops.aten._index_put_impl_(target, [row, col], vals, False, False)
        self.assertEqual(
            target.cpu(),
            torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        )

        target.zero_()
        self.assertEqual(target.cpu(), torch.zeros(3, 3))

    def test_index_put_accepts_cpu_boolean_mask_snapshot(self):
        target = torch.zeros(5, 3, device="mcpu")
        values = torch.arange(1, 10, dtype=torch.float32, device="mcpu").reshape(
            3, 3
        )
        mask = torch.tensor([True, False, True, True, False], device="cpu")
        expected = target.cpu()
        expected[mask] = values.cpu()

        blocker = torch.zeros(1, dtype=torch.int64, device="mcpu")
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
        target[mask] = values
        mask.fill_(False)
        torch.mcpu.synchronize()

        self.assertEqual(target.cpu(), expected)

    def test_explicit_forward_ops_batch_3(self):
        x = torch.tensor([[0.0, 2.0], [3.0, 0.0]], device="mcpu")
        mask = torch.tensor([[True, False], [False, True]], device="mcpu")

        x.masked_fill_(mask, 9.0)
        self.assertEqual(x.cpu(), torch.tensor([[9.0, 2.0], [3.0, 9.0]]))

        ne_out = torch.empty_like(x, dtype=torch.bool)
        torch.ne(x, 9.0, out=ne_out)
        self.assertEqual(ne_out.cpu(), torch.ne(x.cpu(), 9.0))

        gt_scalar = torch.gt(x, 2.0)
        self.assertEqual(gt_scalar.cpu(), torch.gt(x.cpu(), 2.0))

        gt_scalar_out = torch.empty_like(x, dtype=torch.bool)
        torch.gt(x, 2.0, out=gt_scalar_out)
        self.assertEqual(gt_scalar_out.cpu(), torch.gt(x.cpu(), 2.0))

        gt_other = torch.tensor([[8.0], [2.0]], device="mcpu")
        gt_tensor = torch.gt(x, gt_other)
        self.assertEqual(gt_tensor.cpu(), torch.gt(x.cpu(), gt_other.cpu()))

        gt_tensor_out = torch.empty(2, 2, device="mcpu", dtype=torch.bool)
        torch.gt(x, gt_other, out=gt_tensor_out)
        self.assertEqual(gt_tensor_out.cpu(), torch.gt(x.cpu(), gt_other.cpu()))

        bits = torch.tensor([[0, 1], [2, 3]], device="mcpu", dtype=torch.int32)
        self.assertEqual(torch.bitwise_not(bits).cpu(), torch.bitwise_not(bits.cpu()))

        bitwise_not_out = torch.empty_like(bits)
        torch.bitwise_not(bits, out=bitwise_not_out)
        self.assertEqual(bitwise_not_out.cpu(), torch.bitwise_not(bits.cpu()))

        bitwise_not_inplace = bits.clone()
        bitwise_not_inplace.bitwise_not_()
        self.assertEqual(bitwise_not_inplace.cpu(), bits.cpu().bitwise_not_())

        bad_binary_out = torch.empty(1, device="mcpu", dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "aten::gt.Scalar_out"):
            torch.gt(x, 2.0, out=bad_binary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::gt.Tensor_out"):
            torch.gt(x, gt_other, out=bad_binary_out)

        bad_bitwise_not_out = torch.empty(1, device="mcpu", dtype=bits.dtype)
        with self.assertRaisesRegex(RuntimeError, "aten::bitwise_not.out"):
            torch.bitwise_not(bits, out=bad_bitwise_not_out)

        nz = torch.nonzero(x)
        self.assertEqual(nz.cpu(), torch.nonzero(x.cpu()))

        where_self = torch.where(mask, x, gt_other)
        self.assertEqual(
            where_self.cpu(), torch.where(mask.cpu(), x.cpu(), gt_other.cpu())
        )
        self.assertEqual(where_self.device.type, "mcpu")

        where_out = torch.empty_like(x)
        torch.where(mask, x, gt_other, out=where_out)
        self.assertEqual(
            where_out.cpu(), torch.where(mask.cpu(), x.cpu(), gt_other.cpu())
        )

        self.assertEqual(
            torch.where(mask, 1.5, x).cpu(),
            torch.where(mask.cpu(), 1.5, x.cpu()),
        )
        self.assertEqual(
            torch.where(mask, x, -2).cpu(),
            torch.where(mask.cpu(), x.cpu(), -2),
        )
        self.assertEqual(
            torch.where(mask, 1, 2).cpu(),
            torch.where(mask.cpu(), 1, 2),
        )

        where_indices = torch.where(mask)
        ref_indices = torch.where(mask.cpu())
        self.assertEqual(len(where_indices), len(ref_indices))
        for actual, expected in zip(where_indices, ref_indices):
            self.assertEqual(actual.cpu(), expected)

        scalar_condition = torch.tensor(True, device="mcpu")
        scalar_where_indices = torch.where(scalar_condition)
        self.assertEqual(len(scalar_where_indices), 1)
        self.assertEqual(scalar_where_indices[0].cpu(), torch.tensor([0]))

        bad_where_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::where.self_out"):
            torch.where(mask, x, gt_other, out=bad_where_out)

        bad_where_dtype_out = torch.empty_like(x, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "aten::where.self_out"):
            torch.where(mask, x, gt_other, out=bad_where_dtype_out)

        sum_out = torch.empty(2, device="mcpu")
        torch.sum(x, dim=[1], out=sum_out)
        self.assertEqual(sum_out.cpu(), torch.sum(x.cpu(), dim=[1]))

        self.assertEqual(torch.sum(x).cpu(), torch.sum(x.cpu()))
        sum_all_out = torch.empty((), device="mcpu")
        torch.ops.aten.sum.out(x, out=sum_all_out)
        self.assertEqual(sum_all_out.cpu(), torch.sum(x.cpu()))

        self.assertEqual(
            torch.sum(x, dim=[1], keepdim=True).cpu(),
            torch.sum(x.cpu(), dim=[1], keepdim=True),
        )
        sum_keepdim_out = torch.empty(2, 1, device="mcpu")
        torch.sum(x, dim=[1], keepdim=True, out=sum_keepdim_out)
        self.assertEqual(
            sum_keepdim_out.cpu(), torch.sum(x.cpu(), dim=[1], keepdim=True)
        )

        int_sum = torch.sum(torch.tensor([1, 2, 3], device="mcpu"))
        self.assertEqual(int_sum.dtype, torch.int64)
        self.assertEqual(int_sum.cpu(), torch.tensor(6, dtype=torch.int64))

        int_matrix = torch.tensor([[1, 2], [3, 4]], device="mcpu", dtype=torch.int32)
        self.assertEqual(
            torch.sum(int_matrix, dtype=torch.float32).cpu(),
            torch.sum(int_matrix.cpu(), dtype=torch.float32),
        )
        int_sum_dim = torch.sum(int_matrix, dim=[0], dtype=torch.float32)
        self.assertEqual(
            int_sum_dim.cpu(), torch.sum(int_matrix.cpu(), dim=[0], dtype=torch.float32)
        )
        int_sum_dim_out = torch.empty(2, device="mcpu", dtype=torch.float32)
        torch.sum(int_matrix, dim=[0], dtype=torch.float32, out=int_sum_dim_out)
        self.assertEqual(
            int_sum_dim_out.cpu(),
            torch.sum(int_matrix.cpu(), dim=[0], dtype=torch.float32),
        )

        bool_sum = torch.sum(torch.tensor([True, False, True], device="mcpu"))
        self.assertEqual(bool_sum.dtype, torch.int64)
        self.assertEqual(bool_sum.cpu(), torch.tensor(2, dtype=torch.int64))

        self.assertEqual(
            torch.sum(x, dim=[0, 1]).cpu(), torch.sum(x.cpu(), dim=[0, 1])
        )

        bad_sum_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::sum.IntList_out"):
            torch.sum(x, dim=[1], out=bad_sum_out)

        bad_sum_all_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::sum.out"):
            torch.ops.aten.sum.out(x, out=bad_sum_all_out)

    def test_explicit_comparison_ops_and_variants(self):
        lhs_cpu = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
        rhs_cpu = torch.tensor([[1.0], [3.0]])
        lhs = lhs_cpu.to("mcpu")
        rhs = rhs_cpu.to("mcpu")

        for name in ("lt", "eq", "gt", "le", "ge", "ne"):
            op = getattr(torch, name)

            scalar = op(lhs, 2.0)
            self.assertEqual(scalar.cpu(), op(lhs_cpu, 2.0))

            scalar_out = torch.empty_like(lhs, dtype=torch.bool)
            op(lhs, 2.0, out=scalar_out)
            self.assertEqual(scalar_out.cpu(), op(lhs_cpu, 2.0))

            tensor = op(lhs, rhs)
            self.assertEqual(tensor.cpu(), op(lhs_cpu, rhs_cpu))

            tensor_out = torch.empty_like(lhs, dtype=torch.bool)
            op(lhs, rhs, out=tensor_out)
            self.assertEqual(tensor_out.cpu(), op(lhs_cpu, rhs_cpu))

            inplace_scalar = lhs.clone()
            getattr(inplace_scalar, name + "_")(2.0)
            inplace_scalar_cpu = lhs_cpu.clone()
            getattr(inplace_scalar_cpu, name + "_")(2.0)
            self.assertEqual(inplace_scalar.cpu(), inplace_scalar_cpu)

            inplace_tensor = lhs.clone()
            getattr(inplace_tensor, name + "_")(rhs)
            inplace_tensor_cpu = lhs_cpu.clone()
            getattr(inplace_tensor_cpu, name + "_")(rhs_cpu)
            self.assertEqual(inplace_tensor.cpu(), inplace_tensor_cpu)

            cpu_scalar_tensor = torch.tensor(2.0)
            cpu_scalar_tensor_result = op(lhs, cpu_scalar_tensor)
            self.assertEqual(
                cpu_scalar_tensor_result.cpu(), op(lhs_cpu, cpu_scalar_tensor)
            )

            cpu_scalar_tensor_out = torch.empty_like(lhs, dtype=torch.bool)
            op(lhs, cpu_scalar_tensor, out=cpu_scalar_tensor_out)
            self.assertEqual(
                cpu_scalar_tensor_out.cpu(), op(lhs_cpu, cpu_scalar_tensor)
            )

            inplace_cpu_scalar_tensor = lhs.clone()
            getattr(inplace_cpu_scalar_tensor, name + "_")(cpu_scalar_tensor)
            inplace_cpu_scalar_tensor_cpu = lhs_cpu.clone()
            getattr(inplace_cpu_scalar_tensor_cpu, name + "_")(cpu_scalar_tensor)
            self.assertEqual(
                inplace_cpu_scalar_tensor.cpu(), inplace_cpu_scalar_tensor_cpu
            )

        less = torch.ops.aten.less
        self.assertEqual(less.Scalar(lhs, 2.0).cpu(), torch.lt(lhs_cpu, 2.0))
        self.assertEqual(
            less.Tensor(lhs, rhs).cpu(),
            torch.lt(lhs_cpu, rhs_cpu),
        )
        self.assertEqual(
            less.Tensor(lhs, torch.tensor(2.0)).cpu(),
            torch.lt(lhs_cpu, torch.tensor(2.0)),
        )

        less_equal = torch.ops.aten.less_equal
        self.assertEqual(
            less_equal.Scalar(lhs, 2.0).cpu(),
            torch.le(lhs_cpu, 2.0),
        )
        self.assertEqual(
            less_equal.Tensor(lhs, rhs).cpu(),
            torch.le(lhs_cpu, rhs_cpu),
        )
        self.assertEqual(
            less_equal.Tensor(lhs, torch.tensor(2.0)).cpu(),
            torch.le(lhs_cpu, torch.tensor(2.0)),
        )

    def test_explicit_forward_ops_batch_4(self):
        x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="mcpu")

        max_values = torch.empty(2, 1, device="mcpu")
        max_indices = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.max(x, dim=1, keepdim=True, out=(max_values, max_indices))
        ref_values, ref_indices = torch.max(x.cpu(), dim=1, keepdim=True)
        self.assertEqual(max_values.cpu(), ref_values)
        self.assertEqual(max_indices.cpu(), ref_indices)

        mean_res = torch.empty(2, 1, device="mcpu")
        torch.mean(x, dim=[1], keepdim=True, out=mean_res)
        self.assertEqual(mean_res.cpu(), torch.mean(x.cpu(), dim=[1], keepdim=True))

        scatter_self = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mcpu")
        scatter_index = torch.tensor([[0, 1], [1, 0]], device="mcpu", dtype=torch.long)
        scatter_src = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="mcpu")
        scatter_out = torch.empty_like(scatter_self)
        torch.scatter_add(scatter_self, 1, scatter_index, scatter_src, out=scatter_out)
        self.assertEqual(
            scatter_out.cpu(),
            torch.scatter_add(scatter_self.cpu(), 1, scatter_index.cpu(), scatter_src.cpu()),
        )

        scatter_src_out = torch.empty_like(scatter_self)
        torch.scatter(scatter_self, 1, scatter_index, scatter_src, out=scatter_src_out)
        self.assertEqual(
            scatter_src_out.cpu(),
            torch.scatter(scatter_self.cpu(), 1, scatter_index.cpu(), scatter_src.cpu()),
        )
        self.assertEqual(scatter_src_out.device.type, "mcpu")

        softmax_out = torch.empty_like(x)
        torch.ops.aten._softmax.out(x, 1, False, out=softmax_out)
        self.assertEqual(
            softmax_out.cpu(),
            torch.ops.aten._softmax.out(x.cpu(), 1, False, out=torch.empty_like(x.cpu())),
        )

        exp_sample = torch.empty(8, device="mcpu")
        exp_sample.exponential_(2.0)
        self.assertEqual(exp_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(exp_sample.cpu()).all())
        self.assertTrue((exp_sample.cpu() >= 0).all())

        exp_generator = torch.Generator(device="mcpu").manual_seed(123)
        exp_with_generator = torch.empty(2, 4, device="mcpu")
        exp_with_generator[0].exponential_(2.0, generator=exp_generator)
        torch.mcpu.synchronize()
        exp_with_generator_cpu = exp_with_generator[0].cpu()
        self.assertTrue(torch.isfinite(exp_with_generator_cpu).all())
        self.assertTrue((exp_with_generator_cpu >= 0).all())

        exp_generator.manual_seed(123)
        exp_repeat = torch.empty(4, device="mcpu")
        exp_repeat.exponential_(2.0, generator=exp_generator)
        torch.mcpu.synchronize()
        self.assertEqual(exp_with_generator_cpu, exp_repeat.cpu())

        uniform_sample = torch.empty(16, device="mcpu")
        uniform_result = uniform_sample.uniform_(-1.0, 2.0)
        uniform_cpu = uniform_sample.cpu()
        self.assertIs(uniform_result, uniform_sample)
        self.assertEqual(uniform_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(uniform_cpu).all())
        self.assertTrue((uniform_cpu >= -1.0).all())
        self.assertTrue((uniform_cpu < 2.0).all())

        uniform_generator = torch.Generator(device="mcpu").manual_seed(123)
        uniform_with_generator = torch.empty(2, 8, device="mcpu")
        uniform_with_generator[0].uniform_(
            -1.0, 2.0, generator=uniform_generator
        )
        torch.mcpu.synchronize()
        uniform_with_generator_cpu = uniform_with_generator[0].cpu()

        uniform_generator.manual_seed(123)
        uniform_repeat = torch.empty(8, device="mcpu")
        uniform_repeat.uniform_(-1.0, 2.0, generator=uniform_generator)
        torch.mcpu.synchronize()
        self.assertEqual(uniform_with_generator_cpu, uniform_repeat.cpu())

        uniform_out = torch.empty_like(uniform_sample)
        torch.ops.aten.uniform.out(uniform_sample, -1.0, 2.0, out=uniform_out)
        uniform_out_cpu = uniform_out.cpu()
        self.assertEqual(uniform_out.shape, uniform_sample.shape)
        self.assertEqual(uniform_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(uniform_out_cpu).all())
        self.assertTrue((uniform_out_cpu >= -1.0).all())
        self.assertTrue((uniform_out_cpu < 2.0).all())

        bad_uniform_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::uniform.out"):
            torch.ops.aten.uniform.out(
                uniform_sample, -1.0, 2.0, out=bad_uniform_out
            )

        normal_sample = torch.empty(8, device="mcpu")
        normal_result = normal_sample.normal_(0.0, 1.0)
        self.assertIs(normal_result, normal_sample)
        self.assertEqual(normal_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_sample.cpu()).all())

        normal_out = torch.empty_like(x)
        torch.ops.aten.normal.out(x, 0.0, 1.0, out=normal_out)
        self.assertEqual(normal_out.shape, x.shape)
        self.assertEqual(normal_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_out.cpu()).all())

        normal_factory = torch.normal(0.0, 1.0, (2, 3), device="mcpu")
        self.assertEqual(normal_factory.shape, torch.Size([2, 3]))
        self.assertEqual(normal_factory.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_factory.cpu()).all())

        mean = torch.zeros(2, 1, device="mcpu")
        std = torch.ones(1, 3, device="mcpu")
        normal_mean = torch.normal(mean, 1.0)
        self.assertEqual(normal_mean.shape, mean.shape)
        self.assertEqual(normal_mean.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_mean.cpu()).all())

        normal_std = torch.normal(0.0, std)
        self.assertEqual(normal_std.shape, std.shape)
        self.assertEqual(normal_std.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_std.cpu()).all())

        normal_tensor = torch.normal(mean, std)
        self.assertEqual(normal_tensor.shape, torch.Size([2, 3]))
        self.assertEqual(normal_tensor.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_tensor.cpu()).all())

        normal_tensor_out = torch.empty(2, 3, device="mcpu")
        torch.normal(mean, std, out=normal_tensor_out)
        self.assertEqual(normal_tensor_out.shape, torch.Size([2, 3]))
        self.assertEqual(normal_tensor_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_tensor_out.cpu()).all())

        normal_float_float_out = torch.empty(2, 3, device="mcpu")
        torch.ops.aten.normal.float_float_out(
            0.0, 1.0, [2, 3], out=normal_float_float_out
        )
        self.assertEqual(normal_float_float_out.shape, torch.Size([2, 3]))
        self.assertEqual(normal_float_float_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_float_float_out.cpu()).all())

        argmax_out = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.argmax(x, dim=1, keepdim=True, out=argmax_out)
        self.assertEqual(argmax_out.cpu(), torch.argmax(x.cpu(), dim=1, keepdim=True))

        self.assertEqual(torch.max(x).cpu(), torch.max(x.cpu()))
        max_unary_out = torch.empty((), device="mcpu")
        torch.max(x, out=max_unary_out)
        self.assertEqual(max_unary_out.cpu(), torch.max(x.cpu()))

        max_values, max_indices = torch.max(x, dim=1, keepdim=True)
        expected_max_values, expected_max_indices = torch.max(
            x.cpu(), dim=1, keepdim=True
        )
        self.assertEqual(max_values.cpu(), expected_max_values)
        self.assertEqual(max_indices.cpu(), expected_max_indices)
        max_values_out = torch.empty(2, 1, device="mcpu")
        max_indices_out = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.max(x, dim=1, keepdim=True, out=(max_values_out, max_indices_out))
        self.assertEqual(max_values_out.cpu(), expected_max_values)
        self.assertEqual(max_indices_out.cpu(), expected_max_indices)

        self.assertEqual(torch.min(x).cpu(), torch.min(x.cpu()))
        min_unary_out = torch.empty((), device="mcpu")
        torch.min(x, out=min_unary_out)
        self.assertEqual(min_unary_out.cpu(), torch.min(x.cpu()))

        min_values, min_indices = torch.min(x, dim=1, keepdim=True)
        expected_min_values, expected_min_indices = torch.min(
            x.cpu(), dim=1, keepdim=True
        )
        self.assertEqual(min_values.cpu(), expected_min_values)
        self.assertEqual(min_indices.cpu(), expected_min_indices)
        min_values_out = torch.empty(2, 1, device="mcpu")
        min_indices_out = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.min(x, dim=1, keepdim=True, out=(min_values_out, min_indices_out))
        self.assertEqual(min_values_out.cpu(), expected_min_values)
        self.assertEqual(min_indices_out.cpu(), expected_min_indices)

        self.assertEqual(torch.mean(x).cpu(), torch.mean(x.cpu()))
        mean_dtype_out = torch.empty((), device="mcpu")
        torch.mean(x, out=mean_dtype_out)
        self.assertEqual(mean_dtype_out.cpu(), torch.mean(x.cpu()))
        self.assertEqual(
            torch.mean(x, dim=[1], keepdim=True).cpu(),
            torch.mean(x.cpu(), dim=[1], keepdim=True),
        )
        mean_out = torch.empty(2, 1, device="mcpu")
        torch.mean(x, dim=[1], keepdim=True, out=mean_out)
        self.assertEqual(mean_out.cpu(), torch.mean(x.cpu(), dim=[1], keepdim=True))

        self.assertEqual(
            torch.amax(x, dim=[0, 1]).cpu(), torch.amax(x.cpu(), dim=[0, 1])
        )
        amax_out = torch.empty(2, 1, device="mcpu")
        torch.amax(x, dim=[1], keepdim=True, out=amax_out)
        self.assertEqual(amax_out.cpu(), torch.amax(x.cpu(), dim=[1], keepdim=True))
        self.assertEqual(
            torch.amin(x, dim=[0, 1]).cpu(), torch.amin(x.cpu(), dim=[0, 1])
        )
        amin_out = torch.empty(2, 1, device="mcpu")
        torch.amin(x, dim=[1], keepdim=True, out=amin_out)
        self.assertEqual(amin_out.cpu(), torch.amin(x.cpu(), dim=[1], keepdim=True))

        bad_max_values = torch.empty(1, device="mcpu")
        bad_max_indices = torch.empty(1, device="mcpu", dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "aten::max.dim_max"):
            torch.max(x, dim=1, keepdim=True, out=(bad_max_values, bad_max_indices))

        bad_min_values = torch.empty(1, device="mcpu")
        bad_min_indices = torch.empty(1, device="mcpu", dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "aten::min.dim_min"):
            torch.min(x, dim=1, keepdim=True, out=(bad_min_values, bad_min_indices))

        bad_mean_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::mean.out"):
            torch.mean(x, dim=[1], keepdim=True, out=bad_mean_out)

        bad_amax_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::amax.out"):
            torch.amax(x, dim=[1], keepdim=True, out=bad_amax_out)

        bad_amin_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::amin.out"):
            torch.amin(x, dim=[1], keepdim=True, out=bad_amin_out)

        bad_scatter_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::scatter_add.out"):
            torch.scatter_add(scatter_self, 1, scatter_index, scatter_src, out=bad_scatter_out)

        with self.assertRaisesRegex(RuntimeError, "aten::scatter.src_out"):
            torch.scatter(scatter_self, 1, scatter_index, scatter_src, out=bad_scatter_out)

        with self.assertRaisesRegex(RuntimeError, "make_cpu_view_spec expects an mcpu tensor"):
            torch.scatter(
                scatter_self,
                1,
                scatter_index.cpu(),
                scatter_src,
                out=torch.empty_like(scatter_self),
            )

        bad_softmax_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::_softmax.out"):
            torch.ops.aten._softmax.out(x, 1, False, out=bad_softmax_out)

        bad_argmax_out = torch.empty(1, device="mcpu", dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "aten::argmax.out"):
            torch.argmax(x, dim=1, keepdim=True, out=bad_argmax_out)

        bad_normal_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::normal.out"):
            torch.ops.aten.normal.out(x, 0.0, 1.0, out=bad_normal_out)

        with self.assertRaisesRegex(RuntimeError, "aten::normal.Tensor_Tensor_out"):
            torch.normal(mean, std, out=bad_normal_out)

    def test_explicit_gather_out(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mcpu")
        index = torch.tensor([[2, 1], [0, 2]], device="mcpu", dtype=torch.long)
        out = torch.empty(2, 2, device="mcpu")

        torch.gather(x, 1, index, out=out)
        self.assertEqual(
            out.cpu(),
            torch.gather(x.cpu(), 1, index.cpu()),
        )
        self.assertEqual(out.device.type, "mcpu")

        bad_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::gather.out"):
            torch.gather(x, 1, index, out=bad_out)

    def test_explicit_index_select_and_index_copy(self):
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mcpu"
        )
        index = torch.tensor([2, 0], device="mcpu", dtype=torch.long)

        selected = torch.index_select(x, 1, index)
        self.assertEqual(selected.cpu(), torch.index_select(x.cpu(), 1, index.cpu()))
        self.assertEqual(selected.device.type, "mcpu")

        selected_out = torch.empty(2, 2, device="mcpu")
        torch.index_select(x, 1, index, out=selected_out)
        self.assertEqual(
            selected_out.cpu(), torch.index_select(x.cpu(), 1, index.cpu())
        )

        bad_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::index_select.out"):
            torch.index_select(x, 1, index, out=bad_out)

        target = torch.zeros(2, 3, device="mcpu")
        source = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]], device="mcpu"
        )
        result = target.index_copy_(1, index, source)
        self.assertIs(result, target)
        self.assertEqual(
            target.cpu(),
            torch.zeros(2, 3).index_copy_(1, index.cpu(), source.cpu()),
        )

    def test_constant_pad_nd_uses_mcpu_stream(self):
        x_cpu = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        x = x_cpu.to("mcpu")

        result = torch.ops.aten.constant_pad_nd.default(x, [1, 2, 1, 0], -3)
        expected = torch.ops.aten.constant_pad_nd.default(
            x_cpu, [1, 2, 1, 0], -3
        )
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.cpu(), expected)

        cropped = torch.ops.aten.constant_pad_nd.default(x, [-1, 2], 5)
        expected_cropped = torch.ops.aten.constant_pad_nd.default(
            x_cpu, [-1, 2], 5
        )
        self.assertEqual(cropped.cpu(), expected_cropped)

        out = torch.empty(expected.shape, device="mcpu", dtype=x.dtype)
        torch.ops.aten.constant_pad_nd.out(x, [1, 2, 1, 0], -3, out=out)
        self.assertEqual(out.cpu(), expected)

        stream = torch.Stream(device="mcpu")
        blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
        with stream:
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
            ordered = torch.ops.aten.constant_pad_nd.default(x, [0, 1], 7)
            self.assertFalse(stream.query())
        stream.synchronize()
        self.assertEqual(
            ordered.cpu(),
            torch.ops.aten.constant_pad_nd.default(x_cpu, [0, 1], 7),
        )

    def test_tensorlist_op_does_not_fallback_to_cpu(self):
        v_mcpu = torch.Tensor([1, 2, 3]).to("mcpu")
        x = (v_mcpu, v_mcpu)
        y = (v_mcpu, v_mcpu)

        self.assertTrue(v_mcpu.device.type == "mcpu")
        self.assertFalse(v_mcpu.is_cpu)

        z = torch._foreach_add(x, y)
        self.assertEqual(z[0].device.type, "mcpu")
        self.assertEqual(z[1].device.type, "mcpu")


class TestSDPA(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        """Test fused SDP choice for privateuse1 backend"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        assert (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        """Test scaled dot product fused attention overrideable forward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        """Test scaled dot product fused attention overrideable backward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        attn_mask_privateuse1 = attn_mask.to("mcpu")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("mcpu")
        grad_input_mask = [True, True, True, True]
        _grad_q, _grad_k, _grad_v, _grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


class TestFactoryExtended(TestCase):
    def test_empty_with_memory_format(self):
        """Test empty tensor creation with memory format"""
        x = torch.empty(1, 2, 3, 4, device="mcpu", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([1, 2, 3, 4]))

        x = torch.empty(
            2, 3, 4, device="mcpu", memory_format=torch.contiguous_format
        )
        self.assertEqual(x.device.type, "mcpu")
        self.assertTrue(x.is_contiguous())

    def test_empty_strided(self):
        """Test empty_strided tensor creation"""
        size = (3, 4)
        stride = (4, 1)
        x = torch.empty_strided(size, stride, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size(size))
        self.assertEqual(x.stride(), stride)

    def test_ones(self):
        """Test ones tensor creation"""
        x = torch.ones(3, 4, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x.cpu() == 1))

    def test_ones_like(self):
        """Test ones_like tensor creation"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.ones_like(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.all(y.cpu() == 1))

    def test_randn(self):
        """Test randn tensor creation"""
        x = torch.randn(3, 4, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))

    def test_full(self):
        """Test full tensor creation"""
        x = torch.full((3, 4), 5.0, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x.cpu() == 5.0))


class TestCopyExtended(TestCase):
    def test_copy_different_dtypes(self):
        """Test copy with different dtypes"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        y = torch.empty(3, 4, dtype=torch.float64, device="mcpu")
        y.copy_(x)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(y.cpu(), x.cpu().double())

    def test_clone(self):
        """Test tensor clone"""
        x = torch.randn(3, 4, device="mcpu")
        y = x.clone()
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.cpu(), x.cpu())
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_non_blocking(self):
        """Test non-blocking copy"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        y.copy_(x, non_blocking=True)
        self.assertEqual(y.cpu(), x.cpu())


class TestOpsExtended(TestCase):
    def test_view(self):
        """Test tensor view operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        y = x.view(6, 4)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([6, 4]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_reshape(self):
        """Test tensor reshape operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        y = x.reshape(6, 4)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([6, 4]))

    def test_as_strided(self):
        """Test as_strided operation"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.as_strided(x, (2, 2), (4, 1), 1)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_unfold(self):
        """Test unfold view operation"""
        x_cpu = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(
            2, 3, 4, 5, 6
        )
        x = x_cpu.to(device="mcpu")

        y = x.unfold(2, 2, 2).unfold(3, 3, 1).unfold(4, 2, 2)
        y_cpu = x_cpu.unfold(2, 2, 2).unfold(3, 3, 1).unfold(4, 2, 2)

        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, y_cpu.shape)
        self.assertEqual(y.stride(), y_cpu.stride())
        self.assertEqual(y.cpu(), y_cpu)
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_local_scalar_dense(self):
        """Test local scalar dense extraction"""
        x = torch.tensor([5.0], device="mcpu")
        scalar = x.item()
        self.assertEqual(scalar, 5.0)

    def test_set_tensor(self):
        """Test set_ operation with tensor source"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        y.set_(x)
        self.assertEqual(y.cpu(), x.cpu())

    def test_set_storage(self):
        """Test set_ operation with storage source"""
        x = torch.randn(3, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(3, 4, device="mcpu")
        y.set_(storage, 0, y.size())
        self.assertEqual(y.cpu(), x.cpu())


class TestSTUBExtended(TestCase):
    def test_abs_contiguous(self):
        """Test abs operation with contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_non_contiguous(self):
        """Test abs operation with non-contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        x_t = x.t()  # Transpose makes it non-contiguous
        y = torch.abs(x_t)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))

    def test_custom_abs(self):
        """Test custom abs operation"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        y = torch.ops.mcpu.custom_abs(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_out(self):
        """Test abs with output tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "mcpu")
        self.assertTrue(torch.all(out.cpu() >= 0))
        self.assertEqual(out.cpu(), torch.abs(x.cpu()))


@unittest.skip("Skipping all quantization tests for mcpu backend")
class TestQuantizationExtended(TestCase):
    def test_quantize_per_tensor_different_scales(self):
        """Test quantization with different scales"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="mcpu")

        scale = 0.1
        zero_point = 10
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        self.assertEqual(quantized.device.type, "mcpu")
        self.assertEqual(quantized.dtype, torch.qint8)
        self.assertEqual(quantized.q_scale(), scale)
        self.assertEqual(quantized.q_zero_point(), zero_point)

    def test_quantize_per_tensor_quint8(self):
        """Test quantization with quint8 dtype"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        quantized = torch.quantize_per_tensor(x, 0.1, 128, torch.quint8)
        self.assertEqual(quantized.device.type, "mcpu")
        self.assertEqual(quantized.dtype, torch.quint8)

    def test_dequantize(self):
        """Test dequantization"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        quantized = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        dequantized = quantized.dequantize()
        self.assertEqual(dequantized.device.type, "mcpu")
        self.assertEqual(dequantized.dtype, torch.float32)


class TestFallbackExtended(TestCase):
    def _timed_event_names(self, fn):
        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            result = fn()
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        return result, event_names

    def test_contiguous_add_sub_use_cpu_view_kernel(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1).to("mcpu")

        (add_res, sub_res), event_names = self._timed_event_names(
            lambda: (torch.add(x, y), torch.sub(y, x))
        )

        self.assertEqual(add_res.cpu(), x.cpu() + y.cpu())
        self.assertEqual(sub_res.cpu(), y.cpu() - x.cpu())
        self.assertIn("mcpu::aten::add", event_names)
        self.assertIn("mcpu::aten::sub.Tensor", event_names)
        self.assertNotIn("mcpu::aten::add.raw", event_names)
        self.assertNotIn("mcpu::aten::sub.raw", event_names)

    def test_add_sub_cpu_view_kernel_cases(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = torch.arange(4, dtype=torch.float32).to("mcpu")

        broadcast_res = torch.add(x, y)
        self.assertEqual(broadcast_res.cpu(), x.cpu() + y.cpu())

        alpha_res = torch.sub(x, x, alpha=0.5)
        self.assertEqual(alpha_res.cpu(), x.cpu() - x.cpu() * 0.5)

        _, event_names = self._timed_event_names(
            lambda: (
                (
                    lambda base: torch.add(base[:-1], base[:-1], out=base[1:])
                )(torch.arange(5, dtype=torch.float32, device="mcpu"))
            )
        )

        self.assertNotIn("mcpu::aten::add.raw", event_names)

    def test_cpu_zero_dim_tensor_binary_ops_use_tensor_fallback(self):
        x = torch.tensor([[4.0, 6.0], [7.0, 5.0]], device="mcpu")
        other = torch.tensor(2.0)

        def run_ops():
            add_out = torch.empty_like(x)
            sub_out = torch.empty_like(x)
            mul_out = torch.empty_like(x)
            div_out = torch.empty_like(x)
            floor_divide_out = torch.empty_like(x)
            remainder_out = torch.empty_like(x)
            gt_out = torch.empty_like(x, dtype=torch.bool)

            torch.add(x, other, out=add_out)
            torch.sub(x, other, out=sub_out)
            torch.mul(x, other, out=mul_out)
            torch.div(x, other, out=div_out)
            torch.floor_divide(x, other, out=floor_divide_out)
            torch.remainder(x, other, out=remainder_out)
            torch.gt(x, other, out=gt_out)

            return (
                add_out,
                sub_out,
                mul_out,
                div_out,
                floor_divide_out,
                remainder_out,
                gt_out,
            )

        (
            add_out,
            sub_out,
            mul_out,
            div_out,
            floor_divide_out,
            remainder_out,
            gt_out,
        ), event_names = self._timed_event_names(run_ops)

        x_cpu = x.cpu()
        self.assertEqual(add_out.cpu(), torch.add(x_cpu, other))
        self.assertEqual(sub_out.cpu(), torch.sub(x_cpu, other))
        self.assertEqual(mul_out.cpu(), torch.mul(x_cpu, other))
        self.assertEqual(div_out.cpu(), torch.div(x_cpu, other))
        self.assertEqual(floor_divide_out.cpu(), torch.floor_divide(x_cpu, other))
        self.assertEqual(remainder_out.cpu(), torch.remainder(x_cpu, other))
        self.assertEqual(gt_out.cpu(), torch.gt(x_cpu, other))

        self.assertIn("mcpu::aten::add", event_names)
        self.assertIn("mcpu::aten::sub.out", event_names)
        self.assertIn("mcpu::aten::mul.out", event_names)
        self.assertIn("mcpu::aten::div.out", event_names)
        self.assertIn("mcpu::aten::floor_divide.out", event_names)
        self.assertIn("mcpu::aten::remainder.Tensor_out", event_names)
        self.assertIn("mcpu::aten::gt.Tensor_out", event_names)

        for raw_name in (
            "mcpu::aten::add.raw",
            "mcpu::aten::add.Scalar.raw",
            "mcpu::aten::sub.raw",
            "mcpu::aten::sub.Scalar.raw",
            "mcpu::aten::mul.raw",
            "mcpu::aten::div.raw",
            "mcpu::aten::floor_divide.raw",
            "mcpu::aten::gt.Tensor.raw",
            "mcpu::aten::gt.Scalar.raw",
        ):
            self.assertNotIn(raw_name, event_names)

    def test_strided_sigmoid_uses_raw_and_mul_uses_cpu_view_kernel(self):
        tokens = 5
        heads = 3
        head_dim = 4
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            q_gate_cpu = torch.randn(tokens, heads, 2 * head_dim, dtype=dtype)
            attn_cpu = torch.randn(tokens, heads, head_dim, dtype=dtype)
            q_gate = q_gate_cpu.to("mcpu")
            attn = attn_cpu.to("mcpu")
            gate = q_gate[..., head_dim:]

            result, event_names = self._timed_event_names(
                lambda: attn * torch.sigmoid(gate)
            )

            expected = attn_cpu * torch.sigmoid(q_gate_cpu[..., head_dim:])
            self.assertEqual(result.cpu(), expected)
            self.assertFalse(gate.is_contiguous())
            self.assertIn("mcpu::aten::sigmoid.raw", event_names)
            self.assertIn("mcpu::aten::mul.Tensor", event_names)
            self.assertNotIn("mcpu::aten::sigmoid", event_names)
            self.assertNotIn("mcpu::aten::mul.raw", event_names)

    def test_double_unary_falls_back_from_raw_kernel(self):
        x = torch.tensor([1e-300, 2.0], dtype=torch.float64, device="mcpu")
        out = torch.empty_like(x)

        result, event_names = self._timed_event_names(
            lambda: torch.reciprocal(x, out=out)
        )

        self.assertIs(result, out)
        self.assertEqual(out.cpu(), torch.reciprocal(x.cpu()))
        self.assertIn("mcpu::aten::reciprocal.out", event_names)
        self.assertNotIn("mcpu::aten::reciprocal.raw", event_names)

    def test_mul_cpu_view_kernel_cases(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = torch.arange(4, dtype=torch.float32).to("mcpu")

        result, event_names = self._timed_event_names(lambda: torch.mul(x, y))
        self.assertEqual(result.cpu(), x.cpu() * y.cpu())
        self.assertNotIn("mcpu::aten::mul.raw", event_names)

        overlap_base = torch.arange(5, dtype=torch.float32, device="mcpu")
        _, event_names = self._timed_event_names(
            lambda: torch.mul(
                overlap_base[:-1], overlap_base[:-1], out=overlap_base[1:]
            )
        )
        self.assertNotIn("mcpu::aten::mul.raw", event_names)

    def test_strided_add_sub_cpu_view_kernel_matches_cpu(self):
        base_x = torch.arange(96, dtype=torch.float32).reshape(4, 3, 8)
        base_y = torch.arange(96, 192, dtype=torch.float32).reshape(4, 3, 8)
        x = base_x.to("mcpu")[:, :, 2:6]
        y = base_y.to("mcpu")[:, :, 2:6]

        (add_res, sub_res), event_names = self._timed_event_names(
            lambda: (torch.add(x, y), torch.sub(y, x))
        )

        self.assertEqual(add_res.cpu(), base_x[:, :, 2:6] + base_y[:, :, 2:6])
        self.assertEqual(sub_res.cpu(), base_y[:, :, 2:6] - base_x[:, :, 2:6])
        self.assertIn("mcpu::aten::add", event_names)
        self.assertIn("mcpu::aten::sub.Tensor", event_names)
        self.assertNotIn("mcpu::aten::add.raw", event_names)
        self.assertNotIn("mcpu::aten::sub.raw", event_names)

    def test_scalar_gt_uses_cpu_view_and_bitwise_not_raw_and_multi_index_put_falls_back(
        self,
    ):
        x = torch.tensor([[0.0, 2.0], [3.0, 4.0]], device="mcpu")
        gt_out = torch.empty_like(x, dtype=torch.bool)
        bits = torch.tensor([[0, 1], [2, 3]], device="mcpu", dtype=torch.int32)
        bitwise_not_out = torch.empty_like(bits)
        target = torch.zeros(3, 3, device="mcpu")
        row = torch.tensor([0, 2], device="mcpu", dtype=torch.long)
        col = torch.tensor([1, 0], device="mcpu", dtype=torch.long)
        vals = torch.tensor([5.0, 7.0], device="mcpu")

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.gt(x, 2.0, out=gt_out)
            torch.bitwise_not(bits, out=bitwise_not_out)
            torch.ops.aten._index_put_impl_(target, [row, col], vals, False, False)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(gt_out.cpu(), torch.gt(x.cpu(), 2.0))
        self.assertEqual(bitwise_not_out.cpu(), torch.bitwise_not(bits.cpu()))
        self.assertEqual(
            target.cpu(),
            torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        )
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::gt.Scalar_out", event_names)
        self.assertNotIn("mcpu::aten::gt.Scalar.raw", event_names)
        self.assertIn("mcpu::aten::bitwise_not.out.raw", event_names)
        self.assertIn("mcpu::aten::_index_put_impl_", event_names)
        self.assertNotIn("mcpu::aten::_index_put_impl_.raw", event_names)

    def test_index_put_dim0_raw_kernel_supports_outer_strides(self):
        target = torch.empty_strided((7, 2, 3, 4), (32, 12, 4, 1), device="mcpu")
        target.zero_()
        index_base = torch.tensor(
            [5, 99, 1, 99, 5], device="mcpu", dtype=torch.long
        )
        index = index_base[::2]
        values_cpu = torch.arange(3 * 2 * 3 * 4, dtype=torch.float32).reshape(
            3, 2, 3, 4
        )
        values = torch.empty_strided((3, 2, 3, 4), (29, 12, 4, 1), device="mcpu")
        values.copy_(values_cpu.to("mcpu"))

        ref = target.cpu()
        ref.index_put_((index.cpu(),), values.cpu(), accumulate=False)

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._index_put_impl_(target, [index], values, False, False)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(target.cpu(), ref)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::_index_put_impl_.raw", event_names)
        self.assertNotIn("mcpu::aten::_index_put_impl_.raw.validate", event_names)

        scalar_target = torch.empty_strided(
            (1, 2, 3, 4), (24, 12, 4, 1), device="mcpu"
        )
        scalar_target.zero_()
        scalar_index = torch.tensor([0], device="mcpu", dtype=torch.long)
        scalar_value = torch.tensor(9.0, device="mcpu")
        scalar_ref = scalar_target.cpu()
        scalar_ref.index_put_(
            (scalar_index.cpu(),), scalar_value.cpu(), accumulate=False
        )

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._index_put_impl_(
                scalar_target, [scalar_index], scalar_value, False, False
            )
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(scalar_target.cpu(), scalar_ref)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::_index_put_impl_.raw", event_names)

        bad_index = torch.tensor(
            [scalar_target.size(0)], device="mcpu", dtype=torch.long
        )
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            torch.ops.aten._index_put_impl_(
                scalar_target, [bad_index], scalar_value, False, False
            )

        int_index_target = torch.empty_strided(
            (7, 2, 3, 4), (32, 12, 4, 1), device="mcpu"
        )
        int_index_target.zero_()
        int_index = torch.tensor([4], device="mcpu", dtype=torch.int32)
        single_row_values = torch.arange(
            2 * 3 * 4, dtype=torch.float32, device="mcpu"
        ).reshape(2, 3, 4)
        int_index_ref = int_index_target.cpu()
        int_index_ref.index_put_(
            (int_index.cpu(),), single_row_values.cpu(), accumulate=False
        )

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._index_put_impl_(
                int_index_target, [int_index], single_row_values, False, False
            )
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(int_index_target.cpu(), int_index_ref)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::_index_put_impl_.raw", event_names)

        bool_mask_target = torch.empty_strided(
            (1, 2, 3, 4), (24, 12, 4, 1), device="mcpu"
        )
        bool_mask_target.zero_()
        bool_mask = torch.tensor([True], device="mcpu", dtype=torch.bool)
        bool_scalar_value = torch.tensor(11.0, device="mcpu")
        bool_mask_ref = bool_mask_target.cpu()
        bool_mask_ref.index_put_(
            (bool_mask.cpu(),), bool_scalar_value.cpu(), accumulate=False
        )

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._index_put_impl_(
                bool_mask_target, [bool_mask], bool_scalar_value, False, False
            )
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(bool_mask_target.cpu(), bool_mask_ref)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::_index_put_impl_.raw", event_names)

    def test_index_put_dim0_raw_kernel_rejects_fallback_required_cases(self):
        target = torch.zeros(3, 2, device="mcpu")
        index = torch.tensor([1], device="mcpu", dtype=torch.long)
        values = torch.ones(1, 2, device="mcpu")
        with self.assertRaisesRegex(IndexError, "too many indices"):
            torch.ops.aten._index_put_impl_(
                target, [index, None, None], values, False, False
            )

        alias_target = torch.tensor(
            [[1, 0], [2, 0], [0, 0]], device="mcpu", dtype=torch.long
        )
        alias_index = alias_target[:, 0]
        alias_values = torch.tensor(
            [[2, 10], [0, 20], [1, 30]], device="mcpu", dtype=torch.long
        )
        alias_ref = alias_target.cpu()
        torch.ops.aten._index_put_impl_(
            alias_ref, [alias_ref[:, 0]], alias_values.cpu(), False, False
        )

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._index_put_impl_(
                alias_target, [alias_index], alias_values, False, False
            )
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(alias_target.cpu(), alias_ref)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::_index_put_impl_", event_names)
        self.assertFalse(
            any(
                name and name.startswith("mcpu::aten::_index_put_impl_.raw")
                for name in event_names
            )
        )

    def test_abs_uses_explicit_mcpu_kernel(self):
        x = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device="mcpu")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "mcpu")

        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "mcpu")

    def test_registered_binary_operations_stay_on_mcpu(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1).to("mcpu")

        z = torch.add(x, y)
        self.assertEqual(z.device.type, "mcpu")

        z = torch.sub(x, y)
        self.assertEqual(z.device.type, "mcpu")

    def test_registered_scalar_operations_stay_on_mcpu(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = x + 1.0
        self.assertEqual(y.device.type, "mcpu")

        y = x - 2.0
        self.assertEqual(y.device.type, "mcpu")


class TestSDPAExtended(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_with_mask(self):
        """Test fused SDP choice with attention mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        attn_mask_privateuse1 = attn_mask.to("mcpu")

        backend = torch._fused_sdp_choice(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1
        )
        self.assertEqual(backend, SDPBackend.OVERRIDEABLE.value)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_with_dropout(self):
        """Test scaled dot product attention with dropout"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )
        self.assertEqual(output.device.type, "mcpu")
        self.assertEqual(output.shape, shape)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_is_causal(self):
        """Test scaled dot product attention with causal mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        self.assertEqual(output.device.type, "mcpu")
        self.assertEqual(output.shape, shape)


class TestCopyFromAndResize(TestCase):
    # def test_copy_from_and_resize(self):
    #     """Test _copy_from_and_resize operation"""
    #     x = torch.randn(3, 4, device="mcpu")
    #     y = torch.empty(2, 2, device="mcpu")
    #     result = torch.ops.aten._copy_from_and_resize(x, y)
    #     self.assertEqual(result.device.type, "mcpu")
    #     self.assertEqual(result.shape, x.shape)

    def test_copy_from_same_device(self):
        """Test _copy_from operation on same device"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=False)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.cpu(), x.cpu())

    def test_copy_from_same_device_fallback_cases(self):
        """Test same-device _copy_from fallback for non-memcpy-compatible copies."""
        src = torch.randn(8, 16, device="mcpu", dtype=torch.float32)
        dst = torch.empty(8, 16, device="mcpu", dtype=torch.bfloat16)
        torch.ops.aten._copy_from(src, dst, non_blocking=False)

        base = torch.arange(1, device="mcpu", dtype=torch.int32)
        expanded = base.expand(16)
        out = torch.empty(16, device="mcpu", dtype=torch.int32)
        torch.ops.aten._copy_from(expanded, out, non_blocking=False)

        torch.mcpu.synchronize()
        self.assertEqual(dst.cpu(), src.cpu().bfloat16())
        self.assertEqual(out.cpu(), expanded.cpu())

    def test_copy_from_cross_device(self):
        """Test _copy_from operation across devices"""
        x = torch.randn(3, 4, device="cpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=False)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.cpu(), x)

    def test_copy_from_host_device_non_blocking_uses_raw_memcpy(self):
        """Test host-device _copy_from avoids copy_ for memcpy-compatible layouts."""
        cpu_src = torch.randn(3, 7, 5)
        mcpu_dst = torch.empty_like(cpu_src, device="mcpu")
        mcpu_src = cpu_src.to("mcpu")
        cpu_dst = torch.empty_like(cpu_src)

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._copy_from(cpu_src, mcpu_dst, non_blocking=True)
            torch.ops.aten._copy_from(mcpu_src, cpu_dst, non_blocking=True)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(mcpu_dst.cpu(), cpu_src)
        self.assertEqual(cpu_dst, cpu_src)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertGreaterEqual(
            event_names.count("mcpu::_copy_from.host_device.memcpy"),
            2,
            event_names,
        )

    def test_copy_from_non_blocking(self):
        """Test _copy_from with non_blocking=True"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=True)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.cpu(), x.cpu())

    def test_reshape_alias(self):
        """Test _reshape_alias operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        new_size = (6, 4)
        new_stride = (4, 1)
        y = torch.ops.aten._reshape_alias(x, new_size, new_stride)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size(new_size))
        self.assertEqual(y.stride(), new_stride)

    def test_set_storage_storage_offset(self):
        """Test set_ operation with storage and storage offset"""
        x = torch.randn(3, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(2, 2, device="mcpu")
        result = torch.ops.aten.set_.source_Storage_storage_offset(
            y, storage, 0, (2, 2), (2, 1)
        )
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, torch.Size([2, 2]))

    def test_set_storage_storage_offset_with_offset(self):
        """Test set_ operation with non-zero storage offset"""
        x = torch.randn(4, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(2, 2, device="mcpu")
        # Use storage offset to skip first 4 elements
        result = torch.ops.aten.set_.source_Storage_storage_offset(
            y, storage, 4, (2, 2), (2, 1)
        )
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, torch.Size([2, 2]))


class TestCustomAutogradFunctions(TestCase):
    def test_custom_autograd_fn_returns_self_basic(self):
        """Test basic usage of custom_autograd_fn_returns_self"""
        x = torch.randn(4, device="mcpu", requires_grad=True)
        y = torch.ops.mcpu.custom_autograd_fn_returns_self(x)

        # Should return the same tensor
        self.assertEqual(x.cpu(), y.cpu())
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.ones_like(x.cpu()) * 0.5))

    def test_custom_autograd_fn_aliasing_basic(self):
        """Test basic usage of custom_autograd_fn_aliasing"""
        x = torch.randn(4, device="mcpu", requires_grad=True)
        y = torch.ops.mcpu.custom_autograd_fn_aliasing(x)

        # Should return a view of the same tensor
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.ones_like(x.cpu()) * 0.5))

    def test_custom_autograd_fn_returns_self_no_grad(self):
        """Test custom_autograd_fn_returns_self without requires_grad"""
        x = torch.randn(4, device="mcpu", requires_grad=False)
        y = torch.ops.mcpu.custom_autograd_fn_returns_self(x)
        self.assertEqual(x.cpu(), y.cpu())
        self.assertFalse(y.requires_grad)

    def test_custom_autograd_fn_aliasing_no_grad(self):
        """Test custom_autograd_fn_aliasing without requires_grad"""
        x = torch.randn(4, device="mcpu", requires_grad=False)
        y = torch.ops.mcpu.custom_autograd_fn_aliasing(x)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(y.requires_grad)


class TestMM(TestCase):
    def test_mm_basic(self):
        """Test basic mm operation"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.mm(a, b)
        self.assertEqual(c.device.type, "mcpu")
        self.assertEqual(c.shape, torch.Size([3, 5]))
        self.assertEqual(c.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_mm_out(self):
        """Test mm with output tensor"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        out = torch.empty(3, 5, device="mcpu")
        torch.mm(a, b, out=out)
        self.assertEqual(out.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_mm_out_wrong_size(self):
        """Test mm.out rejects wrong output size"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        bad_out = torch.empty(1, 1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::mm.out"):
            torch.mm(a, b, out=bad_out)

    def test_mm_square(self):
        """Test mm with square matrices"""
        a = torch.randn(4, 4, device="mcpu")
        b = torch.randn(4, 4, device="mcpu")
        c = torch.mm(a, b)
        self.assertEqual(c.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_addmm_basic(self):
        """Test basic addmm operation"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b)
        self.assertEqual(c.device.type, "mcpu")
        self.assertEqual(c.shape, torch.Size([3, 5]))
        self.assertEqual(c.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))

    def test_addmm_alpha_beta(self):
        """Test addmm with non-default alpha and beta"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b, beta=0.5, alpha=2.0)
        expected = torch.addmm(bias.cpu(), a.cpu(), b.cpu(), beta=0.5, alpha=2.0)
        self.assertEqual(c.cpu(), expected)

    def test_addmm_out(self):
        """Test addmm with output tensor"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        out = torch.empty(3, 5, device="mcpu")
        torch.addmm(bias, a, b, out=out)
        self.assertEqual(out.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))

    def test_addmm_broadcast_bias(self):
        """Test addmm with broadcastable bias"""
        bias = torch.randn(5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b)
        self.assertEqual(c.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))


if __name__ == "__main__":
    run_tests()
