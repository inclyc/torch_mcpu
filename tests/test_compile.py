"""
Tests for torch.compile support on the mcpu backend.

Mirrors docs/ref/extension_backends/cpp/test_extension_backend.py:
verifies that registering mcpu with torch._inductor allows torch.compile()
to generate and execute fused C++ kernels for mcpu tensors.
"""

import os
import tempfile
import unittest
from importlib.util import find_spec
from itertools import count
from pathlib import Path
from unittest.mock import patch

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.cpp_builder as cpp_builder
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.common import (
    custom_backend_passes,
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
)
from torch._inductor.utils import (
    run_and_get_code,
    run_and_get_cpp_code,
)

import torch_mcpu  # noqa: F401 – registers the mcpu backend with PyTorch
from torch_mcpu.compile_flags import get_compile_definitions, get_compile_flags
from torch_mcpu.inductor.extension_codegen_backend import (
    McpuCppWrapperCodegen,
    McpuScheduling,
    McpuWrapperCodegen,
)
from torch_mcpu.inductor.torch_xcpu_fusion import McpuTorchXcpuFusionPass
from torch_mcpu.paths import get_include, get_library_dir



def _torch_xcpu_aoti_env():
    spec = find_spec("torch_xcpu")
    if spec is None or not spec.submodule_search_locations:
        raise unittest.SkipTest("torch_xcpu is not available")

    install_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    header = install_dir / "include" / "aoti_torch_xcpu.h"
    so_files = sorted(install_dir.glob("_C*.so"))
    if not header.is_file() or not so_files:
        raise unittest.SkipTest("torch_xcpu AOTI bits are not available")

    so_file = so_files[0]
    return {
        "AOTI_EXTRA_CFLAGS": f"-include {header}",
        "AOTI_EXTRA_LDFLAGS": f"-Wl,-rpath,{so_file.parent} {so_file}",
        "TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES": "torch_xcpu",
    }


class TestMcpuInductorRegistration(unittest.TestCase):
    """Verify that the inductor backend classes are registered correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_scheduling_registered(self):
        self.assertIs(get_scheduling_for_device("mcpu"), McpuScheduling)

    def test_mcpu_uses_cpp_backend(self):
        """Inductor must not classify mcpu as the Triton backend."""
        from torch._inductor.utils import get_current_backend

        with inductor_config.patch(cpu_backend="cpp", cuda_backend="triton"):
            self.assertEqual(get_current_backend("mcpu"), "cpp")
            self.assertEqual(get_current_backend("privateuseone"), "cpp")
            self.assertEqual(get_current_backend("cuda"), "triton")

    def test_python_wrapper_registered(self):
        self.assertIs(get_wrapper_codegen_for_device("mcpu"), McpuWrapperCodegen)

    def test_cpp_wrapper_registered(self):
        self.assertIs(
            get_wrapper_codegen_for_device("mcpu", cpp_wrapper=True),
            McpuCppWrapperCodegen,
        )

    def _assert_cpp_device_build_options_include_mcpu_flags(self, device_type):
        options = cpp_builder.CppTorchDeviceOptions(device_type=device_type)

        self.assertIn(get_include(), options.get_include_dirs())
        self.assertIn(get_library_dir(), options.get_libraries_dirs())
        self.assertIn("torch_mcpu", options.get_libraries())
        self.assertIn("openreg", options.get_libraries())
        for name, value in get_compile_definitions().items():
            self.assertIn(f"{name}={value}", options.get_definitions())

    def test_cpp_device_build_options_include_mcpu_flags(self):
        self._assert_cpp_device_build_options_include_mcpu_flags("mcpu")

    def test_cpu_cpp_device_build_options_include_mcpu_flags(self):
        self._assert_cpp_device_build_options_include_mcpu_flags("cpu")

    def test_cpp_wrapper_include_path_does_not_mutate_aoti_flags(self):
        original_flags = "-include /tmp/external_aoti_header.h"
        codegen = object.__new__(McpuCppWrapperCodegen)

        with patch.dict(
            os.environ, {"AOTI_EXTRA_CFLAGS": original_flags}, clear=False
        ):
            for mode, header_dir in (("jit", "cpp_wrapper"), ("aot", "aoti_include")):
                with self.subTest(mode=mode):
                    include = getattr(codegen, f"get_device_include_path_{mode}")("mcpu")
                    self.assertEqual(os.environ["AOTI_EXTRA_CFLAGS"], original_flags)
                    self.assertIn(
                        str(Path(get_include()) / header_dir / "mcpu.h"), include
                    )

    def test_sync_kernel_launch_definition_is_exported(self):
        definitions = get_compile_definitions()
        self.assertIn("TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH", definitions)
        value = definitions["TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH"]
        self.assertIn(value, {"0", "1"})
        self.assertIn(
            f"-DTORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH={value}",
            get_compile_flags(),
        )

    @staticmethod
    def _make_cpp_codegen_for_int_array_tests():
        codegen = object.__new__(McpuCppWrapperCodegen)
        codegen.int_array_id = count()
        codegen.declared_int_array_vars = OrderedSet()
        codegen.codegen_int_array_var_cache = {}
        return codegen

    def test_cpp_wrapper_does_not_cache_dynamic_int_arrays(self):
        codegen = self._make_cpp_codegen_for_int_array_tests()
        lines = []

        first = codegen.codegen_int_array_var(
            "{s0, 2L, 128L}", lines.append, known_statically=False
        )
        second = codegen.codegen_int_array_var(
            "{s0, 2L, 128L}", lines.append, known_statically=False
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len(lines), 2)

    def test_cpp_wrapper_does_not_cache_static_int_arrays(self):
        codegen = self._make_cpp_codegen_for_int_array_tests()
        lines = []

        first = codegen.codegen_int_array_var(
            "{1L, 2L, 128L}", lines.append, known_statically=True
        )
        second = codegen.codegen_int_array_var(
            "{1L, 2L, 128L}", lines.append, known_statically=True
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len(lines), 2)


class TestMcpuCompile(unittest.TestCase):
    """Verify that torch.compile() produces correct results for mcpu tensors."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_generated_cpp_kernel_uses_stream_pointer_guard_and_timing(self):
        def fn(x):
            x[:, :64].sin_()
            return x.cos()

        # Force residual generated loops; the normal device pass intentionally
        # falls back most arithmetic before this codegen path is reached.
        with patch.dict(custom_backend_passes, {"mcpu": None, "privateuseone": None}):
            with inductor_config.patch(cpp_wrapper=True, fallback_by_default=False):
                compiled = torch.compile(fn, fullgraph=True, dynamic=True)
                streams = [torch.Stream(device="mcpu") for _ in range(2)]
                torch.mcpu.reset_kernel_timing()
                torch.mcpu.set_kernel_timing_enabled(True)
                try:
                    code_text = ""
                    for index, rows in enumerate((7, 3, 11)):
                        cpu = torch.randn(rows, 128)
                        expected = fn(cpu.clone() + 2)
                        with torch.mcpu.stream(streams[index % 2]):
                            x = cpu.to("mcpu")
                            x.add_(2)
                            result, code = run_and_get_code(compiled, x)
                            actual = result.cpu()
                        torch.testing.assert_close(actual, expected)
                        code_text += "\n".join(code)
                    torch.mcpu.synchronize()
                    events = [
                        event
                        for thread in torch.mcpu.get_kernel_timing()
                        for event in thread["events"]
                        if event["name"].startswith("mcpu::cpp_fused")
                    ]
                    self.assertGreaterEqual(len(events), 3)
                    self.assertTrue(
                        all(e["end_time"] > e["begin_time"] for e in events)
                    )
                    self.assertIn("MCPU_LAUNCH_TIMED_KERNEL", code_text)
                    self.assertIn("KernelPointerMemoryGuard", code_text)
                    self.assertNotIn("KernelAllMemoryGuard", code_text)
                finally:
                    torch.mcpu.synchronize()
                    torch.mcpu.set_kernel_timing_enabled(False)

    def test_python_wrapper_rejects_generated_host_loops(self):
        with patch.dict(custom_backend_passes, {"mcpu": None, "privateuseone": None}):
            with inductor_config.patch(cpp_wrapper=False, fallback_by_default=False):
                fn = torch.compile(lambda x: x.sin() + 1, fullgraph=True)
                with self.assertRaisesRegex(Exception, "require cpp_wrapper=True"):
                    fn(torch.ones(2, 64, device="mcpu"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tensors(self, rows=2, cols=16):
        device = torch.device("mcpu")
        x = torch.empty(rows, cols, device=device).fill_(1)
        y = torch.empty(rows, cols, device=device).fill_(2)
        z = torch.empty(rows, cols, device=device).fill_(3)
        ref = torch.empty(rows, cols).fill_(5)  # 1*2 + 3 = 5
        return x, y, z, ref

    @staticmethod
    def fn(a, b, c):
        return a * b + c

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_compile_python_wrapper(self):
        """Basic torch.compile with Python wrapper (cpp_wrapper=False)."""
        x, y, z, ref = self._make_tensors()
        with inductor_config.patch({
            "cpp_wrapper": False,
            "fallback_by_default": True,
        }):
            opt_fn = torch.compile(self.fn)
            res, code = run_and_get_code(opt_fn, x, y, z)
        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_python_wrapper_keeps_mcpu_arange_as_iota_fallback(self):
        """The MoE expert_map path creates mcpu int32 arange inside compile."""

        def make_expert_map(x):
            return torch.arange(x.shape[0], device=x.device, dtype=torch.int32)

        x = torch.empty(256, device="mcpu")
        expected = torch.arange(256, dtype=torch.int32)

        with inductor_config.patch({
            "cpp_wrapper": False,
            "fallback_by_default": True,
        }):
            opt_fn = torch.compile(make_expert_map)
            res, code = run_and_get_code(opt_fn, x)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertEqual(res.dtype, torch.int32)
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertIn("torch.ops.prims.iota.default", code_text)

    def test_codegen_supports_mcpu_arange_iota_pointwise(self):
        """Regression for Inductor ComputedBuffer(Pointwise iota) on mcpu."""

        def make_expert_map(x):
            return torch.arange(x.shape[0], device=x.device, dtype=torch.int32)

        x = torch.empty(256, device="mcpu")
        expected = torch.arange(256, dtype=torch.int32)

        with inductor_config.patch({
            "cpp_wrapper": False,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(make_expert_map)
            res, code = run_and_get_code(opt_fn, x)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertEqual(res.dtype, torch.int32)
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_codegen_keeps_mcpu_cat_as_aten_fallback(self):
        """Regression for MTP hidden/input embedding concat compilation."""

        def concat_hidden_inputs(hidden_states, inputs_embeds):
            return torch.cat([inputs_embeds, hidden_states], dim=-1)

        hidden_states = torch.empty(
            3, 1024, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)
        inputs_embeds = torch.empty(
            3, 1024, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        expected = concat_hidden_inputs(hidden_states, inputs_embeds).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": False,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(concat_hidden_inputs)
            res, code = run_and_get_code(opt_fn, hidden_states, inputs_embeds)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_codegen_keeps_mcpu_slice_as_aten_fallback(self):
        """A view slice must not become a host C++ copy over mcpu memory."""

        def slice_hidden_states(hidden_states):
            return hidden_states[: hidden_states.shape[0], :]

        hidden_states = torch.empty(
            32, 2048, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)
        expected = slice_hidden_states(hidden_states).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": False,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(slice_hidden_states)
            res, code = run_and_get_code(opt_fn, hidden_states)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_cpp_wrapper_lowers_mcpu_slice_to_metadata_view(self):
        """AOTI C++ wrapper must keep slice as a metadata-only view."""

        def slice_hidden_states(hidden_states):
            return hidden_states[:, 3:11]

        hidden_states = torch.arange(
            64, device="mcpu", dtype=torch.float32
        ).reshape(4, 16)
        expected = slice_hidden_states(hidden_states).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(slice_hidden_states)
            res, code = run_and_get_cpp_code(opt_fn, hidden_states)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)
        self.assertNotIn("aoti_torch_mcpu_slice_Tensor", code_text)
        self.assertIn("reinterpret_tensor_wrapper", code_text)

    def test_cpp_wrapper_supports_mcpu_cat_fallback(self):
        """AOTI C++ wrapper needs an mcpu cat shim for MTP compilation."""

        def concat_hidden_inputs(hidden_states, inputs_embeds):
            return torch.cat([inputs_embeds, hidden_states], dim=-1)

        hidden_states = torch.empty(
            3, 1024, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)
        inputs_embeds = torch.empty(
            3, 1024, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        expected = concat_hidden_inputs(hidden_states, inputs_embeds).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(concat_hidden_inputs)
            res, code = run_and_get_cpp_code(opt_fn, hidden_states, inputs_embeds)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)
        self.assertIn("aoti_torch_mcpu_cat", code_text)

    def test_cpp_wrapper_supports_mcpu_embedding_fallback(self):
        """AOTI C++ wrapper needs an mcpu embedding shim for MTP compilation."""

        def embed(weight, indices):
            return torch.nn.functional.embedding(indices, weight)

        weight = torch.arange(
            20, device="mcpu", dtype=torch.float32
        ).reshape(5, 4)
        indices = torch.tensor(
            [[0, 2], [4, 1]], device="mcpu", dtype=torch.int64
        )
        expected = torch.nn.functional.embedding(
            indices.to("cpu"), weight.to("cpu")
        )

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": False,
        }):
            opt_fn = torch.compile(embed)
            res, code = run_and_get_cpp_code(opt_fn, weight, indices)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)
        self.assertIn("aoti_torch_mcpu_embedding", code_text)

    def test_compile_cpp_wrapper(self):
        """torch.compile with cpp_wrapper=True (AOT C++ code generation)."""
        x, y, z, ref = self._make_tensors()
        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": True,
        }):
            opt_fn = torch.compile(self.fn)
            res, code = run_and_get_cpp_code(opt_fn, x, y, z)
            # res = opt_fn(x, y, z)
        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_cpp_wrapper_supports_mcpu_arange_iota_fallback(self):
        """Regression for Qwen3.6 MoE compile expert_map creation."""

        def make_expert_map(x):
            return torch.arange(x.shape[0], device=x.device, dtype=torch.int32)

        x = torch.empty(256, device="mcpu")
        expected = torch.arange(256, dtype=torch.int32)

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": True,
        }):
            opt_fn = torch.compile(make_expert_map)
            res, code = run_and_get_cpp_code(opt_fn, x)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertEqual(res.dtype, torch.int32)
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)

    def test_cpp_wrapper_uses_mcpu_aten_fallback_not_cpp_fused_loop(self):
        """mcpu cpp_wrapper must not emit raw Inductor C++ fused loops."""

        def pointwise(a, b, c):
            return a[:, None].sigmoid() * b + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": True,
            "post_grad_custom_post_pass": None,
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)
        self.assertIn("aoti_torch_mcpu_mul_Tensor", code_text)
        self.assertIn("aoti_torch_mcpu_add_Tensor", code_text)

    def test_cpp_wrapper_keeps_constant_pad_nd_on_mcpu_stream(self):
        """Padding must dispatch to MCPU instead of running a raw host loop."""

        def pad_rows(x):
            return torch.nn.functional.pad(x, (0, 0, 0, 1))

        x = torch.arange(24, device="cpu", dtype=torch.bfloat16).reshape(3, 8)
        x = x.to("mcpu")
        expected = pad_rows(x.to("cpu"))

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": False,
            "post_grad_custom_post_pass": None,
        }):
            opt_fn = torch.compile(pad_rows, fullgraph=True)
            res, code = run_and_get_cpp_code(opt_fn, x)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.equal(expected, res.to("cpu")))
        self.assertNotIn("cpp_fused_constant_pad_nd", code_text)
        self.assertIn("constant_pad_nd", code_text)

    def test_cpp_wrapper_fuses_sigmoid_and_add_with_torch_xcpu(self):
        """mcpu post-grad pass replaces add/mul/sigmoid with torch_xcpu op."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(a, b, c):
            return b * a[:, None].sigmoid() + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused_add_mul_sigmoid", code_text)
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_cpp_wrapper_fuses_sigmoid_and_mul_with_torch_xcpu(self):
        """mcpu post-grad pass replaces mul/sigmoid with torch_xcpu op."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(a, b):
            return b * a.sigmoid()

        a = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        ref = pointwise(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused_mul_sigmoid", code_text)
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_cpp_wrapper_direct_dispatch_fusions_with_dynamic_shapes(self):
        """Cover the vLLM path: dynamic dims, direct dispatch, AOTI C++ wrapper."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2_2d, input1_2d, input3_2d, input2_3d, input1_3d):
            add_out = input1_2d * input2_2d.sigmoid() + input3_2d
            mul_out = input1_3d * input2_3d.sigmoid()
            return add_out, mul_out

        input2_2d = torch.empty(
            4, 1, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input1_2d = torch.empty(
            4, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)
        input3_2d = torch.empty(
            4, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(3)
        input2_3d = torch.empty(
            4, 2, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input1_3d = torch.empty(
            4, 2, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)

        ref_add, ref_mul = pointwise(
            input2_2d, input1_2d, input3_2d, input2_3d, input1_3d
        )
        torch._dynamo.mark_dynamic(input2_2d, 0)
        torch._dynamo.mark_dynamic(input1_2d, 0)
        torch._dynamo.mark_dynamic(input3_2d, 0)
        torch._dynamo.mark_dynamic(input2_3d, 0)
        torch._dynamo.mark_dynamic(input1_3d, 0)

        with (
            patch.dict(os.environ, _torch_xcpu_aoti_env(), clear=False),
            inductor_config.patch({"cpp_wrapper": True}),
        ):
            opt_fn = torch.compile(pointwise, dynamic=True)
            res, code = run_and_get_cpp_code(
                opt_fn, input2_2d, input1_2d, input3_2d, input2_3d, input1_3d
            )

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertTrue(torch.allclose(ref_add.to("cpu"), res[0].to("cpu")))
        self.assertTrue(torch.allclose(ref_mul.to("cpu"), res[1].to("cpu")))
        self.assertNotIn("cpp_fused_add_mul_sigmoid", code_text)
        self.assertNotIn("cpp_fused_mul_sigmoid", code_text)
        self.assertIn(
            "aoti_torch_mcpu_fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )
        self.assertIn(
            "aoti_torch_mcpu_fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_aot_compile_direct_dispatch_fusion(self):
        """Cover vLLM's Dynamo AOT artifact path with direct dispatch."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        args = (
            torch.full((4, 1), 1, device="mcpu", dtype=torch.bfloat16),
            torch.full((4, 8), 2, device="mcpu", dtype=torch.bfloat16),
            torch.full((4, 8), 3, device="mcpu", dtype=torch.bfloat16),
        )
        ref = pointwise(*args).to("cpu")

        with (
            tempfile.TemporaryDirectory(prefix="mcpu-aot-") as cache_dir,
            patch.dict(
                os.environ,
                {
                    **_torch_xcpu_aoti_env(),
                    "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                },
                clear=False,
            ),
            torch._dynamo.config.patch(enable_aot_compile=True),
            inductor_config.patch({"cpp_wrapper": True}),
        ):
            compiled = torch.compile(pointwise, fullgraph=True, dynamic=False)
            aot_fn, code = run_and_get_cpp_code(compiled.aot_compile, (args, {}))
            res = aot_fn(*args)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertIn(
            "aoti_torch_mcpu_fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )
        self.assertNotIn(
            'aoti_torch_call_dispatcher("torch_xcpu::'
            'fused_sigmoid_mul_add_3d_lastdim_bf16"',
            code_text,
        )

    def test_torch_xcpu_fusion_fuses_3d_broadcast_add(self):
        """Cover MoE-style [tokens, 1, hidden] * [tokens, 1, 1] + residual."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2 = torch.empty(4, 1, 1, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input1 = torch.empty(4, 1, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        input3 = torch.empty(4, 1, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_follows_pytorch_2d_broadcast_for_3d_add(self):
        """A [seq, hidden] gate broadcasts to [1, seq, hidden], not [seq, 1, hidden]."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2_cpu = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 16
        input2 = input2_cpu.to(torch.bfloat16).to("mcpu")
        input1 = torch.empty(4, 4, 8, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input3 = torch.empty(4, 4, 8, device="mcpu", dtype=torch.bfloat16).fill_(0)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_follows_pytorch_1d_broadcast_for_3d_add(self):
        """A [hidden] gate stays right-aligned even when batch == hidden."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2_cpu = torch.arange(4, dtype=torch.float32) / 4
        input2 = input2_cpu.to(torch.bfloat16).to("mcpu")
        input1 = torch.empty(4, 2, 4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input3 = torch.empty(4, 2, 4, device="mcpu", dtype=torch.bfloat16).fill_(0)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_fuses_3d_lastdim_contiguous_slice_mul(self):
        """Cover attention-style gate slices whose last dimension is contiguous."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1):
            return input1 * input2.sigmoid()

        input1 = torch.empty(4, 8, 256, device="mcpu", dtype=torch.bfloat16).fill_(2)
        input2_base = torch.empty(
            4, 8, 512, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input2 = input2_base[:, :, 256:]
        ref = pointwise(input2, input1).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_unsupported_stride(self):
        """The custom pass only selects kernels matching their layout names."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def noncontig_mul(a, b):
            return b * a.sigmoid()

        a_base = torch.empty(2, 3, 16, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b_base = torch.empty(2, 3, 16, device="mcpu", dtype=torch.bfloat16).fill_(2)
        a = a_base[:, :, ::2]
        b = b_base[:, :, ::2]
        ref = noncontig_mul(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(noncontig_mul)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_mixed_dtype_add(self):
        """Do not reinterpret mixed dtype tensors as a single fused-op dtype."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def mixed_dtype_add(a, b, c):
            return b * a[:, None].sigmoid() + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.float32).fill_(3)
        ref = mixed_dtype_add(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(mixed_dtype_add)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_mixed_dtype_mul(self):
        """Do not fuse sigmoid/mul when the two operands have different dtypes."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def mixed_dtype_mul(a, b):
            return b * a.sigmoid()

        a = torch.empty(2, 3, 8, device="mcpu", dtype=torch.float32).fill_(1)
        b = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        ref = mixed_dtype_mul(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(mixed_dtype_mul)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim",
            code_text,
        )

    def test_compile_repeated_calls(self):
        """Compiled function produces consistent results across multiple calls."""
        x, y, z, ref = self._make_tensors()
        opt_fn = torch.compile(self.fn)
        for _ in range(3):
            res = opt_fn(x, y, z)
            self.assertTrue(torch.allclose(ref, res.to("cpu")))

    def test_compile_larger_tensor(self):
        """Compiled function works on larger tensors."""
        x, y, z, ref = self._make_tensors(rows=64, cols=128)
        opt_fn = torch.compile(self.fn)
        res = opt_fn(x, y, z)
        self.assertTrue(torch.allclose(ref, res.to("cpu")))


if __name__ == "__main__":
    unittest.main()
