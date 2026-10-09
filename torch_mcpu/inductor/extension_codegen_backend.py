"""
Torch Inductor codegen backend for the mcpu device.

Mirrors docs/ref/extension_backends/cpp/extension_codegen_backend.py and
registers mcpu as a CPU-based inductor backend so that torch.compile() can
generate and execute code for mcpu tensors.

Usage::

    from torch_mcpu.inductor.extension_codegen_backend import (
        McpuCppWrapperCodegen,
        McpuScheduling,
        McpuWrapperCodegen,
    )
    from torch._inductor.codegen.common import register_backend_for_device
    from torch._inductor.codegen import cpp_utils

    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"
"""

from pathlib import Path
from textwrap import dedent

import torch

from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.codegen import cpu_device_op_overrides as _cpu_device_op_overrides  # noqa: F401
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    get_hash_for_files,
)
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V
from torch_mcpu.paths import get_include


def _patch_get_current_backend() -> None:
    """Make Inductor classify mcpu as a C++ backend."""
    import sys

    from torch._inductor import config as inductor_config
    from torch._inductor import utils as inductor_utils

    original = inductor_utils.get_current_backend

    def get_current_backend(device_type=None):
        if not device_type:
            try:
                device_type = V.graph.get_current_device_or_throw().type
            except Exception:
                return original(device_type)
        if device_type in {"mcpu", "privateuseone"}:
            return inductor_config.cpu_backend
        return original(device_type)

    inductor_utils.get_current_backend = get_current_backend
    for module in list(sys.modules.values()):
        if getattr(module, "get_current_backend", None) is original:
            module.get_current_backend = get_current_backend


_patch_get_current_backend()


_MCPU_HOST_LOOP_FALLBACK_OPS = frozenset(
    {
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
        torch.ops.aten.fill.Scalar,
        torch.ops.aten.fill.Tensor,
        torch.ops.aten.full.default,
        torch.ops.aten.full_like.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.cat.default,
        # Padding is a data-moving compute op even though its schema is not
        # tagged pointwise/reduction.  Letting Inductor lower it with the CPU
        # C++ scheduler emits a raw ``cpp_fused_constant_pad_nd`` loop over
        # PrivateUse1 storage, bypassing the MCPU stream entirely.
        torch.ops.aten.constant_pad_nd.default,
        torch.ops.aten.zero.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
    }
)

_MCPU_IOTA_FALLBACK_OPS = frozenset(
    {
        torch.ops.prims.iota.default,
    }
)

_MCPU_GATHER_FALLBACK_OPS = frozenset(
    {
        torch.ops.aten.embedding.default,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_select.default,
        torch.ops.aten.index_select.out,
    }
)


def _meta_uses_mcpu(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type in {"mcpu", "privateuseone"}
    if isinstance(value, (list, tuple)):
        return any(_meta_uses_mcpu(item) for item in value)
    if isinstance(value, dict):
        return any(_meta_uses_mcpu(item) for item in value.values())
    return False


def _graph_uses_mcpu(gm) -> bool:
    for node in gm.graph.nodes:
        if _meta_uses_mcpu(node.meta.get("val")) or _meta_uses_mcpu(
            node.meta.get("example_value")
        ):
            return True
    return False


def _is_mcpu_host_loop_op(node) -> bool:
    target = node.target
    return (
        node.op == "call_function"
        and isinstance(target, torch._ops.OpOverload)
        and target.namespace in {"aten", "prims"}
        and (
            torch.Tag.pointwise in target.tags
            or torch.Tag.reduction in target.tags
            or target in _MCPU_HOST_LOOP_FALLBACK_OPS
            or target in _MCPU_IOTA_FALLBACK_OPS
            or target in _MCPU_GATHER_FALLBACK_OPS
        )
        and (
            _meta_uses_mcpu(node.meta.get("val"))
            or _meta_uses_mcpu(node.meta.get("example_value"))
        )
    )


class McpuDisableComputeFusionPass(CustomGraphModulePass):
    """Selectively lower mcpu compute ops through ATen fallback.

    mcpu is used to model an accelerator with device memory that is not directly
    host-accessible. Inductor's generated ``cpp_fused_*`` kernels execute raw
    loops on the host thread over tensor data pointers, so they bypass the
    launch-kernel/page-protection model. Marking pointwise and reduction ATen
    nodes with ``should_fallback`` uses Inductor's selective lowering mechanism
    to keep graph compilation and AOTI wrapper generation while avoiding CPU
    fused-loop codegen for mcpu compute and tensor factory fills. Metadata-only
    view ops such as slice retain their standard ReinterpretView lowering.
    """

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        if not _graph_uses_mcpu(gm):
            return
        for node in gm.graph.nodes:
            if _is_mcpu_host_loop_op(node):
                node.meta["should_fallback"] = True

    def uuid(self):
        return get_hash_for_files((__file__,), extra=self.__class__.__name__)


class McpuDeviceOpOverrides(DeviceOpOverrides):
    """Device-op overrides for mcpu.

    mcpu is a PrivateUse1 accelerator simulation. Generated wrappers should use
    mcpu device/stream APIs instead of inheriting CPU no-op behavior.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            f"""
            def {name}(device_idx):
                import torch
                return torch.mcpu.current_stream(device_idx).stream_id
            """
        )

    def set_device(self, device_idx: int) -> str:
        return f"torch.mcpu.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mcpu.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.mcpu.device({device_idx})"

    def cpp_device_guard(self) -> str:
        return "at::DeviceGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTIMcpuGuard"

    def cpp_stream_guard(self) -> str:
        return "c10::OptionalStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTIMcpuStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "c10::mcpu::getStreamFromExternal"

    def kernel_header(self) -> str:
        return ""

    def kernel_driver(self) -> str:
        return ""

    def cpp_stream_type(self) -> str:
        return "orStream_t"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_mcpu_get_current_stream"

    def cpp_kernel_type(self) -> str:
        return "void*"

    def cpp_device_ptr(self) -> str:
        return "void*"

    def tma_descriptor_helpers(self) -> str:
        return ""

    def cpp_scratch(self, idx: int, workspace, prefix=None):
        return None


class McpuWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        # The base-class create() hard-codes PythonWrapperCodegen(); override it
        # so the correct type is instantiated.
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return wrapper.SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return McpuWrapperCodegen()

    def _generate_kernel_call_helper(
        self, kernel_name, call_args, *, device=None, triton=True, **kwargs
    ):
        device = device or V.graph.get_current_device_or_throw()
        if not triton and device.type in {"mcpu", "privateuseone"}:
            raise NotImplementedError(
                "MCPU generated C++ kernels require cpp_wrapper=True for "
                "stream launch, pointer memory guards and kernel timing. "
                "The Python wrapper supports ATen/custom-op fallback only."
            )
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, triton=triton, **kwargs
        )


class McpuCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def codegen_int_array_var(
        self,
        int_array,
        writeline,
        known_statically=False,
        graph=None,
    ):
        return self._codegen_int_array_var_impl(int_array, writeline, known_statically)

    def _generate_kernel_call_helper(
        self,
        kernel_name,
        call_args,
        *,
        device=None,
        triton=True,
        **kwargs,
    ):
        device = device or V.graph.get_current_device_or_throw()
        if not triton and device.type in {"mcpu", "privateuseone"}:
            # Inductor may introduce ComputedBuffers after the post-grad
            # fallback pass (layout materialization and mutation copy-back are
            # common examples).  Those become host C++ loops over raw device
            # pointers.  Make the storage CPU-accessible for the duration of
            # such a generated kernel, just like native mcpu/torch_xcpu
            # kernels do with KernelPointerMemoryGuard.
            arg_types = kwargs["arg_types"]
            assert arg_types is not None and len(call_args) == len(arg_types)
            captures = []
            kernel_args = []
            pointer_args = []
            for idx, (arg, arg_type) in enumerate(zip(call_args, arg_types)):
                name = f"kernel_arg_{idx}"
                is_pointer = isinstance(arg_type, str) and "*" in arg_type
                value = f"({arg_type})({arg}.data_ptr())" if is_pointer else str(arg)
                captures.append(f"{name} = {value}")
                kernel_args.append(name)
                if is_pointer:
                    pointer_args.append(name)
            # Capture only pointer/scalar values. Allocation ownership stays
            # with the stream-aware allocator, as for native MCPU kernels.
            self.writeline(
                f'MCPU_LAUNCH_TIMED_KERNEL("mcpu::{kernel_name}", '
                f"([{', '.join(captures)}]), {{"
            )
            self.writeline(
                "at::mcpu::KernelPointerMemoryGuard guard({"
                + ", ".join(pointer_args)
                + "});"
            )
            self.writeline(self.wrap_kernel_call(kernel_name, kernel_args))
            self.writeline("});")
            return
        super()._generate_kernel_call_helper(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            **kwargs,
        )

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        # The base-class create() hard-codes CppWrapperCpu(); override it
        # so the correct type (McpuCppWrapperCodegen) is instantiated.
        return McpuCppWrapperCodegen()

    def val_to_arg_str_for_prim_type(self, val, type_) -> str:
        if isinstance(val, torch.device):
            device_type, device_idx = self.codegen_device(val).split(",")
            return (
                "torch::stable::Device("
                f"static_cast<torch::stable::DeviceType>({device_type}),"
                f"{device_idx})"
            )
        return super().val_to_arg_str_for_prim_type(val, type_)

    @staticmethod
    def _get_mcpu_include_path(header_dir: str) -> str:
        include_dir = Path(get_include())
        header = include_dir / header_dir / "mcpu.h"

        if not header.exists():
            raise FileNotFoundError(f"Failed to find header file: {header}")

        return f'#include "{header}"'

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        if device not in {"mcpu", "privateuseone"}:
            return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_jit(device)
        return McpuCppWrapperCodegen._get_mcpu_include_path("cpp_wrapper")

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        if device not in {"mcpu", "privateuseone"}:
            return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_aot(device)
        return McpuCppWrapperCodegen._get_mcpu_include_path("aoti_include")


class McpuScheduling(BaseScheduling):
    """Scheduling for the mcpu device.

    Delegates to CppScheduling since mcpu runs on CPU hardware.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return self._scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2):
        return self._scheduling.can_fuse_horizontal(node1, node2)

    def fuse(self, node1, node2):
        # CppScheduling.fuse() does more than construct a FusedSchedulerNode:
        # for compatible pointwise loops with different dimensionality it
        # recomputes one loop body using the other's ranges.  Delegating the
        # can_fuse checks without also delegating fuse leaves, for example, a
        # (rows, cols) producer fused with a flattened mutation copy-back.
        # CppKernelProxy then sees incompatible iteration groups at codegen.
        return self._scheduling.fuse(node1, node2)

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes=()):
        pass

    def codegen_node(self, node):
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()

    def benchmark_fused_nodes(self, nodes):
        # mcpu does not implement Inductor combo-kernel codegen. If benchmarked,
        # report fusion as unavailable instead of letting the scheduler create a
        # ForeachKernelSchedulerNode that only SIMD/CUDA backends can lower.
        return float("inf"), ""

    def benchmark_combo_kernel(self, node_list):
        # See benchmark_fused_nodes: combo kernels are not a supported mcpu
        # lowering path today.
        return float("inf"), 0.0, []


register_device_op_overrides("mcpu", McpuDeviceOpOverrides())
