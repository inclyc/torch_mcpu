import os
from dataclasses import dataclass
from typing import Any

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
    register_interface_for_device,
)
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import register_backend_for_device

import torch_mcpu._C  # type: ignore[misc]
import torch_mcpu.openreg
from torch_mcpu.compile_flags import get_compile_definitions
from torch_mcpu.paths import get_include, get_library_dir
from torch_mcpu.inductor.extension_codegen_backend import (
    McpuCppWrapperCodegen,
    McpuDisableComputeFusionPass,
    McpuScheduling,
    McpuWrapperCodegen,
)
from torch_mcpu.inductor.torch_xcpu_fusion import (
    McpuTorchXcpuFusionPass,
    append_post_grad_pass,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _setup_inductor_cpp_device_build_flags() -> None:
    import torch._inductor.cpp_builder as cpp_builder

    if getattr(cpp_builder, "_torch_mcpu_device_build_flags_registered", False):
        return

    original = cpp_builder.get_cpp_torch_device_options

    def get_cpp_torch_device_options_with_mcpu(*args, **kwargs):
        device_type = kwargs.get("device_type")
        if device_type is None and args:
            device_type = args[0]

        result = original(*args, **kwargs)
        if device_type not in {"cpu", "mcpu", "privateuseone"}:
            return result

        (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = result

        definitions = [
            *definitions,
            *(
                f"{name}={value}"
                for name, value in get_compile_definitions().items()
            ),
        ]
        lib_dir = get_library_dir()
        include_dirs = [*include_dirs, get_include()]
        ldflags = [*ldflags, f"Wl,-rpath,{lib_dir}"]
        libraries_dirs = [*libraries_dirs, lib_dir]
        libraries = [*libraries, "torch_mcpu", "openreg"]

        return (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        )

    cpp_builder.get_cpp_torch_device_options = get_cpp_torch_device_options_with_mcpu
    cpp_builder._torch_mcpu_device_build_flags_registered = True


def _register_mcpu_aoti_fallback_shims() -> None:
    from torchgen.aoti.fallback_ops import inductor_fallback_ops

    inductor_fallback_ops.setdefault("aten.cat.default", {})
    inductor_fallback_ops.setdefault("aten.constant_pad_nd.default", {})
    inductor_fallback_ops.setdefault("aten.embedding.default", {})
    inductor_fallback_ops.setdefault("aten.index.Tensor", {})
    inductor_fallback_ops.setdefault("aten.index_select.default", {})
    inductor_fallback_ops.setdefault("aten.index_select.out", {})
    inductor_fallback_ops.setdefault("aten.sigmoid.default", {})


def _register_mcpu_inductor_fallback_lowerings() -> None:
    import torch._inductor.lowering as lowering

    fallback_ops = (
        torch.ops.aten.embedding.default,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_select.default,
        torch.ops.aten.index_select.out,
    )
    for op in fallback_ops:
        lowering.add_needs_realized_inputs(op)
        lowering.lowerings[op] = lowering.fallback_handler(
            op, add_to_fallback_set=False
        )


def _register_mcpu_inductor_metadata_lowerings() -> None:
    import torch._inductor.lowering as lowering
    from torch._inductor.virtualized import V

    op = torch.ops.aten.sym_storage_offset.default
    if op in lowering.lowerings:
        return

    # Like sym_size/sym_stride, this must use view metadata. Falling back
    # through a realized buffer loses the offset of an intermediate view.
    @lowering.register_lowering(op)
    def sym_storage_offset(input):
        value = V.graph.current_node.meta["val"]
        return value.node.expr if isinstance(value, torch.SymInt) else int(value)


@dataclass(frozen=True)
class McpuDeviceProperties:
    name: str
    major: int
    minor: int
    multi_processor_count: int
    total_memory: int


def _normalize_mcpu_device_index(device: torch.types.Device = None) -> int:
    if device is None:
        return McpuInterface.Worker.current_device()
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type not in {"mcpu", "privateuseone"}:
            raise ValueError(f"Expected an mcpu device, got {device}")
        if device.index is None:
            return McpuInterface.Worker.current_device()
        return device.index
    return int(device)


class McpuInterface(DeviceInterface):
    """Dynamo device interface for the mcpu PrivateUse1 backend."""

    device = torch_mcpu.openreg.device
    Event = torch_mcpu.openreg.Event
    Stream = torch_mcpu.openreg.Stream

    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["mcpu"] = device

        @staticmethod
        def current_device() -> int:
            if "mcpu" in caching_worker_current_devices:
                return caching_worker_current_devices["mcpu"]
            return torch_mcpu.openreg.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            device_index = _normalize_mcpu_device_index(device)
            if "mcpu" not in caching_worker_device_properties:
                caching_worker_device_properties["mcpu"] = [
                    McpuDeviceProperties(
                        name=f"mcpu:{idx}",
                        major=0,
                        minor=0,
                        multi_processor_count=1,
                        total_memory=0,
                    )
                    for idx in range(torch_mcpu.openreg.device_count())
                ]
            return caching_worker_device_properties["mcpu"][device_index]

    @staticmethod
    def current_device() -> int:
        return torch_mcpu.openreg.current_device()

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        torch_mcpu.openreg.set_device(device)

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        return torch_mcpu._C._exchangeDevice(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        return torch_mcpu._C._exchangeDevice(device)

    @staticmethod
    def device_count() -> int:
        return torch_mcpu.openreg.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch_mcpu.openreg.is_available()

    @staticmethod
    def stream(stream: torch.Stream):
        return torch_mcpu.openreg.stream(stream)

    @staticmethod
    def current_stream(device: torch.types.Device = None) -> torch.Stream:
        return torch_mcpu.openreg.current_stream(device)

    @staticmethod
    def default_stream(device: torch.types.Device = None) -> torch.Stream:
        return torch_mcpu.openreg.default_stream(device)

    @staticmethod
    def set_stream(stream: torch.Stream) -> None:
        torch_mcpu.openreg.set_stream(stream)

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
        stream = torch.Stream(
            stream_id=stream_id,
            device_index=device_index,
            device_type=device_type,
        )
        torch_mcpu.openreg.set_stream(stream)

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        return torch_mcpu.openreg.current_stream(device_idx).stream_id

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        torch_mcpu.openreg.synchronize(device)

    get_device_properties = staticmethod(Worker.get_device_properties)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        return "mcpu"

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return torch.bfloat16 in torch_mcpu.openreg.get_amp_supported_dtype()

    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.memory_allocated(device)

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        return False

    @staticmethod
    def memory_reserved(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.memory_reserved(device)

    @staticmethod
    def max_memory_allocated(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.max_memory_allocated(device)

    @staticmethod
    def max_memory_reserved(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.max_memory_reserved(device)


def setup_mcpu_compile() -> None:
    """Register mcpu with Dynamo and Inductor."""
    _setup_inductor_cpp_device_build_flags()
    _register_mcpu_aoti_fallback_shims()
    _register_mcpu_inductor_fallback_lowerings()
    _register_mcpu_inductor_metadata_lowerings()
    # Keep automatic pattern fusion opt-in. Explicit torch_xcpu/vLLM custom
    # operators remain available, but ordinary ATen graphs must not silently
    # turn into asynchronous raw-pointer fused kernels.
    if _env_flag("TORCH_MCPU_ENABLE_TORCH_XCPU_FUSIONS", False):
        inductor_config.post_grad_custom_post_pass = append_post_grad_pass(
            inductor_config.post_grad_custom_post_pass,
            McpuTorchXcpuFusionPass(),
        )
    register_interface_for_device("mcpu", McpuInterface)
    register_interface_for_device("privateuseone", McpuInterface)
    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
        device_custom_pass=McpuDisableComputeFusionPass(),
    )
    register_backend_for_device(
        "privateuseone",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
        device_custom_pass=McpuDisableComputeFusionPass(),
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"
    cpp_utils.DEVICE_TO_ATEN["privateuseone"] = "at::kPrivateUse1"
