# MCPU profiler 现状与 Kineto 适配计划

MCPU 当前通过 PyTorch profiler 的 PrivateUse1 timing stubs 提供算子设备耗时，并在 Chrome trace 导出后补入自行采集的 kernel 事件。此次 PyTorch 2.13 适配的目标是保持这条 fallback 路径可用。**工程化接入 PyTorch 的 Kineto 接口仍是重要 TODO，本次兼容修复不代表原生 Kineto 接入已经完成。**

本文面向维护 profiler、runtime 和算子启动接口的开发者。PyTorch 接口行为以相邻 `../pytorch` 仓库的 `2.13.0a0` 源码为基准（commit `ee7cfe81a78`），升级时需要重新核对，不能将开发版本的内部接口视为稳定 ABI。

## 当前数据链路

| 层次 | 实现入口 | 职责 |
| --- | --- | --- |
| Python 兼容层 | [torch_mcpu/profiler.py](../torch_mcpu/profiler.py) | 选择 fallback，管理 kernel timing 开关与重置，包装 trace 导出 |
| PrivateUse1 timing stubs | [csrc/profiler/stubs/openreg.cpp](../csrc/profiler/stubs/openreg.cpp) | 通过 `registerPrivateUse1Methods` 注册 `record`、`elapsed`、`synchronize` 等方法 |
| Kernel 启动与计时 | [McpuKernelLaunch.h](../include/runtime/McpuKernelLaunch.h)、[McpuKernelTiming.h](../include/runtime/McpuKernelTiming.h)、[McpuKernelTiming.cpp](../csrc/runtime/McpuKernelTiming.cpp) | 提交任务时预留 event slot，在 worker 执行期间记录起止时间 |
| Python timing API | [torch_mcpu/openreg/__init__.py](../torch_mcpu/openreg/__init__.py)、[Module.cpp](../torch_mcpu/csrc/Module.cpp) | 暴露开关、重置和事件快照；初始化 backend 时安装 profiler 补丁 |

### 算子耗时统计

`McpuMethods::record()` 保存当前 stream 和全局 kernel timing event 数量。`elapsed()` 同步结束标记对应的 stream，然后累加两个 event 下标之间已经完成的 kernel 耗时，转换为微秒并取整。PyTorch fallback 将这些数据用于算子设备耗时统计，例如 `key_averages()` 中的 `self_device_time_total`。

这里得到的是被 timing scope 覆盖的执行时间之和。它不等于 host 算子的执行时间，也不等于从提交到完成的墙钟时间；跨 stream 并行任务的耗时相加可能超过实际经过时间。由于下标区间来自全局 buffer，当前实现没有按 stream 或 correlation ID 精确归属事件的能力。

### Kernel 时间线

算子需要使用 `launch_timed_kernel_on_stream` 等 timed launch 接口，并在 worker 中使用 `MCPU_KERNEL_TIMING_SCOPE_EVENT` 等 scope，才会产生相应的执行区间。普通 launch 不会自动获得 kernel timing。原始事件包含名称、stream 标识和纳秒单位的起止时间；`torch.mcpu.get_kernel_timing()` 返回这些事件的快照。

`torch.profiler.profile.export_chrome_trace()` 的包装函数先调用 PyTorch 原始导出，再补入 `cat="mcpu_kernel"`、`pid="MCPU"` 的事件，按 `tid="mcpu_stream_<stream>"` 显示不同 stream。当前 stream 标识来自 runtime stream 指针转换后的整数，不是跨进程或跨运行稳定的编号。补入过程支持普通 JSON 和 `.gz` 文件。

补入事件的时长使用 `(end_time - begin_time) / 1000`，但起点采用近似对齐：将最早的 kernel 起点放到导出文件中首个匹配的 `aten::` 或 `mcpu::` CPU 算子结束处，再保留 kernel 之间的相对时间差。未找到锚点时使用零。**这不能用于精确推导 host 到 kernel 的启动延迟或因果关系。**

导出后追加的事件只存在于 trace 文件中，不会回填 PyTorch 的 `events()` 或 `key_averages()`。能看到 MCPU stream lane 与能获得算子设备时间是两条独立链路。

## PyTorch 2.13 的兼容修复

### 显式选择 PrivateUse1 fallback

此前兼容层从 `_supported_activities()` 返回值中移除 PrivateUse1，以避免启用 MCPU 尚未实现的原生 tracing。在当前 PyTorch 2.13 源码中，`torch.profiler.profile` 创建底层 autograd profiler 时传入 `use_kineto=True`；当 `use_device="mcpu"` 时，底层仅依据 `use_kineto` 就会选择 `KINETO_PRIVATEUSE1`。此时 supported activities 只决定是否加入 PrivateUse1 activity，不能将状态改为 fallback。

本次修复包装 `torch.autograd.profiler.profile.__init__`：当关键字参数 `use_device` 为 `"mcpu"` 时，将 `use_kineto` 改为 `False`，使其选择 `KINETO_PRIVATEUSE1_FALLBACK` 并继续使用已注册的 timing stubs。原有 supported activities 补丁仍然保留。

`KINETO_PRIVATEUSE1_FALLBACK` 仍属于 PyTorch profiler 的兼容状态；这里的 `use_kineto=False` 是该版本的状态选择手段，不能理解为完全移除了 Kineto 相关基础设施。Fallback 要求 CPU activity，因此使用时应显式指定 CPU 和 PrivateUse1。

### 转发 trace 导出的新增参数

当前 PyTorch 的 `tensorboard_trace_handler` 会调用：

```python
prof.export_chrome_trace(path, use_python_export=use_python_export)
```

原来的包装函数只接收 `self, path`，无法接收这一关键字参数。本次修改为接收并转发 `*args, **kwargs`，随后继续执行 MCPU trace 补入逻辑。参数可转发不代表所有导出格式均已完成兼容验证；当前新增回归用例显式覆盖的是 `use_python_export=False`。

### 回归测试的变化

[tests/test_profiler.py](../tests/test_profiler.py) 中本次调整的测试分别验证：

- `test_profiler_key_averages` 使用 `torch.profiler.profile` 的 CPU 与 PrivateUse1 组合入口，并断言汇总的 `self_device_time_total` 大于零，防止只采到 CPU 事件也通过测试。
- `test_profiler_export_chrome_trace_multiple_mcpu_streams` 显式传入 `use_python_export=False`，并继续检查多个 MCPU stream 的事件。

这些断言覆盖此次兼容修复，尚不能证明原生 Kineto 接入、设备耗时归属或完整时间轴关联正确。

## 使用方式

在已经安装匹配版本的 PyTorch 和 torch_mcpu 的环境中，可以运行：

```python
import torch
import torch_mcpu  # 注册 MCPU backend 并安装 profiler 兼容层

x = torch.ones(1024, device="mcpu")
torch.mcpu.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ],
    record_shapes=True,
) as prof:
    with torch.profiler.record_function("mcpu_add"):
        y = x + x
    torch.mcpu.synchronize()

print(prof.key_averages().table(sort_by="self_device_time_total"))
prof.export_chrome_trace("mcpu_trace.json", use_python_export=False)
```

应在下一个采集窗口开始或手动重置 timing buffer 之前导出。当前导出读取的是全局 kernel timing 快照，profiler 对象没有独立保存这部分事件。

仓库还提供了包含 schedule 和 trace handler 的示例：

```bash
python scripts/profile_mcpu_tensor_ops.py \
  --activities CPU PrivateUse1 \
  --torch-profiler-dir /tmp/mcpu_profile
```

检查 trace 中的 `mcpu_kernel` 执行事件及 stream lane；检查设备耗时时，使用 `self_device_time_total` 等通用字段。脚本中的 `dump_time_totals()` 仍尝试读取旧式后端专用字段，因此未打印设备时间汇总不等于没有设备耗时。

`TORCH_MCPU_PATCH_PROFILER_TRACE=0` 可以关闭 Python 层自动 kernel timing 窗口管理和 trace 补入，便于定位后处理问题。它不会撤销 fallback 选择补丁，也不会阻止 C++ timing stubs 开启计时，因此不是 profiler 的总开关。

## 当前限制

- **采集窗口**：高层 `start()` 在启动前同步、重置并开启 timing，`stop()` 在退出时同步并关闭。autograd context 也有对应包装。当前没有对 schedule 的每个 wait、warmup、active、repeat 转换分别管理 kernel buffer，多轮 trace 可能带入其他阶段的事件，需要逐窗口验证。
- **并发与容量**：虽然数据结构名为 `ThreadBuffer`，实际只有一个全局 `g_worker_buffer`，计数使用普通 `count++`。多个 host producer 即使向不同 stream 提交，也不能据此认为 timing 写入安全。容量为 262144 个事件，满后不再分配 slot，尚无 dropped event 计数。重置或读取快照前必须确保相关提交与 worker 写入已停止。
- **事件归属**：stubs 累加全局下标区间，没有记录 host op 到 kernel 的 correlation ID；仅同步结束标记的 stream，也不能保证其他 stream 对应事件已完成。现有多 stream 测试主要验证事件存在和 lane 区分。
- **时钟与导出**：文件后处理依赖 PyTorch trace 的文本布局，并使用 host 锚点近似平移 kernel 时间轴。多设备事件目前也没有独立的 device 字段用于区分。
- **生命周期与诊断**：timing 开关和 buffer 是进程全局状态，嵌套或并发 profiler session 缺乏隔离。部分补丁安装和 trace 补入异常被静默忽略，可能表现为文件正常生成但缺少 MCPU kernel 事件。
- **观测开销**：同步、事件记录、fallback 汇总以及 JSON 或 gzip 后处理都会产生额外开销，不能将 profiler 下的端到端延迟直接当作无 profiler 的性能结果。

## 重要 TODO 工程化接入 Kineto

目标是让 MCPU 的 kernel 活动通过 PyTorch 和 Kineto 的正式采集链路进入 profiler，统一管理采集生命周期、事件关联、时间轴与导出，并逐步收敛 Python monkey patch。现有 fallback 可以作为迁移期间的兼容路径。

### 接口与版本适配

相邻 PyTorch 源码已经提供 `PrivateUse1ProfilerRegistry` 和 `REGISTER_PRIVATEUSE1_PROFILER`，用于注册实现 `libkineto::IActivityProfiler` 的 backend factory。`prepareProfiler()` 在 `KINETO_PRIVATEUSE1` 状态下调用 registry 的 `onKinetoInit()`，将 factory 接入 Kineto。后续实现应首先评估这条扩展路径。

需要核对的源码入口如下，路径相对于本仓库根目录：

- `../pytorch/torch/autograd/profiler.py`：原生与 fallback 状态选择。
- `../pytorch/torch/profiler/profiler.py`：高层 API、schedule 和 trace handler。
- `../pytorch/torch/csrc/profiler/standalone/privateuse1_profiler.h`：factory 注册及 `USE_KINETO` 条件。
- `../pytorch/torch/csrc/autograd/profiler_kineto.cpp`：注册接入与 profiler 生命周期。
- `../pytorch/third_party/kineto/libkineto/include/IActivityProfiler.h`：`IActivityProfiler` 和 `IActivityProfilerSession`，包括配置、启动、停止、处理 trace、设备与资源信息和 correlation ID 接口。

应建立版本适配层，明确支持的 PyTorch/Kineto 组合、构建选项、头文件与符号可见性，并验证 factory 注册时序。通过构建时能力检查和运行时探测选择 native 或 fallback；不应仅依据版本字符串推断能力。对缺失接口和初始化失败给出可诊断信息。

### 事件模型与缓冲区

设计包含 device、stream、kernel 名称、起止时间、correlation ID 和 session 归属的事件模型。将 host op、runtime 提交和 worker 执行的关联信息随任务传递，并映射为 Kineto 支持的 activity、device 和 resource。

明确 MCPU timer 到 Kineto 时间轴的转换，取代“首个 host op 结束处”锚点。需要分别验证单位、时钟来源和跨线程时间戳的一致性，使启动延迟及并发关系可以被可靠解释。

结合当前 stream 队列的并发约束，优先评估 per-stream timing buffer，定义 slot 发布、读取和回收协议，消除全局非原子计数的竞争。为容量上限、溢出、截断和 dropped events 提供可见的统计信息，并明确 kernel 名称和事件内存的生命周期。

### 采集生命周期

以 Kineto session 为边界管理 configure、warmup、start、stop、flush 和快照所有权，明确在途任务如何归入窗口。覆盖 schedule 多周期、异常退出、重复启停，以及多线程、多 stream、多设备场景。对于不支持的重入或并发 session 应明确报错。

停止采集后，结果应由对应 session 持有，保证延迟导出和重复读取不会受到下一轮 reset 的影响。尽量减少全设备同步，并衡量计时关闭和开启时的热路径成本。

### 迁移步骤与验收

1. **实现最小原生链路**：完成 factory 注册和 session，单 stream kernel 事件通过 Kineto 输出；核对 device、stream、名称、时长与 CPU 关联，不依赖 JSON 补入。
2. **补齐采集语义**：实现时钟转换、关联传播、并发 buffer、窗口隔离和丢事件统计；用可控执行时长及多 stream 工作负载验证归属与边界。
3. **建立兼容矩阵**：覆盖目标 PyTorch 版本、Kineto 可用与不可用配置、CPU-only 与 CPU + PrivateUse1、普通与 gzip 导出、trace handler、多轮 schedule 和异常退出；对两种导出实现分别验证后再声明支持。
4. **收敛兼容补丁**：原生路径达到验收要求后，按能力选择路径，逐步移除强制 `use_kineto=False`、supported activities 修改、Python 生命周期包装和文件补入；保留 fallback 的边界与退出条件。

完成标准应同时包含 `key_averages()` 的设备时间归属、trace 的真实 kernel 活动及关联、多轮 session 隔离、溢出诊断和可接受的运行开销。仅测试“不报错”或“生成 trace 文件”不足以关闭此 TODO。

## 回归验证

在安装匹配的扩展后，从仓库根目录运行现有 profiler 测试：

```bash
python tests/test_profiler.py
```

当前测试覆盖基础事件、标注、耗时统计、窗口启动前同步与清理、时间单位转换、gzip 后处理和多个 stream lane。后续 Kineto 接入需要按上述验收项增加原生采集、精确关联、并发、容量溢出和 schedule 多周期测试。
