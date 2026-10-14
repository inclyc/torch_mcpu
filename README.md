# PyTorch OpenReg

## 背景

基于 PrivateUse1 的第三方设备集成机制已成为新后端集成到 PyTorch 的官方主流方法。确保该机制的可用性对于丰富 PyTorch 的硬件生态系统至关重要。

**注意：**

`torch_mcpu` 的目标**不是实现一个功能完整、高性能的 PyTorch 后端**，而是作为**机制验证的极简参考实现**。

### 目的

- **测试后端**：作为 PrivateUse1 的树内测试后端，通过 CI/CD 确保质量稳定性。
- **集成示例**：作为新后端集成的参考示例。
- **集成文档**：提供与代码对应的模块级集成文档。

### 设计原则

- **最小化原则**：根本目标是启用/验证新后端集成到 PyTorch 的所有集成路径/机制。所有功能遵循"恰到好处"策略，确保相关集成能力的正确性。
- **真实性原则**：以真实加速器后端集成到 PyTorch 的相同方式完成 OpenReg 集成。

## 目录结构

**依赖关系**:

```mermaid
graph LR
    A[Python]
    B[_C.so]
    C[libtorch_bindings.so]
    D[libtorch_mcpu.so]
    E[libmcpu.so]

    A --> B --> C --> D --> E
```

torch_mcpu 中有 4 个 DSO，它们之间的依赖关系如下：

- `_C.so`:
  - **源文件**: torch_mcpu/csrc/stub.c
  - **描述**: Python C 模块入口点。
- `libtorch_bindings.so`: Python 和 C++ 之间的桥接代码应该放在这里。
  - **源文件**: torch_mcpu/csrc
  - **描述**: Python 和 C++ 之间的薄胶水层。
- `libtorch_mcpu.so`: 所有核心实现应该放在这里。
  - **源文件**: csrc
  - **描述**: 所有核心功能，如设备运行时、运算符等。
- `libmcpu.so`: 一个使用 CPU 模拟类似 CUDA 设备的 DSO，你可以忽略它。
  - **源文件**: third_party/mcpu
  - **描述**: 提供类似于 libcudart.so 的底层设备功能。

**关键目录**:

- `csrc/`: 核心设备实现，包括运算符注册、运行时等。
  - `csrc/amp/`: AMP（自动混合精度）
  - `csrc/aten/`: 运算符注册
    - `csrc/aten/ops/`: 显式实现的 `aten` 算子，每个算子单独文件内实现并注册。
    - `csrc/aten/native/`: OpenReg 设备的特定运算符实现。
      - `csrc/aten/native/OpenRegMinimal.cpp`: 最小的运算符实现集（完成后允许创建 Tensor 及相关操作）。
      - `csrc/aten/native/OpenRegExtra.cpp`: 其他类型运算符的实现。
  - `csrc/runtime/`: Host 内存、设备内存、Guard、Hooks 等的实现。
- `third_party/`: 一个使用 CPU 模拟类似 CUDA 设备的 C++ 库。
- `torch_mcpu/`: Python 接口实现（Python 代码和 C++ 绑定）。
  - `torch_mcpu/csrc/`: Python C++ 绑定代码。
  - `torch_mcpu/mcpu/`: Python API。

## 当前已实现的功能

### 运算符注册

- 运算符实现

  - 为内置 PyTorch 运算符注册
    - `TORCH_LIBRARY_IMPL` 形式：参见 `empty.memory_format
    - `STUB` 形式：参见 `abs_stub`
  - 为自定义运算符注册
    - Schema 注册：参见 `custom_abs`
    - Kernel 注册：参见 `custom_abs`
    - 为 `AutogradPriavateUse1` 注册 Fallback：参见 `custom_abs`
    - Meta 注册：参见 `custom_abs`
    - `torch.autograd.Function`：参见 `custom_autograd_fn_aliasing`
  - 为 fallback 注册
    - 按运算符 Fallback：参见 `sub.Tensor`
    - 全局 Fallback：参见 `wrapper_cpu_fallback`

未手动接入的 ATen 算子默认启用通用 CPU fallback：先把所有输入的 D2H copy
排入当前 mcpu stream，再按当前 stream 去重同步，在 host 线程执行 CPU
kernel，最后按 CPU 输出的 size/stride 分配 mcpu tensor，并把回拷排入同一个
stream。后续 mcpu launch 因此会自然排在回拷之后。与 CUDA 一致，跨 stream
的数据依赖需要调用方通过 event 或 `wait_stream` 显式建立。涉及 storage alias
的 view 算子不能通过跨设备复制保持语义，仍需手动实现；可通过
`TORCH_MCPU_ENABLE_CPU_FALLBACK=OFF` 在构建时关闭通用 fallback。

### 自动加载

当 `import torch` 时，已安装的加速器（如 `torch_mcpu`）将被自动加载，实现与内置后端相同的体验。

- 使用 Python `entry points` 注册后端：参见 `setup.py` 中的 `setup`
- 添加后端初始化的可调用函数：参见 `torch_mcpu/__init__.py` 中的 `_autoload`
- 无需显式导入即可动态加载后端：参见 [使用示例](#使用示例)

### AMP（自动混合精度）

`AMP` 为混合精度提供便捷方法，某些操作使用 `torch.float32` 数据类型，其他操作使用`较低精度`的浮点数据类型：`torch.float16` 或 `torch.bfloat16`。

- 注册特定的运算符转换规则：参见 `csrc/amp` 中的 `autocat_mode.cpp`
- 为不同加速器添加新数据类型的支持：参见 `torch_mcpu/mcpu/amp/__init__.py` 中的 `get_amp_supported_dtype`

### Stream host 侧任务提交假设

当前 OpenReg stream 的队列实现面向低开销的热路径：**每个 stream 假设只有一个 host producer 线程提交任务，并由该 stream 独占的 worker 线程顺序消费任务**。

这意味着：

- 多个 stream 可以分别由不同 host 线程提交任务，因为每个 stream 有独立队列。
- 同一个 stream 不支持多个 host 线程并发调用 `orLaunchKernel` / `stream->launch` 提交任务。
- 当前队列是固定容量环形队列，`tail` 通过普通 load/store 推进，不使用 mutex、`fetch_add` 或 CAS 分配多 producer slot。
- 如果多个 host 线程同时向同一个 stream 提交任务，可能发生 slot 覆盖、任务丢失或队列状态损坏；这是未定义行为，不是受支持用法。

这个限制是当前性能取舍的一部分。相关设计和 benchmark 背景见 [docs/openreg_launch_queue_benchmark.md](/shared/vllm-xcpu-dev-kit-0.19/torch_mcpu/docs/openreg_launch_queue_benchmark.md)。

## 安装和使用

### 安装

```python
python -m pip install --no-build-isolation -e . # 用于开发
python -m pip install --no-build-isolation . # 用于安装
```

调试异步算子错误时，可以构建同步启动版本。该选项会在每次 MCPU
算子提交后同步等待其所在 stream 完成；默认关闭，不影响普通构建的异步行为。

```bash
TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH=ON \
  python -m pip install --no-build-isolation -e .
```

构建树中通过 CMake target 链接 `torch_mcpu` 的第三方 target 会继承该宏。
独立构建的第三方扩展应读取已安装包的 `torch_mcpu.get_compile_flags()`，
并把返回值加入其 C/C++ 编译参数，以保持同步启动、内存保护等构建配置一致。

算子启动辅助函数是公共头文件中的内联模板，宏分支会被编译进调用方。因此，
切换 `TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH` 后，本库内的算子会随本库重编而
生效，但使用这些启动辅助函数的既有第三方二进制不会自动改变，必须使用新的
编译参数重新编译。仅通过 dispatcher 调用本库已注册算子的第三方代码不受此
限制，因为实际启动点位于本库。

完成安装后，开发者必须在仓库根目录执行一次以下命令安装 Git hook：

```bash
./scripts/install-git-hooks.sh
```

未安装 hook 时，提交前不会自动执行本仓库要求的 `clang-format` 检查与修复。

### 提交前自动格式化

仓库使用根目录的 `.clang-format` 作为 C/C++ 代码风格配置。

安装后，每次执行 `git commit` 时，pre-commit hook 会：

- 只检查本次已经 `stage` 的 C/C++ 相关文件。
- 自动运行 `clang-format` 修复格式问题，并把修复结果重新加入本次提交。
- 如果某个已暂存文件同时存在未暂存改动，则拒绝提交，避免覆盖工作区中的部分修改。

如果 hook 自动修复了文件，本次提交会先被中断；确认格式化结果后，再次执行 `git commit` 即可。

手动全量格式化可执行：

```bash
./scripts/format.sh
```

这个脚本会扫描仓库内的 C/C++ 文件并执行 `clang-format`，默认排除 `third_party/googletest`。如果只想检查不改文件，可执行：

```bash
./scripts/format.sh --check
```

### 使用示例

安装后，你可以像使用任何其他常规设备一样在 Python 中使用 `mcpu` 设备。

```python
import torch

if not torch.mcpu.is_available():
    print("MCPU backend is not available in this build.")
    exit()

print("MCPU backend is available!")

device = torch.device("mcpu")

x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = x + 2
print("Result y:\n", y)
print(f"Device of y: {y.device}")

z = y.cpu()
print("Result z:\n", z)
print(f"Device of z: {z.device}")
```

## 文档

请参阅[此文档](https://docs.pytorch.org/docs/main/accelerator/index.html)，了解将新加速器集成到 PyTorch 的一系列文档，这些文档将与 `OpenReg` 代码库保持同步。

显式实现 `aten` 算子的仓库内说明见：

- [docs/how_to_impl_aten_ops.md](/shared/vllm-xcpu-dev-kit-0.19/torch_mcpu/docs/how_to_impl_aten_ops.md)

Profiler 的使用方式、PyTorch 2.13 兼容修复和长期 Kineto 适配计划见
[MCPU profiler 文档](docs/profiler.md)。

## 未来计划

- **增强功能**：
  - 设备无关 API
  - 内存管理
  - 生成器
  - 分布式
  - 自定义 Tensor&Storage
  - ...
- **改进测试**：添加更多与集成机制相关的测试用例。
- **Profiler 原生 Kineto 接入（重要 TODO）**：工程化适配 PyTorch PrivateUse1 profiler 与 Kineto 的注册、采集生命周期、时钟和事件关联接口；当前 PyTorch 2.13 修改仅维护 fallback 兼容路径，详见 [profiler 适配计划](docs/profiler.md)。
- **Stream 计时与观测性改进**：
  - 继续以当前 SPSC（single-producer/single-consumer）stream 队列作为主要性能取舍；同一个 stream 多 host producer 并发提交不是当前优化目标。
  - 将 profiler/kernel timing 的 event buffer 与 stream 并发模型对齐；优先评估 per-stream timing buffer，避免全局计数热点。
  - 在 event 写满时暴露 dropped event 计数，避免 trace 静默截断。
  - 强化 profiler start/reset/synchronize 的窗口语义，确保旧任务不会污染新的 profile 区间。
