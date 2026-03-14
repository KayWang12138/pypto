# Vulkan GPU Torch 算子方案模块拆解任务列表

## 1. 目的

本文把 [torch_operator_gpu_vk_pipeline_plan.md](/home/anfield/project/pypto/custom/report/torch_operator_gpu_vk_pipeline_plan.md) 进一步细化为可执行的“模块拆解任务列表”。

目标是直接回答三个问题：

1. 每个阶段要新增哪些文件
2. 每个文件的职责是什么
3. 每个阶段最小需要哪些接口定义

本文默认采用以下约束：

- GPU 后端使用 Vulkan Compute API
- 先用独立 GPU backend 目录，不污染现有 NPU 实现
- 先允许 CPU staging 路径，后续再做 GPU 常驻与零拷贝
- Python 对外接口先独立放在 `python/pypto_gpu_vk/`

## 2. 建议的总代码布局

```text
framework/src/gpu_vk/
├── common/
├── ir/
├── passes/
├── codegen/
├── runtime/
├── cache/
└── profile/

python/src/bindings/
├── gpu_vk_runtime.cpp
├── gpu_vk_ir.cpp
└── gpu_vk_codegen.cpp

python/pypto_gpu_vk/
├── __init__.py
├── runtime.py
├── converter.py
├── config.py
├── backend.py
├── compile.py
└── op/

python/tests/ut/gpu_vk/
framework/tests/ut/gpu_vk/
framework/tests/st/gpu_vk/
```

## 3. 建议的构建开关

建议新增独立编译开关：

- `BUILD_WITH_VULKAN`
- `GPU_VK_ENABLE_VALIDATION`
- `GPU_VK_ENABLE_SHADER_CACHE`

建议新增的构建改动文件：

- `CMakeLists.txt`
- `python/src/CMakeLists.txt`
- `python/src/bindings/bindings.h`
- `python/src/pybind11.cpp`

## 4. 阶段总览

| Phase | 目标 | 关键新增模块 | 是否必须修改现有 NPU 代码 |
|---|---|---|---|
| Phase 0 | 跑通最小 Vulkan torch 算子 | runtime + Python binding + 手写 shader | 否 |
| Phase 1 | 建立 backend-neutral IR 和 GPU pass | `ir/` + `passes/` | 否 |
| Phase 2 | 建立自动 SPIR-V codegen | `codegen/` | 否 |
| Phase 3 | Vulkan runtime 工程化 | `runtime/` + `cache/` | 否 |
| Phase 4 | Torch 与 Vulkan 资源互通 | `converter/interop` | 可能 |
| Phase 5 | 性能优化与 autotune | `profile/` + `passes/fusion` | 否 |

---

## 5. Phase 0: 最小可运行链路

## 5.1 目标

先跑通：

- `add`
- `mul`
- `relu`

输入输出先允许：

- CPU `torch.Tensor`
- CPU staging copy 到 `VkBuffer`

这一阶段不要引入：

- 自动 codegen
- 复杂 IR
- 零拷贝

## 5.2 需要新增的文件

### C++ Runtime

- `framework/src/gpu_vk/common/vk_status.h`
- `framework/src/gpu_vk/common/vk_error.h`
- `framework/src/gpu_vk/runtime/vk_instance.h`
- `framework/src/gpu_vk/runtime/vk_instance.cpp`
- `framework/src/gpu_vk/runtime/vk_device.h`
- `framework/src/gpu_vk/runtime/vk_device.cpp`
- `framework/src/gpu_vk/runtime/vk_buffer.h`
- `framework/src/gpu_vk/runtime/vk_buffer.cpp`
- `framework/src/gpu_vk/runtime/vk_shader_module.h`
- `framework/src/gpu_vk/runtime/vk_shader_module.cpp`
- `framework/src/gpu_vk/runtime/vk_pipeline.h`
- `framework/src/gpu_vk/runtime/vk_pipeline.cpp`
- `framework/src/gpu_vk/runtime/vk_descriptor.h`
- `framework/src/gpu_vk/runtime/vk_descriptor.cpp`
- `framework/src/gpu_vk/runtime/vk_command.h`
- `framework/src/gpu_vk/runtime/vk_command.cpp`
- `framework/src/gpu_vk/runtime/vk_launcher.h`
- `framework/src/gpu_vk/runtime/vk_launcher.cpp`

### Python Binding

- `python/src/bindings/gpu_vk_runtime.cpp`

### Python 包装

- `python/pypto_gpu_vk/__init__.py`
- `python/pypto_gpu_vk/runtime.py`
- `python/pypto_gpu_vk/converter.py`
- `python/pypto_gpu_vk/op/elementwise.py`

### 测试

- `python/tests/ut/gpu_vk/test_runtime_add.py`
- `python/tests/ut/gpu_vk/test_runtime_mul.py`
- `framework/tests/ut/gpu_vk/test_vk_buffer.cpp`
- `framework/tests/ut/gpu_vk/test_vk_pipeline.cpp`

### 构建改动

- `python/src/CMakeLists.txt`
- `python/src/bindings/bindings.h`
- `python/src/pybind11.cpp`

## 5.3 最小接口定义

### `vk_status.h`

```cpp
#pragma once

namespace npu::tile_fwk::gpu_vk {
enum class VkStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    SHADER_COMPILE_FAILED,
    PIPELINE_CREATE_FAILED,
    DISPATCH_FAILED,
};
}
```

### `vk_instance.h`

```cpp
#pragma once
#include <vulkan/vulkan.h>

namespace npu::tile_fwk::gpu_vk {
class VkInstanceContext {
public:
    VkStatus Initialize(bool enableValidation);
    void Destroy();
    VkInstance Get() const;
private:
    VkInstance instance_{VK_NULL_HANDLE};
};
}
```

### `vk_device.h`

```cpp
#pragma once

namespace npu::tile_fwk::gpu_vk {
class VkDeviceContext {
public:
    VkStatus Initialize(VkInstance instance);
    void Destroy();
    VkPhysicalDevice PhysicalDevice() const;
    VkDevice Device() const;
    VkQueue ComputeQueue() const;
    uint32_t ComputeQueueFamily() const;
private:
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue computeQueue_{VK_NULL_HANDLE};
    uint32_t computeQueueFamily_{0};
};
}
```

### `vk_buffer.h`

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace npu::tile_fwk::gpu_vk {
enum class VkBufferUsageKind {
    STAGING,
    STORAGE,
};

class VkBufferHandle {
public:
    VkStatus Allocate(VkDevice device, VkPhysicalDevice physicalDevice, size_t bytes, VkBufferUsageKind usage);
    VkStatus Upload(const void *src, size_t bytes);
    VkStatus Download(void *dst, size_t bytes) const;
    void Destroy();
    VkBuffer Buffer() const;
    size_t Size() const;
private:
    VkBuffer buffer_{VK_NULL_HANDLE};
    VkDeviceMemory memory_{VK_NULL_HANDLE};
    size_t size_{0};
};
}
```

### `vk_pipeline.h`

```cpp
#pragma once

#include <string>
#include <vector>

namespace npu::tile_fwk::gpu_vk {
struct VkBindingDesc {
    uint32_t binding;
    uint32_t descriptorType;
};

struct VkLaunchSpec {
    uint32_t localX;
    uint32_t localY;
    uint32_t localZ;
    uint32_t groupX;
    uint32_t groupY;
    uint32_t groupZ;
};

class VkComputePipelineHandle {
public:
    VkStatus Create(
        VkDevice device,
        const std::vector<uint32_t> &spirv,
        const std::vector<VkBindingDesc> &bindings);
    void Destroy();
    VkPipeline Pipeline() const;
    VkPipelineLayout Layout() const;
    VkDescriptorSetLayout DescriptorLayout() const;
};
}
```

### `vk_launcher.h`

```cpp
#pragma once

#include <vector>

namespace npu::tile_fwk::gpu_vk {
struct VkTensorDesc {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int dtype;
    size_t nbytes;
};

class VkLauncher {
public:
    VkStatus Initialize(bool enableValidation);
    VkStatus RunBinaryElementwise(
        const void *input0,
        const void *input1,
        void *output,
        const VkTensorDesc &desc,
        const std::vector<uint32_t> &spirv,
        const VkLaunchSpec &launchSpec);
    void Destroy();
};
}
```

### `gpu_vk_runtime.cpp`

```cpp
void BindGpuVkRuntime(py::module &m);
```

绑定最少导出：

- `GpuVkLauncher`
- `run_elementwise_binary()`
- `run_elementwise_unary()`

### `python/pypto_gpu_vk/runtime.py`

```python
class VkRuntime:
    def __init__(self, enable_validation: bool = False): ...
    def run_elementwise_binary(self, op_name: str, a, b, out=None): ...
    def run_elementwise_unary(self, op_name: str, x, out=None): ...
```

### `python/pypto_gpu_vk/converter.py`

```python
def torch_to_vk_cpu_staging(tensor) -> dict: ...
def alloc_output_like(tensor): ...
```

## 5.4 本阶段任务顺序

1. 建立 Vulkan 基础上下文
2. 建立 buffer 上传下载路径
3. 手写 `add.comp` shader
4. 在 C++ 侧跑通单次 dispatch
5. 暴露到 Python
6. 用 `torch.Tensor(cpu)` 跑通单元测试

---

## 6. Phase 1: Backend-neutral IR 与 GPU Pass

## 6.1 目标

把“手写 shader + 手写 launch”升级为：

- 有统一 IR
- 有 GPU lowering pass
- 仍然不直接暴露 Vulkan API 到高层 IR

## 6.2 需要新增的文件

### IR

- `framework/src/gpu_vk/ir/gpu_vk_tensor_ir.h`
- `framework/src/gpu_vk/ir/gpu_vk_tensor_ir.cpp`
- `framework/src/gpu_vk/ir/gpu_vk_dispatch_ir.h`
- `framework/src/gpu_vk/ir/gpu_vk_dispatch_ir.cpp`
- `framework/src/gpu_vk/ir/gpu_vk_shader_ir.h`
- `framework/src/gpu_vk/ir/gpu_vk_shader_ir.cpp`

### Pass

- `framework/src/gpu_vk/passes/gpu_vk_pass.h`
- `framework/src/gpu_vk/passes/gpu_vk_pass_manager.h`
- `framework/src/gpu_vk/passes/gpu_vk_pass_manager.cpp`
- `framework/src/gpu_vk/passes/constant_fold.h`
- `framework/src/gpu_vk/passes/constant_fold.cpp`
- `framework/src/gpu_vk/passes/layout_normalize.h`
- `framework/src/gpu_vk/passes/layout_normalize.cpp`
- `framework/src/gpu_vk/passes/broadcast_legalize.h`
- `framework/src/gpu_vk/passes/broadcast_legalize.cpp`
- `framework/src/gpu_vk/passes/op_fusion.h`
- `framework/src/gpu_vk/passes/op_fusion.cpp`
- `framework/src/gpu_vk/passes/tile_lowering.h`
- `framework/src/gpu_vk/passes/tile_lowering.cpp`
- `framework/src/gpu_vk/passes/dispatch_lowering.h`
- `framework/src/gpu_vk/passes/dispatch_lowering.cpp`

### Python 包装

- `python/src/bindings/gpu_vk_ir.cpp`
- `python/pypto_gpu_vk/backend.py`
- `python/pypto_gpu_vk/compile.py`

### 测试

- `python/tests/ut/gpu_vk/test_ir_builder.py`
- `python/tests/ut/gpu_vk/test_dispatch_lowering.py`
- `framework/tests/ut/gpu_vk/test_gpu_vk_pass_manager.cpp`

## 6.3 最小接口定义

### `gpu_vk_tensor_ir.h`

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace npu::tile_fwk::gpu_vk {
enum class GpuVkOpKind {
    INPUT,
    OUTPUT,
    ADD,
    MUL,
    RELU,
    WHERE,
    SUM,
    MATMUL,
};

struct GpuVkValue {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int dtype;
};

struct GpuVkNode {
    GpuVkOpKind op;
    std::vector<GpuVkValue*> inputs;
    std::vector<GpuVkValue*> outputs;
};

class GpuVkTensorGraph {
public:
    GpuVkValue *AddInput(const GpuVkValue &value);
    GpuVkValue *AddIntermediate(const GpuVkValue &value);
    GpuVkValue *AddOutput(const GpuVkValue &value);
    GpuVkNode *AddNode(GpuVkOpKind op, const std::vector<GpuVkValue*> &inputs, const std::vector<GpuVkValue*> &outputs);
    const std::vector<std::unique_ptr<GpuVkNode>> &Nodes() const;
};
}
```

### `gpu_vk_dispatch_ir.h`

```cpp
struct GpuVkDispatchParam {
    uint32_t groupX;
    uint32_t groupY;
    uint32_t groupZ;
    uint32_t localX;
    uint32_t localY;
    uint32_t localZ;
};

struct GpuVkBufferBinding {
    std::string name;
    uint32_t binding;
    bool isOutput;
};

class GpuVkDispatchGraph {
public:
    void AddBinding(const GpuVkBufferBinding &binding);
    void SetDispatch(const GpuVkDispatchParam &param);
    const GpuVkDispatchParam &Dispatch() const;
};
```

### `gpu_vk_shader_ir.h`

```cpp
struct GpuVkShaderOp {
    GpuVkOpKind op;
    std::string expr;
};

class GpuVkShaderFunction {
public:
    void SetName(const std::string &name);
    void AddOp(const GpuVkShaderOp &op);
    const std::vector<GpuVkShaderOp> &Ops() const;
};
```

### `gpu_vk_pass.h`

```cpp
class GpuVkPass {
public:
    virtual ~GpuVkPass() = default;
    virtual const char *Name() const = 0;
    virtual VkStatus Run(GpuVkTensorGraph &graph) = 0;
};
```

### `gpu_vk_pass_manager.h`

```cpp
class GpuVkPassManager {
public:
    void RegisterDefaultPasses();
    VkStatus RunTensorPasses(GpuVkTensorGraph &graph);
    VkStatus LowerToDispatch(GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph);
};
```

### `python/pypto_gpu_vk/compile.py`

```python
def build_tensor_graph(op_name: str, *inputs): ...
def lower_to_dispatch(graph): ...
```

## 6.4 本阶段任务顺序

1. 建立 tensor IR
2. 建立 dispatch IR
3. 落常量折叠与 broadcast legalize
4. 落 tile/workgroup lowering
5. 让 `add/mul/relu/where` 从 IR 自动 lower 到 dispatch graph

---

## 7. Phase 2: SPIR-V CodeGen

## 7.1 目标

自动生成：

- GLSL compute shader
- SPIR-V
- ShaderMeta

## 7.2 需要新增的文件

- `framework/src/gpu_vk/codegen/shader_meta.h`
- `framework/src/gpu_vk/codegen/shader_meta.cpp`
- `framework/src/gpu_vk/codegen/glsl_codegen.h`
- `framework/src/gpu_vk/codegen/glsl_codegen.cpp`
- `framework/src/gpu_vk/codegen/spirv_compiler.h`
- `framework/src/gpu_vk/codegen/spirv_compiler.cpp`
- `framework/src/gpu_vk/codegen/gpu_vk_artifact.h`
- `framework/src/gpu_vk/codegen/gpu_vk_artifact.cpp`
- `python/src/bindings/gpu_vk_codegen.cpp`
- `python/tests/ut/gpu_vk/test_glsl_codegen.py`
- `python/tests/ut/gpu_vk/test_spirv_compile.py`

## 7.3 最小接口定义

### `shader_meta.h`

```cpp
struct ShaderBindingMeta {
    uint32_t binding;
    std::string name;
    int dtype;
    bool isOutput;
};

struct PushConstantMeta {
    std::string name;
    uint32_t offset;
    uint32_t size;
};

struct ShaderMeta {
    std::string kernelName;
    std::vector<ShaderBindingMeta> bindings;
    std::vector<PushConstantMeta> pushConstants;
    GpuVkDispatchParam dispatch;
};
```

### `glsl_codegen.h`

```cpp
class GlslComputeCodegen {
public:
    std::string Emit(const GpuVkShaderFunction &shaderFunc, const ShaderMeta &meta) const;
};
```

### `spirv_compiler.h`

```cpp
struct SpirvCompileOptions {
    bool optimize{true};
    bool debugInfo{false};
};

class SpirvCompiler {
public:
    VkStatus CompileGlslToSpirv(
        const std::string &glsl,
        const SpirvCompileOptions &options,
        std::vector<uint32_t> &spirvOut) const;
};
```

### `gpu_vk_artifact.h`

```cpp
struct GpuVkArtifact {
    ShaderMeta meta;
    std::string glsl;
    std::vector<uint32_t> spirv;
};
```

### `python/pypto_gpu_vk/compile.py`

```python
def lower_to_shader(graph): ...
def compile_to_spirv(shader_ir): ...
```

## 7.4 本阶段任务顺序

1. 先为 `add/mul/relu` 生成 GLSL
2. 接 `glslangValidator` 或 `shaderc`
3. 输出 `ShaderMeta + SPIR-V`
4. 把 runtime 从“读手写 shader”切到“读 codegen artifact”

---

## 8. Phase 3: Vulkan Runtime 工程化

## 8.1 目标

把一次性 demo runtime 升级成可复用执行器。

## 8.2 需要新增的文件

- `framework/src/gpu_vk/runtime/vk_allocator.h`
- `framework/src/gpu_vk/runtime/vk_allocator.cpp`
- `framework/src/gpu_vk/runtime/vk_descriptor_cache.h`
- `framework/src/gpu_vk/runtime/vk_descriptor_cache.cpp`
- `framework/src/gpu_vk/runtime/vk_pipeline_cache.h`
- `framework/src/gpu_vk/runtime/vk_pipeline_cache.cpp`
- `framework/src/gpu_vk/runtime/vk_executable.h`
- `framework/src/gpu_vk/runtime/vk_executable.cpp`
- `framework/src/gpu_vk/runtime/vk_runner.h`
- `framework/src/gpu_vk/runtime/vk_runner.cpp`
- `framework/src/gpu_vk/cache/shape_cache.h`
- `framework/src/gpu_vk/cache/shape_cache.cpp`
- `framework/src/gpu_vk/cache/artifact_cache.h`
- `framework/src/gpu_vk/cache/artifact_cache.cpp`
- `framework/tests/ut/gpu_vk/test_vk_runner.cpp`
- `python/tests/ut/gpu_vk/test_runtime_cache.py`

## 8.3 最小接口定义

### `vk_allocator.h`

```cpp
class VkAllocator {
public:
    VkStatus Initialize(VkDevice device, VkPhysicalDevice physicalDevice);
    VkStatus AllocStorageBuffer(size_t bytes, VkBufferHandle &out);
    VkStatus AllocStagingBuffer(size_t bytes, VkBufferHandle &out);
    void Destroy();
};
```

### `vk_pipeline_cache.h`

```cpp
struct PipelineCacheKey {
    std::string kernelName;
    std::vector<int64_t> shapeSignature;
};

class VkPipelineCacheManager {
public:
    bool Find(const PipelineCacheKey &key, VkComputePipelineHandle *&pipeline);
    void Insert(const PipelineCacheKey &key, std::unique_ptr<VkComputePipelineHandle> pipeline);
};
```

### `vk_executable.h`

```cpp
struct VkExecutable {
    GpuVkArtifact artifact;
    std::unique_ptr<VkComputePipelineHandle> pipeline;
};
```

### `vk_runner.h`

```cpp
class VkRunner {
public:
    VkStatus Initialize(bool enableValidation);
    VkStatus Run(
        const GpuVkArtifact &artifact,
        const std::vector<VkTensorDesc> &inputs,
        const std::vector<VkTensorDesc> &outputs,
        const std::vector<const void*> &hostInputPtrs,
        const std::vector<void*> &hostOutputPtrs);
    void Destroy();
};
```

## 8.4 本阶段任务顺序

1. 落 allocator
2. 落 descriptor/pipeline cache
3. 落 executable
4. 落统一 `VkRunner`
5. 让 `compile -> artifact -> run` 成为固定入口

---

## 9. Phase 4: Torch 与 Vulkan 资源互通

## 9.1 目标

从 CPU staging 模式升级到：

- Vulkan 常驻 buffer
- 或 Torch 与 Vulkan 外部内存互通

这是高风险阶段，必须单独切分。

## 9.2 需要新增的文件

- `framework/src/gpu_vk/runtime/vk_tensor_storage.h`
- `framework/src/gpu_vk/runtime/vk_tensor_storage.cpp`
- `framework/src/gpu_vk/runtime/torch_vk_interop.h`
- `framework/src/gpu_vk/runtime/torch_vk_interop.cpp`
- `python/src/bindings/torch_vk_interop.cpp`
- `python/pypto_gpu_vk/converter.py`
- `python/tests/ut/gpu_vk/test_torch_vk_interop.py`

## 9.3 最小接口定义

### `vk_tensor_storage.h`

```cpp
enum class VkTensorStorageMode {
    CPU_STAGING,
    VK_BUFFER,
    EXTERNAL_MEMORY,
};

struct VkTensorStorage {
    VkTensorStorageMode mode;
    VkBufferHandle buffer;
    VkTensorDesc desc;
};
```

### `torch_vk_interop.h`

```cpp
class TorchVkInterop {
public:
    VkStatus ImportFromTorchCpu(const py::object &tensor, VkTensorStorage &out);
    VkStatus ExportToTorchCpu(const VkTensorStorage &storage, py::object &outTensor);
    VkStatus TryImportExternalMemory(const py::object &tensor, VkTensorStorage &out);
};
```

### `python/pypto_gpu_vk/converter.py`

```python
def from_torch(tensor, storage_mode: str = "cpu_staging"): ...
def to_torch(vk_tensor): ...
```

## 9.4 本阶段任务顺序

1. 先把 `CPU_STAGING` 路径封装成统一 converter
2. 再做 Vulkan 原生 tensor storage
3. 最后再评估 external memory

---

## 10. Phase 5: 性能优化与 Autotune

## 10.1 目标

让 Vulkan backend 从“能跑”走到“可用”。

## 10.2 需要新增的文件

- `framework/src/gpu_vk/profile/vk_profiler.h`
- `framework/src/gpu_vk/profile/vk_profiler.cpp`
- `framework/src/gpu_vk/profile/workgroup_tuner.h`
- `framework/src/gpu_vk/profile/workgroup_tuner.cpp`
- `framework/src/gpu_vk/passes/fusion_planner.h`
- `framework/src/gpu_vk/passes/fusion_planner.cpp`
- `framework/src/gpu_vk/passes/shared_memory_planner.h`
- `framework/src/gpu_vk/passes/shared_memory_planner.cpp`
- `python/pypto_gpu_vk/tune.py`
- `python/tests/ut/gpu_vk/test_workgroup_tuner.py`
- `framework/tests/st/gpu_vk/test_vk_matmul_perf.cpp`

## 10.3 最小接口定义

### `vk_profiler.h`

```cpp
struct VkKernelProfile {
    uint64_t dispatchNs;
    uint32_t groupX;
    uint32_t groupY;
    uint32_t groupZ;
    uint32_t localX;
    uint32_t localY;
    uint32_t localZ;
};

class VkProfiler {
public:
    VkStatus Begin();
    VkStatus End();
    VkKernelProfile Read() const;
};
```

### `workgroup_tuner.h`

```cpp
struct WorkgroupCandidate {
    uint32_t localX;
    uint32_t localY;
    uint32_t localZ;
};

class WorkgroupTuner {
public:
    std::vector<WorkgroupCandidate> GenerateCandidates(const VkTensorDesc &desc) const;
    WorkgroupCandidate SelectBest(const std::vector<VkKernelProfile> &profiles) const;
};
```

### `python/pypto_gpu_vk/tune.py`

```python
def tune_workgroup(op_name: str, *inputs): ...
def profile_once(op_name: str, *inputs): ...
```

## 10.4 本阶段任务顺序

1. 落 timestamp profile
2. 落 workgroup 候选生成
3. 落 tuner
4. 落 fusion planner
5. 把 tuned 参数回灌到 dispatch lowering

---

## 11. Python 侧统一接口建议

为了避免 Python API 过早碎片化，建议最终收敛为以下接口：

### `python/pypto_gpu_vk/__init__.py`

```python
from .runtime import VkRuntime
from .converter import from_torch, to_torch
from .compile import compile_op
```

### `python/pypto_gpu_vk/compile.py`

```python
def compile_op(op_name: str, *inputs, options: dict | None = None):
    """
    输入 torch tensor 或 vk tensor，返回可执行对象。
    """

class CompiledVkOp:
    def __call__(self, *inputs): ...
    def dump_glsl(self) -> str: ...
    def dump_spirv(self) -> bytes: ...
```

### `python/pypto_gpu_vk/runtime.py`

```python
class VkRuntime:
    def __init__(self, enable_validation: bool = False, enable_cache: bool = True): ...
    def execute(self, compiled_op, *inputs): ...
    def synchronize(self): ...
```

## 12. Binding 层统一改动建议

需要改动的现有文件：

- [python/src/CMakeLists.txt](/home/anfield/project/pypto/python/src/CMakeLists.txt)
- [python/src/bindings/bindings.h](/home/anfield/project/pypto/python/src/bindings/bindings.h)
- [python/src/pybind11.cpp](/home/anfield/project/pypto/python/src/pybind11.cpp)

建议新增绑定入口：

```cpp
void BindGpuVkRuntime(py::module &m);
void BindGpuVkIr(py::module &m);
void BindGpuVkCodegen(py::module &m);
```

对应 `pybind11.cpp` 中新增：

```cpp
BindGpuVkRuntime(m);
BindGpuVkIr(m);
BindGpuVkCodegen(m);
```

## 13. 测试拆解建议

### Python UT

- `test_runtime_add.py`
- `test_runtime_mul.py`
- `test_ir_builder.py`
- `test_dispatch_lowering.py`
- `test_glsl_codegen.py`
- `test_spirv_compile.py`
- `test_runtime_cache.py`
- `test_torch_vk_interop.py`
- `test_workgroup_tuner.py`

### C++ UT

- `test_vk_buffer.cpp`
- `test_vk_pipeline.cpp`
- `test_vk_runner.cpp`
- `test_gpu_vk_pass_manager.cpp`

### ST

- `test_vk_add_e2e.py`
- `test_vk_matmul_e2e.py`
- `test_vk_matmul_perf.cpp`

## 14. 建议的实际落地顺序

建议严格按以下顺序推进：

1. Phase 0 Runtime 基础骨架
2. Phase 0 Python 接口与最小 e2e
3. Phase 1 IR
4. Phase 1 Pass
5. Phase 2 GLSL CodeGen
6. Phase 2 SPIR-V 编译
7. Phase 3 Cache/Runner
8. Phase 4 Interop
9. Phase 5 Tuner/Profile

不要一开始就做：

- matmul
- attention
- external memory
- 跨队列复杂同步

## 15. 可直接开工的第一批文件

如果现在就开始实现，建议第一批只创建这些文件：

- `framework/src/gpu_vk/common/vk_status.h`
- `framework/src/gpu_vk/runtime/vk_instance.h`
- `framework/src/gpu_vk/runtime/vk_instance.cpp`
- `framework/src/gpu_vk/runtime/vk_device.h`
- `framework/src/gpu_vk/runtime/vk_device.cpp`
- `framework/src/gpu_vk/runtime/vk_buffer.h`
- `framework/src/gpu_vk/runtime/vk_buffer.cpp`
- `framework/src/gpu_vk/runtime/vk_pipeline.h`
- `framework/src/gpu_vk/runtime/vk_pipeline.cpp`
- `framework/src/gpu_vk/runtime/vk_launcher.h`
- `framework/src/gpu_vk/runtime/vk_launcher.cpp`
- `python/src/bindings/gpu_vk_runtime.cpp`
- `python/pypto_gpu_vk/__init__.py`
- `python/pypto_gpu_vk/runtime.py`
- `python/tests/ut/gpu_vk/test_runtime_add.py`

第一批只需要跑通：

- CPU tensor 输入
- 手写 `add` shader
- 单次 Vulkan dispatch
- 输出回到 CPU tensor

## 16. 总结

如果把这份 Vulkan 方案落到“文件和接口”的级别，最重要的工程决策是：

1. 新建 `framework/src/gpu_vk/`，不要把 Vulkan 逻辑混入现有 NPU backend
2. 新建 `python/pypto_gpu_vk/`，不要一开始就改动现有 `python/pypto/`
3. 先做 runtime 骨架，再做 IR/pass/codegen
4. 先做 CPU staging，再做 Torch/Vulkan GPU 内存互通

一句话总结：

> 这条链路的最小落地路径应该是“先用独立 Vulkan runtime 跑通一个手写 shader 的 Torch 自定义算子”，然后再逐步把 runtime 外围替换成 `IR -> Pass -> SPIR-V CodeGen -> Cache -> Interop` 的完整编译执行系统。
