# Vulkan GPU 真实编译与执行替换 Stub 实施计划

## 1. 概要

目标是把当前 `pypto_gpu_vk` 中的伪 `SPIR-V`、伪 `VkRunner` 和 Python 侧 CPU fallback 执行路径，替换为一条真实可用的 Vulkan Compute 执行链，同时保持现有 `IR -> Pass -> CodeGen -> Runtime -> Cache -> Python API` 的分层不变。

本计划锁定以下范围与策略：

- 首批真实支持算子：`add`、`mul`、`relu`、`where`、`matmul`
- `matmul` 首版采用 `Balanced Tile`
  - `buffer-only`
  - `shared memory + 16x16x1 local size`
  - `K tile = 16`
  - 每个 invocation 计算 1 个输出元素
- 真实 `SPIR-V` 编译采用 `CLI glslangValidator`
- CPU fallback 仅保留为开发调试能力
  - 默认公共运行路径不自动 fallback
  - 仅显式配置或环境变量开启时才允许 fallback
- 首版真实执行只支持 `float32` CPU staging 输入输出
  - 不做 Vulkan-native tensor 常驻
  - 不做 external memory
  - `float16/bfloat16/image path/subgroup optimization` 不进入本次范围

## 2. 当前状态

当前仓里已经有完整 scaffold，但真实能力仍是 stub：

- `framework/src/gpu_vk/codegen/spirv_compiler.cpp` 生成的是伪 `SPIR-V`
- `framework/src/gpu_vk/runtime/vk_launcher.cpp` / `vk_runner.cpp` 仍返回 `UNAVAILABLE`
- `python/pypto_gpu_vk/runtime.py` 仍默认走 torch CPU fallback
- `IR / Pass / CodeGen / Cache / Interop / Profile` 分层已经存在，可以直接替换实现，不需要推翻结构

因此这次工作是“替换实现”，不是“重做架构”。

## 3. 目标完成标准

实现完成后必须满足：

1. `compile_op("add" | "mul" | "relu" | "where" | "matmul", ...)` 产出的 `artifact.spirv` 来自 `glslangValidator` 真实编译结果
2. `VkRuntime.execute()` 在默认配置下真正走 Vulkan，不再默认回退 CPU
3. `VkRunner` 能完成：
   - 创建 instance/device/queue
   - 创建 shader module / descriptor set layout / pipeline layout / compute pipeline
   - 创建 staging/storage buffer
   - 上传输入
   - dispatch
   - 下载输出
4. `add/mul/relu/where/matmul` 与 PyTorch CPU 结果对齐
5. 第二次相同 shape/signature 调用命中 pipeline/artifact cache
6. 无 Vulkan 设备、无 `glslangValidator`、shader 编译失败、pipeline 创建失败、dispatch 失败等场景均返回明确错误并附带诊断信息

## 4. 公共接口与类型变更

### 4.1 `VkStatus` 扩展

更新 `framework/src/gpu_vk/common/vk_status.h`：

新增：

- `DEVICE_NOT_FOUND`
- `TOOL_NOT_FOUND`
- `SHADER_COMPILE_FAILED`
- `SHADER_MODULE_CREATE_FAILED`
- `DESCRIPTOR_CREATE_FAILED`
- `PIPELINE_LAYOUT_CREATE_FAILED`
- `PIPELINE_CREATE_FAILED`
- `BUFFER_CREATE_FAILED`
- `MEMORY_ALLOC_FAILED`
- `COMMAND_RECORD_FAILED`
- `QUEUE_SUBMIT_FAILED`
- `DISPATCH_FAILED`
- `DOWNLOAD_FAILED`

保留：

- `SUCCESS`
- `INVALID_ARGUMENT`
- `OUT_OF_MEMORY`
- `NOT_SUPPORTED`
- `INTERNAL_ERROR`

### 4.2 `GpuVkArtifact` 真实化

更新 `framework/src/gpu_vk/codegen/gpu_vk_artifact.h`：

新增字段：

- `entryPoint = "main"`
- `glslPath`
- `spirvPath`
- `compileLog`
- `sourceHash`
- `isRealSpirv`

语义：

- `spirv` 存真实 `.spv` 文件内容
- `compileLog` 存 `glslangValidator` stderr/stdout 摘要
- `sourceHash` 参与 cache key
- `isRealSpirv` 用于测试与运行时断言

### 4.3 `RuntimeConfig` 默认行为调整

更新 `python/pypto_gpu_vk/config.py` 与 `python/pypto_gpu_vk/runtime.py`：

- `fallback_to_torch` 默认从 `True` 改为 `False`
- 新增：
  - `prefer_real_vulkan: bool = True`
  - `allow_cpu_fallback_for_dev: bool = False`
  - `glslang_validator_path: str | None = None`
  - `dump_artifacts: bool = False`
  - `dump_dir: str | None = None`

开发环境允许通过以下方式显式开启 fallback：

- `VkRuntime(..., fallback_to_torch=True)`
- 或环境变量 `PYPTO_VK_ALLOW_CPU_FALLBACK=1`

### 4.4 Python 执行接口保持不变，但语义变化

保留：

- `compile_op(...)`
- `VkRuntime.execute(compiled_op, *inputs)`

变化：

- 默认执行真实 Vulkan
- 只有显式允许时才走 CPU fallback
- `run_elementwise_binary/unary` 也遵守同一行为

## 5. 详细实现方案

## 5.1 真实 SPIR-V 编译链

### 5.1.1 `SpirvCompiler` 改为调用 `glslangValidator`

修改 `framework/src/gpu_vk/codegen/spirv_compiler.cpp`：

实现固定为 CLI：

1. 将 GLSL 写入临时文件
   - `/tmp/pypto_vk/<sourceHash>.comp`
2. 输出目标
   - `/tmp/pypto_vk/<sourceHash>.spv`
3. 调用命令
   - `glslangValidator -S comp -V --target-env vulkan1.2 -o <spv> <glsl>`
4. 当 `options.optimize=True`
   - 首版只走 `glslangValidator`
   - 不引入 `spirv-opt`
5. 当 `options.debugInfo=True`
   - 增加 `-g`

失败处理：

- 工具不存在：`TOOL_NOT_FOUND`
- 返回码非 0：`SHADER_COMPILE_FAILED`
- 输出 `.spv` 不存在或为空：`SHADER_COMPILE_FAILED`

### 5.1.2 `glslangValidator` 路径解析规则

固定顺序：

1. `RuntimeConfig.glslang_validator_path`
2. 环境变量 `PYPTO_VK_GLSLANG_VALIDATOR`
3. `PATH` 中查找 `glslangValidator`

工具解析逻辑放在 `SpirvCompiler` 内部，不放到 Python 层。

### 5.1.3 产物 dump 行为

当 `dump_artifacts=True` 时：

- GLSL 存到 `<dump_dir>/<hash>.comp`
- SPIR-V 存到 `<dump_dir>/<hash>.spv`
- 编译日志存到 `<dump_dir>/<hash>.log`

默认 `dump_dir`：

- `./build_out/gpu_vk_dump`

## 5.2 Vulkan Runtime 真实执行链

### 5.2.1 `VkInstanceContext`

修改 `framework/src/gpu_vk/runtime/vk_instance.cpp`：

- 调 `vkCreateInstance`
- 当 `enableValidation=True` 时尝试开启：
  - `VK_LAYER_KHRONOS_validation`
- 若 layer 不存在：
  - 不报错
  - 记录 warning
  - 继续无 validation 运行

默认目标：

- Vulkan API 1.2

### 5.2.2 设备选择规则

目标平台固定为：

- Linux x86_64
- 桌面级 Vulkan 驱动
- 至少 1 个支持 compute queue 的 GPU

选择顺序：

1. discrete GPU
2. integrated GPU
3. 其他支持 compute 的 physical device

### 5.2.3 `VkDeviceContext`

修改 `framework/src/gpu_vk/runtime/vk_device.cpp`：

- 枚举 physical device
- 选择满足条件的 compute device
- 创建 logical device
- 获取 compute queue
- 记录 `queueFamilyIndex`

若找不到 compute 设备：

- 返回 `DEVICE_NOT_FOUND`

### 5.2.4 `VkBufferHandle`

修改 `framework/src/gpu_vk/runtime/vk_buffer.h/.cpp`：

从当前 `std::vector<uint8_t>` stub 改为真实 Vulkan buffer：

- `VkBuffer`
- `VkDeviceMemory`
- `size`
- `usage`
- `mappedPtr` 仅对 host-visible staging buffer 生效

首版内存策略固定：

- staging buffer：
  - `HOST_VISIBLE | HOST_COHERENT`
- storage buffer：
  - `DEVICE_LOCAL`
- 上传/下载通过 copy 完成，不直接 map storage buffer

### 5.2.5 `VkAllocator`

修改 `framework/src/gpu_vk/runtime/vk_allocator.cpp`：

职责：

- 根据 usage 创建 buffer
- 查询 memory requirements
- 按固定策略选择 memory type
- 绑定 buffer 与 memory

首版不做：

- 子分配器
- slab
- memory pooling

### 5.2.6 `VkDescriptorCache` / `VkPipelineCacheManager`

`VkDescriptorCache`：

- 缓存对象改为真实：
  - `VkDescriptorSetLayout`
  - `VkDescriptorPool`
  - `VkDescriptorSet`
- cache key：
  - binding count
  - binding order
  - readonly/writeonly 标记

`VkPipelineCacheManager`：

- cache key：
  - `kernelName`
  - `sourceHash`
  - input/output shape signature
  - local size
  - entryPoint
- 缓存对象：
  - `VkShaderModule`
  - `VkPipelineLayout`
  - `VkPipeline`

### 5.2.7 `VkRunner::Run`

修改 `framework/src/gpu_vk/runtime/vk_runner.cpp`：

真实执行步骤固定为：

1. 校验 artifact 与输入输出描述
2. 构建/命中 artifact cache
3. 构建/命中 pipeline cache
4. 分配 staging/storage buffer
5. 上传输入：
   - host -> staging
   - staging -> storage
6. 写 descriptor set
7. 录 command buffer：
   - bind pipeline
   - bind descriptor set
   - push constants
   - dispatch
   - memory barrier
   - storage -> staging
8. queue submit + fence wait
9. 下载输出：
   - staging -> host output ptr
10. 返回 `SUCCESS`

命令录制固定设计：

- 单 command pool
- 单 reusable command buffer
- 单 fence，同步等待
- 不做 async queue
- 不做 timeline semaphore
- 不做多 stream

Python binding `GpuVkRunner.run_artifact(...)` 改为：

- 从 `torch.Tensor` 提取 `data_ptr()`
- 创建 CPU output tensor
- 调 C++ `Run`
- 返回 output tensor 或 output tuple

不再只返回 `VkStatus`。

## 5.3 Shader 与 CodeGen 真实化

### 5.3.1 Elementwise GLSL

更新 `framework/src/gpu_vk/codegen/glsl_codegen.cpp`：

首版 elementwise 统一为 1D dispatch：

- `local_size_x = 256`
- `group_x = ceil(numel / 256)`
- `group_y = 1`
- `group_z = 1`

支持：

- `add`
- `mul`
- `relu`
- `where`

每个 shader 模板都使用 storage buffer。

`where` 规则固定：

- `cond != 0.0f` 视为 true
- 条件输入也按 `float32` buffer 处理

### 5.3.2 Matmul GLSL

新增或重构 `matmul` shader 模板，仍放在 `framework/src/gpu_vk/codegen/glsl_codegen.cpp`：

首版 `Balanced Tile` 固定方案：

- 输入：
  - `A[M, K]`
  - `B[K, N]`
- 输出：
  - `C[M, N]`
- layout：
  - `local_size_x = 16`
  - `local_size_y = 16`
  - `local_size_z = 1`
- shared memory：
  - `shared float Asub[16][16]`
  - `shared float Bsub[16][16]`
- group：
  - `group_x = ceil(N / 16)`
  - `group_y = ceil(M / 16)`
- 每个 invocation 计算一个 `C[row, col]`
- 尾块处理：
  - 越界元素补 0
  - 输出边界写保护

首版不支持：

- batched matmul
- transpose flag
- alpha/beta
- mixed precision
- subgroup matrix extension

### 5.3.3 Push constants 设计

`ShaderMeta.pushConstants` 固定包含：

elementwise：

- `numel`

matmul：

- `M`
- `N`
- `K`

## 5.4 Pass 与 IR 调整

### 5.4.1 `GpuVkTensorGraph` 保持 backend-neutral

不把 Vulkan 资源类型泄漏进 tensor graph。

### 5.4.2 `LowerToDispatch` 的真实输出

修改 `framework/src/gpu_vk/passes/gpu_vk_pass_manager.cpp` 和相关 lowering pass：

对每种真实支持算子生成确定的 dispatch 语义：

- `add/mul/relu/where`
  - 1D launch
  - 1 output
- `matmul`
  - 2D launch
  - push constants = `M,N,K`

`dispatchGraph.loweredOps` 不再只是调试字符串，必须能用于 shader 模板选择：

- `elementwise_add`
- `elementwise_mul`
- `elementwise_relu`
- `elementwise_where`
- `matmul_f32_tile16`

### 5.4.3 shape inference 与约束

在 `build_tensor_graph` 或 pass 阶段补强 shape 校验：

`add/mul/relu/where`：

- 首版只接受完全相同 shape
- 不做真正 broadcast

`matmul`：

- 严格要求 rank=2
- `A.shape[1] == B.shape[0]`

不满足直接报 `INVALID_ARGUMENT`。

## 5.5 Python 层行为调整

### 5.5.1 `compile.py`

更新 `python/pypto_gpu_vk/compile.py`：

`compile_op()` 的返回对象必须包含：

- `artifact`
- `dispatch_graph`
- `shader_ir`
- `compiled_kind`
- `supports_real_vulkan`

规则：

- 当前 op 在真实支持列表中且 dtype/shape 条件满足，则为 `True`
- 否则为 `False`

### 5.5.2 `runtime.py`

更新 `python/pypto_gpu_vk/runtime.py`：

默认行为：

1. 若 `compiled_op.supports_real_vulkan=False`
   - 若允许 dev fallback，则走 CPU fallback
   - 否则直接报错
2. 若支持真实 Vulkan
   - 调 `GpuVkRunner.run_artifact(...)`
   - 若成功，返回真实输出
   - 若失败：
     - 若允许 dev fallback，记录 warning 后 fallback
     - 否则抛异常

`available()` 只反映 Vulkan 真实 runtime 是否 ready，不再把 fallback 算作 available。

### 5.5.3 `converter.py`

更新 `python/pypto_gpu_vk/converter.py`：

首版真实执行仍保留 CPU staging converter，但其角色变为“真实 Vulkan 输入准备”，不是测试桩。

行为固定：

- `torch_to_vk_cpu_staging()` 只处理 CPU tensor
- 若输入 tensor 在 CUDA/NPU/其他 device：
  - 直接报错
  - 不做隐式设备搬移

## 6. Build 与环境接入

### 6.1 CMake

更新顶层 `CMakeLists.txt` 与 `python/src/CMakeLists.txt`：

- `find_package(Vulkan REQUIRED)` 或等效 include/lib 路径接入
- 保留 `BUILD_WITH_VULKAN`
- 新增 cache 变量：
  - `GPU_VK_GLSLANG_VALIDATOR`
- 当缺少 Vulkan headers/libs 时：
  - `BUILD_WITH_VULKAN=OFF`
  - Python 绑定仍可编译，但真实 runtime 相关 API 返回 `NOT_SUPPORTED`

### 6.2 依赖策略

本次不引入：

- `shaderc`
- in-process `glslang`

唯一新增工具依赖：

- `glslangValidator`

## 7. 实施顺序

### 第 1 步：真实编译链先打通

修改：

- `framework/src/gpu_vk/codegen/spirv_compiler.cpp`
- `framework/src/gpu_vk/codegen/gpu_vk_artifact.h`
- `python/src/bindings/gpu_vk_codegen.cpp`

结果：

- `compile_op("add", ...)` 可产出真实 `.spv`
- 暂不要求能执行

### 第 2 步：真实 Vulkan 基础上下文与 buffer

修改：

- `vk_instance.*`
- `vk_device.*`
- `vk_buffer.*`
- `vk_allocator.*`

结果：

- 能创建设备
- 能分配 staging/storage buffer
- 能完成 upload/download

### 第 3 步：真实 elementwise 执行

修改：

- `vk_descriptor_cache.*`
- `vk_pipeline_cache.*`
- `vk_runner.*`
- `glsl_codegen.cpp`
- Python runtime binding

结果：

- `add/mul/relu/where` 默认走真实 Vulkan
- 第二次执行命中 cache

### 第 4 步：真实 matmul 执行

修改：

- `dispatch_lowering.*`
- `glsl_codegen.cpp`
- `vk_runner.*`

结果：

- `matmul` 走 tile16 shared-memory path
- correctness 通过

### 第 5 步：收紧 fallback 与诊断

修改：

- `runtime.py`
- `config.py`
- 错误码与日志路径

结果：

- fallback 仅 dev-mode 可用
- 错误可定位

## 8. 测试与验收

### 8.1 Python UT

更新/新增：

- `python/tests/ut/gpu_vk/test_glsl_codegen.py`
- `python/tests/ut/gpu_vk/test_spirv_compile.py`
- `python/tests/ut/gpu_vk/test_runtime_cache.py`
- `python/tests/ut/gpu_vk/test_workgroup_tuner.py`
- `python/tests/ut/gpu_vk/test_torch_vk_interop.py`

### 8.2 Python E2E

新增真实 Vulkan e2e：

- `test_vk_add_e2e.py`
- `test_vk_mul_e2e.py`
- `test_vk_relu_e2e.py`
- `test_vk_where_e2e.py`
- `test_vk_matmul_e2e.py`

每个测试要求：

- 默认 `fallback_to_torch=False`
- 结果与 torch 对比
- `atol=1e-6`
- `rtol=1e-6`

### 8.3 失败场景测试

必须覆盖：

- 没有 Vulkan device
- 没有 `glslangValidator`
- GLSL 编译失败
- pipeline 创建失败
- 输入 shape 非法
- `matmul` 维度不匹配
- fallback 关闭时的报错路径

### 8.4 设备依赖测试分层

分两层：

无设备也能跑：

- IR / pass / GLSL / tool lookup / compile error path

需要 Vulkan 设备：

- 真正 dispatch 的 E2E

## 9. 验收门槛

完成时必须同时满足：

- `compile_op(...).artifact.isRealSpirv == True`
- `VkRuntime(...).available() == True` 时默认路径不经过 CPU fallback
- `add/mul/relu/where/matmul` 真实运行输出正确
- `pipeline_cache_hit_count()` 第二次调用增长
- 没有工具/设备时错误信息可直接指导修复
- 现有 Python facade 调用方式保持一致

## 10. 显式假设与默认值

- 目标平台默认为 Linux x86_64 桌面 Vulkan 环境
- 首版真实执行只支持 `float32`
- 首版只支持 CPU tensor 输入
- 首版不做 broadcast
- 首版不做 batched matmul
- 首版不做 external memory
- `glslangValidator` 为唯一 SPIR-V 编译后端
- CPU fallback 仅开发调试可用，默认关闭
- `matmul` 首版采用 `16x16` shared-memory tile，`K tile = 16`
