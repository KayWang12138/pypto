# Hello World 端到端流程详解

本文档以 `examples/00_hello_world/hello_world.py` 为例，详细梳理从 Python 代码到 C++ 实现的完整执行流程。

## 示例代码概览

```python
@pypto.frontend.jit(runtime_options={"run_mode": mode})
def add_kernel(
    x: pypto.Tensor(shape, pypto.DT_FP32),
    y: pypto.Tensor(shape, pypto.DT_FP32),
) -> pypto.Tensor(shape, pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out = x + y
    return out

# 调用
output_data = add_kernel(input_data0, input_data1)
```

## 完整流程图

```mermaid
flowchart TD
    Start([Python: hello_world.py<br/>main函数]) --> ParseArgs[解析命令行参数<br/>--run_mode npu/sim]
    ParseArgs --> GetDeviceId{获取设备ID<br/>TILE_FWK_DEVICE_ID}
    GetDeviceId -->|NPU模式| SetNPUDevice[设置NPU设备<br/>torch.npu.set_device]
    GetDeviceId -->|SIM模式| SkipDevice[跳过设备设置]
    SetNPUDevice --> CreateKernel[create_add_kernel<br/>创建JIT函数]
    SkipDevice --> CreateKernel
    
    CreateKernel --> Decorate[应用@pypto.frontend.jit装饰器<br/>entry.py:jit]
    Decorate --> CreateWrapper[创建JitCallableWrapper<br/>延迟编译]
    CreateWrapper --> CallKernel[调用add_kernel<br/>input_data0, input_data1]
    
    CallKernel --> ValidateInputs[验证输入参数<br/>检查是否为torch.Tensor]
    ValidateInputs --> CheckCompiled{是否已编译?<br/>_is_compiled}
    
    CheckCompiled -->|否| Compile[编译阶段]
    CheckCompiled -->|是| Execute[执行阶段]
    
    Compile --> CreateParser[创建Parser实例<br/>_create_parser]
    CreateParser --> ParseAST[解析Python AST<br/>parser.parse]
    ParseAST --> DeviceInit[初始化设备<br/>pypto_impl.DeviceInit]
    DeviceInit --> OperatorBegin[开始操作符<br/>pypto_impl.OperatorBegin<br/>返回handler]
    OperatorBegin --> SetOptions[设置配置选项<br/>_set_config_option]
    SetOptions --> BindDims[绑定动态维度<br/>bind_dynamic_dims_from_inputs]
    BindDims --> ExecuteParse[执行解析<br/>parser.execute<br/>生成PTO IR]
    ExecuteParse --> OperatorEnd[结束操作符<br/>pypto_impl.OperatorEnd]
    OperatorEnd --> MarkCompiled[标记已编译<br/>_is_compiled = True]
    MarkCompiled --> Execute
    
    Execute --> AllocOutputs[分配输出张量<br/>根据签名创建torch.empty]
    AllocOutputs --> ConvertTensors[转换张量<br/>torch.Tensor -> pypto.Tensor]
    ConvertTensors --> Dispatch[根据运行模式分发<br/>_dispatch_with_run_mode]
    
    Dispatch --> CheckMode{运行模式?<br/>run_mode}
    CheckMode -->|NPU=0| RunNPU[NPU执行路径]
    CheckMode -->|SIM=1| RunSIM[SIM执行路径]
    
    RunNPU --> CheckCANN{CANN环境?<br/>ASCEND_HOME_PATH}
    CheckCANN -->|否| ErrorCANN[抛出错误<br/>需要配置CANN环境]
    CheckCANN -->|是| RunWithNPU[_run_with_npu]
    
    RunWithNPU --> ConvertToData[转换为DeviceTensorData<br/>_pto_to_tensor_data]
    ConvertToData --> GetWorkspace[获取工作空间大小<br/>GetWorkSpaceSize]
    GetWorkspace --> AllocWorkspace[分配工作空间<br/>torch.empty workspace]
    AllocWorkspace --> CallCpp[调用C++绑定<br/>OperatorDeviceRunOnceDataFromDevice<br/>runtime.cpp]
    
    RunSIM --> RunWithCPU[_run_with_cpu]
    RunWithCPU --> CostModel[调用代价模型<br/>_cost_model_run_once_data_from_host]
    CostModel --> Emulation[仿真执行<br/>EmulationLauncher]
    
    CallCpp --> CppEntry[C++入口<br/>runtime.cpp::<br/>OperatorDeviceRunOnceDataFromDevice]
    CppEntry --> GetOperator[获取操作符<br/>ExportedOperator]
    GetOperator --> GetFunction[获取Function对象]
    GetFunction --> CheckProfile{性能分析?<br/>PROFILE_ENABLE}
    
    CheckProfile -->|是| EmulationLaunch[仿真启动<br/>EmulationLauncher::<br/>EmulationLaunchDeviceTensorData]
    CheckProfile -->|否| DeviceLaunch[设备启动]
    EmulationLaunch --> DeviceLaunch
    
    DeviceLaunch --> GetStreams[获取设备流<br/>aicpuStream, aicoreStream]
    GetStreams --> DeviceLaunchOnce[DeviceLauncher::<br/>DeviceLaunchOnceWithDeviceTensorData]
    
    DeviceLaunchOnce --> InitEnv[环境初始化]
    InitEnv --> SetRunType[设置运行类型: npu]
    SetRunType --> SetBinData[设置Kernel二进制数据]
    SetBinData --> SetCapture[设置捕获流<br/>SetCaptureStream]
    SetCapture --> ACLInit[ACL初始化<br/>aclInit]
    ACLInit --> CheckCache{缓存操作符?}
    
    CheckCache -->|是| UseCache[使用缓存的配置数据]
    CheckCache -->|否| InitKArgs[初始化Kernel参数]
    UseCache --> InitKArgs
    
    InitKArgs --> CheckDevice[检查设备ID<br/>CheckDeviceId]
    CheckDevice --> FillDeviceInfo[填充设备信息<br/>DeviceLauncherConfigFillDeviceInfo]
    FillDeviceInfo --> InitTiling[初始化Tiling数据<br/>DeviceInitTilingData<br/>使用DeviceMemoryUtils]
    InitTiling --> SetCacheKernel[设置缓存Kernel<br/>DeviceRunCacheKernelSet]
    SetCacheKernel --> InitIO[初始化输入输出<br/>DeviceInitKernelInOuts]
    
    InitIO --> RegisterBin[注册Kernel二进制<br/>RegisterKernelBin]
    RegisterBin --> DynamicLaunch[动态启动<br/>DeviceRunner::DynamicLaunch]
    
    DynamicLaunch --> InitAicpuSo[初始化AICPU SO二进制<br/>首次执行]
    InitAicpuSo --> PrepareArgs[准备设备参数<br/>DeviceArgs]
    PrepareArgs --> CopyArgs[复制参数到设备内存<br/>rtMemcpy]
    CopyArgs --> RunPrepare[执行准备阶段<br/>RunPrepare]
    RunPrepare --> LaunchInit[启动AICPU初始化<br/>launchDynamicAiCpuInit]
    LaunchInit --> LaunchCpu[启动AICPU任务<br/>launchDynamicAiCpu]
    LaunchCpu --> LaunchCore[启动AICore任务<br/>launchDynamicAiCore]
    
    LaunchCore --> RunProfile[运行性能分析<br/>RunWithProfile]
    RunProfile --> Sync[流同步<br/>DynamicLaunchSynchronize]
    Sync --> ReturnResult[返回结果<br/>torch.Tensor]
    
    ReturnResult --> Verify[验证结果<br/>与golden对比]
    Verify --> End([结束])
    
    ErrorCANN --> End
    
    style Start fill:#e1f5ff
    style End fill:#ffe1f5
    style Compile fill:#fff4e1
    style Execute fill:#e1ffe1
    style DeviceLaunch fill:#ffe1f5
    style DynamicLaunch fill:#fff4e1
    style LaunchInit fill:#e1ffe1
    style LaunchCpu fill:#e1ffe1
    style LaunchCore fill:#e1ffe1
```

## 详细阶段说明

### 阶段1: Python入口 (hello_world.py)

**文件**: `examples/00_hello_world/hello_world.py`

1. **main()** (112-222行)
   - 解析命令行参数
   - 获取设备ID（NPU模式）
   - 调用测试函数

2. **test_add_direct()** (89-109行)
   - 准备输入数据（torch.rand）
   - 创建kernel函数
   - 调用kernel并验证结果

3. **create_add_kernel()** (68-86行)
   - 根据run_mode设置运行模式
   - 使用`@pypto.frontend.jit`装饰器
   - 返回装饰后的函数

### 阶段2: JIT装饰器处理 (frontend/parser/entry.py)

**文件**: `python/pypto/frontend/parser/entry.py`

1. **jit()** (744-843行)
   - 装饰器工厂函数
   - 创建`JitCallableWrapper`实例
   - **延迟编译**：不立即编译，等待第一次调用

2. **JitCallableWrapper.__init__()** (153-214行)
   - 保存原始函数和配置选项
   - 初始化状态：`_is_compiled = False`
   - 捕获局部变量（用于闭包）

### 阶段3: 函数调用 (frontend/parser/entry.py)

**文件**: `python/pypto/frontend/parser/entry.py`

1. **JitCallableWrapper.__call__()** (216-332行)
   - 验证输入参数（必须是torch.Tensor）
   - 检查张量连续性
   - 解析符号维度
   - 分配输出张量
   - 调用`_compile_if_needed()`
   - 调用`_dispatch_with_run_mode()`

### 阶段4: 编译阶段 (frontend/parser/entry.py)

**文件**: `python/pypto/frontend/parser/entry.py`

1. **_compile_if_needed()** (485-532行)
   - 检查是否已编译
   - 创建Parser实例
   - 调用`parser.parse()`解析AST
   - 初始化后端：`pypto_impl.DeviceInit()`
   - 开始操作符：`pypto_impl.OperatorBegin()`
   - 设置配置选项
   - 绑定动态维度
   - 执行解析：`parser.execute()` → 生成PTO IR
   - 结束操作符：`pypto_impl.OperatorEnd()`
   - 标记已编译

2. **Parser.parse()** (parser/parser.py)
   - 使用访问者模式遍历AST
   - 解析`pypto.set_vec_tile_shapes()`调用
   - 解析`x + y`表达式
   - 生成PTO中间表示（IR）

### 阶段5: 执行分发 (frontend/parser/entry.py)

**文件**: `python/pypto/frontend/parser/entry.py`

1. **_dispatch_with_run_mode()** (650-686行)
   - 检查运行模式（NPU=0 或 SIM=1）
   - NPU模式：调用`_run_with_npu()`
   - SIM模式：调用`_run_with_cpu()`

2. **_run_with_npu()** (577-616行)
   - 转换为DeviceTensorData
   - 获取工作空间大小
   - 分配工作空间
   - 调用C++绑定：`pypto_impl.OperatorDeviceRunOnceDataFromDevice()`

3. **_run_with_cpu()** (618-635行)
   - 调用代价模型接口
   - `_cost_model_run_once_data_from_host()`

### 阶段6: Python到C++绑定 (bindings/runtime.cpp)

**文件**: `python/src/bindings/runtime.cpp`

1. **OperatorDeviceRunOnceDataFromDevice()** (108-203行)
   - 获取ExportedOperator
   - 获取Function对象
   - 验证输入输出数量
   - 如果启用性能分析，先执行仿真
   - 获取设备流（aicpuStream, aicoreStream）
   - 调用`DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData()`

### 阶段7: C++设备启动 (machine/runtime/device_launcher.cpp)

**文件**: `framework/src/machine/runtime/device_launcher.cpp`

1. **DeviceLaunchOnceWithDeviceTensorData()** (125-183行)
   - 环境初始化
   - 设置捕获流
   - ACL初始化
   - 处理缓存操作符
   - 初始化Kernel参数
   - 注册Kernel二进制
   - 调用`DeviceRunner::DynamicLaunch()`

### 阶段8: 动态启动 (machine/runtime/device_runner.cpp)

**文件**: `framework/src/machine/runtime/device_runner.cpp`

1. **DynamicLaunch()** (711-762行)
   - 初始化AICPU SO二进制（首次）
   - 准备设备参数（DeviceArgs）
   - 复制参数到设备内存
   - 执行准备阶段
   - 调用`DynamicKernelLaunch()`

2. **DynamicKernelLaunch()** (637-662行)
   - `launchDynamicAiCpuInit()` - 初始化AICPU
   - `launchDynamicAiCpu()` - 启动AICPU任务
   - `launchDynamicAiCore()` - 启动AICore任务

3. **性能分析和同步**
   - `RunWithProfile()` - 收集性能数据
   - `DynamicLaunchSynchronize()` - 等待任务完成

## 关键函数调用链

### NPU模式完整调用链

```
hello_world.py::main()
  └─> test_add_direct()
      └─> create_add_kernel() [装饰器应用]
          └─> @pypto.frontend.jit
              └─> JitCallableWrapper.__init__()
      
      └─> add_kernel(input_data0, input_data1) [函数调用]
          └─> JitCallableWrapper.__call__()
              ├─> _compile_if_needed() [首次调用]
              │   ├─> Parser.parse() [解析AST]
              │   ├─> pypto_impl.DeviceInit()
              │   ├─> pypto_impl.OperatorBegin()
              │   ├─> parser.execute() [生成PTO IR]
              │   └─> pypto_impl.OperatorEnd()
              │
              └─> _dispatch_with_run_mode()
                  └─> _run_with_npu()
                      └─> pypto_impl.OperatorDeviceRunOnceDataFromDevice()
                          └─> runtime.cpp::OperatorDeviceRunOnceDataFromDevice()
                              └─> DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData()
                                  └─> DeviceRunner::DynamicLaunch()
                                      └─> DynamicKernelLaunch()
                                          ├─> launchDynamicAiCpuInit()
                                          ├─> launchDynamicAiCpu()
                                          └─> launchDynamicAiCore()
```

### SIM模式完整调用链

```
hello_world.py::main()
  └─> test_add_direct()
      └─> add_kernel(input_data0, input_data1)
          └─> JitCallableWrapper.__call__()
              └─> _dispatch_with_run_mode()
                  └─> _run_with_cpu()
                      └─> _cost_model_run_once_data_from_host()
                          └─> cost_model.cpp::CostModelRunOnceDataFromHost()
                              └─> CostModelLauncher::CostModelRunOnce()
                                  └─> EmulationLauncher::EmulationRunOnce()
                                      └─> EmulationLaunchOnce()
                                          └─> DynTileFwkBackendKernelServer()
```

## 关键数据结构

### Python层
- `torch.Tensor` - PyTorch张量
- `pypto.Tensor` - PTO张量包装
- `DeviceTensorData` - 设备张量数据（Python绑定）

### C++层
- `DeviceTensorData` - 设备张量数据（C++）
- `DeviceKernelArgs` - Kernel参数
- `DevAscendProgram` - 设备程序
- `Function` - PTO函数对象

## 内存管理

### NPU模式
- **设备内存**: 使用`DeviceMemoryUtils()`管理NPU设备内存
- **工作空间**: 动态分配工作空间
- **流管理**: 使用`rtStream_t`管理异步执行

### SIM模式
- **主机内存**: 使用`EmulationMemoryUtils()`管理主机内存
- **线程模型**: 多线程模拟AICPU执行

## 配置选项流转

```
runtime_options={"run_mode": mode}
  └─> JitCallableWrapper._runtime_options
      └─> _set_config_option()
          └─> pypto.set_runtime_options()
              └─> config::SetRuntimeOption()
                  └─> C++层配置系统
```

## 性能分析流程

```
PROFILE_ENABLE = True
  └─> EmulationLauncher::EmulationLaunchDeviceTensorData() [仿真执行]
      └─> DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData() [NPU执行]
          └─> RunWithProfile() [收集性能数据]
              └─> 生成泳道图
```

## 错误处理

1. **编译阶段错误**: 在Parser中捕获，返回诊断信息
2. **运行时错误**: 在C++层返回错误码，Python层转换为异常
3. **设备错误**: ACL错误码转换为Python异常

## 缓存机制

1. **编译缓存**: `_is_compiled`标志避免重复编译
2. **Kernel缓存**: `DeviceRunCacheKernelEnable()`支持Kernel重用
3. **控制流缓存**: `BuildControlFlowCache()`缓存控制流信息

## 总结

整个流程从Python装饰器开始，经过AST解析、IR生成、编译优化，最终在NPU设备或仿真环境中执行。关键特点是：

1. **延迟编译**: 第一次调用时才编译
2. **动态形状**: 支持运行时确定张量形状
3. **多模式执行**: 支持NPU硬件和CPU仿真
4. **性能分析**: 集成性能数据收集
5. **缓存优化**: 多层缓存机制提升性能
