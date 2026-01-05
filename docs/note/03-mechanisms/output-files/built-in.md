# built_in 目录详细说明

## 目录概述

`built_in` 目录包含 PyPTO 内置操作的元数据信息，用于运行时识别和调用内置操作。

**目录位置**：输出目录下的 `built_in/`（例如：`output/output_<timestamp>_<pid>/built_in/`）

## 目录结构

```
built_in/
└── pypto_op_info.json                                  # PyPTO 操作信息文件
```

## 文件详细说明

### pypto_op_info.json

**作用**：PyPTO 内置操作的元数据文件，定义了 PyPTO 框架内置操作的运行时信息。

**文件格式**：JSON

**文件内容结构**：

```json
{
  "操作名": {
    "opInfo": {
      "computeCost": "计算成本",
      "engine": "执行引擎",
      "flagAsync": "是否异步",
      "flagPartial": "是否部分执行",
      "functionName": "函数名",
      "kernelSo": "内核共享库",
      "opKernelLib": "操作内核库",
      "userDefined": "是否用户定义"
    }
  }
}
```

### 内置操作说明

#### 1. PyptoInit

**作用**：PyPTO 初始化操作

**配置**：
- `engine`: `DNN_VM_AICPU` - 使用 AICPU 引擎
- `functionName`: `DynPyptoKernelServerInit` - 初始化函数名
- `kernelSo`: `libtilefwk_backend_server.so` - 后端服务器共享库
- `opKernelLib`: `KFCKernel` - KFC 内核库

**用途**：在程序开始时初始化 PyPTO 运行时环境

#### 2. PyptoNull

**作用**：PyPTO 空操作（占位符）

**配置**：
- `engine`: `DNN_VM_AICPU` - 使用 AICPU 引擎
- `functionName`: `DynPyptoKernelServerNull` - 空操作函数名
- `kernelSo`: `libtilefwk_backend_server.so` - 后端服务器共享库
- `opKernelLib`: `AICPUKernel` - AICPU 内核库

**用途**：用于占位或调试

#### 3. PyptoRun

**作用**：PyPTO 运行操作

**配置**：
- `engine`: `DNN_VM_AICPU` - 使用 AICPU 引擎
- `functionName`: `DynPyptoKernelServer` - 运行函数名
- `kernelSo`: `libtilefwk_backend_server.so` - 后端服务器共享库
- `opKernelLib`: `KFCKernel` - KFC 内核库

**用途**：执行 PyPTO 内核计算

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `computeCost` | string | 计算成本（用于调度优化） |
| `engine` | string | 执行引擎类型 |
| `flagAsync` | string | 是否支持异步执行 |
| `flagPartial` | string | 是否支持部分执行 |
| `functionName` | string | 运行时调用的函数名 |
| `kernelSo` | string | 内核共享库文件名 |
| `opKernelLib` | string | 操作内核库名称 |
| `userDefined` | string | 是否为用户自定义操作 |

### 执行引擎类型

- `DNN_VM_AICPU`：AICPU 虚拟机引擎，用于执行控制流和复杂逻辑
- `DNN_VM_AICORE`：AICore 引擎，用于执行计算密集型操作

### 使用场景

1. **运行时加载**：运行时根据操作名查找对应的元数据
2. **函数调用**：根据 `functionName` 调用对应的运行时函数
3. **库加载**：根据 `kernelSo` 加载对应的共享库
4. **调度优化**：根据 `computeCost` 进行任务调度

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [输出目录与产物总览](./README.md)
- [run.log 文件详细说明](./run-log.md)
- [kernel_aicore 目录说明](./kernel-aicore.md)
- [kernel_aicpu 目录说明](./kernel-aicpu.md)

