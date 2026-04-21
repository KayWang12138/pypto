# PyPTO 适配 CANN Mobile 编译修改记录

## 背景

PyPTO 需要适配端侧 CANN Mobile 包进行编译。CANN Mobile 与云侧 CANN 的主要差异是**缺少 HCCL（分布式通信）模块**——不提供 `hccl/` 头文件目录和 `libhccl.so` 库文件。

## 环境信息

- 编译环境：Linux x86 Docker (`pypto-dev`)
- CANN Mobile 路径：`/opt/cann_mobile/`
- CANN Mobile 版本：`cann-8.5.0`
- 编译命令：
```bash
export ASCEND_PATH=/opt/cann_mobile/
export PTO_TILE_LIB_CODE_PATH=/opt/pto-isa/
export PYPTO_THIRD_PARTY_PATH=/workspace/third_party_path/
source /opt/cann_mobile/ascend-toolkit/latest/set_env.sh
export LD_LIBRARY_PATH=/opt/cann_mobile/ascend-toolkit/latest/x86_64-linux/simulator/Kirin9030/lib/:$LD_LIBRARY_PATH
PYPTO_BUILD_EXT_ARGS='--cmake-build-type Release --cmake-options "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF"' python3 -m pip install -e . --verbose
```

## 遇到的问题及修复

### 问题 1：缺少 HCCL 头文件（编译错误）

**错误信息：**
```
/workspace/framework/src/adapter/api/hcomm_api.cpp:20: fatal error: hccl/hccl_types.h: No such file or directory
```

**原因：** `hcomm_api.cpp` 在 `#ifdef BUILD_WITH_CANN` 条件块中 `#include "hccl/hccl_types.h"` 和 `"hccl/hccl_rank_graph.h"`，CANN Mobile 不提供这些头文件。

**修复文件：** `framework/src/adapter/api/hcomm_api.cpp`

**修复方式：** 将文件中所有 `#ifdef BUILD_WITH_CANN` 替换为 `#if defined(BUILD_WITH_CANN) && !defined(BUILD_WITH_CANN_MOBILE)`。涉及 9 处修改：

- 头文件 include 守卫
- 7 个 HCCL 函数实现（HcommGetCommName, HcommGetL0TopoTypeEx, HcommGetCommHandleByGroup, HcommGetRootInfo, HcommCommInitRootInfo, HcommCommDestroy, HcommAllocComResourceByTiling）
- static_assert 类型校验块

### 问题 2：HCCL 符号未定义（链接错误）

**错误信息：**
```
OSError: libtile_fwk_runtime.so: undefined symbol: HcommGetCommHandleByGroup
```

**原因：** `framework/src/machine/runtime/distributed/distributed_context.cpp` 等模块调用 HCCL 函数，这些符号定义在 `hcomm_api.cpp` 中。CANN Mobile 下 HCCL 函数走 stub 实现，符号仍然存在，问题自然解决。

### 新增：传递 BUILD_WITH_CANN_MOBILE 编译定义

**修复文件：** `framework/src/adapter/CMakeLists.txt`

**修复方式：** 在 `target_compile_definitions` 中添加 `BUILD_WITH_CANN_MOBILE` 宏传递，使源文件可以使用该宏进行条件编译。

## CANN Mobile 与云侧 CANN 差异确认

| 组件 | 云侧 CANN | CANN Mobile | 状态 |
|------|-----------|-------------|------|
| `libascendcl.so` | ✅ | ✅ | 路径一致 |
| `libruntime.so` | ✅ | ✅ | 路径一致 |
| `libascend_hal.so` | ✅ | ✅ | 路径一致 |
| `libprofapi.so` | ✅ | ✅ | 路径一致 |
| `libascend_dump.so` | ✅ | ✅ | 路径一致 |
| `libhccl.so` | ✅ | ❌ | **缺失** — 分布式通信不可用 |
| `include/hccl/` | ✅ | ❌ | **缺失** — HCCL 头文件不存在 |
| 目录结构 (`lib64/`, `include/`, `pkg_inc/`) | ✅ | ✅ | 基本一致 |

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `CMakeLists.txt` | 新增编译开关 | 添加 `BUILD_WITH_CANN_MOBILE` option（ON） |
| `framework/src/adapter/api/hcomm_api.cpp` | 源码修改 | HCCL 代码用 `BUILD_WITH_CANN_MOBILE` 条件排除，走 stub 路径 |
| `framework/src/adapter/CMakeLists.txt` | 构建修改 | 添加 `BUILD_WITH_CANN_MOBILE` 编译定义传递 |

## 验证结果

- ✅ 编译成功（所有 target 构建通过）
- ✅ `import pypto` 成功
- ✅ `pypto.Tensor`、`pypto.Element`、`pypto.op` 核心类可正常使用
