# 🚀 PyPTO 上手指南

> **目标读者：** 第一次使用 PyPTO 的用户与开发者  
> **学习目标：** 跑通第一个算子、知道常用配置/调试入口、需要时能从源码构建与跑测试

---

## 快速导航 🧭

- **先跑通**：从[第1步：确认环境](#第1步确认环境-)开始
- **看结果与产物**：见[第4步：查看结果与产物 📄](#第4步查看结果与产物-)
- **常用配置**：见[第5步：常用配置速查 ⚙️](#第5步常用配置速查-)
- **遇到问题**：见[第6步：出现问题再排查 🧰](#第6步出现问题再排查-)
- **需要从源码构建/跑测试**：见[从源码构建与跑测试](00-getting-started/02-build-and-test.md)

---

## 第1步：确认环境 ✅

这一节只做两件事：确认 **PyPTO 可导入**，以及当前机器是否具备 **NPU 运行条件**（否则默认会走 SIM）。

```bash
# 检查PyPTO是否已安装
python3 -c "import pypto; print('PyPTO已安装')"

# 检查 NPU 是否可用（如果在 NPU 设备上）
# 说明：仿真环境通常不安装 torch_npu；这里用"可选导入"避免直接报错
python3 -c "import importlib; m=importlib.util.find_spec('torch_npu'); print('torch_npu 已安装' if m else '未安装 torch_npu，将使用SIM模式')"
```

**如果报错：** 请先完成 note 内的环境准备与安装：见[环境准备与安装](00-getting-started/01-environment-setup.md)

**重要说明：**
- **安装 ≠ 可用**：`torch_npu` 已安装不代表 NPU 环境已就绪，还需要：
  - CANN 环境已正确安装并 `source setenv.sh`（真实 NPU 环境）
  - 环境变量（如 `LD_LIBRARY_PATH`）已正确设置
  - 设备驱动与固件已安装（真实 NPU 环境）
- **详细检查方法：** 参考[环境准备与安装](00-getting-started/01-environment-setup.md)中的 CANN 安装与验证章节

### （强烈建议）安全运行与权限最小化

为避免权限风险与产物泄露，建议在开始跑样例/产线任务前先确认：

- **不要使用 root 账户运行**：尽量使用普通用户账号；按需对输出目录/设备访问做最小授权。
- **设置更严格的默认权限**：建议 `umask >= 0027`（新建目录最高 `750`，文件最高 `640`）。

常见场景的权限建议（上限值）：

| 类型 | Linux 权限建议最大值 |
|---|---|
| 用户主目录 | `750` |
| 程序文件（含脚本/库） | `550` |
| 程序目录 | `550` |
| 配置文件 | `640` |
| 配置目录 | `750` |
| 日志文件（正在记录） | `640` |
| 日志文件（归档后） | `440` |
| 日志目录 | `750` |
| Debug 文件 | `640` |
| Debug 目录 | `750` |
| 临时文件目录 | `750` |
| 私钥/证书/密文文件 | `600` |
| 私钥/证书/密文目录 | `700` |

补充说明：
- 源码编译会生成中间文件；编译完成后也建议对 `build/`、产物目录做权限收敛。
- 运行异常退出时通常会打印报错信息并生成日志；排查问题时优先在可控权限目录下收集 `run.log` 等产物。

---

## 第2步：编写第一个程序 📝

目标是用最小代码跑通一次 `@pypto.jit`：输入 `torch.Tensor` → `pypto.from_torch()` 转换 → `z[:] = ...` 写回输出。

创建文件 `hello_pypto.py`：

```python
import pypto
import torch

@pypto.jit
def add_vectors(x, y, z):
    """向量加法：z = x + y"""
    pypto.set_vec_tile_shapes(16)
    z[:] = x + y

x = torch.ones(16, dtype=torch.float32)
y = torch.ones(16, dtype=torch.float32)
z = torch.empty(16, dtype=torch.float32)

add_vectors(
    pypto.from_torch(x),
    pypto.from_torch(y),
    pypto.from_torch(z),
)

print("输出 z:", z.numpy())
```

参考：
- 相关 API 名称与用法模式见：[API 使用总结（含全量目录附录）](02-core/02-api-reference.md)

---

## 第3步：运行程序 ▶️

```bash
python3 hello_pypto.py
```

---

## 第4步：查看结果与产物 📄

跑通之后建议马上做两件事：**确认结果正确**，以及 **知道本次运行的日志/产物在哪里**，便于后续排查与对比。

### 1) 确认结果

如果你的输入是全 1（如示例代码），输出 `z` 应该是全 2；若结果不符合预期，请直接跳到[第6步：出现问题再排查](#第6步出现问题再排查-)。

### 2) 找到本次输出目录（重点 `run.log`）

PyPTO 默认会在当前目录创建 `output/output_<timestamp>_<pid>/`，并生成 `run.log` 等产物。你也可以指定输出目录：

```bash
export TILE_FWK_OUTPUT_DIR=/tmp/pypto_output

# 查看最新一次输出（默认目录）
ls -dt output/output_* | head -1
```

建议先把产物体系串起来（后续排查/性能分析会高频用到）：

- [输出目录与产物总览](03-mechanisms/output-files/README.md)
- [run.log 文件详细说明](03-mechanisms/output-files/run-log.md)

### 3) （可选）通过可视化查看计算图/泳道图

如果你在使用 PyPTO 配套可视化工具（如 VSCode 插件），建议先把“产物是什么/怎么看”理解清楚：

- 术语解释见：[术语对照表](TERMINOLOGY.md)
- 插件安装与可视化入口见：[环境准备与安装](00-getting-started/01-environment-setup.md#7-产物可视化pypto-toolkitvscode-插件)

---

## 第5步：常用配置速查 ⚙️

进入“能用”之后，最常遇到两个问题：**在哪跑（NPU/SIM）**、以及 **怎么调 Tiling**。这两项优先掌握即可。

### 1) 运行模式：NPU / SIM

```python
import pypto

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x, y, z):
    z[:] = x + y
```

参考：
- API 索引见：[API 使用总结（含全量目录附录）](02-core/02-api-reference.md)

### 2) Tiling：先掌握 Vector Tiling

要点：
- `pypto.set_vec_tile_shapes(*args)` 最多 4 维
- **最后一维需要按 dtype 满足 32Byte 对齐**

排障与问题库：见 [常见问题与已知问题库](05-debugging/03-troubleshooting-and-known-issues.md)

---

## 第6步：出现问题再排查 🧰

当结果不对/性能不符合预期时，优先做“看 `run.log` + 查排障”，不要一上来改复杂配置。

- [完整调试指南](05-debugging/00-complete-guide.md)
- [Segfault 调试实战案例](05-debugging/01-segfault-practice.md)
- [常见问题与已知问题库](05-debugging/03-troubleshooting-and-known-issues.md)

---

## 下一步：跑官方示例与进阶阅读 📖

到这里你已经能写并运行最小算子了。下一步建议“先跑示例，再读概念”，效率最高。

- **示例（推荐）**
  - [Hello World](01-examples/00-hello-world.md)
  - [Softmax 实战](01-examples/03-softmax.md)

- **进阶主题**
  - [环境准备与安装](00-getting-started/01-environment-setup.md)
  - [精度调试（流程与方法）](05-debugging/02-precision-debugging.md)
  - [常见问题与已知问题库](05-debugging/03-troubleshooting-and-known-issues.md)
  - [PyTorch 集成与接入](08-best-practices/02-pytorch-integration.md)
  - [API 使用总结（含全量目录附录）](02-core/02-api-reference.md)

- **理解框架核心概念**
  - [核心概念](02-core/01-concepts.md)
  - [框架总览](02-core/00-overview.md)

- **性能调优流程**
  - [性能优化指南](07-features/01-performance-optimization.md)

- **更多示例（仓库 examples/，不依赖外链的速览）**
  - **分层结构**：`01_beginner`（基础）→ `02_intermediate`（算子/NN/控制流）→ `03_advanced`（attention/模式/系统优化）→ `models`（大模型算子）
  - **运行前（NPU）常用环境**：

```bash
# 配置 CANN 环境变量（NPU 模式）
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置 NPU 设备 ID（运行 NPU 样例时通常必需）
export TILE_FWK_DEVICE_ID=0
```

  - **通用运行方式**（多数样例脚本都支持这些参数形态）：

```bash
# 直接运行脚本（部分样例默认 NPU）
python3 examples/01_beginner/basic/basic_ops.py

# 指定仿真模式运行（无需硬件）
python3 examples/01_beginner/basic/basic_ops.py --run_mode sim

# 列出脚本内所有可用用例
python3 examples/01_beginner/basic/basic_ops.py --list

# 运行某个特定用例（形如：<group>::<case>）
python3 examples/01_beginner/basic/basic_ops.py view_operations::test_view_operations
```

---

## 从源码构建与跑测试（可选）🛠️

如果你需要从源码编译 PyPTO 或运行测试，详见：[从源码构建与跑测试](00-getting-started/02-build-and-test.md)

该文档包含：
- 构建流程与命令
- 运行测试方法
- 常用构建开关与环境变量
- 构建与调试的关系

---

*如果你在阅读其它文档时遇到“前置知识/快速上手”的入口链接，统一以本文件为准。*

