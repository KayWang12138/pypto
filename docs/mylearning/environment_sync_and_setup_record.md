# PyPTO 环境同步与配置记录

**日期**: 2026-03-03  
**任务**: 同步仓库并配置 PyPTO 开发环境  
**执行人**: OpenCode AI Assistant  

---

## 📋 任务概述

将当前 PyPTO 仓库与原始仓库同步，并重新安装配置开发环境，确保基于最新版本运行。

### 任务目标
1. ✅ 保留当前工作分支的所有更改
2. ✅ 与原始仓库 (cann/pypto) 同步最新代码
3. ✅ 重新编译安装 PyPTO 环境
4. ✅ 配置 pto-isa 源码环境
5. ✅ 验证环境正确性

---

## 🔧 执行步骤

### 步骤 1: Git 仓库状态检查与暂存

**初始状态:**
```
On branch master
Changes not staged for commit:
  modified:   models/glm_v4_5/glm_attention_ifa_pfa.py

Untracked files:
  custom/
  docs/mylearning/optimization_workflow.md
  docs/mylearning/pfa_config_guide.md
  docs/mylearning/pfa_optimization_guide.md
  docs/mylearning/pfa_optimization_record.md
  docs/mylearning/readme.md
  docs/plans/
  models/glm_v4_5/glm_attention_ifa_pfa_opt_v1.py
  models/glm_v4_5/glm_attention_ifa_pfa_opt_v2.py
  models/glm_v4_5/test_pfa_performance.py
  tools/compare_swimlane.py
```

**执行命令:**
```bash
# 暂存所有更改（包括未跟踪文件）
git stash push -u -m "Save current changes before sync with upstream"
```

---

### 步骤 2: 添加原始仓库并同步

**添加 upstream 远程仓库:**
```bash
git remote add upstream https://gitcode.com/cann/pypto.git
git fetch upstream
```

**合并最新代码:**
```bash
# 设置 Git 用户信息（仅当前仓库）
git config user.email "developer@example.com"
git config user.name "Developer"

# 合并 upstream/master
git merge upstream/master -m "Merge upstream/master to sync with original repository"
```

**合并结果:**
- **变更文件**: 466 个文件
- **新增**: 30,987 行
- **删除**: 20,446 行
- **主要变更**:
  - 新增多个算子支持 (LReLU, exp2, expm1, gcd, log1p, prelu, remainder, var 等)
  - 新增分布式计算支持
  - 删除 IR Builder 相关代码
  - 重构控制流示例（controflow → controlflow）
  - 更新文档和测试用例

---

### 步骤 3: 恢复本地更改

**执行命令:**
```bash
git stash pop
```

**恢复结果:**
- ✅ 所有修改文件已恢复
- ✅ 所有未跟踪文件已恢复
- 当前分支领先 origin/master 75 个提交

---

### 步骤 4: 重新编译安装 PyPTO

**卸载旧版本:**
```bash
pip3 uninstall pypto -y
# 卸载版本: pypto 0.1.0
```

**编译新版本:**
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

**编译参数:**
- Python: 3.11.4
- CMake Build Type: Release
- Parallel Jobs: 29
- Backend: NPU

**编译产物:**
```
build_out/
├── pypto-0.1.1-cp311-cp311-linux_aarch64.whl  (8.4M)
├── pypto-0.1.1.tar.gz                          (1.6M)
└── pypto-0.1.0.tar.gz                          (1.7M) [旧版本]
```

**安装新版本:**
```bash
pip3 install build_out/pypto-0.1.1-cp311-cp311-linux_aarch64.whl --force-reinstall --no-deps
```

**验证安装:**
```bash
pip3 show pypto
# Version: 0.1.1
# Location: /home/developer/.local/lib/python3.11/site-packages
```

---

### 步骤 5: 配置 pto-isa 源码环境

#### 问题诊断

**初始测试失败:**
```bash
python3 examples/00_hello_world/hello_world.py --run_mode=npu
```

**错误信息:**
```
fatal error: 'pto/comm/pto_comm_inst.hpp' file not found
```

**根本原因:**
- 系统 CANN 版本: 9.0.0
- PyPTO 要求版本: 8.5.0
- 版本不匹配导致缺少头文件

#### 解决方案

**采用方案 2: 使用 pto-isa 源码（避免修改系统环境）**

**克隆 pto-isa 仓库:**
```bash
cd /mnt/workspace/gitCode/cann/mce
git clone https://gitcode.com/cann/pto-isa.git
```

**目录结构验证:**
```
pto-isa/
├── include/
│   └── pto/
│       ├── comm/          # ✅ 包含所需通信头文件
│       ├── common/        # ✅ 通用定义
│       ├── cpu/           # ✅ CPU 相关
│       ├── npu/           # ✅ NPU 相关
│       └── pto-inst.hpp   # ✅ 主安装文件
├── CMakeLists.txt
├── build.sh
└── README.md
```

**设置环境变量:**
```bash
# 当前会话
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa

# 持久化到 bashrc
echo 'export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa' >> ~/.bashrc
```

---

### 步骤 6: 环境验证

#### 测试 1: Hello World 示例（仿真模式）

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa
cd examples/00_hello_world
python3 hello_world.py --run_mode=sim
```

**输出结果:**
```
============================================================
PyPTO hello_world Example
============================================================

Running Example hello_world::test_add_direct: hello_world
Input0 shape: torch.Size([1, 4, 1, 64])
Input1 shape: torch.Size([1, 4, 1, 64])
Output shape: torch.Size([1, 4, 1, 64])
✓ Hello world example passed
```

**✅ 仿真模式测试通过**

---

#### 测试 2: Hello World 示例（NPU 模式）

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa
export TILE_FWK_DEVICE_ID=0
python3 hello_world.py --run_mode=npu
```

**输出结果:**
```
============================================================
PyPTO hello_world Example
============================================================

Running examples that require NPU hardware...
(Make sure CANN environment is configured and NPU is available)

Running Example hello_world::test_add_direct: hello_world
Input0 shape: torch.Size([1, 4, 1, 64])
Input1 shape: torch.Size([1, 4, 1, 64])
Output shape: torch.Size([1, 4, 1, 64])
Max difference: 0.000000
✓ Hello world example passed
```

**✅ NPU 模式测试通过**

---

#### 测试 3: 完整功能测试

```bash
cd examples/01_beginner/basic
python3 basic_ops.py --run_mode=npu
```

**测试项目:**
```
============================================================
Example 1: Tensor Creation
============================================================
  name=my_tensor, shape=[4, 4], dtype=DataType.DT_FP16, format=TileOpFormat.TILEOP_ND, dim=2
✓ Tensor creation completed successfully

============================================================
Example 2: Element-wise Operations
============================================================
  Max difference: 0.000000
✓ Element-wise operations completed successfully

============================================================
Example 3: Matrix Multiplication
============================================================
  Max difference: 0.000000
✓ Matrix multiplication completed successfully

============================================================
Example 4: Reduction Operations (sum)
============================================================
  Input:    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  Output:   [6.0, 15.0]
✓ Reduction operations completed successfully

============================================================
Example 5: Tiling Configuration
============================================================
  vec_tile_shapes set to (2, 8)
✓ Tiling configuration completed successfully

============================================================
Example 6: Transform Operations (view + assemble)
============================================================
  Max difference: 0.000000
✓ Transform operations completed successfully

============================================================
All examples completed successfully!
============================================================
```

**✅ 所有功能测试通过**

---

## 📊 最终环境配置

### 软件版本信息

| 组件 | 版本 | 安装路径 | 状态 |
|------|------|----------|------|
| **PyPTO** | 0.1.1 | `~/.local/lib/python3.11/site-packages` | ✅ 已安装 |
| **pto-isa** | 源码编译 | `/mnt/workspace/gitCode/cann/mce/pto-isa` | ✅ 已配置 |
| **CANN** | 9.0.0 | `/home/developer/Ascend/cann-9.0.0` | ✅ 兼容 |
| **Python** | 3.11.4 | `/usr/local/bin/python3` | ✅ 正常 |
| **PyTorch** | NPU版本 | `/opt/buildtools/Python-3.11.4/` | ✅ 正常 |

### 硬件环境

| 项目 | 信息 | 状态 |
|------|------|------|
| **NPU 设备** | 910B3 (Chip ID: 0) | ✅ 可用 |
| **NPU 状态** | Health: OK | ✅ 正常 |
| **温度** | 36°C | ✅ 正常 |
| **内存使用** | 3435MB / 65536MB | ✅ 正常 |

### 环境变量配置

```bash
# 必需环境变量
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa
export TILE_FWK_DEVICE_ID=0  # NPU 设备 ID

# CANN 环境（已在 bashrc 中配置）
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
```

**验证命令:**
```bash
# 验证环境变量
echo $PTO_TILE_LIB_CODE_PATH
# 输出: /mnt/workspace/gitCode/cann/mce/pto-isa

echo $TILE_FWK_DEVICE_ID
# 输出: 0

# 验证路径存在
ls $PTO_TILE_LIB_CODE_PATH/include/pto/comm/
# 应看到相关头文件
```

---

## 🚀 使用指南

### 快速开始

#### 方式 1: 运行单个示例

```bash
# 设置环境变量
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa
export TILE_FWK_DEVICE_ID=0

# 进入示例目录
cd examples/00_hello_world

# NPU 模式运行
python3 hello_world.py --run_mode=npu

# 仿真模式运行（无需 NPU）
python3 hello_world.py --run_mode=sim
```

#### 方式 2: 运行完整测试套件

```bash
# 运行基础操作示例
cd examples/01_beginner/basic
python3 basic_ops.py --run_mode=npu

# 运行计算算子示例
cd examples/01_beginner/compute
python3 elementwise_ops.py --run_mode=npu

# 运行控制流示例
cd examples/02_intermediate/controlflow/loop
python3 loop.py --run_mode=npu
```

### 开发流程

#### 1. 开发自定义算子

```bash
# 创建算子目录
cd custom/
mkdir my_operator
cd my_operator

# 创建算子文件
# my_operator.py  - 算子实现
# test_my_operator.py - 测试文件
# README.md - 文档
```

#### 2. 编译和测试

```bash
# 如果修改了 PyPTO 源码，重新编译
python3 build_ci.py -f python3 --disable_auto_execute
pip3 install build_out/pypto-*.whl --force-reinstall

# 运行测试
python3 test_my_operator.py --run_mode=npu
```

#### 3. 性能分析

```bash
# 运行性能分析
python3 my_operator.py --run_mode=npu

# 查看泳道图
# 输出路径: output/output_*/CostModelSimulationOutput/merged_swimlane.json
# 在 https://ui.perfetto.dev/ 中打开
```

---

## 📝 常见问题与解决方案

### 问题 1: 找不到 pto_comm_inst.hpp

**错误信息:**
```
fatal error: 'pto/comm/pto_comm_inst.hpp' file not found
```

**原因:** 
- 未设置 `PTO_TILE_LIB_CODE_PATH` 环境变量
- 或 CANN 版本不匹配

**解决方案:**
```bash
# 设置环境变量
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa

# 验证路径
ls $PTO_TILE_LIB_CODE_PATH/include/pto/comm/
```

---

### 问题 2: Invalid Device ID

**错误信息:**
```
RuntimeError: Invalid device ID
```

**原因:** 
- `TILE_FWK_DEVICE_ID` 设置不正确

**解决方案:**
```bash
# 检查可用 NPU 设备
npu-smi info

# 设置正确的设备 ID（通常是 0 或 1）
export TILE_FWK_DEVICE_ID=0
```

---

### 问题 3: Git 合并冲突

**场景:** 本地修改与 upstream 冲突

**解决方案:**
```bash
# 查看冲突文件
git status

# 手动解决冲突后
git add <conflicted_file>
git commit -m "Resolve merge conflicts"

# 如果需要重新开始
git merge --abort
```

---

### 问题 4: 编译失败

**可能原因:**
- 缺少依赖
- CMake 配置问题

**解决方案:**
```bash
# 清理构建
python3 build_ci.py -c

# 检查依赖
python3 -m pip install -r python/requirements.txt

# 详细输出模式
python3 build_ci.py -f python3 --disable_auto_execute --verbose
```

---

## 📚 参考资源

### 官方文档
- **PyPTO 文档中心**: https://pypto.gitcode.com
- **Git 仓库**: https://gitcode.com/cann/pypto
- **pto-isa 仓库**: https://gitcode.com/cann/pto-isa

### 本地文档
- **API 文档**: `docs/api/`
- **编程指南**: `docs/tutorials/`
- **安装指南**: `docs/install/`

### 示例代码
- **入门示例**: `examples/00_hello_world/`
- **初级示例**: `examples/01_beginner/`
- **中级示例**: `examples/02_intermediate/`
- **高级示例**: `examples/03_advanced/`
- **模型实现**: `models/`

---

## ✅ 任务完成确认

- [x] Git 仓库状态检查与暂存
- [x] 添加 upstream 远程仓库
- [x] 从 upstream/master 拉取最新代码
- [x] 合并最新代码到当前分支
- [x] 恢复本地更改
- [x] 卸载旧版本 PyPTO
- [x] 编译安装新版本 PyPTO (0.1.1)
- [x] 克隆 pto-isa 源码仓库
- [x] 配置 PTO_TILE_LIB_CODE_PATH 环境变量
- [x] 环境变量持久化到 bashrc
- [x] 验证仿真模式运行
- [x] 验证 NPU 模式运行
- [x] 完整功能测试
- [x] 编写归档文档

---

## 📌 后续建议

### 1. 环境管理
- 定期同步 upstream 仓库更新
- 保持 `PTO_TILE_LIB_CODE_PATH` 环境变量正确设置
- 关注 PyPTO 版本更新日志

### 2. 开发规范
- 遵循 `AGENTS.md` 中的开发规范
- 使用 `custom/` 目录存放自定义算子
- 编写完整的测试用例和文档

### 3. 性能优化
- 使用泳道图分析性能瓶颈
- 参考 `docs/tutorials/debug/matmul_performance_guide.md`
- 合理配置 Tiling 参数

### 4. 学习路径
- 从 `examples/00_hello_world` 开始
- 逐步学习 `01_beginner` → `02_intermediate` → `03_advanced`
- 参考 `models/` 中的大模型实现

---

## 📊 变更记录

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-03-03 | 1.0 | 初始版本 - 完成环境同步与配置 |

---

**文档维护**: 此文档应随环境变更及时更新  
**最后更新**: 2026-03-03 16:59  
**状态**: ✅ 环境配置完成并验证通过
