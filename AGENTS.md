# AGENTS.md

本文件为 PyPTO 算子开发项目级指令。

## 项目概述

本项目用于开发华为昇腾 AI 处理器（CANN PyPTO）自定义算子，支持完整的开发、测试及性能调优流程。

### 核心功能

- 使用 PyPTO 编程语言开发昇腾 AI 处理器自定义算子
- 提供完整的开发、构建、测试及性能调优工作流支持
- 遵循官方开发规范和性能优化最佳实践
- 支持批量生成
---

## Skills目录说明

本项目使用的skills位于 `.agents/skills/` 目录下，每个skill包含一个SKILL.md文件。
由于Skill工具无法直接从项目目录加载skills，开发时应：
1. 直接读取对应skill的SKILL.md文件内容
2. 将skill内容作为指导文档使用
3. 使用Task工具调用Explore和Plan Agent来执行相应任务

### 可用skills列表

#### 算子开发类
- `pypto-operator-develop-workflow`: PyPTO 算子开发工作流程

#### 性能分析类
- `pypto-operator-perf-analyzer`: 分析 PyPTO 算子的性能指标
- `pypto-operator-perf-autotuner`: PyPTO 算子性能分析和自动调优

#### 精度验证与调试类
- `pypto-operator-accuracy-verify`: PyPTO 算子精度验证
- `pypto-binary-search-verify`: 利用精度工具通过二分查找定位算子精度问题
- `pypto-binary-search-without-verify`: 不依赖精度工具的精度对比技能
- `pypto-aicore-error-locator`: 定位 aicore error 时的问题 CCE 文件

#### 环境与工具类
- `pypto-environment-setup`: PyPTO 环境安装与环境问题修复
- `gitcode-mcp-install`: 安装和配置 GitCode MCP Server

#### PR与代码质量类
- `pypto-pr-creator`: PyPTO 项目 PR 创建全流程指南
- `pypto-pr-fixer`: 修复 PyPTO PR 的 CodeCheck CI 失败和 review 评论
- `pypto-skill-reviewer`: 对 skill 目录进行质量与最佳实践合规性评审

#### Pass 开发类
- `pypto-pass/pypto-pass-module-analyzer`: PyPTO Pass 模块代码分析
- `pypto-pass/pypto-pass-ut-generate`: 根据Pass业务描述生成单元测试用例
- `pypto-pass/pypto-pass-workflow-analyzer`: PyPTO Pass 业务流分析

---

## 集成的技能列表
| 技能 | 触发时机 | 说明 |
| :--- | :--- | :--- |
| **算子开发** |||
| pypto-operator-develop-workflow | 接收到算子开发任务时 | 详细开发流程及环境准备、运行测试等，确保开发过程规范、高效 |
| **性能分析** |||
| pypto-operator-perf-analyzer | 需要分析算子性能指标时 | 从性能数据文件中提取关键指标，计算性能评级，提供优化建议 |
| pypto-operator-perf-autotuner | 接收到算子性能统计及调优指令时 | 生成泳道图、分析性能数据、查看性能统计和提供优化建议 |
| **精度验证与调试** |||
| pypto-operator-accuracy-verify | 需要验证算子精度时 | 验证 PyPTO 算子计算精度，提供多种容差配置和调试方法 |
| pypto-binary-search-verify | 需要调试算子精度问题时 | 利用精度工具通过二分查找快速定位算子精度问题 |
| pypto-binary-search-without-verify | 算子精度不满足要求时 | 通过添加检查点tensor进行原地修改，对比中间结果精度 |
| pypto-aicore-error-locator | 出现 aicore error 时 | 定位测试案例中出现 aicore error 时的问题 CCE 文件 |
| **环境与工具** |||
| pypto-environment-setup | 环境安装或环境问题修复时 | 包括CANN、torch_npu、编译工具链、第三方依赖和PyPTO编译运行等 |
| gitcode-mcp-install | 需要配置 GitCode MCP 时 | 安装和配置 GitCode MCP Server，使 AI 客户端能与 GitCode 平台交互 |
| **PR与代码质量** |||
| pypto-pr-creator | 需要创建 PR 时 | PR 创建全流程指南，包括仓库发现、分支创建、commit、PR创建等 |
| pypto-pr-fixer | PR CI 失败或需要修复评论时 | 修复 CodeCheck CI 失败和 review 评论 |
| pypto-skill-reviewer | 需要审计 skill 质量时 | 对 skill 目录进行质量与最佳实践合规性评审并评分 |
| **Pass 开发** |||
| pypto-pass-module-analyzer | 需要理解 Pass 模块时 | 分析 PyPTO pass 代码，生成模块分析文档 |
| pypto-pass-ut-generate | 需要生成 Pass 测试用例时 | 根据Pass业务描述，生成单元测试用例 |
| pypto-pass-workflow-analyzer | 需要理解 Pass 执行流程时 | 分析 PyPTO pass 文档中的业务流，帮助理解执行流程和数据流转 |
---
=======
- 编程框架：PyPTO
- 目标硬件：昇腾 AI 处理器（A3 服务器，CANN 8.5.0）
- 算子目录：`custom/{op}/`

---

## 核心原则

1. **文档优先** — 遇到问题先查 `docs/api/` 和 `examples/`，禁止凭直觉实现
2. **定位修复不推翻** — 定位问题点后修复该部分，禁止遇错推翻重写
3. **方案可用即完成** — 方案走通后即完成，不做额外优化探索

---

## 快速开始

算子开发使用 `pypto-op-orchestrator` 作为默认入口，它管理从需求到性能的完整流程。

---

## 开发技能系统

### 端到端工作流
| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-op-orchestrator` | 端到端算子开发编排 | **算子开发默认入口** |

### 阶段能力
| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-intent-understanding` | 需求理解，生成 spec.md | 需求分析阶段 |
| `pypto-golden-generator` | 生成 golden 参考实现 | spec 完成后 |
| `pypto-op-design` | 生成 design 文档 | golden 完成后 |
| `pypto-op-develop` | 实现 kernel + test + README | design 完成后 |
| `pypto-op-perf-autotuner` | 性能采集与调优 | 精度通过后 |
| `pypto-op-perf-analyzer` | 性能分析报告 | 性能数据采集后 |

### 调试辅助
| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-op-accuracy-verify` | 精度问题初步排查 | 精度不通过时（第一步） |
| `pypto-binary-search-verify` | verify-based 精度定位 | 精度不通过时（默认） |
| `pypto-binary-search-without-verify` | checkpoint-based 精度定位 | verify 方式不适用时 |

---

## 项目目录结构

所有算子必须放在 `custom/` 目录下，每个算子独立子目录：

**每个算子目录应包含：**

采用**eager模式**进行算子开发：

- **`.py` 文件**：包含完整的算子实现（必需）
- **`.py` 文件**：包含算子golden及测试用例（必需）
- **`README.md`**：编写算子文档（必须）
- **`data_utils.h`**：数据读写工具函数（可选，推荐包含）
- **`scripts/` 目录**：辅助脚本（可选）

### 核心开发规范 ⭐

> **严格遵循以下三条黄金法则，可避免 95% 的开发问题**

1. **理解官方示例原理后实现**
   - 生成一个测试及golden文件
   - 生成一个算子实现文件
   - 生成一个性能分析数据报告
2. **黄金法则：遇问题处理流程**
   - 第一步：直接搜索 `docs/` API 文档
   - 第二步：查阅官方示例 `examples/`
   - 第三步：定位问题点后修复，**禁止简化代码或推翻重写**
   - 第四步：完成功能即可，无需继续探索

## 分阶段开发指南
加载`pypto-operator-develop-workflow`技能，进行算子开发.

### 阶段一：需求分析与方案设计

1. **理解需求**：
   - 根据用户需求，设计算子的原型（输入、输出、数据类型）
   - 明确算子的数学公式（如：sinh(x) = (e^x - e^(-x)) / 2）

2. **API 可行性验证**：
   - 将公式拆解为 PyPTO API 的组合
   - 搜索 `docs/api/` 验证 API 存在性
   - 确认每个 API 支持的数据类型和约束

### 阶段二：算子实现

1. **从零实现算子**：
   - 参考 `examples/03_advanced/advanced_nn/attention/attention.py` 的代码结构
   - 创建 `custom/your_operator/` 目录
   - 按照设计方案实现算子逻辑

2. **核心实现**：
   - 实现文件中的 golden 函数
   - 实现文件中的 jit 函数
   - **一次只实现一个函数，立即验证编译**

3. **关键实现要点**：
   - 使用 `eager` 的方式进行开发
   - 可以先开发基础版本，再进行性能调试

### 阶段三：构建和测试

1. **测试准备**：
   - 生成覆盖多种场景的测试数据（边界值、零值、大值等）
   - 明确验证标准：误差容忍度、通过率等

2. **构建与执行**：

    **⚠️ 环境变量设置（非常重要）**
    ```bash
    # 设置NPU Chip ID（使用实际可用的chip）
    export TILE_FWK_DEVICE_ID=0
    export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
    ```
     **⚠️ 重要提示**：
    - **先设置 `export TILE_FWK_DEVICE_ID=0`**
    - 如果设置TILE_FWK_DEVICE_ID=0执行失败了，报错为`Invalid Device`时，再检查npu设备
    - 运行 `npu-smi info` 查看可用的NPU chip，进行设置`export TILE_FWK_DEVICE_ID=x`
    
    **⚠️ PTO_TILE_LIB_CODE_PATH 配置检查（关键）**：
    - **必须确保 `$PTO_TILE_LIB_CODE_PATH` 路径存在**
    - **必须确保 `$PTO_TILE_LIB_CODE_PATH/include/pto` 路径存在**，该路径下应包含 `pto_comm_inst.hpp` 等头文件
    - **如果路径不存在，需要查找正确的路径**：
      ```bash
      # 查找 include/pto 目录所在位置
      find /usr/local/Ascend -type d -name "pto" 2>/dev/null | grep include
      # 或者查找 pto_comm_inst.hpp 文件
      find /usr/local/Ascend -name "pto_comm_inst.hpp" 2>/dev/null
      ```
    - **找到正确路径后，设置 PTO_TILE_LIB_CODE_PATH 为包含 `include/pto` 的上级目录**
      - 例如：如果 `include/pto` 在 `/usr/local/Ascend/cann/aarch64-linux/include/pto`
      - 则设置：`export PTO_TILE_LIB_CODE_PATH=/usr/local/Ascend/cann/aarch64-linux`

   - 编译whl包并安装
     python3 build_ci.py -f python3 --disable_auto_execute
   - 执行算子进行验证
    **⚠️ 重要提示**：
    - **检查到存在npu卡的时候，必须使用run_mode=npu执行**

3. **验证与调试**：

   **Level 0~N 多级用例**：
   ```
   Level 0: 8-16 元素  ──▶ 基础功能验证
       ↓ 通过
   Level 1: 1K 元素     ──▶ 典型场景验证
       ↓ 通过
   Level 2: 极值/零值   ──▶ 边界情况验证
       ↓ 通过
   Level 3: 大数据量    ──▶ 性能验证
   ```
**✓ 检查清单**：
- [ ] 功能和精度验证通过
- [ ] 性能达到合理水平
- [ ] 多种输入规模测试通过

### 阶段四：开发结果总结

1. **总结开发情况**：
   - 编译运行成功，无错误无警告
   - 功能和精度验证通过
   - 性能验证结果（NPU 利用率、带宽利用率等）
   - 泛化验证结果（多种输入规模测试情况）
   - 记录已知限制（如 FP16 精度损失）

**✓ 检查清单**：
- [ ] 明确标注成功/失败
- [ ] 记录功能验证结果
- [ ] 记录精度测试结果
- [ ] 记录性能验证结果
- [ ] 记录泛化测试结果
- [ ] 记录已知限制和问题
- [ ] 如失败，总结失败原因

### 阶段五：编写算子文档

1. **编写 README.md**（必须使用中文）：
   - 记录数学公式和 API 映射关系
   - 参考标准写法：`examples/03_advanced/advanced_nn/attention/README.md`
   - 包含：
     - 算子概述（功能、数学公式）
     - 编译运行指南
     - 测试结果说明
     - 已知限制和注意事项
     - 常见问题

2. **更新当前状态至plan中**：`custom/plan/{算子名称}.md`

### 阶段六：性能数据采集及优化建议
加载`pypto-operator-perf-autotuner`技能进行性能分析。
1. **采集性能数据，并进行性能分析**
2. **生成性能分析报告**
**⚠️ 重要提示**：
性能分析报告不要过于复杂，列举出性能数据、分析汇总、性能优化建议即可。
=======
```text
custom/{op}/
├── spec.md              # 需求规格
├── design.md            # 设计方案
├── {op}_golden.py       # 纯 torch 参考实现，导出 {op}_golden()
├── {op}_impl.py         # PyPTO kernel 实现，导出 {op}_wrapper()
├── test_{op}.py         # 测试入口（import golden + impl，assert_allclose）
├── README.md            # 算子文档
├── .orchestrator_state.json  # 状态持久化
└── output/              # 性能数据
```

---

## 环境关键提示

```bash
export TILE_FWK_DEVICE_ID=0          # NPU 设备 ID（必须先设置）
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/  # pto-isa 源码路径
```

- 未设置 `TILE_FWK_DEVICE_ID` 会导致 "If no NPU environment is available" 错误
- 若 `TILE_FWK_DEVICE_ID=0` 报 `Invalid Device`，用 `npu-smi info` 查看可用设备
- API 文档：`docs/api/`，官方示例：`examples/`
