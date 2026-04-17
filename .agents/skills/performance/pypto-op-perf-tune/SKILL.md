---
name: pypto-op-perf-tune
description: PyPTO 算子性能分析和自动调优技能。用于对生成及新开发的算子进行性能分析及自动调优，包括算子用例执行及精度校验、性能数据采集及分析、分步骤性能调优和生成性能分析报告。当用户需要分析 PyPTO 算子性能、进行性能调优、生成性能报告时使用此技能。触发词：算子性能调优、性能分析、自动调优、性能优化、泳道图分析。
---

# PyPTO 算子性能分析和自动调优

## Stage Gating (Router Policy)

This skill is the **router** for the three performance tuning sub-skills.
When used by the Optimization Agent (see `skills/orchestration/lead-orchestrator/references/agents.md`), stages must be
entered sequentially and **only one tune-\* sub-skill may be active at a
time**. This keeps the active-skill count ≤ 4.

| Stage | Sub-skill to load | Enter when | Exit when | Unload before entering the next stage |
|-------|-------------------|------------|-----------|---------------------------------------|
| 1. Frontend | `skills/performance/tune-frontend/SKILL.md` | GATE 4 passed (correctness frozen); baseline perf measured | No further frontend-level gain, or target reached | ✅ yes |
| 2. Swimlane | `skills/performance/tune-swimlane/SKILL.md` | Stage 1 exited; scheduling / graph-fusion potential remains | No further swimlane-level gain, or target reached | ✅ yes |
| 3. Incore | `skills/performance/tune-incore/SKILL.md` | Stage 2 exited; single-task incore pipeline is the remaining bottleneck | Target reached, or no further incore gain | ✅ yes |
| Automation | `skills/performance/pypto-operator-auto-tuner/SKILL.md` | Swimlane extraction / AIV dependency automation needed in any stage | Automation task finished | ✅ yes — unload back to the stage sub-skill |

**Router rules (mandatory):**

- Do **not** enter Stage N+1 until Stage N exits cleanly (see 步骤 4 / 4.3
  iterative tuning flow below).
- Do **not** load more than one `tune-*` sub-skill at once.
- Activation precondition: GATE 4 has passed — E2E `detailed_tensor_compare`
  returns `all_close: true` on all outputs AND layout check exits 0. If not,
  return control to the Lead Agent; do not load any `tune-*` sub-skill.
- Every tuning change must pass the regression loop with the Verification
  Agent (see `skills/orchestration/lead-orchestrator/references/agents.md` §8): tensor compare + layout check + perf delta,
  rollback on any regression.
- Record which sub-skill was loaded, which stage it belonged to, and the
  resulting perf delta to `custom/plan/<op>.md` under the tuning log.

---

## 概述

此技能提供 PyPTO 算子性能调优的完整工作流程，包括精度校验、性能数据采集、性能分析和迭代调优。

## 核心原则

**⚠️⚠️⚠️ 非常重要：所有的调优验证必须上板执行，拒绝理论猜测，凭空捏造！！！**

### 1. 性能调优前提（最高优先级）
**⚠️ 非常重要：性能调优必须建立在精度正确的基础上！**

**⛔ 禁止：没有精度验证通过的记录，绝对禁止进入任何调优步骤！**

**核心要求**：
1. ✅ **精度必须通过**：首先确保算子精度校验通过，才能进行性能调优
2. ✅ **每次验证精度**：每次调优修改后，必须重新验证精度（不要怕麻烦！）
3. ❌ **精度失败不修复**：
   - 首轮失败：不进行修复，可以换卡尝试，多次失败让用户确认
   - 调优修改导致失败：可以进行简单分析后，如果不能解决，则回退修改，记录失败原因，尝试其他优化方案
   - ⚠️ 精度问题是算子实现问题，不是调优能解决的，可以尝试，但不强制解决

**详细处理流程**：见步骤 1.4（精度校验）和步骤 4.3（迭代调优流程）

### 2. 迭代优化原则

**⚠️ 重要：性能调优是一个循环迭代的过程！**

1. 修改一处改动点后，立即验证精度
2. 精度通过后，立即测试性能
3. **不要**全部改完再测
4. 对比修改前后的性能数据
5. 如果性能回退或者执行超时等异常情况，尝试修改，如果不能解决，则回退修改，记录失败，尝试其他方案
6. 重复上述过程直到达到目标性能

**⚠️ 注意：当长时间无法达到性能目标，或者识别没有调优空间时，可以尝试重新设计算子**

### 3. 留痕可溯原则

**⚠️ 重要：性能调优是一个可追溯的过程！**

1. 所有尝试过的调优手段及性能表现，需要留痕记录到结果文件中
2. 不要自己判断性能表现，只要是正向的收益，就保留。负向的收益，代码回退，过程留痕
3. 调优过程中遇到的报错及失败场景，请保留详细的过程及关键报错日志

### 4. 主动学习原则

**⚠️ 重要：拒绝盲目调优，主动查询，主动学习**

1. 拒绝盲目无脑试错式调优
不瞎猜、不瞎改、不凭感觉乱配置。
2. 以文档 / 资料库为依据
遇到问题查官方文档、权威资料、经典案例。
3. 遇到不清晰的接口，不确定使用方法时，主动查询 API 接口文档

**资料库**
1. [高性能编程实践](../../../models/) -- 介绍了很多高性能的编程案例，可以参考其中的高性能写法进行优化
2. [API 接口文档](../../../docs/api/) -- 介绍了整个 pypto 仓库的所有接口及调优参数使用说明

### 5. 进度可视化原则

**⚠️ 重要：调优开始时必须创建todo list，让用户清晰看到进度！**

**创建时机**：精度校验通过后立即创建

**Todo 模板**：
```markdown
## 📊 性能调优进度

### 目标
- 算子: [算子名称]
- 基准: [基准性能] us
- 目标: [目标性能] us (提升X倍)

### 进度
- ✅ 精度校验通过
- 🔄 [当前阶段]
- ⏸️ [待优化阶段]

### 性能记录
| 轮次 | 优化内容 | 执行时间 | 提升 |
|------|---------|---------|------|
| 基准 | - | XX us | - |
```

**更新时机**：
- 每次优化后（无论成功失败）
- 阶段切换时
- 性能提升>5%时
- 连续 3 次无提升时

**状态看板**（每完成 5 轮优化输出）：
```markdown
## 当前状态
- 性能: XX us (累计提升 XX%)
- 进度: ████████░░ XX%
- 成功率: X/Y 轮
```

### 6. 阶段摘要与上下文压缩原则

**⚠️ 重要：调优过程上下文会持续膨胀，必须在阶段切换时进行压缩，确保后续阶段的调优质量！**

**背景**：性能调优涉及 3 个子技能（开箱调优 → 深度调优 → 核内调优），每个子技能内部会进行多轮迭代（修改代码 → 验证精度 → 测性能 → 记录结果），上下文会随轮次快速增长。到后期阶段时，模型的有效注意力严重衰减，导致：
- 忘记早期约束（某些参数为何不能改、哪些优化已失败）
- 忽略代码中的隐含依赖
- 重复尝试已失败的优化路径

**核心要求**：
1. ✅ **每个子技能阶段结束时**，必须生成阶段交接摘要（详见步骤 4.5）
2. ✅ 摘要必须覆盖：当前性能状态、已采纳优化、已失败优化、约束发现、代码关键片段
3. ✅ 摘要生成后，后续阶段以摘要为核心上下文继续工作
4. ❌ 禁止：不做摘要直接进入下一阶段
5. ❌ 禁止：简单截断历史而不保留关键信息

**压缩原则**：
- **保留**：已采纳/已失败的优化列表、当前代码关键配置、约束发现、性能数据
- **丢弃**：中间过程的调试日志、失败代码的完整内容、冗余的性能数据对比细节
- **浓缩**：多轮迭代的性能记录合并为趋势摘要

**详细流程**：见步骤 4.5（阶段摘要与上下文压缩）

---

## 步骤 0：确定性能调优目标

**⚠️ 重要：必须明确性能目标，否则无法判断何时停止调优！**

### 0.1 询问用户性能目标

**必须询问用户**：
```
请明确性能调优目标：
- 需要提升几倍性能？（例如：提升 5 倍）
- 或者需要达到多少执行时间？（例如：≤5000 us）
```

**如果用户未说明**，请主动询问，不要猜测！

### 0.2 计算具体目标值

根据用户输入计算具体目标：
```python
# 示例：用户要求提升 5 倍
原始执行时间 = 27469.66 us
目标执行时间 = 原始执行时间 / 5 = 5493.93 us

# 示例：用户要求执行时间≤5000 us
目标执行时间 = 5000 us
```

### 0.3 设置调优终止条件

**自动终止条件**：
1. ✅ 达到性能目标（执行时间 ≤ 目标值）
2. ✅ 核心利用率 > 80% 且 气泡率 < 10%
3. ✅ 达到调优时间限制（默认 12 小时）

**手动终止条件**：
1. 用户明确要求停止

---

## 步骤 1：算子用例执行及精度校验

### 1.1 设置环境变量

```bash
export TILE_FWK_DEVICE_ID=0  # 或其他可用 NPU 卡
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

**⚠️ 执行超时配置**：
- 所有算子执行命令必须设置 timeout=300 秒（5 分钟）
- 使用 Bash 工具执行时，添加 `timeout: 300000` 参数（单位：毫秒）

**验证环境**：
```bash
# 检查 NPU 设备
npu-smi info

# 检查路径存在
ls -la $PTO_TILE_LIB_CODE_PATH/include/pto/
```

### 1.2 编译策略

**⚠️ 重要：首次进行精度校验，需要进行编译。**
**⚠️ 重要：如果只修改了算子测试或 impl 代码，直接运行即可，不需要编译。**

| 修改类型 | 是否需要编译 | 原因 |
|---------|------------|------|
| 首次执行 | ✅ 需要编译 | 第一次执行需要更新 whl 包 |
| 算子测试或 impl 代码（*.py） | ❌ 不需要 | Python 代码即时生效 |
| framework 代码 | ✅ 需要编译 | C++ 代码需要重新编译 |
| python/pypto目录 | ✅ 需要编译 | 核心框架代码 |

**编译命令**（仅在需要时执行）：
```bash
# 执行编译
python3 build_ci.py -f python3 --disable_auto_execute
# 设置环境变量
export PYTHONPATH=./pypto/build_out/:$PYTHONPATH
export LD_LIBRARY_PATH=./pypto/build_out/pypto/lib/:$LD_LIBRARY_PATH
```

### 1.3 执行算子用例

```bash
python3 custom/operator_name/operator.py --run-mode npu
```

### 1.4 精度校验（⛔ 强制检查点)

**⛔ 禁止：必须完成本步骤并通过后，才能进入步骤2！**

**执行精度校验**：
```bash
python3 custom/operator_name/operator.py --run-mode npu
```
**⛔ 强制检查流程（每次调优修改后必须执行）**：
1. ✅ 必须运行测试用例
2. ✅ 必须看到 "passed/success" 或类似成功输出，出现 "time out" 都是失败
3. ✅ 必须记录精度验证结果（包含验证时间和命令）
4. ❌ 禁止：假设精度通过、跳过验证、使用之前的验证结果

**⛔ 强制记录验证结果（必须填写）**：
```markdown
### 精度验证记录
- 验证时间: YYYY-MM-DD HH:MM:SS
- 验证命令: python3 xxx.py --run-mode npu
- 验证结果: ✅ 通过 / ❌ 失败
- 关键输出: [粘贴 "test passed" 或报错信息]
```

**✅ 通过：继续执行步骤 2（性能数据采集）**

**⚡ 立即创建调优Todo List**：
```markdown

---

## Detailed Workflow (Steps 2-5, Templates, Reporting)

Read `references/workflow-detail.md` for:
- Performance tuning progress board template
- Step 2: Performance data collection
- Step 3: Performance data analysis
- Step 4: Iterative tuning with stage handoff
- Step 5: Final tuning report generation
- Common errors and troubleshooting
