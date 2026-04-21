---
name: pypto-kernel-phase0-phase1
description: Phase 0（分诊、API 可用性、计划文件）和 Phase 1（golden 准备 — 审计、规范化、冻结）。涵盖编写 PyPTO 代码之前的所有工作。
---

# PyPTO 复杂 Kernel — Phase 0–1：规划与 Golden 准备

## 目录

| 文件 | 用途 |
|------|------|
| **本文件（SKILL.md）** | Phase 0-1 工作流指南 |
| **`op_index.json`** | 可选的本地算子索引快照 — `query_op` / `list_ops` 的离线回退方案（实时查询见 `pypto-api-explorer`） |

## Phase 0：分诊与规划

在生成任何代码之前，确定是否适用复杂 kernel 工作流。

以下任一条件为真时使用本工作流：
- backward kernel，
- 递归或有状态 kernel，
- 类 scan kernel，
- 多个依赖的数学阶段，
- 多次 layout 转换，
- 多次归约，
- 嵌套循环结构，
- 之前尝试失败，
- 想要"直接实现整个东西"的冲动。

### 步骤 0.1：API 可用性检查

将公式分解为原子操作，逐一验证其在 PyPTO 中是否存在：

```
# 优先使用 MCP
list_ops(category="")
query_op(names=["<op1>", "<op2>"])
```

CLI 回退方案：
```bash
python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py --list-categories
python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py --op <op1> --op <op2> --compact
```

对于按名称未找到的操作，按类别或语义搜索：
```
list_ops(category="<相关类别>")
retrieve_docs(query="<公式步骤> PyPTO API", chunk_type="api_doc")
```

在继续之前，将每个公式步骤标记为 **已支持**、**需要替代** 或 **不支持**。对于不支持的操作，在确认替代方案之前不得开始实现。

### 子技能委托：需求与环境（可选）

**如果需求非结构化：** 阅读 `skills/pypto-intent-understand/SKILL.md` 并按其工作流产出 `SPEC.md`。使用 SPEC.md 输出来填充计划文件的任务摘要和 API 映射。

**如果出现环境问题：** 阅读 `skills/pypto-environment-setup/SKILL.md` 并按其工作流诊断和修复 CANN、torch_npu 或构建链问题。

### 子技能委托：增强 API 探索

完成步骤 0.1 后，如果需要更深入的约束验证（3 层验证、跨 `models/` 和 `examples/` 的参考实现搜索），阅读 `skills/pypto-api-explore/SKILL.md` 并产出 `API_REPORT.md`。将 API_REPORT.md 的发现合并到计划的 API 映射部分。

### 步骤 0.2：查找结构相似的示例

搜索具有类似结构的现有 kernel：
```
retrieve_docs(query="<kernel 类型> example loop structure tiling", chunk_type="example")
```

在计划文件中记录最相关的示例路径。

### 计划文件（必须）

首先创建 `custom/plan/<算子名称>.md`，然后立即继续实现。

计划必须包含：
- 任务摘要，
- 参考位置（包括步骤 0.2 的示例路径），
- API 可用性映射（来自步骤 0.1），
- 规范化 golden 状态，
- 模块列表，
- 模块契约，
- 已冻结项，
- 尝试历史，
- 集成状态，
- 优化状态，
- 阻塞列表，
- **设计格式合规性：** `skills/kernel-code-format/pypto-kernel-design-format.md`（A–L）中哪些层适用，哪些省略及原因，
- **`active_module`** 和 **`modules_pypto_verified`**（见 `skills/lead-orchestrator/references/rules.md` → 逐模块执行规则）。

---

## Phase 1：Golden 准备

目标：定义一个数学正确且在结构上可映射到 PyPTO 的可信参考。

### 步骤 1：从最强可用的参考开始

优先级顺序：
1. PyTorch forward/backward 参考
2. NumPy 参考
3. 仅当没有现有参考时才编写新的数学参考

在编写新参考之前：
```
retrieve_docs(query="<算子名称> golden reference implementation", chunk_type="example")
```

### 步骤 2：审计参考中不兼容 PyPTO 的模式

主动搜索：
- 隐式多轴广播，
- 不透明的库操作，
- 复杂的组合调用，
- 隐藏的 layout 变更，
- 必须显式化的控制流，
- 在 PyPTO 中可能脆弱的 4D/5D 操作，
- 不能干净映射到 tile_fwk IR 的 host 侧便利操作。

对每个可疑模式，查询 op_index：
```
query_op(names=["<op>"])
```

### 步骤 3：规范化 golden

将 golden 重写为 PyPTO 友好的参考。

规则：
- 保留语义，不保留源代码语法，
- 显式化 shape，
- 显式化 dtype 转换，
- 用有意义的名称暴露中间 tensor，
- 暴露语义模块边界（用 `# --- Module M1: <角色> ---` 注释标记），
- 必要时将隐藏的广播链重写为逐轴形式，
- 将窄向量或别扭的 layout 重写为对齐友好的表示，
- **不使用 `.T` / `.t()`：** 替换为 `torch.transpose(t, dim0, dim1)`。对于 matmul `a @ b.T`，写 `torch.matmul(a, b.transpose(-2, -1))` 并注释 `# a @ b^T → pypto: b_trans=True`，
- 为每个中间 tensor 添加 shape 注释 `# [B, H, T, K]`。

### 子技能委托：golden 生成（可选）

要使用带置信度评分和自动修复的自动化 golden 生成，阅读 `skills/pypto-golden-generate/SKILL.md`。该子技能产出带验证套件的 `{op}_golden.py`。

**复杂 kernel 覆盖规则适用：** 无论子技能的输出如何，golden 必须符合上述所有 Phase 1 规则（不使用 `.T`/`.t()`、每个中间结果有 shape 注释、`# --- Module M1 ---` 边界标记）。如有需要，在子技能执行后手动应用这些规则。

### 步骤 3a. Golden 实现策略：完整计算 vs. 分块计算

规范化的 golden 可以采用**两种等价策略**：

#### 策略 1：完整计算（默认）
一次性处理整个输入 tensor。最简单直接。
```python
def attention_golden(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

#### 策略 2：分块计算（可选，推荐用于复杂 kernel）
将输入拆分为小块，独立计算每块，然后拼接或累加结果。这种实现模式：
- 模拟 PyPTO kernel 的实际执行方式（逐块计算）
- 允许在完整 PyPTO 实现之前提前验证边界处理、填充和累加逻辑
- 可以在完整 PyPTO 实现之前暴露 tile 大小对数值精度的影响
- 对于具有内在分块结构的 kernel（带窗口大小的 attention、分块 matmul、FlashAttention 模式）至关重要

示例（批量分块）：
```python
def attention_golden_tiled(q, k, v, window_size=None):
    """分块 attention golden（匹配 PyPTO kernel 逐块执行）。"""
    outputs = []
    for b in range(q.shape[0]):
        q_tile = q[b:b+1, ...]  # [1, h, t, d]
        k_tile = k[b:b+1, ...]
        v_tile = v[b:b+1, ...]
        scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))  # [1, h, t, t]
        probs = torch.softmax(scores, dim=-1)
        out_tile = torch.matmul(probs, v_tile)  # [1, h, t, d]
        outputs.append(out_tile)
    return torch.cat(outputs, dim=0)
```

**何时选择分块实现：**
- Kernel 规格明确描述了分块或基于循环的计算
- 算法涉及拆分、部分结果或状态累加
- 需要在完整 PyPTO 实现之前验证 tile 边界边缘情况

**两种策略必须产生相同的数值结果**（在浮点容差范围内）。如果同时实现两种，都包含在 `{op}_golden.py` 中并在验证套件中验证等价性。

### 步骤 3b. 构建 Golden function inventory（必须）

规范化 golden 编写完成后，在 `custom/plan/<算子名称>.md` → Golden function inventory 中列出每个数学操作：

```
| # | Golden 操作               | Shape 变换                         | PyPTO 实现          | 行号 | 状态 |
|---|---------------------------|-----------------------------------|---------------------|------|------|
| 1 | matmul(q, k^T)            | [B,H,T,K]@[B,H,K,T]->[B,H,T,T]  | pypto.matmul(...)   | L.42 | ✅    |
| 2 | softmax(scores, dim=-1)   | [B,H,T,T]->[B,H,T,T]             |                     |      | ❌    |
```

**Gate：** 在 inventory 存在且 golden 包含零个 `.T`/`.t()` 调用之前，不得进入 Phase 2。

### 步骤 4：验证规范化 golden 与原始 golden

始终使用以下方式验证：
- 相同随机种子，
- 小 shape，
- 典型 shape，
- 边界/极端 shape，
- dtype 感知比较，
- NaN/Inf 检查，
- 带有所需容差策略的 `assert_allclose`。

如果规范化 golden 不匹配，停止并修复。不要开始 PyPTO 实现。

### 步骤 5：冻结规范化 golden

规范化 golden 匹配后：
- 在计划中标记为已冻结，
- 作为后续的唯一参考，
- 除非有证据表明规范化本身有误，否则不要更改。
