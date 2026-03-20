# PyPTO V2 Architecture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the entire PyPTO agent skill system to V1 architecture — 7-stage state machine, 3 Subagents for context isolation, enhanced Skill independence, new API explorer skill, and simplified knowledge base.

**Architecture:** Three-layer system: Orchestrator (state machine) → Subagent (context isolation) → Skill (self-contained, independently usable). Stage 1-2 bypass Subagent layer for shared context. All Skills must work standalone without hardcoded file paths.

**Tech Stack:** Markdown agent/skill definitions (OpenCode frontmatter primary, Claude Code compatible), Python test templates, JSON state persistence.

**Spec:** `docs/plans/2026-03-20-pypto-agent-skill-architecture-v2-design.md`

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `.opencode/agents/pypto-op-analyst.md` | Analyst Subagent: Stage 3-4 context isolation + path management |
| `.opencode/agents/pypto-op-developer.md` | Developer Subagent: Stage 5-6 context isolation + path management |
| `.opencode/agents/pypto-op-perftuner.md` | PerfTuner Subagent: Stage 7 context isolation + iteration control |
| `.agents/skills/pypto-api-explorer/SKILL.md` | API exploration skill (new Stage 2) |
| `.agents/skills/pypto-api-explorer/templates/api_report.md` | API report output template |
| `.agents/skills/pypto-op-design/references/quick_ref.md` | Condensed knowledge base (~60 lines) |

### Files to Rewrite

| File | What Changes |
|------|-------------|
| `.opencode/agents/pypto-op-orchestrator.md` | Full rewrite: 7-stage (1-7), Subagent dispatch, three-state classification, V2 state format |

### Files to Modify (Skill Independence)

| File | What Changes |
|------|-------------|
| `.agents/skills/pypto-intent-understanding/SKILL.md` | Remove hardcoded `custom/{op}/` paths, add standalone usage section |
| `.agents/skills/pypto-golden-generator/SKILL.md` | Remove hardcoded paths, add standalone usage section |
| `.agents/skills/pypto-op-design/SKILL.md` | Remove hardcoded paths, update references from 4 files → quick_ref.md |
| `.agents/skills/pypto-op-develop/SKILL.md` | Remove hardcoded paths, add three-state test marker template, add standalone usage |
| `.agents/skills/pypto-precision-debugger/SKILL.md` | Remove hardcoded paths, add standalone usage |
| `.agents/skills/pypto-op-perf-analyzer/SKILL.md` | Remove hardcoded paths, add standalone usage |
| `.agents/skills/pypto-op-perf-autotuner/SKILL.md` | Remove hardcoded paths, add standalone usage |
| `AGENTS.md` | Slim down to project-level principles only |

### Files to Delete

| File | Reason |
|------|--------|
| `.agents/skills/pypto-binary-search-without-verify/SKILL.md` | Capability no longer needed |
| `.agents/skills/pypto-op-design/references/api_mapping.md` | Replaced by quick_ref.md + dynamic search |
| `.agents/skills/pypto-op-design/references/tiling_rules.md` | Replaced by quick_ref.md + dynamic search |
| `.agents/skills/pypto-op-design/references/loop_strategy.md` | Replaced by quick_ref.md + dynamic search |
| `.agents/skills/pypto-op-design/references/performance_params.md` | Replaced by quick_ref.md + dynamic search |

---

## Chunk 1: Cleanup & Foundation

### Task 1: Slim AGENTS.md to Project-Level Principles

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Read current AGENTS.md**

Read `AGENTS.md` (71 lines). Identify which sections to keep and remove.

Keep:
- 项目概述 (project overview)
- 核心原则 (core principles)
- 环境关键提示 (environment tips)

Remove:
- 开发技能系统 (entire "阶段能力" and "调试辅助" tables)
- 项目目录结构 (the `custom/{op}/` directory listing)

- [ ] **Step 2: Rewrite AGENTS.md**

Replace with:

```markdown
# AGENTS.md

本文件为 PyPTO 项目级指令。

## 项目概述

本项目用于开发华为昇腾 AI 处理器（CANN PyPTO）自定义算子，支持完整的开发、测试及性能调优流程。

- 编程框架：PyPTO
- 目标硬件：昇腾 AI 处理器（A3 服务器，CANN 8.5.0）

---

## 核心原则

1. **文档优先** — 遇到问题先查 `docs/api/` 和 `examples/`，禁止凭直觉实现
2. **定位修复不推翻** — 定位问题点后修复该部分，禁止遇错推翻重写
3. **方案可用即完成** — 方案走通后即完成，不做额外优化探索

---

## 环境关键提示

```bash
export TILE_FWK_DEVICE_ID=0          # NPU 设备 ID（必须先设置）
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/  # pto-isa 源码路径
```

- 未设置 `TILE_FWK_DEVICE_ID` 会导致 "If no NPU environment is available" 错误
- 若 `TILE_FWK_DEVICE_ID=0` 报 `Invalid Device`，用 `npu-smi info` 查看可用设备
- API 文档：`docs/api/`，官方示例：`examples/`
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "refactor: slim AGENTS.md to project-level principles only

Remove operator-development-specific content (skill lists, debug
workflow, directory structure). That information now lives in the
Orchestrator agent definition."
```

---

### Task 2: Delete pypto-binary-search-without-verify

**Files:**
- Delete: `.agents/skills/pypto-binary-search-without-verify/` (entire directory)

- [ ] **Step 1: Verify no other files reference this skill**

Search the codebase for `pypto-binary-search-without-verify`. Expected: only the orchestrator (which will be rewritten) and AGENTS.md (already slimmed).

```bash
grep -r "binary-search-without-verify" --include="*.md" .
```

- [ ] **Step 2: Delete the directory**

```bash
rm -rf .agents/skills/pypto-binary-search-without-verify/
```

- [ ] **Step 3: Commit**

```bash
git add -A .agents/skills/pypto-binary-search-without-verify/
git commit -m "refactor: remove pypto-binary-search-without-verify skill

Capability covered by pypto-precision-debugger which calls
pypto-binary-search-verify internally."
```

---

### Task 3: Replace op-design Knowledge Base (4 → 1 file)

**Files:**
- Create: `.agents/skills/pypto-op-design/references/quick_ref.md`
- Delete: `.agents/skills/pypto-op-design/references/api_mapping.md`
- Delete: `.agents/skills/pypto-op-design/references/tiling_rules.md`
- Delete: `.agents/skills/pypto-op-design/references/loop_strategy.md`
- Delete: `.agents/skills/pypto-op-design/references/performance_params.md`

- [ ] **Step 1: Create quick_ref.md**

Content is specified in `pypto-op-design-kb-simplification.md` §四. Write the ~60-line quick reference:

```markdown
# PyPTO 开发速查

> 详细信息通过搜索 docs/ 获取，本文件仅提供核心原则和约束。

---

## 1. Tiling 原则

### 1.1 算子类型判断

```
含 matmul/@ → Cube → set_cube_tile_shapes
仅逐元素/归约 → Vector → set_vec_tile_shapes
混合 → 两者都需要
```

### 1.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| Vector TileShape | 每维 > 0，最多 4 维 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Cube 必须设置 | matmul 前必须调用 set_cube_tile_shapes | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| 32B 对齐 | 尾轴需满足对齐要求 | `docs/tutorials/development/tiling.md` |

---

## 2. Loop 原则

### 2.1 是否需要 Loop

```
所有轴编译期已知 & 无多步骤分块 → 不需要 loop
存在动态轴 → 需要 pypto.loop 或 pypto.loop_unroll
多步骤计算（如 attention 分块）→ 需要 loop
```

### 2.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| 静态轴优先 Python for | pypto.loop 将静态轴展开增加编译复杂度 | `docs/tutorials/debug/performance.md` |
| 动态轴使用 pypto.loop | 并补齐边界控制 | `docs/tutorials/development/loops.md` |

### 2.3 标准写法

```python
# 静态轴 — Python for
for i in range(num_heads):
    head_i = pypto.view(x, [seq_len, head_dim], [0, i * head_dim])

# 动态轴 — pypto.loop
for i in pypto.loop(batch_size, name="LOOP_BATCH"):
    x_i = pypto.view(x, [seq_len, hidden], [i * seq_len, 0])
```

---

## 3. Runtime 硬约束

| 规则 | 说明 | 证据 |
|------|------|------|
| run_mode | 0=NPU，1=模拟器 | `docs/api/config/pypto-set_runtime_options.md` |
| NPU 需 CANN | run_mode=0 时需 source CANN 环境 | `docs/install/prepare_environment.md` |

### 示例

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def kernel(...):
    ...
```

---

## 4. from_torch 约束

- **dtype**: FP16/BF16/FP32/INT8-64/BOOL
- **contiguous**: 必须连续（is_contiguous() == True）
- **证据**: `docs/api/others/pypto-from_torch.md`

---

## 5. 搜索优先级

```
docs/（官方文档）→ 最高优先级
models/（生产代码）→ 次优先级
examples/（示例）→ 参考优先级
```
```

- [ ] **Step 2: Delete old reference files**

```bash
rm .agents/skills/pypto-op-design/references/api_mapping.md
rm .agents/skills/pypto-op-design/references/tiling_rules.md
rm .agents/skills/pypto-op-design/references/loop_strategy.md
rm .agents/skills/pypto-op-design/references/performance_params.md
```

- [ ] **Step 3: Update pypto-op-design/SKILL.md references**

In SKILL.md, find all references to the 4 deleted files and update to point to `quick_ref.md`. Look for patterns like:
- `references/api_mapping.md` → remove or replace with `references/quick_ref.md`
- `references/tiling_rules.md` → remove
- `references/loop_strategy.md` → remove
- `references/performance_params.md` → remove

Also update any "## 知识库" or "## References" section to list only `quick_ref.md`.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/pypto-op-design/references/
git add .agents/skills/pypto-op-design/SKILL.md
git commit -m "refactor(op-design): replace 4 reference files with quick_ref.md

Reduces knowledge base from ~600 lines to ~60 lines. Detailed API
mapping, tiling, loop, and performance info now obtained via dynamic
search of docs/ directory."
```

---

## Chunk 2: New pypto-api-explorer Skill

### Task 4: Create pypto-api-explorer Skill

**Files:**
- Create: `.agents/skills/pypto-api-explorer/SKILL.md`
- Create: `.agents/skills/pypto-api-explorer/templates/api_report.md`

Reference: `docs/plans/2026-03-20-pypto-agent-skill-architecture-v2-design.md` §六 and `.agents/docs/pypto-api-explorer-design.md`

- [ ] **Step 1: Create templates directory**

```bash
mkdir -p .agents/skills/pypto-api-explorer/templates
```

- [ ] **Step 2: Create api_report.md template**

Write the template from the api-explorer design doc §九. Must include `<!-- REQUIRED -->` markers on 4 sections:
- `## 1. 概述`
- `## 3. API 映射`
- `## 7. 证据索引`
- `## 8. 结论`

Full template content from the design doc.

- [ ] **Step 3: Create SKILL.md**

Write SKILL.md following these principles:
- **Frontmatter**: name, description with Chinese trigger words
- **输入约定**: Accept any form of input (natural language, formulas, code). Do NOT reference specific file names or paths.
- **输出约定**: Output `api_report.md` using template. Path decided by caller.
- **核心工作流**: Input parsing → Formula decomposition → API exploration (search `docs/api/`) → Constraint validation → Report generation
- **内嵌知识**: Operation→API mapping quick reference, operator type judgment, hard constraint quick reference (from design doc §八)
- **搜索目录**: `docs/api/operation/index.md`, `docs/api/operation/pypto-*.md`, `docs/api/others/pypto-from_torch.md`, `docs/api/config/`
- **Checklist**: File exists + 4 REQUIRED sections exist + sections non-empty
- **独立使用**: When called directly by user, prompt for operator description, output to current directory

Key sections of SKILL.md:

```markdown
---
name: pypto-api-explorer
description: "探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证 API 约束、分析算子可行性时使用。Triggers: API 探索、查找 API、PyPTO 有没有 xxx、支持什么 dtype、约束是什么、tiling 怎么配、API 映射、可行性分析、这个算子能做吗。"
---

# pypto-api-explorer

探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。

## 输入约定

接受任意形式的输入，提取算子计算逻辑：
- 自然语言描述（如"计算 softmax"）
- 数学公式（如 softmax(x) = exp(x)/sum(exp(x))）
- 代码片段（PyTorch 或伪代码）
- 已有的 spec 文档内容

**不约定输入文件名或路径。**

## 输出约定

- 输出件：api_report.md
- 格式：markdown，使用 templates/api_report.md 模板
- 输出路径由调用者决定

## 独立使用

当用户直接调用本 Skill 时：
1. 从用户输入提取算子信息
2. 如果信息不足，向用户提问补充
3. 按工作流生成 api_report.md
4. 输出到当前目录

## 核心工作流

### Stage 1: 输入解析
...（from design doc）

### Stage 2: 公式分解
...（from design doc）

### Stage 3: API 探索
...（from design doc, search docs/api/）

### Stage 4: 约束验证
...（from design doc, 3-layer validation）

### Stage 5: 生成报告
...（from design doc, use template）

## 内嵌知识

### 操作 → API 映射速查
（from design doc §八.1）

### 算子类型判断
（from design doc §八.2）

### 硬约束速查
（from design doc §八.3）

## 搜索目录

| 目录 | 搜索内容 | 优先级 |
|------|----------|--------|
| docs/api/operation/index.md | API 列表 | 入口 |
| docs/api/operation/pypto-*.md | 具体 API | 主要 |
| docs/api/others/pypto-from_torch.md | 入口约束 | 必查 |
| docs/api/config/ | Tiling 配置 | 条件 |

## Checklist

验证 api_report.md 的门禁条件：
1. 文件存在
2. 以下 4 个章节存在且内容不为空：
   - `## 1. 概述`
   - `## 3. API 映射`
   - `## 7. 证据索引`
   - `## 8. 结论`
```

- [ ] **Step 4: Verify template and SKILL.md are consistent**

Check: template REQUIRED markers match SKILL.md checklist sections.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/pypto-api-explorer/
git commit -m "feat: add pypto-api-explorer skill for Stage 2 API exploration

New skill that explores PyPTO API to provide API mapping, constraint
checking, and Tiling analysis. Accepts any input form, outputs
api_report.md via template. Can be used standalone or within
orchestrated workflow."
```

---

## Chunk 3: Skill Independence Optimization

All existing Skills need their hardcoded `custom/{op}/` paths removed and standalone usage sections added. Handle one at a time.

### Task 5: Optimize pypto-intent-understanding for Independence

**Files:**
- Modify: `.agents/skills/pypto-intent-understanding/SKILL.md`

- [ ] **Step 1: Read current SKILL.md**

Read `.agents/skills/pypto-intent-understanding/SKILL.md` (512 lines). Identify:
- All `custom/{op}/` path references
- All hardcoded file name references (e.g., "输出到 custom/{op}/spec.md")
- Missing standalone usage section

- [ ] **Step 2: Apply independence changes**

Changes to make:
1. In **输出约定** section: Replace any "输出到 `custom/{op}/spec.md`" with "输出件：spec.md，路径由调用者决定"
2. In frontmatter `description`: Keep trigger words, don't reference paths
3. Add **独立使用** section:
```markdown
## 独立使用

当用户直接调用本 Skill 时：
1. 从用户输入提取算子信息（名称、公式、数据规格等）
2. 如果信息不足，向用户逐步提问补充
3. 按工作流生成 spec.md
4. 输出到当前目录或用户指定位置
```
4. Remove any hardcoded `custom/{op}/` paths throughout the file. Replace with generic descriptions like "算子工作目录" or remove the path entirely.
5. Keep all domain knowledge, templates, and checklist logic intact.

- [ ] **Step 3: Verify no hardcoded paths remain**

```bash
grep -n "custom/" .agents/skills/pypto-intent-understanding/SKILL.md
```

Expected: 0 matches.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/pypto-intent-understanding/SKILL.md
git commit -m "refactor(intent-understanding): remove hardcoded paths, add standalone usage

Skill no longer references custom/{op}/ paths. Output path decided
by caller. Can be used independently outside orchestrator."
```

---

### Task 6: Optimize pypto-golden-generator for Independence

**Files:**
- Modify: `.agents/skills/pypto-golden-generator/SKILL.md`

- [ ] **Step 1: Read current SKILL.md**

Read `.agents/skills/pypto-golden-generator/SKILL.md` (402 lines). Identify hardcoded paths.

- [ ] **Step 2: Apply independence changes**

Same pattern as Task 5:
1. Remove `custom/{op}/` path references from input/output sections
2. Input section: "接受算子规格信息（如 spec 文档的内容、自然语言描述等）" — not "读取 custom/{op}/spec.md"
3. Output section: "输出件：`{op}_golden.py`，导出 `{op}_golden()` 函数，路径由调用者决定"
4. Add **独立使用** section
5. Keep verification logic, confidence system, repair mechanism intact

- [ ] **Step 3: Verify no hardcoded paths remain**

```bash
grep -n "custom/" .agents/skills/pypto-golden-generator/SKILL.md
```

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/pypto-golden-generator/SKILL.md
git commit -m "refactor(golden-generator): remove hardcoded paths, add standalone usage"
```

---

### Task 7: Optimize pypto-op-design for Independence

**Files:**
- Modify: `.agents/skills/pypto-op-design/SKILL.md`

- [ ] **Step 1: Read current SKILL.md**

Read `.agents/skills/pypto-op-design/SKILL.md` (259 lines). This file also needs reference updates from Task 3.

- [ ] **Step 2: Apply independence changes**

1. Remove all `custom/{op}/` path references
2. Input: "接受算子规格信息、API 映射信息、golden 代码" — not file paths
3. Output: "输出件：design.md，路径由调用者决定"
4. Update knowledge base references to point to `references/quick_ref.md` only (if not already done in Task 3)
5. Add **独立使用** section
6. Keep template reference, 9-chapter structure, search strategy intact

- [ ] **Step 3: Verify**

```bash
grep -n "custom/" .agents/skills/pypto-op-design/SKILL.md
```

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/pypto-op-design/SKILL.md
git commit -m "refactor(op-design): remove hardcoded paths, add standalone usage"
```

---

### Task 8: Optimize pypto-op-develop for Independence + Three-State Markers

**Files:**
- Modify: `.agents/skills/pypto-op-develop/SKILL.md`

This task has an extra requirement: the Skill must now generate test files that output `[PRECISION_PASS]` or `[PRECISION_FAIL]` markers.

- [ ] **Step 1: Read current SKILL.md**

Read `.agents/skills/pypto-op-develop/SKILL.md` (160 lines). Identify:
- Hardcoded paths
- Current test template (uses `numpy.testing.assert_allclose`)
- Missing three-state marker logic

- [ ] **Step 2: Apply independence changes**

1. Remove all `custom/{op}/` path references
2. Input: "接受设计方案内容、golden 代码、算子规格" — not file paths
3. Output: "输出件：`{op}_impl.py` + `test_{op}.py` + `README.md`，路径由调用者决定"
4. Add **独立使用** section

- [ ] **Step 3: Add three-state marker template**

Add a new section **三态标记约定** to SKILL.md. The generated `test_{op}.py` MUST wrap the assertion in a try/except that outputs markers:

```markdown
## 三态标记约定

生成的 `test_{op}.py` 必须使用以下模式输出精度判定标记：

```python
import sys
import numpy as np

def run_test():
    # ... setup inputs, call golden and impl ...
    try:
        np.testing.assert_allclose(impl_output, golden_output, rtol=rtol, atol=atol)
        print("[PRECISION_PASS]")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # 功能问题（无标记），exit code ≠ 0
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    run_test()
```

- `[PRECISION_PASS]`: 精度验证通过
- `[PRECISION_FAIL]`: 精度验证失败（数值不匹配）
- 无标记 + exit ≠ 0: 功能问题（代码崩溃、逻辑错误等）
```

- [ ] **Step 4: Verify**

```bash
grep -n "custom/" .agents/skills/pypto-op-develop/SKILL.md
grep -n "PRECISION_PASS\|PRECISION_FAIL" .agents/skills/pypto-op-develop/SKILL.md
```

First grep: 0 matches. Second grep: should find the new section.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/pypto-op-develop/SKILL.md
git commit -m "refactor(op-develop): add three-state markers, remove hardcoded paths

Generated test files now output [PRECISION_PASS] or [PRECISION_FAIL]
markers for the orchestrator's three-state classification. Removed
all hardcoded custom/{op}/ paths."
```

---

### Task 9: Optimize Remaining Skills for Independence

**Files:**
- Modify: `.agents/skills/pypto-precision-debugger/SKILL.md`
- Modify: `.agents/skills/pypto-op-perf-analyzer/SKILL.md`
- Modify: `.agents/skills/pypto-op-perf-autotuner/SKILL.md`

- [ ] **Step 1: Read all three files**

Read each file and identify hardcoded `custom/{op}/` paths.

- [ ] **Step 2: Apply independence changes to pypto-precision-debugger**

1. Remove `custom/{op}/` path references
2. Input: "接受精度失败的 impl 代码、golden 代码、错误信息"
3. Output: "修复后的 impl 代码，路径由调用者决定"
4. Keep the reference to `pypto-binary-search-verify` (by skill name, not path)
5. Add **独立使用** section

- [ ] **Step 3: Apply independence changes to pypto-op-perf-analyzer**

1. Remove `custom/{op}/` path references
2. Input: "性能数据（output 目录内容）"
3. Output: "性能分析报告，路径由调用者决定"
4. Add **独立使用** section

- [ ] **Step 4: Apply independence changes to pypto-op-perf-autotuner**

1. Remove `custom/{op}/` path references
2. Input: "精度通过的 impl + 性能分析报告"
3. Output: "调优后的 impl 代码，路径由调用者决定"
4. Add **独立使用** section

- [ ] **Step 5: Verify no hardcoded paths remain in any of the three**

```bash
grep -rn "custom/" .agents/skills/pypto-precision-debugger/SKILL.md .agents/skills/pypto-op-perf-analyzer/SKILL.md .agents/skills/pypto-op-perf-autotuner/SKILL.md
```

- [ ] **Step 6: Commit**

```bash
git add .agents/skills/pypto-precision-debugger/SKILL.md .agents/skills/pypto-op-perf-analyzer/SKILL.md .agents/skills/pypto-op-perf-autotuner/SKILL.md
git commit -m "refactor(skills): remove hardcoded paths from precision-debugger, perf-analyzer, perf-autotuner

All three skills now accept content-based input and let callers
decide output paths. Added standalone usage sections."
```

---

## Chunk 4: Subagent Definitions

### Task 10: Create Analyst Subagent

**Files:**
- Create: `.opencode/agents/pypto-op-analyst.md`

- [ ] **Step 1: Write Analyst Subagent definition**

```markdown
---
description: "PyPTO 算子分析 Subagent。负责 Golden 生成（Stage 3）和 Design 设计（Stage 4）阶段。接收算子目录路径和 Stage 指令，在隔离上下文中调用相应 Skill 完成分析工作，将输出写入指定路径。"
mode: subagent
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# PyPTO 算子分析 Subagent

你是 Analyst Subagent，负责在隔离上下文中完成算子分析工作。

## 职责

1. 接收 Orchestrator 的任务指令（算子目录路径 + Stage 编号）
2. 从算子目录读取所需工件内容
3. 调用对应 Skill 完成分析
4. 将 Skill 输出写入算子目录
5. 返回执行结果摘要

## Stage 3: Golden 生成

1. 读取算子目录下的 `spec.md` 内容
2. 调用 `pypto-golden-generator` Skill，传递 spec 内容
3. 将生成的 golden 代码写入算子目录下的 `{op}_golden.py`
4. 验证：执行 `python {op}_golden.py` 确认无报错
5. 返回摘要：golden 文件路径、导出函数名、验证结果

## Stage 4: Design 设计

1. 读取算子目录下的 `spec.md`、`api_report.md`、`{op}_golden.py` 内容
2. 调用 `pypto-op-design` Skill，传递以上内容
3. 将生成的设计方案写入算子目录下的 `design.md`
4. 验证：检查 design.md 必选章节存在且不为空
5. 返回摘要：design 文件路径、验证结果

## 约束

- 不能调用其他 Subagent
- 不能跳过 Skill 直接实现
- 不能修改非本 Stage 的工件
- 工件路径由 Orchestrator 在任务指令中指定
```

- [ ] **Step 2: Commit**

```bash
git add .opencode/agents/pypto-op-analyst.md
git commit -m "feat: add Analyst Subagent for Stage 3-4 (golden + design)

Lightweight subagent for context isolation. Reads artifacts from
operator directory, calls golden-generator and op-design skills,
writes output back to operator directory."
```

---

### Task 11: Create Developer Subagent

**Files:**
- Create: `.opencode/agents/pypto-op-developer.md`

- [ ] **Step 1: Write Developer Subagent definition**

```markdown
---
description: "PyPTO 算子开发 Subagent。负责代码实现（Stage 5）和精度修复（Stage 6）阶段。在隔离上下文中完成 kernel 实现、测试生成、首跑判定、精度调试修复。"
mode: subagent
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# PyPTO 算子开发 Subagent

你是 Developer Subagent，负责在隔离上下文中完成算子实现和精度修复。

## 职责

1. 接收 Orchestrator 的任务指令（算子目录路径 + Stage 编号）
2. 从算子目录读取所需工件内容
3. 调用对应 Skill 完成实现/修复
4. 将 Skill 输出写入算子目录
5. 执行测试并返回三态判定结果

## Stage 5: 代码实现

1. 读取算子目录下的 `spec.md`、`design.md`、`{op}_golden.py` 内容
2. 调用 `pypto-op-develop` Skill，传递以上内容
3. 将生成的文件写入算子目录：`{op}_impl.py`、`test_{op}.py`、`README.md`
4. 执行首跑：`python test_{op}.py`
5. 三态判定：
   - stdout 含 `[PRECISION_PASS]` → 报告"精度通过"
   - stdout/stderr 含 `[PRECISION_FAIL]` → 报告"精度失败"
   - exit code ≠ 0 且无标记 → 报告"运行失败"
6. 返回摘要：文件路径、判定结果、错误信息（如有）

## Stage 6: 精度修复

1. 读取算子目录下的 `{op}_impl.py`、`{op}_golden.py`、上次错误信息
2. 备份当前 `{op}_impl.py` 到 `history_version/`
3. 调用 `pypto-precision-debugger` Skill（内部可调用 `pypto-binary-search-verify`）
4. 将修复后的 impl 写入算子目录
5. 重新执行测试，进行三态判定
6. 判定结果处理：
   - `[PRECISION_PASS]` → 保留修改，报告"精度通过"
   - `[PRECISION_FAIL]` → 对比精度指标：提升则保留，下降则回滚到备份
   - 功能问题（无标记报错）→ 必须回滚到备份版本
7. 返回摘要：判定结果、是否回滚、精度指标变化

## 约束

- 不能调用其他 Subagent
- 不能跳过 Skill 直接实现
- Stage 6 每次修复前必须备份 impl.py
- 功能问题（无标记报错）必须回滚，不可保留
```

- [ ] **Step 2: Commit**

```bash
git add .opencode/agents/pypto-op-developer.md
git commit -m "feat: add Developer Subagent for Stage 5-6 (implement + precision fix)

Handles code implementation with three-state classification and
precision fix with rollback strategy. Calls op-develop and
precision-debugger skills in isolated context."
```

---

### Task 12: Create PerfTuner Subagent

**Files:**
- Create: `.opencode/agents/pypto-op-perftuner.md`

- [ ] **Step 1: Write PerfTuner Subagent definition**

```markdown
---
description: "PyPTO 算子性能调优 Subagent。负责性能分析和调优迭代（Stage 7）。在隔离上下文中完成性能采集、分析、调优，每次调优后验证精度。"
mode: subagent
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# PyPTO 算子性能调优 Subagent

你是 PerfTuner Subagent，负责在隔离上下文中完成性能分析和调优。

## 职责

1. 接收 Orchestrator 的任务指令（算子目录路径）
2. 迭代执行：性能分析 → 调优 → 精度验证
3. 管理中止条件
4. 返回最终性能报告

## Stage 7: 性能调优

### 初始化

1. 读取算子目录下精度通过的 `{op}_impl.py`
2. 记录初始性能基准

### 迭代流程

每次迭代：
1. 调用 `pypto-op-perf-analyzer` Skill → 获取性能分析报告
2. 调用 `pypto-op-perf-autotuner` Skill → 获取调优后的 impl
3. 将调优后的 impl 写入算子目录
4. 执行测试验证精度（`python test_{op}.py`）
5. 判定：
   - 精度通过 + 性能提升 → 采纳修改，重置连续无提升计数
   - 精度通过 + 性能下降 → 回滚到上一版本，连续无提升计数 +1
   - 精度失败 → 回滚到上一版本，连续无提升计数 +1

### 中止条件

满足任一条件即中止，返回结果：
- 迭代次数 >= 10
- 连续三次无性能提升
- 满足 spec 中定义的性能目标（如有）

### 返回结果

返回摘要包含：
- 最终性能指标
- 迭代次数
- 性能提升比例
- 中止原因

## 约束

- 不能调用其他 Subagent
- 每次调优后必须验证精度
- 精度失败时必须回滚
- 性能下降时必须回滚
```

- [ ] **Step 2: Commit**

```bash
git add .opencode/agents/pypto-op-perftuner.md
git commit -m "feat: add PerfTuner Subagent for Stage 7 (performance tuning)

Iterative performance analysis and tuning with precision validation
after each iteration. Manages stop conditions (10 iterations, 3
consecutive no-improvement, or target met)."
```

---

## Chunk 5: Orchestrator Rewrite

### Task 13: Rewrite Orchestrator for V2 Architecture

**Files:**
- Rewrite: `.opencode/agents/pypto-op-orchestrator.md`

This is the largest single change. The orchestrator must be rewritten from scratch to support:
- 7-stage state machine (1-7) instead of (0-6 with 2a/2b)
- Subagent dispatch for Stage 3-7
- Direct Skill call for Stage 1-2
- Three-state classification
- V2 state persistence format
- Old state format migration

- [ ] **Step 1: Read current orchestrator**

Read `.opencode/agents/pypto-op-orchestrator.md` (368 lines) to understand structure.

- [ ] **Step 2: Write new orchestrator**

The new orchestrator must contain these sections:

**§1 Identity & Principles**
- Primary agent, sole process owner
- Only orchestrates, never writes code
- Artifact-driven state, not conversation-driven
- Gate must pass before next stage — no skipping

**§2 Startup Flow**
- Determine operator name `{op}` and directory `custom/{op}/`
- Check `.orchestrator_state.json` — if exists, detect format:
  - New format (keys "1"-"7") → resume from `current_stage`
  - Old format (keys "0", "2a", "2b") → migrate to new format, then resume
- If no state file → start from Stage 1
- List existing artifacts, notify user

**§3 Artifact Contract**
- Standard directory (includes `api_report.md`)
- Owner/Consumer table (updated for 8 artifacts)
- Three-file separation pattern
- Override strategy (version management for user artifacts)

**§4 Stage State Machine**
- 7 stages with exact execution method:
  - Stage 1: Orchestrator calls `pypto-intent-understanding` directly
  - Stage 2: Orchestrator calls `pypto-api-explorer` directly
  - Stage 3: Orchestrator dispatches Analyst Subagent (`@pypto-op-analyst`) for golden
  - Stage 4: Orchestrator dispatches Analyst Subagent for design
  - Stage 5: Orchestrator dispatches Developer Subagent (`@pypto-op-developer`) for implementation
  - Stage 6 (optional): Orchestrator dispatches Developer Subagent for precision fix
  - Stage 7: Orchestrator dispatches PerfTuner Subagent (`@pypto-op-perftuner`) for tuning
- Mermaid flowchart (from design doc §3.2)
- Three-state classification for Stage 5
- Retry limits per stage

**§5 Precision Fix Loop (Stage 5→6)**
- Three-state detection: `[PRECISION_PASS]`, `[PRECISION_FAIL]`, no marker
- Stage 6 rollback strategy
- Retry limits: Stage 5 = 10, Stage 6 = 5

**§6 Performance Tuning (Stage 7)**
- Iterative: perf-analyzer → perf-autotuner → precision check
- Stop conditions: 10 iterations / 3 consecutive no-improvement / target met
- Rollback on precision failure or performance decrease

**§7 State Persistence**
- V2 JSON format (from design doc §十一)
- Update timing: stage start → `in_progress`, stage end success → `completed`
- Field descriptions

**§8 State Migration**
- Old format detection: presence of "0", "2a", "2b" keys
- Migration mapping table
- Stage 2 (API explorer) auto-completed if old Stage 1+2a both completed

**§9 Terminal States**
- SUCCESS, BLOCKED_SPEC, BLOCKED_API, BLOCKED_GOLDEN, BLOCKED_DESIGN, BLOCKED_IMPL, BLOCKED_ACCURACY, BLOCKED_ENVIRONMENT
- Recovery: intent recognition → resume from current_stage → reset retry count

**§10 Final Report**
- Structured output template (operator info, precision result, performance result, known issues)

- [ ] **Step 3: Verify completeness**

Check the new orchestrator covers all V2 design requirements:
- [ ] 7 stages numbered 1-7
- [ ] Stage 1-2 direct Skill call
- [ ] Stage 3-7 Subagent dispatch
- [ ] Three-state classification
- [ ] V2 state format
- [ ] State migration logic
- [ ] All retry limits from design doc
- [ ] Stage 6 rollback strategy
- [ ] Stage 7 stop conditions
- [ ] All terminal states
- [ ] `api_report.md` in artifact contract

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/pypto-op-orchestrator.md
git commit -m "feat: rewrite orchestrator for V2 architecture

7-stage state machine (1-7), Subagent dispatch for Stage 3-7,
three-state precision classification, V2 state persistence format,
old state migration support. Adds Stage 2 (API exploration) and
replaces Stage 4 binary-search with Stage 6 precision-debugger."
```

---

## Chunk 6: Final Verification

### Task 14: Cross-File Consistency Check

- [ ] **Step 1: Verify all skill names are consistent**

```bash
# All SKILL.md frontmatter names
grep -r "^name:" .agents/skills/*/SKILL.md

# All orchestrator skill references
grep -o "pypto-[a-z-]*" .opencode/agents/pypto-op-orchestrator.md | sort -u

# All subagent skill references
grep -o "pypto-[a-z-]*" .opencode/agents/pypto-op-analyst.md .opencode/agents/pypto-op-developer.md .opencode/agents/pypto-op-perftuner.md | sort -u
```

Verify: every skill name referenced in orchestrator and subagents has a matching SKILL.md.

- [ ] **Step 2: Verify no orphan references to deleted files**

```bash
# Check for references to deleted skills/files
grep -rn "binary-search-without-verify" .opencode/ .agents/ AGENTS.md
grep -rn "api_mapping\.md\|tiling_rules\.md\|loop_strategy\.md\|performance_params\.md" .agents/skills/pypto-op-design/
```

Expected: 0 matches for both.

- [ ] **Step 3: Verify no hardcoded custom/ paths in Skills**

```bash
grep -rn "custom/{op}/" .agents/skills/*/SKILL.md
grep -rn "custom/" .agents/skills/*/SKILL.md
```

Expected: 0 matches. (Paths should only appear in orchestrator and subagents, which manage file system layout.)

- [ ] **Step 4: Verify state format consistency**

Check orchestrator state format matches design doc §十一:
- Keys are "1" through "7" (strings)
- Has `stage_retry_count`, `perf_iteration` fields
- Migration logic handles old "0", "2a", "2b" keys

- [ ] **Step 5: Verify three-state marker consistency**

Check pypto-op-develop SKILL.md contains `[PRECISION_PASS]` and `[PRECISION_FAIL]` template.
Check orchestrator references these exact marker strings.

```bash
grep -n "PRECISION_PASS\|PRECISION_FAIL" .agents/skills/pypto-op-develop/SKILL.md .opencode/agents/pypto-op-orchestrator.md .opencode/agents/pypto-op-developer.md
```

All three files should reference these markers.

- [ ] **Step 6: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: cross-file consistency fixes for V2 architecture"
```

(Only if fixes were needed.)

---

## Summary: Execution Order

| Task | Description | Files Changed | Estimated Steps |
|------|-------------|---------------|-----------------|
| 1 | Slim AGENTS.md | 1 modify | 3 |
| 2 | Delete binary-search-without-verify | 1 delete | 3 |
| 3 | Replace op-design KB (4→1) | 1 create, 4 delete, 1 modify | 4 |
| 4 | Create api-explorer skill | 2 create | 5 |
| 5 | Independence: intent-understanding | 1 modify | 4 |
| 6 | Independence: golden-generator | 1 modify | 4 |
| 7 | Independence: op-design | 1 modify | 4 |
| 8 | Independence: op-develop + 3-state | 1 modify | 5 |
| 9 | Independence: remaining 3 skills | 3 modify | 6 |
| 10 | Create Analyst Subagent | 1 create | 2 |
| 11 | Create Developer Subagent | 1 create | 2 |
| 12 | Create PerfTuner Subagent | 1 create | 2 |
| 13 | Rewrite Orchestrator | 1 rewrite | 4 |
| 14 | Cross-file verification | 0 (verify only) | 6 |

**Total: 14 tasks, ~54 steps, ~13 commits**

Tasks 1-3 can run in parallel (cleanup). Tasks 5-9 can run in parallel (skill independence). Tasks 10-12 can run in parallel (subagent creation). Task 13 (orchestrator) depends on all previous. Task 14 depends on everything.
