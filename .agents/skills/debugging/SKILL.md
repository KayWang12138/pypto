---
name: pypto-kernel-debugging
description: 调试技能概览 — 失败历史、策略切换、逐算子检查协议，以及完整的调试手册（DEBUG.md）。
---

# PyPTO 复杂 Kernel — 调试

本技能涵盖 agent 陷入困境时的应对措施。完整的错误手册和 agent 学习到的模式请阅读 **`DEBUG.md`**（同目录）。

## 内容概览

| 文档 | 涵盖内容 |
|----------|---------------|
| **本文件（SKILL.md）** | 失败历史记录、策略切换规则、逐算子检查协议 |
| **`DEBUG.md` §1–§7** | 不透明错误手册：`FFFFF`、`UNKNOWN`、`F21004`、AICore、错误码速查、停止条件 |
| **`DEBUG.md` §8** | 示例 kernel 调试实践（全局模式、反复出现的失败、集成追踪） |
| **`DEBUG.md` §9** | Agent 学习到的开发模式：JIT 签名、动态 shape、`pypto.view`、SIM 模式、matmul API、tile shape、reduction 对齐等 |

---

## 失败历史

每次迭代必须记录：
- 假设，
- 确切修改位置，
- 结果，
- 下一步操作，
- 是否发生了回退。

此为强制要求。

---

## 策略切换规则

以下规则为强制要求。

- 反复出现编译错误 → 先运行 `validate_kernel_structure`；重新检查接口、shape、dtype、kernel 结构
- 反复出现运行时错误 → 先运行 `diagnose_error(error_log=..., kernel_code=...)`；若无匹配，切换到设备/运行时错误定位（`pypto-aicore-error-locator`）
- 反复出现精度不匹配 → 运行 `validate_kernel_structure` 排除写回 bug；然后切换到二分检查点调试（`pypto-binary-search-verify`）
- 三次基于证据的尝试后仍然失败 → 重新审视模块边界或架构
- 整图内存冲突 / copy-pass 失败 → 停止假设局部行修复；使用分阶段回退或创建最小复现
- 以上步骤后仍出现不透明 PyPTO 错误 → 运行 `extract_pypto_calls.py`，然后遵循下方的逐算子检查协议

---

## PyPTO 逐算子检查协议（agent 陷入困境时）

**目标：** 当遇到 `validate_kernel_structure`、`diagnose_error` 或文档无法解决的 PyPTO 特定失败时，使用此机械性序列使调试收敛。

### 步骤 0 — 枚举每个 `pypto` 调用（强制检查清单）

在失败的 kernel 文件上运行：

```bash
python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py custom/<operator_name>/<kernel_or_impl>.py
```

- 输出为有序的编号列表：行号 + 调用 shape。
- 使用此列表作为"下一个 PyPTO op 是什么"的唯一事实来源。
- 可选：`--json` 用于机器消费或粘贴到计划中。

### 步骤 1 — 按文档逐个验证每个调用点（按顺序）

对于可疑区域中的调用点（如果故障未知则从索引 1 开始）：

1. 映射 `pypto.<name>` → `docs/api/operation/pypto-<name>.md` 或 `docs/api/config/...`。
2. 确认 dtype、shape/axes、tile 配置、转置标志和写回规则与该行的文档匹配。
3. 在计划中记录不匹配项，使用步骤 0 中的行号。

### 步骤 2 — 数值结果错误：检查点二分法（非随机编辑）

如果图运行但输出错误：
- 使用 `pypto-precision-compare` / 检查点保存，使 golden 和 kernel 在对齐的逻辑点导出 tensor。
- 二分查找第一个分叉的检查点索引；将该索引映射回步骤 0 中的调用点范围。

### 步骤 3 — **不要**"注释掉文件的一半"

在一个融合的 `@jit` 中禁用任意 `pypto` 行通常会使图失效或隐藏真正的 bug。

- 优先使用逐模块存根（见 `skills/lead-orchestrator/references/rules.md` → 逐模块执行）：缩小活跃区域，然后在更小的文件上重新运行步骤 0。
- 如果必须在一个模块内部进行二分，在调用点 k 和 k+1 之间插入一个中间检查点，使用编号列表二分查找 k — 除非最小复现需要，否则不要删除 op。

### 步骤 4 — 计划文件记录（交接安全）

追加到 `custom/plan/<operator_name>.md`：
- `extract_pypto_calls.py` 输出的路径（或粘贴表格），
- 第一个文档不匹配或第一个分叉的检查点索引，
- 假设和补丁；重新运行验证。

此协议与完全自主运行兼容：agent 在陷入困境时无需等待用户即可应用，除非停止条件适用。

---

## 编写 PyPTO 代码之前 — 参考 DEBUG.md §9

在编写每个模块的 PyPTO 代码之前，阅读对应的子章节：

| 你即将编写的内容 | 先阅读 |
|------------------------------|------------|
| 任何 `@pypto.frontend.jit` 函数 | `DEBUG.md` §9.1（`from __future__ import annotations` 会破坏 JIT） |
| `pypto.view` / `pypto.assemble` | `DEBUG.md` §9.4（黄金规则：`len(shape)==len(offsets)`、padding、reshape） |
| `pypto.matmul` | `DEBUG.md` §9.19（转置标志 `a_trans`/`b_trans`，不是 `.T`；需要 cube+vec tile） |
| `.sum()` / reduction op | `DEBUG.md` §9.19（32 字节对齐；基于 matmul 的变通方案） |
| 动态 shape / `pypto.loop` | `DEBUG.md` §9.2（具体循环边界、符号偏移） |
| JIT 签名中的 Tensor 类型提示 | `DEBUG.md` §9.13（使用 `pypto.Tensor([], dtype)`，不是显式 `DYNAMIC` 维度） |
| JIT 内的 element-wise op | `DEBUG.md` §9.14（Python `*`、`+`、`.exp()` 可用；优先使用而非冗长的 `pypto.mul`） |
| Tile shape 配置 | `DEBUG.md` §9.15 + §9.19（matmul 需要 vec+cube；≥4 个 vec 参数） |
| 开发期间的任何错误 | `DEBUG.md` §9.11（常见错误 → 原因 → 解决方案速查表） |

---

## 子技能回退决策树（路由策略）

本技能是所有调试子技能的**路由器**。当由 Verification Agent 使用时（见 `skills/lead-orchestrator/references/agents.md`），每次失败加载**恰好一个**子技能，在处理下一个失败之前卸载它。这将活跃技能数保持在 ≤ 4。

**调度顺序 — 从上到下评估；在第一个匹配项处停止：**

1. **精度 / 准确度不匹配**
   *信号：* `detailed_tensor_compare` 返回 `all_close: false`；构建和布局检查成功；无崩溃，日志中无 aicore error。
   → 加载 `skills/pypto-precision-debug/SKILL.md`
   （代码级变通方案：inplace、unroll、`+0.0`）。

2. **需要精度二分**
   *信号：* (1) 之后精度仍然失败，或分叉的检查点尚未定位；涉及多个模块。
   → 加载 `skills/pypto-precision-compare/SKILL.md`
   （`pass_verify_save` 或检查点 tensor；二分查找第一个分叉的检查点索引）。

3. **aicore error / CCE 文件报告**
   *信号：* 日志包含 `aicore error`、`ERROR_CODE: EE...`，或 CCE 文件路径。
   → 加载 `skills/pypto-aicore-error-locator/SKILL.md`
   （映射错误 → CCE 文件 → 问题源码行）。

4. **Host 侧崩溃 / 堆栈**
   *信号：* 段错误、Python/C++ 堆栈、进程在 kernel 启动完成前被杀死。
   → 加载 `skills/pypto-host-stacktrace-analyzer/SKILL.md`
   （地址到源码映射、常见 host 崩溃模式）。

5. **怀疑 Workspace 内存重叠**
   *信号：* 随 tensor 布局变化的非确定性精度失败；隔离模块通过，仅在整图中失败。
   → 加载 `skills/pypto-memory-overlap-detector/SKILL.md`。

6. **OOM / workspace 大小异常**
   *信号：* `rtMalloc failed`、OOM 错误，或 workspace 大小远超预期。
   → 加载 `skills/pypto-machine-workspace/SKILL.md`。

**路由器规则（强制）：**

- **不要**同时加载多个子技能。如果出现新的失败类别，先卸载当前子技能。
- **不要**投机性预加载子技能。
- 如果没有匹配的行，留在本 SKILL.md + `DEBUG.md` 中。不要升级。
- `skills/lead-orchestrator/references/rules.md` 在冲突时优先于任何子技能的指导。
- 将调度决策记录到 `custom/plan/<op>.md` 的 **开发与调试日志** 下（匹配了哪一行、加载了哪个子技能、结果）。
