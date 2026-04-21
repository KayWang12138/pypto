---
name: pypto-kernel-phase2-phase3
description: Phase 2（语义模块分解 — 按含义拆分、定义契约、冻结）和 Phase 3（模块构建 — 逐模块实现、验证、交叉检查 golden inventory）。
---

# PyPTO 复杂 Kernel — Phase 2–3：分解与构建

## Phase 2：语义模块分解

目标：将 kernel 拆分为有语义含义、可验证的代码块。

### 将分解写入计划（必须）

一旦确定模块拆分，在 `custom/plan/<算子名称>.md` 中写入模块分解：命名模块、边界 tensor 和理由。参见 `skills/plan-template/plan.template.md` 了解日志格式。

### 规则：按语义拆分，不按等复杂度拆分

优先选择如下边界：
- matmul 块，
- 归一化 / softmax 块，
- 递归状态更新块，
- 归约块，
- 写回 / 组装块，
- layout 转换块，
- 衰减 / 门控块，
- 逐步递归块。

避免：
- 任意等大小分块，
- 将单一语义操作拆分到多个模块，
- 将无关的 layout 和计算变换合并到一个模块，
- 无法独立验证的模块。

### 为每个模块定义契约

在为模块编写任何代码之前，查阅确切的 PyPTO API 签名：

```
query_op(names=["<op1>", "<op2>"])
```

CLI 回退方案：
```bash
python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py --op <op1> --op <op2>
```

对于索引中未捕获的约束详情：
```
retrieve_docs(query="<操作名称> constraints dtype tile shape", chunk_type="api_doc")
```

每个模块必须指定：
- 名称、用途，
- 输入、输出、shape、dtype，
- 不变量，
- 语义前驱和后继，
- 是否包含循环携带状态，
- 是否包含归约，
- 是否包含对齐敏感的 tensor，
- 使用的 PyPTO API（含 op_index 中的确切签名）。

### 模块冻结规则

模块验证通过后，标记为已冻结。不要因为后续阶段失败而编辑已冻结模块。首先检查最早的未冻结失败边界。

### 子技能委托：DESIGN.md 生成（可选）

要产出包含 API 映射、tiling 策略、循环结构和验证计划的独立设计文档，阅读 `skills/pypto-op-design/SKILL.md` 并生成 `DESIGN.md`。设计文档补充（而非替代）计划文件中的模块分解和契约。

---

## Phase 3：模块构建

目标：在集成之前独立构建每个模块。

**硬性规则：** 每次迭代最多扩展一个新语义模块的真实 PyPTO 逻辑到生产 kernel。下游部分保持桩化或从 golden 边界 tensor 获取数据（见 `skills/lead-orchestrator/references/rules.md` → 逐模块执行规则）。

### 编写 PyPTO 代码之前 — 查阅 `skills/debugging/DEBUG.md` §9

在编写每个模块的 PyPTO 代码之前阅读相关子节：

| 即将编写的内容 | 先阅读 |
|----------|----------|
| 任何 `@pypto.frontend.jit` 函数 | §9.1（`from __future__ import annotations` 会破坏 JIT） |
| `pypto.view` / `pypto.assemble` | §9.4（黄金法则：`len(shape)==len(offsets)`、填充、reshape） |
| `pypto.matmul` | §9.19（转置标志 `a_trans`/`b_trans`，不是 `.T`；需要 cube+vec tile） |
| `.sum()` / 归约操作 | §9.19（32 字节对齐；基于 matmul 的替代方案） |
| 动态 shape / `pypto.loop` | §9.2（具体循环边界、符号偏移） |
| JIT 签名中的 Tensor 类型提示 | §9.13（使用 `pypto.Tensor([], dtype)`，不使用显式 `DYNAMIC` 维度） |
| JIT 内的逐元素操作 | §9.14（Python `*`、`+`、`.exp()` 可用；优于冗长的 `pypto.mul`） |
| Tile shape 配置 | §9.15 + §9.19（matmul 需要 vec+cube；≥4 个 vec 参数） |
| 开发过程中的任何错误 | §9.11（常见错误 → 原因 → 解决方案速查表） |

### 子技能参考：实现模板与执行约束

编写模块代码时，查阅 `skills/pypto-op-develop/SKILL.md` 获取额外的实现约束和模板（`references/execution-constraints.md`、`references/impl-template.py`、`references/test-template.py`）。**复杂 kernel 覆盖规则适用：** Layer A–L 模板（`skills/kernel-code-format/pypto_kernel_template.py`）、分阶段文件链和逐模块执行规则优先于子技能的单文件方式。

### 分阶段模块文件（必须 — 在代码中执行）

将每个步骤实现为 `custom/<op>/<op>_module1.py` → `…_module12.py` → … → `…_module1…N.py`（见 `skills/lead-orchestrator/references/rules.md` 规则 14）。每个文件是该里程碑的制品：golden + PyPTO + 可运行比较。后缀 = 拼接的模块索引（`1`、`12`、`123`、…）。

### 步骤 1. 将模块表达为 kernel 语义

设计模块以适配最终的生产 kernel。

编写之前，检索每个 API 的确切签名：
```
query_op(names=["<op_name>"])
```

如果对齐或 tiling 约束不明确：
```
retrieve_docs(query="<op_name> tile shape alignment constraint", chunk_type="api_doc")
```

允许的形式：语义伪代码、辅助函数、临时检查点逻辑、可选的临时验证 kernel。

默认不允许的形式：每个语义模块使用单独的生产 `@jit` kernel。

### 步骤 2. 构建模块级验证路径

每个模块必须在下一个模块开始之前可验证。

验证可以使用临时检查点输出、临时渐进式 kernel 或模块边界的 host golden 提取。但验证路径必须清晰地映射回预期的最终集成 kernel。

在模块边界使用 `detailed_tensor_compare`（内置）。每次边界运行后，在计划中的每模块验证日志追加一行。

### 步骤 2b. 交叉检查 Golden function inventory（运行前必须）

在首次执行模块之前，打开 `custom/plan/<算子名称>.md` → Golden function inventory 并交叉检查此模块范围内的每个操作：

- 对属于当前模块的每个 golden 操作，标记 ✅ 并注明 PyPTO 调用和行号，或标记 ❌ 如果尚未实现。
- **如果仍有任何 ❌，不要运行测试。** 先实现遗漏的操作。

此步骤是防止因遗漏操作导致精度错误的主要防线。

### 步骤 3. 验证模块正确性

**先运行 AST 检查 — 在编译或执行之前：**
```
validate_kernel_structure(source_code=<完整模块源代码>)
```

修复所有 `error` 级别的问题后再继续。

然后验证：编译/结构、shape、dtype 以及边界 tensor 与 golden 的比较（使用 `detailed_tensor_compare`）。

### 步骤 4. 冻结与记录

如果模块通过：
- 冻结它，
- 在每模块验证日志中记录通过的边界，
- 进入下一个模块。

如果模块失败：

1. 先检查已知错误模式：
   ```
   diagnose_error(error_log=<完整错误输出>, kernel_code=<模块源代码>)
   ```
   如果找到匹配，应用修复并重新运行。

2. 如果没有匹配的模式：
   - 检查 shape/dtype/接口，
   - 检查内部中间检查点，
   - 切换到二分查找式调试。

### 子技能委托：调试升级

当 `skills/debugging/DEBUG.md` 策略和 `diagnose_error` 无法解决问题时，按顺序升级到以下子技能：

1. **精度变通方案**：阅读 `skills/pypto-precision-debug/SKILL.md` — 尝试变通方案检查清单（前端切换、避免 inplace、unroll_list=[1]、submit_before_loop、+0.0、shape 调整）。
2. **精度二分定位**：阅读 `skills/pypto-precision-compare/SKILL.md` — 使用 `pass_verify_save` 或检查点 tensor 精确定位发散的操作。
3. **内存重叠**：如果精度失败疑似由 workspace 问题引起，阅读 `skills/pypto-memory-overlap-detector/SKILL.md`。
4. **AICore 错误**：如果错误日志包含 `aicore error`，阅读 `skills/pypto-aicore-error-locator/SKILL.md` 定位 CCE 文件和问题代码行。
5. **Host 崩溃**：如果进程崩溃并输出堆栈信息，阅读 `skills/pypto-host-stacktrace-analyzer/SKILL.md` 将地址解析到源代码行。
6. **MACHINE workspace**：对于 workspace 相关分析，阅读 `skills/pypto-machine-workspace/SKILL.md`。
