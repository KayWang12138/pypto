---
name: pypto-kernel-validation
description: 验证运行器要求、detailed_tensor_compare 用法、成功标准、必需交付物，以及 PyPTO 复杂 kernel 工作流的必需输出结构。
---

# PyPTO 复杂 Kernel — 验证与交付物

## 验证运行器与 `detailed_tensor_compare`（强制）

**目的：** 端到端正确性是 golden 与 PyPTO 在同一进程中的对比。**不要**使用 `pytest` 作为默认驱动。使用用户显式运行的普通 Python 脚本。

### 运行器文件与命令

| 项目 | 要求 |
|------|------|
| **路径** | `custom/<operator_name>/test_<operator_name>.py` |
| **工作目录** | 仓库根目录 |
| **命令** | `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py` |
| **导入** | `from detailed_tensor_compare import detailed_tensor_compare`（由 `PYTHONPATH` 提供；必须使用随附的实现） |

### 运行器必须执行的操作

1. 构建输入，运行 golden（参考实现），运行 PyPTO（`pypto_function` / kernel）。
2. 对于**每个**输出张量 —— 包括 tuple/list 的所有元素、dict 的所有键，以及展平到叶子张量的嵌套结构 —— 调用 `detailed_tensor_compare(golden_tensor, pypto_tensor, tensor_name, ...)` 并要求每个都 `all_close`。**禁止：** 存在多个输出时仅比较其中一个。
3. 任何不匹配时以非零退出码退出或抛出异常；至少打印清晰的 PASS/FAIL 摘要，列出每个已比较的张量名称。
4. `if __name__ == "__main__":` 入口 —— 不是仅以 `pytest` 测试函数作为唯一可运行路径。

### 模块边界（Phase 3）

中间模块边界检查必须使用同一个随附的 `detailed_tensor_compare`。在 `custom/plan/<operator_name>.md` → 逐模块验证日志中记录每次运行（参见 `skills/plan-template/plan.template.md`）。

### `pytest`

- **禁止**作为 golden vs PyPTO 端到端对比的默认机制。
- **仅允许**用于小型、可选的附加功能，前提是在 `custom/plan/<operator_name>.md` 中记录。

## 用户侧运行器要求

用户不应需要手动编排验证。脚本 `test_<operator_name>.py` 必须：
- 准备输入，
- 运行生产 kernel，
- 运行 golden，
- 使用 `detailed_tensor_compare` 比较所有输出，
- 可选地暴露调试/检查点模式，
- 打印简洁的验证摘要。

从仓库根目录的默认命令：
```bash
PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py
```

---

## 成功标准

仅当以下所有条件为真时，任务才算完成：
- 标准化后的 golden 在容差范围内与原始 golden 一致，
- 每个模块与其预期的 golden 输出匹配，
- `custom/plan/<operator_name>.md` 记录了模块分解（理由）、分阶段文件链，以及带有 `detailed_tensor_compare` 证据的最新逐模块验证日志，
- 每个里程碑的分阶段模块文件都存在，且最终文件等于完整的集成 PyPTO kernel，
- 渐进集成在每个边界保持正确性，
- 最终生产设计是一个集成的 `@pypto.frontend.jit` kernel 或明确记录的分阶段回退方案，
- 用户可以运行一个脚本来执行验证，且该脚本比较每个 kernel 输出张量。

---

## 必需交付物

agent 必须产出以下所有内容：

1. **`custom/plan/<operator_name>.md`** —— 包括模块分解、分阶段模块文件表和逐模块验证日志；边界检查必须使用 `detailed_tensor_compare`。
2. **分阶段模块文件** —— `<op>_module1.py`、`…_module12.py`、…、`…_module1…N.py`；每个阶段通过后才进入下一个。
3. **标准化 golden 参考** —— 与最终 kernel 一致（可位于分阶段文件和/或共享辅助模块中）。
4. **验证运行器脚本** —— `custom/<operator_name>/test_<operator_name>.py`，从仓库根目录以 `PYTHONPATH=.agents` 运行；必须比较所有输出。不要使用 `pytest` 作为默认。
5. **生产 kernel 实现** —— 通常是最终的分阶段文件和/或薄的重导出；在计划中记录。
6. **可选调试辅助文件** —— 仅在必要时。
7. **摘要：** 模块契约、冻结检查点、当前已知限制，以及最终架构是融合方案还是分阶段回退。

---

## 必需输出结构

最终，agent 必须能够报告：
- 标准化 golden 摘要，
- 语义模块映射，
- 模块验证状态，
- 冻结检查点，
- 生产架构决策，
- 集成验证结果，
- 优化状态，
- 已知限制，
- 供用户运行的精确命令。
