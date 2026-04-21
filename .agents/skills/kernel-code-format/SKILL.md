---
name: pypto-kernel-code-format
description: Kernel 代码结构 — A-L 层设计文档、命名规范、shape 注解规范以及强制使用的 Python 代码骨架（pypto_kernel_template.py）。
---

# PyPTO Complex Kernel — Kernel 代码格式

本 Skill 定义**如何组织 kernel 代码**。包含设计文档和每个阶段文件及生产 kernel 必须遵循的强制 Python 模板。

## 目录

| 文件 | 用途 |
|------|------|
| **`pypto-kernel-design-format.md`** | A-L 层设计文档、命名规范、参考约束、shape 注解规范（§11）、检查清单 |
| **`pypto_kernel_template.py`** | 实现 A-L 层的强制 Python 代码骨架 — 复制到 `custom/<op>/` 作为起点 |

---

## 何时阅读

- **Phase 0：** 了解哪些层适用于你的 kernel（仅前向、反向、融合等）
- **Phase 2：** 将模块分解映射到各层和阶段标记
- **Phase 3-5：** 每个阶段文件（`<op>_module*.py`）和最终 kernel 都必须遵循此模板
- **任何时间：** shape 注解规范（设计文档 §11）适用于所有 kernel 代码

## A-L 层（快速参考）

| 层 | 职责 | 典型名称 |
|----|------|----------|
| A | 工具函数（可选） | `tensor_compare_report`，辅助函数 |
| B | 小型数学构建块 | `norm_fwd`、`softmax_chunk` |
| C | 前向参考 | `forward_ref` |
| D | Host 侧常量 | `make_chunk_constants` |
| E | 反向参考辅助 | `_slice_chunk_inputs`、`_stage_attn` |
| F | Golden 反向/前向 | `torch_golden_*` |
| G | 缓存/桥接 | `prepare_cache_for_npu` |
| H | PyPTO 子 kernel | `pypto_slice_inputs`、`pypto_fused_stage_ab` |
| I | Kernel 实现 | `_your_op_kernel_impl`（包含 `pypto.loop`） |
| J | JIT 入口 | `@pypto.frontend.jit` `your_op_kernel_npu` |
| K | Host 封装 | `pypto_function`（仅 I/O，不含 `for...in range`） |
| L | 驱动/测试 | `main()` 或测试运行器 |

并非每个 kernel 都需要所有层。根据你的复杂度选择合适的层。

## 关键规则

- **强制骨架：** 每个交付物必须以 `pypto_kernel_template.py` 作为起始布局（`skills/lead-orchestrator/references/rules.md` 规则 17）
- **Layer K（`pypto_function`）中禁止 `for...in range`**：kernel 迭代属于 Layer I，使用 `pypto.loop`（`skills/lead-orchestrator/references/rules.md` 规则 18）
- **shape 注解：** 每个 tensor 赋值必须附带行内 shape 注释（设计文档 §11）
- **命名：** 遵循设计文档 §2 中的命名规范
