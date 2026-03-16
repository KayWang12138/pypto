---
name: pypto-op-develop
description: "读取 spec.md、design.md 和 {op}_golden.py，产出完整可运行的 PyPTO 算子：{op}_impl.py（kernel 实现，导出 {op}_wrapper()）、test_{op}.py（测试入口）、README.md。当需要编写 PyPTO 算子实现时使用此 skill。Triggers: 实现算子、写 kernel、编写实现、写 impl、算子编码、开始编码、code the op、写 test、生成测试、写实现代码、op develop、算子开发、kernel 实现。"
---

# PyPTO 算子功能实现

读取 `spec.md`、`design.md`、`{op}_golden.py`，产出一个能正常运行的 PyPTO 算子实现。

## Contract

### Inputs

- `custom/{op}/spec.md` — 算子需求规格
- `custom/{op}/design.md` — 设计方案
- `custom/{op}/{op}_golden.py` — 纯 torch 参考实现
- 可选：已有 `custom/{op}/` 目录中的历史实现文件（续跑场景）

### Outputs

- `custom/{op}/test_{op}.py` — 测试入口（从 golden 和 impl 导入，不含实现代码）
- `custom/{op}/{op}_impl.py` — PyPTO kernel 实现（必须使用pypto接口实现，导出 `{op}_wrapper()` 函数）
- `custom/{op}/README.md` — PyPTO 算子文档

### Side Effects

- 首跑 `test_{op}.py` 进行功能验证

### Overwrite Policy

- 覆盖前需确认

### Failure Exit

- 3 个输出文件中任一缺失 → FAIL
- 脚本不可执行（语法错误等） → FAIL

---

## Read spec.md

从 `spec.md` 读取：

| 字段 | 用途 |
|------|------|
| 算子名称 | 文件命名、函数命名 |
| 数学公式 | 理解计算逻辑 |
| 输入输出规格 | 测试数据生成 |
| dtype 范围 | 测试覆盖、impl 类型处理 |
| 精度要求 | 测试容差设定 |

## Read design.md

从 `design.md` 读取：

| 字段 | 用途 |
|------|------|
| API 映射设计 | `{op}_impl.py` 核心实现 |
| 数据规格设计 | `test_{op}.py` 数据生成 + `{op}_impl.py` tensor 描述 |
| Tiling 策略 | `{op}_impl.py` tiling 配置 |
| Loop 结构设计 | `{op}_impl.py` kernel 逻辑 |
| 验证方案 | `test_{op}.py` 测试用例设计 |
| 性能指标 | `README.md` 性能说明 |

## Read {op}_golden.py

从 `{op}_golden.py` 读取：

| 信息 | 用途 |
|------|------|
| golden 函数名 `{op}_golden()` | `test_{op}.py` import 语句 |
| 参数签名 | `test_{op}.py` 数据准备 |
| 输出形式 | `{op}_impl.py` 输出 tensor 构造 |

---

## Generate test_{op}.py

torch golden函数实现，基于固定模板 `references/test-template.py` 生成。

### 结构

| 区块 | 内容 |
|------|------|
| Import | `from {op}_golden import {op}_golden` + `from {op}_impl import {op}_wrapper` |
| 环境工具 | `get_device_id()` — 读取 `TILE_FWK_DEVICE_ID` |
| 测试函数 | `test_{op}_levelN()` — 数据生成 → `{op}_wrapper(x)` → `{op}_golden(x)` → `assert_allclose` |
| CLI 入口 | `argparse`，支持 `example_id` / `--list` / `--run_mode` |

### 精度对比强制规范

| 规范 | 说明 |
|------|------|
| 精度对比 | 必须使用 `from numpy.testing import assert_allclose` |
| 容差 | 简单算子 `rtol=1e-3, atol=1e-3`；复杂算子 `rtol=3e-3, atol=3e-3` |
| NPU 条件对比 | `if run_mode == "npu": assert_allclose(...)` |
| 禁止手写对比 | `assert max_diff < tolerance` / `np.allclose()` 均禁止 |

`assert_allclose` 抛出的 `AssertionError` 包含 `Not equal to tolerance` 关键字，orchestrator 据此区分"运行失败"和"精度失败"。

---

## Generate {op}_impl.py

PyPTO kernel函数实现，基于固定模板 `references/impl-template.py` 生成。

| 规范 | 说明 |
|------|------|
| 导出函数 | `{op}_wrapper(x: torch.Tensor) -> torch.Tensor` |
| Kernel 装饰器 | `@pypto.frontend.jit` |
| Tensor 描述符 | `pypto.Tensor()` 或 `pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)` |
| Tiling 配置 | 必须调用 `pypto.set_vec_tile_shapes(...)` 或 `pypto.set_cube_tile_shapes(...)` |
| 输出写回 | `output[:] = result` 或 `pypto.assemble(result, offset, output)` |
| 可选辅助函数 | `{op}_core()` — 复杂算子拆分核心计算逻辑 |

---

## Generate README.md

必须包含：

- 中文说明
- 算子概述与公式
- 目录结构
- 运行方式
- 验证入口
- 已知限制

---

## Design 到代码文件的映射

| design 章节 | `test_{op}.py` | `{op}_impl.py` | `README.md` |
|------------|----------------|----------------|-------------|
| 概述 | 间接引用 | 否 | 是 |
| API 映射设计 | 否 | 是 | 可摘要 |
| 数据规格设计 | 是 | 是 | 可摘要 |
| Tiling 策略 | 否 | 是 | 可摘要 |
| Loop 结构设计 | 否 | 是 | 可摘要 |
| 验证方案 | 是 | 否 | 是 |
| 性能指标 | 部分 | 部分 | 是 |
| 交付件清单 | 是 | 是 | 是 |

---

## First Run and Minimum Pass Criteria

1. 3 个文件（`test_{op}.py` + `{op}_impl.py` + `README.md`）全部存在
2. `test_{op}.py` 可执行（无语法错误）
3. 精度判定由 orchestrator 负责（非本 skill 职责）

## 注意事项

1. **BFloat16 转 NumPy 失败**：必须先 `.float()` 再 `.numpy()`
2. **环境变量未设置**：需要 `export TILE_FWK_DEVICE_ID=0`
3. **动态轴定义位置错误**：必须在 jit 函数外部定义
4. **Tile Shape 未设置**：matmul 前必须调用 `set_cube_tile_shapes`
5. **使用 PyTorch 作为 Golden**：golden 必须独立在 `{op}_golden.py`，使用纯 torch 实现
6. **使用 PyPTO 接口实现 kernel 函数**：kernel 函数必须使用纯 pypto 实现
