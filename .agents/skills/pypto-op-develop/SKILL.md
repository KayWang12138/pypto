---
name: pypto-op-develop
description: "读取 spec.md、design.md 和 {op}_golden.py，产出完整可运行的 PyPTO 算子：{op}_impl.py（kernel 实现，导出 {op}_wrapper()）、test_{op}.py（测试入口）、README.md。当需要编写 PyPTO 算子实现时使用此 skill。Triggers: 实现算子、写 kernel、编写实现、写 impl、算子编码、开始编码、code the op、写 test、生成测试、写实现代码、op develop、算子开发、kernel 实现。"
---

# PyPTO 算子功能实现

读取 `spec.md`、`design.md`、`{op}_golden.py`，产出一个能正常运行的 PyPTO 算子实现。

## Contract

### Inputs

- 算子需求规格（spec.md 内容）
- 设计方案（design.md 内容）
- 纯 torch 参考实现（`{op}_golden.py`）
- 可选：已有的历史实现文件（续跑场景）

### Outputs

- `test_{op}.py` — 测试入口（从 golden 和 impl 导入，不含实现代码），路径由调用者决定
- `{op}_impl.py` — PyPTO kernel 实现（必须使用pypto接口实现，导出 `{op}_wrapper()` 函数），路径由调用者决定
- `README.md` — PyPTO 算子文档，路径由调用者决定

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

## 实现注意点

1. **PyPTO tensor 初始化与 torch 不同**：不要机械照搬 `torch.zeros` / `torch.empty` 的习惯，优先按照 PyPTO 张量描述与后续完整写回方式组织代码。
2. **禁止无中生有 op**：实现时只能使用 PyPTO 已支持的 API，遇到缺失能力应回退到 API 探索或设计阶段重新确认。
3. **优先使用当前项目推荐的 `@pypto.frontend.jit` 写法**：实现应与现有示例和当前文档保持一致，不要混用过时包装方式。
4. **golden / impl / test 必须职责分离**：不要把 golden 逻辑、实现逻辑和测试逻辑混写到同一个文件中。

## 动态与循环实现提醒

- 动态 shape 场景下，`pypto.view` / `pypto.reshape` 需要认真检查 `valid_shape`
- 多层循环场景下，谨慎使用 `unroll_list`，尤其注意是否只放在最内层循环
- matmul / cube 场景必须确认 `set_cube_tile_shapes(...)` 已正确配置
- 如果设计中已有 tiling / loop 约束，编码时优先遵循 `design.md`，不要临时拍脑袋改写

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

## 推荐验证顺序

1. 小规模功能验证：先确认代码能运行、基础输出形状正确
2. 典型规模验证：使用目标场景下的常规输入做主路径验证
3. 边界与极值验证：检查零值、大值、特殊 shape、动态 shape 等情况
4. 大规模或性能相关验证：在功能和精度稳定后再推进性能相关验证

## 注意事项

1. **BFloat16 转 NumPy 失败**：必须先 `.float()` 再 `.numpy()`
2. **环境变量未设置**：需要 `export TILE_FWK_DEVICE_ID=0`
3. **动态轴定义位置错误**：必须在 jit 函数外部定义
4. **Tile Shape 未设置**：matmul 前必须调用 `set_cube_tile_shapes`
5. **使用 PyTorch 作为 Golden**：golden 必须独立在 `{op}_golden.py`，使用纯 torch 实现
6. **使用 PyPTO 接口实现 kernel 函数**：kernel 函数必须使用纯 pypto 实现

---

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

标记含义：
- `[PRECISION_PASS]`: 精度验证通过
- `[PRECISION_FAIL]`: 精度验证失败（数值不匹配）
- 无标记 + exit ≠ 0: 功能问题（代码崩溃、逻辑错误等）

---

## 独立使用

当用户直接调用本 Skill 时：
1. 从用户输入提取算子名称，获取 spec、design、golden 等必要信息
2. 如果信息不足，向用户逐步提问补充
3. 按工作流执行代码生成（test + impl + README）
4. 输出到当前目录或用户指定位置
