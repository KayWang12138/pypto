# 新增 KernelBench Case 指南

本文档说明如何为 `benchmark` 增加新的 KernelBench 风格 case，并让
`case_loader.py` 自动生成符合 PyPTO 工作流要求的 `SPEC.md`。

## 1. 放置位置

仓内自维护 case 放在：

```text
benchmark/testdata/KernelBench/<level>/<N>_<Name>.py
```

示例：

```text
benchmark/testdata/KernelBench/level1/101_DynamicAxisAdd.py
```

命名要求：

- `<level>` 使用 `level1` / `level2` / `level3`。
- 文件名使用 KernelBench 扁平布局：`<序号>_<CaseName>.py`。
- 序号建议从当前 level 未占用编号继续递增，例如 `101_...`。

## 2. 必需代码结构

每个 case 必须包含以下 3 个 KernelBench 标准入口：

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, *init_args):
        super().__init__()

    def forward(self, *inputs):
        ...


def get_inputs():
    return [...]


def get_init_inputs():
    return [...]
```

约束：

- `Model(*get_init_inputs())(*get_inputs())` 必须能直接运行。
- `get_inputs()` 返回 forward 输入列表。
- `get_init_inputs()` 返回构造参数列表；没有参数时返回 `[]`。
- 输入 tensor 的 shape/dtype 应尽量固定、可复现，便于 `case_loader.py` 探针生成 `p0_shapes` 和 `supported_dtypes`。

## 3. 新接口：公式和动态轴

新增 case 如果能提供数学语义和动态轴，请在文件顶层添加两个大写全局变量：

```python
FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
DYNAMIC_AXIS = ["B", "S"]
```

字段含义：

- `FORMULA`: 数学公式或简洁计算语义。`case_loader.py` 会写入 `SPEC.md` 的 `### 1.3 数学公式` 小节。
- `DYNAMIC_AXIS`: 动态轴名称列表，例如 `["B", "S"]`。`case_loader.py` 会写入 `SPEC.md` front matter 的 `dynamic_axis` 字段。

兼容规则：

- 老 KernelBench case 没有这两个变量时，不会输出对应内容。
- 新 case 有其中一个变量时，只输出对应部分。
- 变量必须是可被 `ast.literal_eval` 解析的常量表达式；不要写成运行时计算结果。

## 4. 推荐模板

```python
#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn


FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


def get_inputs():
    batch = 2
    seq_len = 4
    hidden_size = 8
    x = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
    bias = torch.randn(hidden_size, dtype=torch.float32)
    return [x, bias]


def get_init_inputs():
    return [8]
```

## 5. 本地校验

先确认 case 能被 loader 解析，并检查生成的 `SPEC.md` 内容：

```bash
python3 -m benchmark.case_loader \
  benchmark/testdata/KernelBench/level1/101_DynamicAxisAdd.py \
  --write /tmp/pypto_case_check
```

检查输出：

```bash
sed -n '1,80p' /tmp/pypto_case_check/DynamicAxisAdd/SPEC.md
```

应看到：

```yaml
---
schema_version: 1
op_name: DynamicAxisAdd
supported_dtypes: ["float32"]
p0_shapes: [[2, 4, 8], [8]]
tolerance: {"rtol": 0.001, "atol": 0.001}
dynamic_axis: ["B", "S"]
---
```

正文中应包含：

```text
### 1.3 数学公式
```

## 6. 运行 benchmark

用仓内 testdata 跑新增 case：

```bash
python3 -m benchmark.run_kernelbench \
  --bench-dir benchmark/testdata/KernelBench \
  --level level1 \
  --cases 101_DynamicAxisAdd \
  --devices 0 \
  --mode correctness \
  --skip-stage7-perf-tune
```

也可以只用序号选择：

```bash
python3 -m benchmark.run_kernelbench \
  --bench-dir benchmark/testdata/KernelBench \
  --level level1 \
  --cases 101 \
  --devices 0 \
  --mode correctness \
  --skip-stage7-perf-tune
```

## 7. 提交前检查

建议至少执行：

```bash
python3 -m py_compile \
  benchmark/case_loader.py \
  benchmark/testdata/KernelBench/level1/101_DynamicAxisAdd.py

python3 -m benchmark.case_loader \
  benchmark/testdata/KernelBench/level1/101_DynamicAxisAdd.py \
  --write /tmp/pypto_case_check
```

若环境安装了仓库 pytest 依赖，可继续执行：

```bash
python3 -m pytest benchmark/tests/test_case_loader_front_matter.py
```
