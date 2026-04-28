#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KernelBench 用例加载器 (上游 PyTorch 扁平布局).

数据集来源: https://github.com/ScalingIntelligence/KernelBench
固定 commit: 21fbe5a642898cd60b8f60c7aefb43d475e11f33. 由
``scripts/download_kernelbench.sh`` 落到
``pypto/benchmark/.cache/KernelBench/``, 布局为
``KernelBench/<level>/{N}_{name}.py`` 的扁平结构, 每个 .py 内含一个
``class Model(nn.Module)`` + ``get_inputs()`` + ``get_init_inputs()``.

负责把一个 KernelBench 用例 .py 解析成两个产物:

1. ``task_desc``: 原始源码字符串, 直接喂给 ``KernelVerifier``
   (作为 ``framework_code`` 参数).
2. ``SPEC.md``: 自然语言 + 半结构化的算子规格, 喂给 pypto 7 阶段 agent
   工作流.

设计要点:
- 优先 AST 解析 (无副作用); 形状/dtype 推断走"在子进程中真实执行
  ``get_inputs()`` 并打印 shape/dtype" 以避免 torch / numpy 表达式自行求值
  的复杂度, 同时不污染主进程.
- 对解析失败的字段做 best-effort fallback: 即便没拿到 shape, SPEC.md 仍可
  落地, 让 pypto 工作流自己按源码推断.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 数据模型
# ────────────────────────────────────────────────────────────

@dataclass
class TensorSpec:
    """单个输入/输出张量的规格."""
    name: str = ""
    shape: Optional[List[int]] = None
    dtype: str = ""


@dataclass
class CaseSpec:
    """KernelBench 用例派生的结构化规格."""
    op_name: str
    case_id: str                           # 上游文件 stem, 例如 "19_relu"
    source_file: str                       # 绝对路径
    task_desc: str                         # 原始源码 (KernelBench 风格)
    level: str = ""                        # KernelBench level, 例如 level1 / level2
    framework_module: str = "torch"        # 上游 KernelBench 一律 torch; 探针时若 import 不同, 会被覆盖
    init_source: str = ""                  # Model.__init__ 的源码片段
    forward_source: str = ""               # Model.forward / __call__ 的源码片段
    init_args_repr: str = "[]"             # get_init_inputs() 的 repr
    inputs: List[TensorSpec] = field(default_factory=list)
    supported_dtypes: List[str] = field(default_factory=lambda: ["float32"])
    p0_shapes: List[List[int]] = field(default_factory=list)
    tolerance: Dict[str, float] = field(
        default_factory=lambda: {"rtol": 1e-3, "atol": 1e-3}
    )
    dynamic_axis: Optional[List[str]] = None
    formula: str = ""


# ────────────────────────────────────────────────────────────
# AST 解析
# ────────────────────────────────────────────────────────────

def _detect_framework(tree: ast.Module) -> str:
    """根据 import 语句推断框架; 默认 numpy."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    return "torch"
                if alias.name == "mindspore":
                    return "mindspore"
                if alias.name == "numpy":
                    return "numpy"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torch"):
                return "torch"
            if node.module and node.module.startswith("mindspore"):
                return "mindspore"
    return "numpy"


def _extract_model_method_source(tree: ast.Module, source: str,
                                 *method_names: str) -> str:
    """提取 ``Model`` 类里指定方法的源码片段."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in method_names:
                    return ast.get_source_segment(source, item) or ""
    return ""


def _has_kernelbench_layout(tree: ast.Module) -> List[str]:
    """返回缺失的关键组件列表; 空列表表示合规."""
    has_model = False
    has_inputs = False
    has_init = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            has_model = True
        if isinstance(node, ast.FunctionDef):
            if node.name == "get_inputs":
                has_inputs = True
            elif node.name == "get_init_inputs":
                has_init = True
    missing: List[str] = []
    if not has_model:
        missing.append("class Model")
    if not has_inputs:
        missing.append("def get_inputs")
    if not has_init:
        missing.append("def get_init_inputs")
    return missing


def _extract_new_interface_globals(tree: ast.Module) -> tuple[str, Optional[List[str]]]:
    """提取新增 KernelBench case 顶层接口: ``FORMULA`` / ``DYNAMIC_AXIS``.

    旧 case 没有这两个全局变量时保持空值, SPEC.md 渲染时不会输出对应字段。
    """
    formula = ""
    dynamic_axis: Optional[List[str]] = None
    for node in tree.body:
        targets: List[ast.expr]
        value_node: Optional[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if value_node is None:
            continue

        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not names:
            continue

        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            continue

        if "FORMULA" in names and isinstance(value, str):
            formula = value.strip()
        if "DYNAMIC_AXIS" in names and isinstance(value, (list, tuple)):
            axis = [str(item) for item in value]
            if axis:
                dynamic_axis = axis
    return formula, dynamic_axis


# ────────────────────────────────────────────────────────────
# 子进程探针: 跑 get_inputs() / get_init_inputs() 拿到 shape & dtype
# ────────────────────────────────────────────────────────────

_PROBE_TEMPLATE = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("kb_case", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def _shape_dtype(obj):
    shape = list(getattr(obj, "shape", ()))
    shape = [int(s) for s in shape]
    dtype = str(getattr(obj, "dtype", type(obj).__name__))
    return shape, dtype

inputs_info = []
try:
    inputs = mod.get_inputs()
    for i, t in enumerate(inputs):
        shp, dt = _shape_dtype(t)
        inputs_info.append({{"name": f"x{{i}}", "shape": shp, "dtype": dt}})
except Exception as e:
    inputs_info.append({{"error": str(e)}})

init_repr = "[]"
try:
    init_repr = repr(mod.get_init_inputs())
except Exception as e:
    init_repr = f"<unavailable: {{e}}>"

print("__PROBE_RESULT__")
print(json.dumps({{"inputs": inputs_info, "init_args_repr": init_repr}}))
"""


def _probe_inputs(case_path: Path, timeout_sec: int = 30) -> tuple[List[TensorSpec], str]:
    """在子进程中执行 ``get_inputs()`` 并捕获 shape/dtype.

    失败时返回空列表 + ``"[]"`` 作为 fallback, 不抛异常 (静默降级到 SPEC 自行推断).
    """
    script = _PROBE_TEMPLATE.format(path=str(case_path))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "[]"

    if proc.returncode != 0:
        return [], "[]"

    marker = "__PROBE_RESULT__"
    if marker not in proc.stdout:
        return [], "[]"
    payload = proc.stdout.split(marker, 1)[1].strip()
    try:
        data = json.loads(payload.splitlines()[0])
    except (ValueError, IndexError):
        return [], "[]"

    inputs: List[TensorSpec] = []
    for entry in data.get("inputs", []):
        if "error" in entry:
            continue
        inputs.append(TensorSpec(
            name=entry.get("name", ""),
            shape=entry.get("shape"),
            dtype=str(entry.get("dtype", "")),
        ))
    return inputs, str(data.get("init_args_repr", "[]"))


# ────────────────────────────────────────────────────────────
# op 名规范化
# ────────────────────────────────────────────────────────────

_OP_NAME_RE = re.compile(r"\W+")


def derive_op_name(case_id: str) -> str:
    """从 ``19_relu`` / ``1_square_matrix_multiplication_`` 等 case id 推一个合法 Python 标识符.

    - 去掉前缀数字 + 下划线 (``19_relu`` → ``relu``)
    - 非法字符替换为 ``_``
    - 收尾下划线裁掉
    - 若以数字开头则前缀 ``op_``
    """
    s = case_id.strip()
    s = re.sub(r"^\d+_", "", s)
    s = _OP_NAME_RE.sub("_", s).strip("_")
    if not s:
        s = "op"
    if s[0].isdigit():
        s = f"op_{s}"
    return s


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

def load_case(case_path: Path, op_name: Optional[str] = None,
              case_id: Optional[str] = None,
              probe_timeout_sec: int = 30) -> CaseSpec:
    """加载并解析一个 KernelBench 用例 (上游 PyTorch 扁平布局).

    Args:
        case_path: 用例 ``.py`` 绝对/相对路径; 上游 KernelBench 中即
            ``KernelBench/<level>/{N}_{name}.py``.
        op_name: 可选, 指定算子名; 缺省时按 ``case_id`` 推导.
        case_id: 可选, 用例标识; 缺省时取 ``case_path.stem``
            (上游扁平布局下文件名即标识).
        probe_timeout_sec: 子进程执行 ``get_inputs()`` 的超时.

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError: 文件不符合 KernelBench 格式 (缺 ``Model`` /
            ``get_inputs`` / ``get_init_inputs``).
    """
    case_path = case_path.resolve()
    if not case_path.exists():
        raise FileNotFoundError(f"KernelBench case not found: {case_path}")

    source = case_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    missing = _has_kernelbench_layout(tree)
    if missing:
        raise ValueError(
            f"{case_path} 不符合 KernelBench 格式, 缺少: {', '.join(missing)}"
        )

    if not case_id:
        case_id = case_path.stem
    if not op_name:
        op_name = derive_op_name(case_id)

    framework = _detect_framework(tree)
    init_src = _extract_model_method_source(tree, source, "__init__")
    forward_src = _extract_model_method_source(tree, source, "__call__", "forward")
    formula, dynamic_axis = _extract_new_interface_globals(tree)
    inputs, init_repr = _probe_inputs(case_path, timeout_sec=probe_timeout_sec)
    supported_dtypes, p0_shapes, tolerance = _derive_front_matter_fields(inputs)

    return CaseSpec(
        op_name=op_name,
        case_id=case_id,
        source_file=str(case_path),
        task_desc=source,
        level=case_path.parent.name,
        framework_module=framework,
        init_source=init_src,
        forward_source=forward_src,
        init_args_repr=init_repr,
        inputs=inputs,
        supported_dtypes=supported_dtypes,
        p0_shapes=p0_shapes,
        tolerance=tolerance,
        dynamic_axis=dynamic_axis,
        formula=formula,
    )


# ────────────────────────────────────────────────────────────
# SPEC.md 渲染
# ────────────────────────────────────────────────────────────

_SPEC_TEMPLATE = """\
---
schema_version: 1
op_name: {op_name}
supported_dtypes: {supported_dtypes_json}
p0_shapes: {p0_shapes_json}
tolerance: {tolerance_json}
{dynamic_axis_front_matter}---

# {op_name} 算子需求规格 (派生自上游 KernelBench)

> 本 SPEC 由 ``benchmark.case_loader`` 自动生成, 用于驱动
> ``pypto-op-orchestrator`` 7 阶段工作流.
>
> 数据集来源: github.com/ScalingIntelligence/KernelBench @ 21fbe5a

## 元数据

- **算子名 (op_name)**: `{op_name}`
- **来源用例 (case_id)**: `{case_id}`
- **来源 level**: `{level}`
- **来源文件**: `{source_file}`
- **参考框架 (framework)**: `{framework}`

{formula_section}
## 输入规格

{inputs_section}

## 初始化参数 (get_init_inputs)

```python
init_args = {init_args_repr}
```

## KernelBench 调用约定

```python
model = Model(*get_init_inputs())
outputs = model(*get_inputs())
```

- `get_init_inputs()` 与 `get_inputs()` 是两段不同的调用面.
- 若 PyPTO wrapper 需要消费 init 参数, 应由 `ModelNew.__init__` 保存, 并在
  `ModelNew.forward()` 内部按正确顺序转发给 wrapper.
- 禁止要求下游验证器把 init 参数和 forward 输入拍平成一个外部调用接口.

## 构造逻辑 (Model.__init__ 参考实现)

```python
{init_source}
```

## 计算逻辑 (Model 参考实现)

```python
{forward_source}
```

## 精度要求

- 默认: ``rtol=1e-3, atol=1e-3`` (FP32) / ``rtol=4e-3, atol=4e-3`` (FP16/BF16).
- 验证通过条件: ``test_{op_name}.py`` 输出 ``[PRECISION_PASS]``.
- 桥接层 KernelVerifier 端按 ``mode=correctness`` 复测.

## 算子开发约束

1. 必须导出 ``{op_name}_wrapper(...) -> torch.Tensor``.
2. 对外桥接后的调用约定必须与 KernelBench 一致:
   ``ModelNew(*get_init_inputs()).forward(*get_inputs())`` 必须可用.
3. 若存在 init 参数, 不得要求外部把 init_args 和 forward inputs 错误拍平后再调用 wrapper.
4. 本算子由外部 KernelBench 桥接消费; 调用方会在 prompt 中要求额外产出
   ``{op_name}_pypto_impl.py`` (含 ``ModelNew`` 类), 文件契约以 prompt 为准,
   本 SPEC 不重复声明.
5. golden / impl / test 三文件分离.
6. 输入/输出 dtype 必须与原 KernelBench 用例一致.

## 原始 KernelBench 任务描述 (task_desc)

```python
{task_desc}
```
"""


def _render_inputs_section(inputs: List[TensorSpec]) -> str:
    if not inputs:
        return ("> 输入 shape/dtype 探针执行失败 (子进程超时或环境缺依赖). "
                "请由 pypto-intent-understand 从下方 task_desc 自行推断.\n")
    lines = ["| # | name | shape | dtype |", "|---|------|-------|-------|"]
    for i, spec in enumerate(inputs):
        shp = "x".join(str(s) for s in (spec.shape or [])) or "scalar"
        lines.append(f"| {i} | `{spec.name}` | `{shp}` | `{spec.dtype}` |")
    return "\n".join(lines) + "\n"


def _render_dynamic_axis_front_matter(dynamic_axis: Optional[List[str]]) -> str:
    if not dynamic_axis:
        return ""
    return f"dynamic_axis: {json.dumps(dynamic_axis, ensure_ascii=False)}\n"


def _render_formula_section(formula: str) -> str:
    formula = textwrap.dedent(formula or "").strip()
    if not formula:
        return ""
    return (
        "### 1.3 数学公式\n\n"
        "```text\n"
        f"{formula}\n"
        "```\n\n"
    )


def _normalize_dtype(dtype: str) -> str:
    """把 ``torch.float32`` / ``numpy.float32`` 等归一成 front matter dtype."""
    value = str(dtype or "").strip()
    if not value:
        return ""
    value = value.replace("torch.", "")
    value = value.replace("mindspore.", "")
    value = value.replace("numpy.", "")
    match = re.search(
        r"float(?:16|32|64)|bfloat16|int(?:8|16|32|64)|uint8|bool",
        value,
    )
    return match.group(0) if match else value


def _derive_front_matter_fields(
    inputs: List[TensorSpec],
) -> tuple[List[str], List[List[int]], Dict[str, float]]:
    """从探针输入规格派生 SPEC.md YAML front matter 字段."""
    supported_dtypes: List[str] = []
    seen_dtypes = set()
    for spec in inputs:
        dtype = _normalize_dtype(spec.dtype)
        if dtype and dtype not in seen_dtypes:
            seen_dtypes.add(dtype)
            supported_dtypes.append(dtype)
    if not supported_dtypes:
        supported_dtypes = ["float32"]

    p0_shapes = [
        [int(dim) for dim in spec.shape]
        for spec in inputs
        if spec.shape
    ]

    has_low_precision = any(dt in ("float16", "bfloat16") for dt in supported_dtypes)
    tolerance = (
        {"rtol": 4e-3, "atol": 4e-3}
        if has_low_precision else
        {"rtol": 1e-3, "atol": 1e-3}
    )
    return supported_dtypes, p0_shapes, tolerance


def render_spec_md(case: CaseSpec) -> str:
    """把 ``CaseSpec`` 渲染成 SPEC.md 文本."""
    supported_dtypes, p0_shapes, tolerance = _derive_front_matter_fields(
        case.inputs
    )
    return _SPEC_TEMPLATE.format(
        op_name=case.op_name,
        supported_dtypes_json=json.dumps(supported_dtypes, ensure_ascii=False),
        p0_shapes_json=json.dumps(p0_shapes, ensure_ascii=False),
        tolerance_json=json.dumps(tolerance, ensure_ascii=False),
        dynamic_axis_front_matter=_render_dynamic_axis_front_matter(case.dynamic_axis),
        case_id=case.case_id,
        level=case.level,
        source_file=case.source_file,
        framework=case.framework_module,
        formula_section=_render_formula_section(case.formula),
        inputs_section=_render_inputs_section(case.inputs),
        init_args_repr=case.init_args_repr,
        init_source=textwrap.dedent(case.init_source).strip() or "# (未提取到 __init__ 源码)",
        forward_source=textwrap.dedent(case.forward_source).strip() or "# (未提取到 forward 源码)",
        task_desc=case.task_desc.strip(),
    )


def write_spec(case: CaseSpec, workdir: Path) -> Path:
    """把 SPEC.md 写到 ``workdir/{op}/SPEC.md`` 并返回路径.

    若文件已存在且内容一致, 不重写以利于断点续跑.
    """
    op_dir = workdir / case.op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    spec_path = op_dir / "SPEC.md"
    new_content = render_spec_md(case)
    if spec_path.exists() and spec_path.read_text(encoding="utf-8") == new_content:
        return spec_path
    spec_path.write_text(new_content, encoding="utf-8")
    return spec_path


def write_task_desc(case: CaseSpec, workdir: Path) -> Path:
    """把原始 KernelBench task_desc 缓存到 ``workdir/{op}/task_desc.py``.

    给 ``verifier_runner`` 直接读取使用, 避免再次 IO 原 KernelBench 路径.
    """
    op_dir = workdir / case.op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    out = op_dir / "task_desc.py"
    out.write_text(case.task_desc, encoding="utf-8")
    return out


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a KernelBench case → CaseSpec / SPEC.md")
    parser.add_argument("case_file", type=Path, help="Path to KernelBench .py")
    parser.add_argument("--op-name", type=str, default=None)
    parser.add_argument("--write", type=Path, default=None,
                        help="写出 SPEC.md 到此目录的 {op}/SPEC.md")
    parser.add_argument("--probe-timeout", type=int, default=30)
    args = parser.parse_args()

    case = load_case(args.case_file, op_name=args.op_name,
                     probe_timeout_sec=args.probe_timeout)
    payload = asdict(case)
    payload["task_desc"] = f"<{len(case.task_desc)} chars>"
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.write:
        spec_path = write_spec(case, args.write)
        td_path = write_task_desc(case, args.write)
        logger.info("Wrote: %s", spec_path)
        logger.info("Wrote: %s", td_path)
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())
