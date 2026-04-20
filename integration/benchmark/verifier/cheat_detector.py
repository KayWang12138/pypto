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
"""PyPTO 算子产物的机械层反作弊检测.

仅做**确定性**检查 — AST 计数 + 文件系统枚举 + 字符串匹配, 不依赖 LLM.
语义层的隐性作弊判定 (jit 函数体是 pass-through, forward 走 if-branch
绕开 pypto, 等等) 由 ``.opencode/skills/pypto-kernel-validate`` 让 LLM
亲自审阅, 与本检测互补不替代.

机械检查项:
    - ``import_pypto``       源码中存在 ``import pypto`` 或 ``from pypto``.
    - ``has_jit``            源码中至少有一个 ``@pypto.jit`` (装饰器或函数式调用).
    - ``jit_def_count``      ``@pypto.jit`` 装饰器数量必须 ≤ 1 (融合算子约定).
    - ``jit_call_count``     ``forward`` 中 jit 函数调用次数 ≤ 1 (best-effort 启发).
    - ``swimlane_dirs``      若已跑过 profile, ``prof_generation_output_*`` 下的
                             ``output_*`` 子目录数 ≤ 1.
    - ``forbidden_patterns`` 注释/字符串里出现 "for testing", "TODO use pypto",
                             "fallback to torch", "workaround" 等可疑文本时
                             标记为 SUSPICIOUS (不直接判 CHEAT, 让 LLM 复核).

裁定层级:
    - ``pass``       —— 检查通过.
    - ``suspicious`` —— 触发软警告, 需要 LLM 进一步审阅.
    - ``cheat``      —— 触发硬铁证, 直接判作弊.

CLI 用法:
    python -m integration.benchmark.verifier.cheat_detector \\
        <op_dir> [--op-name <name>] [--json-out <file>]

输出 JSON schema:
    {
      "op_name": "...",
      "op_dir": "...",
      "verdict": "pass | suspicious | cheat",
      "checks": [
        {"name": "...", "status": "pass|fail|warn|skip",
         "level": "info|warn|fatal", "detail": "..."},
        ...
      ],
      "summary": "<人类可读总结>"
    }
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 数据模型
# ────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str           # "pass" | "fail" | "warn" | "skip"
    level: str            # "info" | "warn" | "fatal"
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheatReport:
    op_name: str
    op_dir: str
    verdict: str          # "pass" | "suspicious" | "cheat"
    checks: List[CheckResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ────────────────────────────────────────────────────────────
# 静态文本分析
# ────────────────────────────────────────────────────────────

_FORBIDDEN_TEXT_PATTERNS = [
    re.compile(r"for\s+testing\s+only", re.IGNORECASE),
    re.compile(r"TODO\s*[:\-]?\s*use\s+pypto", re.IGNORECASE),
    re.compile(r"fallback\s+to\s+torch", re.IGNORECASE),
    re.compile(r"workaround", re.IGNORECASE),
    re.compile(r"bypass\s+pypto", re.IGNORECASE),
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _is_pypto_frontend_jit_attr(node: ast.Attribute) -> bool:
    """``pypto.frontend.jit`` 属性链."""
    val = node.value
    return (
        node.attr == "jit"
        and isinstance(val, ast.Attribute)
        and val.attr == "frontend"
        and isinstance(val.value, ast.Name)
        and val.value.id == "pypto"
    )


def _is_pypto_jit_attr(node: ast.AST) -> bool:
    """判断 AST 节点是否引用 ``pypto.jit`` / ``pypto.frontend.jit``."""
    if isinstance(node, ast.Attribute) and node.attr == "jit":
        if isinstance(node.value, ast.Name) and node.value.id == "pypto":
            return True
        if _is_pypto_frontend_jit_attr(node):
            return True
    if isinstance(node, ast.Name) and node.id == "jit":
        # `from pypto import jit` 后裸用; best-effort 接受.
        return True
    return False


def _count_jit_decorators(tree: ast.Module) -> List[str]:
    """枚举所有以 @pypto.jit / @pypto.frontend.jit 装饰的函数名."""
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if _is_pypto_jit_attr(target):
                    names.append(node.name)
                    break
    return names


def _count_jit_function_calls(tree: ast.Module) -> int:
    """统计 ``pypto.jit(fn)`` 这种函数式调用的出现次数 (非装饰器形式)."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_pypto_jit_attr(node.func):
            count += 1
    return count


def _has_pypto_import(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "pypto" or a.name.startswith("pypto."):
                    return True
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "pypto" or mod.startswith("pypto."):
                return True
    return False


def _find_forward_jit_calls(tree: ast.Module, jit_func_names: List[str]) -> int:
    """启发式: 数 ``ModelNew.forward`` (或任何叫 forward 的方法) 里调用 jit
    装饰过的函数名出现次数."""
    if not jit_func_names:
        return 0
    target = set(jit_func_names)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name) and fn.id in target:
                        count += 1
                    elif isinstance(fn, ast.Attribute) and fn.attr in target:
                        count += 1
    return count


def _scan_forbidden_text(src: str) -> List[str]:
    hits: List[str] = []
    for pat in _FORBIDDEN_TEXT_PATTERNS:
        m = pat.search(src)
        if m:
            hits.append(m.group(0))
    return hits


# ────────────────────────────────────────────────────────────
# 文件系统检查
# ────────────────────────────────────────────────────────────

def _list_swimlane_kernel_dirs(op_dir: Path) -> List[Path]:
    """枚举 op_dir 下任何 ``prof_generation_output_*`` 内的 ``output_*`` 子目录.

    PyPTO profile 在 runtime_debug_mode=1 下, 每个 jit kernel 写一个独立子目录.
    """
    found: List[Path] = []
    for prof in glob.glob(str(op_dir / "prof_generation_output_*")):
        for sub in glob.glob(os.path.join(prof, "output_*")):
            if os.path.isdir(sub):
                found.append(Path(sub))
    return found


# ────────────────────────────────────────────────────────────
# 主检查逻辑
# ────────────────────────────────────────────────────────────

def _candidate_source_files(op_dir: Path, op_name: str) -> List[Path]:
    """收集 op_dir 下要扫描的 PyPTO 实现源码文件.

    优先 ``{op}_impl.py`` (核心算子), ``{op}_pypto_impl.py`` (ModelNew 包装).
    遇到 fallback 场景 (只有 _impl.py) 也兼容.
    """
    candidates = [
        op_dir / f"{op_name}_impl.py",
        op_dir / f"{op_name}_pypto_impl.py",
    ]
    return [p for p in candidates if p.exists()]


def _parse_module(path: Path) -> Optional[ast.Module]:
    src = _read_text(path)
    if not src:
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def detect_cheats(op_dir: Path, op_name: str) -> CheatReport:
    """对单个算子产物目录执行机械层反作弊检测."""
    op_dir = op_dir.resolve()
    report = CheatReport(op_name=op_name, op_dir=str(op_dir), verdict="pass")

    sources = _candidate_source_files(op_dir, op_name)
    if not sources:
        report.checks.append(CheckResult(
            name="sources_present",
            status="fail",
            level="fatal",
            detail=(
                f"No source files found in {op_dir}; "
                f"expected at least one of {op_name}_impl.py / {op_name}_pypto_impl.py."
            ),
        ))
        report.verdict = "cheat"
        report.summary = "无源码可分析."
        return report
    report.checks.append(CheckResult(
        name="sources_present",
        status="pass",
        level="info",
        detail=f"Scanning: {', '.join(p.name for p in sources)}.",
    ))

    has_pypto_import = False
    jit_decorated_funcs: List[str] = []
    jit_call_count = 0
    forward_jit_calls = 0
    forbidden_hits: List[str] = []

    for path in sources:
        src = _read_text(path)
        forbidden_hits.extend(_scan_forbidden_text(src))
        tree = _parse_module(path)
        if tree is None:
            report.checks.append(CheckResult(
                name=f"parse:{path.name}",
                status="fail",
                level="fatal",
                detail=f"SyntaxError parsing {path}; cannot analyze.",
            ))
            report.verdict = "cheat"
            continue
        has_pypto_import = has_pypto_import or _has_pypto_import(tree)
        decorated = _count_jit_decorators(tree)
        jit_decorated_funcs.extend(decorated)
        jit_call_count += _count_jit_function_calls(tree)
        forward_jit_calls += _find_forward_jit_calls(tree, decorated)

    if has_pypto_import:
        report.checks.append(CheckResult(
            name="import_pypto",
            status="pass",
            level="info",
            detail="源码中存在 import pypto / from pypto.",
        ))
    else:
        report.checks.append(CheckResult(
            name="import_pypto",
            status="fail",
            level="fatal",
            detail=(
                "源码中找不到 import pypto / from pypto — 这意味着算子可能"
                "完全没用 PyPTO, 而是纯 torch 实现. CHEAT."
            ),
        ))
        report.verdict = "cheat"

    total_jit = len(jit_decorated_funcs) + jit_call_count
    if total_jit == 0:
        report.checks.append(CheckResult(
            name="has_jit",
            status="fail",
            level="fatal",
            detail=(
                "源码中未发现任何 @pypto.jit / pypto.jit(...) 调用 — "
                "算子没有走 PyPTO jit 编译路径. CHEAT."
            ),
        ))
        report.verdict = "cheat"
    else:
        report.checks.append(CheckResult(
            name="has_jit",
            status="pass",
            level="info",
            detail=f"发现 jit 装饰器 {len(jit_decorated_funcs)} 个 + 函数式调用 {jit_call_count} 次.",
            extra={"decorated_funcs": jit_decorated_funcs},
        ))

    if len(jit_decorated_funcs) > 1:
        report.checks.append(CheckResult(
            name="jit_def_count",
            status="fail",
            level="fatal",
            detail=(
                f"@pypto.jit 装饰函数有 {len(jit_decorated_funcs)} 个: "
                f"{jit_decorated_funcs}. PyPTO 算子要求融合 kernel, "
                f"多 jit 定义 = 多 kernel = 没融合 = CHEAT."
            ),
            extra={"jit_funcs": jit_decorated_funcs},
        ))
        report.verdict = "cheat"
    else:
        report.checks.append(CheckResult(
            name="jit_def_count",
            status="pass",
            level="info",
            detail=f"@pypto.jit 装饰函数: {len(jit_decorated_funcs)} 个 (≤1 OK).",
        ))

    if jit_call_count > 1:
        report.checks.append(CheckResult(
            name="jit_func_call_count",
            status="fail",
            level="fatal",
            detail=(
                f"pypto.jit(...) 函数式调用出现 {jit_call_count} 次. "
                f"约定 ≤1; 多次调用 = 多 kernel = CHEAT."
            ),
        ))
        report.verdict = "cheat"
    else:
        report.checks.append(CheckResult(
            name="jit_func_call_count",
            status="pass",
            level="info",
            detail=f"pypto.jit(...) 函数式调用: {jit_call_count} 次.",
        ))

    if forward_jit_calls > 1:
        report.checks.append(CheckResult(
            name="forward_jit_calls",
            status="warn",
            level="warn",
            detail=(
                f"forward 中调用 jit 函数 {forward_jit_calls} 次 (启发式). "
                f"可能算子被拆分; 请 LLM 语义层复核."
            ),
        ))
        if report.verdict == "pass":
            report.verdict = "suspicious"
    else:
        report.checks.append(CheckResult(
            name="forward_jit_calls",
            status="pass",
            level="info",
            detail=f"forward 中调用 jit 函数: {forward_jit_calls} 次 (启发式).",
        ))

    swimlane_dirs = _list_swimlane_kernel_dirs(op_dir)
    if not swimlane_dirs:
        report.checks.append(CheckResult(
            name="swimlane_kernel_dirs",
            status="skip",
            level="info",
            detail="尚未跑过 profile (无 prof_generation_output_*); 跳过运行时多 kernel 检查.",
        ))
    elif len(swimlane_dirs) > 1:
        report.checks.append(CheckResult(
            name="swimlane_kernel_dirs",
            status="fail",
            level="fatal",
            detail=(
                f"profile 产物中发现 {len(swimlane_dirs)} 个 jit kernel 子目录: "
                f"{[str(d.name) for d in swimlane_dirs]}. CHEAT."
            ),
            extra={"dirs": [str(d) for d in swimlane_dirs]},
        ))
        report.verdict = "cheat"
    else:
        report.checks.append(CheckResult(
            name="swimlane_kernel_dirs",
            status="pass",
            level="info",
            detail=f"profile 产物中只有 1 个 jit kernel 子目录 ({swimlane_dirs[0].name}).",
        ))

    if forbidden_hits:
        report.checks.append(CheckResult(
            name="forbidden_text_patterns",
            status="warn",
            level="warn",
            detail=(
                f"源码中出现可疑字符串: {forbidden_hits!r}. "
                f"请 LLM 复核, 可能为占位 / fallback / workaround 实现."
            ),
            extra={"hits": forbidden_hits},
        ))
        if report.verdict == "pass":
            report.verdict = "suspicious"
    else:
        report.checks.append(CheckResult(
            name="forbidden_text_patterns",
            status="pass",
            level="info",
            detail="源码中未出现 testing-only / fallback / workaround 等可疑字符串.",
        ))

    fatal_fails = [c for c in report.checks if c.status == "fail" and c.level == "fatal"]
    warns = [c for c in report.checks if c.status == "warn"]
    if fatal_fails:
        report.summary = (
            f"CHEAT: {len(fatal_fails)} 项硬铁证不通过 — "
            f"{', '.join(c.name for c in fatal_fails)}."
        )
    elif warns:
        report.summary = (
            f"SUSPICIOUS: {len(warns)} 项软警告需要 LLM 语义层复核 — "
            f"{', '.join(c.name for c in warns)}."
        )
    else:
        report.summary = "PASS: 机械层未发现作弊迹象 (语义层仍需 LLM 复核)."

    return report


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m integration.benchmark.verifier.cheat_detector",
        description="PyPTO 算子产物机械层反作弊检测.",
    )
    p.add_argument("op_dir", type=Path,
                   help="算子产物目录 (含 {op}_impl.py / {op}_pypto_impl.py).")
    p.add_argument("--op-name", default=None,
                   help="算子名; 缺省取 op_dir 的 basename.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON 报告输出路径; 缺省打印到 stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    op_dir: Path = args.op_dir.resolve()
    op_name: str = args.op_name or op_dir.name

    if not op_dir.is_dir():
        logger.error("[cheat-detector] op_dir 不存在或非目录: %s", op_dir)
        return 2

    report = detect_cheats(op_dir, op_name)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        logger.info("[cheat-detector] 报告已写入: %s", args.json_out)
    sys.stdout.write(payload + "\n")
    return {"pass": 0, "suspicious": 0, "cheat": 1}.get(report.verdict, 0)


if __name__ == "__main__":
    sys.exit(main())
