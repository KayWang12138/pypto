#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""pypto_op_lint.py — 算子开发流程确定性检查工具

通过 Claude Code hooks / OpenCode plugin 自动触发，
也可手动执行进行调试。
"""

import argparse
import ast
import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
LOGS_EVENTS_FILE = os.path.join(LOGS_DIR, "lint_events.jsonl")
TEST_COMMAND_PATTERN = re.compile(r"python3?\s+.*test_\w+\.py")
PYTHON_BIN = os.path.abspath(sys.executable)
GIT_BIN = shutil.which("git")

SPEC_FILE = "SPEC.md"
API_REPORT_FILE = "API_REPORT.md"
DESIGN_FILE = "DESIGN.md"
OP_WORKSPACE_DIR = "custom"

IMPL_RULE_IDS = [
    "OL01", "OL02", "OL03", "OL04", "OL05", "OL06", "OL07", "OL08",
    "OL16", "OL23", "OL25", "OL26", "OL28", "OL29", "OL37", "OL38",
]
GOLDEN_RULE_IDS = ["OL15"]
TEST_RULE_IDS = ["OL17", "OL18", "OL19", "OL20", "OL21", "OL22", "OL42"]
CONSISTENCY_RULE_IDS = ["OL30", "OL31", "OL32", "OL33", "OL34", "OL39", "OL40", "OL41", "OL43"]
STRICT_ENV = "PYPTO_OP_LINT_STRICT"
MODE_ENV = "PYPTO_OP_LINT_MODE"
HOOK_INPUT_ENV = "PYPTO_OP_LINT_HOOK_INPUT"
POST_EDIT_BLOCK_ENV = "PYPTO_OP_LINT_POST_EDIT_BLOCK"


# ─── 数据结构 ───

@dataclass
class Finding:
    rule_id: str = ""
    severity: str = ""
    dimension: str = ""
    status: str = "SKIP"
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass
class CheckContext:
    op_dir: str
    op_name: str
    stage: int
    rules: list[dict[str, Any]]
    _ast_cache: dict[str, ast.Module] = field(default_factory=dict, repr=False)
    _parse_errors: dict[str, str] = field(default_factory=dict, repr=False)

    def file_path(self, filename: str) -> str:
        return os.path.join(self.op_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(self.file_path(filename))

    def read_file(self, filename: str) -> str:
        path = self.file_path(filename)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_file(self, filename: str) -> Optional[ast.Module]:
        if filename in self._ast_cache:
            return self._ast_cache[filename]
        source = self.read_file(filename)
        if not source:
            return None
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            self._parse_errors[filename] = str(e)
            return None
        self._ast_cache[filename] = tree
        return tree

    def parse_error(self, filename: str) -> str:
        return self._parse_errors.get(filename, "")

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        return {}

    def make_finding(self, rule_id: str, status: str, message: str,
                     file: str = "", line: int = 0) -> Finding:
        rule = self.get_rule(rule_id)
        return Finding(
            rule_id=rule_id,
            severity=rule.get("severity", "S2"),
            dimension=rule.get("dimension", ""),
            status=status,
            message=message,
            file=file,
            line=line,
        )


# ─── 规则注册 ───

CHECKERS: dict[str, Callable[[CheckContext], Finding]] = {}


def register(rule_id: str):
    """装饰器：将检查函数注册到规则 ID"""
    def decorator(fn: Callable[[CheckContext], Finding]):
        CHECKERS[rule_id] = fn
        return fn
    return decorator


# ─── AST 辅助函数 ───

def _get_jit_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """找到所有被 @pypto.frontend.jit 装饰的函数"""
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_str = ast.dump(dec)
            if "pypto" in dec_str and "jit" in dec_str:
                result.append(node)
                break
    return result


def _get_wrapper_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    wrappers: list[ast.FunctionDef] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_wrapper"):
            wrappers.append(node)
    return wrappers


def _resolve_primary_kernel_names(tree: ast.Module) -> set[str]:
    """解析主 kernel 名称（优先取 wrapper 内直接调用的 jit 函数）。"""
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return set()
    jit_names = {f.name for f in jit_funcs}

    resolved: set[str] = set()
    for wrapper in _get_wrapper_functions(tree):
        for node in ast.walk(wrapper):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in jit_names:
                resolved.add(node.func.id)
    if resolved:
        return resolved

    # 无法解析 wrapper 调用关系时，降级为“全部 jit”。
    return jit_names


def _get_primary_jit_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    primary_names = _resolve_primary_kernel_names(tree)
    if not primary_names:
        return []
    return [f for f in _get_jit_functions(tree) if f.name in primary_names]


def _has_loop_structure(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return True
        if isinstance(node, ast.Call):
            call_str = ast.dump(node.func).lower()
            if "loop" in call_str:
                return True
            if any(keyword.arg in ("unroll_list", "submit_before_loop")
                   for keyword in node.keywords if keyword.arg):
                return True
    return False


def _is_pypto_tensor_annotation(annotation: ast.AST) -> bool:
    """检查注解是否为 pypto.Tensor 类型"""
    func: ast.AST
    if isinstance(annotation, ast.Call):
        func = annotation.func
    else:
        func = annotation

    if isinstance(func, ast.Attribute):
        return isinstance(func.value, ast.Name) and func.value.id == "pypto" and func.attr == "Tensor"
    if isinstance(func, ast.Name):
        return func.id == "Tensor"
    return False


def _is_non_tensor_annotation(annotation: ast.AST) -> bool:
    """检查注解是否为显式声明的非 Tensor 参数类型。"""
    return annotation is not None and not _is_pypto_tensor_annotation(annotation)


def _has_test_level_markers(tree: ast.Module, source: str) -> tuple[bool, bool]:
    """识别仓内常见的两级测试命名方式。

    支持：
    - 函数名中的 level0 / level1
    - 功能_P0 / 性能_P0（含 func_p0 / perf_p0）这类 case 命名或元数据
    """
    func_names = [
        node.name.lower() for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    has_level0 = any("level0" in name for name in func_names)
    has_level1 = any("level1" in name for name in func_names)
    if has_level0 and has_level1:
        return True, True

    lower = source.lower()
    p0_level0_patterns = (
        "功能_p0",
        "func_p0",
        "test_p0",
    )
    p0_level1_patterns = (
        "性能_p0",
        "perf_p0",
    )
    has_level0 = has_level0 or any(pattern in lower for pattern in p0_level0_patterns)
    has_level1 = has_level1 or any(pattern in lower for pattern in p0_level1_patterns)
    return has_level0, has_level1


# ─── D1: 框架约束合规 (OL01-OL08) ───

@register("OL01")
def check_ol01(ctx: CheckContext) -> Finding:
    """kernel 函数必须有 @pypto.frontend.jit 装饰器"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL01", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL01", "PASS",
            f"找到 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=impl_file, line=func.lineno)
    return ctx.make_finding("OL01", "FAIL",
        f"[S0 致命] {impl_file} 中未找到任何 @pypto.frontend.jit 装饰的函数。"
        f"这是原则性错误——缺少 jit 装饰器的文件不构成有效的 kernel 实现，"
        f"无法编译、无法在 NPU 上执行。"
        f"禁止在此文件上做局部修补或变通处理，"
        f"必须删除当前 {impl_file} 并基于 SPEC.md / DESIGN.md 从零重新生成。",
        file=impl_file)


@register("OL02")
def check_ol02(ctx: CheckContext) -> Finding:
    """输出写回必须用 [:]/move()/assemble()，禁止 out = expr"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL02", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL02", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        param_names = {arg.arg for arg in func.args.args}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in param_names:
                        return ctx.make_finding("OL02", "FAIL",
                            f"禁止 `{target.id} = expr` 写回，应使用 "
                            f"`{target.id}[:] = ...` 或 `{target.id}.move(...)`",
                            file=impl_file, line=node.lineno)
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id in param_names:
                    return ctx.make_finding("OL02", "FAIL",
                        f"禁止 `{node.target.id} += expr` 写回，应使用 "
                        f"`{node.target.id}[:] = {node.target.id} + ...`",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL02", "PASS", "输出写回方式正确", file=impl_file)


@register("OL03")
def check_ol03(ctx: CheckContext) -> Finding:
    """kernel 函数不能有 return 语句"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL03", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL03", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                return ctx.make_finding("OL03", "FAIL",
                    f"jit 函数 {func.name} 内存在 return 语句",
                    file=impl_file, line=node.lineno)
    return ctx.make_finding("OL03", "PASS", "jit 函数无 return 语句", file=impl_file)


@register("OL04")
def check_ol04(ctx: CheckContext) -> Finding:
    """必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL04", "SKIP", f"{impl_file} 不存在或无法解析")
    for func in _get_jit_functions(tree):
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                call_str = ast.dump(node.func)
                if "set_vec_tile_shapes" in call_str or "set_cube_tile_shapes" in call_str:
                    return ctx.make_finding("OL04", "PASS",
                        "找到 tile shapes 配置调用",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL04", "FAIL",
        "jit 函数体内未找到 set_vec_tile_shapes 或 set_cube_tile_shapes 调用"
        "（注意：必须在 @jit 装饰的函数内部调用）", file=impl_file)


@register("OL05")
def check_ol05(ctx: CheckContext) -> Finding:
    """kernel 张量参数必须有 pypto.Tensor 类型注解"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL05", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL05", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for arg in func.args.args:
            if arg.annotation is None:
                return ctx.make_finding("OL05", "FAIL",
                    f"jit 函数参数 `{arg.arg}` 缺少类型注解",
                    file=impl_file, line=func.lineno)
            # 非张量参数（int/float 等标量）跳过 Tensor 注解检查
            if _is_non_tensor_annotation(arg.annotation):
                continue
            if not _is_pypto_tensor_annotation(arg.annotation):
                return ctx.make_finding("OL05", "FAIL",
                    f"jit 函数张量参数 `{arg.arg}` 注解必须为 pypto.Tensor",
                    file=impl_file, line=func.lineno)
    return ctx.make_finding("OL05", "PASS",
        "jit 函数张量参数均有 pypto.Tensor 类型注解", file=impl_file)


@register("OL06")
def check_ol06(ctx: CheckContext) -> Finding:
    """kernel 内禁用 Python 原生 min()/max()"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL06", "SKIP", f"{impl_file} 不存在或无法解析")
    for func in _get_jit_functions(tree):
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("min", "max")):
                return ctx.make_finding("OL06", "FAIL",
                    f"jit 函数内使用了 Python 原生 {node.func.id}()，"
                    "应使用 pypto 等价函数",
                    file=impl_file, line=node.lineno)
    return ctx.make_finding("OL06", "PASS", "未使用原生 min/max", file=impl_file)


@register("OL07")
def check_ol07(ctx: CheckContext) -> Finding:
    """必须 import pypto"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL07", "SKIP", f"{impl_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pypto" or alias.name.startswith("pypto."):
                    return ctx.make_finding("OL07", "PASS", "找到 import pypto",
                        file=impl_file, line=node.lineno)
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("pypto")):
            return ctx.make_finding("OL07", "PASS", "找到 from pypto import ...",
                file=impl_file, line=node.lineno)
    return ctx.make_finding("OL07", "FAIL",
        f"[S0 致命] {impl_file} 中未 import pypto——"
        "这是 PyPTO 算子实现的基础前提，缺少 import 说明该文件不是合法的 kernel 实现。"
        "禁止任何形式的局部修补或绕过。"
        f"必须删除当前 {impl_file} 并基于 DESIGN.md 重新实现。",
        file=impl_file)


@register("OL08")
def check_ol08(ctx: CheckContext) -> Finding:
    """wrapper 函数必须导出且以 _wrapper 结尾"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL08", "SKIP", f"{impl_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_wrapper"):
            return ctx.make_finding("OL08", "PASS",
                f"找到 wrapper 函数: {node.name}",
                file=impl_file, line=node.lineno)
    return ctx.make_finding("OL08", "FAIL",
        "未找到以 _wrapper 结尾的模块级函数", file=impl_file)


@register("OL23")
def check_ol23(ctx: CheckContext) -> Finding:
    impl_file = f"{ctx.op_name}_impl.py"
    syntax_error = _syntax_error_finding(ctx, "OL23", impl_file)
    if syntax_error:
        return syntax_error
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL23", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_primary_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL23", "SKIP", "无 jit 函数")
    if any(_has_loop_structure(func) for func in jit_funcs):
        return ctx.make_finding("OL23", "PASS",
            "检测到 loop 相关结构", file=impl_file)
    return ctx.make_finding("OL23", "WARN",
        "未检测到 loop 相关结构；若该算子需要分块或迭代，请确认设计已说明无需 loop",
        file=impl_file)


@register("OL25")
def check_ol25(ctx: CheckContext) -> Finding:
    """JIT Tensor 参数注解建议显式声明 shape/dtype（官方示例允许无参数形式）。"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL25", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL25", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            if _is_non_tensor_annotation(ann):
                continue
            if not _is_pypto_tensor_annotation(ann):
                continue
            if not isinstance(ann, ast.Call):
                continue
            n_args = len(ann.args)
            if n_args == 0:
                return ctx.make_finding("OL25", "WARN",
                    f"参数 `{arg.arg}` 使用 pypto.Tensor()（未声明 shape/dtype）。"
                    "建议改为 pypto.Tensor([shape_dims], pypto.DT_xxx) 以提升可维护性",
                    file=impl_file, line=ann.lineno)
    return ctx.make_finding("OL25", "PASS",
        "JIT Tensor 参数注解已包含 shape 与 dtype", file=impl_file)


@register("OL38")
def check_ol38(ctx: CheckContext) -> Finding:
    """JIT Tensor 参数注解建议写全 shape 与 dtype（告警项）。"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL38", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL38", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            if _is_non_tensor_annotation(ann):
                continue
            if not _is_pypto_tensor_annotation(ann):
                continue
            if not isinstance(ann, ast.Call):
                return ctx.make_finding("OL38", "WARN",
                    f"参数 `{arg.arg}` 的 Tensor 注解未显式给出参数，"
                    "建议写成 pypto.Tensor([shape_dims], pypto.DT_xxx)",
                    file=impl_file, line=arg.lineno)
            if len(ann.args) == 1:
                return ctx.make_finding("OL38", "WARN",
                    f"参数 `{arg.arg}` 只声明了 shape，缺少 dtype。"
                    "建议写成 pypto.Tensor([shape_dims], pypto.DT_xxx)",
                    file=impl_file, line=ann.lineno)
    return ctx.make_finding("OL38", "PASS",
        "JIT Tensor 参数注解已包含 shape 与 dtype", file=impl_file)


@register("OL26")
def check_ol26(ctx: CheckContext) -> Finding:
    """JIT 函数中张量参数必须在非张量参数之前"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL26", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL26", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        seen_non_tensor = False
        for arg in func.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            if _is_non_tensor_annotation(ann):
                seen_non_tensor = True
            elif _is_pypto_tensor_annotation(ann):
                if seen_non_tensor:
                    return ctx.make_finding("OL26", "FAIL",
                        f"jit 函数 {func.name} 中张量参数 `{arg.arg}` 出现在非张量参数之后，"
                        "JIT 要求张量参数在前、非张量参数在后",
                        file=impl_file, line=func.lineno)
    return ctx.make_finding("OL26", "PASS",
        "jit 函数参数顺序正确（张量在前、标量在后）", file=impl_file)


FP32_ONLY_OPS = {"sigmoid", "softmax", "sin", "cos"}


@register("OL28")
def check_ol28(ctx: CheckContext) -> Finding:
    """sigmoid/softmax/sin/cos 仅支持 DT_FP32，非 FP32 dtype 时警告"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL28", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL28", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        used_fp32_only = set()
        for node in ast.walk(func):
            if _is_fp32_only_call(node):
                used_fp32_only.add(node.func.attr)
        if not used_fp32_only:
            continue
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann):
                continue
            if len(ann.args) >= 2:
                dtype_str = ast.dump(ann.args[1])
                if "DT_FP32" not in dtype_str:
                    return ctx.make_finding("OL28", "WARN",
                        f"jit 函数使用了仅支持 DT_FP32 的 API "
                        f"({', '.join(sorted(used_fp32_only))})，"
                        f"但参数 `{arg.arg}` 的 dtype 不是 DT_FP32，"
                        "请确认已正确处理 dtype 转换（cast）",
                        file=impl_file, line=func.lineno)
    return ctx.make_finding("OL28", "PASS",
        "FP32-only API 与 dtype 注解一致", file=impl_file)


def _is_fp32_only_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    owner = node.func.value
    if not isinstance(owner, ast.Name):
        return False
    if owner.id != "pypto":
        return False
    return node.func.attr in FP32_ONLY_OPS


@register("OL29")
def check_ol29(ctx: CheckContext) -> Finding:
    """Tensor 注解的 shape 中应声明 pypto.DYNAMIC/pypto.DYN 维度"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL29", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_primary_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL29", "SKIP", "无 jit 函数")
    tensor_count = 0
    has_dynamic = False
    dynamic_aliases = _extract_symbolic_dynamic_aliases(tree)
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann):
                continue
            tensor_count += 1
            if ann.args:
                if _shape_has_dynamic(ann.args[0], dynamic_aliases):
                    has_dynamic = True
    if tensor_count == 0:
        return ctx.make_finding("OL29", "SKIP", "无 Tensor 注解")
    if has_dynamic:
        return ctx.make_finding("OL29", "PASS",
            "Tensor 注解中包含 DYNAMIC 维度声明", file=impl_file)
    return ctx.make_finding("OL29", "WARN",
        "所有 Tensor 注解的 shape 中均未声明 pypto.DYNAMIC/pypto.DYN，"
        "若有输入维度在运行时可变，必须标记为 DYNAMIC 以避免重编译",
        file=impl_file)


# ─── D2: 工件完整性与流程合规 (OL09-OL14, OL24) ───

@register("OL09")
def check_ol09(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL09", "FAIL", f"{SPEC_FILE} 不存在")
    content = ctx.read_file(SPEC_FILE)
    if ctx.op_name not in content:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 中未包含算子名 '{ctx.op_name}'", file=SPEC_FILE)

    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "数学公式"):
        missing.append("数学公式")
    if not _has_heading_like(headings, "输入输出规格"):
        missing.append("输入输出规格")
    if not _has_heading_like(headings, "精度要求"):
        missing.append("精度要求")
    if missing:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 缺少必需内容: {', '.join(missing)}", file=SPEC_FILE)

    # 数学/算法章节：尝试多个可能的章节名
    math_text = _extract_section_text(content, "数学")
    if not math_text:
        math_text = _extract_section_text(content, "算法")
    if not math_text:
        math_text = _extract_section_text(content, "基础信息")
    # 公式建议统一使用 $$...$$ 块公式；为兼容少量纯文本公式，仍保留 '=' 判定。
    if not math_text or not any(token in math_text for token in ("$$", "=", "\\begin{equation}", "round", "clamp")):
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 数学定义章节内容不足（缺少可解析公式特征）", file=SPEC_FILE)

    # 输入输出/数据规格章节
    io_text = _extract_section_text(content, "输入输出规格")
    if not io_text:
        io_text = _extract_section_text(content, "数据规格")
    if not io_text or "dtype" not in io_text.lower() or "shape" not in io_text.lower():
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 输入输出规格章节内容不足（需包含 shape/dtype）", file=SPEC_FILE)

    precision_text = _extract_section_text(content, "精度")
    if not precision_text or ("atol" not in precision_text.lower() and "rtol" not in precision_text.lower() and "mare" not in precision_text.lower()):
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 精度要求章节内容不足（需包含 atol/rtol 或指标阈值）", file=SPEC_FILE)

    # Front matter schema 校验：在 SPEC 生成时即拦截格式问题（如 p0_shapes 非 list）
    spec_meta, _ = _parse_front_matter(content)
    if not spec_meta:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 缺少 front matter（必须以 --- 开头，包含 schema_version/op_name/supported_dtypes/p0_shapes/tolerance）",
            file=SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    return ctx.make_finding("OL09", "PASS",
        f"{SPEC_FILE} 含算子名、公式、输入输出规格与精度要求，front matter schema 合法", file=SPEC_FILE)


@register("OL10")
def check_ol10(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(API_REPORT_FILE):
        return ctx.make_finding("OL10", "FAIL", f"{API_REPORT_FILE} 不存在")
    content = ctx.read_file(API_REPORT_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "API 映射"):
        missing.append("API 映射")
    if not _has_heading_like(headings, "约束"):
        missing.append("约束")
    if not _has_heading_like(headings, "Tiling"):
        missing.append("Tiling")
    if missing:
        return ctx.make_finding("OL10", "FAIL",
            f"{API_REPORT_FILE} 缺少必需内容: {', '.join(missing)}",
            file=API_REPORT_FILE)
    return ctx.make_finding("OL10", "PASS",
        f"{API_REPORT_FILE} 含 API 映射、约束与 Tiling 说明", file=API_REPORT_FILE)


@register("OL11")
def check_ol11(ctx: CheckContext) -> Finding:
    """进入 Stage 4 需 {op}_golden.py 可导入"""
    golden_file = f"{ctx.op_name}_golden.py"
    if not ctx.file_exists(golden_file):
        return ctx.make_finding("OL11", "FAIL", f"{golden_file} 不存在")
    probe_code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {json.dumps(ctx.op_dir)})\n"
        f"importlib.import_module({json.dumps(f'{ctx.op_name}_golden')})\n"
    )
    try:
        result = subprocess.run(
            [PYTHON_BIN, "-c", probe_code],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入超时（>10s）", file=golden_file)
    except OSError as e:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入探测失败: {e}", file=golden_file)
    if result.returncode != 0:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入失败: {result.stderr[:200]}", file=golden_file)
    return ctx.make_finding("OL11", "PASS", f"{golden_file} 可导入", file=golden_file)


@register("OL12")
def check_ol12(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL12", "FAIL", f"{DESIGN_FILE} 不存在")
    content = ctx.read_file(DESIGN_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "计算图"):
        missing.append("计算图")
    if not _has_heading_like(headings, "Tiling"):
        missing.append("Tiling")
    if not _has_heading_like(headings, "验证方案"):
        missing.append("验证方案")
    if missing:
        return ctx.make_finding("OL12", "FAIL",
            f"{DESIGN_FILE} 缺少必需内容: {', '.join(missing)}", file=DESIGN_FILE)
    return ctx.make_finding("OL12", "PASS",
        f"{DESIGN_FILE} 含计算图、Tiling 与验证方案", file=DESIGN_FILE)


@register("OL13")
def check_ol13(ctx: CheckContext) -> Finding:
    """生成文件三件套完整"""
    files = [
        f"{ctx.op_name}_impl.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    missing = [f for f in files if not ctx.file_exists(f)]
    if missing:
        return ctx.make_finding("OL13", "FAIL",
            f"缺少文件: {', '.join(missing)}")
    return ctx.make_finding("OL13", "PASS", "三件套文件完整")


@register("OL14")
def check_ol14(ctx: CheckContext) -> Finding:
    """进入 Stage 7 需精度通过"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL14", "FAIL", "状态文件不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("stage_status", {})
        if status.get("5") == "completed" or status.get("6") == "completed":
            return ctx.make_finding("OL14", "PASS", "精度已通过")
    except ValueError:
        pass
    return ctx.make_finding("OL14", "FAIL", "精度未通过（Stage 5/6 未 completed）")


@register("OL24")
def check_ol24(ctx: CheckContext) -> Finding:
    """.orchestrator_state.json 结构合法"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL24", "FAIL", ".orchestrator_state.json 不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ctx.make_finding("OL24", "FAIL", f"JSON 解析失败: {e}")
    required = ["operator_name", "current_stage", "stage_status"]
    missing = [k for k in required if k not in data]
    if missing:
        return ctx.make_finding("OL24", "FAIL",
            f"缺少必需字段: {', '.join(missing)}")
    return ctx.make_finding("OL24", "PASS", "状态文件结构合法")


# ─── D3: 三文件分离 (OL15-OL18) ───

@register("OL15")
def check_ol15(ctx: CheckContext) -> Finding:
    """golden 文件禁止 import pypto"""
    golden_file = f"{ctx.op_name}_golden.py"
    tree = ctx.parse_file(golden_file)
    if tree is None:
        return ctx.make_finding("OL15", "SKIP", f"{golden_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pypto" or alias.name.startswith("pypto."):
                    return ctx.make_finding("OL15", "FAIL",
                        "golden 文件禁止 import pypto",
                        file=golden_file, line=node.lineno)
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("pypto")):
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 from pypto import ...",
                file=golden_file, line=node.lineno)
    return ctx.make_finding("OL15", "PASS",
        "golden 文件未导入 pypto", file=golden_file)


@register("OL16")
def check_ol16(ctx: CheckContext) -> Finding:
    """impl 文件不应导入 golden 模块"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL16", "SKIP", f"{impl_file} 不存在或无法解析")
    golden_module = f"{ctx.op_name}_golden"
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == golden_module:
            return ctx.make_finding("OL16", "FAIL",
                f"impl 文件不应导入 {golden_module}",
                file=impl_file, line=node.lineno)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == golden_module:
                    return ctx.make_finding("OL16", "FAIL",
                        f"impl 文件不应导入 {golden_module}",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL16", "PASS",
        "impl 文件未导入 golden", file=impl_file)


@register("OL17")
def check_ol17(ctx: CheckContext) -> Finding:
    """test 文件不应包含 kernel 实现代码"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL17", "SKIP", f"{test_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL17", "FAIL",
            f"test 文件包含 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=test_file, line=func.lineno)
    return ctx.make_finding("OL17", "PASS",
        "test 文件未包含 kernel 实现", file=test_file)


@register("OL18")
def check_ol18(ctx: CheckContext) -> Finding:
    """test 文件必须从 impl 和 golden 分别导入"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL18", "SKIP", f"{test_file} 不存在或无法解析")
    impl_module = f"{ctx.op_name}_impl"
    golden_module = f"{ctx.op_name}_golden"
    has_impl = False
    has_golden = False

    def _matches_module(module_name: str | None, expected: str) -> bool:
        if module_name is None:
            return False
        return module_name == expected or module_name.endswith(f".{expected}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _matches_module(node.module, impl_module):
                has_impl = True
            if _matches_module(node.module, golden_module):
                has_golden = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_module(alias.name, impl_module):
                    has_impl = True
                if _matches_module(alias.name, golden_module):
                    has_golden = True
    missing = []
    if not has_impl:
        missing.append(impl_module)
    if not has_golden:
        missing.append(golden_module)
    if missing:
        return ctx.make_finding("OL18", "FAIL",
            f"test 文件缺少导入: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL18", "PASS",
        "test 文件正确导入了 impl 和 golden", file=test_file)


# ─── D4: 测试规范 (OL19-OL22) ───

@register("OL19")
def check_ol19(ctx: CheckContext) -> Finding:
    """test 必须使用 assert_allclose"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 无法解析")
    has_allclose = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = ast.dump(node.func)
            if "assert_allclose" in call_str:
                has_allclose = True
                break
    if not has_allclose:
        return ctx.make_finding("OL19", "FAIL",
            "未找到 assert_allclose 调用，禁止手写 assert max_diff",
            file=test_file)
    return ctx.make_finding("OL19", "PASS",
        "使用了 assert_allclose", file=test_file)


@register("OL20")
def check_ol20(ctx: CheckContext) -> Finding:
    """test 必须处理 TILE_FWK_DEVICE_ID 并调用 set_device"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL20", "SKIP", f"{test_file} 不存在")
    has_device_id = "TILE_FWK_DEVICE_ID" in source
    has_set_device = "set_device" in source
    if has_device_id and has_set_device:
        return ctx.make_finding("OL20", "PASS",
            "找到 TILE_FWK_DEVICE_ID 处理和 set_device 调用", file=test_file)
    missing = []
    if not has_device_id:
        missing.append("TILE_FWK_DEVICE_ID 环境变量处理")
    if not has_set_device:
        missing.append("set_device 调用")
    return ctx.make_finding("OL20", "FAIL",
        f"缺少: {', '.join(missing)}", file=test_file)


@register("OL21")
def check_ol21(ctx: CheckContext) -> Finding:
    """test 必须有 Level 0 和 Level 1 两级测试函数"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在或无法解析")
    has_level0, has_level1 = _has_test_level_markers(tree, source)
    missing = []
    if not has_level0:
        missing.append("level0")
    if not has_level1:
        missing.append("level1")
    if missing:
        return ctx.make_finding("OL21", "FAIL",
            f"缺少测试级别: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL21", "PASS",
        "包含 Level 0 和 Level 1 测试", file=test_file)


@register("OL22")
def check_ol22(ctx: CheckContext) -> Finding:
    """test 应设置 torch.manual_seed 保证可复现"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL22", "SKIP", f"{test_file} 不存在")
    if "manual_seed" in source:
        return ctx.make_finding("OL22", "PASS",
            "找到 manual_seed 设置", file=test_file)
    return ctx.make_finding("OL22", "FAIL",
        "未设置 torch.manual_seed，测试结果可能不可复现", file=test_file)


@register("OL42")
def check_ol42(ctx: CheckContext) -> Finding:
    """NPU 环境下 test 不得硬编码 sim 模式"""
    test_file = f"test_{ctx.op_name}.py"
    content = ctx.read_file(test_file)
    if not content:
        return ctx.make_finding("OL42", "SKIP", f"{test_file} 不存在")

    # 检测是否有 NPU 环境
    if not _check_npu_available():
        return ctx.make_finding("OL42", "SKIP",
            "未检测到 NPU 环境（npu-smi 不可用），跳过 sim 模式检查")

    # 在有 NPU 的环境下，检查是否硬编码了 sim 模式
    problems = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        # 跳过注释行
        if stripped.startswith("#"):
            continue
        # 检查 default="sim" 或 default='sim' (argparse default)
        if re.search(r"""default\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: argparse default 设置为 sim")
        # 检查 run_mode="sim" 或 run_mode='sim' 的硬编码赋值
        elif re.search(r"""run_mode\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: run_mode 硬编码为 sim")

    if problems:
        return ctx.make_finding("OL42", "FAIL",
            f"NPU 环境下不应使用 sim 模式: {'; '.join(problems)}",
            file=f"test_{ctx.op_name}.py")
    return ctx.make_finding("OL42", "PASS",
        "未发现 sim 模式硬编码", file=f"test_{ctx.op_name}.py")


def _check_npu_available() -> bool:
    """通过 npu-smi info 检测 NPU 环境是否可用"""
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─── D5: 跨文件一致性检查 (Consistency) ───

DTYPE_ALIASES: dict[str, str] = {
    "float32": "fp32", "fp32": "fp32", "dt_fp32": "fp32", "torch.float32": "fp32",
    "float16": "fp16", "fp16": "fp16", "dt_fp16": "fp16", "torch.float16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16", "dt_bf16": "bf16", "torch.bfloat16": "bf16",
}


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in ("", "null", "None"):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            items = [x.strip() for x in inner.split(",")]
            parsed: list[Any] = []
            for item in items:
                if item.startswith("[") and item.endswith("]"):
                    parsed.append(_parse_scalar(item))
                elif item.startswith(("'", '"')) and item.endswith(("'", '"')):
                    parsed.append(item[1:-1])
                elif re.fullmatch(r"[+-]?\d+", item):
                    parsed.append(int(item))
                elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", item):
                    parsed.append(float(item))
                else:
                    parsed.append(item)
            return parsed
    if value.startswith("{") and value.endswith("}"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        # 回退：解析 YAML 风格的无引号 key 字典，如 {key1: val1, key2: val2}
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for pair in inner.split(","):
            pair = pair.strip()
            if ":" not in pair:
                return value  # 无法解析，返回原始字符串
            k, v = pair.split(":", 1)
            k = k.strip().strip("'\"")
            result[k] = _parse_scalar(v)
        return result
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        return float(value)
    return value


def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """解析 markdown front matter。

    约定：仅支持文件开头的 `---` 包裹块；值支持标量、list、dict（可使用
    Python/JSON 字面量形式），满足 lint 结构化字段读取需求。
    """
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end < 0:
        return {}, content

    header = content[4:end]
    body = content[end + 5:]
    meta: dict[str, Any] = {}

    for raw in header.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _parse_scalar(value)
    return meta, body


def _load_doc_meta(ctx: CheckContext, filename: str) -> dict[str, Any]:
    content = ctx.read_file(filename)
    if not content:
        return {}
    meta, _ = _parse_front_matter(content)
    return meta


def _load_tolerance_schema() -> list[dict[str, Any]]:
    """从 rules.json 加载 tolerance_schema.oneOf 定义。"""
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tolerance_schema", {}).get("oneOf", [])
    except (OSError, json.JSONDecodeError):
        return []


def _validate_tolerance(tol: dict[str, Any]) -> list[str]:
    """基于 rules.json 中 tolerance_schema.oneOf 校验 tolerance dict。

    只要匹配任一模式即通过。
    """
    schemas = _load_tolerance_schema()
    if not schemas:
        # 无 schema 定义时回退到基础校验：只检查是否为非空 dict
        return []

    for schema in schemas:
        required = schema.get("required", [])
        if all(k in tol for k in required):
            return []  # 匹配成功

    # 所有模式都不匹配，列出可接受的模式
    modes = [f"{s.get('mode', '?')}({', '.join(s.get('required', []))})" for s in schemas]
    return [f"tolerance must match one of: {' | '.join(modes)}"]


def _validate_doc_schema(doc_type: str, meta: dict[str, Any]) -> list[str]:
    required: dict[str, list[str]] = {
        "SPEC": ["schema_version", "op_name", "supported_dtypes", "p0_shapes", "tolerance"],
        "DESIGN": ["schema_version", "op_name", "dynamic_axes"],
        "API_REPORT": ["schema_version", "op_name"],
    }
    errors: list[str] = []
    for key in required.get(doc_type, []):
        if key not in meta or meta[key] in (None, "", []):
            errors.append(f"missing field: {key}")
    if doc_type == "SPEC":
        if "supported_dtypes" in meta and not isinstance(meta.get("supported_dtypes"), list):
            errors.append("invalid type: supported_dtypes must be list")
        if "p0_shapes" in meta and not isinstance(meta.get("p0_shapes"), list):
            errors.append("invalid type: p0_shapes must be list")
        if "tolerance" in meta and not isinstance(meta.get("tolerance"), dict):
            errors.append("invalid type: tolerance must be dict")
        if isinstance(meta.get("tolerance"), dict):
            errors.extend(_validate_tolerance(meta["tolerance"]))
    if doc_type == "DESIGN":
        if "dynamic_axes" in meta and not isinstance(meta.get("dynamic_axes"), list):
            errors.append("invalid type: dynamic_axes must be list")
    return errors


def _extract_spec_dtypes_from_meta(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("supported_dtypes", [])
    if not isinstance(raw, list):
        return set()
    result: set[str] = set()
    for item in raw:
        key = str(item).strip().lower()
        canonical = DTYPE_ALIASES.get(key)
        if canonical:
            result.add(canonical)
    return result


def _extract_test_dtypes(source: str) -> set[str]:
    """从 test 文件源码中提取使用的 dtype"""
    dtypes: set[str] = set()
    lower = source.lower()
    for alias, canonical in DTYPE_ALIASES.items():
        if alias in lower:
            dtypes.add(canonical)
    return dtypes


def _extract_symbolic_dynamic_aliases(tree: ast.Module) -> set[str]:
    """提取模块级别指向 pypto.DYNAMIC/pypto.DYN 的符号名。"""
    aliases: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        value = node.value
        if isinstance(value, ast.Attribute):
            if isinstance(value.value, ast.Name) and value.value.id == "pypto" and value.attr in ("DYNAMIC", "DYN"):
                aliases.add(target)
        elif isinstance(value, ast.Name) and value.id in ("DYNAMIC", "DYN"):
            aliases.add(target)
    return aliases


def _shape_has_dynamic(shape_node: ast.AST, dynamic_aliases: set[str]) -> bool:
    """递归判断 shape 注解中是否包含动态维度声明。"""
    if isinstance(shape_node, ast.Attribute):
        if isinstance(shape_node.value, ast.Name) and shape_node.value.id == "pypto" and shape_node.attr in ("DYNAMIC", "DYN"):
            return True
    if isinstance(shape_node, ast.Name):
        return shape_node.id in dynamic_aliases or shape_node.id in ("DYNAMIC", "DYN")
    if isinstance(shape_node, (ast.List, ast.Tuple)):
        return any(_shape_has_dynamic(elem, dynamic_aliases) for elem in shape_node.elts)
    for child in ast.iter_child_nodes(shape_node):
        if _shape_has_dynamic(child, dynamic_aliases):
            return True
    return False


def _extract_tolerance(content: str, key: str) -> list[float]:
    """从文本中提取所有 atol 或 rtol 数值"""
    values: list[float] = []
    for pat in [
        rf"{key}\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
        rf"\*\*{key}\*\*\s*[:=]?\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
    ]:
        for m in re.finditer(pat, content, re.IGNORECASE):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                continue
    return values


def _get_func_param_count(tree: ast.Module, func_name: str) -> Optional[int]:
    """获取指定函数的必需参数个数"""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            total = len(node.args.args)
            defaults = len(node.args.defaults)
            required = total - defaults
            if required > 0 and node.args.args[0].arg in ("self", "cls"):
                required -= 1
            return required
    return None


def _extract_shapes_from_text(content: str) -> set[tuple[int, ...]]:
    """从文本中提取 shape 元组，如 [1, 2048, 4096] 或 (1, 2048, 4096)"""
    shapes: set[tuple[int, ...]] = set()
    for m in re.finditer(r'[\[\(](\d+(?:\s*,\s*\d+)+)[\]\)]', content):
        try:
            dims = tuple(int(x.strip()) for x in m.group(1).split(","))
            if len(dims) >= 2:
                shapes.add(dims)
        except ValueError:
            continue
    return shapes


def _extract_shapes_from_test_ast(tree: ast.Module) -> set[tuple[int, ...]]:
    """从测试 AST 中提取 shape 覆盖。

    支持模式：
    1) _run_and_check(name, n, m, ...)
    2) torch.randn(n, m, ...) / torch.zeros((n, m), ...) 等
    """
    shapes: set[tuple[int, ...]] = set()

    def _const_int(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "_run_and_check":
            if len(node.args) >= 3:
                n_val = _const_int(node.args[1])
                m_val = _const_int(node.args[2])
                if n_val is not None and m_val is not None:
                    shapes.add((n_val, m_val))

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id != "torch":
                continue
            if node.func.attr not in ("randn", "zeros", "ones", "empty", "full"):
                continue

            if node.func.attr == "randn" and len(node.args) >= 2:
                n_val = _const_int(node.args[0])
                m_val = _const_int(node.args[1])
                if n_val is not None and m_val is not None:
                    shapes.add((n_val, m_val))

            if node.args and isinstance(node.args[0], (ast.Tuple, ast.List)) and len(node.args[0].elts) >= 2:
                n_val = _const_int(node.args[0].elts[0])
                m_val = _const_int(node.args[0].elts[1])
                if n_val is not None and m_val is not None:
                    shapes.add((n_val, m_val))

    return shapes


@register("OL30")
def check_ol30(ctx: CheckContext) -> Finding:
    """SPEC.md front matter 的 supported_dtypes 必须在测试文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL30", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL30", "SKIP", f"{test_file} 不存在")

    spec_dtypes = _extract_spec_dtypes_from_meta(spec_meta)
    if not spec_dtypes:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter 中未声明 supported_dtypes",
            file=SPEC_FILE)

    test_dtypes = _extract_test_dtypes(test_source)
    missing = spec_dtypes - test_dtypes
    if missing:
        canonical_names = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        missing_names = sorted(canonical_names.get(d, d) for d in missing)
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} 声明支持的 dtype ({', '.join(missing_names)}) "
            "在测试文件中未覆盖；请补充对应 dtype 的测试用例",
            file=test_file)
    return ctx.make_finding("OL30", "PASS", "spec dtype 覆盖与 test 一致")


@register("OL31")
def check_ol31(ctx: CheckContext) -> Finding:
    """DESIGN.md front matter dynamic_axes 与 impl 动态注解一致。"""
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL31", "SKIP", f"{DESIGN_FILE} 不存在")
    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    schema_errors = _validate_doc_schema("DESIGN", design_meta)
    if schema_errors:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=DESIGN_FILE)

    impl_file = f"{ctx.op_name}_impl.py"
    if not ctx.read_file(impl_file):
        return ctx.make_finding("OL31", "SKIP", f"{impl_file} 不存在")
    syntax_error = _syntax_error_finding(ctx, "OL31", impl_file)
    if syntax_error:
        return syntax_error
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL31", "SKIP", f"{impl_file} 无法解析")
    jit_funcs = _get_primary_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL31", "SKIP", "无 jit 函数")

    dynamic_axes = design_meta.get("dynamic_axes", [])
    if not isinstance(dynamic_axes, list) or not dynamic_axes:
        return ctx.make_finding("OL31", "PASS",
            f"{DESIGN_FILE} front matter 未声明动态轴，无需检查")

    dynamic_aliases = _extract_symbolic_dynamic_aliases(tree)
    has_dynamic_in_impl = False
    has_unparameterized_tensor = False
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann):
                continue
            # pypto.Tensor() 无参数 => 所有维度隐式动态，兼容任何 dynamic_axes 声明
            if len(ann.args) == 0:
                has_unparameterized_tensor = True
                break
            if _shape_has_dynamic(ann.args[0], dynamic_aliases):
                has_dynamic_in_impl = True
                break
        if has_dynamic_in_impl or has_unparameterized_tensor:
            break

    if has_unparameterized_tensor:
        return ctx.make_finding("OL31", "PASS",
            "impl 使用 pypto.Tensor() 无参数注解，所有维度隐式动态，"
            "与 design 动态轴声明兼容", file=impl_file)
    if not has_dynamic_in_impl:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter 声明了动态轴，但 impl 的 Tensor 注解中未使用 "
            "pypto.DYNAMIC/pypto.DYN，动态轴必须在类型注解中显式标记",
            file=impl_file)
    return ctx.make_finding("OL31", "PASS",
        "design 动态轴声明与 impl 注解一致", file=impl_file)


@register("OL43")
def check_ol43(ctx: CheckContext) -> Finding:
    """DESIGN 声明动态轴时 impl 必须包含 pypto.loop 调用"""
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL43", "SKIP", f"{DESIGN_FILE} 不存在")

    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    dynamic_axes = design_meta.get("dynamic_axes") or design_meta.get("dynamic_axis")

    # 若 front matter 无声明，在正文中搜索动态轴相关关键词
    if not dynamic_axes:
        content = ctx.read_file(DESIGN_FILE)
        has_dynamic_keyword = bool(
            re.search(r'(?:pypto\.DYNAMIC|pypto\.DYN|动态轴|dynamic.{0,10}axis)', content, re.IGNORECASE)
        )
        if not has_dynamic_keyword:
            return ctx.make_finding("OL43", "SKIP",
                "DESIGN.md 未声明动态轴，无需检查 pypto.loop")

    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL43", "SKIP", f"{impl_file} 不存在或无法解析")

    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL43", "SKIP", "未找到 JIT 函数")

    # 在 impl 中搜索 pypto.loop 调用
    impl_source = ctx.read_file(impl_file) or ""
    has_pypto_loop = bool(re.search(r'pypto\.loop\s*\(', impl_source))
    if not has_pypto_loop:
        has_pypto_loop = _has_loop_structure(jit_funcs[0])

    if has_pypto_loop:
        return ctx.make_finding("OL43", "PASS",
            "DESIGN 声明动态轴，impl 中存在 loop 结构", file=f"{ctx.op_name}_impl.py")

    return ctx.make_finding("OL43", "FAIL",
        f"DESIGN.md 声明了动态轴，但 {ctx.op_name}_impl.py 中未找到 "
        f"pypto.loop / pypto.lang.loop 调用。建议补充 pypto.loop 以覆盖动态轴，"
        f"可对现有 impl 做局部修补。",
        file=f"{ctx.op_name}_impl.py")


@register("OL32")
def check_ol32(ctx: CheckContext) -> Finding:
    """SPEC.md front matter tolerance 须与 test 文件一致（支持 tolerance_schema 多模式）。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL32", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL32", "SKIP", f"{test_file} 不存在")

    tolerance_meta = spec_meta.get("tolerance", {})
    if not isinstance(tolerance_meta, dict):
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter 中 tolerance 必须是 dict",
            file=SPEC_FILE)

    # 根据 tolerance_schema.oneOf 确定当前使用的模式及其 required 字段
    schemas = _load_tolerance_schema()
    matched_keys: list[str] = []
    if schemas:
        for schema in schemas:
            required = schema.get("required", [])
            if all(k in tolerance_meta for k in required):
                matched_keys = required
                break
        if not matched_keys:
            modes = [f"{s.get('mode', '?')}({', '.join(s.get('required', []))})" for s in schemas]
            return ctx.make_finding("OL32", "FAIL",
                f"{SPEC_FILE} tolerance 不匹配任何已知模式: {' | '.join(modes)}",
                file=SPEC_FILE)
    else:
        # 无 schema 定义时回退到标准模式
        matched_keys = ["atol", "rtol"]

    mismatches = []
    for key in matched_keys:
        if key not in tolerance_meta:
            continue
        try:
            spec_vals = [float(tolerance_meta[key])]
        except (ValueError, TypeError):
            continue
        test_vals = _extract_tolerance(test_source, key)
        if not spec_vals or not test_vals:
            continue
        spec_strictest = min(spec_vals)
        test_loosest = max(test_vals)
        if test_loosest > spec_strictest * 10:
            mismatches.append(
                f"{key}: spec 最严={spec_strictest}, test 最松={test_loosest}")
    if mismatches:
        return ctx.make_finding("OL32", "WARN",
            f"{SPEC_FILE} 与 test 的精度容差差距较大: {'; '.join(mismatches)}",
            file=test_file)
    return ctx.make_finding("OL32", "PASS", "spec 精度容差与 test 一致")


@register("OL33")
def check_ol33(ctx: CheckContext) -> Finding:
    """golden 函数签名须与 impl wrapper 函数兼容"""
    golden_file = f"{ctx.op_name}_golden.py"
    impl_file = f"{ctx.op_name}_impl.py"
    syntax_error = _syntax_error_finding(ctx, "OL33", impl_file)
    if syntax_error:
        return syntax_error
    golden_tree = ctx.parse_file(golden_file)
    impl_tree = ctx.parse_file(impl_file)
    if golden_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{golden_file} 不存在或无法解析")
    if impl_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{impl_file} 不存在或无法解析")
    golden_func = f"{ctx.op_name}_golden"
    wrapper_func = f"{ctx.op_name}_wrapper"
    golden_count = _get_func_param_count(golden_tree, golden_func)
    wrapper_count = _get_func_param_count(impl_tree, wrapper_func)
    if golden_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {golden_func} 函数")
    if wrapper_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {wrapper_func} 函数")
    if golden_count != wrapper_count:
        return ctx.make_finding("OL33", "WARN",
            f"{golden_func} 需要 {golden_count} 个必需参数，"
            f"但 {wrapper_func} 需要 {wrapper_count} 个必需参数，接口可能不兼容",
            file=impl_file)
    return ctx.make_finding("OL33", "PASS",
        f"golden ({golden_count} 参数) 与 wrapper ({wrapper_count} 参数) 签名兼容")


@register("OL34")
def check_ol34(ctx: CheckContext) -> Finding:
    """SPEC.md front matter p0_shapes 应在 test 文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL34", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在")
    test_tree = ctx.parse_file(test_file)
    if test_tree is None:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在或无法解析")
    spec_p0_shapes: set[tuple[int, ...]] = set()
    raw_shapes = spec_meta.get("p0_shapes", [])
    if not isinstance(raw_shapes, list):
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中 p0_shapes 必须是 list",
            file=SPEC_FILE)
    for item in raw_shapes:
        if not isinstance(item, (list, tuple)):
            continue
        try:
            dims = tuple(int(x) for x in item)
        except (ValueError, TypeError):
            continue
        if len(dims) >= 2:
            spec_p0_shapes.add(dims)

    if not spec_p0_shapes:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中未找到有效 p0_shapes",
            file=SPEC_FILE)
    test_shapes = _extract_shapes_from_test_ast(test_tree) | _extract_shapes_from_text(test_source)
    missing = spec_p0_shapes - test_shapes
    if missing:
        missing_str = ", ".join(str(list(s)) for s in sorted(missing))
        return ctx.make_finding("OL34", "WARN",
            f"{SPEC_FILE} 中 P0 配置的 shape {missing_str} 在测试文件中未覆盖",
            file=test_file)
    return ctx.make_finding("OL34", "PASS",
        "spec P0 配置 shape 在 test 中均有覆盖")


@register("OL39")
def check_ol39(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档必须包含 front matter。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL39", "SKIP", "strict 模式关闭")
    for filename in (SPEC_FILE, DESIGN_FILE, API_REPORT_FILE):
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL39", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        if not meta:
            return ctx.make_finding("OL39", "FAIL",
                f"{filename} 缺少 front matter（必须以 --- 开头）",
                file=filename)
    return ctx.make_finding("OL39", "PASS", "front matter 完整")


@register("OL40")
def check_ol40(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档 front matter 必填字段必须完整。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL40", "SKIP", "strict 模式关闭")
    docs = ((SPEC_FILE, "SPEC"), (DESIGN_FILE, "DESIGN"), (API_REPORT_FILE, "API_REPORT"))
    for filename, doc_type in docs:
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL40", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        errors = _validate_doc_schema(doc_type, meta)
        if errors:
            return ctx.make_finding("OL40", "FAIL",
                f"{filename} front matter schema 非法: {'; '.join(errors)}",
                file=filename)
    return ctx.make_finding("OL40", "PASS", "front matter schema 完整")


@register("OL41")
def check_ol41(ctx: CheckContext) -> Finding:
    """禁止将 lint/门禁输出文本污染到代码工件。"""
    suspicious_tokens = (
        "[pypto-op-lint]",
        "交付门禁阻断",
        "以下规则违规",
        "fix_hints:",
        "blocking_rules:",
        "docs_ref:",
    )
    targets = [
        f"{ctx.op_name}_impl.py",
        f"{ctx.op_name}_golden.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    for filename in targets:
        source = ctx.read_file(filename)
        if not source:
            continue
        lower = source.lower()
        for token in suspicious_tokens:
            if token.lower() in lower:
                return ctx.make_finding(
                    "OL41",
                    "FAIL",
                    f"{filename} 检测到 lint 输出污染片段: {token}",
                    file=filename,
                )
    return ctx.make_finding("OL41", "PASS", "未检测到 lint 输出污染")


# ─── 辅助函数 ───

def _load_rules() -> list[dict[str, Any]]:
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])



def _extract_markdown_headings(content: str) -> set[str]:
    _, body = _parse_front_matter(content)
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return {h.strip().lower() for h in headings}


def _has_heading_like(headings: set[str], *keywords: str) -> bool:
    keys = [k.strip().lower() for k in keywords if k.strip()]
    for h in headings:
        if any(k in h for k in keys):
            return True
    return False


def _extract_section_text(content: str, heading_keyword: str) -> str:
    _, body = _parse_front_matter(content)
    lines = body.splitlines()
    key = heading_keyword.strip().lower()
    start = -1
    start_level = 0
    for idx, line in enumerate(lines):
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip().lower()
        if key in title:
            start = idx + 1
            start_level = level
            break
    if start < 0:
        return ""

    buff: list[str] = []
    for line in lines[start:]:
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if m and len(m.group(1)) <= start_level:
            break
        buff.append(line)
    return "\n".join(buff).strip()


def _syntax_error_finding(ctx: CheckContext, rule_id: str, filename: str) -> Optional[Finding]:
    tree = ctx.parse_file(filename)
    if tree is not None:
        return None
    error = ctx.parse_error(filename)
    if not error:
        return None
    return ctx.make_finding(
        rule_id,
        "FAIL",
        f"{filename} 存在语法错误，无法解析: {error}",
        file=filename,
    )


def _infer_op_dir(file_path: str) -> Optional[str]:
    if not file_path:
        return None
    op_dir = os.path.dirname(os.path.abspath(file_path))
    if os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        return op_dir
    basename = os.path.basename(file_path)
    inferred_op_name = _infer_op_name_from_filename(basename)
    if inferred_op_name and _looks_like_stateless_op_dir(op_dir, inferred_op_name):
        return op_dir
    return None


def _infer_op_name_from_filename(filename: str) -> str:
    if filename.startswith("test_") and filename.endswith(".py"):
        return filename[len("test_"):-len(".py")]
    if filename.endswith("_impl.py"):
        return filename[:-len("_impl.py")]
    if filename.endswith("_golden.py"):
        return filename[:-len("_golden.py")]
    return ""


def _looks_like_stateless_op_dir(op_dir: str, op_name: str) -> bool:
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return False
    expected = {
        f"{op_name}_impl.py",
        f"{op_name}_golden.py",
        f"test_{op_name}.py",
        SPEC_FILE,
        API_REPORT_FILE,
        DESIGN_FILE,
        "README.md",
    }
    return len(files & expected) >= 2


def _infer_stage_from_filename(filename: str) -> int:
    if filename == SPEC_FILE:
        return 1
    if filename == API_REPORT_FILE:
        return 2
    if filename == DESIGN_FILE:
        return 4
    if filename.endswith("_golden.py"):
        return 3
    if filename.endswith("_impl.py") or (filename.startswith("test_") and filename.endswith(".py")):
        return 5
    return 0


def _infer_stage_from_artifacts(op_dir: str) -> int:
    op_name = os.path.basename(op_dir)
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return 0

    if len({f"{op_name}_impl.py", f"test_{op_name}.py", "README.md"} & files) >= 2:
        return 5
    if DESIGN_FILE in files or f"{op_name}_golden.py" in files:
        return 4
    if API_REPORT_FILE in files:
        return 3
    if SPEC_FILE in files:
        return 2
    return 0


def _get_current_stage(op_dir: str) -> int:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return 0
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("current_stage", 0))
    except ValueError:
        return 0


def _get_op_name(op_dir: str) -> str:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                name = json.load(f).get("operator_name", "")
                if name:
                    return name
        except ValueError:
            pass
    return os.path.basename(op_dir)


def _build_context(op_dir: str, stage: Optional[int] = None) -> CheckContext:
    rules = _load_rules()
    op_dir = os.path.abspath(op_dir)
    if stage is None:
        stage = _get_current_stage(op_dir)
    op_name = _get_op_name(op_dir)
    return CheckContext(op_dir=op_dir, op_name=op_name, stage=stage, rules=rules)


def _run_checks(ctx: CheckContext, rule_ids: list[str]) -> list[Finding]:
    """执行指定规则的检查"""
    findings = []
    mode = os.environ.get(MODE_ENV, "cli")
    strict = os.environ.get(STRICT_ENV, "1") == "1"
    for rid in rule_ids:
        start = time.perf_counter()
        rule = ctx.get_rule(rid)
        if ctx.stage not in rule.get("stages", []):
            finding = ctx.make_finding(rid, "SKIP", "当前阶段不适用")
            findings.append(finding)
            _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
            continue
        checker = CHECKERS.get(rid)
        if not checker:
            finding = ctx.make_finding(rid, "SKIP", "检查函数未注册")
            findings.append(finding)
            _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
            continue
        finding = checker(ctx)
        findings.append(finding)
        _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
    return findings


def _output_hook_json(event: str, **kwargs):
    """输出 hookSpecificOutput JSON 到 stdout"""
    output = {"hookSpecificOutput": {"hookEventName": event, **kwargs}}
    _write_stdout_json(output)


def _load_hook_input() -> dict[str, Any]:
    raw = ""
    # Try stdin first — non-blocking probe. Works for any caller that pipes data.
    # select(0) returns immediately: data ready → read; no data → skip to env var.
    # On a TTY stdin, select(0) also returns immediately (TTY has no pending data),
    # so this is safe for interactive terminals too — no isatty() check needed.
    if not sys.stdin.closed:
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.read()
        except (ValueError, OSError):
            pass
    # Fall back to env var (OpenCode plugin path)
    if not raw.strip():
        raw = os.environ.get(HOOK_INPUT_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


SPEC_RULE_IDS = ["OL09"]


def _rule_ids_for_filename(filename: str) -> list[str]:
    if filename.endswith("_impl.py"):
        return IMPL_RULE_IDS + CONSISTENCY_RULE_IDS
    if filename.endswith("_golden.py"):
        return GOLDEN_RULE_IDS + CONSISTENCY_RULE_IDS
    is_test_file = filename.startswith("test_") and filename.endswith(".py")
    if is_test_file:
        return TEST_RULE_IDS + CONSISTENCY_RULE_IDS
    if filename == SPEC_FILE:
        return SPEC_RULE_IDS
    return []


def _print_findings(findings: list[Finding]):
    has_error_fail = _has_error_fail(findings)
    result = {
        "passed": not any(f.status == "FAIL" for f in findings),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "pass": sum(1 for f in findings if f.status == "PASS"),
            "warn": sum(1 for f in findings if f.status == "WARN"),
            "info": sum(1 for f in findings if f.status == "INFO"),
            "fail": sum(1 for f in findings if f.status == "FAIL"),
            "skip": sum(1 for f in findings if f.status == "SKIP"),
            "has_error_fail": has_error_fail,
        },
    }
    _write_stdout_json(result, indent=2)


def _write_stdout_json(payload: dict[str, Any], indent: Optional[int] = None) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=indent)
    os.write(1, f"{content}\n".encode("utf-8"))


def _ensure_logs_dir() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    _ensure_logs_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emit_metric_event(ctx: CheckContext, finding: Finding, mode: str,
                       strict: bool, duration_ms: float) -> None:
    try:
        message_hash = hashlib.sha1(finding.message.encode("utf-8")).hexdigest()[:12]
        event = {
            "ts": int(time.time()),
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "rule_id": finding.rule_id,
            "severity": finding.severity,
            "status": finding.status,
            "mode": mode,
            "strict": strict,
            "duration_ms": round(float(duration_ms), 3),
            "file": finding.file,
            "message_hash": message_hash,
            "event_type": "rule_check",
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass


def _emit_gate_event(ctx: CheckContext, blocked: bool, blocking_rules: list[str]) -> None:
    try:
        event = {
            "ts": int(time.time()),
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "mode": "stop",
            "strict": os.environ.get(STRICT_ENV, "1") == "1",
            "event_type": "gate_decision",
            "blocked": blocked,
            "blocking_rules": blocking_rules,
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass


def _has_error_fail(findings: list[Finding]) -> bool:
    for finding in findings:
        if finding.status == "FAIL" and finding.severity in ("S0", "S1"):
            return True
    return False


# ─── Hook 适配层 ───

def hook_post_edit() -> int:
    """PostToolUse[Write|Edit] — 按文件类型 lint。

    默认开启“产物生成即阻断”模式：当检测到 S0/S1 FAIL 时返回 decision=block，
    由插件侧拦截本次工具调用。
    """
    os.environ[MODE_ENV] = "post-edit"
    data = _load_hook_input()
    file_path = data.get("tool_input", {}).get("file_path", "")
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return 0

    basename = os.path.basename(file_path)
    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_filename(basename)
        if stage == 0:
            return 0
    ctx = _build_context(op_dir, stage)

    # ── SPEC.md 冻结：Stage 2 完成后禁止修改 ──
    if basename == SPEC_FILE and ctx.stage is not None and ctx.stage >= 3:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=(
                f"[pypto-op-lint] {SPEC_FILE} 在 Stage 2 完成后已冻结，不允许修改。"
                "如需变更需求规格，应通过 state_transition 回退到 Stage 2 重新审核。"
            ),
        )
        return 0

    rule_ids = _rule_ids_for_filename(basename)
    if not rule_ids:
        return 0

    findings = _run_checks(ctx, rule_ids)
    fails = [f for f in findings if f.status == "FAIL"]
    error_fails = [f for f in fails if f.severity in ("S0", "S1")]
    warns = [f for f in findings if f.status == "WARN"]
    infos = [f for f in findings if f.status == "INFO"]
    if not fails and not warns and not infos:
        return 0

    sections = []
    if fails:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in fails]
        sections.append(
            "[pypto-op-lint] 以下规则违规，请立即修正后重新写入文件：\n"
            + "\n".join(lines)
            + "\n\n参考 .agents/skills/pypto-op-develop/references/execution-constraints.md"
        )
    if warns:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in warns]
        sections.append(
            "[pypto-op-lint] 以下提醒建议确认：\n"
            + "\n".join(lines)
            + "\n\n请确认以上提醒项是否需要处理。"
        )
    if infos:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in infos]
        sections.append(
            "[pypto-op-lint] 以下信息提示（不影响门禁）：\n"
            + "\n".join(lines)
        )
    context_msg = "\n\n".join(sections)

    strict_block = os.environ.get(POST_EDIT_BLOCK_ENV, "1") == "1"
    if strict_block and error_fails:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in error_fails]
        reason = (
            "[pypto-op-lint] 产物写入后即时门禁未通过（S0/S1）：\n"
            + "\n".join(lines)
            + "\n\n请先修复后再继续。"
        )
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=reason,
            additionalContext=context_msg,
        )
        # 不返回非零，避免插件侧拿不到结构化结果；实际阻断由插件根据 decision 实施。
        return 0

    _output_hook_json("PostToolUse", decision="allow", reason="", additionalContext=context_msg)
    return 0


def hook_post_bash() -> int:
    """PostToolUse[Bash] — 三态解析测试输出"""
    os.environ[MODE_ENV] = "post-bash"
    data = _load_hook_input()
    command = data.get("tool_input", {}).get("command", "")
    if not TEST_COMMAND_PATTERN.search(command):
        return 0

    stdout = data.get("tool_result", {}).get("stdout", "")
    stderr = data.get("tool_result", {}).get("stderr", "")
    exit_code = data.get("tool_result", {}).get("exit_code", 0)

    verdict = _parse_verdict(stdout, stderr, exit_code)
    detail = _verdict_detail(verdict)
    context_msg = (
        f"[pypto-op-lint parse-result] 确定性判定: {verdict}。{detail}。"
        "请以此结果为准，不要自行解读测试输出。"
    )
    _output_hook_json("PostToolUse", additionalContext=context_msg)
    return 0


def hook_pre_edit_backup() -> int:
    """PreToolUse[Write|Edit] — Stage 6 编辑 impl 前 git auto-commit（独立于 lint）"""
    data = _load_hook_input()
    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith("_impl.py"):
        return 0
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return 0
    if _get_current_stage(op_dir) != 6:
        return 0

    impl_file = os.path.basename(file_path)
    try:
        _ensure_git_init(op_dir)
        subprocess.run(
            [_git_executable(), "add", impl_file],
            cwd=op_dir, check=True, capture_output=True,
        )
        attempt = _get_stage6_attempt(op_dir)
        subprocess.run(
            [_git_executable(), "commit", "-m",
             f"backup: {impl_file} before stage6 attempt {attempt}"],
            cwd=op_dir, check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # git commit 失败（可能无变更），不拦截
    return 0


def _git_executable() -> str:
    if GIT_BIN:
        return GIT_BIN
    return "/usr/bin/git"


def _ensure_git_init(op_dir: str):
    """确保算子工作区根目录有 git 仓库"""
    current = op_dir
    while current and os.path.basename(current) != OP_WORKSPACE_DIR:
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    git_dir = current if os.path.basename(current) == OP_WORKSPACE_DIR else op_dir
    if not os.path.isdir(os.path.join(git_dir, ".git")):
        subprocess.run([_git_executable(), "init"], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run([_git_executable(), "add", "."], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run([_git_executable(), "commit", "-m", "init: operator workspace"],
                       cwd=git_dir, check=True, capture_output=True)


def _get_stage6_attempt(op_dir: str) -> int:
    """从 .orchestrator_state.json 获取 Stage 6 重试次数"""
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("stage_retry_count", {}).get("6", 0)) + 1
    except (ValueError, FileNotFoundError):
        return 1


def hook_stop() -> int:
    """Stop — agent 结束前交付门禁"""
    os.environ[MODE_ENV] = "stop"
    data = _load_hook_input()
    cwd = data.get("cwd", os.getcwd())

    op_dir = _find_nearest_op_dir(cwd)
    if not op_dir:
        return 0

    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_artifacts(op_dir)
        if stage == 0:
            return 0
    ctx = _build_context(op_dir, stage)
    applicable = [r["id"] for r in ctx.rules if ctx.stage in r.get("stages", [])]
    findings = _run_checks(ctx, applicable)
    error_fails = []
    for finding in findings:
        if finding.status == "FAIL" and finding.severity in ("S0", "S1"):
            error_fails.append(finding)

    if error_fails:
        blocking_rules = [f.rule_id for f in error_fails]
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in error_fails]
        hint_lines = [f"  - {f.rule_id}: {_rule_fix_hint(f.rule_id)}" for f in error_fails]
        _emit_gate_event(ctx, blocked=True, blocking_rules=blocking_rules)
        _output_hook_json("Stop",
            decision="block",
            reason="[pypto-op-lint] 交付门禁未通过，存在 ERROR（S0/S1）级违规：\n"
                   + "\n".join(lines)
                   + "\n\nblocking_rules: " + ", ".join(blocking_rules)
                   + "\nfix_hints:\n" + "\n".join(hint_lines)
                   + "\n\ndocs_ref: .agents/hooks/pypto-op-lint/rules.json"
                   + "\n\n**⛔ 门禁已阻断：请先修复上述 ERROR 级违规，再继续后续操作。"
                     "修复后重新运行即可。**")
        return 2
    _emit_gate_event(ctx, blocked=False, blocking_rules=[])
    return 0


def _rule_fix_hint(rule_id: str) -> str:
    hints = {
        "OL30": "在 SPEC.md front matter 中填写 supported_dtypes，并在 test 中覆盖对应 dtype",
        "OL31": "在 DESIGN.md front matter 设置 dynamic_axes，并在 impl Tensor 注解使用 pypto.DYNAMIC",
        "OL32": "在 SPEC.md front matter 的 tolerance 中填写 atol/rtol，并校准 test 断言阈值",
        "OL34": "在 SPEC.md front matter 的 p0_shapes 填写 P0 形状，并在 test 用例覆盖",
        "OL39": "为 SPEC.md/DESIGN.md/API_REPORT.md 添加 front matter 块（--- 包裹）",
        "OL40": "补齐 front matter 必填字段并修正字段类型（list/dict）",
        "OL41": "删除代码文件中的 lint 门禁输出文本，确保仅保留可执行源码/文档内容",
    }
    return hints.get(rule_id, "参考 rules.json 中该规则说明修复")


_MAX_OP_DIR_SEARCH_DEPTH = 8


def _find_nearest_op_dir(cwd: str) -> Optional[str]:
    """从 cwd 向上查找所属算子目录，避免跨算子误判。"""
    abs_cwd = os.path.abspath(cwd)
    current = abs_cwd
    for _ in range(_MAX_OP_DIR_SEARCH_DEPTH):
        if os.path.isfile(os.path.join(current, ".orchestrator_state.json")):
            return current
        if _looks_like_stateless_op_dir(current, os.path.basename(current)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _parse_verdict(stdout: str, stderr: str, exit_code: int) -> str:
    combined = f"{stdout}\n{stderr}"
    # FAIL 优先于 PASS：多 case 测试中部分失败应判定为整体失败
    if "[PRECISION_FAIL]" in combined:
        return "precision_fail"
    if "[PRECISION_PASS]" in combined:
        return "precision_pass"
    if exit_code != 0:
        return "runtime_error"
    return "no_marker"


def _verdict_detail(verdict: str) -> str:
    details = {
        "precision_pass": "输出中包含 [PRECISION_PASS] 标记，精度通过",
        "precision_fail": "输出中包含 [PRECISION_FAIL] 标记，精度失败",
        "runtime_error": "测试进程返回非零退出码，运行失败",
        "no_marker": "测试运行完毕但未检测到精度标记（[PRECISION_PASS]/[PRECISION_FAIL]）",
    }
    return details.get(verdict, f"未知判定结果: {verdict}")


def _extract_design_identifiers(content: str) -> set[str]:
    # 仅从“代码相关上下文”提取变量名，减少自然语言文本噪声：
    # 1) markdown fenced code block
    # 2) inline code (`...`)
    code_blocks = re.findall(r"```[\w-]*\n(.*?)```", content, re.S)
    inline_codes = re.findall(r"`([^`\n]+)`", content)
    scoped_text = "\n".join(code_blocks + inline_codes)

    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", scoped_text))
    blacklist = {
        "input", "output", "tensor", "shape", "dtype", "dynamic", "algorithm",
        "default", "stage", "spec", "design", "report", "loop", "tiling",
        "step", "max", "min", "round", "clamp", "cast", "mul", "add", "sub", "div",
    }
    # 只保留更像“中间变量”的命名（snake_case 优先），减少误报
    return {t for t in tokens if t not in blacklist and ("_" in t or len(t) >= 6)}


def _extract_jit_assigned_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for func in _get_jit_functions(tree):
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    return names


@register("OL37")
def check_ol37(ctx: CheckContext) -> Finding:
    """design 与 impl 的关键命名可追溯性检查（信息提示）"""
    design_content = ctx.read_file(DESIGN_FILE)
    if not design_content:
        return ctx.make_finding("OL37", "SKIP", f"{DESIGN_FILE} 不存在")
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL37", "SKIP", f"{impl_file} 不存在或无法解析")

    design_names = _extract_design_identifiers(design_content)
    impl_names = _extract_jit_assigned_names(tree)
    impl_blacklist = {
        "i", "j", "k", "n", "m", "x", "y", "z", "tmp", "temp", "result", "out",
        "input", "output",
    }
    impl_names = {n for n in impl_names if n not in impl_blacklist}

    # design 中缺少可对齐变量名时直接跳过，避免误报
    if len(design_names) < 3:
        return ctx.make_finding(
            "OL37",
            "SKIP",
            f"{DESIGN_FILE} 中可用于对齐的代码变量名不足（<3），跳过可追溯性检查",
        )

    if not impl_names:
        return ctx.make_finding("OL37", "SKIP", "未检测到 jit 内局部变量赋值")

    overlap = sorted(design_names & impl_names)
    # 命中 >=2 视为具备可追溯性；该规则为 S3 信息提示，避免因 design 关键名
    # 抽取范围较大导致“有重合却仍提示不重合”的假阳性。
    if len(overlap) >= 2:
        return ctx.make_finding(
            "OL37",
            "PASS",
            f"design/impl 命名可追溯性良好（命中 {len(overlap)} 个：{', '.join(overlap[:5])}）",
            file=impl_file,
        )

    sample_impl = ", ".join(sorted(list(impl_names))[:5])
    return ctx.make_finding(
        "OL37",
        "INFO",
        "design 与 impl 的关键命名重合较少，建议对齐中间变量命名以提升可追溯性；"
        f"当前 impl 示例变量：{sample_impl}",
        file=impl_file,
    )


# ─── 手动模式 ───

def _cmd_run(findings: list[Finding]) -> int:
    _print_findings(findings)
    return 2 if _has_error_fail(findings) else 0


def cmd_lint_impl(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, IMPL_RULE_IDS))


def cmd_lint_golden(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, GOLDEN_RULE_IDS))


def cmd_lint_test(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, TEST_RULE_IDS))


def cmd_lint_consistency(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, CONSISTENCY_RULE_IDS))


def cmd_check_gate(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    gate_rules: list[str] = []
    for rule in ctx.rules:
        # 门禁检查应覆盖当前 stage 的全部交付规则，而不仅是 gate/flow。
        # 否则会出现 Stage 5/6/7 对 impl/test 关键 S1 规则不阻断的问题。
        if rule.get("target") not in ("gate", "flow", "impl", "test", "golden"):
            continue
        if stage not in rule.get("stages", []):
            continue
        gate_rules.append(rule["id"])
    return _cmd_run(_run_checks(ctx, gate_rules))



# ─── 入口 ───

def main() -> int:
    parser = argparse.ArgumentParser(description="PyPTO 算子开发流程确定性检查工具")
    parser.add_argument("--hook",
                        choices=["post-edit", "post-bash",
                                 "pre-edit-backup", "stop"],
                        help="Hook 模式（从 stdin 读 JSON）")
    parser.add_argument("--lint-impl", action="store_true",
                        help="检查 impl 文件")
    parser.add_argument("--lint-golden", action="store_true",
                        help="检查 golden 文件")
    parser.add_argument("--lint-test", action="store_true",
                        help="检查 test 文件")
    parser.add_argument("--lint-consistency", action="store_true",
                        help="检查跨文件一致性（D5 规则）")
    parser.add_argument("--check-gate", action="store_true",
                        help="检查阶段门禁")
    parser.add_argument("--op-dir", help="算子工作目录")
    parser.add_argument("--stage", type=int, default=5,
                        help="当前阶段 (1-7)")
    args = parser.parse_args()

    if args.hook:
        return {"post-edit": hook_post_edit,
                "post-bash": hook_post_bash,
                "pre-edit-backup": hook_pre_edit_backup,
                "stop": hook_stop,
                }[args.hook]()
    elif args.lint_impl:
        if not args.op_dir:
            parser.error("--lint-impl 需要 --op-dir")
        return cmd_lint_impl(args.op_dir, args.stage)
    elif args.lint_golden:
        if not args.op_dir:
            parser.error("--lint-golden 需要 --op-dir")
        return cmd_lint_golden(args.op_dir, args.stage)
    elif args.lint_test:
        if not args.op_dir:
            parser.error("--lint-test 需要 --op-dir")
        return cmd_lint_test(args.op_dir, args.stage)
    elif args.lint_consistency:
        if not args.op_dir:
            parser.error("--lint-consistency 需要 --op-dir")
        return cmd_lint_consistency(args.op_dir, args.stage)
    elif args.check_gate:
        if not args.op_dir:
            parser.error("--check-gate 需要 --op-dir")
        return cmd_check_gate(args.op_dir, args.stage)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
