#!/usr/bin/env python3
"""pypto_op_lint.py — 算子开发流程确定性检查工具

通过 Claude Code hooks / OpenCode plugin 自动触发，
也可手动执行进行调试。
"""

import ast
import json
import os
import re
import sys
import subprocess
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_COMMAND_PATTERN = re.compile(r"python3?\s+.*test_\w+\.py")

IMPL_RULE_IDS = ["OL01", "OL02", "OL03", "OL04", "OL05", "OL06", "OL07", "OL08", "OL16", "OL23"]
GOLDEN_RULE_IDS = ["OL15"]
TEST_RULE_IDS = ["OL17", "OL18", "OL19", "OL20", "OL21", "OL22"]


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
        except SyntaxError:
            return None
        self._ast_cache[filename] = tree
        return tree

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
    """检查注解是否为合法的非 Tensor 标量参数类型（int/float/bool）"""
    if isinstance(annotation, ast.Name):
        return annotation.id in ("int", "float", "bool")
    return False


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
        "未找到 @pypto.frontend.jit 装饰器", file=impl_file)


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
    return ctx.make_finding("OL07", "FAIL", "缺少 import pypto", file=impl_file)


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
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL23", "SKIP", f"{impl_file} 不存在或无法解析")
    jit_funcs = _get_jit_functions(tree)
    if not jit_funcs:
        return ctx.make_finding("OL23", "SKIP", "无 jit 函数")
    if any(_has_loop_structure(func) for func in jit_funcs):
        return ctx.make_finding("OL23", "PASS",
            "检测到 loop 相关结构", file=impl_file)
    return ctx.make_finding("OL23", "WARN",
        "未检测到 loop 相关结构；若该算子需要分块或迭代，请确认设计已说明无需 loop",
        file=impl_file)


# ─── D2: 工件完整性与流程合规 (OL09-OL14, OL24) ───

@register("OL09")
def check_ol09(ctx: CheckContext) -> Finding:
    if not ctx.file_exists("spec.md"):
        return ctx.make_finding("OL09", "FAIL", "spec.md 不存在")
    content = ctx.read_file("spec.md")
    if ctx.op_name not in content:
        return ctx.make_finding("OL09", "FAIL",
            f"spec.md 中未包含算子名 '{ctx.op_name}'", file="spec.md")
    missing = _missing_required_groups(content, ctx.get_rule("OL09"))
    if missing:
        return ctx.make_finding("OL09", "FAIL",
            f"spec.md 缺少必需内容: {', '.join(missing)}", file="spec.md")
    return ctx.make_finding("OL09", "PASS",
        "spec.md 含算子名、公式、输入输出规格与精度要求", file="spec.md")


@register("OL10")
def check_ol10(ctx: CheckContext) -> Finding:
    if not ctx.file_exists("api_report.md"):
        return ctx.make_finding("OL10", "FAIL", "api_report.md 不存在")
    content = ctx.read_file("api_report.md")
    missing = _missing_required_groups(content, ctx.get_rule("OL10"))
    if missing:
        return ctx.make_finding("OL10", "FAIL",
            f"api_report.md 缺少必需内容: {', '.join(missing)}",
            file="api_report.md")
    return ctx.make_finding("OL10", "PASS",
        "api_report.md 含 API 映射、约束与 Tiling 说明", file="api_report.md")


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
            ["python3", "-c", probe_code],
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
    if not ctx.file_exists("design.md"):
        return ctx.make_finding("OL12", "FAIL", "design.md 不存在")
    content = ctx.read_file("design.md")
    missing = _missing_required_groups(content, ctx.get_rule("OL12"))
    if missing:
        return ctx.make_finding("OL12", "FAIL",
            f"design.md 缺少必需内容: {', '.join(missing)}", file="design.md")
    return ctx.make_finding("OL12", "PASS",
        "design.md 含 API 映射、数据切分/Tiling 与验证方案", file="design.md")


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
    except (json.JSONDecodeError, ValueError):
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
    """test 必须处理 TILE_FWK_DEVICE_ID 环境变量"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL20", "SKIP", f"{test_file} 不存在")
    if "TILE_FWK_DEVICE_ID" in source:
        return ctx.make_finding("OL20", "PASS",
            "找到 TILE_FWK_DEVICE_ID 处理", file=test_file)
    return ctx.make_finding("OL20", "FAIL",
        "未找到 TILE_FWK_DEVICE_ID 环境变量处理", file=test_file)


@register("OL21")
def check_ol21(ctx: CheckContext) -> Finding:
    """test 必须有 Level 0 和 Level 1 两级测试函数"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在或无法解析")
    func_names = [
        node.name.lower() for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    has_level0 = any("level0" in name for name in func_names)
    has_level1 = any("level1" in name for name in func_names)
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


# ─── 辅助函数 ───

def _load_rules() -> list[dict[str, Any]]:
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def _missing_required_groups(content: str, rule: dict[str, Any]) -> list[str]:
    normalized = content.lower()
    missing = []
    for group in rule.get("required_groups", []):
        keywords = group.get("keywords", [])
        if not any(str(keyword).lower() in normalized for keyword in keywords):
            missing.append(group.get("name", "未命名分组"))
    return missing


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
        "spec.md",
        "api_report.md",
        "design.md",
        "README.md",
    }
    return len(files & expected) >= 2


def _infer_stage_from_filename(filename: str) -> int:
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
    if "design.md" in files or f"{op_name}_golden.py" in files:
        return 4
    if "api_report.md" in files:
        return 3
    if "spec.md" in files:
        return 2
    return 0


def _get_current_stage(op_dir: str) -> int:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return 0
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("current_stage", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _get_op_name(op_dir: str) -> str:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                name = json.load(f).get("operator_name", "")
                if name:
                    return name
        except (json.JSONDecodeError, ValueError):
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
    for rid in rule_ids:
        rule = ctx.get_rule(rid)
        if ctx.stage not in rule.get("stages", []):
            findings.append(ctx.make_finding(rid, "SKIP", "当前阶段不适用"))
            continue
        checker = CHECKERS.get(rid)
        if not checker:
            findings.append(ctx.make_finding(rid, "SKIP", "检查函数未注册"))
            continue
        findings.append(checker(ctx))
    return findings


def _output_hook_json(event: str, **kwargs):
    """输出 hookSpecificOutput JSON 到 stdout"""
    output = {"hookSpecificOutput": {"hookEventName": event, **kwargs}}
    print(json.dumps(output, ensure_ascii=False))


def _load_hook_input_or_exit() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)
    if isinstance(data, dict):
        return data
    return {}


def _rule_ids_for_filename(filename: str) -> list[str]:
    if filename.endswith("_impl.py"):
        return IMPL_RULE_IDS
    if filename.endswith("_golden.py"):
        return GOLDEN_RULE_IDS
    if filename.startswith("test_") and filename.endswith(".py"):
        return TEST_RULE_IDS
    return []


def _print_findings(findings: list[Finding]):
    result = {
        "passed": not any(f.status == "FAIL" for f in findings),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "pass": sum(1 for f in findings if f.status == "PASS"),
            "warn": sum(1 for f in findings if f.status == "WARN"),
            "fail": sum(1 for f in findings if f.status == "FAIL"),
            "skip": sum(1 for f in findings if f.status == "SKIP"),
            "has_s0_fail": any(
                f.status == "FAIL" and f.severity == "S0" for f in findings
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ─── Hook 适配层 ───

def hook_post_edit():
    """PostToolUse[Write|Edit] — 按文件类型 lint"""
    data = _load_hook_input_or_exit()
    file_path = data.get("tool_input", {}).get("file_path", "")
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        sys.exit(0)

    basename = os.path.basename(file_path)
    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_filename(basename)
        if stage == 0:
            sys.exit(0)
    ctx = _build_context(op_dir, stage)

    rule_ids = _rule_ids_for_filename(basename)
    if not rule_ids:
        sys.exit(0)

    findings = _run_checks(ctx, rule_ids)
    fails = [f for f in findings if f.status == "FAIL"]
    warns = [f for f in findings if f.status == "WARN"]
    if not fails and not warns:
        sys.exit(0)

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
            + "\n\n若当前算子确实无需 loop，可忽略此提醒。"
        )
    context_msg = "\n\n".join(sections)
    _output_hook_json("PostToolUse", additionalContext=context_msg)
    sys.exit(0)


def hook_post_bash():
    """PostToolUse[Bash] — 三态解析测试输出"""
    data = _load_hook_input_or_exit()
    command = data.get("tool_input", {}).get("command", "")
    if not TEST_COMMAND_PATTERN.search(command):
        sys.exit(0)

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
    sys.exit(0)


def hook_pre_edit():
    """PreToolUse[Write|Edit] — 纯 lint 检查，无副作用。

    Git 备份逻辑已拆离到独立的 hook_pre_edit_backup()，
    通过 --hook pre-edit-backup 触发，避免 lint 与版本控制写操作耦合。
    """
    # 当前 pre-edit 阶段暂无需前置 lint 检查，保留入口供后续扩展
    sys.exit(0)


def hook_pre_edit_backup():
    """PreToolUse[Write|Edit] — Stage 6 编辑 impl 前 git auto-commit（独立于 lint）"""
    data = _load_hook_input_or_exit()
    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith("_impl.py"):
        sys.exit(0)
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        sys.exit(0)
    if _get_current_stage(op_dir) != 6:
        sys.exit(0)

    impl_file = os.path.basename(file_path)
    try:
        _ensure_git_init(op_dir)
        subprocess.run(
            ["git", "add", impl_file],
            cwd=op_dir, check=True, capture_output=True,
        )
        attempt = _get_stage6_attempt(op_dir)
        subprocess.run(
            ["git", "commit", "-m",
             f"backup: {impl_file} before stage6 attempt {attempt}"],
            cwd=op_dir, check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # git commit 失败（可能无变更），不拦截
    sys.exit(0)


def _ensure_git_init(op_dir: str):
    """确保 custom/ 级别有 git 仓库"""
    current = op_dir
    while current and os.path.basename(current) != "custom":
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    git_dir = current if os.path.basename(current) == "custom" else op_dir
    if not os.path.isdir(os.path.join(git_dir, ".git")):
        subprocess.run(["git", "init"], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init: operator workspace"],
                       cwd=git_dir, check=True, capture_output=True)


def _get_stage6_attempt(op_dir: str) -> int:
    """从 .orchestrator_state.json 获取 Stage 6 重试次数"""
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("stage_retry_count", {}).get("6", 0)) + 1
    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        return 1


def hook_stop():
    """Stop — agent 结束前交付门禁"""
    data = _load_hook_input_or_exit()
    cwd = data.get("cwd", os.getcwd())

    op_dir = _find_nearest_op_dir(cwd)
    if not op_dir:
        sys.exit(0)

    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_artifacts(op_dir)
        if stage == 0:
            sys.exit(0)
    ctx = _build_context(op_dir, stage)
    applicable = [r["id"] for r in ctx.rules if ctx.stage in r.get("stages", [])]
    findings = _run_checks(ctx, applicable)
    s0_fails = [f for f in findings if f.status == "FAIL" and f.severity == "S0"]

    if s0_fails:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in s0_fails]
        _output_hook_json("Stop",
            decision="block",
            reason="[pypto-op-lint] 交付门禁未通过，存在 S0 级违规：\n"
                   + "\n".join(lines))
        sys.exit(2)
    sys.exit(0)


def _find_nearest_op_dir(cwd: str) -> Optional[str]:
    """从 cwd 查找最近的算子目录。

    策略：先检查 cwd 自身，再向下搜索最多 2 层（仅在 custom/ 子树内）。
    避免从仓库根目录或 /tmp 等大目录误匹配到不相关的子目录。
    """
    abs_cwd = os.path.abspath(cwd)

    # 1. cwd 自身就是算子目录
    if os.path.isfile(os.path.join(abs_cwd, ".orchestrator_state.json")):
        return abs_cwd
    if _looks_like_stateless_op_dir(abs_cwd, os.path.basename(abs_cwd)):
        return abs_cwd

    # 2. 仅在 custom/ 子树内向下搜索（最多 2 层）
    custom_dir = abs_cwd if os.path.basename(abs_cwd) == "custom" else None
    if custom_dir is None:
        candidate = os.path.join(abs_cwd, "custom")
        if os.path.isdir(candidate):
            custom_dir = candidate
    if custom_dir is None:
        return None

    stateless_candidates: list[str] = []
    for root, dirs, files in os.walk(custom_dir):
        depth = root.replace(custom_dir, "").count(os.sep)
        if depth > 2:
            dirs.clear()
            continue
        if ".orchestrator_state.json" in files:
            return root
        if _looks_like_stateless_op_dir(root, os.path.basename(root)):
            stateless_candidates.append(root)
    if len(stateless_candidates) == 1:
        return stateless_candidates[0]
    return None


def _parse_verdict(stdout: str, stderr: str, exit_code: int) -> str:
    combined = f"{stdout}\n{stderr}"
    # FAIL 优先于 PASS：多 case 测试中部分失败应判定为整体失败
    if "[PRECISION_FAIL]" in combined:
        return "precision_fail"
    if "[PRECISION_PASS]" in combined:
        return "precision_pass"
    if exit_code != 0 and (stderr.strip() or "traceback" in combined.lower()):
        return "other"
    return "other"


def _verdict_detail(verdict: str) -> str:
    if verdict == "precision_pass":
        return "输出中包含 [PRECISION_PASS] 标记，精度通过"
    if verdict == "precision_fail":
        return "输出中包含 [PRECISION_FAIL] 标记，精度失败"
    return "未检测到精度标记，可能是运行失败或其他错误"


# ─── 手动模式 ───

def _cmd_run_and_exit(findings: list[Finding]):
    _print_findings(findings)
    has_s0_fail = any(f.status == "FAIL" and f.severity == "S0" for f in findings)
    sys.exit(2 if has_s0_fail else 0)


def cmd_lint_impl(op_dir: str, stage: int):
    ctx = _build_context(op_dir, stage)
    _cmd_run_and_exit(_run_checks(ctx, IMPL_RULE_IDS))


def cmd_lint_golden(op_dir: str, stage: int):
    ctx = _build_context(op_dir, stage)
    _cmd_run_and_exit(_run_checks(ctx, GOLDEN_RULE_IDS))


def cmd_lint_test(op_dir: str, stage: int):
    ctx = _build_context(op_dir, stage)
    _cmd_run_and_exit(_run_checks(ctx, TEST_RULE_IDS))


def cmd_check_gate(op_dir: str, stage: int):
    ctx = _build_context(op_dir, stage)
    gate_rules = [r["id"] for r in ctx.rules
                  if r.get("target") in ("gate", "flow")
                  and stage in r.get("stages", [])]
    # OL23 target=impl 但属于门禁 advisory，显式纳入
    if stage in (5, 6, 7) and "OL23" not in gate_rules:
        gate_rules.append("OL23")
    _cmd_run_and_exit(_run_checks(ctx, gate_rules))


# ─── 入口 ───

def main():
    parser = argparse.ArgumentParser(description="PyPTO 算子开发流程确定性检查工具")
    parser.add_argument("--hook",
                        choices=["post-edit", "post-bash", "pre-edit",
                                 "pre-edit-backup", "stop"],
                        help="Hook 模式（从 stdin 读 JSON）")
    parser.add_argument("--lint-impl", action="store_true",
                        help="检查 impl 文件")
    parser.add_argument("--lint-golden", action="store_true",
                        help="检查 golden 文件")
    parser.add_argument("--lint-test", action="store_true",
                        help="检查 test 文件")
    parser.add_argument("--check-gate", action="store_true",
                        help="检查阶段门禁")
    parser.add_argument("--op-dir", help="算子工作目录")
    parser.add_argument("--stage", type=int, default=5,
                        help="当前阶段 (1-7)")
    args = parser.parse_args()

    if args.hook:
        {"post-edit": hook_post_edit,
         "post-bash": hook_post_bash,
         "pre-edit": hook_pre_edit,
         "pre-edit-backup": hook_pre_edit_backup,
         "stop": hook_stop,
         }[args.hook]()
    elif args.lint_impl:
        if not args.op_dir:
            parser.error("--lint-impl 需要 --op-dir")
        cmd_lint_impl(args.op_dir, args.stage)
    elif args.lint_golden:
        if not args.op_dir:
            parser.error("--lint-golden 需要 --op-dir")
        cmd_lint_golden(args.op_dir, args.stage)
    elif args.lint_test:
        if not args.op_dir:
            parser.error("--lint-test 需要 --op-dir")
        cmd_lint_test(args.op_dir, args.stage)
    elif args.check_gate:
        if not args.op_dir:
            parser.error("--check-gate 需要 --op-dir")
        cmd_check_gate(args.op_dir, args.stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
