#!/usr/bin/env python3
# coding: utf-8

"""Pure-Python guards for flash_attention frontend UT case selection."""

import ast
import sys
import types
from pathlib import Path


def _install_pass_context_stub():
    pypto_block_mod = sys.modules.get("pypto_block")
    if pypto_block_mod is None:
        pypto_block_mod = types.ModuleType("pypto_block")
        sys.modules["pypto_block"] = pypto_block_mod

    pypto_core_mod = sys.modules.get("pypto_block.pypto_core")
    if pypto_core_mod is None:
        pypto_core_mod = types.ModuleType("pypto_block.pypto_core")
        sys.modules["pypto_block.pypto_core"] = pypto_core_mod

    passes_mod = types.ModuleType("pypto_block.pypto_core.passes")

    class VerificationMode:
        BEFORE_AND_AFTER = "before_and_after"

    class VerificationInstrument:
        def __init__(self, *_args, **_kwargs):
            pass

    class PassContext:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    passes_mod.PassContext = PassContext
    passes_mod.VerificationInstrument = VerificationInstrument
    passes_mod.VerificationMode = VerificationMode

    pypto_core_mod.passes = passes_mod
    pypto_block_mod.pypto_core = pypto_core_mod
    sys.modules["pypto_block.pypto_core.passes"] = passes_mod


_install_pass_context_stub()


FLASH_ATTENTION_TEST = (
    Path(__file__).resolve().parents[1] / "flash_attention" / "test_fa.py"
)


def _parse_module():
    source = FLASH_ATTENTION_TEST.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(FLASH_ATTENTION_TEST))


def _extract_int_constant(module: ast.Module, name: str) -> int:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                    return node.value.value
    raise AssertionError(f"Could not find integer constant {name} in {FLASH_ATTENTION_TEST}")


def _extract_test_cases(module: ast.Module):
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "test_fa_k":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.For):
                continue
            if not isinstance(inner.target, ast.Tuple):
                continue
            names = [elt.id for elt in inner.target.elts if isinstance(elt, ast.Name)]
            if names != ["sq", "skv", "d", "num_cores"]:
                continue
            if not isinstance(inner.iter, ast.List):
                continue
            cases = []
            for elt in inner.iter.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) != 4:
                    continue
                sq_node, skv_node, _d_node, cores_node = elt.elts
                if not (
                    isinstance(sq_node, ast.Constant)
                    and isinstance(sq_node.value, int)
                    and isinstance(skv_node, ast.Constant)
                    and isinstance(skv_node.value, int)
                    and isinstance(cores_node, ast.Constant)
                    and isinstance(cores_node.value, int)
                ):
                    continue
                cases.append((sq_node.value, skv_node.value, cores_node.value))
            return cases
    raise AssertionError(f"Could not find test_fa_k case list in {FLASH_ATTENTION_TEST}")


def test_test_fa_k_keeps_supported_single_kv_tile_cases():
    """The UT should stay within the file's documented skv_tiles==1 support window."""
    module = _parse_module()
    tkv = _extract_int_constant(module, "TKV")
    cases = _extract_test_cases(module)

    assert cases, "Expected at least one flash_attention UT case"
    assert all(skv <= tkv for _sq, skv, _cores in cases), (
        "test_fa_k contains skv > TKV even though the file documents "
        "multi-KV-tile cross-core sync as a known limitation"
    )
