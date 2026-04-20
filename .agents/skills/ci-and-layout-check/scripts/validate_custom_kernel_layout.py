#!/usr/bin/env python3
"""
Validate custom/<operator_name>/ layout against pypto-kernel-custom-skills rules.

Runs without NPU. Intended for CI and pre-commit.

When custom/<op>/ contains kernel work (test_<op>.py and/or <op>_module*.py), checks:
  - custom/plan/<op>.md exists
  - test_<op>.py exists and references detailed_tensor_compare
  - Staged files <op>_module{SUFFIX}.py use cumulative suffixes 1, 12, 123, ...
  - No `for ... in range(...)` in PyPTO kernel-side code in custom/<op>/**/*.py (AST scan):
    - Applies to `pypto_function`, `*_kernel_impl*`, `*_kernel_npu*`, and any `pypto_*` helper
      (including nested functions inside those).
    - Does not apply inside golden/reference functions (`torch_golden_reference`, `forward_ref`,
      `detailed_tensor_compare`, or names starting with `torch_golden`), where plain Python loops are OK.
  - `pypto.view(...)` is not used as a reshape (AST scan):
    - When `shape` and `offsets` are both literal lists, their lengths MUST match.
    - When `shape` and `valid_shape` are both literal lists, their lengths MUST match.
    - `pypto.view` extracts a same-rank sub-view — it is NOT a reshape. Changing the
      rank of a tensor via view is the #1 misuse pattern agents produce.
  - `pypto.set_cube_tile_shapes(m, k, n, ...)` uses valid literal tile lists (AST scan):
    - Each of `m`, `k`, `n` must be a 2-element list `[L0, L1]` (not 1-element, not 3+).
    - When `L0` and `L1` are int literals: require `0 < L0 <= L1` and `L1 % L0 == 0`.
    - See `docs/api/config/pypto-set_cube_tile_shapes.md`. The "one-element list"
      misuse (`[16], [32], [64]`) is the #1 tile-config crash pattern.
  - `pypto.loop` usage in kernel files (AST scan):
    - Warns if a PyPTO kernel function is detected but no `pypto.loop` calls are found.
    - Intended as a guidance check for kernels that should use loop constructs.

Exit 0 if nothing to check or all checks pass; 1 on failure.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


def _expected_staged_suffixes(n: int) -> list[str]:
    """Suffixes for N cumulative modules: '1', '12', ..., '123...N'."""
    return ["".join(str(j) for j in range(1, k + 1)) for k in range(1, n + 1)]


def _is_range_call(node: ast.AST) -> bool:
    """True if `node` is a call to builtin `range` (Name only; not methods)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    )


def _is_golden_function_name(name: str) -> bool:
    """Reference / golden paths may use normal Python `for ... in range`."""
    if name in ("torch_golden_reference", "forward_ref", "detailed_tensor_compare"):
        return True
    return name.startswith("torch_golden")


def _is_kernel_function_name(name: str) -> bool:
    """PyPTO-side code where tile/host iteration should use `pypto.loop`, not `for ... in range`."""
    if name == "pypto_function":
        return True
    if "kernel_impl" in name or "kernel_npu" in name:
        return True
    return name.startswith("pypto_")


def _stack_has_golden(stack: list[str]) -> bool:
    return any(_is_golden_function_name(n) for n in stack)


def _stack_has_kernel(stack: list[str]) -> bool:
    return any(_is_kernel_function_name(n) for n in stack)


def find_kernel_for_range_violations(source: str, rel_path: Path) -> list[str]:
    """
    Flag `for` loops whose iterator is `range(...)` when they appear in kernel-side functions.

    Golden/reference functions (see `_is_golden_function_name`) are skipped, including nested
    defs under them. Nested defs under `_your_op_kernel_impl` / `pypto_*` / etc. are still checked.

    Reinforces skills/lead-orchestrator/references/rules.md Prohibition B: use `pypto.loop` for kernel iteration, not Python `range` loops.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{rel_path}: syntax error (cannot scan for range in kernel): {e}"]

    stack: list[str] = []

    def _append_for_range_error(lineno: int | str) -> None:
        innermost = stack[-1] if stack else "?"
        if innermost == "pypto_function":
            hint = (
                "forbidden in `pypto_function` — use `pypto.loop` in "
                "`_your_op_kernel_impl` / `your_op_kernel_npu`"
            )
        else:
            hint = (
                "forbidden in PyPTO kernel code — use `pypto.loop` for this iteration "
                "(not `for ... in range`)"
            )
        errors.append(
            f"{rel_path}:{lineno}: `for ... in range(...)` {hint} "
            f"(see skills/lead-orchestrator/references/rules.md rule 18). "
            f"Context: {' -> '.join(stack)}"
        )

    def _scan_block(stmts: list[ast.stmt]) -> None:
        for st in stmts:
            _scan_stmt(st)

    def _scan_stmt(st: ast.stmt) -> None:
        if isinstance(st, ast.For):
            if (
                _is_range_call(st.iter)
                and not _stack_has_golden(stack)
                and _stack_has_kernel(stack)
            ):
                _append_for_range_error(getattr(st, "lineno", "?"))
            _scan_block(st.body)
            _scan_block(st.orelse)
            return
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(st)
            return
        if isinstance(st, ast.ClassDef):
            for item in st.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_function(item)
                else:
                    _scan_stmt(item)
            return
        if isinstance(st, ast.If):
            _scan_block(st.body)
            _scan_block(st.orelse)
            return
        if isinstance(st, ast.While):
            _scan_block(st.body)
            _scan_block(st.orelse)
            return
        if isinstance(st, (ast.With, ast.AsyncWith)):
            _scan_block(st.body)
            return
        if isinstance(st, ast.Try):
            _scan_block(st.body)
            for h in st.handlers:
                _scan_block(h.body)
            _scan_block(st.orelse)
            _scan_block(st.finalbody)
            return
        if hasattr(ast, "TryStar") and isinstance(st, ast.TryStar):
            _scan_block(st.body)
            for h in st.handlers:
                _scan_block(h.body)
            _scan_block(st.orelse)
            _scan_block(st.finalbody)
            return
        if hasattr(ast, "Match") and isinstance(st, ast.Match):
            for case in st.cases:
                _scan_block(case.body)
            return

    def visit_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        stack.append(node.name)
        try:
            # Always walk statements so nested defs under golden / non-kernel parents are reached;
            # `for ... in range` is only flagged when `_scan_stmt` sees non-golden + kernel stack.
            _scan_block(node.body)
        finally:
            stack.pop()

    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(top)
        elif isinstance(top, ast.ClassDef):
            for item in top.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_function(item)
                else:
                    _scan_stmt(item)

    return errors


def _is_pypto_view_call(node: ast.AST) -> bool:
    """True if `node` is a call to `pypto.view(...)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "view"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pypto"
    )


def _get_call_arg(call: ast.Call, pos_index: int, kw_name: str) -> ast.AST | None:
    """Return the AST node for the arg at positional index or matching keyword, else None."""
    if len(call.args) > pos_index:
        return call.args[pos_index]
    for kw in call.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _list_len(node: ast.AST | None) -> int | None:
    """Return len(elts) if node is a literal list, else None. Empty list returns 0."""
    if isinstance(node, ast.List):
        return len(node.elts)
    return None


def find_pypto_view_reshape_misuse(source: str, rel_path: Path) -> list[str]:
    """
    Flag `pypto.view(...)` calls that change tensor rank — the common reshape misuse.

    Detectable statically when arguments are literal lists:
      - len(shape) != len(offsets)         → clear bug (user's prompt example)
      - len(shape) != len(valid_shape)     → clear bug when valid_shape is non-empty

    pypto.view extracts a same-rank sub-view. It is NOT a reshape. See
    `docs/api/operation/pypto-view.md` and `skills/debugging/DEBUG.md` §9.4.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{rel_path}: syntax error (cannot scan pypto.view): {e}"]

    for node in ast.walk(tree):
        if not _is_pypto_view_call(node):
            continue

        shape_node = _get_call_arg(node, 1, "shape")
        offsets_node = _get_call_arg(node, 2, "offsets")
        valid_shape_node = _get_call_arg(node, 3, "valid_shape")

        shape_len = _list_len(shape_node)
        offsets_len = _list_len(offsets_node)
        valid_len = _list_len(valid_shape_node)

        lineno = getattr(node, "lineno", "?")

        if (
            shape_len is not None
            and offsets_len is not None
            and shape_len != offsets_len
        ):
            errors.append(
                f"{rel_path}:{lineno}: pypto.view shape/offsets rank mismatch "
                f"(shape has {shape_len} dims, offsets has {offsets_len} dims). "
                f"pypto.view is NOT reshape — it extracts a same-rank sub-view. "
                f"See `docs/api/operation/pypto-view.md` and "
                f"`skills/debugging/DEBUG.md` §9.4."
            )

        if (
            shape_len is not None
            and valid_len is not None
            and valid_len != 0
            and shape_len != valid_len
        ):
            errors.append(
                f"{rel_path}:{lineno}: pypto.view shape/valid_shape rank mismatch "
                f"(shape has {shape_len} dims, valid_shape has {valid_len} dims). "
                f"All of shape, offsets, and valid_shape must share the same rank. "
                f"See `docs/api/operation/pypto-view.md`."
            )

    return errors


def _is_pypto_loop_call(node: ast.AST) -> bool:
    """True if `node` is a call to `pypto.loop(...)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loop"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pypto"
    )


def find_missing_pypto_loop(source: str, rel_path: Path) -> list[str]:
    """
    Check if a PyPTO kernel file contains at least one `pypto.loop` call.

    PyPTO kernels that use iteration should employ `pypto.loop` for dynamic axes.
    This is a warning check to help ensure kernels follow best practices.

    Applies to files matching:
      - <op>_module*.py (staged kernel files)
      - <op>_test.py and test_<op>.py

    Returns list of warnings (may be empty if pypto.loop is present or not applicable).
    """
    warnings: list[str] = []

    # Check if this is a kernel file that should have pypto.loop
    # (typically staged kernel modules or full kernels)
    is_kernel_file = (
        "_module" in rel_path.name and rel_path.name.endswith(".py")
    ) or (
        rel_path.name.startswith("test_") and rel_path.name.endswith(".py")
    ) or (
        "_test.py" in rel_path.name
    )

    if not is_kernel_file:
        return warnings

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{rel_path}: syntax error (cannot scan for pypto.loop): {e}"]

    # Search for pypto.loop calls anywhere in the tree
    found_loop = False
    found_pypto_function = False

    for node in ast.walk(tree):
        if _is_pypto_loop_call(node):
            found_loop = True
            break
        # Also check if any pypto_function or kernel function exists
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_kernel_function_name(node.name):
                found_pypto_function = True

    # Only warn if we found kernel functions but no pypto.loop
    if found_pypto_function and not found_loop:
        warnings.append(
            f"{rel_path}: PyPTO kernel function detected but no `pypto.loop` calls found. "
            f"If the kernel uses iteration over dynamic axes, consider using `pypto.loop`. "
            f"See `skills/debugging/DEBUG.md` §9.15 and "
            f"`docs/api/controlflow/pypto-loop.md`."
        )

    return warnings


def _is_set_cube_tile_shapes_call(node: ast.AST) -> bool:
    """True if `node` is a call to `pypto.set_cube_tile_shapes(...)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_cube_tile_shapes"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pypto"
    )


def _int_literal_value(node: ast.AST | None) -> int | None:
    """Return int value if `node` is a positive int literal; else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    # `-1` etc. would be UnaryOp; we do NOT accept negative so leave as None.
    return None


def _tile_pair_errors(axis: str, node: ast.AST | None) -> list[str]:
    """
    Validate that `node` is a 2-element list `[L0, L1]` with 0 < L0 <= L1 and L1 % L0 == 0.

    Returns a list of human-readable error fragments (without filename/lineno prefix).
    Silently skips when the argument is not a literal list (e.g. variable reference).
    """
    errs: list[str] = []
    if not isinstance(node, ast.List):
        return errs  # not a literal list; can't statically validate
    n = len(node.elts)
    if n != 2:
        errs.append(
            f"`{axis}` must be a 2-element list `[{axis}L0, {axis}L1]`, got {n}-element list. "
            f"pypto.set_cube_tile_shapes takes [L0, L1] per axis, NOT [L0] only."
        )
        return errs
    l0 = _int_literal_value(node.elts[0])
    l1 = _int_literal_value(node.elts[1])
    if l0 is None or l1 is None:
        return errs  # non-literal ints → skip numeric checks
    if l0 <= 0 or l1 <= 0:
        errs.append(f"`{axis}` values must be positive: got [{l0}, {l1}].")
        return errs
    if l0 > l1:
        errs.append(
            f"`{axis}` requires {axis}L0 <= {axis}L1, got [{l0}, {l1}] "
            f"({axis}L0 > {axis}L1)."
        )
    if l1 % l0 != 0:
        errs.append(
            f"`{axis}` requires {axis}L1 % {axis}L0 == 0, got [{l0}, {l1}] "
            f"({l1} % {l0} = {l1 % l0})."
        )
    return errs


def find_set_cube_tile_shapes_misuse(source: str, rel_path: Path) -> list[str]:
    """
    Flag `pypto.set_cube_tile_shapes(m, k, n, enable_split_k=...)` calls with invalid
    literal tile lists. Skips non-literal args.

    Statically caught bugs:
      - m / k / n given as 1-element or 3+-element list  → crash at tile-config time
      - L0 > L1 or L1 % L0 != 0                          → tile divisibility violation

    See `docs/api/config/pypto-set_cube_tile_shapes.md`.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{rel_path}: syntax error (cannot scan set_cube_tile_shapes): {e}"]

    for node in ast.walk(tree):
        if not _is_set_cube_tile_shapes_call(node):
            continue

        m_node = _get_call_arg(node, 0, "m")
        k_node = _get_call_arg(node, 1, "k")
        n_node = _get_call_arg(node, 2, "n")
        lineno = getattr(node, "lineno", "?")

        for axis, arg in (("m", m_node), ("k", k_node), ("n", n_node)):
            for msg in _tile_pair_errors(axis, arg):
                errors.append(
                    f"{rel_path}:{lineno}: pypto.set_cube_tile_shapes {msg} "
                    f"See `docs/api/config/pypto-set_cube_tile_shapes.md`."
                )

    return errors


def should_validate_kernel(custom_op: Path, op: str) -> bool:
    """Only validate dirs that clearly started kernel work."""
    if (custom_op / f"test_{op}.py").is_file():
        return True
    if any(custom_op.glob(f"{op}_module*.py")):
        return True
    return False


def validate_operator(repo_root: Path, op: str) -> list[str]:
    errors: list[str] = []
    custom_op = repo_root / "custom" / op
    plan_md = repo_root / "custom" / "plan" / f"{op}.md"
    test_py = custom_op / f"test_{op}.py"

    if not plan_md.is_file():
        errors.append(f"Missing plan file: {plan_md.relative_to(repo_root)}")

    if not test_py.is_file():
        errors.append(f"Missing E2E runner: {test_py.relative_to(repo_root)}")
    else:
        text = test_py.read_text(encoding="utf-8", errors="replace")
        if "detailed_tensor_compare" not in text:
            errors.append(
                f"{test_py.relative_to(repo_root)} must reference detailed_tensor_compare "
                "(import or call)."
            )

    module_re = re.compile(rf"^{re.escape(op)}_module(\d+)\.py$")
    staged = sorted(
        p for p in custom_op.iterdir() if p.is_file() and module_re.match(p.name)
    )
    if staged:
        suffixes_sorted = []
        for p in staged:
            m = module_re.match(p.name)
            assert m is not None
            suffixes_sorted.append(m.group(1))
        suffixes_sorted = sorted(suffixes_sorted, key=lambda s: (len(s), s))
        n = len(suffixes_sorted)
        expected = _expected_staged_suffixes(n)
        if suffixes_sorted != expected:
            errors.append(
                f"Staged module files must be {op}_module1.py, {op}_module12.py, … "
                f"(cumulative digit suffixes). Found: {[p.name for p in staged]}, "
                f"expected suffixes {expected}."
            )

    for py_file in sorted(custom_op.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            errors.append(f"Cannot read {py_file.relative_to(repo_root)}: {e}")
            continue
        rel = py_file.relative_to(repo_root)
        for msg in find_kernel_for_range_violations(text, rel):
            errors.append(msg)
        for msg in find_pypto_view_reshape_misuse(text, rel):
            errors.append(msg)
        for msg in find_set_cube_tile_shapes_misuse(text, rel):
            errors.append(msg)
        for msg in find_missing_pypto_loop(text, rel):
            errors.append(msg)

    return errors


def discover_operators(custom_dir: Path) -> list[str]:
    if not custom_dir.is_dir():
        return []
    return sorted(
        p.name for p in custom_dir.iterdir() if p.is_dir() and p.name != "plan"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    args = ap.parse_args()
    repo_root = args.repo_root.resolve()
    custom_dir = repo_root / "custom"
    ops = discover_operators(custom_dir)
    if not ops:
        print("validate_custom_kernel_layout: no custom/<operator>/ — skip.")
        return 0

    all_errors: list[str] = []
    checked = 0
    for op in ops:
        custom_op = repo_root / "custom" / op
        if not should_validate_kernel(custom_op, op):
            continue
        checked += 1
        for e in validate_operator(repo_root, op):
            all_errors.append(f"[{op}] {e}")

    if all_errors:
        print("validate_custom_kernel_layout: FAILED", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    if checked == 0:
        print(
            f"validate_custom_kernel_layout: OK ({len(ops)} custom dir(s), "
            "no kernel artifacts yet — skip)."
        )
    else:
        print(f"validate_custom_kernel_layout: OK ({checked} operator kernel layout(s) checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
