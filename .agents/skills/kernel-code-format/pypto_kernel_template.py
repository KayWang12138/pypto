# Copyright / license: follow your project policy.
#
# =============================================================================
# pypto_kernel_template.py — canonical single-file kernel layout (generic)
# =============================================================================
#
# CANONICAL PATH (this bundle)
# -----------------------------
#   `.agents/skills/kernel-code-format/pypto_kernel_template.py`
#
# MANDATORY FOR AGENTS — pypto-kernel-custom-skills workflow
# ------------------------------------------------------------
# **You must use this template** for **both**:
#   (1) **Per-module staged development** — each `custom/<op>/<op>_module1.py`, `…_module12.py`,
#       …, `…_module1…N.py` must follow the **same layers (A–L)** here (trim unused layers; keep
#       naming: `torch_golden_reference`, `pypto_function`, `_*_kernel_impl`, `your_op_kernel_npu`, …).
#   (2) **Full / production kernel** — the final integrated file must **not** abandon this structure
#       for ad-hoc layout; document any naming exceptions in `custom/plan/<op>.md`.
# Copy this file into `custom/<op>/` as a starting skeleton, or merge its sections into existing files
# without dropping the layer discipline. **Do not** claim a kernel is complete if the only remaining
# code path is `assemble`/glue with core `pypto` math removed — golden vs PyPTO checks must still pass.
#
# **`pypto_function` (Layer K) vs `pypto.loop` (Layers I–J)** — enforced by **skills/lead-orchestrator/references/rules.md** rule **18**,
# **skills/lead-orchestrator/references/rules.md** rule **22**, and **`skills/ci-and-layout-check/scripts/validate_custom_kernel_layout.py`** (AST scan of
# `custom/<op>/**/*.py`):
#   - **Forbidden in `pypto_function`:** `for ... in range(...)` (or any Python loop that drives
#     batch / sequence / chunk / tile computation that belongs on device).
#   - **Required:** express that iteration inside **`_your_op_kernel_impl`** and/or the body behind
#     **`your_op_kernel_npu`** using **`pypto.loop`** (+ `pypto.view`, etc.). Layer K only packs tensors,
#     allocates outputs, calls the JIT once (or as your API requires without host tile loops).
#
# See also: **skills/lead-orchestrator/references/rules.md** (template rule, rule 18), **pypto-kernel-design-format.md**, **skills/lead-orchestrator/references/rules.md** (rule 22),
# **pypto-kernel-design-format.md**.
#
# PURPOSE
# -------
# This file defines a **standard vertical structure** for **any** PyPTO custom
# kernel so that humans and coding agents can:
#   - Navigate by responsibility (reference math vs device vs host glue vs test).
#   - Keep reference steps aligned with `pypto_*` stages for easier debugging.
#   - Deliver a complete artifact: **torch_golden_reference** (primary oracle) + JIT + tests.
#
# GOLDEN FUNCTION — single primary entry (Layer F)
# ------------------------------------------------
# **`torch_golden_reference`** is the **main** numeric reference for precision tests.
# Not every kernel has both forward and backward: implement only what you verify.
# Examples:
#   - Forward inference kernel: golden returns the expected **forward output(s)**.
#   - Custom backward kernel: golden returns **gradients** (and may omit forward in the check).
#   - Fused forward+backward: golden returns **tuple of tensors** to compare with PyPTO.
# You may call `forward_ref` (Layer C) from inside the golden, or fold all PyTorch math
# into `torch_golden_reference` alone—Layer C is optional helper structure.
#
# SCOPE (not tied to one operator family)
# ---------------------------------------
# Works for elementwise ops, reductions, matmul variants, fused MLP blocks,
# attention-like or recurrent patterns, backward-only custom autograd, etc.
# **Pick only the layers you need** (see “Kernel modes” below).
#
# KERNEL MODES — which layers to implement (illustrative)
# -------------------------------------------------------
# | Mode               | Golden (F) typically returns              | Other layers      |
# |--------------------|-------------------------------------------|-------------------|
# | Forward-only       | Reference **y** (or tuple of outputs)    | C optional; H–K+L |
# | Backward-only      | Reference **grads** w.r.t. inputs        | E optional; H–K+L |
# | Forward + backward | **Outputs and/or grads** as you test     | C, E as needed    |
# | Inference-only     | Same as forward-only for numeric check   | Often no backward |
#
# HOW TO USE (agents)
# -------------------
# 1. Set OP_NAME and KERNEL_MODE (documentation intent; see block below).
# 2. Implement **`torch_golden_reference` (Layer F)** as the **main** oracle—what you
#    compare against `pypto_function` via `detailed_tensor_compare`.
# 3. Optionally add **Layer C** (`forward_ref`) as a building block the golden calls.
# 4. Optionally add **Layer E** private helpers if the golden grows large.
# 5. Implement **H–I** (`pypto_*` + `_*_kernel_impl`) on device; mirror the same math as F.
#    Put iterative/tensor-tile logic in **`pypto.loop`** here — **not** in `pypto_function` (Layer K).
# 6. Complete **J** (`@pypto.frontend.jit` signatures, runtime_options).
# 7. Complete **K** (`pypto_function`): **host-only** I/O (layouts, allocations, single JIT launch).
#    **Do not** add `for ... in range(...)` over dynamic batch/seq/tile here (CI will fail; RULES 18).
# 8. Run **L** (`main` / pytest): **golden → PyPTO → compare** (same order of tensors).
# 9. Record evidence (commands, rtol/atol, shapes) in your project plan / README.
#
# DEPENDENCIES
# ------------
#   torch, pypto, typing (add numpy, pytest, etc. only if your tests need them)
#
# ---------------------------------------------------------------------------
# Complex-kernel bundle (repo): `.agents/`
# ---------------------------------------------------------------------------
# This file **lives in** that folder and is the **mandatory code skeleton** for staged + full kernels.
# When developing **custom/** multi-stage fused kernels, treat this folder as the
# source of truth alongside this template. Read at least:
#   - **skills/lead-orchestrator/references/rules.md** — zero tolerance, staged `*_module*.py` chain, plan updates,
#     `skills/ci-and-layout-check/run_validate_layout.sh`, `set_vec_tile_shapes` (≥4 args), `view` ≤4D, **rule 18** (`pypto_function`
#     must not use `for ... in range(...)` for kernel iteration).
#   - **skills/** — sub-skills: module-at-a-time, golden vs PyPTO boundaries,
#     prohibited patterns (one-shot JIT, host `for` as tile loop in the **wrong** layer, one-JIT-per-module).
#   - **skills/debugging/DEBUG.md** — §1–§8: opaque `Errcode` / `FFFFF` / `UNKNOWN`: do not stop without playbook.
#     **§9**: agent-learned dev patterns — consult BEFORE writing PyPTO code (RULES 7b).
#   - **skills/ci-and-layout-check/CI.md** — layout validator command (no NPU).
# Tools (see README.md “Available Tools”):
#   - `python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py <file.py>`
#   - `bash .agents/skills/ci-and-layout-check/run_validate_layout.sh` (repo root) — also runs an AST
#     check: **`for ... in range(...)` inside `def pypto_function`** under `custom/<op>/` → **exit 1**
#     (fix by moving loops to **`pypto.loop`** in Layer I; see Layer K comments).
#   - Validation runner: `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<op>/test_<op>.py`
# Staged files under **custom/<op>/** use the bundled **detailed_tensor_compare** via that PYTHONPATH
# (same semantics as Layer A below — do not swap for `torch.allclose` alone as primary proof).
# ---------------------------------------------------------------------------
#
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# PyPTO is required for Layers H–J. Import early so missing env fails fast during dev.
import pypto


# -----------------------------------------------------------------------------
# Global placeholders — replace for your operator
# -----------------------------------------------------------------------------

# Short identifier for logs and error messages (e.g. "my_softmax", "fused_add_rms").
OP_NAME: str = "YOUR_OP_NAME"

# Set True after you have a runnable end-to-end path on NPU (or sim if allowed).
KERNEL_READY: bool = False

# Document intent only (no runtime behavior). Agents: align golden + kernel with this.
#   "forward_only"   — `torch_golden_reference` implements forward oracle; no grad check.
#   "backward_only"  — golden implements gradient oracle (PyPTO backward kernel).
#   "forward_backward" — golden returns both output and grad checks as needed.
KERNEL_MODE: str = "forward_only"


# =============================================================================
# Layer A — Utilities (optional but recommended for delivery)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Shared diagnostics used by tests and manual debugging. Keep them deterministic
#   and independent of PyPTO so CPU-side checks stay simple.
#
# Typical contents:
#   - detailed per-tensor comparison reports
#   - optional: save/load intermediates for bisection (see project skills)
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **Mandatory comparison helper** for golden vs PyPTO in the complex-kernel workflow:
#   import from `.agents/skills/validation-and-deliverables/detailed_tensor_compare.py` when using
#   `custom/<op>/test_<op>.py` (see **skills/validation-and-deliverables/SKILL.md**).
# - Do **not** replace primary validation with `torch.allclose` alone (RULES / skill: use bundled report).
# - Compare **every leaf output** (tuple/list/dict/nested) — not only the first tensor (RULES 10, skill 16).
# =============================================================================


def detailed_tensor_compare(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tensor_name: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True,
    max_outliers_display: int = 20,
) -> Dict[str, Any]:
    """
    Detailed tensor comparison: tolerance stats, outlier ratio, top outliers.

    Use this in Layer L to compare:
      - **PyPTO output** (or left arg), vs
      - **`torch_golden_reference` output** (or right arg), same layout.

    Works for any comparable pair: forward activations, gradients, auxiliary tensors, etc.

    Args:
        tensor1: First tensor (e.g. PyPTO output).
        tensor2: Second tensor (e.g. reference). Must match shape.
        tensor_name: Label for logs.
        rtol, atol: Same semantics as torch.allclose.
        verbose: Print human-readable report.
        max_outliers_display: Cap on printed outlier rows.

    Returns:
        dict with keys including:
          all_close (bool), out_of_tolerance_count, max_diff, diff_tensor, ...
    """
    t1, t2 = tensor1.detach().cpu().float(), tensor2.detach().cpu().float()
    if t1.shape != t2.shape:
        raise ValueError(
            f"[{tensor_name}] shape mismatch: {tuple(t1.shape)} vs {tuple(t2.shape)}"
        )

    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)

    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    total_elements = t1.numel()
    out_of_tolerance_count = int(out_of_tolerance_mask.sum().item())
    out_of_tolerance_ratio = out_of_tolerance_count / max(total_elements, 1)

    max_diff = float(torch.max(diff).item())
    mean_diff = float(torch.mean(diff).item())
    std_diff = float(torch.std(diff).item())

    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = float(torch.max(out_of_tolerance_diff).item())
        mean_out_diff = float(torch.mean(out_of_tolerance_diff).item())

        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]
    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None

    result: Dict[str, Any] = {
        "total_elements": total_elements,
        "out_of_tolerance_count": out_of_tolerance_count,
        "out_of_tolerance_ratio": out_of_tolerance_ratio,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_out_of_tolerance_diff": max_out_diff,
        "mean_out_of_tolerance_diff": mean_out_diff,
        "all_close": out_of_tolerance_count == 0,
        "tolerance_mask": tolerance_mask,
        "diff_tensor": diff,
        "outlier_indices": sorted_outlier_indices,
        "outlier_values1": sorted_outlier_values1,
        "outlier_values2": sorted_outlier_values2,
        "outlier_diffs": sorted_outlier_diffs,
        "outlier_relative_diffs": sorted_outlier_relative_diffs,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Tensor detailed comparison report")
        print(f"name: {tensor_name}")
        print("=" * 60)
        print(f"total elements: {total_elements:,}")
        print(f"out of tolerance: {out_of_tolerance_count:,}")
        print(f"out ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio * 100:.4f}%)")
        print(f"max |diff|: {max_diff:.6f}")
        print(f"mean |diff|: {mean_diff:.6f}")
        print(f"std(diff): {std_diff:.6f}")
        print(f"rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"max |diff| (only outliers): {max_out_diff:.6f}")
            print(f"mean |diff| (only outliers): {mean_out_diff:.6f}")
            print(
                f"\nTop outliers (showing up to {min(max_outliers_display, out_of_tolerance_count)}):"
            )
            print("-" * 80)
            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                assert sorted_outlier_indices is not None
                idx_str = str(
                    tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices)))
                )
                print(
                    f"{idx_str:<24} {sorted_outlier_values1[i].item():<14.6f} "
                    f"{sorted_outlier_values2[i].item():<14.6f} "
                    f"{sorted_outlier_diffs[i].item():<12.6f} "
                    f"{sorted_outlier_relative_diffs[i].item():<12.6f}"
                )

        print(f"\nall_close: {result['all_close']}")
        print("=" * 60)

    return result


# =============================================================================
# Layer B — Small math building blocks (pure PyTorch)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Reusable fragments of the algorithm: elementwise pieces, reductions, blocks
#   that you may call from forward_ref and mirror in PyPTO sub-kernels (Layer H).
#
# Naming:
#   Use problem-specific names (e.g. `rms_norm_fwd`, `window_sum`, `gelu_approx`).
#   Delete this stub if your reference is short enough to live only in Layer C.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - Keep math **normalizable** and aligned with what you will port to PyPTO (skill: normalized golden).
# - Do not use host Python `for` here to stand in for **kernel tile loops** — that belongs in
#   `pypto.loop` inside Layer I (skill **Prohibition B**).
# =============================================================================


def math_helper_placeholder(x: torch.Tensor) -> torch.Tensor:
    """
    Replace with real helpers for YOUR_OP, or remove Layer B entirely.

    Keep ops aligned with what you can lower on NPU (see Layer C constraints).
    """
    raise NotImplementedError(
        f"[{OP_NAME}] Agent: implement Layer B helpers or delete math_helper_placeholder."
    )


# =============================================================================
# Layer C — Reference implementation (optional helper; ground truth building block)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   **Optional** PyTorch reference used to build **`torch_golden_reference` (Layer F)**.
#   Some teams implement all oracle math only in `torch_golden_reference` and delete Layer C.
#
# Typical contract (customize freely):
#   Inputs:  logical tensors in user layout (any rank your spec allows).
#   Outputs: primary tensor(s) + optional **cache** dict for training / PyPTO bridge.
#
# Relation to Layer F:
#   `forward_ref` is **not** the primary test hook—**`torch_golden_reference` is**. Call
#   `forward_ref` from the golden when sharing forward math helps readability.
#
# CONSTRAINTS (example — customize per kernel)
#   Allowed:    matmul, elementwise, sum/reduction over named dims, Python loops.
#   Disallowed:  ops that you cannot match on NPU (list them explicitly).
#
# When a PyTorch op is “forbidden”, replace with equivalent math (e.g. prefix sum
# via a fixed matrix multiply) and reuse the same constants in Layer D.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **Golden frozen:** do not change reference behavior without evidence + plan log (RULES Non-negotiable 5,
#   skill Non-Negotiable 6). Primary test oracle remains **`torch_golden_reference`** (Layer F).
# - **Shape comments** on tensor lines in real kernel code are required by the workflow (RULES 6 / skill).
# - Optional here: staged work often keeps a shared golden in **`custom/<op>/`** per **skills/plan-template/plan.template.md**.
# =============================================================================


def forward_ref(
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Optional reference forward for YOUR_OP (edit signature and returns as needed).
    Primary precision baseline is **`torch_golden_reference`** (Layer F); implement this
    only when splitting forward math out of the golden helps clarity or reuse.

    Returns:
        out:       primary output (float32 often improves comparison stability).
        aux_out:   optional second output (state, mask logits, etc.); use None if N/A.
        cache:     dict of tensors for device path; use {} if N/A.

    Agent: document outputs/cache in the Appendix if you use this layer.
    """
    raise NotImplementedError(
        f"[{OP_NAME}] Agent: implement forward_ref (Layer C) if used, or delete and keep golden-only in F."
    )


# =============================================================================
# Layer D — Host-side constants (optional)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Precompute tensors shared across reference and PyPTO: masks, one-hot grids,
#   prefix/suffix operators, cos/sin tables, **or** chunk-local BT×BT structures.
#
# Omit Layer D entirely if your kernel only needs inputs + weights (e.g. simple GEMM).
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - Same constants must feed **both** golden path and PyPTO path so boundaries stay comparable.
# - Document any constant tensor shapes in **`custom/plan/<op>.md`** when using the full workflow.
# =============================================================================


def make_host_constants(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """
    Build static host tensors your reference and JIT entry both consume.

    Signature is **kernel-specific** (add dimensions like tile size, seq length, …).

    Agent: return a tuple in a fixed order; document that order next to `pypto_function`.
    """
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement make_host_constants (Layer D) or delete if unused.")


# =============================================================================
# Layer E — Reference helpers for complex passes (private, one stage per function)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   For long backward (or multi-phase forward) references, split into `_stage_*`
#   helpers that map 1:1 to `pypto_*` blocks in Layer H. Use the same stage labels
#   (A)(B)(C) in comments for both sides.
#
# Omit this layer if the golden in F is short, or if you only verify forward.
#
# Example generic stages (rename to your problem):
#   _load_tile_views, _apply_operator, _reduce_partial, _scatter_result
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - Map each stage label **(A)(B)(C)** to a **semantic module** in the plan (**skills/phase2-phase3-construction/SKILL.md**:
#   Module decomposition — split by meaning/checkpointability, not “balanced complexity”).
# - One **active** PyPTO module at a time: later stages stubbed with **golden-fed** tensors until the
#   current module passes `detailed_tensor_compare` (**skills/lead-orchestrator/references/rules.md**).
# - Comment stubs explicitly: `# STUB: until Mk verified; golden-fed tensor`.
# =============================================================================


def _reference_stage_placeholder() -> None:
    """Replace with real `_`-prefixed helpers or delete Layer E."""
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement Layer E helpers or remove.")


# =============================================================================
# Layer F — PRIMARY golden: `torch_golden_reference` (required for standard precision flow)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   **`torch_golden_reference` is the main numeric oracle** used with `detailed_tensor_compare`
#   against `pypto_function` outputs. Implement **only** the tensors your kernel API exposes
#   for verification (forward outputs, backward grads, or both—not every kernel needs both).
#
# Contract:
#   - Return type is **kernel-specific**: one `torch.Tensor` or a `Tuple` of tensors.
#   - Document the **fixed order** of tuple elements in Appendix tables and in Layer L.
#   - May call `forward_ref` (Layer C), `torch.autograd.grad`, hand-written formulas, etc.
#
# Not every kernel implements forward **and** backward: one golden can still cover the
# project (e.g. forward-only golden returns `y_ref`; backward-only golden returns `grads`).
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - This is the **trusted golden** for the workflow: finalize it **before** unlocking real PyPTO
#   logic for the next semantic module (RULES: golden frozen; skill: do not change without evidence).
# - **Zero skipping:** do not omit reading **skills/lead-orchestrator/references/rules.md** / sub-skill sections that apply (skills/lead-orchestrator/references/rules.md).
# - Default user intent: “implement the kernel” ⇒ **next unverified module only** unless user asks for
#   full integration after listing verified modules (RULES User prompt default).
# - Every boundary check: record **`detailed_tensor_compare`** results in **`custom/plan/<op>.md`**
#   Per-module verification log (RULES Plan file / skill).
# =============================================================================


def torch_golden_reference(
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    **Primary golden function** for this kernel: PyTorch (or numpy) reference values that
    you compare against the PyPTO path in Layer L.

    What to return depends on KERNEL_MODE and what you ship:
      - Forward-only: typically the reference **forward output(s)**.
      - Backward-only: typically **gradient tensor(s)** w.r.t. inputs.
      - Mixed: a tuple combining outputs and/or grads—document the order explicitly.

    This is the **single** canonical name for the golden in this template; avoid scattering
    multiple unrelated `*_golden_*` entry points unless your project policy requires it.

    Agent: implement for every tensor you intend to check numerically; omit sub-results you
    do not compare (no need to mirror every internal PyPTO temporary).
    """
    raise NotImplementedError(
        f"[{OP_NAME}] Agent: implement torch_golden_reference (Layer F) — primary golden for precision tests."
    )


# =============================================================================
# Layer G — Cache / layout bridge (optional)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Convert reference cache (nested lists, per-tile tensors, Python scalars) into
#   the flat/multi-buffer layout the JIT entry expects. Often inlined into K instead.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - Layout rules for **`custom/<op>/`** are checked by **`skills/ci-and-layout-check/run_validate_layout.sh`** (skills/ci-and-layout-check/CI.md) — run after
#   meaningful edits under `custom/`, fix exit 1 before claiming layout complete (RULES 7, 15).
# - Flattening order (batch/head/seq) must match what **`pypto.view`** uses in Layer I.
# =============================================================================


def prepare_buffers_for_pypto(
    cache: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> List[Optional[torch.Tensor]]:
    """
    Normalize cache entries for the device path.

    Agent: return list/tuple matching kernel input order, or fold logic into `pypto_function`.
    """
    raise NotImplementedError(
        f"[{OP_NAME}] Agent: implement prepare_buffers_for_pypto (Layer G) or inline in pypto_function."
    )


# =============================================================================
# Layer H — PyPTO sub-kernels (small named regions)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Each function does **one** conceptual step:
#     - slice views with `pypto.view`
#     - matmul / elementwise blocks
#     - optional: `pypto.set_vec_tile_shapes`, `pypto.set_pass_options`, etc.
#
# Simple kernels may use a single `pypto_*` or inline into Layer I.
#
# Rules:
#   - Return all tensors the next step needs (avoid hidden globals).
#   - Names should mirror reference helpers when applicable.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **`set_vec_tile_shapes`:** pass positive tile dimensions as required by `docs/api/config/pypto-set_vec_tile_shapes.md` (RULES 16, skill 20 / 5.4b).
# - **`pypto.view` / internal tensors:** keep ranks **≤ 4D** where required (same refs).
# - Add **shape comments** on tensor lines in real code (RULES 6).
# - If stuck: **skills/debugging/DEBUG.md** → `extract_pypto_calls.py` → op-by-op protocol in **skills/debugging/SKILL.md** (RULES 7, 13).
# - Before writing PyPTO code: **skills/debugging/DEBUG.md §9** (RULES 7b) — JIT §9.1, view §9.4, matmul §9.19, etc.
# =============================================================================


def pypto_stage_placeholder() -> None:
    """Replace with real `pypto_*` functions or merge into Layer I."""
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement pypto_* stages (Layer H).")


# =============================================================================
# Layer I — Kernel implementation (`pypto.loop` nest, no @jit here)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Device-side recipe: `pypto.loop` over batch/tiles/dims as needed, calls to
#   Layer H, writes to output buffers provided by Layer J.
#
# Not every kernel needs explicit loops (some are single-shot matmul + epilogue).
#
# Why split from Layer J:
#   - Reuse / test without re-running JIT shell.
#   - Keep J thin (signatures + options only).
#
# Document:
#   Any flattening / indexing convention (row-major order, packed heads, …).
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **Algorithmic** iteration belongs in **`pypto.loop`** (and views) **here (Layer I) or behind Layer J** —
#   **not** inside **`pypto_function`** (Layer K). Host **`for ... in range(...)`** in `pypto_function`
#   is **forbidden** for that purpose (RULES **18**, skill **22**; enforced by **`validate_custom_kernel_layout.py`**).
# - **Algorithmic** iteration also does **not** belong in a Python `for` on the host that fakes per-tile
#   execution anywhere (skill **Prohibition B**; RULES Non-negotiable 3).
# - **Progressive integration:** implement **one semantic module** at a time; stub later stages with
#   golden boundary tensors until the active module matches golden (**skills/lead-orchestrator/references/rules.md**).
# - **Forbidden:** wiring all `pypto_*` helpers into one `kernel_impl` in a single edit before any
#   intermediate boundary matches golden (skill Module-at-a-time).
# - Opaque errors (`Errcode`, `FFFFF`, …): follow **skills/debugging/DEBUG.md** — do not abandon without playbook (RULES 13).
# =============================================================================


def _your_op_kernel_impl(
    x_in: pypto.Tensor,
    y_out: pypto.Tensor,
) -> None:
    """
    Full PyPTO implementation (no @jit on this function).

    Parameters must mirror `your_op_kernel_npu` (Layer J) one-to-one.

    Use **`pypto.loop`** for batch/sequence/tile iteration required by the algorithm — not Python
    `for ... in range(...)` in `pypto_function` (Layer K). See module header and RULES 18.

    Agent: implement.
    """
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement _your_op_kernel_impl (Layer I).")


# =============================================================================
# Layer J — JIT entry (`@pypto.frontend.jit`)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   - Declare tensor types: dynamic vs static dimensions, dtypes.
#   - Set `runtime_options` (memory, stitch, run_mode NPU/SIM per project policy).
#   - Set `debug_options` during bring-up.
#   - Body: delegate to `_your_op_kernel_impl` only.

# Notes:
#   - Prefer pre-allocated output tensors as arguments.
#   - Host reshape stays in Layer K.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - Aim for **one production `@pypto.frontend.jit`** for the final shipped kernel (RULES Non-negotiable 4;
#   skill **Prohibition C**). Semantic modules are **helpers inside the graph**, not separate production
#   JIT entry points unless an explicit **documented fallback** (toolchain limits).
# - Staged milestone files under **`custom/<op>/`** may each contain **golden + one `@jit`** for that
#   **cumulative** scope — still follow **`…_module1.py` → `…_module12.py` → … → `…_module1…N.py`**
#   naming; do not advance until **all** outputs pass `detailed_tensor_compare` (RULES 14, skill 17).
# =============================================================================


@pypto.frontend.jit(
    runtime_options={
        # Example keys — adjust per official PyPTO docs / your stack version.
        "stitch_function_inner_memory": 128 * 16,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 16,
        "device_sched_mode": 1,
        "run_mode": pypto.RunMode.NPU,
    },
    debug_options={"runtime_debug_mode": 1},
)
def your_op_kernel_npu(
    x_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    y_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
) -> None:
    """
    JIT-compiled entry (rename to match your operator).

    Replace parameters with your real buffers; extend the list as needed.

    Agent: use explicit `pypto.Tensor([...], pypto.DT_*)` for each I/O; do not use *args.
    """
    _your_op_kernel_impl(x_in, y_out)


# =============================================================================
# Layer K — Host wrapper (`pypto_function`, `launch_*`, …)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Role:
#   Pre-JIT orchestration on the torch side **only**:
#     1. Device / dtype placement.
#     2. Flatten, transpose, pad to the layout Layer J expects.
#     3. Allocate outputs (`torch.empty`, …).
#     4. Call `your_op_kernel_npu(...)` (typically **once** per public API call — no host loop over tiles).
#     5. Reshape outputs to the user-facing API.
#
# This is the usual **Python entry point** for integration tests and product code.
#
# **FORBIDDEN (RULES 18, skill 22, CI AST check):** `for ... in range(...)` **inside this function**
# when it drives **kernel** work (batch, sequence, chunks, tiles). That logic must live under
# **`pypto.loop`** in **`_your_op_kernel_impl`** / the JIT graph (`your_op_kernel_npu`), not here.
# **`validate_custom_kernel_layout.py`** fails if it detects `for ... in range(` in `def pypto_function`.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **Allowed** Python loops here: only **incidental** uses (e.g. iterating a small fixed list of output
#   names, zipping pre-built tensors) — **not** dynamic `range(B)`, `range(S)`, `range(NT)`, etc. for
#   compute orchestration (use **`pypto.loop`** in Layer I).
# - Keep I/O reshape and allocation here; JIT entry (Layer J) should stay thin.
# =============================================================================


def pypto_function(
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Pack inputs, invoke `your_op_kernel_npu`, unpack outputs.

    Do **not** implement device-side iteration with **`for ... in range(...)`** here — use
    **`pypto.loop`** in `_your_op_kernel_impl` (see module header, RULES 18, skill Non-negotiable 22).

    Agent: match return structure to Layer L tests and to public API docs.
    """
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement pypto_function (Layer K).")


# =============================================================================
# Layer L — Test driver (`main`, pytest, or CI script)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (Role, bundle reminders, workflow constraints).
# =============================================================================
#
# Suggested flow (trim steps you do not need):
#
#   1) Config: seed, device, dtype, shapes, run_mode.
#   2) Inputs: random or fixture tensors.
#   3) Constants: `make_host_constants` if Layer D exists.
#   4) **Golden:** `ref = torch_golden_reference(...)`  ← primary oracle (Layer F).
#   5) **PyPTO:** `pto = pypto_function(...)` (same logical inputs / upstream grads as golden).
#   6) **Compare:** `detailed_tensor_compare(pto_i, ref_i, name)` for each output slot.
#
# Optional: call `forward_ref` inside the golden or separately for debugging—tests should
# still treat **`torch_golden_reference` as the baseline** for pass/fail unless you
# document otherwise.
#
# Do not claim pass without running the intended environment when policy requires NPU.
#
# --- pypto-kernel-custom-skills (bundle) — reminders for this layer ---
# - **Default E2E runner** for `custom/<op>/` is **`test_<op>.py`** with:
#     `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<op>/test_<op>.py`
#   from repo root — **not** `pytest` as the primary correctness driver (skill Non-Negotiable 11).
# - After edits under **`custom/`**, run **`bash .agents/skills/ci-and-layout-check/run_validate_layout.sh`**
#   (skills/ci-and-layout-check/CI.md; RULES 7, 15) — includes **`pypto_function` + `for ... in range`** detection (RULES 18).
# - **Claim success only with real runs** — paste/compare evidence into **`custom/plan/<op>.md`**
#   (Development & debug log, Per-module verification log); no “should pass” (RULES).
# - Same failure **three** times ⇒ **change strategy** (skill Non-Negotiable 8).
# =============================================================================


def main() -> None:
    """
    End-to-end test harness — **`torch_golden_reference` vs `pypto_function`** is the core loop.

    Typical pattern (single forward output):
      ref_out = torch_golden_reference(inputs, ...)
      pto_out = pypto_function(inputs, ...)
      detailed_tensor_compare(pto_out, ref_out, "out")

    Typical pattern (tuple of tensors, e.g. grads or multi-output):
      ref_pack = torch_golden_reference(...)
      pto_pack = pypto_function(...)
      for name, a, b in zip(names, pto_pack, ref_pack):
          detailed_tensor_compare(a, b, name)

    Optional: use `forward_ref` only as a helper inside `torch_golden_reference`.
    """
    torch.manual_seed(0)

    # --- 1) Config (device, dtype, shapes, seeds) ---
    # device = "npu:0"
    # dtype = torch.bfloat16

    if not KERNEL_READY:
        raise NotImplementedError(
            f"[{OP_NAME}] Set KERNEL_READY=True after implementation is ready, "
            f"or remove this guard during incremental testing."
        )

    # --- 2) Build inputs ---
    # x = ...

    # --- 3) Host constants (optional) ---
    # consts = make_host_constants(torch.device(device), torch.float32, ...)

    # --- 4) Primary golden (Layer F) — baseline for all comparisons ---
    # with torch.no_grad():
    #     ref_out = torch_golden_reference(x, ...)

    # --- 5) Optional: forward_ref only if you use it inside golden or for debug ---
    # out_dbg, aux_dbg, cache = forward_ref(...)

    # --- 6) PyPTO kernel path ---
    # with torch.no_grad():
    #     pto_out = pypto_function(x, ...)

    # --- 7) Precision: golden vs PyPTO (repeat per tensor if tuple) ---
    # detailed_tensor_compare(pto_out, ref_out, "out")

    print(
        f"[{OP_NAME}] main() template: wire torch_golden_reference → pypto_function → "
        f"detailed_tensor_compare (KERNEL_MODE={KERNEL_MODE!r})."
    )


if __name__ == "__main__":
    main()


# =============================================================================
# Appendix — Contract tables (fill in for your kernel; keep near README/plan)
# **LOCKED — no edits, no deletions:** Do not edit or remove the following comment block (contract tables template, bundle notes).
# =============================================================================
#
# Public API / tensor contract (example — replace rows):
#
# | Name   | Shape (symbolic) | dtype    | Role / compared in golden      |
# |--------|------------------|----------|--------------------------------|
# | x      | [B, N, D]        | bf16     | input                          |
# | weight | [D, D]           | fp32     | parameter                      |
# | y      | [B, N, D]        | fp32     | `torch_golden_reference` out   |
#
# Cache keys (if any):
#
# | Key    | Needed for bwd / device | Notes        |
# |--------|-------------------------|--------------|
# | scale  | optional                | fusion param |
#
# Reference ↔ PyPTO mapping (for debugging):
#
# | Ref helper           | PyPTO function      |
# |----------------------|---------------------|
# | _ref_stage_alpha     | pypto_stage_alpha   |
#
# --- pypto-kernel-custom-skills (bundle) — plan file (outside this single-file template) ---
# For work under **`custom/<op>/`**, maintain **`custom/plan/<operator_name>.md`** from
# **`skills/plan-template/plan.template.md`**: `active_module`, `current_staged_file`, **Staged module files** table,
# **Module decomposition** (how + why), **Per-module verification log** with `detailed_tensor_compare`
# evidence, `next_mandatory_step`, **blockers** if stuck (RULES “Plan file (every turn)”, skill).
#
# =============================================================================
