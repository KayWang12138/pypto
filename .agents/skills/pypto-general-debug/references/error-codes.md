# Opaque error / error-code playbook (DEBUG_GUIDEBOOK.md §1–§7)

**Audience:** Agents working on complex PyPTO kernels under this bundle.

**Do not skip:** This file is part of the mandatory path when stuck — see the Debug Agent contract (`.opencode/agents/pypto-op-debugger.md`).

**Policy:** **`Errcode: FFFFF!`**, **`UNKNOWN`**, **`0x3FFFF`**, or other opaque / scary-looking error codes are **not** a reason to end the session, report failure, or "give up" until the Debug Agent's Hard rules (`.opencode/agents/pypto-op-debugger.md`, "If after 3 fix/re-verify cycles the module still fails, stop and report the blocker") actually apply. **Token and turn budget are not a constraint** for this workflow: keep iterating, gathering evidence, and trying the next structured strategy until the problem is resolved or a true blocker is proven.

**What "give up" looks like (forbidden here):** stopping after only printing the error code, declaring the task impossible without consulting docs, skipping logs, or refusing to try another angle.

---

## 1. First actions (always)

1. **Capture the full message** — stderr, Python traceback, and any **`pypto-log*.log`** / device log lines. Search for `Errcode`, `ErrCode`, `F` + digits, `aicore`.
2. **Route the error code** — read `docs/trouble_shooting/README.md` and open the component doc for that prefix (e.g. FUNCTION / `docs/trouble_shooting/function.md` for many `F2xxxx`-style codes). For **`F21004`** / **`REGISTER_COPY`** / invalid vector tile, see **§4** below first.
3. **Append to `custom/<op>/MEMORY.md` → Development & debug log** — what failed, command, hypothesis, next step. No empty "stopped" endings.

---

## 2. UNKNOWN / FFFFF / insufficient detail / silent failure

This section covers **three related failure modes**:
- An **opaque error code** (`FFFFF`, `UNKNOWN`, `0x3FFFF`) with no narrative.
- A run that **looks successful in Python** but produces wrong / zero outputs and emits **no Python-level error message** at all.
- A run that dies early on the NPU with only a terse line like `Inner Error, please contact Huawei Engineer`.

In all three cases, the **Python stderr is insufficient**. The real error is in the device slog. You must pull it into stdout, capture to a file, and search it efficiently.

### 2a. Force the device log into stdout and capture it

Set the two ASCEND env vars and re-run with stdout redirected to a log file:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0       # 0 = DEBUG, 1 = INFO, 2 = WARN, 3 = ERROR, 4 = NULL
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # route slog to stdout instead of on-disk slog files
python <failing_file>.py >> result.log 2>&1
```

Notes:
- **Use `>>` (append) rather than `>`** so repeated runs accumulate evidence you can diff later.
- `2>&1` is required — some ASCEND layers emit to stderr even with `PRINT_TO_STDOUT=1`.
- `ASCEND_GLOBAL_LOG_LEVEL=0` is verbose; expect `result.log` to commonly exceed **10 MB** on any non-trivial kernel (single op, L1 shape, full NT loop).
- Run from the NPU server over `Run <file> on npu:<N>`; the env vars apply to that process.

### 2b. Search `result.log` efficiently with ast-grep

Straight `grep -n ERROR result.log` gets you the list of ERROR-lines, but **loses the structural context** around each one (the preceding `[TRACE]` breadcrumbs, the CCE file path, the stack frame). And scrolling 10 MB+ of slog by hand is a dead end.

The **ast-grep agent-skill** (`https://github.com/ast-grep/agent-skill/tree/main/ast-grep/skills/ast-grep`) gives you **syntax-aware** and **context-preserving** pattern search over large structured logs at high throughput. Install it per that repo's README before starting. It is the preferred search tool when `result.log` exceeds a few MB or when the failing pattern spans multiple lines (e.g. an `ERROR:` line followed by 10 lines of `[INFO]` device context that is the actual signal).

**Recommended first pass** (run in this order — each narrows the noise further):

1. **All ERROR-class lines with 5 lines of trailing context** (the trailing lines usually carry the device stack frame the ERROR line references):

   ```bash
   ast-grep run --pattern 'ERROR' --context-after 5 result.log
   ```

   Fall back to `grep -nA 5 ERROR result.log` only if `ast-grep` is not installed.

2. **Filter to ERROR lines that carry a code** (`F2xxxx`, `E1xxxx`, `0xXXXXX`, `Errcode`, `ErrCode`):

   ```bash
   ast-grep run --pattern 'ERROR $CODE' --regex 'F[0-9]{5}|E[0-9]{5}|0x[0-9A-Fa-f]+|Errcode|ErrCode' result.log
   ```

3. **Pull matching CCE file paths** (when the error references generated device code):

   ```bash
   ast-grep run --pattern '$PATH.cce' --regex '/.*\.cce' result.log
   ```

4. **Narrow to the last ERROR block only** (if the run died early, the last error before the Python traceback is usually the real cause):

   ```bash
   tac result.log | ast-grep run --pattern 'ERROR' --context-before 5 --context-after 20 --max-count 1
   ```

If the op uses `pypto.loop` and you suspect an iteration-dependent failure, also search for the loop-index breadcrumbs the framework emits: `loop iter=`, `chunk=`, `NT step`. These are in the same log but often **below** the ERROR line, so `--context-after 20` on step 1 captures them.

### 2c. Classify what you found

Once the relevant ERROR block is isolated (usually 20–50 lines, not 10 MB):

- If the code starts with `F2xxxx`: route via `docs/trouble_shooting/function.md` → see **§4** if it is `F21004`.
- If the code starts with `F4` / `F5`: **pass / compile error** — see §3a.
- If the code is an AICore code / CCE file path appears: load **`pypto-aicore-error-locator`**.
- If the log shows `rtMalloc failed` / `Out of memory`: load **`pypto-machine-workspace`**.
- If the log shows a host stack trace (`Segmentation fault`, `_ZN…` mangled frames): load **`pypto-host-stacktrace-analyzer`**.
- If the run **completed cleanly** but outputs are wrong and **no ERROR line appears even at `LOG_LEVEL=0`**: this is a **true silent precision / correctness failure**. Proceed to §3 (narrow blast radius) and §9 lookup.

### 2d. Computation-graph / program dump

- If the graph is suspect: follow the **computation-graph / program-dump** guidance linked from troubleshooting docs (upstream docs may label this "view computation graph").
- **Do not** treat "unknown" as terminal; treat it as **need more signal** (logs, smaller repro, earlier checkpoint).

### 2e. Append findings to the memory

Every ASCEND-log search that produced a useful signal (ERROR code, CCE file, loop-index breadcrumb) is logged to `custom/<op>/MEMORY.md` → Development & debug log with:
- the exact `ast-grep` command used,
- the matched block (not the full 10 MB log — include only the narrowed result),
- the classification it led to.

This preserves the search trail for future bisection and for the verification handoff.

### 2f. `fp32_unstable` — CPU FP32 reproducer protocol (distinguish kernel-bug from math-bug)

When pypto-op-verifier reports `divergence_fingerprint: "algorithmic"` (i.e. `torch` mode also fails), the failure is either:
- **(A) Kernel is genuinely wrong** — the algorithm is misimplemented. `failure_category: precision` is correct, load `pypto-precision-debug`.
- **(B) FP32 arithmetic is fundamentally unstable** for this op's expression. `failure_category: fp32_unstable` is correct, and the right response is Safeguard B (return to architecture) per Hard rules.

You MUST run the CPU FP32 reproducer below **before** deciding between (A) and (B). This takes ≤ 15 minutes and prevents the infinite tile-tweaking loop.

**Procedure:**

1. Write `custom/<op>/_debug/fp32_stability_check.py` — a pure-CPU script (no NPU, no pypto import needed). It:
   - Reads the failing case from `custom/<op>/eval/adversarial_suite.json` (seed, shape, scale knobs).
   - Rebuilds inputs identically to `test_inputs.py::make_inputs(case)` (import `make_inputs` directly).
   - Runs the golden-side expression for the failing output **twice on the same inputs**: once in FP32 (default), once in FP64 (wrap each input with `.double()` before the compute, cast result back with `.float()` for comparison).
   - Prints `max_abs_diff` and `max_rel_diff` between FP32-golden and FP64-golden-cast-back-to-FP32.

2. Interpret the result:

   | FP32 vs FP64 diff | Interpretation | Verdict to return |
   |---|---|---|
   | ≤ pypto-op-verifier's `atol`/`rtol` | Golden is numerically stable in FP32. The kernel is genuinely wrong — case **(A)**. | `failure_category: precision`, `divergence_fingerprint: "algorithmic"`. Load `pypto-precision-debug`. |
   | > pypto-op-verifier's `atol`/`rtol`, comparable to NPU's failing diff | Golden itself drifts in FP32 — the math is fundamentally unstable — case **(B)**. | `failure_category: fp32_unstable`, `divergence_fingerprint: "algorithmic"`. DO NOT load a sub-skill. Return "RETURN TO architecture — algebraic reformulation required" per Safeguard B. |
   | FP32-FP64 diff is tiny but NPU diff is large | The kernel diverges from the (stable) golden. Case **(A)**, but the NPU-side flavour is likely tile/pipe/memory rather than pure algorithm. Reconsider `divergence_fingerprint` — it should be `kernel_ok_npu_only`, not `algorithmic`. Escalate to verification to re-run the 3-way dispatcher. |

3. Attach `fp32_stability_check.py` + its log to the memory entry. Architecture, when invoked under Safeguard B, reads this as the primary evidence for the reformulation decision.

**Common `fp32_unstable` patterns to cite when returning to architecture:**
- `A + B − C` where `|A + B| ≈ |C|` (gated delta rule backward's `d_s_new`, RetNet bwd `dq`, linear attention bwd dK row sums).
- `logsumexp_left − logsumexp_right` without a shared max shift (unstable softmax backward variants).
- `new_state − old_state * decay` when `decay ≈ 1` and the two terms are nearly equal.
- Chained matmul accumulations where the matmul is fp32-accumulate but the downstream sub runs on the matmul's fp32 result without fp64 promotion.

Suggesting one of these patterns to architecture with the failing case's numerical fingerprint is MUCH more useful than simply saying "precision fails" — it directs the reformulation to the exact expression.

---

## 3. Kernel-specific: narrow the blast radius

1. Confirm which **staged set** is current (e.g. `staged/<op>_module12_impl.py` + `_golden.py` + `test_*.py`; see `skills/pypto-decompose-construct/SKILL.md` → Staged sets). Run `extract_pypto_calls.py` on the staged `*_impl.py` (or, after Phase D, on the canonical `custom/<op>/<op>_impl.py`).
2. Run **`python3 .agents/skills/pypto-kernel-layout-check/scripts/extract_pypto_calls.py <kernel.py>`** and follow the **op-by-op check protocol** in `.agents/skills/pypto-general-debug/SKILL.md`.
3. Stay in **one active module** at a time; stub downstream with golden-fed tensors if needed.
4. Re-run **module boundary** checks with **`detailed_tensor_compare`** and log results in **Per-module verification log**.
5. After fixes, re-run the current staged set's `test_<op>_module<suffix_k>.py` (or, after Phase D, `test_<op>.py`) and confirm **every** kernel output — not only the first tensor.

### 3a. Layout CI and pass regressions (quick pointers)

- **Layout / `pypto_function` loops:** After meaningful edits under `custom/`, run `.agents/skills/pypto-kernel-layout-check/scripts/run_validate_layout.sh` from the repository root using the `bash` line in `.agents/skills/pypto-kernel-layout-check/references/CI.md`. **Exit 1** means fix plan / test / staged naming issues **or** remove `for ... in range(...)` inside `pypto_function` — express that iteration with `pypto.loop` in `_<op>_kernel_impl` / the JIT kernel per `skills/pypto-op-develop/references/kernel-layer-format.md` §7.1.
- **Pass / compile failure right after a graph edit:** If logs show **PASS**-range codes (**`F4` / `F5`**, see **`docs/trouble_shooting/README.md`** → **`pass.md`**), follow **`.agents/skills/pypto-pass-error-locator/SKILL.md`**, **bisect** the PyPTO graph (e.g. last known-good **staged** file vs current), and re-check API constraints (**`query_op_index` / `docs/`**) before large rewrites. Re-run **`extract_pypto_calls.py`** on the failing file to see whether a new op ordering triggered the pass.

---

## 4. F21004 (`INVALID_VAL` / `TileShape::Current().GetVecTile()` invalid) — `REGISTER_COPY` and vector tile shapes

**Cause (framework):** **`F21004`** is raised in **`Operation`'s constructor** when **`TileShape::Current().GetVecTile()`** is invalid (e.g. `framework/src/interface/operation/operation.cpp` ~191–195).

**Why `REGISTER_COPY` shows up:** **`REGISTER_COPY`** is an **AIV** op. It needs a **valid vector tile** even when the **compiler inserts** that op (e.g. memory-conflict passes). There is **no** separate "REGISTER_COPY tiling" registration — the same **vec tile** rules apply.

**When is `VecTile` valid?** The stored list must be **non-empty** and **every value must be &gt; 0** (`tile_shape.cpp`).

### Fix — what to do in PyPTO Python

1. **Before any vector/tensor work in that scope** (including code that eventually leads to **`REGISTER_COPY`**), set **vector** tile sizes:

   ```python
   import pypto
   pypto.set_vec_tile_shapes(1, 1, 128, 128)   # positive ints per doc
   ```

2. Call it at the **start** of your **`@jit` / kernel function body**, and **again after any scope change** if your API uses **nested scopes**.

3. **Do not rely only on `set_cube_tile_shapes`** — cube tile does **not** replace vector tile for AIV ops like **`REGISTER_COPY`**.

4. If you use **both** cube and vector ops:

   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
   pypto.set_vec_tile_shapes(1, 1, 128, 128)
   ```

### If it still fails

- **Invalid args or zeros** — ensure all args are positive integers. Refer to **`docs/api/config/pypto-set_vec_tile_shapes.md`** for your PyPTO version's requirements (see `skills/pypto-op-develop/SKILL.md` §实现注意点 #7 / 常见错误 #4).
- **Wrong order** — anything that **adds ops** (reshape, view, passes inserting copy) must run **after** **`set_vec_tile_shapes`** on that execution path.
- **Symbolic / dynamic shapes** — ensure tile args resolve to **concrete positive integers** (see **`set_vec_tile_shapes`** + **`SymbolicScalar`** in `python/pypto/_controller.py` in your tree).

**Bottom line:** **`F21004` on `REGISTER_COPY`** is almost always **"no valid `vec_tile_shapes` in the current scope when this op was created."** Fix it with **`pypto.set_vec_tile_shapes(...)`** with **all positive sizes** before those ops run.

---

## 5. When logs point to device / AICore

- Load **`.agents/skills/pypto-aicore-error-locator/SKILL.md`** and follow its steps when the failure is an **aicore error** / device-side trace problem.

---

## 6. Error-code quick reference (repo)

- `.agents/skills/pypto-op-develop/references/error-code-troubleshooting.md` — flow for `Errcode: Fxxxxx!`.
- `docs/trouble_shooting/function.md` — e.g. **INVALID_VAL (0x21004)**, **UNKNOWN (0x3FFFF)**.
- **F21004 / vec tile / `REGISTER_COPY`:** see **§4** above.

---

## 7. When stopping is allowed

Only align with the Debug Agent's stop conditions (`.opencode/agents/pypto-op-debugger.md` Hard rules: 3 fix/re-verify cycles exhausted, missing reference, impossible golden, fundamental framework block, missing user-provided logs when required, or proven blind speculation after exhaustive structured attempts). **A single cryptic error line is never enough.**

---

## Final reminder

**Prefer ten documented failed attempts with log citations over one early exit.** Unknown error codes mean **escalate evidence and method**, not **stop**.
