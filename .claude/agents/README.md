# PyPTO 9-Agent System for Claude Code

This directory contains **8 sub-agent definitions** ported from `.opencode/agents/`. The team structure and phase ownership match the opencode setup, with two additions: **all kernel execution runs on the remote NPU server**, and **Lead lives in `CLAUDE.md`** (the primary-agent file at the project root) because only the primary Claude Code agent can dispatch sub-agents via the Task tool.

## Agent roster

| Agent | Location | Phase | Role |
|---|---|---|---|
| **Lead** (primary) | `/CLAUDE.md` | 0–6 | **Primary orchestrator.** Dispatches sub-agents via Task tool, enforces gates. Never writes code. |
| `planning` | `.claude/agents/planning.md` | 0 | Translates user request → SPEC.md, API_REPORT.md, plan seed. |
| `algorithm` | `.claude/agents/algorithm.md` | 1 | PyPTO-friendly golden.py in PyTorch/NumPy. Zero implicit transposes. |
| `architecture` | `.claude/agents/architecture.md` | 1–2 boundary | DESIGN.md: tiling, loop structure, memory plan, perf targets. |
| `design` | `.claude/agents/design.md` | 2 | Module decomposition, contracts, staged file layout. |
| `coding` | `.claude/agents/coding.md` | 3–5 | Implements ONE staged file per dispatch. Never debugs. |
| `verification` | `.claude/agents/verification.md` | 2 (modular golden + adversarial suite), 3–5 gates, 6 regression | Judge-only. Builds `<op>_golden_modular.py` and `adversarial_runner.py` (supports prefix-eval `--up-to-module k`), runs checks on NPU, emits pass/fail + failure category. |
| `debug` | `.claude/agents/debug.md` | on GATE failure | Investigator-only. Proposes patches. Never edits production. |
| `optimization` | `.claude/agents/optimization.md` | 6 | Perf tuning after GATE 4. 3 stages: frontend → swimlane → incore. |

### Why Lead lives in `CLAUDE.md` (not `.claude/agents/lead.md`)

Claude Code's Task tool — the only way to dispatch a sub-agent — is available **only to the primary agent**. Sub-agents cannot dispatch further sub-agents. Because Lead's entire job is orchestration (dispatching the other 8 sub-agents across phases and gates), Lead **must** be the primary agent.

`CLAUDE.md` is auto-loaded by Claude Code at session start, so every session in this project starts with the primary Claude agent already configured as Lead. There is no `lead.md` sub-agent file — opening one would create a non-functional Lead (no Task tool).

## Execution model (different from opencode)

All opencode agents assumed local execution. **These Claude Code agents route every kernel run through the NPU server.**

- **Mac side (Claude Code):** code generation, editing, static checks (`validate_kernel_structure`, `extract_pypto_calls.py`), plan orchestration, log analysis.
- **NPU side (remote server):** kernel execution, `detailed_tensor_compare`, layout checks, precision tests, perf measurement.

### Primary mechanism: `Run <file> on npu:<N>` prompt

Claude Code has a built-in NPU execution mechanism. When an agent needs to run a kernel file, it issues a natural-language prompt of the form:

```
Run <file> on npu:<N>
```

Examples:
- `Run custom/relu/relu_module1.py on npu:8`
- `Run custom/matmul/test_e2e.py on npu:0`

Claude Code automatically uploads the project state, runs the file on the specified NPU device, and returns the stdout/stderr inline. **This is the primary mechanism for single-file kernel execution across all agents.**

Each operator is pinned to an NPU device number recorded in `custom/plan/<op>.md` under `execution.npu_device`. Lead asks for this on the first user turn.

### Fallback: batch scripts

For multi-file test runs (pytest across a directory), layout checks, perf profiling, or log aggregation that doesn't fit a single-file run, the helper scripts exist:

| Script | Purpose |
|---|---|
| `./scripts/npu_sync.sh` | Rsync the project to NPU manually |
| `./scripts/npu_test.sh <op>` | Sync + pytest over an operator dir + layout check + pull logs |
| `./scripts/npu_run.sh "<cmd>"` | Arbitrary remote command on NPU |
| `./scripts/npu_shell.sh` | Interactive shell on NPU, landing in project dir |

Golden scripts (pure PyTorch/NumPy) still run locally on Mac — they're not NPU kernels.

## One-time setup (before using these agents)

1. **SSH config**: add an entry in `~/.ssh/config`:
   ```
   Host npu
       HostName 1.95.147.197
       User y00960395
       ServerAliveInterval 60
       IdentityFile ~/.ssh/id_ed25519
   ```
2. **SSH key auth**: `ssh-copy-id npu` (so Claude Code can run ssh/rsync non-interactively)
3. **Test connectivity**: `ssh npu "hostname"` should return the NPU hostname without prompting for a password.
4. **Create helper scripts**: see `scripts/README.md` (or ask the Lead Agent to generate them on first run).
5. **Verify project exists on NPU**: `./scripts/npu_sync.sh` followed by `ssh npu "ls ~/pypto-multi/"`.

If any of these are missing, Lead will halt and ask for setup.

## Dispatch flow (Phase 3 example)

```
User → lead (asks for op name, shapes, perf target, NPU device)
     → planning (Phase 0) → lead
     → algorithm (Phase 1) → lead
     → architecture (Phase 1–2) → lead
     → design (Phase 2 decomposition + module_interfaces.yaml) → lead
     → verification (Phase 2 scaffolding: build <op>_golden_modular.py +
                     adversarial_suite.json + adversarial_runner.py;
                     run GATE A.5 composition-verify + GATE B --self-test) → lead
     → for each module M_k:
         coding (writes M_k file) → lead
         verification (Run <file> on npu:<N>
                      + Run adversarial_runner.py --up-to-module k on npu:<N>) → lead
             if pass → commit, next module
             if fail → debug (proposes patch; uses failing_module_boundary
                              from evaluation_report.json) → lead
                       coding (applies patch) → lead
                       verification (re-run on NPU) → ...
     → optimization (Phase 6, NPU perf tuning + full adversarial sweep regression)
```

### Modular golden + prefix evaluation (Phase 2/3)

Verification produces two artifacts that support per-module debuggability (ported from the Joshua evaluator design):

1. **Modular torch golden** (`custom/<op>/eval/<op>_golden_modular.py`) — pure-torch per-module reference derived from `module_interfaces.yaml`. `golden_composed(*primary_inputs)` must match `<op>_golden(*primary_inputs)` within tolerance on every `(seed, shape)` pair from `composition_verification` (GATE A.5, sub-gate of GATE 2).
2. **Adversarial test suite + prefix-evaluation runner** (`adversarial_suite.json`, `test_inputs.py`, `adversarial_runner.py`) — ≥ 2 cases per level L1 (structural) / L2 (precision) / L3 (edge) / L4 (regression) / L5 (adversarial). The runner's `--up-to-module k` flag composes modules [1..k] from the impl with modules [k+1..N] from the modular golden, giving Phase 3 a per-module correctness signal even before the full kernel is implemented (GATE B, sub-gate of GATE 2).

Every Phase 3 verification dispatch runs prefix-eval in addition to the module's own entry-point test. The `evaluation_report.json` fields `status`, `first_failure.failing_module_boundary`, and `first_failure.failure_category` feed directly into the verdict the Lead relays to @debug — the failing module boundary narrows the fix domain and distinguishes an intra-module bug from an output-contract mismatch.

### Information barrier

Verification reports sent back to Lead are sanitized by `_sanitize` before leaving the eval workspace — they never contain golden tensor values, golden source excerpts, or raw input tensor contents. @debug and @coding derive correctness independently from `SPEC.md`, `module_interfaces.yaml`, and the user-provided golden.

## How Claude Code invokes these agents

In a Claude Code session, the user talks to the primary agent = Lead directly. Lead dispatches sub-agents via the Task tool, specifying the sub-agent name with `subagent_type`:

```
Task(subagent_type="verification", prompt="Run GATE 3 checks on custom/relu/relu_module1.py using npu:8")
```

Claude Code reads the sub-agent's frontmatter, loads its system prompt, gives it the declared tools, and runs it as an isolated context.

## Differences from .opencode/agents/

- `mode: subagent` removed (Claude Code uses primary + sub-agents natively)
- **Lead is now the primary agent**, with its system prompt living in `/CLAUDE.md` (auto-loaded at session start). The other 8 agents are sub-agents in this directory.
- `tools:` converted from object-of-booleans to Claude Code tool names (Read, Write, Edit, Bash, etc.). Lead additionally has `Task` (only available to primary).
- Explicit NPU execution rules added to Lead (in CLAUDE.md) and to `verification`, `debug`, `optimization`, `coding`
- Primary execution mechanism is the `Run <file> on npu:<N>` prompt (Claude Code built-in), with scripts as batch fallback
- Each operator is pinned to an NPU device number in its plan file (`execution.npu_device`)
- `coding` and `debug` now explicitly forbid local kernel execution
- `verification` has a new `infra` failure category for SSH/rsync issues
