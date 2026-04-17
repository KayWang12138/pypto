---
name: pypto-operator-auto-tuner
description: "Automated performance tuning scripts for PyPTO operators. Swimlane data extraction, AIV dependency chain analysis, and leafhash-to-code mapping."
triggers: ["swimlane analysis", "AIV dependency", "leafhash", "auto tune", "performance scripts"]
---

# PyPTO Operator Auto-Tuner

Automation scripts for performance analysis and tuning of PyPTO operators.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/analyze_swimlane.py` | Extract and analyze swimlane data from profiling output | `python3 scripts/analyze_swimlane.py <profiling_data>` |
| `scripts/analyze_aiv_dep_chains.py` | Analyze AIV dependency chains to identify pipeline stalls | `python3 scripts/analyze_aiv_dep_chains.py <profiling_data>` |
| `scripts/leafhash_to_code.py` | Map leafhash identifiers back to source code locations | `python3 scripts/leafhash_to_code.py <leafhash>` |

## Workflow

1. Run the target operator with profiling enabled.
2. Execute `analyze_swimlane.py` on the profiling output to extract per-stage timing.
3. Execute `analyze_aiv_dep_chains.py` to identify dependency bottlenecks.
4. Use `leafhash_to_code.py` to map performance-critical nodes back to source code.
5. Apply optimizations based on findings, re-profile, and compare.

## References

- `references/leafhash-to-code-mapping.md` — Mapping guide for leafhash identifiers to frontend source code.
