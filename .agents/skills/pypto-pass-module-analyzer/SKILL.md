---
name: pypto-pass-module-analyzer
description: "Analyze one specific PyPTO pass module and produce a structured markdown report with evidence from source code. Use when users ask to analyze, understand, review, or explain a pass implementation (for example: 分析 AutoCast Pass, 解释 RemoveRedundantReshape 的处理逻辑, 分析 pass_manager 里的某个 pass). This skill is for single-pass deep analysis, not full-repo pass inventory."
---

# PyPTO Pass Module Analyzer

Generate a deterministic, evidence-based report for one pass.

## Inputs

Required:
- Pass name (for example `AutoCast`, `RemoveRedundantReshape`)

Optional:
- Existing notes/doc that should be merged into analysis
- Output directory (default current directory)

## Bundled files

- `Pass_Analysis_Template.md`: output template
- `scripts/collect_pass_metadata.py`: collect strategy index and candidate source files
- `scripts/validate_report.py`: validate output report format

## Workflow

1. Normalize the pass name
- Accept `AutoCast`, `AUTO_CAST`, `auto_cast` and normalize to canonical pass name.

2. Collect metadata (mandatory)
- Run:
```bash
python3 scripts/collect_pass_metadata.py --pass "<PASS_NAME>" --repo-root .
```
- Read stdout JSON fields:
  - `canonical_name`
  - `pvc2_ooo_index` (or `null`)
  - `output_file_name`
  - `candidate_files`

3. Analyze code with evidence
- Focus on:
  - `RunOnFunction`
  - `PreChecker` / `PostChecker`
  - key helper functions and I/O behavior
  - opcode special handling (`GetOpcode`, `Opcode::OP_`)
- For each major conclusion, attach at least one concrete evidence item:
  - file path
  - function name
  - key condition/action (short quote <= 20 words)

4. Render report using template
- Output filename must be exactly `output_file_name` from metadata script.
- Keep code snippets short. Prefer function names + behavior summary.

5. Validate report (mandatory)
- Run:
```bash
python3 scripts/validate_report.py --report "<OUTPUT_FILE>"
```
- If validation fails, revise report and rerun until pass.

## Output contract

Must produce exactly one markdown report for the requested pass, containing these sections:

1. `## 1. Pass 概述`
2. `## 2. 代码分析`
3. `## 3. 业务分析`
4. `## 4. OPCode 特判分析`
5. `## 5. 总结`
6. `## 6. 相关文件`
7. `## 7. 附录`

Mandatory quality gates:
- No unresolved placeholders like `[Pass名称]`, `[描述...]`, `TODO`, `TBD`.
- At least 3 evidence lines in format:
  - `- Evidence: <file>:<function> - <fact>`
- `## 6. 相关文件` must include at least 2 project file paths.

## Scope boundaries

In scope:
- Deep analysis for one pass.
- Explain design intent and special-case handling.

Out of scope:
- Bulk generation for all passes in one request.
- Refactoring code or changing pass behavior.
- Performance benchmarking.

## Failure handling

- If pass cannot be found, return a short error report with:
  - searched keyword
  - checked locations
  - suggested closest pass names from metadata script
- If no source file candidate is found, stop and report `BLOCKED: source file unresolved`.
