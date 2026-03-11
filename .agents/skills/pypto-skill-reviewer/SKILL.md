---
name: pypto-skill-reviewer
description: "Review and score a skill directory for quality and best-practice compliance. Use when you need to audit a skill, check if a skill follows conventions, or evaluate a skill before publishing. Runs static checks via a Python script and semantic review via checklist, then produces a scored report with concrete fix suggestions."
user-invocable: true
---

# PyPTO Skill Reviewer

Review a skill directory against 51 rules across 9 dimensions, producing a scored report with actionable fix suggestions.

## Input

The user provides a `<skill-path>` — the path to the skill directory to review. This directory must contain a `SKILL.md` file.

## Reference Files

| File | Purpose | Load Timing |
|------|---------|-------------|
| [references/rules.json](references/rules.json) | Single source of truth for all 51 rules, dimensions, weights, and severity levels | Read at the start of Phase 1 and Phase 2 |
| [references/scoring-spec.md](references/scoring-spec.md) | Scoring algorithm: dimension weights, deduction formula, S0 veto, grade mapping | Read at the start of Phase 3 |
| [references/semantic-checklist.md](references/semantic-checklist.md) | Detailed check requirements, evidence standards, and judgment criteria for 22 semantic rules | Read at the start of Phase 2 |
| [scripts/validate_skill.py](scripts/validate_skill.py) | Deterministic static checker for 29 rules, outputs JSON findings | Executed in Phase 1 |
| [templates/report-template.md](templates/report-template.md) | Markdown template for the final review report | Read at the start of Phase 3 |

## Workflow

Execute three phases. Phase 1 and Phase 2 run in parallel because they have no data dependencies.

### Phase 1: Static Checks

1. Read [references/rules.json](references/rules.json) to understand the rule definitions.
2. Run the static checker against the target skill:
   ```
   python3 <reviewer-dir>/scripts/validate_skill.py --score <skill-path>
   ```
   Where `<reviewer-dir>` is this skill's own directory (`.opencode/skills/pypto-skill-reviewer`).
3. Capture the JSON output — an array of finding objects.
4. Verify the script exited successfully. If it fails, report the error and continue to Phase 2 results only.

### Phase 2: Semantic Review

1. Read [references/rules.json](references/rules.json) to identify which rules are `type: "semantic"`.
2. Read [references/semantic-checklist.md](references/semantic-checklist.md) for the detailed check procedures.
3. Read all files in the target skill directory:
   - `SKILL.md` (required)
   - All files in subdirectories (`references/`, `scripts/`, `templates/`, etc.)
4. Evaluate each semantic rule against the target skill content, following the checklist procedures exactly.
5. For each rule, produce a finding object with the required fields:
   ```json
   {
     "rule_id": "R07",
     "status": "FAIL|PASS|SKIP",
     "severity": "S1",
     "dimension": "D1",
     "message": "(must reference specific content from the target skill)",
     "evidence": {
       "file": "SKILL.md",
       "line": 3,
       "snippet": "(verbatim excerpt from target skill, ≥10 chars)"
     },
     "suggested_fix": "(concrete edit for this specific skill)"
   }
   ```
6. Run self-validation on all semantic findings:
   - **Snippet match**: Verify every `evidence.snippet` exists verbatim in the target skill files. Remove or fix any finding with a fabricated snippet.
   - **Uniqueness**: Ensure no two findings have identical `message` text. Merge or differentiate duplicates.
   - **Specificity**: Confirm every `message` and `suggested_fix` references the target skill's specific content, not generic advice.

### Phase 3: Scoring and Report

1. Read [references/scoring-spec.md](references/scoring-spec.md) for the scoring algorithm.
2. Read [templates/report-template.md](templates/report-template.md) for the report format.
3. Merge all findings from Phase 1 and Phase 2 into a single list.
4. Aggregate findings into issues by location:
   - **Aggregation key**: `file + line_range` (findings within ±5 lines of each other merge into one issue)
   - Each issue records all matched `rule_id` values
   - Generate a single unified fix suggestion per issue (with before/after comparison)
5. Calculate scores per dimension:
   ```
   dimension_raw   = max(0, 100 - sum_of_FAIL_deductions)
   dimension_score = dimension_raw × weight
   ```
6. Calculate total score: `total = Σ dimension_scores (D1–D9)`.
7. Apply S0 veto: if any S0 rule has status FAIL, cap total at 59.9 and set max grade to D.
8. D9 special case: if no `scripts/` directory exists in the target skill, award D9 full marks (5.0).
9. Map total score to grade: A (90–100), B (75–89), C (60–74), D (40–59), F (0–39).
10. Compute rule coverage: static rules (PASS + FAIL from script output) + semantic rules (PASS + FAIL + SKIP from Phase 2) = total evaluated. Coverage = evaluated / 51 × 100%.
11. Apply quality gates — filter findings before scoring:
    - **Internal misbound**: If a semantic finding's evidence clearly refers to the reviewer itself rather than the target skill, remove it and tag reason `internal_misbound_rule_or_evidence`.
    - **Insufficient evidence**: If a semantic finding's `evidence.snippet` cannot be found verbatim in the target files, remove it and tag reason `low_information_snippet`.
    - Record all filtered items for the quality gate section of the report.
12. Render the final report using the template, filling in all placeholders.

## Output

Output the complete review report in Markdown format directly to the user. The report MUST contain ALL 6 sections below — missing any section means the report is incomplete.

1. **Review summary** — skill name, total score (0-100, two decimal places), grade (A/B/C/D/F), S0 veto status (Yes/No), rule statistics (pass/fail/warn/skip counts)
2. **Dimension scores table** — 9 rows (D1-D9), each with: raw score (0-100), weight, weighted score, deduction breakdown listing every FAIL rule_id and its deduction value
3. **Rule coverage** — static count + semantic count = total evaluated, with per-status breakdown (PASS/FAIL/SKIP), coverage percentage = evaluated / 51 × 100%
4. **Quality gate** — filtered items with count and removal reason (internal_misbound / low_information_snippet)
5. **Issue list** — grouped by severity (S0 → S3), each issue with:
   - Issue description referencing specific target skill content
   - Matched rule IDs with severity tags
   - Location (`file:line`) and evidence (verbatim snippet ≥10 chars from target)
   - Concrete fix suggestion with before/after comparison
6. **Passed rules summary** — all passed rule IDs grouped by dimension

**Success criteria**: A valid report satisfies:
  (a) all 6 sections present,
  (b) total score = sum of dimension weighted scores (±0.01),
  (c) every finding's evidence.snippet exists verbatim in target files,
  (d) coverage denominator = 51.

## Error Handling

- **SKILL.md not found**: Report as a single S0 finding (R01), skip all other checks, output a minimal report with score 0 and grade F.
- **Script execution failure**: Log the error, proceed with semantic review only, note in the report that static checks were incomplete. Mark all 29 static rules as SKIP in coverage.
- **Script output malformed**: If validate_skill.py stdout is not valid JSON:
  1. Report: "Static analysis script returned non-JSON output"
  2. Include raw stderr (first 500 chars) in the report's quality gate section
  3. Proceed with semantic-only review
  4. Mark all 29 static rules as SKIP in coverage
- **Empty skill directory**: Same as SKILL.md not found.
- **Invalid frontmatter**: R01 FAIL triggers S0 veto. Continue checking other rules where possible (those not dependent on frontmatter data).

## Constraints

- Do not modify any files in the target skill directory — this is a read-only review.
- Do not fabricate evidence — every snippet must exist in the actual target files.
- **`rules.json` is the SINGLE SOURCE OF TRUTH** for all 51 rules. Do NOT:
  - Invent rules not defined in rules.json
  - Modify severity levels or dimension assignments from rules.json
  - Skip any rule — if a rule cannot be checked, mark it as SKIP with reason
  - Override rule definitions based on assumptions or external knowledge
- Every finding MUST reference a `rule_id` that exists in rules.json.
- Use the scoring formula from `scoring-spec.md` precisely — do not estimate or approximate scores.
