# Semantic Review Checklist

This checklist defines the 21 semantic rules that require LLM judgment. For each rule, follow the check requirements, evidence standards, and judgment criteria exactly.

## Output Format

For each rule, output a JSON object:

```json
{
  "rule_id": "R07",
  "status": "FAIL",
  "severity": "S1",
  "dimension": "D1",
  "message": "(must quote specific content from the target skill, never generic text)",
  "evidence": {
    "file": "SKILL.md",
    "line": 3,
    "snippet": "(must be a verbatim excerpt from the target skill, ≥10 characters)"
  },
  "suggested_fix": "(must be a concrete edit specific to this skill)"
}
```

For PASS results, set `status: "PASS"` and provide a brief `message` confirming what was found. Evidence is optional for PASS.

---

## D1 Frontmatter 元数据

### R07 — Description Must Answer "What" and "When" (S1)

**Check**: Read the `description` frontmatter field. It must clearly answer:
1. **What does this skill do?** — the purpose/outcome
2. **When should it be used?** — the trigger condition or user scenario

**Evidence standard**: Quote the description text. Identify which of the two questions is missing or poorly answered.

**Judgment**:
- PASS: Both questions are answered (explicitly or implicitly clear)
- FAIL: One or both questions are not addressed

### R08 — Description Contains Natural Trigger Phrases (S2)

**Check**: The `description` should contain phrases a user would naturally say when they want to invoke this skill. These are the keywords/phrases the AI assistant uses to match user requests to skills.

**Evidence standard**: Quote the description. If trigger phrases are present, list them. If absent, suggest what phrases should be added.

**Judgment**:
- PASS: Contains at least one natural trigger phrase
- FAIL: Only uses technical/formal language without user-facing trigger phrases

### R09 — Description Is Outcome-Oriented (S2)

**Check**: The `description` should focus on the outcome or result for the user, not just list features or capabilities.

**Evidence standard**: Quote the description. Identify whether it states outcomes ("reviews skills for quality") vs features ("checks frontmatter, validates structure, runs scripts").

**Judgment**:
- PASS: Primarily outcome-oriented
- FAIL: Primarily feature-listing without clear outcomes

---

## D2 简洁性与效率

### R14 — No Redundant Duplicate Content (S2)

**Check**: Scan all sections of SKILL.md for content that is repeated in substantially the same form. Identical instructions appearing in multiple sections are wasteful.

**Evidence standard**: Quote both occurrences with their section headings and line numbers.

**Judgment**:
- PASS: No significant duplication found
- FAIL: Same instruction or content block appears in two or more places

---

## D3 文件结构与导航

### R19 — Referenced Files Have Purpose and Timing Described (S2)

**Check**: For each file referenced in SKILL.md (via links or mentions of files in references/, scripts/, templates/), check that the skill explains:
1. **Purpose**: What the file is for
2. **Load timing**: When it should be read/executed

**Evidence standard**: List each referenced file and quote the context where it is mentioned. Note which files lack purpose or timing descriptions.

**Judgment**:
- PASS: All referenced files have both purpose and timing described
- FAIL: One or more files lack purpose or timing context

---

## D4 语言与表达

### R20 — Instructions Use Imperative Mood (S2)

**Check**: Instructions and directives in the skill body should use imperative mood ("Run the script", "Read the file") rather than passive suggestions ("The script should be run", "The file could be read").

**Evidence standard**: Quote specific lines that use passive/suggestive voice. Suggest imperative rewording.

**Judgment**:
- PASS: Instructions are predominantly imperative
- FAIL: Multiple key instructions use passive voice or indirect phrasing

### R21 — Avoid Vague Language (S3)

**Check**: Scan for hedging words and vague qualifiers: "consider", "you might", "perhaps", "maybe", "try to", "it would be good to", "ideally".

**Evidence standard**: Quote lines containing vague language with line numbers.

**Judgment**:
- PASS: No significant vague language in instructions
- FAIL: Hedging language appears in actionable instructions (not in background context)

### R23 — Instructions Explain Why (S2)

**Check**: Key instructions should include rationale. "Do X because Y" is stronger than just "Do X".

**Evidence standard**: Identify instructions that lack rationale. Quote them and suggest how to add reasoning.

**Judgment**:
- PASS: Most key instructions include reasoning or the reason is self-evident from context
- FAIL: Multiple important instructions lack any rationale

---

## D5 精确性与可执行性

### R24 — Steps Have Verifiable Success Criteria (S2)

**Check**: Workflow steps should define how to know they succeeded. "Run the script" alone is insufficient; "Run the script and verify the output contains no FAIL entries" is verifiable.

**Evidence standard**: Quote steps that lack success criteria.

**Judgment**:
- PASS: Most steps have explicit or clearly implied success criteria
- FAIL: Multiple steps have no way to verify completion

### R25 — Commands and Paths Are Specific (S2)

**Check**: Any commands, file paths, or tool references should be concrete and executable, not placeholder-like or generic.

**Evidence standard**: Quote any placeholder commands (`<command>`, `$SOME_VAR` without definition) or unresolved paths.

**Judgment**:
- PASS: All commands and paths are specific
- FAIL: Contains unresolved placeholders or generic references in executable context

### R26 — All Operations Provide Concrete Methods (S1)

**Check**: When the skill says to "validate X" or "check Y", it must specify how — a specific script, tool command, or step-by-step procedure.

**Evidence standard**: Quote operations that lack implementation details.

**Judgment**:
- PASS: All mentioned operations have concrete methods
- FAIL: One or more operations are mentioned without specifying how to do them

### R27 — Completion Criteria Are Defined (S2)

**Check**: The skill should clearly state when the overall task is complete. What is the final deliverable? How does the user know it's done?

**Evidence standard**: Quote the completion criteria if found, or note their absence.

**Judgment**:
- PASS: Clear completion criteria exist
- FAIL: No explicit definition of what constitutes completion

---

## D6 工作流完整性

### R28 — Clear Step-by-Step Workflow Defined (S1)

**Check**: The skill must contain a structured workflow — numbered steps, ordered phases, or clear sequential stages.

**Evidence standard**: Identify the workflow structure. If absent, note what a workflow should look like for this skill.

**Judgment**:
- PASS: Clear ordered workflow exists
- FAIL: No discernible step-by-step flow; instructions are scattered or unordered

### R29 — Workflow Steps Flow Smoothly (S2)

**Check**: The output of one step should connect logically to the input of the next. No "jumps" where context is lost between steps.

**Evidence standard**: Identify any gaps where a step's output doesn't feed into the next step's requirements.

**Judgment**:
- PASS: Steps have clear data/control flow between them
- FAIL: Gaps exist between steps where information is lost or undefined

### R30 — Error Handling or Failure Recovery Included (S2)

**Check**: The skill should address what to do when things go wrong — script failures, missing files, invalid input, unexpected states.

**Evidence standard**: Quote error handling instructions if found, or note their absence.

**Judgment**:
- PASS: At least basic error handling is described
- FAIL: No mention of error cases or recovery procedures

### R31 — Conditional Branches Are Fully Described (S2)

**Check**: When the skill uses if/else logic or conditional paths (e.g., "if X exists, do A; otherwise do B"), all branches must be specified.

**Evidence standard**: Quote conditional logic and identify any branches that lack instructions.

**Judgment**:
- PASS: All conditional branches have clear instructions, or no conditional logic exists
- FAIL: One or more branches are missing or underspecified

---

## D7 模式与最佳实践

### R32 — Progressive Disclosure Pattern (S3)

**Check**: The skill should follow a layered structure:
1. **Entry**: Brief summary/overview at the top
2. **Body**: Detailed instructions and workflow
3. **References**: Deep content in separate files

**Evidence standard**: Describe the document's structure and how well it follows this pattern.

**Judgment**:
- PASS: Clear layered structure from summary to detail to references
- FAIL: Content is flat (all at same detail level) or inverted (deep details before overview)

### R33 — Deterministic Scripts for Validation (S3)

**Check**: Verification tasks (checking correctness, validating format) should use deterministic scripts rather than relying solely on LLM judgment.

**Evidence standard**: Identify validation tasks and note whether they use scripts or rely on LLM alone.

**Judgment**:
- PASS: Key validation tasks use scripts or deterministic checks
- FAIL: All validation relies on LLM judgment with no deterministic component

### R51 — Multi-Option Default Recommendation (S2)

**Check**: When the skill presents multiple options, alternatives, or approaches for the user or the AI to choose from, a default or recommended option should be clearly indicated.

**Evidence standard**: Quote the section presenting options. Note whether a default is marked (e.g., "(recommended)", "prefer X", "default: Y").

**Judgment**:
- PASS: All multi-option sections have a clear default or recommendation, or no multi-option sections exist
- FAIL: Options are presented without any guidance on which to prefer

---

## D9 脚本与代码质量

### R42 — Scripts Include Basic Error Handling (S2)

**Check**: Script files in `scripts/` should include:
- Try/except or try/catch blocks for operations that can fail
- Meaningful error messages on failure
- Non-zero exit codes on error

**Evidence standard**: Quote the error handling code or note its absence.

**Judgment**:
- PASS: Scripts have basic error handling
- FAIL: Scripts lack try/except blocks and may silently fail
- SKIP: No scripts/ directory exists

### R50 — Scripts Handle Missing Dependencies Gracefully (S2)

**Check**: Scripts that import non-standard-library modules or invoke external tools should handle the case where the dependency is missing — either with a try/except around the import or an explicit check before use.

**Evidence standard**: Quote import statements. Note whether missing-dependency errors would produce a helpful message or an opaque traceback.

**Judgment**:
- PASS: All imports are stdlib-only, or non-stdlib imports are guarded with try/except and a clear error message
- FAIL: Non-stdlib imports exist without any guard, so a missing dependency would crash with an unhelpful traceback
- SKIP: No scripts/ directory exists

---

## D0 附加检查

### R44 — Fork Skills Include Explicit Task Instructions (S2)

**Check**: When `context: fork` is set, the skill body must contain clear instructions for what the spawned agent should do.

**Evidence standard**: Quote the task instructions for the forked agent.

**Judgment**:
- PASS: Clear task delegation instructions exist
- FAIL: Fork context is set but no clear task instructions for the agent
- SKIP: `context` is not set to `fork`

---

## Self-Validation Checklist

After completing all semantic checks, perform these validations on your own output:

1. **Snippet Match**: Every `evidence.snippet` MUST be a verbatim excerpt from the target skill files. Verify each one exists in the source.

2. **Uniqueness**: No two findings should have identical `message` text. If duplicates exist, merge them or differentiate.

3. **Specificity**: Every `message` and `suggested_fix` must reference the target skill's specific content (name, phrases, structure). Generic advice like "improve the description" without referencing the actual description content is invalid.
