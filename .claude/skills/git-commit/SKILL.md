---
name: git-commit
description: Complete git commit workflow for PyPTO including pre-commit review, staging, message generation, and verification. Use when creating commits, preparing changes for commit, or when the user asks to commit changes.
---

# PyPTO Git Commit Workflow

## Step 1: Summarize Changes

Run the following in parallel to understand what changed:

```bash
git status
git diff HEAD
git log -5 --oneline
```

Read the changed files as needed to understand the context.

## Step 2: Stage Changes

Stage all relevant modified files:

```bash
git add <file1> <file2> ...
```

**Never stage**: Build artifacts (`build/`, `*.o`), temp files, IDE configs.

## Step 3: Generate Commit Message

**Format**: `type(scope): description (≤72 chars)`

**Types**: feat, fix, refactor, test, docs, style, chore, perf
**Scope**: Module/component (ir, printer, builder, language, frontend, …)
**Description**: Present tense, action verb, no period

**Good examples**:

```text
feat(language): Add Array field support in tiling parameters
fix(printer): Update printer to use yield_ instead of yield
refactor(builder): Simplify tensor construction logic
test(ir): Add edge case coverage for structural comparison
```

Optionally include a short body (1-3 lines) explaining *why* if the change is non-obvious.

## Step 4: Commit

```bash
git commit -m "type(scope): description"
# Or with body:
git commit -m "$(cat <<'EOF'
type(scope): description

Body explaining the why.
EOF
)"
```

## Step 5: Verify

```bash
git show HEAD --name-only
```

## Co-Author Policy

**❌ NEVER add AI assistants**: No Claude, ChatGPT, Cursor AI, etc.
**✅ Only credit human contributors**: `Co-authored-by: Name <email>`

## Checklist

- [ ] Only relevant files staged (no build artifacts)
- [ ] Message format: `type(scope): description` (≤72 chars, present tense, no period)
- [ ] No AI co-authors
- [ ] `git show HEAD --name-only` confirms correct files
