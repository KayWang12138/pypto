# 算子开发流程：history_version/ → Git 版本控制

## 背景

原算子生成工作流在 Stage 5（代码实现）、Stage 6（精度修复）和 Stage 7（性能调优）中，每次 attempt 前需要把当前 `{op}_impl.py` 备份到 `custom/{op}/history_version/` 目录，用文件命名（`{op}_impl_s6_attempt{N}.py`）承载版本信息，回滚时靠文件复制。

痛点：
- 命名承载版本信息，难以 diff、难以审计
- `history_version/` 与工件目录混杂
- 无法原生查看跨 attempt 的变更历史
- 多算子平行开发时，各算子版本孤立、缺乏统一视角

## 方案

每个 `custom/{op}/` 作为**独立的 git 仓库**，全阶段产物通过 `git commit` 记录版本、通过 `git reset --hard` 完成回滚，不再使用 `history_version/` 备份目录。

### 仓库归属

- `custom/{op}/` 是一个**独立 git 仓库**，与外层 pypto 主仓互不影响
- 每个算子拥有各自的 `.git/`，互不干扰
- 主仓 `.gitignore` 已排除 `custom/`（无需额外调整）

### Commit 责任（Q2: 2b 方案）

每个 Subagent 自己负责所在阶段的 commit，Orchestrator 负责仓库初始化与自己亲自调用的 Stage 1/2：

| 阶段 | 责任方 | 触发时机 | commit message 模板 |
|------|--------|---------|---------------------|
| 初始化 | Orchestrator | `custom/{op}/` 首次创建 | `chore: init {op} sub-git repo` |
| Stage 1 | Orchestrator | SPEC.md 写入后 | `stage1: SPEC` |
| Stage 2 | Orchestrator | API_REPORT.md 写入后 | `stage2: API_REPORT` |
| Stage 3 | `pypto-op-analyst` | `{op}_golden.py` 门禁通过 | `stage3: golden` |
| Stage 4 | `pypto-op-analyst` | `DESIGN.md` 门禁通过 | `stage4: DESIGN` |
| Stage 5 | `pypto-op-developer` | attempt 前后各一次 | `stage5 attempt-{K}: before/after` |
| Stage 6 | `pypto-op-developer` | 精度修复 attempt 前后各一次 | `stage6 attempt-{K}: before/after` |
| Stage 7 | `pypto-op-perf-tuner` | 每轮调优前后各一次 | `stage7 iter-{K}: before/after` |

### .gitignore 约定

屏蔽噪声文件，但**保留**工件与状态文件：
```
__pycache__/
*.pyc
*.pyo
*.log
*.tmp
*.bak
.pytest_cache/
build/
dist/
```

显式保留版本控制（不 ignore）：
- `.orchestrator_state.json`（阶段状态机）
- `debug_log.md`（每次 attempt 的调试痕迹）

### 回滚契约

| 操作 | 命令 |
|------|------|
| 固化当前状态 | `git -C custom/{op} commit -am "<msg>"` |
| 回滚到上一 commit | `git -C custom/{op} reset --hard HEAD~1` |
| 回滚到指定 sha | `git -C custom/{op} reset --hard <baseline_sha>` |
| 审计历史 | `git -C custom/{op} log --oneline` |

### Subagent 返回字段

Stage 3/4 (`analyst`)：`commit_sha`

Stage 5/6 (`developer`)：`baseline_sha`、`candidate_sha`、`rollback: true/false`

Stage 7 (`perf-tuner`)：`baseline_sha`、`candidate_sha`、`rollback: true/false`

---

## 已完成的代码改动

| 文件 | 关键改动 |
|------|---------|
| `.opencode/agents/pypto-op-orchestrator.md` | 新增 Step 0 git init bootstrap；目录树 `history_version/` → `.git/ + .gitignore`；新增"版本控制"章节；Stage 1/2 commit 职责归到 Orchestrator |
| `.opencode/agents/pypto-op-analyst.md` | Stage 3/4 执行清单加 `git commit` 步骤；约束章节新增 commit 责任；输出 schema 加 `commit_sha` |
| `.opencode/agents/pypto-op-developer.md` | 「备份规则」→「git 版本管理规则」；`history_version/` 引用全部移除；输出 schema: `backup_path` → `baseline_sha` + `candidate_sha` |
| `.opencode/agents/pypto-op-perf-tuner.md` | Stage 7 单轮：pre-commit baseline → 候选 commit → 采纳或 `git reset --hard` 回滚；输出 schema 加 `baseline_sha` / `candidate_sha` |

## 未改动的位置（可能仍提及 history_version）

- `.claude/agents/pypto-op-*.md` — Claude 镜像，按约定只改 `.opencode/`
- `docs/plans/*.md`, `docs/agents/*.md`, `docs/architecture/*.md` — 历史设计文档
- `session-*.md`, `fracture-point-*.md` — 会话记录
- 其他 skill 文档（如适用）

需要时可按需单独批量迁移。