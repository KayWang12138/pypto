# PyPTO PR 详细规范

> 来源：GitCode cann/pypto 仓库 `$PYPTO_REPO/docs/contribute/pull-request.md`

## PR 标题格式

```
tag(scope): [Short summary]
```

### Tag 类型

| Tag | 用途 | 示例 |
|-----|------|------|
| `feat` | 新功能 | `feat(interface): Add new API for tensor slicing` |
| `fix` | Bug 修复 | `fix(compiler): Fix memory leak in graph optimization` |
| `docs` | 文档变更 | `docs(api): Update tensor creation documentation` |
| `style` | 代码格式 | `style(core): Fix indentation in utils module` |
| `refactor` | 重构 | `refactor(backend): Simplify device manager logic` |
| `test` | 测试相关 | `test(ops): Add unit tests for softmax operator` |
| `perf` | 性能优化 | `perf(kernel): Optimize memory access pattern` |

### Scope 规则

- 填写受影响的模块/组件名称
- 多模块用 `|` 分隔：`feat(frontend|backend): ...`
- 涉及过多模块时使用 `all`：`refactor(all): ...`

### Summary 规则

- **英文编写**
- **首字母大写**
- **结尾不加句号**
- **祈使语气**（imperative mood）

| ✅ 正确 | ❌ 错误 |
|---------|---------|
| `Add support for FP16` | `Added support for FP16` |
| `Fix memory leak in optimizer` | `Fixes memory leak in optimizer.` |
| `Update API documentation` | `updated API documentation` |

---

## PR Body 格式

### 结构

```
tag(scope): [Short summary]

[变更描述 - 动机、背景、上下文]

Changes:
- [变更点1]
- [变更点2]

Related Issues: #1234,#5678
```

### 要求

1. **禁止空 Body** — 必须清晰传达变更意图
2. **描述动机和上下文** — 让 reviewer 理解 why，不只是 what
3. **移除无关模板内容** — 不要保留模板注释和占位符
4. **关联 Issue** — 相关 Issue 放在最后一行 `Related Issues: #xxx`
5. **多 commit 场景** — 在 Body 中简要列出每个 commit 摘要

### 完整示例

```
feat(interface): Optimize the pypto.cond with concrete value

The origin implementation of pypto.cond generate both if/else branch,
even if the condition is always true or false, in this PR, we optimize
the implementation to generate only one branch.

Changes:
- Optimize the pypto.cond with concrete value, which can reduce the number of branches in the program
- Update the test cases to cover the new features

Related Issues: #1234,#5678
```

---

## Cross-Fork PR 技术细节

### gitcode_create_pull_request 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `owner` | str | ✅ | **上游仓库**所有者，固定为 `cann`（不是 fork owner） |
| `repo` | str | ✅ | **上游仓库**名称，固定为 `pypto` |
| `title` | str | ✅ | PR 标题，格式 `tag(scope): Summary` |
| `head` | str | ✅ | 源分支，格式 **`<username>:<branch_name>`**（冒号分隔） |
| `base` | str | ✅ | 目标分支（上游仓库的分支，通常为 `master`） |
| `body` | str | ❌ | PR 描述（建议必填，描述动机与变更） |

### 创建 PR 示例

```python
gitcode_create_pull_request(
    owner="cann",                                # 目标（upstream）owner
    repo="pypto",                                # 目标仓库
    title="fix(ops): Fix precision issue in softmax",
    head="<username>:fix/softmax-precision",     # ⚠️ fork_owner:branch
    base="master",
    body="## 问题描述\n...\n\n## 修复方案\n..."
)
```

### 更新 PR 示例

```python
gitcode_update_pull_request(
    owner="cann",              # 上游仓库 owner
    repo="pypto",              # 上游仓库名
    pull_number=<pr_number>,   # PR 编号
    title="新标题",             # 可选
    body="新描述",              # 可选
    state="open"               # 可选: "open" 或 "closed"
)
```

### head 参数常见错误

```python
head="feat/add-pr-guide"              # ❌ 缺少 fork owner
head="<username>/feat/add-pr-guide"   # ❌ 用了 / 而非 :
head="<username>:feat/add-pr-guide"   # ✅ 正确格式
```

### MCP 参数注意

- 所有参数名**必须小写**：`owner`, `repo`, `title`, `head`, `base`, `body`
- 大写参数名（如 `Owner`, `Repo`）会导致验证错误

---

## Commit 规范

### 单 Commit

- 每个 Commit 只做一件事
- 格式：`tag(scope): [Short summary]`
- PyPTO 使用 **squash merge**，无需详细的 commit body

### 多 Commit 顺序

推荐顺序：`fixup` → `refactor` → `feat` → `test`

---

## PR 准则

1. **单一职责** — 每个 PR 只做一件事，易于 review 和 revert
2. **避免无关变更** — 不在同一 PR 中混合不相关修改
3. **请求审查** — 在 PR 评论中 @committer 或贡献者
4. **描述清晰** — Body 必须清楚传达变更意图

## FAQ

### pre-hooks declined

PR/commit 标题必须遵循 `tag(scope): [Short summary]` 格式。Summary 首字母必须大写，结尾不加句号。

### push 被拒绝 "shallow update not allowed"

本地仓库为浅克隆，需先执行：
```bash
git fetch --unshallow origin
```
