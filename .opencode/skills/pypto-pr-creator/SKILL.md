---
name: pypto-pr-creator
description: "PyPTO 项目 Pull Request 创建全流程指南。当需要为 cann/pypto 仓库创建 PR、编写 commit message、准备代码提交、检查 PR 规范时使用。覆盖：(1) 仓库发现与 fork 验证, (2) 用户确认检查点（mandatory blocking confirmation）, (3) 分支创建与 commit, (4) 通过 GitCode MCP 创建/更新 PR, (5) Post-PR 结构化报告。触发词：创建PR、提交PR、PR规范、commit message、pypto贡献、代码提交到pypto、更新PR。"
---

# PyPTO PR Creator

## 强制约束

| 规则 | 说明 |
|------|------|
| **远程操作** | 所有远程操作必须通过 GitCode MCP，禁止直接使用 `GITCODE_TOKEN` |
| **Origin 配置** | `origin` 必须指向用户 fork（如 `<username>/pypto`），不能是 `cann/pypto` |
| **隐私保护** | 禁止在屏幕、日志、错误信息中打印或暴露 `GITCODE_TOKEN` |
| **用户确认** | 创建分支、commit、push、创建/更新 PR 前必须获得用户明确确认 |
| **文件路径** | 使用 `$PYPTO_REPO` 指代用户的 pypto 本地仓库根目录 |

### GitCode MCP 工具

- PR 操作：`gitcode_create_pull_request`, `gitcode_update_pull_request`, `gitcode_list_pull_requests`
- 仓库查询：`gitcode_get_repository`

未安装时按 `gitcode-mcp-install` skill 完成配置。

---

## 完整工作流（7 Phase）

### Phase 1: 仓库发现与验证

**目标**：找到用户的 pypto fork 仓库，验证 origin 配置正确。

**执行逻辑**（按优先级）：

1. 检查当前工作目录是否是 pypto 仓库（`git rev-parse --is-inside-work-tree`）
2. 检查 origin：必须包含 `pypto` 且不能包含 `cann/pypto`
3. 若当前目录不符合，在工作区搜索（`find` 搜索 `.git` 目录，检查 remote）
4. 检测浅克隆：`git rev-parse --is-shallow-repository`（若 true，后续需 `git fetch --unshallow origin`）

| Origin 配置 | 判定 | 处理 |
|-------------|------|------|
| `<username>/pypto`（用户 fork） | ✅ | 直接使用 |
| `cann/pypto`（upstream） | ❌ | 需修复 origin：`git remote set-url origin https://gitcode.com/<username>/pypto.git`；或重新克隆 fork：`git clone https://gitcode.com/<username>/pypto.git` |
| 无 origin 或非 pypto | ❌ | 询问用户正确的仓库路径 |

**通过 GitCode MCP 验证 fork 关系**：

```python
result = gitcode_get_repository(owner="<username>", repo="pypto")
# 验证: result.parent.full_name == "cann/pypto"
```

### Phase 2: 用户确认（强制阻塞）

> **在获得用户明确确认之前，禁止执行任何 git 操作。**

向用户展示执行计划表（包含以下字段），等待确认：

- 本地仓库路径（`$PYPTO_REPO`）
- Fork 仓库（`<username>/pypto`）
- 本地分支名
- Commit 信息（`tag(scope): Summary` 格式）
- Push 目标（origin → 分支名）
- PR 目标（`cann/pypto` → `master`）
- PR 标题与 Body 预览

### Phase 3: 预检修复

用户确认后执行：

- 浅克隆修复：若 `git rev-parse --is-shallow-repository` 为 true → `git fetch --unshallow origin`
- 再次验证 origin 不是 `cann/pypto`

### Phase 4: 创建分支、Commit、Push

```bash
git -C "$PYPTO_REPO" checkout -b <branch_name>
git -C "$PYPTO_REPO" add <files>
git -C "$PYPTO_REPO" commit -m "tag(scope): Summary"
git -C "$PYPTO_REPO" push origin <branch_name>
```

> **认证说明**：push 依赖 git credential helper 或 `.gitconfig` 中已配置的凭据。

#### Push 失败诊断与修复

若 push 报认证错误（401/403），按以下步骤排查：

**Step 1: 检查 http.extraheader 配置**

```bash
git config --local --get http.extraheader
# 如输出包含 "Authorization: Bearer ..." → 这是问题根源
```

**原因**：GitCode 不支持 Bearer token 认证，只支持 HTTP Basic Auth。若配置了 `http.extraheader=Authorization: Bearer <token>`，git 会强制使用 Bearer token 导致认证失败。

**Step 2: 删除错误的配置并修复认证**

```bash
# 删除错误的 Bearer token 配置
git config --local --unset http.extraheader

# 配置正确的 credential helper
git config --local credential.helper store

# 存储凭据（使用 HTTP Basic Auth）
git credential-store store << 'EOF'
protocol=https
host=gitcode.com
username=<your_username>
password=<your_token>
EOF
```

**Step 3: 调试认证问题（可选）**

```bash
# 查看实际发送的认证头
GIT_CURL_VERBOSE=1 git push origin <branch_name> 2>&1 | grep -i authorization
# 正确: Authorization: Basic <base64>
# 错误: Authorization: Bearer <token>
```
### Phase 5: 创建或更新 PR

#### 5.1 判断创建还是更新

```python
# 查询是否已有该分支的 open PR
prs = gitcode_list_pull_requests(owner="cann", repo="pypto")
# 筛选: state == "opened" 且 source_branch 匹配当前分支

# 如存在匹配 PR → 询问用户：更新现有 PR 还是创建新 PR
# 如不存在 → 创建新 PR
```

#### 5.2 创建新 PR

```python
gitcode_create_pull_request(
    owner="cann",                          # 上游仓库 owner（固定）
    repo="pypto",                          # 上游仓库名（固定）
    title="tag(scope): Summary",           # PR 标题
    head="<username>:<branch_name>",       # ⚠️ 必须是 "fork_owner:branch" 格式
    base="master",                         # 目标分支
    body="..."                             # PR 描述（禁止为空）
)
```

**关键细节**：
- `head` 格式为 `<username>:<branch_name>`（冒号分隔），不是纯分支名
- `owner`/`repo` 指向**上游仓库**（`cann/pypto`），不是 fork
- 所有参数名必须**小写**

> 完整参数说明和示例见 [references/pr-spec.md](references/pr-spec.md)。

#### 5.3 更新现有 PR

```python
gitcode_update_pull_request(
    owner="cann",
    repo="pypto",
    pull_number=<pr_number>,    # 目标 PR 编号
    title="新标题",              # 可选
    body="新描述",               # 可选
    state="open"                # 可选: "open" 或 "closed"
)
```

### Phase 6: Post-PR 报告

PR 操作成功后，向用户展示结构化报告，包含：

- PR 链接（取自 `gitcode_create_pull_request` 返回值的 `html_url` 字段）
- 操作类型（创建/更新）
- PR 标题、源分支 → 目标分支
- Commit hash 与 message
- 后续操作提示（发送 `compile` 触发 CI、等待 review）

### Phase 7: 追加修改（可选）

若需修改已有 PR：在同一分支追加 commit 并 push，PR 自动更新。如需修改标题/描述，用 `gitcode_update_pull_request`。

---

## PR 标题与 Body 规范

详见 [references/pr-spec.md](references/pr-spec.md)。

### 速查

| 规则 | 说明 |
|------|------|
| 格式 | `tag(scope): Summary` |
| Tag | feat / fix / docs / style / refactor / test / perf |
| Summary | 英文、首字母大写、无句号、祈使语气 |
| Body | 禁止为空，描述动机+变更列表 |

---

## 提交前检查清单

详见 [references/checklist.md](references/checklist.md)。

---

## 常见陷阱

| 陷阱 | 解决方案 |
|------|---------|
| origin 指向 `cann/pypto` | `git remote set-url origin https://gitcode.com/<username>/pypto.git`，或重新克隆你的 fork |
| push 报 "shallow update not allowed" | `git fetch --unshallow origin` |
| push 报认证错误 401/403 | 检查并删除 `http.extraheader` Bearer 配置，使用 credential.helper store |
| git 强制使用 Bearer token | `git config --local --unset http.extraheader` 并重新配置 credential |
| PR 创建报 "pre receive hook check failed" | commit message 不符合 `tag(scope): Summary` |
| PR 创建返回 400 | 检查 `head` 是否用了 `<username>:<branch_name>` 格式 |
| `head` 只有分支名 | 必须是 `<username>:<branch_name>` |
| `owner` 指向了 fork | `owner` 应为 `cann`（上游仓库） |
---

## 相关文件

| 文件 | 路径 |
|------|------|
| 贡献指南 | `$PYPTO_REPO/CONTRIBUTION.md` |
| PR 模板 (EN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE.md` |
| PR 模板 (CN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` |
| 详细规范 | `$PYPTO_REPO/docs/contribute/pull-request.md` |
| 代码检查规则 | `$PYPTO_REPO/docs/contribute/code-check-rule.yaml` |

> `$PYPTO_REPO` = 用户的 pypto 本地仓库根目录
