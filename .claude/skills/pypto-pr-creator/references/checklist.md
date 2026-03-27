# PyPTO PR 提交前检查清单

## Phase 1: 环境预检

- [ ] 确认本地仓库路径正确（`$PYPTO_REPO`）
- [ ] `git remote -v` 确认 origin 指向用户 fork（非 `cann/pypto`）
- [ ] **添加 upstream remote**：`git remote add upstream https://gitcode.com/cann/pypto.git`（如不存在）
- [ ] 通过 `gitcode_get_repository` 确认 fork 链关系（parent.full_name == "cann/pypto")
- [ ] `git rev-parse --is-shallow-repository` 检测浅克隆
- [ ] 若浅克隆 → `git fetch --unshallow origin`

## Phase 2: Git 认证检查

### 2.1 检测认证状态
- [ ] 检测 `GITCODE_TOKEN` 环境变量：`[ -n "$GITCODE_TOKEN" ] && echo '已设置'`
- [ ] 检测 SSH Key：`ls ~/.ssh/*.pub 2>/dev/null | wc -l`
- [ ] 检测 credential.helper：`git config --global credential.helper`
- [ ] 检测 .git-credentials：`[ -f ~/.git-credentials ] && echo '存在'`

### 2.2 选择认证方式（向用户展示）
- [ ] 方式 1: **cache** - 内存缓存，安全但临时（容器推荐）
- [ ] 方式 2: **store** - 明文存储，方便但有风险
- [ ] 方式 3: **SSH Key** - 最安全，需预先在 GitCode 网站配置
- [ ] 方式 4: **GITCODE_TOKEN + URL** - 环境变量，适合 CI/CD
- [ ] 方式 5: **libsecret** - 系统加密，Linux 桌面推荐

### 2.3 配置认证
- [ ] 用户选择认证方式后，按 SKILL.md Phase 2.3 配置
- [ ] 验证认证生效：`git push --dry-run origin <branch>`
- [ ] **认证配置成功后才可继续后续步骤**

## Phase 3: 用户确认

- [ ] 已向用户展示完整执行计划表
- [ ] 用户已明确确认（y / 修改参数后确认）
- [ ] 确认项包含：本地仓库、分支名、commit 信息、fork 仓库+分支、目标仓库+分支、PR 标题、PR Body 预览

## Phase 4: 代码准备

- [ ] **同步 upstream**：`git fetch upstream master`
- [ ] **检查分支是否落后**：`git log --oneline HEAD..upstream/master`（输出为空则同步）
- [ ] **若落后则 rebase**：`git rebase upstream/master && git push -f origin <branch>`
- [ ] feat/fix 类型已添加测试用例
- [ ] 代码变更已文档化
- [ ] code-check 警告已修复（参考 `$PYPTO_REPO/docs/contribute/code-check-rule.yaml`）

## Phase 5: PR 规范

- [ ] 标题遵循 `tag(scope): Summary` 格式
- [ ] Tag 为合法类型：feat / fix / docs / style / refactor / test / perf
- [ ] Summary 英文编写、首字母大写、无句号、祈使语气
- [ ] Body 清晰描述变更意图（禁止为空）
- [ ] 单一职责 — 无不相关变更混入

## Phase 6: Commit 规范

- [ ] 每个 commit 只做一件事
- [ ] commit message 格式：`tag(scope): Summary`
- [ ] 多 commit 按推荐顺序：fixup → refactor → feat → test

## Phase 7: 创建或更新 PR

- [ ] 通过 `gitcode_list_pull_requests` 查询是否存在相同分支的 open PR
- [ ] 如存在 → 询问用户：更新现有 PR 还是创建新 PR
- [ ] `head` 使用 `<username>:<branch_name>` 格式（冒号分隔，cross-fork 必须）
- [ ] MCP 参数名全小写（owner, repo, title, head, base, body）
- [ ] `owner`/`repo` 指向上游仓库（`cann/pypto`），不是 fork

## Phase 8: 操作完成

- [ ] 已获得用户确认后才执行
- [ ] PR 创建/更新成功后输出结构化报告（含 PR 链接）
- [ ] 若为更新 PR，确认 PR 编号正确
- [ ] **若 MCP 创建失败**：使用 curl fallback（见 SKILL.md Phase 6.4）
