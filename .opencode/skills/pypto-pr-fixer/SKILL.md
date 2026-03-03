---
name: pypto-pr-fixer
description: "PyPTO 仓库 PR 自动修复工具。两大能力：(1) 通用 Review 评论修复 — 理解任意人工 review 评论并智能生成修复方案，不限于特定评论类型；(2) CodeCheck CI 修复 — 解析 cann-robot 报告的 CodeArts-Check 静态分析失败，从 openlibing.com 获取违规详情并自动修复。触发词：PR评论修复、review意见修复、PR review fixer、修复PR评论、codecheck修复、codecheck失败。"
---

# PyPTO PR Fixer

PyPTO 仓库 PR 自动修复工具，提供两大能力：


## 输入协议

### 模式 1: PR URL

```
https://gitcode.com/<owner>/<repo>/pull/<number>
https://gitcode.com/<owner>/<repo>/merge_requests/<number>
```

### 模式 2: 三元组

```
owner: cann
repo: pypto
pull_number: 1276
```

### 本地仓库位置

- 默认：当前工作目录（须为 git 仓库）
- pypto 仓库推荐路径：`/workspace/opencode/pypto_repo`

## 核心流程

```
1. 解析 PR 标识 → owner/repo/number
2. 获取 PR 元数据和全部评论
3. 过滤机器人评论，分离人工评论与 CI 报告
4. 对人工评论：通用理解 → 定位文件 → 生成修复方案
5. 对 codecheck 失败：提取报告 URL → 获取违规详情 → 匹配规则修复
6. 用户确认修复方案
7. 应用修复 + 验证
8. 委托 pypto-pr-creator 完成 push + PR 创建
```

## 评论获取与分类

### 获取评论

```python
pr_info = gitcode_get_pull_request(owner, repo, pull_number)
comments = gitcode_list_pull_request_comments(owner, repo, pull_number)
```

### 机器人评论过滤

```python
def is_robot_comment(comment):
    login = comment.get("user", {}).get("login", "").lower()
    if login == "cann-robot":
        return True
    return any(kw in login for kw in ["bot", "robot", "ci", "automation"])
```

### 评论分流

将评论分为两类：

1. **cann-robot CI 报告** — 包含 HTML 表格（`codecheck ❌ FAILED`），走 CodeCheck 修复流程
2. **人工 review 评论** — 其余所有评论，走通用修复流程

## 能力一：通用 Review 评论修复

### 设计原则

**不预设固定分类**。LLM 直接理解评论语义，动态判断修复方案。

### 处理流程

对每条人工评论：

1. **理解意图** — 阅读评论全文，理解 reviewer 要求什么
2. **判断可行性** — 该修改是否可以自动执行？
   - 可自动修复：明确的代码修改指令（改名、移动字段、修复格式、添加缺失内容等）
   - 需人工判断：涉及设计决策、业务逻辑、架构变更等
3. **定位文件** — 通过以下方式定位受影响文件：
   - `diff_comment` 的 `diff_position`（行号信息）
   - 评论中提及的文件路径
   - `grep` 搜索评论中提到的关键内容
4. **生成修复** — 使用 `edit`/`write` 工具直接修复
5. **标记待办** — 无法自动修复的评论输出为结构化 TODO

### 输出格式

```yaml
# 可自动修复
- comment_id: 164054117
  intent: "将 source_url 从顶层移至 metadata 下"
  auto_fixable: true
  files:
    - path/to/SKILL.md
  action: "移动 YAML 字段位置"

# 需人工处理
- comment_id: 164054200
  intent: "重新设计 API 接口"
  auto_fixable: false
  reason: "涉及架构决策"
  suggested_action: "与 reviewer 讨论方案后再修改"
```

### diff_comment 注意事项

GitCode MCP 的 `diff_comment` 包含 `diff_position`（`start_new_line`/`end_new_line`），但**不包含文件 `path`**。定位文件时需结合 grep 搜索评论提及的代码内容。

详见 @references/review-guide.md。

## 能力二：CodeCheck CI 修复

### 触发条件

cann-robot 评论中包含 codecheck 失败的 HTML 表格：

```html
<tr>
  <td><strong>codecheck</strong></td>
  <td>❌ FAILED</td>
  <td><a href="https://www.openlibing.com/apps/entryCheckDashCode/{MR_ID}/{hash}?projectId=300033&codeHostingPlatformFlag=gitcode">>>>>>></a></td>
</tr>
```

### 处理流程

1. **提取报告 URL** — 从 cann-robot 评论 HTML 中解析 `href`
2. **获取违规详情** — 使用 Playwright Python 提取违规列表：
   ```bash
   # 方法一：使用内置脚本（推荐）
   python ${UNIFIED_SKILLS_ROOT}/library/shared/gitcode-pr-review-fixer/scripts/fetch_codecheck_violations.py "$CODECHECK_URL" --output json
   ```
   
   **注意**：openlibing.com 是 SPA，受 WAF 保护，Playwright MCP 不支持 ARM64，必须使用 Playwright Python。详见 @references/codecheck-rules.md。
3. **匹配规则** — 根据规则 ID 在 @references/codecheck-rules.md 中查找修复方案
4. **分类处理**：
   - 可自动修复（格式类 G.FMT、命名类 G.NAM、日志类 G.LOG 等）→ 直接修复
   - 需人工判断（安全类 G.EDV、业务逻辑类 G.CTL 等）→ 生成修复建议
5. **应用修复并验证**

### 提取脚本

内置脚本：`scripts/fetch_codecheck_violations.py`

输出示例：
```json
{
  "total": 79,
  "by_rule": {"G.LOG.02": 25, "G.FMT.02": 22, "G.CLS.11": 13, ...},
  "violations": [
    {"file": "path/to/file.py", "line": 42, "rule_id": "G.FMT.02", ...}
  ]
}
```

### 环境依赖

| 依赖 | 版本 | 安装命令 |
|------|------|----------|
| Python | 3.10+ | 系统自带 |
| playwright | 1.58+ | `pip install playwright` |
| Chromium Headless Shell | v1208 | `playwright install chromium-headless-shell` |

**注意**：ARM64 环境只能用 `chromium-headless-shell`，不能用完整 Chrome。

### CodeCheck 规则参考

完整规则映射见 @references/codecheck-rules.md，包含 111 条 Python 规则的修复方案分类。

## PR 策略选择

获取 PR 信息后，确定修复提交策略：

| 条件 | 策略 | 操作 |
|------|------|------|
| 文件存在于 target_branch | `new_pr` | 从 target_branch 创建新分支，独立 PR |
| 文件仅存在于 source_branch | `append` | 追加到现有 PR 的 source_branch |
| 混合场景 | `append` | 优先追加，避免拆分修复 |

**确认门**：向用户展示策略选择结果，等待确认后继续。

## 修复验证

对修改的 skill 目录运行验证：

```bash
python ${UNIFIED_SKILLS_ROOT}/library/shared/skill-creator/scripts/quick_validate.py <skill_dir>
```

确保输出 "Skill is valid!"。

## 提交与 PR

### Commit Message 格式

```
fix(skills): <summary>

- <change 1>
- <change 2>
```

### 委托 pypto-pr-creator

PR 创建逻辑**完全委托给 pypto-pr-creator 技能**，传递：
- 本地仓库路径
- commit 信息
- PR 标题和描述

### PR 创建后检查

PR 创建成功后检查 CLA 和 LGTM 状态：
- CLA 未签署 → `⚠️ WARNING`（不阻塞提交，仅影响合并）
- LGTM 不足 → `ℹ️ INFO`（合并通常需要 ≥ 2 个 LGTM）

## PR 创建失败诊断

当遇到 `pre receive hook check failed` 时，执行诊断：

| 检查项 | 诊断命令 | 修复建议 |
|--------|----------|----------|
| Commit message 格式 | `git log -1 --format="%s"` | 必须匹配 `tag(scope): Summary` |
| 分支同步状态 | `git log HEAD..origin/<target> --oneline` | `git pull --rebase` |
| 提交者身份 | `git log -1 --format="%ae"` | 配置 `git user.email` |

详见 @references/error-handling.md。

## 限制说明

1. **diff_comment 不含文件路径** — 通过 grep 定位，可能存在误匹配
2. **openlibing.com 为 SPA** — 必须使用 Playwright 渲染，无法 HTTP 直接获取
3. **隐私保护** — 禁止打印 `GITCODE_TOKEN`（屏幕、日志、调试信息均禁止）

## 参考文档

- 通用修复指南 (@references/review-guide.md) — 评论理解与修复策略
- CodeCheck 规则参考 (@references/codecheck-rules.md) — CodeArts-Check 规则映射与修复方案
- 错误处理参考 (@references/error-handling.md) — MCP 错误、平台错误、pre-receive hook 诊断
- pypto-pr-creator (@${UNIFIED_SKILLS_ROOT}/library/shared/pypto-pr-creator/SKILL.md) — PR 创建委托
