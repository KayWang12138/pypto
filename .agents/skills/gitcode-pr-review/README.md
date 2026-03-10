# gitcode-pr-review — Cursor Agent Skill

一个用于 GitCode PR 代码检视的 Cursor Agent Skill，可自动分析 PR diff、生成检视意见，并通过 GitCode API 将行内评论直接提交到 PR 页面。

## 功能

- 自动拉取 PR 变更文件和 diff
- 按 C++/Python 专项标准分析代码问题（必须修改 / 建议修改 / 可选优化）
- 检查与仓库现有代码风格的一致性
- 提交前展示意见清单，由用户选择哪些条目提交
- 将行内评论精确投递到对应文件的具体行号

## 安装

```bash
# 克隆到个人 skills 目录
git clone <本仓库地址> ~/.cursor/skills/gitcode-pr-review
```

## 配置

配置个人 GitCode Token（每人使用自己的 Token，**不要提交到代码仓库**）：

```bash
# 写入 ~/.bashrc，永久生效
echo 'export GITCODE_TOKEN="your_personal_access_token"' >> ~/.bashrc
source ~/.bashrc
```

Token 获取：gitcode.com → 设置 → 私人令牌，勾选 `pull_requests` 权限。

## 使用

在 Cursor 中直接描述任务即可触发：

```
帮我 review 一下 cann/pypto 仓库的 PR #1041，把意见提交到 GitCode
```

Skill 会自动：
1. 获取 PR 信息和 diff
2. 分析代码，生成带编号的意见清单
3. 等你确认选择后，再提交到 GitCode

## 文件说明

| 文件 | 内容 |
|------|------|
| `SKILL.md` | 主流程：工作步骤、API 调用、确认流程 |
| `api-reference.md` | GitCode API 详细参数参考 |
| `review-standards.md` | 代码检视标准（通用 + C++ 专项 + Python 专项） |
