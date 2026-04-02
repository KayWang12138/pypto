---
name: gitcode-issue-archiver
description: "GitCode issue archiver for fetching and storing issues locally using MCP tools. Use when archiving GitCode repository issues with comments, tracking state changes, and maintaining local sync. Supports incremental updates, deletion handling, and markdown export. Read-only operations - never modifies remote repository. **MUST prompt user for config when config.json doesn't exist - never auto-generate default config**."
---

# GitCode Issue Archiver

归档 GitCode 仓库 issue 到本地 markdown 文件。**只读操作，不修改远程仓库。**

## 关键规则

1. **无配置时必须询问** — 当 `config.json` 不存在时，询问用户仓库路径 (`owner/repo`)、归档目录、Token（可选）。绝不自动生成默认配置。
2. **有配置时直接使用** — 配置文件存在时，使用已保存的 `repo_path` 和 `last_archive_dir`，不再询问。
3. **时间戳自动追加** — 每次归档自动在路径后追加 `YYYYMMDD_HHMMSS` 子目录。

## 用法

```bash
python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues
# → /workspace/archive/pypto_issues/

python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues --token YOUR_TOKEN
```

后续运行可省略参数，自动读取 `config.json`。

## 配置

**文件位置**: `{skill_dir}/config.json`

```json
{
  "repo_path": "cann/pypto",
  "last_archive_dir": "/workspace/archive/pypto_issues/20260204_091547"
}
```

- `repo_path`: 仓库路径 (`owner/repo`)
- `last_archive_dir`: 上次归档目录
- 命令行参数覆盖配置文件值

## 归档流程

使用 GitCode MCP 工具（`gitcode_list_issues`、`gitcode_get_issue`、`gitcode_list_issue_comments`）：

1. 列出所有 issue，获取 `max_issue_number`
2. 遍历 `1..max_issue_number`，逐个获取 issue 和评论
3. 生成 markdown 保存为 `issue-{number}.md`
4. 更新 `archive_record.json` 记录状态

**增量更新**: 仅处理新增、状态变化或时间戳变化的 issue，跳过未变更项。

**错误处理**: 404 标记为已删除，其他错误记录后继续。

## 输出结构

```
archive_dir/
├── archive_record.json
├── issue-1.md
├── issue-2.md
└── ...
```

每个 markdown 包含：标题、元数据（状态/作者/时间/URL）、标签、里程碑、描述、所有评论。

**Record 文件格式**: 见 [references/record_schema.md](references/record_schema.md)

## 脚本

- `scripts/archive_issues_mcp.py` — 主归档脚本，支持重试和分页
- `scripts/test_archive_issues.py` — 单元测试
