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

### 归档issue
```bash
python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues
```

### 分析issue（归档后由AI分析）
归档完成后，AI 会询问是否需要分析 Bug-Report Issue。如果选择分析，AI 会：
1. 读取归档的 issue 文件
2. 筛选 Bug-Report 类型的 issue
3. 分析提取不支持场景、精度问题等
4. 生成报告到项目根目录 `pypto_unsupported_scenarios.md`

**分析能力由大模型提供，无需额外脚本。**

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
5. **归档完成后提示用户可以分析 issue**

**增量更新**: 仅处理新增、状态变化或时间戳变化的 issue，跳过未变更项。

**错误处理**: 404 标记为已删除，其他错误记录后继续。

---

## 分析流程

归档完成后会询问：**是否分析 Bug-Report Issue？**

如果选择分析，AI 会按照 [docs/issue_analysis_workflow.md](docs/issue_analysis_workflow.md) 执行详细分析：

### 核心要求

1. **不要臆想**：严格按照 issue 实际提到的规避方式描述
2. **图片处理**：使用 OCR 技术解析 issue 中的图片内容
3. **引用来源**：每个结论都标注来源（Issue #XXX）
4. **状态标注**：区分 closed / open / resolved（已修复无需规避）
5. **问题筛选**：只关注 PyPTO 写法/Pass/Machine 问题，过滤环境类问题
6. **质量优先**：宁缺毋滥，只包含有明确触发条件的场景

### 分析步骤

```
1. 读取 archive_record.json，筛选 Bug-Report issue
2. 应用问题领域筛选规则：
   ├─ ✅ 保留 PyPTO 写法/Pass/Machine 问题
   └─ ❌ 过滤环境类问题（安装/编译/硬件/文档）
3. 逐个分析保留的 issue：
   ├─ 读取 issue-{num}.md
   ├─ 处理图片（OCR提取内容）
   ├─ 提取：现象、触发条件、根因、规避方案
   └─ 按类型分类（编译/运行时/精度/性能/功能缺失）
4. 生成两个独立文件：
   ├─ pypto_unsupported_scenarios.md（只包含有明确触发条件的场景）
   └─ pypto_issue_index.md（完整的问题索引）
```

### 输出文件

**文件一：`pypto_unsupported_scenarios.md`**
- **位置**：`.agents/skills/gitcode-issue-archiver/docs/`
- **用途**：开发前必读
- **内容**：明确的不支持场景清单
- **特点**：
  - 只包含有明确触发条件的场景
  - 触发条件具体、可操作
  - 规避方案明确
  - 宁缺毋滥，不记流水账
  - 不包含附录

**文件二：`pypto_issue_index.md`**
- **位置**：`.agents/skills/gitcode-issue-archiver/docs/`
- **用途**：遇到精度问题时检索
- **内容**：按类型分类的问题索引
- **特点**：
  - 包含所有保留的 issue
  - 按 A-G 类型分类
  - 便于快速检索

**详细流程**：见 [docs/issue_analysis_workflow.md](docs/issue_analysis_workflow.md)

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
