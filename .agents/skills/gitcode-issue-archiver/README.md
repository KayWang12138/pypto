# GitCode Issue Archiver & Analyzer

归档 GitCode 仓库 issue 并分析 Bug-Report 问题，生成 PyPTO 不支持场景报告。

---

## 功能特性

### 1. Issue 归档
- ✅ 增量更新：仅处理新增或变化的 issue
- ✅ 完整记录：issue 描述、评论、图片
- ✅ 状态追踪：记录 issue 状态变化
- ✅ 只读操作：不修改远程仓库

### 2. Bug-Report 分析
- ✅ 自动识别：筛选 Bug-Report 类型 issue
- ✅ 图片处理：OCR 提取图片中的关键信息
- ✅ 智能分类：按问题类型分类整理
- ✅ 生成报告：输出不支持场景清单和问题索引

---

## 快速开始

### 步骤 1：归档 Issue

```bash
cd .agents/skills/gitcode-issue-archiver
python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues
```

### 步骤 2：分析 Bug-Report Issue

归档完成后，脚本会询问：
```
是否分析归档的 Bug-Report Issue? (y/n):
```

输入 `y` 或 `yes` 开始分析。

### 步骤 3：查看报告

分析完成后，报告生成在项目根目录：
```
pypto_unsupported_scenarios.md
```

---

## 文档结构

```
gitcode-issue-archiver/
├── SKILL.md                          # Skill 主文档
├── README.md                          # 本文档
├── config.json                        # 配置文件（自动生成）
├── scripts/
│   ├── archive_issues_mcp.py         # 归档脚本
│   └── test_archive_issues.py        # 测试脚本
└── docs/
    ├── issue_analysis_workflow.md    # 详细分析流程
    └── record_schema.md              # 记录格式说明
```

---

## 使用场景

### 场景 1：首次归档
```bash
python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues
# 等待归档完成
# 选择 y 开始分析
```

### 场景 2：增量更新
```bash
# 配置文件已保存，直接运行
python scripts/archive_issues_mcp.py
# 仅更新新增或变化的 issue
```

### 场景 3：仅归档不分析
```bash
python scripts/archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues
# 归档完成后选择 n
```

### 场景 4：手动分析已有归档
如果已有归档数据，可以请求 AI 手动分析：
```
请分析 /workspace/archive/pypto_issues 下的 Bug-Report Issue
```

AI 会按照 `docs/issue_analysis_workflow.md` 执行分析。

---

## 配置说明

### config.json

首次运行会自动生成配置文件：

```json
{
  "repo_path": "cann/pypto",
  "last_archive_dir": "/workspace/archive/pypto_issues/20260330_143000"
}
```

- `repo_path`: GitCode 仓库路径（格式：owner/repo）
- `last_archive_dir`: 上次归档目录

---

## 分析报告说明

### 报告结构

```markdown
# PyPTO 不支持场景与问题检索目录

## 第一部分：明确的不支持场景清单
（可直接规避的已知问题）

### 1. 内存连续性 - 合轴优化
- 触发条件：...
- 规避方案：...
- 状态：仍需规避
- 来源：Issue #108

## 第二部分：问题现象索引
（按类型分类的所有问题）

### A. 编译错误类
### B. 运行时错误类
### C. 精度问题类
...
```

### 使用方法

1. **开发前检查**：查看第一部分，了解已知限制
2. **遇到问题时**：在第二部分搜索相似问题
3. **快速定位**：根据 issue 编号查看原始 issue

---

## 集成到其他 Skill

### pypto-op-workflow
- Stage 0：开发前查看 `pypto_unsupported_scenarios.md`
- 避免使用已知不支持的写法

### pypto-precision-debugger
- 步骤 -1：精度问题优先查看不支持场景
- 快速匹配已知问题和规避方案

---

## 常见问题

### Q1: 报告文件在哪里？
**A**: 报告生成在项目根目录（当前工作目录），文件名为 `pypto_unsupported_scenarios.md`

### Q2: 如何更新报告？
**A**: 重新运行归档和分析流程即可。建议定期更新以获取最新的 issue 信息。

### Q3: 分析会处理图片吗？
**A**: 会。AI 会使用 OCR 技术提取 issue 图片中的错误信息、代码、日志等内容。

### Q4: 只想分析，不想重新归档怎么办？
**A**: 直接请求 AI 分析已有归档目录即可：
```
请分析 /workspace/archive/pypto_issues 下的 Bug-Report Issue
```

### Q5: 分析结果不准确怎么办？
**A**: 查看 `docs/issue_analysis_workflow.md` 了解分析流程，可以手动调整或补充。

---

## 参考文档

- [详细分析流程](docs/issue_analysis_workflow.md)
- [记录格式说明](references/record_schema.md)
- [Skill 主文档](SKILL.md)

---

**维护者**: PyPTO Team  
**更新日期**: 2026-03-30