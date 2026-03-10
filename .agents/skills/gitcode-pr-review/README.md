# gitcode-pr-review Skill 说明

`gitcode-pr-review` 是用于 **在 GitCode 平台上对 PyPTO 仓的 Pull Request 进行代码检视** 的辅助 Skill，支持：

- 自动获取 PR 的基本信息和变更文件列表；
- 解析 diff，按「必须修改 / 建议修改 / 可选优化」分级输出检视意见；
- 在向用户展示完整检视清单并获得确认后，通过 GitCode API 提交行内评论和整体审查意见；
- 支持查找并删除自己之前提交的错误评论（如 position 计算错误）。

## 目录结构

- `SKILL.md`：Skill 的总体说明、使用流程和评论格式规范；
- `api-reference.md`：与本 Skill 相关的 GitCode API 接口参考；
- `review-standards.md`：在 PyPTO 仓做代码检视时的检查重点与分级标准（C++ / Python / 通用风格等）。

## 使用前提

1. 已在本机配置 GitCode 个人访问令牌：
   - 在 `~/.bashrc` 中设置 `GITCODE_TOKEN` 环境变量；
   - Token 需要至少包含 `pull_requests` 权限；
2. 已确认当前检视的仓库为 PyPTO 相关仓库（如 `CANN/pypto` 或其 fork）。

更多细节请参考同目录下的 `SKILL.md`、`api-reference.md` 和 `review-standards.md`。

