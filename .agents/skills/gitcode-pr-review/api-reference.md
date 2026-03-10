# GitCode PR Review API 参考

本文档汇总 `gitcode-pr-review` skill 涉及的 GitCode API 接口，便于快速查阅和排错。

## 1. 获取 PR 详情

- **Method**: `GET`
- **URL**: `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}`
- **Headers**:
  - `Authorization: Bearer $GITCODE_TOKEN`
- **用途**:
  - 获取 `head.sha`（提交评论时必填的 `commit_id`）
  - 获取 PR 的基本信息（标题、作者、状态等）。

## 2. 获取 PR 变更文件列表

- **Method**: `GET`
- **URL**: `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/files`
- **Headers**:
  - `Authorization: Bearer $GITCODE_TOKEN`
- **核心字段**:
  - `filename`: 变更文件相对路径
  - `patch`: 该文件的 unified diff 内容，用于分析每一处修改。

## 3. 创建行内评论

- **Method**: `POST`
- **URL**: `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments`
- **Headers**:
  - `Authorization: Bearer $GITCODE_TOKEN`
  - `Content-Type: application/json`
- **Body 字段**:
  - `body` (`string`): 评论正文
  - `commit_id` (`string`): 来自 PR 的 `head.sha`
  - `path` (`string`): 文件相对路径（如 `framework/xxx.cpp`）
  - `position` (`number`): **新文件中的实际行号**
- **常见错误**:
  - `422 Unprocessable Entity`: 通常是 `position` 不正确，需要根据 diff 重新计算新文件行号。

## 4. 创建整体评论（非行内）

- **Method**: `POST`
- **URL**: `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments`
- **Headers**:
  - `Authorization: Bearer $GITCODE_TOKEN`
  - `Content-Type: application/json`
- **Body 字段**:
  - `body` (`string`): 整体审查意见
  - `commit_id` (`string`): 来自 PR 的 `head.sha`
- **说明**:
  - 不携带 `path` 与 `position`，等价于在 PR 讨论区里追加一条总评。

## 5. 列出 PR 评论

- **Method**: `GET`
- **URL**: `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments`
- **Query 参数**:
  - `page`: 页码（从 1 开始）
  - `per_page`: 每页条数
- **用途**:
  - 排查自己之前提交的评论（包括行内评论）
  - 获取评论的 `id`，为后续删除做准备。

## 6. 删除评论

- **Method**: `DELETE`
- **URL**: `https://gitcode.com/api/v5/repos/cann/pypto/pulls/comments/{note_id}`
- **Headers**:
  - `Authorization: Bearer $GITCODE_TOKEN`
- **注意事项**:
  - 使用的是评论的 **数字 `id`**（如 `164610876`），不是其它字符串字段。

---

## 调试建议

- 调用前先用 `echo $GITCODE_TOKEN` 检查环境变量是否配置正确。
- 使用 `-v` 或保存响应 body，便于排查 4xx/5xx 错误。
- 对于 422 错误，优先检查 `position`、`commit_id`、`path` 三个字段是否正确。

