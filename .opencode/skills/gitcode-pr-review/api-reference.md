# GitCode API 参考

Base URL: `https://gitcode.com/api/v5`

## 认证

所有请求需携带 Token：

```bash
-H "Authorization: Bearer $GITCODE_TOKEN"
```

或通过 query 参数：`?access_token=$GITCODE_TOKEN`

---

## PR 相关接口

### 获取 PR 详情

```
GET /repos/{owner}/{repo}/pulls/{number}
```

关键响应字段：

| 字段 | 说明 |
|------|------|
| `head.sha` | PR 最新 commit SHA（提交评论必须） |
| `base.sha` | 目标分支 SHA |
| `state` | PR 状态：open / closed / merged |
| `title` | PR 标题 |
| `body` | PR 描述 |

---

### 获取 PR 变更文件列表

```
GET /repos/{owner}/{repo}/pulls/{number}/files
```

关键响应字段（数组）：

| 字段 | 说明 |
|------|------|
| `filename` | 文件相对路径（提交评论时用） |
| `status` | 变更类型：added / modified / removed |
| `patch` | 文件的 diff 内容 |
| `additions` | 新增行数 |
| `deletions` | 删除行数 |

---

### 提交行内评论

```
POST /repos/{owner}/{repo}/pulls/{number}/comments
```

请求体（JSON）：

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `body` | string | 是 | 评论内容 |
| `commit_id` | string | 是 | PR 最新 commit SHA（head.sha） |
| `path` | string | 是 | 文件相对路径，需与文件列表中 filename 一致 |
| `position` | integer | 是 | **新文件中的实际行号**（不是 diff 位置计数器） |

**position 正确计算方法**：

`position` = 目标行在**变更后新文件**中的行号。从 diff hunk 头 `@@ -A,B +C,D @@` 中的 `C`（新文件起始行号）开始，逐行累加（跳过删除行 `-`）。

```diff
@@ -10,6 +10,7 @@ class Foo {       <- 新文件从第 10 行开始
 int a;                               <- 新文件第 10 行 → position=10
 int b;                               <- 新文件第 11 行 → position=11
-int c;                               <- 删除行，新文件无此行，不可评论
+int c = 0;                           <- 新文件第 12 行 → position=12（可评论）
+int d = 0;                           <- 新文件第 13 行 → position=13（可评论）
};                                    <- 新文件第 14 行 → position=14
```

**常见错误处理**：

| HTTP 状态码 | 原因 | 解决方法 |
|-------------|------|---------|
| 422 | position 不在变更范围内或指向删除行 | 重新检查新文件行号 |
| 401 | Token 无效或过期 | 检查 GITCODE_TOKEN 环境变量 |
| 404 | 仓库或 PR 不存在 | 确认 owner/repo/number 正确 |

---

### 删除行内评论

```
DELETE /repos/{owner}/{repo}/pulls/comments/{comment_id}
```

> ⚠️ 注意：删除端点是 `/pulls/comments/{id}`（无 PR number），而非 `/pulls/{number}/comments`。

---

### 获取已有评论

```
GET /repos/{owner}/{repo}/pulls/{number}/comments
```

响应为评论对象数组，每条含：`id`、`body`、`path`、`position`、`user.login`

---

### 回复评论

```
POST /repos/{owner}/{repo}/pulls/{number}/discussions/{discussion_id}/comments
```

请求体：

```json
{
  "body": "回复内容"
}
```

---

### 提交整体审查

```
POST /repos/{owner}/{repo}/pulls/{number}/review
```

请求体：

| 字段 | 类型 | 说明 |
|------|------|------|
| `body` | string | 整体审查意见 |
| `action` | string | `comment` / `approve` / `request_changes` |

---

## Token 获取方式

1. 登录 https://gitcode.com
2. 进入「设置」→「私人令牌」（Personal Access Token）
3. 创建新令牌，勾选 `projects` 和 `pull_requests` 权限
4. 将令牌保存到环境变量：`export GITCODE_TOKEN="glpat-xxxx"`