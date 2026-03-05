---
name: gitcode-pr-review
description: 对 GitCode 平台（gitcode.com）上的 pypto 代码仓 Pull Request 进行代码检视，并通过 GitCode API 将行内评论直接提交到 PR 页面。当用户要求检视 PR、review 代码、提交检视意见时使用本 skill。
---

# GitCode PR 代码检视

## 前提条件

使用前需配置个人 GitCode Token（每人配置自己的，Token 不进入代码仓库）：

```bash
# 添加到 ~/.bashrc（一次性配置，永久生效）
echo 'export GITCODE_TOKEN="your_personal_access_token"' >> ~/.bashrc
source ~/.bashrc

# 验证是否生效
echo $GITCODE_TOKEN
```

Token 获取方式：gitcode.com → 设置 → 私人令牌，勾选 `pull_requests` 权限。

默认仓库信息（pypto 项目）：
- owner: 仓库组织名/用户名
- repo: `pypto`
- base_url: `https://gitcode.com/api/v5`

## 工作流程

```
Task Progress:
- [ ] Step 1: 获取 PR 基本信息和文件列表
- [ ] Step 2: 分析 diff，生成检视意见
- [ ] Step 3: 向用户展示意见清单，等待确认
- [ ] Step 4: 按用户选择提交行内评论
- [ ] Step 5: 提交整体审查结论（可选）
```

### Step 1: 获取 PR 信息

```bash
# 获取 PR 详情（含 head.sha）
curl -s -H "Authorization: Bearer $GITCODE_TOKEN" \
  "https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}"

# 获取 PR 变更文件列表（含每个文件的 patch/diff）
curl -s -H "Authorization: Bearer $GITCODE_TOKEN" \
  "https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/files"
```

关键字段：
- `head.sha`：PR 最新 commit SHA，提交评论时必须
- 文件列表中的 `filename`（文件路径）和 `patch`（diff 内容）

### Step 2: 分析 diff，生成检视意见

解析每个文件的 `patch` 字段，重点检查：

1. **正确性**：逻辑错误、边界条件、空指针/空引用
2. **性能**：不必要的循环、内存分配、算法复杂度
3. **安全性**：输入验证、资源泄漏
4. **可读性**：命名规范、注释完整性
5. **C++ 规范**（framework/ 目录）：RAII、智能指针使用、const 正确性
6. **Python 规范**（python/ 和 models/ 目录）：类型注解、异常处理
7. **仓库风格一致性**：与当前仓库中同类代码的风格保持一致（命名惯例、文件结构、错误处理模式、日志输出方式等），参见 review-standards.md 中的"仓库风格一致性"章节

**position 计算规则**：
- diff 中每一行（`@@` 标记行、上下文行 ` `、新增行 `+`、删除行 `-`）各占一个 position，从 1 开始
- 只能对 `+`（新增）行或上下文行（` `）添加 inline comment
- 不能对 `-`（删除）行添加 inline comment

### Step 3: 向用户展示意见清单，等待确认

**在提交任何评论之前，必须先以清单形式向用户展示所有检视意见，并等待用户确认。**

展示格式如下（每条一行，编号）：

```
共发现 N 条检视意见，请选择要提交到 GitCode 的条目（可输入编号，如 "1 3 5"，或 "全部"/"all"，或 "跳过"/"none"）：

[1] 🔴 monitor_manager.cpp (pos=52)：持锁调用 impl_->Stop() 导致死锁
[2] 🔴 monitor_manager.cpp (pos=108)：setenv() 多线程不安全
[3] 🟡 monitor_impl.cpp (pos=77)：快照不一致
[4] 🟢 monitor_config.h (pos=16)：MonitorConfig 未使用
...
```

**等待用户回复**，根据用户选择决定提交哪些条目：
- 用户输入 `全部` 或 `all`：提交所有条目
- 用户输入编号（如 `1 2 5`）：只提交对应编号的条目
- 用户输入 `跳过` 或 `none`：不提交任何行内评论
- 用户可以在自然语言中描述，如"只提交红色的"、"不要提交可选优化"，需据此过滤

### Step 4: 按用户选择提交行内评论

仅提交用户确认的条目：

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GITCODE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "body": "检视意见内容",
    "commit_id": "{head_sha}",
    "path": "相对文件路径",
    "position": 数字（diff 中的行位置）
  }' \
  "https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments"
```

返回 422 时说明 position 错误，需重新检查 diff 行计数。

### Step 5: 提交整体审查结论（可选）

**提交整体总结前同样需要询问用户是否需要**，并询问审查结论是 `comment`（仅评论）还是 `request_changes`（要求修改）。

整体总结通过不带 `position`/`path` 的评论接口提交（`review` 接口需要特殊权限，优先用 `comments` 接口）：

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GITCODE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "body": "整体审查意见",
    "commit_id": "{head_sha}"
  }' \
  "https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments"
```

## 评论格式规范

每条行内评论使用以下格式：

```
【问题级别】简短标题

具体描述原因和潜在影响。

建议：`推荐的修改方式`
```

问题级别：
- `🔴 必须修改`：影响功能、安全或编译
- `🟡 建议修改`：影响可维护性或性能
- `🟢 可选优化`：风格或可读性改进

## 注意事项

- **在提交任何内容到 GitCode 之前，必须先展示意见清单并等待用户确认**，这是不可跳过的步骤
- Token 未设置时，引导用户执行 `echo 'export GITCODE_TOKEN="..."' >> ~/.bashrc && source ~/.bashrc`
- 一次最多提交 20 条行内评论，优先提交高优先级问题
- 详细 API 参数见 [api-reference.md](api-reference.md)
- 代码检视标准见 [review-standards.md](review-standards.md)