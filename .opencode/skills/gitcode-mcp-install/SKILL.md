---
name: gitcode-mcp-install
description: >-
  安装和配置 GitCode MCP Server，使 AI 客户端能与 GitCode 平台交互（仓库/分支/Issue/PR 管理）。
  触发词：安装 gitcode mcp、配置 gitcode mcp、gitcode mcp server。
source_url: https://gitcode.com/gitcode-ai/gitcode_mcp_server
---

# GitCode MCP Server 安装与配置

## 约定

- `$SKILL_DIR`：由 agent 运行时自动注入的环境变量，指向当前 skill 的根目录（即本 `gitcode-mcp-install/` 目录）。手动执行时需自行设置，例如：`export SKILL_DIR=/path/to/gitcode-mcp-install`。

## ⛔ 隐私保护（强制）

**Token 相关行为绝对禁止：**

| 禁止项 | 说明 |
|--------|------|
| 明文显示 | 输出必须遮罩（如 `abcd****efgh`），仅保留首尾各4位 |
| 非配置存储 | Token 只能存于 `~/.config/opencode/opencode.json` |
| 询问用户 | 不得要求用户提供 Token，由用户自行编辑配置文件 |
| 硬编码 | 文档/代码示例必须用占位符 `<YOUR_GITCODE_TOKEN>` |

---

## 安装

两种安装方式按环境选择。

### 方式一：Go 二进制

```bash
go install gitcode.com/gitcode-ai/gitcode_mcp_server@latest
```

> **注意**：Go 二进制方式在国内网络可能失败（GitCode 不支持 Go module 代理）。
> 若失败请使用 Python 方式。

### 方式二：Python 源码安装

```bash
git clone https://gitcode.com/gitcode-ai/gitcode_mcp_server.git /tmp/gitcode_mcp_server
pip3 install -e /tmp/gitcode_mcp_server
```

> Python >= 3.8
> 
> **重要**：Python 安装方式会自动注册 `gitcode-mcp` 命令，配置方式与 Go 二进制相同。

## OpenCode 配置（不存在会自动创建）

脚本会在 `~/.config/opencode/opencode.json` 不存在时创建默认配置，并把 token 写成占位符。
你只需要在安装完成后把占位符替换成真实 token，**修改后需重启 OpenCode 才能生效**。

模板如下（按需求固定格式）：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "gitcode": {
      "type": "local",
      "command": [
        "gitcode-mcp"
      ],
      "enabled": true,
      "environment": {
        "GITCODE_TOKEN": "<YOUR_GITCODE_TOKEN>",
        "GITCODE_API_URL": "https://api.gitcode.com/api/v5"
      }
    }
  }
}
```

注意：不要在对话里发送或者询问GITCODE_TOKEN，若用户主动提供需要提醒用户存在泄露风险。只需要修改本机文件 `~/.config/opencode/opencode.json`。

## 获取 Token（不在这里索取）

1. 登录 https://gitcode.com → 设置 → 访问令牌
2. 创建 Personal Access Token（建议包含 `repo`、`read:user` 权限）
3. 保存 Token（仅显示一次）

## 验证（会列出仓库列表）

运行验证脚本：

```bash
python3 "$SKILL_DIR/scripts/verify_install.py"
```

验证逻辑：
- 先检测 `GITCODE_TOKEN/GITCODE_KEY` 是否仍为占位符
- 若仍为占位符：脚本会轮询等待你手动修改 `~/.config/opencode/opencode.json`（**修改后需重启 OpenCode 才能生效**）
- 检测到 token 已更新后：脚本会要求你输入 `y` 确认，再继续
- 最终通过调用 `GET https://api.gitcode.com/api/v5/user/repos` 获取仓库列表来验证 token 可用，并打印前 10 个仓库

**可选参数**（用于非交互式场景如 CI/CD）：
- `--no-wait`：token 仍为占位符时不等待，直接退出
- `--no-confirm`：检测到 token 更新后不要求用户输入 y 确认

```bash
# CI/CD 场景示例
python3 "$SKILL_DIR/scripts/verify_install.py" --no-wait --no-confirm
```
## 代理（可选）

如需代理访问 GitCode API，把代理环境变量加入 OpenCode 的 `environment`：

```json
"environment": {
  "GITCODE_TOKEN": "<YOUR_GITCODE_TOKEN>",
  "GITCODE_API_URL": "https://api.gitcode.com/api/v5",
  "HTTP_PROXY": "http://proxy:8080",
  "HTTPS_PROXY": "http://proxy:8080"
}
```

## 故障排查

| 问题 | 处理 |
|------|------|
| `gitcode-mcp: command not found` | 确认 PATH：`which gitcode-mcp` |
| API 401/403 | token 无效/权限不足，更新 `~/.config/opencode/opencode.json` 后再验证 |
| 连接超时 | 检查网络或配置代理 |
| 配置文件不存在 | 运行验证脚本会自动创建 `~/.config/opencode/opencode.json` |

## 外部参考

- 源码: https://gitcode.com/gitcode-ai/gitcode_mcp_server
- MCP 协议: https://modelcontextprotocol.io
- GitCode API: https://api.gitcode.com/api/v5
