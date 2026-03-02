---
name: gitcode-mcp-install
description: >-
  安装和配置 GitCode MCP Server，使 AI 客户端能与 GitCode 平台交互（仓库/分支/Issue/PR 管理）。
  触发词：安装 gitcode mcp、配置 gitcode mcp、gitcode mcp server。
metadata:
  source_url: https://gitcode.com/gitcode-ai/gitcode_mcp_server
  tags:
    - GitCode
    - MCP
    - 安装配置
---

# GitCode MCP Server 安装与配置

## 约定

- `$SKILL_DIR`：由 agent 运行时自动注入的环境变量，指向当前 skill 的根目录（即本 `gitcode-mcp-install/` 目录）。手动执行时需自行设置，例如：`export SKILL_DIR=/path/to/gitcode-mcp-install`。

## ⛔ 隐私保护（强制）

**Token 相关行为绝对禁止：**

| 禁止项 | 说明 |
|--------|------|
| 明文显示 | 输出必须遮罩（如 `abcd****efgh`），仅保留首尾各4位 |
| 非配置存储 | Token 只能存于 `~/.config/opencode/opencode.json` 或环境变量 |
| 询问用户 | 不得要求用户提供 Token，由用户自行编辑配置文件 |
| 硬编码 | 文档/代码示例必须用占位符 `<YOUR_GITCODE_TOKEN>` |
| **打印环境变量** | ⚠️ 禁止在任何输出中打印 `GITCODE_TOKEN` 或 `GITCODE_KEY`，包括调试信息 |

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

## 验证

**手动验证**：

```bash
# 1. 检查命令是否存在
which gitcode-mcp

# 2. 检查配置文件
cat ~/.config/opencode/opencode.json | grep -A5 gitcode

# 3. 测试 API 连接（替换 <YOUR_TOKEN> 为真实 token）
curl -s "https://api.gitcode.com/api/v5/user/repos?access_token=<YOUR_TOKEN>&per_page=5" | jq '.[].full_name'
```

**验证要点**：
- `which gitcode-mcp` 返回路径
- 配置文件中 `GITCODE_TOKEN` 不是占位符
- API 调用返回仓库名称列表

> **注意**：修改 `~/.config/opencode/opencode.json` 后需重启 OpenCode 才能生效。

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
| API 401/403 | token 无效/权限不足，更新 `~/.config/opencode/opencode.json` |
| 连接超时 | 检查网络或配置代理 |
| 配置文件不存在 | 手动创建 `~/.config/opencode/opencode.json` |

## 外部参考

- 源码: https://gitcode.com/gitcode-ai/gitcode_mcp_server
- MCP 协议: https://modelcontextprotocol.io
- GitCode API: https://api.gitcode.com/api/v5
