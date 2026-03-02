#!/usr/bin/env python3
"""GitCode MCP Server 安装与配置验证脚本。

行为约束（按需求实现）：
- 不向用户询问 Token；默认使用占位符 <YOUR_GITCODE_TOKEN>
- 若 ~/.config/opencode/opencode.json 不存在：自动创建为指定模板
- 运行检查前先检测 GITCODE_KEY / GITCODE_TOKEN 是否仍为占位符
- 安装完成后轮询等待用户手动修改 GITCODE_TOKEN；检测到更新后，等待用户确认再执行验证
- 验证方式：调用 GitCode API 列出当前用户仓库列表并打印

提示：此脚本只会遮罩打印 token，不会输出明文。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OPENCODE_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
TOKEN_PLACEHOLDER = "<YOUR_GITCODE_TOKEN>"

DEFAULT_OPENCODE_CONFIG: Dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "mcp": {
        "gitcode": {
            "type": "local",
            "command": ["gitcode-mcp"],
            "enabled": True,
            "environment": {
                "GITCODE_TOKEN": TOKEN_PLACEHOLDER,
                "GITCODE_API_URL": "https://api.gitcode.com/api/v5",
            },
        }
    },
}


PLACEHOLDER_TOKENS = {
    TOKEN_PLACEHOLDER,
    "<your_token>",
    "<your_personal_access_token>",
    "<your_personal_access_token_here>",
    "your_personal_access_token",
    "your_personal_access_token_here",
}


def is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = value.strip()
    if not v:
        return True
    if v in PLACEHOLDER_TOKENS:
        return True
    return "YOUR_GITCODE_TOKEN" in v.upper()


def mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 8:
        return "****"
    return v[:4] + "****" + v[-4:]


def ensure_opencode_config_exists() -> bool:
    """确保 opencode.json 存在；若新建则返回 True。"""
    if OPENCODE_CONFIG_PATH.exists():
        return False
    OPENCODE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENCODE_CONFIG_PATH.write_text(json.dumps(DEFAULT_OPENCODE_CONFIG, ensure_ascii=True, indent=2) + "\n")
    print(f"✅ 已创建 OpenCode 配置: {OPENCODE_CONFIG_PATH}")
    print("⚠️  请把 mcp.gitcode.environment.GITCODE_TOKEN 从占位符改为真实 Token，然后保存。")
    return True


def read_opencode_config() -> Dict[str, Any]:
    return json.loads(OPENCODE_CONFIG_PATH.read_text())


def get_config_token_and_api_url() -> Tuple[Optional[str], str]:
    cfg = read_opencode_config()
    gitcode = cfg.get("mcp", {}).get("gitcode", {})
    env_cfg = gitcode.get("environment", {}) if isinstance(gitcode, dict) else {}
    token = env_cfg.get("GITCODE_TOKEN") if isinstance(env_cfg, dict) else None
    api_url = env_cfg.get("GITCODE_API_URL") if isinstance(env_cfg, dict) else None
    return (token if isinstance(token, str) else None, api_url or "https://api.gitcode.com/api/v5")


@dataclass(frozen=True)
class EffectiveAuth:
    token: Optional[str]
    api_url: str
    source: str


@dataclass(frozen=True)
class ConfigCheckResult:
    """配置检查结果"""
    valid: bool  # 配置是否有效（文件存在且结构正确）
    enabled: bool  # MCP 是否启用
    has_token_key: bool  # 是否包含 GITCODE_TOKEN 键

def get_effective_auth() -> EffectiveAuth:
    """优先级：env GITCODE_TOKEN > env GITCODE_KEY > opencode.json GITCODE_TOKEN"""
    cfg_token, api_url = get_config_token_and_api_url()

    env_token = os.environ.get("GITCODE_TOKEN")
    if env_token and not is_placeholder(env_token):
        return EffectiveAuth(token=env_token, api_url=api_url, source="env:GITCODE_TOKEN")

    env_key = os.environ.get("GITCODE_KEY")
    if env_key and not is_placeholder(env_key):
        return EffectiveAuth(token=env_key, api_url=api_url, source="env:GITCODE_KEY")

    if cfg_token and not is_placeholder(cfg_token):
        return EffectiveAuth(token=cfg_token, api_url=api_url, source=f"file:{OPENCODE_CONFIG_PATH}")

    return EffectiveAuth(token=None, api_url=api_url, source="unset")


def wait_for_token_update(max_wait_seconds: int, poll_seconds: int, require_confirm: bool, no_wait: bool) -> EffectiveAuth:
    auth = get_effective_auth()
    if auth.token:
        return auth

    print("⚠️  未检测到有效 Token（或仍为占位符）。")
    print(f"请编辑 {OPENCODE_CONFIG_PATH} 并将 GITCODE_TOKEN 从 {TOKEN_PLACEHOLDER} 改为真实 Token。")

    if no_wait:
        print("已按 --no-wait 退出（未等待用户修改）。")
        sys.exit(2)

    start = time.time()
    while True:
        auth = get_effective_auth()
        if auth.token:
            print(f"✅ 已检测到 Token 已设置（{auth.source}）：{mask(auth.token)}")
            if require_confirm and sys.stdin.isatty():
                reply = input("请确认已填写正确 Token，输入 y 回车继续验证: ").strip().lower()
                if reply not in {"y", "yes"}:
                    print("已取消验证。")
                    sys.exit(2)
            return auth

        elapsed = int(time.time() - start)
        if elapsed >= max_wait_seconds:
            print("❌ 等待超时：仍未检测到有效 Token。")
            sys.exit(2)

        time.sleep(poll_seconds)


def check_gitcode_mcp_installed() -> bool:
    try:
        result = subprocess.run(["which", "gitcode-mcp"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ gitcode-mcp 已安装: {result.stdout.strip()}")
            return True
    except Exception:
        pass

    print("❌ gitcode-mcp 未安装")
    return False


def check_opencode_config() -> ConfigCheckResult:
    """检查 OpenCode 配置，返回包含 valid/enabled/has_token_key 的结果。"""
    if not OPENCODE_CONFIG_PATH.exists():
        print(f"❌ OpenCode 配置不存在: {OPENCODE_CONFIG_PATH}")
        return ConfigCheckResult(valid=False, enabled=False, has_token_key=False)

    try:
        cfg = read_opencode_config()
        gitcode = cfg.get("mcp", {}).get("gitcode", {})
        if not isinstance(gitcode, dict) or not gitcode:
            print("❌ OpenCode 配置中未找到 mcp.gitcode")
            return ConfigCheckResult(valid=False, enabled=False, has_token_key=False)

        enabled = bool(gitcode.get("enabled", False))
        env_cfg = gitcode.get("environment", {}) if isinstance(gitcode.get("environment", {}), dict) else {}
        has_token_key = "GITCODE_TOKEN" in env_cfg

        icon = "✅" if enabled else "⚠️ "
        print(f"{icon} OpenCode GitCode MCP: enabled={enabled}, has_token_key={has_token_key}")
        return ConfigCheckResult(valid=True, enabled=enabled, has_token_key=has_token_key)
    except Exception as e:
        print(f"❌ 读取 OpenCode 配置失败: {e}")
        return ConfigCheckResult(valid=False, enabled=False, has_token_key=False)


def test_mcp_startup(token: str) -> bool:
    """STDIO 模式：无 stdin 时可能正常退出（exit 0），同样视为启动可用。"""
    try:
        result = subprocess.run(
            ["gitcode-mcp"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "GITCODE_TOKEN": token},
        )
        if result.returncode == 0:
            print("✅ MCP 启动测试通过（STDIO 模式可能正常退出）")
            return True
        print(f"❌ MCP 退出码 {result.returncode}: {(result.stderr or result.stdout)[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  MCP 启动测试：进程运行中（超时未退出，可能正常）")
        return True
    except FileNotFoundError:
        print("❌ gitcode-mcp 命令未找到")
        return False
    except Exception as e:
        print(f"⚠️  启动测试跳过: {e}")
        return False


def fetch_user_repos(api_url: str, token: str, per_page: int) -> List[Dict[str, Any]]:
    base = api_url.rstrip("/")
    url = f"{base}/user/repos?{urlencode({'per_page': min(per_page, 100), 'page': 1})}"

    req = Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "opencode-gitcode-mcp-install-verify/1.0")

    with urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        if not isinstance(data, list):
            raise ValueError(f"unexpected response type: {type(data)}")
        return data


def verify_repo_list(api_url: str, token: str, per_page: int) -> bool:
    try:
        repos = fetch_user_repos(api_url=api_url, token=token, per_page=per_page)
    except HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        print(f"❌ GitCode API 返回 HTTP {e.code}: {msg[:200]}")
        return False
    except URLError as e:
        print(f"❌ 访问 GitCode API 失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 获取仓库列表失败: {e}")
        return False

    print(f"✅ GitCode API 仓库列表获取成功: {len(repos)} 个")
    for r in repos[:10]:
        if not isinstance(r, dict):
            continue
        full_name = r.get("full_name") or r.get("path_with_namespace")
        name = r.get("name")
        web = r.get("html_url") or r.get("web_url")
        label = full_name or name or "<unknown>"
        print(f"- {label}{(' ' + web) if web else ''}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify GitCode MCP install by listing repos")
    p.add_argument("--no-wait", action="store_true", help="token 仍为占位符时不等待，直接退出")
    p.add_argument("--max-wait-seconds", type=int, default=900, help="轮询等待最长秒数")
    p.add_argument("--poll-seconds", type=int, default=3, help="轮询间隔秒数")
    p.add_argument("--no-confirm", action="store_true", help="检测到 token 更新后不要求用户输入 y 确认")
    p.add_argument("--per-page", type=int, default=20, help="仓库列表 per_page（最大 100）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 50)
    print("GitCode MCP Server 安装验证")
    print("=" * 50)
    print()

    ensure_opencode_config_exists()

    # 运行检查前：先检查 GITCODE_KEY / GITCODE_TOKEN 是否是占位符
    auth = get_effective_auth()
    if not auth.token:
        auth = wait_for_token_update(
            max_wait_seconds=args.max_wait_seconds,
            poll_seconds=args.poll_seconds,
            require_confirm=not args.no_confirm,
            no_wait=args.no_wait,
        )

    results: List[Tuple[str, bool]] = []

    print("【安装检查】")
    installed = check_gitcode_mcp_installed()
    results.append(("gitcode-mcp 安装", installed))

    print("\n【配置检查】")
    config_result = check_opencode_config()
    results.append(("OpenCode 配置", config_result.valid))
    # 只有当 MCP 已安装且 enabled=true 时才执行启动测试
    if installed and config_result.enabled:
        print("\n【启动测试】")
        results.append(("MCP 启动测试", test_mcp_startup(auth.token)))
    elif installed and not config_result.enabled:
        print("\n【启动测试】")
        print("⏭️  跳过：MCP 已禁用 (enabled=false)")

    print("\n【功能验证】")
    # 显示 token 来源，便于用户理解验证结果
    if auth.source.startswith("env:"):
        print(f"ℹ️  使用环境变量 token ({auth.source}) 进行 API 验证")
    results.append(("GitCode API 仓库列表", verify_repo_list(auth.api_url, auth.token, per_page=args.per_page)))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")

    print(f"\n通过: {passed}/{total}")
    if passed == total:
        print("\n🎉 验证通过：GitCode MCP 配置可用")
        return 0

    print("\n⚠️  部分检查未通过，请按输出提示修复")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
