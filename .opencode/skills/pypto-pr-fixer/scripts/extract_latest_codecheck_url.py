#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""从 GitCode PR 评论 JSON 中提取最新 codecheck URL。

说明:
- 本脚本不调用任何 GitCode API。
- 输入必须是已有的评论 JSON 数据（通常为数组，每个元素含 body 和 created_at）。

用法示例:
  python3 /tmp/extract_latest_codecheck_url.py --input comments.json
  cat comments.json | python3 /tmp/extract_latest_codecheck_url.py
  python3 /tmp/extract_latest_codecheck_url.py --input comments.json --evidence

退出码:
  0: 成功找到并输出 URL
  1: 输入错误或未找到 URL
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

CODECHECK_URL_RE = re.compile(
    r"https://www\.openlibing\.com/apps/entryCheckDashCode/[^\s'\">]+",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 PR 评论 JSON 中提取最新 codecheck URL（不调用 GitCode API）"
    )
    parser.add_argument(
        "--input",
        help="评论 JSON 文件路径；不传时从 stdin 读取",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="输出完整 JSON 证据链（包含所有匹配的 codecheck URL）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息（comment_id、created_at、url）",
    )
    return parser.parse_args()


def load_json(input_path: str | None) -> Any:
    try:
        if input_path:
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError(f"输入不是合法 JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"读取输入失败: {exc}") from exc


def parse_time(ts: Any) -> datetime:
    if not isinstance(ts, str) or not ts.strip():
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def iter_comments(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for key in ("comments", "data", "items"):
            maybe = data.get(key)
            if isinstance(maybe, list):
                for item in maybe:
                    if isinstance(item, dict):
                        yield item
                return
        raise ValueError("JSON 对象中未找到 comments/data/items 数组")
    else:
        raise ValueError("JSON 顶层必须是数组或对象")


def extract_latest_codecheck(comments: Iterable[dict[str, Any]]) -> tuple[int | None, str | None, str]:
    latest_key: tuple[datetime, int] | None = None
    latest_comment_id: int | None = None
    latest_created_at: str | None = None
    latest_url: str | None = None

    for c in comments:
        body = c.get("body")
        if not isinstance(body, str) or "codecheck" not in body.lower():
            continue

        match = CODECHECK_URL_RE.search(body)
        if not match:
            continue

        created_at_raw = c.get("created_at")
        created_at_dt = parse_time(created_at_raw)
        cid = c.get("id")
        cid_num = cid if isinstance(cid, int) else -1
        key = (created_at_dt, cid_num)

        if latest_key is None or key > latest_key:
            latest_key = key
            latest_comment_id = cid if isinstance(cid, int) else None
            latest_created_at = created_at_raw if isinstance(created_at_raw, str) else None
            latest_url = match.group(0)

    if latest_url is None:
        raise ValueError("未在评论中找到 codecheck URL")

    return latest_comment_id, latest_created_at, latest_url


def extract_with_evidence(comments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """提取所有 codecheck URL 并返回完整证据链。
    
    Returns:
        包含以下结构的字典：
        {
            "total_found": int,  # 共找到多少个 codecheck URL
            "latest": {
                "comment_id": int | None,
                "created_at": str | None,
                "url": str
            },
            "evidence_chain": [
                {
                    "index": int,
                    "comment_id": int | None,
                    "created_at": str | None,
                    "url": str,
                    "is_latest": bool
                },
                ...  # 按时间倒序排列
            ]
        }
    """
    matches: list[tuple[datetime, int, str | None, str]] = []  # (created_at, comment_id, created_at_raw, url)
    
    for c in comments:
        body = c.get("body")
        if not isinstance(body, str) or "codecheck" not in body.lower():
            continue

        match = CODECHECK_URL_RE.search(body)
        if not match:
            continue

        created_at_raw = c.get("created_at")
        created_at_dt = parse_time(created_at_raw)
        cid = c.get("id")
        cid_num = cid if isinstance(cid, int) else -1
        url = match.group(0)
        
        matches.append((created_at_dt, cid_num, created_at_raw if isinstance(created_at_raw, str) else None, url))
    
    if not matches:
        raise ValueError("未在评论中找到 codecheck URL")
    
    # 按时间倒序排序（最新的在前）
    matches.sort(reverse=True, key=lambda x: (x[0], x[1]))
    
    # 构建证据链
    evidence_chain = []
    for idx, (created_at_dt, cid_num, created_at_raw, url) in enumerate(matches):
        evidence_chain.append({
            "index": idx,
            "comment_id": cid_num if cid_num >= 0 else None,
            "created_at": created_at_raw,
            "url": url,
            "is_latest": idx == 0
        })
    
    # 提取最新的
    latest_match = matches[0]
    latest = {
        "comment_id": latest_match[1] if latest_match[1] >= 0 else None,
        "created_at": latest_match[2],
        "url": latest_match[3]
    }
    
    return {
        "total_found": len(matches),
        "latest": latest,
        "evidence_chain": evidence_chain
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        data = load_json(args.input)
        
        if args.evidence:
            # 证据链模式：输出完整 JSON
            result = extract_with_evidence(iter_comments(data))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # 默认模式：仅输出最新 URL
            comment_id, created_at, url = extract_latest_codecheck(iter_comments(data))
            if args.verbose:
                logging.info("comment_id=%s", comment_id)
                logging.info("created_at=%s", created_at)
                logging.info("codecheck_url=%s", url)
            else:
                print(url)
    except ValueError as exc:
        logging.error("ERROR: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
