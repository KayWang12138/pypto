#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""
GitCode Issue Archiver (MCP Version)

Archives GitCode repository issues to local markdown files using MCP tools.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
import subprocess
import urllib.parse

CONFIG_FILE = Path(__file__).parent.parent / "config.json"
MAX_RETRIES = 3
NOT_FOUND_RETRY_DELAY_SEC = 1

logger = logging.getLogger(__name__)


class GitCodeMCPArchiver:
    """Archive GitCode repository issues using MCP tools."""

    def __init__(self, repo_path: str, archive_dir: str, start: Optional[int] = None, end: Optional[int] = None):
        self.repo_path = repo_path
        self.archive_dir = Path(archive_dir)
        self.start = start
        self.end = end

        parts = repo_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repo_path format: {repo_path}. Expected 'owner/repo'")

        self.owner, self.repo = parts
        self.record_file = self.archive_dir / "archive_record.json"

    @staticmethod
    def load_config() -> Dict[str, Optional[str]]:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return {
                        "last_archive_dir": config.get("last_archive_dir"),
                        "repo_path": config.get("repo_path"),
                    }
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load config file: %s", e)
        return {"last_archive_dir": None, "repo_path": None}

    @staticmethod
    def save_config(archive_dir: str, repo_path: str):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {"last_archive_dir": archive_dir, "repo_path": repo_path}
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _extract_asset_urls(markdown: str) -> List[str]:
        pattern = re.compile(r"https?://\S+?\.(?:png|jpg|jpeg|gif|webp)(?:\?\S*)?", re.IGNORECASE)
        urls = pattern.findall(markdown)
        seen = set()
        out: List[str] = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    @staticmethod
    def _request_with_retry(method: str, url: str, params: Optional[Dict[str, Any]] = None,
                            headers: Optional[Dict[str, str]] = None) -> Tuple[Any, Optional[int]]:
        """HTTP request with exponential backoff retry."""
        try:
            import requests
        except ImportError:
            logger.error("requests library is required. Install with: pip install requests")
            return None, -1

        if headers is None:
            headers = {}

        def _looks_like_not_found(resp: "requests.Response") -> bool:
            if resp.status_code == 404:
                return True
            if resp.status_code != 400:
                return False
            try:
                payload = resp.json()
            except Exception:
                return False
            if not isinstance(payload, dict):
                return False
            if payload.get("error_code") == 404:
                return True
            msg = str(payload.get("error_message", "")).lower()
            return "not found" in msg

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)

                if _looks_like_not_found(response):
                    if attempt == 0:
                        time.sleep(NOT_FOUND_RETRY_DELAY_SEC)
                        continue
                    return None, 404

                response.raise_for_status()
                return response.json(), None

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if 400 <= status_code < 500 and status_code != 429:
                    return None, status_code

                logger.warning("HTTP Error (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.info("  Retrying in %ss...", wait_time)
                    time.sleep(wait_time)
                else:
                    return None, status_code

            except requests.exceptions.RequestException as e:
                logger.warning("Request Error (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.info("  Retrying in %ss...", wait_time)
                    time.sleep(wait_time)
                else:
                    return None, None

        return None, None

    @staticmethod
    def _get_headers() -> Dict[str, str]:
        headers = {}
        token = os.environ.get("GITCODE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _generate_markdown(issue: Dict[str, Any], comments: List[Dict[str, Any]]) -> str:
        lines = []

        lines.append(f"# Issue #{issue['number']}: {issue['title']}")
        lines.append("")
        lines.append(f"**State**: {issue.get('state', 'unknown')}")
        lines.append(f"**Author**: {issue.get('user', {}).get('login', 'unknown')}")

        created_at = issue.get('created_at', '')
        updated_at = issue.get('updated_at', '')
        lines.append(f"**Created**: {created_at}")
        lines.append(f"**Updated**: {updated_at}")

        html_url = issue.get('html_url', '')
        if html_url:
            lines.append(f"**URL**: {html_url}")

        assignee = issue.get('assignee')
        if assignee:
            lines.append(f"**Assignee**: {assignee.get('login', 'unknown')}")

        lines.append("")

        labels = issue.get('labels', [])
        if labels:
            lines.append("## Labels")
            for label in labels:
                lines.append(f"- {label.get('name', '')} ({label.get('color', '#000000')})")
            lines.append("")

        milestone = issue.get('milestone')
        if milestone:
            lines.append(f"**Milestone**: {milestone.get('title', '')}")
            lines.append("")

        body = issue.get('body', '')
        if body:
            lines.append("## Description")
            lines.append(body)
            lines.append("")

        if comments:
            comments.sort(key=lambda c: c.get('created_at', ''))
            lines.append(f"## Comments ({len(comments)})")
            lines.append("")

            for comment in comments:
                user = comment.get('user', {}).get('login', 'unknown')
                created = comment.get('created_at', '')
                body_text = comment.get('body', '')
                lines.append(f"### {user} - {created}")
                lines.append(body_text)
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _needs_update(record: Dict[str, Any], issue_number: int, issue: Dict[str, Any]) -> bool:
        issue_record = record["issues"].get(str(issue_number))

        if not issue_record:
            return True

        if not issue_record.get("exists", True):
            return True

        old_state = issue_record.get("state")
        new_state = issue.get("state")
        if old_state != new_state:
            return True

        old_updated = issue_record.get("updated_at", "")
        new_updated = issue.get("updated_at", "")
        if old_updated != new_updated:
            return True

        return False

    def archive(self):
        logger.info("Archiving issues from %s using MCP tools...", self.repo_path)
        logger.info("Archive directory: %s", self.archive_dir)

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        record = self._load_record()

        logger.info("Fetching issue list using gitcode_list_issues MCP tool...")
        all_issues = self._get_all_issues()

        if not all_issues:
            logger.info("No issues found in repository.")
            return

        max_issue_number = int(all_issues[0]["number"])
        logger.info("Max issue number from list: %s", max_issue_number)

        if max_issue_number > record["max_issue_number"]:
            record["max_issue_number"] = max_issue_number

        processed = 0
        updated = 0
        skipped = 0
        deleted = 0
        errors = 0

        start_num = self.start if self.start is not None else 1
        end_num = self.end if self.end is not None else max_issue_number

        total_range = end_num - start_num + 1

        for issue_number in range(start_num, end_num + 1):
            processed += 1

            if processed % 10 == 0 or issue_number == end_num:
                logger.info("Processing %s/%s issues...", processed, total_range)

            issue, error_status = self._get_issue(issue_number)

            if error_status == 404:
                issue_record = record["issues"].get(str(issue_number), {})
                if issue_record.get("exists", True):
                    record["issues"][str(issue_number)] = {
                        "exists": False,
                        "reason": "deleted",
                        "last_checked": datetime.utcnow().isoformat() + "Z"
                    }
                    deleted += 1
                continue

            if error_status:
                errors += 1
                logger.warning("Issue #%s: Failed with status %s", issue_number, error_status)
                continue

            if not issue:
                continue

            if not GitCodeMCPArchiver._needs_update(record, issue_number, issue):
                skipped += 1
                continue

            comments = self._get_issue_comments(issue_number)

            markdown = GitCodeMCPArchiver._generate_markdown(issue, comments)
            markdown = self._download_issue_assets_and_rewrite_markdown(issue_number, markdown)
            self._save_issue_markdown(issue_number, markdown)

            record["issues"][str(issue_number)] = {
                "exists": True,
                "state": issue.get("state"),
                "title": issue.get("title", ""),
                "updated_at": issue.get("updated_at", ""),
                "comment_count": len(comments)
            }

            updated += 1
            logger.info("Issue #%s: %s - %s", issue_number, issue.get("state"), issue.get("title", "")[:50])

        self._save_record(record)

        logger.info("=" * 50)
        logger.info("Archive Summary")
        logger.info("=" * 50)
        logger.info("Total processed: %s", processed)
        logger.info("Updated:        %s", updated)
        logger.info("Skipped:        %s", skipped)
        logger.info("Deleted:        %s", deleted)
        logger.info("Errors:         %s", errors)
        logger.info("Record file:    %s", self.record_file)
        logger.info("Issues saved:   %s", self.archive_dir)

        logger.info("=" * 50)
        logger.info("提示")
        logger.info("=" * 50)
        logger.info("归档已完成，共 %s 个issue更新", updated)
        logger.info("归档目录: %s", self.archive_dir)
        logger.info("如需分析 Bug-Report Issue 并生成不支持场景报告，")
        logger.info("请在归档完成后告知 AI 开始分析。")

    def _get_all_issues(self) -> List[Dict[str, Any]]:
        base_url = "https://api.gitcode.com/api/v5"
        url = f"{base_url}/repos/{self.owner}/{self.repo}/issues"
        headers = GitCodeMCPArchiver._get_headers()
        all_issues = []
        page = 1

        while True:
            params = {"state": "all", "per_page": 100, "sort": "number", "direction": "desc", "page": page}
            data, error = GitCodeMCPArchiver._request_with_retry("GET", url, params=params, headers=headers)
            if error or not data:
                break
            all_issues.extend(data)
            if len(data) < 100:
                break
            page += 1

        return all_issues

    def _get_issue(self, issue_number: int) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        base_url = "https://api.gitcode.com/api/v5"
        url = f"{base_url}/repos/{self.owner}/{self.repo}/issues/{issue_number}"
        return GitCodeMCPArchiver._request_with_retry("GET", url, headers=GitCodeMCPArchiver._get_headers())

    def _get_issue_comments(self, issue_number: int) -> List[Dict[str, Any]]:
        base_url = "https://api.gitcode.com/api/v5"
        url = f"{base_url}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        headers = GitCodeMCPArchiver._get_headers()
        all_comments = []
        page = 1

        while True:
            params = {"per_page": 100, "page": page}
            data, error = GitCodeMCPArchiver._request_with_retry("GET", url, params=params, headers=headers)
            if error or not data:
                break
            all_comments.extend(data)
            if len(data) < 100:
                break
            page += 1

        return all_comments

    def _load_record(self) -> Dict[str, Any]:
        if self.record_file.exists():
            try:
                with open(self.record_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load record file: %s. Starting fresh.", e)
                return self._create_empty_record()
        else:
            return self._create_empty_record()

    def _create_empty_record(self) -> Dict[str, Any]:
        return {
            "last_check": datetime.utcnow().isoformat() + "Z",
            "repository": self.repo_path,
            "max_issue_number": 0,
            "issues": {}
        }

    def _save_record(self, record: Dict[str, Any]):
        record["last_check"] = datetime.utcnow().isoformat() + "Z"
        with open(self.record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    def _download_issue_assets_and_rewrite_markdown(self, issue_number: int, markdown: str) -> str:
        """Download image assets referenced in markdown and rewrite links to local relative paths."""

        urls = self._extract_asset_urls(markdown)
        if not urls:
            return markdown

        assets_dir = self.archive_dir / "assets" / f"issue-{issue_number}"
        assets_dir.mkdir(parents=True, exist_ok=True)

        rewritten = markdown
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            filename = Path(parsed.path).name
            if not filename:
                continue

            local_path = assets_dir / filename

            if not local_path.exists():
                try:
                    subprocess.run(
                        ["/usr/bin/curl", "-L", url, "-o", str(local_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    logger.warning("Issue #%s: Warning: failed to download asset %s: %s", issue_number, url, e)
                    continue

            rel = Path("assets") / f"issue-{issue_number}" / filename
            rewritten = rewritten.replace(url, rel.as_posix())

        return rewritten

    def _save_issue_markdown(self, issue_number: int, content: str):
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.archive_dir / f"issue-{issue_number}.md"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Archive GitCode repository issues to local markdown files using MCP tools."
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        help="Repository path in format 'owner/repo' (default: cann/pypto)"
    )
    parser.add_argument(
        "archive_dir",
        nargs="?",
        help="Archive directory (default: last saved path, first use required)"
    )
    parser.add_argument(
        "--token",
        help="GitCode Personal Access Token (optional, for private repos)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start issue number (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End issue number (inclusive)"
    )

    args = parser.parse_args()

    config = GitCodeMCPArchiver.load_config()

    if not args.repo_path:
        args.repo_path = config.get("repo_path") or "cann/pypto"

    if not args.archive_dir:
        args.archive_dir = config.get("last_archive_dir")
        if not args.archive_dir:
            logger.error("首次使用必须指定归档路径")
            logger.info("Usage: python archive_issues_mcp.py [repo_path] <archive_dir>")
            logger.info("Example: python archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues")
            sys.exit(1)

    GitCodeMCPArchiver.save_config(args.archive_dir, args.repo_path)

    if args.token:
        os.environ["GITCODE_TOKEN"] = args.token

    try:
        archiver = GitCodeMCPArchiver(args.repo_path, args.archive_dir, start=args.start, end=args.end)
        archiver.archive()
    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
