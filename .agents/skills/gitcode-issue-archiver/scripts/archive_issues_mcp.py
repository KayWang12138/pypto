#!/usr/bin/env python3
"""
GitCode Issue Archiver (MCP Version)

Archives GitCode repository issues to local markdown files using MCP tools.
"""

import argparse
import json
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


class GitCodeMCPArchiver:
    """Archive GitCode repository issues using MCP tools."""

    def __init__(self, repo_path: str, archive_dir: str):
        self.repo_path = repo_path
        self.archive_dir = Path(archive_dir)

        parts = repo_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repo_path format: {repo_path}. Expected 'owner/repo'")

        self.owner, self.repo = parts
        self.record_file = self.archive_dir / "archive_record.json"

    def _request_with_retry(self, method: str, url: str, params: Optional[Dict[str, Any]] = None,
                            headers: Optional[Dict[str, str]] = None) -> Tuple[Any, Optional[int]]:
        """HTTP request with exponential backoff retry."""
        try:
            import requests
        except ImportError:
            print("Error: requests library is required. Install with: pip install requests")
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
                # Don't retry 4xx errors (except rate limit 429)
                if 400 <= status_code < 500 and status_code != 429:
                    return None, status_code

                print(f"HTTP Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, status_code

            except requests.exceptions.RequestException as e:
                print(f"Request Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, None

        return None, None

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        token = os.environ.get("GITCODE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_all_issues(self) -> List[Dict[str, Any]]:
        base_url = "https://api.gitcode.com/api/v5"
        url = f"{base_url}/repos/{self.owner}/{self.repo}/issues"
        headers = self._get_headers()
        all_issues = []
        page = 1

        while True:
            params = {"state": "all", "per_page": 100, "sort": "number", "direction": "desc", "page": page}
            data, error = self._request_with_retry("GET", url, params=params, headers=headers)
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
        return self._request_with_retry("GET", url, headers=self._get_headers())

    def _get_issue_comments(self, issue_number: int) -> List[Dict[str, Any]]:
        base_url = "https://api.gitcode.com/api/v5"
        url = f"{base_url}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        headers = self._get_headers()
        all_comments = []
        page = 1

        while True:
            params = {"per_page": 100, "page": page}
            data, error = self._request_with_retry("GET", url, params=params, headers=headers)
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
                print(f"Warning: Could not load record file: {e}. Starting fresh.")
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

    @staticmethod
    def _load_config() -> Dict[str, Optional[str]]:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return {
                        "last_archive_dir": config.get("last_archive_dir"),
                        "repo_path": config.get("repo_path"),
                    }
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file: {e}")
        return {"last_archive_dir": None, "repo_path": None}

    @staticmethod
    def _save_config(archive_dir: str, repo_path: str):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {"last_archive_dir": archive_dir, "repo_path": repo_path}
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _save_record(self, record: Dict[str, Any]):
        record["last_check"] = datetime.utcnow().isoformat() + "Z"
        with open(self.record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    def _generate_markdown(self, issue: Dict[str, Any], comments: List[Dict[str, Any]]) -> str:
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
    def _extract_asset_urls(markdown: str) -> List[str]:
        # GitCode issue bodies often embed images as markdown like:
        # ![](https://raw.gitcode.com/user-images/assets/.../xx.png)
        # Also handle plain URLs.
        pattern = re.compile(r"https?://\S+?\.(?:png|jpg|jpeg|gif|webp)(?:\?\S*)?", re.IGNORECASE)
        urls = pattern.findall(markdown)

        # Preserve order but dedupe.
        seen = set()
        out: List[str] = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

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

            # Download only if missing to make re-runs cheap.
            if not local_path.exists():
                try:
                    subprocess.run(
                        ["curl", "-L", url, "-o", str(local_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    # Don't fail the archive on asset download errors.
                    print(f"  Issue #{issue_number}: Warning: failed to download asset {url}: {e}")
                    continue

            # Always use forward slashes in markdown.
            rel = Path("assets") / f"issue-{issue_number}" / filename
            rewritten = rewritten.replace(url, rel.as_posix())

        return rewritten

    def _save_issue_markdown(self, issue_number: int, content: str):
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.archive_dir / f"issue-{issue_number}.md"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Note: asset download/rewrite is handled by archive() before saving.

    def _needs_update(self, record: Dict[str, Any], issue_number: int, issue: Dict[str, Any]) -> bool:
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
        print(f"Archiving issues from {self.repo_path} using MCP tools...")
        print(f"Archive directory: {self.archive_dir}")
        print()

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        record = self._load_record()

        print("Fetching issue list using gitcode_list_issues MCP tool...")
        all_issues = self._get_all_issues()

        if not all_issues:
            print("No issues found in repository.")
            return

        max_issue_number = int(all_issues[0]["number"])
        print(f"Max issue number from list: {max_issue_number}")

        if max_issue_number > record["max_issue_number"]:
            record["max_issue_number"] = max_issue_number

        processed = 0
        updated = 0
        skipped = 0
        deleted = 0
        errors = 0

        for issue_number in range(1, max_issue_number + 1):
            processed += 1

            if processed % 10 == 0 or issue_number == max_issue_number:
                print(f"Processing {processed}/{max_issue_number + 1} issues...")

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
                print(f"  Issue #{issue_number}: Failed with status {error_status}")
                continue

            if not issue:
                continue

            if not self._needs_update(record, issue_number, issue):
                skipped += 1
                continue

            comments = self._get_issue_comments(issue_number)

            markdown = self._generate_markdown(issue, comments)
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
            print(f"  Issue #{issue_number}: {issue.get('state')} - {issue.get('title', '')[:50]}")

        self._save_record(record)

        print()
        print("=" * 50)
        print("Archive Summary")
        print("=" * 50)
        print(f"Total processed: {processed}")
        print(f"Updated:        {updated}")
        print(f"Skipped:        {skipped}")
        print(f"Deleted:        {deleted}")
        print(f"Errors:         {errors}")
        print()
        print(f"Record file:    {self.record_file}")
        print(f"Issues saved:   {self.archive_dir}")
        
        # 提示用户可以分析issue
        print()
        print("=" * 50)
        print("💡 提示")
        print("=" * 50)
        print(f"归档已完成，共 {updated} 个issue更新")
        print(f"归档目录: {self.archive_dir}")
        print()
        print("如需分析 Bug-Report Issue 并生成不支持场景报告，")
        print("请在归档完成后告知 AI 开始分析。")


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

    args = parser.parse_args()

    config = GitCodeMCPArchiver._load_config()

    if not args.repo_path:
        args.repo_path = config.get("repo_path") or "cann/pypto"

    if not args.archive_dir:
        args.archive_dir = config.get("last_archive_dir")
        if not args.archive_dir:
            print("Error: 首次使用必须指定归档路径")
            print("Usage: python archive_issues_mcp.py [repo_path] <archive_dir>")
            print("Example: python archive_issues_mcp.py cann/pypto /workspace/archive/pypto_issues")
            sys.exit(1)

    GitCodeMCPArchiver._save_config(args.archive_dir, args.repo_path)

    if args.token:
        os.environ["GITCODE_TOKEN"] = args.token

    try:
        archiver = GitCodeMCPArchiver(args.repo_path, args.archive_dir)
        archiver.archive()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
