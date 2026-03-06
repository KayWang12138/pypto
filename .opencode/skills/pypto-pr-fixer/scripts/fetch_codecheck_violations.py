#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""从 openlibing.com CodeCheck 页面提取违规项。"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format="%(message)s")

VIOLATION_RE = re.compile(r"文件路径:([^\n:]+):(\d+)\s*问题描述[：:]([^\n]+)\s*规则[：:]([^\n]+)")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    description: str
    rule_id: str
    rule_description: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
        }


class Args(argparse.Namespace):
    url: str = ""
    output: str = "json"
    group: bool = False
    retries: int = 3


def parse_violations_from_text(text: str) -> list[Violation]:
    violations: list[Violation] = []
    for file_path, line_text, description, rule in VIOLATION_RE.findall(text):
        parts = rule.strip().split(" ", 1)
        rule_id = parts[0] if parts else ""
        rule_description = parts[1] if len(parts) > 1 else ""
        violations.append(
            Violation(
                file=file_path.strip(),
                line=int(line_text),
                description=description.strip(),
                rule_id=rule_id,
                rule_description=rule_description,
            )
        )
    return violations


def extract_violations_with_playwright(url: str) -> list[Violation]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _ = page.goto(url, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(8000)

            _ = page.evaluate(
                """
                const cookieDivs = document.querySelectorAll('[class*="cookie"]');
                cookieDivs.forEach(div => div.remove());
                const overlays = document.querySelectorAll('[style*="position: fixed"], [style*="position:fixed"]');
                overlays.forEach(el => {
                    if (el.style.zIndex > 1000) {
                        el.remove();
                    }
                });
                """
            )
            page.wait_for_timeout(500)

            try:
                page.locator(".el-pagination__sizes").click(timeout=5000)
                page.wait_for_timeout(500)

                options = page.locator(".el-select-dropdown__item").all()
                largest_opt = None
                largest_num = 0
                for opt in options:
                    text = opt.inner_text()
                    digits = "".join(ch for ch in text if ch.isdigit())
                    num = int(digits) if digits else 0
                    if num > largest_num:
                        largest_num = num
                        largest_opt = opt

                if largest_opt is not None:
                    largest_opt.click()
                    page.wait_for_timeout(2000)
            except Exception as exc:
                logging.debug("pagination size expand skipped: %s", exc)

            body_text = page.inner_text("body")
            return parse_violations_from_text(body_text)
        finally:
            browser.close()


def extract_with_retry(url: str, max_retries: int = 3) -> list[Violation]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries}...")
            return extract_violations_with_playwright(url)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if attempt < max_retries:
                wait_s = 2 ** (attempt - 1)
                logging.warning(f"  Failed: {exc}")
                logging.info(f"  Waiting {wait_s}s before retry...")
                time.sleep(wait_s)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def group_by_rule(violations: list[Violation]) -> dict[str, list[Violation]]:
    by_rule: dict[str, list[Violation]] = {}
    for violation in violations:
        by_rule.setdefault(violation.rule_id, []).append(violation)
    return by_rule


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="从 openlibing.com 提取 CodeCheck 违规列表")
    _ = parser.add_argument("url", help="CodeCheck 报告 URL")
    _ = parser.add_argument("--output", "-o", choices=["json", "text"], default="json", help="输出格式")
    _ = parser.add_argument("--group", "-g", action="store_true", help="按规则分组输出")
    _ = parser.add_argument("--retries", type=int, default=3, help="提取重试次数（默认 3）")
    return parser.parse_args(namespace=Args())


def main() -> int:
    args = parse_args()

    try:
        retries = max(1, int(args.retries))
        violations = extract_with_retry(args.url, max_retries=retries)

        if args.output == "json":
            grouped = group_by_rule(violations)
            result: dict[str, object] = {
                "total": len(violations),
                "by_rule": {rule: len(items) for rule, items in grouped.items()},
                "violations": [v.to_dict() for v in violations],
            }
            if args.group:
                result["grouped"] = {rule: [item.to_dict() for item in items] for rule, items in grouped.items()}
            logging.info(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logging.info(f"Total violations: {len(violations)}\n")
            if args.group:
                grouped = group_by_rule(violations)
                for rule, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    desc = items[0].rule_description if items else ""
                    logging.info(f"## {rule}: {len(items)} violations")
                    logging.info(f"   {desc}\n")
                    for item in items:
                        logging.info(f"   - {item.file}:{item.line}")
                        logging.info(f"     {item.description}\n")
            else:
                for item in violations:
                    logging.info(f"{item.rule_id} | {item.file}:{item.line}")
                    logging.info(f"  {item.description}\n")

        return 0
    except Exception as exc:
        logging.error(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
