#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
CodeCheck 违规提取脚本

从 openlibing.com 的 CodeArts-Check 报告页面提取违规列表。

使用方法:
    python fetch_codecheck_violations.py <codecheck_url> [--output json|text]

参数:
    codecheck_url: openlibing.com 的 codecheck 报告 URL
    --output: 输出格式，json 或 text (默认: json)

示例:
    python fetch_codecheck_violations.py \
        "https://www.openlibing.com/apps/entryCheckDashCode/MR_xxx/yyy?projectId=300033"
"""

import argparse
import json
import logging
import re
import sys
from typing import Any


def extract_violations_with_playwright(url: str) -> list[dict[str, Any]]:
    """使用 Playwright 提取违规列表"""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 访问页面
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)
        
        # 移除 cookie 对话框（会阻挡分页控件）
        page.evaluate("""
            const cookieDivs = document.querySelectorAll('[class*="cookie"]');
            cookieDivs.forEach(div => div.remove());
            const overlays = document.querySelectorAll('[style*="position: fixed"], [style*="position:fixed"]');
            overlays.forEach(el => {
                if (el.style.zIndex > 1000) {
                    el.remove();
                }
            });
        """)
        page.wait_for_timeout(500)
        
        # 尝试切换到最大页面大小以显示所有违规
        try:
            page.locator(".el-pagination__sizes").click(timeout=5000)
            page.wait_for_timeout(500)
            
            options = page.locator(".el-select-dropdown__item").all()
            if options:
                # 找到最大的数字选项
                largest_opt = None
                largest_num = 0
                for opt in options:
                    text = opt.inner_text()
                    num = int(''.join(filter(str.isdigit, text))) if any(c.isdigit() for c in text) else 0
                    if num > largest_num:
                        largest_num = num
                        largest_opt = opt
                
                if largest_opt:
                    largest_opt.click()
                    page.wait_for_timeout(2000)
        except Exception:
            pass  # 分页控件可能不存在
        
        # 提取违规列表
        text = page.inner_text("body")
        browser.close()
        
        return parse_violations_from_text(text)


def parse_violations_from_text(text: str) -> list[dict[str, Any]]:
    """从页面文本解析违规列表"""
    pattern = r'文件路径:([^\n:]+):(\d+)\s*问题描述[：:]([^\n]+)\s*规则[：:]([^\n]+)'
    matches = re.findall(pattern, text)
    
    violations = []
    for match in matches:
        file_path, line, description, rule = match
        rule_parts = rule.strip().split(" ", 1)
        
        violations.append({
            "file": file_path.strip(),
            "line": int(line),
            "description": description.strip(),
            "rule_id": rule_parts[0] if rule_parts else "",
            "rule_description": rule_parts[1] if len(rule_parts) > 1 else ""
        })
    
    return violations


def group_by_rule(violations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按规则分组违规"""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for v in violations:
        rule = v['rule_id']
        if rule not in by_rule:
            by_rule[rule] = []
        by_rule[rule].append(v)
    return by_rule


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description="从 openlibing.com 提取 CodeCheck 违规列表")
    parser.add_argument("url", help="CodeCheck 报告 URL")
    parser.add_argument("--output", "-o", choices=["json", "text"], default="json", help="输出格式")
    parser.add_argument("--group", "-g", action="store_true", help="按规则分组输出")
    
    args = parser.parse_args()
    
    try:
        violations = extract_violations_with_playwright(args.url)
        
        if args.output == "json":
            result: dict[str, Any] = {
                "total": len(violations),
                "by_rule": {k: len(v) for k, v in group_by_rule(violations).items()},
                "violations": violations
            }
            if args.group:
                result["grouped"] = group_by_rule(violations)
            logging.info(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logging.info(f"Total violations: {len(violations)}\n")
            
            if args.group:
                by_rule = group_by_rule(violations)
                for rule, items in sorted(by_rule.items(), key=lambda x: -len(x[1])):
                    desc = items[0]['rule_description'] if items else ''
                    logging.info(f"## {rule}: {len(items)} violations")
                    logging.info(f"   {desc}\n")
                    for v in items:
                        logging.info(f"   - {v['file']}:{v['line']}")
                        logging.info(f"     {v['description']}\n")
            else:
                for v in violations:
                    logging.info(f"{v['rule_id']} | {v['file']}:{v['line']}")
                    logging.info(f"  {v['description']}\n")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
