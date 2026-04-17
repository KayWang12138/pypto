#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

REQUIRED_SECTIONS = [
    "## 1. Pass 概述",
    "## 2. 代码分析",
    "## 3. 业务分析",
    "## 4. OPCode 特判分析",
    "## 5. 总结",
    "## 6. 相关文件",
    "## 7. 附录",
]

BAD_PATTERNS = [
    r"\[Pass名称\]",
    r"\[描述",
    r"TODO",
    r"TBD",
    r"<fact>",
    r"<project-relative-file-path>",
]


def extract_section(text: str, header: str):
    idx = text.find(header)
    if idx < 0:
        return ""
    next_idx = len(text)
    for h in REQUIRED_SECTIONS:
        if h == header:
            continue
        h_idx = text.find(h, idx + len(header))
        if h_idx != -1:
            next_idx = min(next_idx, h_idx)
    return text[idx:next_idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pass analysis report")
    parser.add_argument("--report", required=True, help="Report markdown path")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"FAIL: report not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8", errors="ignore")
    errors = []

    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"missing section: {section}")

    for pat in BAD_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            errors.append(f"unresolved placeholder or todo: pattern `{pat}`")

    evidence_lines = re.findall(r"^- Evidence:\s+.+:.+[ \t]-[ \t].+$", text, flags=re.M)
    if len(evidence_lines) < 3:
        errors.append("at least 3 evidence lines are required")

    file_section = extract_section(text, "## 6. 相关文件")
    project_paths = re.findall(r"^-\s+([A-Za-z0-9_./\-]+\.(?:h|hpp|cc|cpp|md))\s*$", file_section, flags=re.M)
    if len(project_paths) < 2:
        errors.append("section `## 6. 相关文件` must include at least 2 file paths")

    if errors:
        print("FAIL")
        for item in errors:
            print(f"- {item}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
