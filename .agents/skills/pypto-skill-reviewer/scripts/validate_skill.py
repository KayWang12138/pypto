#!/usr/bin/env python3
"""
validate_skill.py — Static checker for skill directories.

Checks 28 static rules from rules.json against a target skill directory.
Outputs a JSON array of findings to stdout.

Usage:
    python3 validate_skill.py <skill-directory-path>

Requirements: Python 3.8+, pyyaml.
"""

import json
import os
import re
import sys
import py_compile
import tempfile
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def parse_frontmatter(lines):
    """Parse YAML frontmatter delimited by --- lines.

    Returns (frontmatter_dict, fm_start_line, fm_end_line, body_start_line)
    or (None, None, None, None) if no valid frontmatter found.
    Line numbers are 1-based.
    """
    if not lines or lines[0].rstrip("\n\r") != "---":
        return None, None, None, None

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\n\r") == "---":
            end_idx = i
            break

    if end_idx is None:
        return None, None, None, None

    fm_text = "".join(lines[1:end_idx])
    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None, 1, end_idx + 1, end_idx + 1
    if not isinstance(fm, dict):
        return None, 1, end_idx + 1, end_idx + 1

    return fm, 1, end_idx + 1, end_idx + 1


# ---------------------------------------------------------------------------
# Code-block-aware line classifier
# ---------------------------------------------------------------------------

class CodeBlockTracker:
    """Tracks whether the current line is inside a fenced code block."""

    def __init__(self):
        self.in_block = False
        self.fence_pattern = re.compile(r"^(`{3,}|~{3,})")

    def update(self, line):
        """Call with each line. Returns True if this line is inside a code block."""
        stripped = line.rstrip("\n\r")
        m = self.fence_pattern.match(stripped)
        if m:
            if not self.in_block:
                self.in_block = True
                return True  # the fence line itself is part of the block
            else:
                self.in_block = False
                return True  # closing fence is part of the block
        return self.in_block


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------

def check_r01(lines, **_):
    """R01: SKILL.md must start with frontmatter delimited by ---"""
    fm, start, end, _ = parse_frontmatter(lines)
    if fm is None:
        return finding("R01", "S0", "D1", "FAIL",
                       "SKILL.md 未以有效的 `---` 分隔 YAML frontmatter 块开头",
                       "SKILL.md", 1,
                       lines[0].rstrip() if lines else "(empty file)")
    return None


def check_r02(fm, **_):
    """R02: name field must exist, non-empty, kebab-case, ≤64 chars"""
    if fm is None:
        return None  # R01 already catches this
    name = fm.get("name", "")
    if isinstance(name, str):
        name = name.strip().strip("'\"")
    else:
        name = ""
    if not name:
        return finding("R02", "S0", "D1", "FAIL",
                       "frontmatter 中 `name` 字段缺失或为空",
                       "SKILL.md", 1, "")
    kebab = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
    if not kebab.match(name):
        return finding("R02", "S0", "D1", "FAIL",
                       f"`name` 值 `{name}` 不是合法的 kebab-case 格式",
                       "SKILL.md", 1, name)
    if len(name) > 64:
        return finding("R02", "S0", "D1", "FAIL",
                       f"`name` 值超过 64 个字符（当前 {len(name)} 个字符）",
                       "SKILL.md", 1, name)
    return None


def check_r03(fm, **_):
    """R03: description field must exist and be ≥20 characters"""
    if fm is None:
        return None
    desc = fm.get("description", "")
    if isinstance(desc, str):
        desc = desc.strip()
    else:
        desc = str(desc).strip()
    if len(desc) < 20:
        return finding("R03", "S0", "D1", "FAIL",
                       f"`description` 过短（{len(desc)} 个字符，最少 20 个字符）",
                       "SKILL.md", 1, desc or "(empty)")
    return None


def check_r04(fm, skill_dir, **_):
    """R04: name value should match the skill directory name"""
    if fm is None:
        return None
    name = fm.get("name", "")
    if isinstance(name, str):
        name = name.strip().strip("'\"")
    dir_name = os.path.basename(os.path.normpath(skill_dir))
    if name.lower() != dir_name.lower():
        return finding("R04", "S1", "D1", "FAIL",
                       f"`name` 值 `{name}` 与目录名 `{dir_name}` 不匹配",
                       "SKILL.md", 1, name)
    return None


def check_r05(fm, **_):
    """R05: description should not exceed 1024 characters"""
    if fm is None:
        return None
    desc = fm.get("description", "")
    if isinstance(desc, str):
        desc = desc.strip()
    if len(desc) > 1024:
        return finding("R05", "S2", "D1", "FAIL",
                       f"`description` 超过 1024 个字符（当前 {len(desc)} 个字符）",
                       "SKILL.md", 1, desc[:80] + "...")
    return None


def check_r06(fm, known_fields, **_):
    """R06: Unknown frontmatter fields produce a warning"""
    if fm is None:
        return None
    unknown = [k for k in fm.keys() if k not in known_fields]
    if unknown:
        return finding("R06", "S3", "D1", "FAIL",
                       f"未知的 frontmatter 字段: {', '.join(unknown)}",
                       "SKILL.md", 1, ", ".join(unknown))
    return None


def check_r10(fm, **_):
    """R10: allowed-tools format check; boolean fields validation"""
    if fm is None:
        return None
    findings = []
    # Check allowed-tools if present
    at = fm.get("allowed-tools")
    if at is not None:
        if not isinstance(at, (str, list)):
            findings.append(finding("R10", "S2", "D1", "FAIL",
                                    "`allowed-tools` 必须是字符串（逗号分隔）或数组",
                                    "SKILL.md", 1, str(at)))
    # Check boolean fields
    for field in ("user-invocable", "intercept"):
        val = fm.get(field)
        if val is not None and not isinstance(val, bool):
            if isinstance(val, str) and val.lower() not in ("true", "false"):
                findings.append(finding("R10", "S2", "D1", "FAIL",
                                        f"`{field}` 必须为布尔值，当前为 `{val}`",
                                        "SKILL.md", 1, str(val)))
    return findings if findings else None


def check_r11(lines, **_):
    """R11: SKILL.md must not exceed 600 lines"""
    count = len(lines)
    if count > 600:
        return finding("R11", "S1", "D2", "FAIL",
                       f"SKILL.md 共 {count} 行（最大 600 行）",
                       "SKILL.md", 1, f"{count} lines")
    return None


def check_r12(lines, fm_end_line, **_):
    """R12: SKILL.md body must not exceed 6000 words"""
    body_start = fm_end_line if fm_end_line else 0
    body = "".join(lines[body_start:])
    word_count = len(body.split())
    if word_count > 6000:
        return finding("R12", "S2", "D2", "FAIL",
                       f"SKILL.md 正文共 {word_count} 个词（最大 6000 个词）",
                       "SKILL.md", body_start + 1, f"{word_count} words")
    return None


def check_r13(lines, fm_end_line, **_):
    """R13: No TODO/FIXME/HACK/XXX outside code blocks"""
    tracker = CodeBlockTracker()
    # Match TODO/FIXME/HACK/XXX as standalone markers — must be preceded by
    # start-of-line or whitespace (not a hyphen/letter), and typically
    # followed by colon, whitespace, or end-of-line.
    pattern = re.compile(r"(?:^|(?<=\s))(TODO|FIXME|HACK|XXX)(?=\s|:|$)", re.IGNORECASE)
    body_start = fm_end_line if fm_end_line else 0
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        # Skip frontmatter region and code blocks
        if in_block or i < body_start:
            continue
        m = pattern.search(line)
        if m:
            results.append(finding("R13", "S1", "D2", "FAIL",
                                   f"在代码块外发现 `{m.group(1)}` 占位标记",
                                   "SKILL.md", i + 1, line.rstrip()))
    return results if results else None


def check_r15(skill_dir, **_):
    """R15: Skill directory name must be kebab-case and ≤64 characters"""
    dir_name = os.path.basename(os.path.normpath(skill_dir))
    kebab = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
    if not kebab.match(dir_name):
        return finding("R15", "S2", "D3", "FAIL",
                       f"目录名 `{dir_name}` 不是合法的 kebab-case 格式",
                       dir_name, 0, dir_name)
    if len(dir_name) > 64:
        return finding("R15", "S2", "D3", "FAIL",
                       f"目录名超过 64 个字符（当前 {len(dir_name)} 个字符）",
                       dir_name, 0, dir_name)
    return None


def check_r16(lines, **_):
    """R16: Large content blocks (>100 lines of reference material) should be in references/"""
    tracker = CodeBlockTracker()
    consecutive_non_code = 0
    max_run = 0
    max_run_start = 0
    current_start = 0
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        stripped = line.strip()
        if not in_block and stripped.startswith(">"):
            if consecutive_non_code == 0:
                current_start = i
            consecutive_non_code += 1
            if consecutive_non_code > max_run:
                max_run = consecutive_non_code
                max_run_start = current_start
        else:
            consecutive_non_code = 0

    if max_run > 100:
        return finding("R16", "S2", "D3", "FAIL",
                       f"发现连续 {max_run} 行内联引用内容（应放在 references/）",
                       "SKILL.md", max_run_start + 1,
                       f"Lines {max_run_start + 1}-{max_run_start + max_run}")
    return None


def check_r17(lines, fm_end_line, **_):
    """R17: Referenced paths should use relative paths"""
    tracker = CodeBlockTracker()
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    body_start = fm_end_line if fm_end_line else 0
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block or i < body_start:
            continue
        for m in link_pattern.finditer(line):
            target = m.group(2)
            # Skip URLs and anchors
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if os.path.isabs(target):
                results.append(finding("R17", "S2", "D3", "FAIL",
                                       f"链接中包含绝对路径: `{target}` — 请使用相对路径",
                                       "SKILL.md", i + 1, line.rstrip()))
    return results if results else None


def check_r18(lines, fm_end_line, skill_dir, **_):
    """R18: Referenced file paths must actually exist"""
    tracker = CodeBlockTracker()
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    body_start = fm_end_line if fm_end_line else 0
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block or i < body_start:
            continue
        for m in link_pattern.finditer(line):
            target = m.group(2)
            # Skip URLs, anchors, and absolute paths (R17 handles those)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            # Strip anchor from path
            path_part = target.split("#")[0]
            if not path_part:
                continue
            resolved = os.path.normpath(os.path.join(skill_dir, path_part))
            if not os.path.exists(resolved):
                results.append(finding("R18", "S2", "D3", "FAIL",
                                       f"引用路径 `{path_part}` 不存在",
                                       "SKILL.md", i + 1, line.rstrip()))
    return results if results else None


def check_r22(lines, **_):
    """R22: Code blocks must be properly closed"""
    fence_re = re.compile(r"^(`{3,}|~{3,})")
    open_fence = None
    open_line = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        m = fence_re.match(stripped)
        if m:
            marker = m.group(1)[0]  # ` or ~
            length = len(m.group(1))
            if open_fence is None:
                open_fence = (marker, length)
                open_line = i
            elif marker == open_fence[0] and length >= open_fence[1]:
                # Check closing: must be only the fence chars (possibly with trailing space)
                after = stripped[len(m.group(1)):].strip()
                if after == "":
                    open_fence = None
                    open_line = None

    if open_fence is not None:
        unclosed_line = open_line if open_line is not None else 0
        return finding("R22", "S2", "D4", "FAIL",
                       f"从第 {unclosed_line + 1} 行开始的代码块未闭合",
                       "SKILL.md", unclosed_line + 1,
                       lines[unclosed_line].rstrip() if unclosed_line < len(lines) else "")
    return None


def check_r34(lines, **_):
    """R34: Must not embed secrets/credentials/sensitive data"""
    tracker = CodeBlockTracker()
    patterns = [
        (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "API key pattern (sk-...)"),
        (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GitHub personal access token"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key ID"),
        (re.compile(r"-----BEGIN.*PRIVATE KEY-----"), "Private key"),
        (re.compile(r"""password\s*[:=]\s*['"][^'"]{8,}['"]"""), "Hardcoded password"),
    ]
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block:
            continue
        for pat, label in patterns:
            if pat.search(line):
                results.append(finding("R34", "S0", "D8", "FAIL",
                                       f"检测到可能嵌入的密钥信息: {label}",
                                       "SKILL.md", i + 1, line.rstrip()))
                break
    return results if results else None


def check_r35(lines, **_):
    """R35: Must not hardcode absolute paths (user directory paths)"""
    tracker = CodeBlockTracker()
    path_pattern = re.compile(r"(/home/|/Users/|/root/)\S+")
    exclude_pattern = re.compile(r"(https?://|/usr/bin/env|/dev/null)")
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block:
            continue
        if path_pattern.search(line) and not exclude_pattern.search(line):
            results.append(finding("R35", "S1", "D8", "FAIL",
                                   "检测到硬编码的绝对用户路径",
                                   "SKILL.md", i + 1, line.rstrip()))
    return results if results else None


def check_r36(lines, **_):
    """R36: Must not inline large data blocks (>50 lines) in SKILL.md"""
    fence_re = re.compile(r"^(`{3,}|~{3,})\s*(\w*)")
    data_langs = {"json", "yaml", "yml", "xml", "csv", "toml"}
    in_block = False
    block_lang = ""
    block_start = 0
    block_lines = 0
    results = []
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        m = fence_re.match(stripped)
        if m:
            if not in_block:
                in_block = True
                block_lang = (m.group(2) or "").lower()
                block_start = i
                block_lines = 0
            else:
                if block_lang in data_langs and block_lines > 50:
                    results.append(finding("R36", "S1", "D8", "FAIL",
                                           f"内联 {block_lang} 数据块过大（{block_lines} 行，最大 50 行）",
                                           "SKILL.md", block_start + 1,
                                           f"Lines {block_start + 1}-{i + 1}"))
                in_block = False
                block_lines = 0
        elif in_block:
            block_lines += 1
    return results if results else None


def check_r37(fm_lines, **_):
    """R37: Frontmatter must not contain XML tags"""
    if not fm_lines:
        return None
    xml_pattern = re.compile(r"<[a-zA-Z]")
    for i, line in enumerate(fm_lines):
        if xml_pattern.search(line):
            return finding("R37", "S1", "D8", "FAIL",
                           "frontmatter 中发现 XML 标签",
                           "SKILL.md", i + 2, line.rstrip())  # +2: 1-based + skip opening ---
    return None


def check_r38(lines, **_):
    """R38: Must not use Windows-style paths"""
    tracker = CodeBlockTracker()
    win_path = re.compile(r"[A-Z]:\\")
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block:
            continue
        if win_path.search(line):
            results.append(finding("R38", "S2", "D8", "FAIL",
                                   "检测到 Windows 风格路径",
                                   "SKILL.md", i + 1, line.rstrip()))
    return results if results else None


def check_r39(skill_dir, **_):
    """R39: Python scripts must have valid syntax"""
    scripts_dir = os.path.join(skill_dir, "scripts")
    if not os.path.isdir(scripts_dir):
        return None
    results = []
    for f in os.listdir(scripts_dir):
        if not f.endswith(".py"):
            continue
        fpath = os.path.join(scripts_dir, f)
        try:
            with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
                py_compile.compile(fpath, tmp.name, doraise=True)
        except py_compile.PyCompileError as e:
            results.append(finding("R39", "S2", "D9", "FAIL",
                                   f"`{f}` 中存在 Python 语法错误: {e}",
                                   f"scripts/{f}", getattr(e, "lineno", 0) or 0,
                                   str(e)))
    return results if results else None


def check_r40(skill_dir, **_):
    """R40: Scripts must include a shebang line"""
    scripts_dir = os.path.join(skill_dir, "scripts")
    if not os.path.isdir(scripts_dir):
        return None
    results = []
    for f in os.listdir(scripts_dir):
        if not any(f.endswith(ext) for ext in (".py", ".sh", ".bash")):
            continue
        # 豁免 __init__.py（Python 包初始化文件不需要 shebang）
        if f == "__init__.py":
            continue
        fpath = os.path.join(scripts_dir, f)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                first_line = fh.readline()
            if not first_line.startswith("#!"):
                results.append(finding("R40", "S2", "D9", "FAIL",
                                       f"脚本 `{f}` 缺少 shebang 行",
                                       f"scripts/{f}", 1, first_line.rstrip()))
        except OSError:
            pass
    return results if results else None


def check_r41(skill_dir, **_):
    """R41: Script paths must be portable (no hardcoded user paths)"""
    scripts_dir = os.path.join(skill_dir, "scripts")
    if not os.path.isdir(scripts_dir):
        return None
    path_pattern = re.compile(r"(/home/|/Users/|/root/)\S+")
    exclude_pattern = re.compile(r"(https?://|/usr/bin/env|/dev/null)")
    # Lines containing regex definitions or string patterns for detection are not real hardcoded paths
    detection_pattern = re.compile(r"""(re\.compile|compile\(|["'].*(/home/|/Users/|/root/).*["'])""")
    results = []
    for f in os.listdir(scripts_dir):
        fpath = os.path.join(scripts_dir, f)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if (path_pattern.search(line)
                            and not exclude_pattern.search(line)
                            and not detection_pattern.search(line)):
                        results.append(finding("R41", "S2", "D9", "FAIL",
                                               f"脚本 `{f}` 中存在硬编码绝对路径",
                                               f"scripts/{f}", i, line.rstrip()))
        except OSError:
            pass
    return results if results else None


def check_r45(skill_dir, standard_dirs, **_):
    """R45: Subdirectory structure should use standard naming"""
    results = []
    try:
        entries = os.listdir(skill_dir)
    except OSError:
        return None
    for entry in entries:
        full = os.path.join(skill_dir, entry)
        if os.path.isdir(full) and entry not in standard_dirs and not entry.startswith("."):
            results.append(finding("R45", "S3", "D3", "FAIL",
                                   f"非标准子目录 `{entry}`（期望: {', '.join(standard_dirs)}）",
                                   entry, 0, entry))
    return results if results else None


def check_r46(fm, **_):
    """R46: description must not contain angle-bracket placeholders like <xxx>"""
    if fm is None:
        return None
    desc = fm.get("description", "")
    if not isinstance(desc, str):
        return None
    placeholder_re = re.compile(r"<[a-zA-Z][a-zA-Z0-9_-]*>")
    matches = placeholder_re.findall(desc)
    if matches:
        return finding("R46", "S2", "D1", "FAIL",
                       f"`description` 包含未替换的占位符: {', '.join(matches)}",
                       "SKILL.md", 1, desc[:100])
    return None


def check_r47(lines, fm_end_line, **_):
    """R47: No duplicate section headings within the same file"""
    tracker = CodeBlockTracker()
    body_start = fm_end_line if fm_end_line else 0
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
    seen = {}  # heading_text -> first line number
    results = []
    for i, line in enumerate(lines):
        in_block = tracker.update(line)
        if in_block or i < body_start:
            continue
        m = heading_re.match(line.rstrip())
        if m:
            text = m.group(2).strip()
            if text in seen:
                results.append(finding("R47", "S2", "D2", "FAIL",
                                       f"重复的标题 `{text}`（首次出现在第 {seen[text]} 行）",
                                       "SKILL.md", i + 1, line.rstrip()))
            else:
                seen[text] = i + 1
    return results if results else None


def check_r48(lines, **_):
    """R48: Fenced code blocks should have a language annotation"""
    fence_re = re.compile(r"^(`{3,}|~{3,})\s*(\S*)")
    in_block = False
    results = []
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        m = fence_re.match(stripped)
        if m:
            if not in_block:
                in_block = True
                lang = m.group(2)
                if not lang:
                    results.append(finding("R48", "S3", "D4", "FAIL",
                                           "代码块缺少语言标注",
                                           "SKILL.md", i + 1, stripped))
            else:
                in_block = False
    return results if results else None





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finding(rule_id, severity, dimension, status, message, file, line, snippet):
    """Create a standardized finding dict."""
    return {
        "rule_id": rule_id,
        "severity": severity,
        "dimension": dimension,
        "status": status,
        "message": message,
        "evidence": {
            "file": file,
            "line": line,
            "snippet": str(snippet)[:200]
        }
    }


def flatten(results):
    """Flatten a list that may contain None, dicts, or lists of dicts."""
    out = []
    for r in results:
        if r is None:
            continue
        if isinstance(r, list):
            out.extend(r)
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS = {
    "D1": 0.25, "D2": 0.15, "D3": 0.10,
    "D4": 0.10, "D5": 0.10, "D6": 0.10, "D7": 0.05,
    "D8": 0.10, "D9": 0.05,
}

SEVERITY_DEDUCTIONS = {"S0": 20, "S1": 10, "S2": 5, "S3": 2}


def calculate_score(findings, skill_dir):
    """Calculate dimension scores, total, and grade from findings.

    Formula (per scoring-spec.md):
        dimension_raw   = max(0, 100 - sum_of_FAIL_deductions)
        dimension_score = dimension_raw * weight
        total           = sum(dimension_scores for D1-D9)

    Special rules:
        - S0 veto: any S0 FAIL caps total at 59.9, max grade D
        - D9: no scripts/ directory in target skill -> full marks (5.0)
    """
    dim_deductions = {f"D{i}": 0 for i in range(1, 10)}
    s0_veto = False

    for f in findings:
        if f.get("status") != "FAIL":
            continue
        sev = f.get("severity", "S3")
        dim = f.get("dimension", "D1")
        dim_deductions[dim] = dim_deductions.get(dim, 0) + SEVERITY_DEDUCTIONS.get(sev, 0)
        if sev == "S0":
            s0_veto = True

    scripts_dir = os.path.join(skill_dir, "scripts")
    d9_no_scripts = not os.path.isdir(scripts_dir)

    dimensions = {}
    for dim, weight in DIMENSION_WEIGHTS.items():
        if dim == "D9" and d9_no_scripts:
            dimensions[dim] = {
                "raw": 100, "weight": weight,
                "score": round(100 * weight, 2),
                "deductions": 0,
                "note": "no scripts/ directory — full marks",
            }
        else:
            raw = max(0, 100 - dim_deductions.get(dim, 0))
            dimensions[dim] = {
                "raw": raw, "weight": weight,
                "score": round(raw * weight, 2),
                "deductions": dim_deductions.get(dim, 0),
            }

    total = round(sum(d["score"] for d in dimensions.values()), 2)

    if s0_veto and total > 59.9:
        total = 59.9

    if total >= 90:
        grade = "A"
    elif total >= 75:
        grade = "B"
    elif total >= 60:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    if s0_veto and grade not in ("D", "F"):
        grade = "D"

    return {
        "dimensions": dimensions,
        "total": total,
        "grade": grade,
        "s0_veto": s0_veto,
        "d9_no_scripts": d9_no_scripts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(skill_dir):
    """Run all static checks against the given skill directory."""
    skill_md_path = os.path.join(skill_dir, "SKILL.md")

    if not os.path.isfile(skill_md_path):
        return [finding("R01", "S0", "D1", "FAIL",
                        "技能目录中未找到 SKILL.md",
                        "SKILL.md", 0, "(file not found)")]

    with open(skill_md_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    fm, fm_start, fm_end, body_start = parse_frontmatter(lines)

    # Extract frontmatter lines (between the --- delimiters) for R37
    fm_lines = lines[1:fm_end - 1] if fm_end and fm_end > 2 else []

    # Load known fields and standard dirs from rules.json
    rules_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "references", "rules.json")
    known_fields = [
        "name", "description", "context", "agent", "allowed-tools",
        "user-invocable", "intercept", "model"
    ]
    standard_dirs = ["references", "scripts", "templates", "assets", "examples"]
    if os.path.isfile(rules_json_path):
        try:
            with open(rules_json_path, "r", encoding="utf-8") as rf:
                rules_data = json.load(rf)
            known_fields = rules_data.get("known_frontmatter_fields", known_fields)
            standard_dirs = rules_data.get("standard_subdirectories", standard_dirs)
        except (json.JSONDecodeError, OSError):
            pass

    ctx = {
        "lines": lines,
        "fm": fm,
        "fm_lines": fm_lines,
        "fm_end_line": fm_end,
        "skill_dir": skill_dir,
        "known_fields": known_fields,
        "standard_dirs": standard_dirs,
    }

    results = [
        check_r01(**ctx),
        check_r02(**ctx),
        check_r03(**ctx),
        check_r04(**ctx),
        check_r05(**ctx),
        check_r06(**ctx),
        check_r10(**ctx),
        check_r11(**ctx),
        check_r12(**ctx),
        check_r13(**ctx),
        check_r15(**ctx),
        check_r16(**ctx),
        check_r17(**ctx),
        check_r18(**ctx),
        check_r22(**ctx),
        check_r34(**ctx),
        check_r35(**ctx),
        check_r36(**ctx),
        check_r37(**ctx),
        check_r38(**ctx),
        check_r39(**ctx),
        check_r40(**ctx),
        check_r41(**ctx),
        check_r45(**ctx),
        check_r46(**ctx),
        check_r47(**ctx),
        check_r48(**ctx),
    ]

    findings = flatten(results)

    # Inject PASS status for rules that did not fail
    STATIC_RULES = {
        "R01": ("S0", "D1"), "R02": ("S0", "D1"), "R03": ("S0", "D1"),
        "R04": ("S1", "D1"), "R05": ("S2", "D1"), "R06": ("S3", "D1"),
        "R10": ("S2", "D1"), "R11": ("S1", "D2"), "R12": ("S2", "D2"),
        "R13": ("S1", "D2"), "R15": ("S2", "D3"), "R16": ("S2", "D3"),
        "R17": ("S2", "D3"), "R18": ("S2", "D3"), "R22": ("S2", "D4"),
        "R34": ("S0", "D8"), "R35": ("S1", "D8"), "R36": ("S1", "D8"),
        "R37": ("S1", "D8"), "R38": ("S2", "D8"), "R39": ("S2", "D9"),
        "R40": ("S2", "D9"), "R41": ("S2", "D9"),
        "R45": ("S3", "D3"), "R46": ("S2", "D1"), "R47": ("S2", "D2"),
        "R48": ("S3", "D4"),
    }
    failed_rules = {f["rule_id"] for f in findings}
    for rule_id, (sev, dim) in STATIC_RULES.items():
        if rule_id not in failed_rules:
            findings.append({
                "rule_id": rule_id,
                "severity": sev,
                "dimension": dim,
                "status": "PASS",
                "message": "",
                "evidence": {"file": "", "line": 0, "snippet": ""},
            })

    return findings


def main():
    """CLI entry point."""
    score_flag = "--score" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--score"]

    if not args:
        print("用法: python3 validate_skill.py [--score] <技能目录路径>",
              file=sys.stderr)
        sys.exit(1)

    skill_dir = os.path.abspath(args[0])
    if not os.path.isdir(skill_dir):
        print(json.dumps([finding("R01", "S0", "D1", "FAIL",
                                  f"路径不是目录: {skill_dir}",
                                  skill_dir, 0, "(not a directory)")]))
        sys.exit(1)

    findings = validate(skill_dir)

    if score_flag:
        score_result = calculate_score(findings, skill_dir)
        output = {"findings": findings, "score": score_result}
    else:
        output = findings

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
