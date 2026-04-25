#!/usr/bin/env python3
"""Extract `hash order -> Subgraph IDs` mappings from pass logs.

Usage:
  python extract_hash_order_mapping.py <log_file_or_dir> [<log_file_or_dir> ...]

For each input log file, this script writes an output file with suffix
`_HASH_ORDER`.
For each input directory, this script recursively scans:
  - Pass_*_NBufferMerge/*.log
  - Pass_*_L1CopyInReuseMerge/*.log
  - pypto-log*.log
  - plog-*.log
and writes one merged summary file: HASH_ORDER.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERN = re.compile(
    r"(L1\s+reuse|Cube\s+nbuffer|Vec\s+nbuffer)\s+hash\s+order:\s*(\d+)\s*,\s*"
    r"Subgraph\s+hash:\s*(\d+)\s*,\s*"
    r"Subgraph\s+count:\s*\d+\s*,\s*"
    r"Subgraph\s+IDs:\s*\[([^\]]*)\]",
    re.IGNORECASE,
)


def normalize_source(raw_source: str) -> str:
    source = raw_source.lower().replace(" ", "")
    if source == "l1reuse":
        return "L1 reuse"
    if source == "cubenbuffer":
        return "Cube nbuffer"
    if source == "vecnbuffer":
        return "Vec nbuffer"
    return raw_source


def parse_file(path: Path) -> list[tuple[str, str, str, str]]:
    mappings: list[tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = PATTERN.search(line)
            if not match:
                continue
            source = normalize_source(match.group(1))
            hash_order = match.group(2)
            subgraph_hash = match.group(3)
            subgraph_ids_content = match.group(4).strip()
            subgraph_ids = f"[{subgraph_ids_content}]" if subgraph_ids_content else "[]"
            mappings.append((source, hash_order, subgraph_hash, subgraph_ids))
    return mappings


def write_output(path: Path, mappings: list[tuple[str, str, str, str]]) -> Path:
    if path.suffix:
        out_path = path.with_name(f"{path.stem}_HASH_ORDER{path.suffix}")
    else:
        out_path = Path(str(path) + "_HASH_ORDER.log")

    l1_mappings = [m for m in mappings if m[0] == "L1 reuse"]
    cube_mappings = [m for m in mappings if m[0] == "Cube nbuffer"]
    vec_mappings = [m for m in mappings if m[0] == "Vec nbuffer"]

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# source: {path.name}\n")
        f.write(f"# total_matches: {len(mappings)}\n")
        f.write(f"# l1_reuse_matches: {len(l1_mappings)}\n")
        f.write(f"# cube_nbuffer_matches: {len(cube_mappings)}\n")
        f.write(f"# vec_nbuffer_matches: {len(vec_mappings)}\n")

        f.write("\n[L1 reuse]\n")
        for _, hash_order, subgraph_hash, ids in l1_mappings:
            f.write(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}\n")

        f.write("\n[Cube nbuffer]\n")
        for _, hash_order, subgraph_hash, ids in cube_mappings:
            f.write(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}\n")

        f.write("\n[Vec nbuffer]\n")
        for _, hash_order, subgraph_hash, ids in vec_mappings:
            f.write(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}\n")

        other_mappings = [
            m for m in mappings if m[0] not in {"L1 reuse", "Cube nbuffer", "Vec nbuffer"}
        ]
        if other_mappings:
            f.write("\n[Other]\n")
            for source, hash_order, subgraph_hash, ids in other_mappings:
                f.write(f"{source} | hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}\n")

    return out_path


def format_mappings_block(mappings: list[tuple[str, str, str, str]]) -> str:
    l1_mappings = [m for m in mappings if m[0] == "L1 reuse"]
    cube_mappings = [m for m in mappings if m[0] == "Cube nbuffer"]
    vec_mappings = [m for m in mappings if m[0] == "Vec nbuffer"]

    lines: list[str] = []
    lines.append(f"# total_matches: {len(mappings)}")
    lines.append(f"# l1_reuse_matches: {len(l1_mappings)}")
    lines.append(f"# cube_nbuffer_matches: {len(cube_mappings)}")
    lines.append(f"# vec_nbuffer_matches: {len(vec_mappings)}")
    lines.append("")
    lines.append("[L1 reuse]")
    for _, hash_order, subgraph_hash, ids in l1_mappings:
        lines.append(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}")
    lines.append("")
    lines.append("[Cube nbuffer]")
    for _, hash_order, subgraph_hash, ids in cube_mappings:
        lines.append(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}")
    lines.append("")
    lines.append("[Vec nbuffer]")
    for _, hash_order, subgraph_hash, ids in vec_mappings:
        lines.append(f"hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}")
    lines.append("")

    other_mappings = [
        m for m in mappings if m[0] not in {"L1 reuse", "Cube nbuffer", "Vec nbuffer"}
    ]
    if other_mappings:
        lines.append("[Other]")
        for source, hash_order, subgraph_hash, ids in other_mappings:
            lines.append(f"{source} | hash order: {hash_order} | Subgraph hash: {subgraph_hash} | Subgraph IDs: {ids}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_logs_from_dir(root: Path) -> list[Path]:
    """Recursively collect log files from directory.

    Matches:
      - Pass_*_NBufferMerge/*.log
      - Pass_*_L1CopyInReuseMerge/*.log
      - pypto-log*.log
      - plog-*.log
    """
    patterns = [
        "Pass_*_NBufferMerge/*.log",
        "Pass_*_L1CopyInReuseMerge/*.log",
        "pypto-log*.log",
        "plog-*.log",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))

    files = [
        p
        for p in files
        if not p.name.endswith("_HASH_ORDER.log") and not p.name.endswith(".log_HASH_ORDER")
    ]
    return sorted(set(files))


def write_dir_summary(root: Path, logs: list[Path]) -> Path:
    out_path = root / "HASH_ORDER.log"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# root: {root}\n")
        f.write(f"# total_log_files: {len(logs)}\n\n")
        for log_path in logs:
            mappings = parse_file(log_path)
            f.write(f"## source: {log_path.name}\n")
            f.write(format_mappings_block(mappings))
            f.write("\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract hash order to Subgraph IDs mapping from log files."
    )
    parser.add_argument("inputs", nargs="+", help="Input log file paths or root directories")
    args = parser.parse_args()

    exit_code = 0
    for input_name in args.inputs:
        path = Path(input_name)
        if not path.is_file():
            if path.is_dir():
                logs = collect_logs_from_dir(path)
                out_path = write_dir_summary(path, logs)
                print(f"[OK] directory {path} -> {out_path} (log_files: {len(logs)})")
                continue
            print(f"[ERROR] File or directory not found: {path}")
            exit_code = 1
            continue

        mappings = parse_file(path)
        out_path = write_output(path, mappings)
        print(f"[OK] {path} -> {out_path} (matches: {len(mappings)})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())