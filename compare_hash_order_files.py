#!/usr/bin/env python3
"""Compare two HASH_ORDER.log files and map old hash orders to new hash orders.

Match key: config section + Subgraph hash.

Usage:
  python tools/compare_hash_order_files.py <old_hash_order_log> <new_hash_order_log> [-o output_file] [--aggregate]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"^##\s+source:\s*(.+)$")
SECTION_RE = re.compile(r"^\[(L1 reuse|Cube nbuffer|Vec nbuffer)\]$")
ENTRY_RE = re.compile(
    r"^hash\s+order:\s*(-?\d+)\s*\|\s*Subgraph\s+hash:\s*(\d+)\s*\|\s*Subgraph\s+IDs:\s*\[([^\]]*)\]\s*$"
)

CONFIGS = ("L1 reuse", "Cube nbuffer", "Vec nbuffer")


def parse_hash_order_file(path: Path) -> dict[str, dict[str, dict[int, tuple[set[int], str]]]]:
    """Parse HASH_ORDER.log file.

    Returns: {source: {section: {subgraph_hash: (set of hash_orders, subgraph_ids_str)}}}
    """
    data: dict[str, dict[str, dict[int, tuple[set[int], str]]]] = {}
    current_source: str | None = None
    current_section: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            source_match = SOURCE_RE.match(line)
            if source_match:
                current_source = source_match.group(1).strip()
                data.setdefault(current_source, {cfg: {} for cfg in CONFIGS})
                current_section = None
                continue

            section_match = SECTION_RE.match(line)
            if section_match and current_source is not None:
                current_section = section_match.group(1)
                continue

            entry_match = ENTRY_RE.match(line)
            if entry_match and current_source is not None and current_section is not None:
                hash_order = int(entry_match.group(1))
                subgraph_hash = int(entry_match.group(2))
                ids_str = entry_match.group(3).strip()
                subgraph_ids = f"[{ids_str}]" if ids_str else "[]"
                # Collect all hashOrders for the same (source, section, subgraph_hash)
                if subgraph_hash in data[current_source][current_section]:
                    existing_ho_set, _ = data[current_source][current_section][subgraph_hash]
                    existing_ho_set.add(hash_order)
                else:
                    data[current_source][current_section][subgraph_hash] = ({hash_order}, subgraph_ids)

    return data


def aggregate_by_config(
    data: dict[str, dict[str, dict[int, tuple[set[int], str]]]]
) -> dict[str, dict[int, tuple[set[int], str]]]:
    """Aggregate hashOrder mappings across all sources by config type.

    Returns: {config: {subgraph_hash: (set of hashOrder values, subgraph_ids_str)}}
    """
    result: dict[str, dict[int, tuple[set[int], str]]] = {cfg: {} for cfg in CONFIGS}
    for source, cfg_data in data.items():
        for cfg in CONFIGS:
            for sg_hash, (ho_set, ids_str) in cfg_data.get(cfg, {}).items():
                if sg_hash in result[cfg]:
                    existing_ho_set, existing_ids = result[cfg][sg_hash]
                    existing_ho_set.update(ho_set)
                else:
                    result[cfg][sg_hash] = (set(ho_set), ids_str)
    return result


def compare(old_data, new_data, aggregate: bool = False) -> list[str]:
    lines: list[str] = []

    if aggregate:
        old_agg = aggregate_by_config(old_data)
        new_agg = aggregate_by_config(new_data)

        lines.append("# mode: aggregate (compare across all sources by Subgraph hash)")
        lines.append("")

        for cfg in CONFIGS:
            lines.append(f"[{cfg}]")
            old_map = old_agg.get(cfg, {})
            new_map = new_agg.get(cfg, {})

            old_hashes = set(old_map.keys())
            new_hashes = set(new_map.keys())
            common_hashes = sorted(old_hashes & new_hashes)
            missing_in_new = sorted(old_hashes - new_hashes)
            added_in_new = sorted(new_hashes - old_hashes)

            if not common_hashes and not missing_in_new and not added_in_new:
                lines.append("(empty)")
                lines.append("")
                continue

            for sg_hash in common_hashes:
                old_ho_set, old_ids = old_map[sg_hash]
                new_ho_set, new_ids = new_map[sg_hash]
                old_ho = "{" + ", ".join(map(str, sorted(old_ho_set))) + "}"
                new_ho = "{" + ", ".join(map(str, sorted(new_ho_set))) + "}"
                lines.append(
                    f"old hash order: {old_ho} -> new hash order: {new_ho} | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {old_ids}"
                )

            for sg_hash in missing_in_new:
                old_ho_set, old_ids = old_map[sg_hash]
                old_ho = "{" + ", ".join(map(str, sorted(old_ho_set))) + "}"
                lines.append(
                    f"old hash order: {old_ho} -> new hash order: N/A | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {old_ids}"
                )

            for sg_hash in added_in_new:
                new_ho_set, new_ids = new_map[sg_hash]
                new_ho = "{" + ", ".join(map(str, sorted(new_ho_set))) + "}"
                lines.append(
                    f"old hash order: N/A -> new hash order: {new_ho} | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {new_ids}"
                )

            lines.append("")

        return lines

    # Original per-source comparison
    old_sources = set(old_data.keys())
    new_sources = set(new_data.keys())
    common_sources = sorted(old_sources & new_sources)
    only_old_sources = sorted(old_sources - new_sources)
    only_new_sources = sorted(new_sources - old_sources)

    lines.append(f"# compared_common_sources: {len(common_sources)}")
    lines.append(f"# only_in_old_sources: {len(only_old_sources)}")
    lines.append(f"# only_in_new_sources: {len(only_new_sources)}")
    lines.append("")

    for source in common_sources:
        lines.append(f"## source: {source}")
        for cfg in CONFIGS:
            lines.append(f"[{cfg}]")
            old_map = old_data[source].get(cfg, {})
            new_map = new_data[source].get(cfg, {})

            old_hashes = set(old_map.keys())
            new_hashes = set(new_map.keys())
            common_hashes = sorted(old_hashes & new_hashes)
            missing_in_new = sorted(old_hashes - new_hashes)
            added_in_new = sorted(new_hashes - old_hashes)

            if not common_hashes and not missing_in_new and not added_in_new:
                lines.append("(empty)")
                lines.append("")
                continue

            for sg_hash in common_hashes:
                old_ho_set, old_ids = old_map[sg_hash]
                new_ho_set, new_ids = new_map[sg_hash]
                old_ho = "{" + ", ".join(map(str, sorted(old_ho_set))) + "}"
                new_ho = "{" + ", ".join(map(str, sorted(new_ho_set))) + "}"
                lines.append(
                    f"old hash order: {old_ho} -> new hash order: {new_ho} | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {old_ids}"
                )

            for sg_hash in missing_in_new:
                old_ho_set, old_ids = old_map[sg_hash]
                old_ho = "{" + ", ".join(map(str, sorted(old_ho_set))) + "}"
                lines.append(
                    f"old hash order: {old_ho} -> new hash order: N/A | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {old_ids}"
                )

            for sg_hash in added_in_new:
                new_ho_set, new_ids = new_map[sg_hash]
                new_ho = "{" + ", ".join(map(str, sorted(new_ho_set))) + "}"
                lines.append(
                    f"old hash order: N/A -> new hash order: {new_ho} | "
                    f"Subgraph hash: {sg_hash} | Subgraph IDs: {new_ids}"
                )

            lines.append("")

    if only_old_sources:
        lines.append("## only_in_old_sources")
        for source in only_old_sources:
            lines.append(source)
        lines.append("")

    if only_new_sources:
        lines.append("## only_in_new_sources")
        for source in only_new_sources:
            lines.append(source)
        lines.append("")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two HASH_ORDER.log files and map old hash order to new hash order."
    )
    parser.add_argument("old_file", help="Old HASH_ORDER.log path")
    parser.add_argument("new_file", help="New HASH_ORDER.log path")
    parser.add_argument(
        "-o",
        "--output",
        default="HASH_ORDER_COMPARE.log",
        help="Output file path (default: HASH_ORDER_COMPARE.log)",
    )
    parser.add_argument(
        "-a",
        "--aggregate",
        action="store_true",
        help="Aggregate all sources and compare by Subgraph hash (ignore source file names).",
    )
    args = parser.parse_args()

    old_path = Path(args.old_file)
    new_path = Path(args.new_file)
    out_path = Path(args.output)

    if not old_path.is_file():
        print(f"[ERROR] Old file not found: {old_path}")
        return 1
    if not new_path.is_file():
        print(f"[ERROR] New file not found: {new_path}")
        return 1

    old_data = parse_hash_order_file(old_path)
    new_data = parse_hash_order_file(new_path)

    lines = [f"# old_file: {old_path}", f"# new_file: {new_path}", ""]
    lines.extend(compare(old_data, new_data, aggregate=args.aggregate))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[OK] {old_path} vs {new_path} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
