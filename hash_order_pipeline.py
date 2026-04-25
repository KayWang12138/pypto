#!/usr/bin/env python3
"""Pipeline for HASH_ORDER extraction + comparison.

Flow:
1) Run extract script on old directory.
2) Run extract script on new directory.
3) Run compare script on generated HASH_ORDER.log files.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT = THIS_DIR / "extract_hash_order_mapping.py"
COMPARE_SCRIPT = THIS_DIR / "compare_hash_order_files.py"
SECTION_RE = re.compile(r"^\[(L1 reuse|Cube nbuffer|Vec nbuffer)\]$")
MAPPING_RE = re.compile(
    r"^old hash order:\s*(\{[^}]+\}|\S+)\s*->\s*new hash order:\s*(\{[^}]+\}|\S+)\s*\|\s*"
    r"Subgraph hash:\s*(\d+)\s*\|\s*Subgraph IDs:\s*(\[.*\])$"
)
SET_VALUE_RE = re.compile(r"(-?\d+)")


def parse_hash_order_set(value_str: str) -> set[str]:
    """Parse hash order value which may be a set like '{0, 1, 2}' or a single value."""
    if value_str.startswith("{"):
        return set(SET_VALUE_RE.findall(value_str))
    elif value_str == "N/A":
        return set()
    else:
        return {value_str}
FILTER_ENTRY_RE = re.compile(r"(-?\d+)\s*:")
FILTER_PAIR_RE = re.compile(r"(-?\d+)\s*:\s*([^,}]+)")
CFG_KEY_TO_SECTION = {
    "cube_l1_reuse_setting": "L1 reuse",
    "cube_nbuffer_setting": "Cube nbuffer",
    "vec_nbuffer_setting": "Vec nbuffer",
}
SECTION_TO_CFG_KEY = {v: k for k, v in CFG_KEY_TO_SECTION.items()}
SECTION_TO_JSON_KEY = {
    "L1 reuse": "l1ReuseHashOrder",
    "Cube nbuffer": "cubeNBufferHashOrder",
    "Vec nbuffer": "vecNBufferHashOrder",
}


def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def extract_valid_hash_orders_from_json(
    json_path: Path,
) -> dict[str, set[int]]:
    """Extract valid hashOrder values from a merged_swimlane.json file.

    Returns a dict mapping section name -> set of valid hashOrder values.
    """
    import json as _json

    with json_path.open("r", encoding="utf-8", errors="ignore") as f:
        data = _json.load(f)

    valid: dict[str, set[int]] = {cfg: set() for cfg in SECTION_TO_JSON_KEY}
    events = data.get("traceEvents", [])
    for event in events:
        args = event.get("args", {})
        for section, json_key in SECTION_TO_JSON_KEY.items():
            if json_key in args:
                valid[section].add(args[json_key])
    return valid


def parse_filter_config(
    config_text: str,
) -> tuple[dict[str, set[str] | None], dict[str, dict[str, str]]]:
    selected: dict[str, set[str] | None] = {
        "L1 reuse": None,
        "Cube nbuffer": None,
        "Vec nbuffer": None,
    }
    value_map_by_section: dict[str, dict[str, str]] = {
        "L1 reuse": {},
        "Cube nbuffer": {},
        "Vec nbuffer": {},
    }
    for cfg_key, section in CFG_KEY_TO_SECTION.items():
        # Support both `key={...}` and `"key": {...}` styles.
        match = re.search(rf'["\']?{re.escape(cfg_key)}["\']?\s*[:=]\s*\{{([^}}]*)\}}', config_text)
        if not match:
            continue
        inside = match.group(1)
        old_hash_orders = {m.group(1) for m in FILTER_ENTRY_RE.finditer(inside)}
        pairs = {m.group(1): m.group(2).strip() for m in FILTER_PAIR_RE.finditer(inside)}
        value_map_by_section[section] = pairs
        if not old_hash_orders:
            continue
        if "-1" in old_hash_orders:
            selected[section] = None
        else:
            selected[section] = old_hash_orders
    return selected, value_map_by_section


def filter_compare_output(
    compare_file: Path,
    selected_old_hash_orders: dict[str, set[str] | None],
) -> None:
    lines_out: list[str] = []
    current_section: str | None = None
    with compare_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            section_match = SECTION_RE.match(line.strip())
            if section_match:
                current_section = section_match.group(1)
                lines_out.append(raw_line)
                continue

            mapping_match = MAPPING_RE.match(line.strip())
            if mapping_match and current_section is not None:
                old_ho_str = mapping_match.group(1)
                old_ho_set = parse_hash_order_set(old_ho_str)
                selected = selected_old_hash_orders.get(current_section)
                # Filter: keep if any old hashOrder is in selected set
                if selected is not None and not old_ho_set.intersection(selected):
                    continue
            lines_out.append(raw_line)

    with compare_file.open("w", encoding="utf-8") as f:
        f.writelines(lines_out)


def summarize_compare(
    compare_file: Path,
    selected_old_hash_orders: dict[str, set[str] | None] | None = None,
    old_value_map_by_section: dict[str, dict[str, str]] | None = None,
    valid_hash_orders: dict[str, set[int]] | None = None,
) -> Path:
    # Mapping per Subgraph hash: {section: {sg_hash: (old_ho_set, new_ho_set, ids_str)}}
    mappings_by_cfg: dict[str, dict[int, tuple[set[str], set[str], str]]] = {
        "L1 reuse": {},
        "Cube nbuffer": {},
        "Vec nbuffer": {},
    }
    current_cfg: str | None = None

    with compare_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            section_match = SECTION_RE.match(line)
            if section_match:
                current_cfg = section_match.group(1)
                continue

            match = MAPPING_RE.match(line)
            if not match or current_cfg is None:
                continue

            old_ho_str = match.group(1)
            new_ho_str = match.group(2)
            sg_hash = int(match.group(3))
            ids_str = match.group(4)

            old_ho_set = parse_hash_order_set(old_ho_str)
            new_ho_set = parse_hash_order_set(new_ho_str)

            # Merge with existing entry for this Subgraph hash
            if sg_hash in mappings_by_cfg[current_cfg]:
                existing_old, existing_new, _ = mappings_by_cfg[current_cfg][sg_hash]
                existing_old.update(old_ho_set)
                existing_new.update(new_ho_set)
            else:
                mappings_by_cfg[current_cfg][sg_hash] = (old_ho_set, new_ho_set, ids_str)

    # Also build old hashOrder -> new hashOrders mapping (for config suggestion)
    old_to_new_by_cfg: dict[str, dict[str, set[str]]] = {
        "L1 reuse": defaultdict(set),
        "Cube nbuffer": defaultdict(set),
        "Vec nbuffer": defaultdict(set),
    }
    for cfg in ("L1 reuse", "Cube nbuffer", "Vec nbuffer"):
        for sg_hash, (old_set, new_set, _) in mappings_by_cfg[cfg].items():
            for old_val in old_set:
                old_to_new_by_cfg[cfg][old_val].update(new_set)

    summary_path = compare_file.with_name(f"{compare_file.stem}_SUMMARY{compare_file.suffix}")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# source_compare_file: {compare_file}\n")
        f.write(
            "# unique_subgraph_hashes_l1_reuse: "
            f"{len(mappings_by_cfg['L1 reuse'])}\n"
        )
        f.write(
            "# unique_subgraph_hashes_cube_nbuffer: "
            f"{len(mappings_by_cfg['Cube nbuffer'])}\n"
        )
        f.write(
            "# unique_subgraph_hashes_vec_nbuffer: "
            f"{len(mappings_by_cfg['Vec nbuffer'])}\n\n"
        )

        def order_key(val: str) -> tuple[int, int | str]:
            if val.lstrip("-").isdigit():
                return (0, int(val))
            return (1, val)

        for cfg in ("L1 reuse", "Cube nbuffer", "Vec nbuffer"):
            f.write(f"[{cfg}]\n")
            cfg_mappings = mappings_by_cfg[cfg]
            if not cfg_mappings:
                f.write("(empty)\n\n")
                continue

            for sg_hash in sorted(cfg_mappings.keys()):
                old_set, new_set, ids_str = cfg_mappings[sg_hash]
                old_sorted = sorted(old_set, key=order_key)
                new_sorted = sorted(new_set, key=order_key)
                f.write(
                    f"Subgraph hash: {sg_hash} | "
                    f"old hash orders: {{{', '.join(old_sorted)}}} -> "
                    f"new hash orders: {{{', '.join(new_sorted)}}} | "
                    f"Subgraph IDs: {ids_str}\n"
                )
            f.write("\n")

        if selected_old_hash_orders is not None:
            old_value_map_by_section = old_value_map_by_section or {
                "L1 reuse": {},
                "Cube nbuffer": {},
                "Vec nbuffer": {},
            }

            def parse_value(raw: str) -> int | str:
                val = raw.strip()
                if val.lstrip("-").isdigit():
                    return int(val)
                return val

            def build_new_cfg_dict(section: str, filter_invalid: bool = True) -> dict[str, int | str]:
                src_values = old_value_map_by_section.get(section, {})
                selected = selected_old_hash_orders.get(section)
                # Get valid hashOrder set for this section from swimlane JSON
                valid_set = valid_hash_orders.get(section) if valid_hash_orders else None

                # Build mapping from Subgraph hash level
                # Each Subgraph hash contributes: each old hashOrder in its set maps to each new hashOrder
                sg_hash_contributions: dict[str, list[tuple[int, int]]] = defaultdict(list)
                for sg_hash, (old_set, new_set, _) in mappings_by_cfg[section].items():
                    for old_val in old_set:
                        if old_val == "N/A" or not old_val.lstrip("-").isdigit():
                            continue
                        if old_val not in src_values:
                            continue
                        if selected is not None and old_val not in selected:
                            continue
                        target_value = parse_value(src_values[old_val])
                        for new_val in new_set:
                            if new_val == "N/A" or not new_val.lstrip("-").isdigit():
                                continue
                            new_int = int(new_val)
                            if filter_invalid and valid_set is not None and new_int not in valid_set:
                                continue
                            sg_hash_contributions[old_val].append((new_int, target_value, sg_hash))

                # For config, pick the first Subgraph hash's contribution for each old hashOrder
                result: dict[str, int | str] = {}
                if "-1" in src_values:
                    result["-1"] = parse_value(src_values["-1"])

                for old_val, contributions in sorted(sg_hash_contributions.items(), key=lambda x: int(x[0])):
                    # Add ALL new hashOrders mapped from this old hashOrder
                    # (All contributions for same old_val share the same target_value from old config)
                    for new_int, target_value, _ in contributions:
                        result[str(new_int)] = target_value
                return result

            def build_old_cfg_dict(section: str) -> dict[str, int | str]:
                src_values = old_value_map_by_section.get(section, {})
                if not src_values:
                    return {}
                keys = sorted(src_values.keys(), key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
                return {k: parse_value(src_values[k]) for k in keys}

            def render_scalar(value: int | str) -> str:
                if isinstance(value, int):
                    return str(value)
                if value.lstrip("-").isdigit():
                    return value
                return f"\"{value}\""

            def render_inner_dict(mapping: dict[str, int | str]) -> str:
                if not mapping:
                    return "{}"
                # Sort by numeric key for consistent output
                items = sorted(mapping.items(), key=lambda x: int(x[0]) if x[0].lstrip("-").isdigit() else x[0])
                parts: list[str] = []
                for k, v in items:
                    key_repr = k if k.lstrip("-").isdigit() else f"\"{k}\""
                    parts.append(f"{key_repr}: {render_scalar(v)}")
                return "{" + ", ".join(parts) + "}"

            def render_config_object(config_obj: dict[str, dict[str, int | str]]) -> str:
                lines = ["{"]
                items = list(config_obj.items())
                for idx, (cfg_key, cfg_value) in enumerate(items):
                    comma = "," if idx < len(items) - 1 else ""
                    inner = render_inner_dict(cfg_value)
                    lines.append(f"  \"{cfg_key}\": {inner}{comma}")
                lines.append("}")
                return "\n".join(lines)

            old_config_json = {
                SECTION_TO_CFG_KEY["L1 reuse"]: build_old_cfg_dict("L1 reuse"),
                SECTION_TO_CFG_KEY["Cube nbuffer"]: build_old_cfg_dict("Cube nbuffer"),
                SECTION_TO_CFG_KEY["Vec nbuffer"]: build_old_cfg_dict("Vec nbuffer"),
            }
            f.write("# input_old_config\n")
            f.write(render_config_object(old_config_json))
            f.write("\n")
            f.write("\n")

            # Generate unfiltered config (all mapped hashOrders)
            suggested_config_unfiltered = {
                SECTION_TO_CFG_KEY["L1 reuse"]: build_new_cfg_dict("L1 reuse", filter_invalid=False),
                SECTION_TO_CFG_KEY["Cube nbuffer"]: build_new_cfg_dict("Cube nbuffer", filter_invalid=False),
                SECTION_TO_CFG_KEY["Vec nbuffer"]: build_new_cfg_dict("Vec nbuffer", filter_invalid=False),
            }
            suggested_config_unfiltered = {
                k: v for k, v in suggested_config_unfiltered.items() if v
            }
            f.write("# suggested_new_config_unfiltered\n")
            f.write(render_config_object(suggested_config_unfiltered))
            f.write("\n")
            f.write("\n")

            # Generate filtered config (only valid hashOrders from swimlane JSON)
            suggested_config_filtered = {
                SECTION_TO_CFG_KEY["L1 reuse"]: build_new_cfg_dict("L1 reuse", filter_invalid=True),
                SECTION_TO_CFG_KEY["Cube nbuffer"]: build_new_cfg_dict("Cube nbuffer", filter_invalid=True),
                SECTION_TO_CFG_KEY["Vec nbuffer"]: build_new_cfg_dict("Vec nbuffer", filter_invalid=True),
            }
            suggested_config_filtered = {
                k: v for k, v in suggested_config_filtered.items() if v
            }
            f.write("# suggested_new_config_filtered\n")
            f.write(render_config_object(suggested_config_filtered))
            f.write("\n")

        # Write valid hashOrders from swimlane JSON if provided
        if valid_hash_orders:
            f.write("\n")
            f.write("# valid_hash_orders_from_swimlane_json\n")
            for section in ("L1 reuse", "Cube nbuffer", "Vec nbuffer"):
                vals = sorted(valid_hash_orders.get(section, set()))
                f.write(f"# {section}: {vals}\n")

    return summary_path


DEFAULT_OLD_DIR = "/mnt/workspace/gitCode/Catherie/bak/pypto/log/debug/plog"
DEFAULT_NEW_DIR = "/mnt/workspace/gitCode/Catherie/pypto/log/debug/plog"


def cleanup_log_files(log_dir: Path) -> int:
    """Remove all .log files in the given directory except HASH_ORDER.log."""
    count = 0
    for f in log_dir.glob("*.log"):
        if f.name != "HASH_ORDER.log":
            f.unlink()
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run extraction for two folders then compare HASH_ORDER results."
    )
    parser.add_argument(
        "old_dir",
        nargs="?",
        default=DEFAULT_OLD_DIR,
        help=f"Old output root directory (default: {DEFAULT_OLD_DIR})",
    )
    parser.add_argument(
        "new_dir",
        nargs="?",
        default=DEFAULT_NEW_DIR,
        help=f"New output root directory (default: {DEFAULT_NEW_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Compare output file path. Default: <cwd>/HASH_ORDER_COMPARE.log",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help=(
            "Enable old hash order filtering via interactive config input. "
            "Use -1 in a config to keep all for that section."
        ),
    )
    parser.add_argument(
        "-j",
        "--swimlane-json",
        default=None,
        help="Path to merged_swimlane.json for deduplication. "
        "New hashOrders not found in this file will be filtered out from the suggested config.",
    )
    args = parser.parse_args()

    old_dir = Path(args.old_dir).resolve()
    new_dir = Path(args.new_dir).resolve()

    if not old_dir.is_dir():
        print(f"[ERROR] old_dir is not a directory: {old_dir}")
        return 1
    if not new_dir.is_dir():
        print(f"[ERROR] new_dir is not a directory: {new_dir}")
        return 1

    if not EXTRACT_SCRIPT.is_file():
        print(f"[ERROR] extract script not found: {EXTRACT_SCRIPT}")
        return 1
    if not COMPARE_SCRIPT.is_file():
        print(f"[ERROR] compare script not found: {COMPARE_SCRIPT}")
        return 1

    selected_old_hash_orders = {
        "L1 reuse": None,
        "Cube nbuffer": None,
        "Vec nbuffer": None,
    }
    old_value_map_by_section = {
        "L1 reuse": {},
        "Cube nbuffer": {},
        "Vec nbuffer": {},
    }
    print("\n\n*****")

    if args.filter:
        print(
            "[INPUT] Paste config (example: "
            "cube_l1_reuse_setting={1: 4, 3: 4}, "
            "vec_nbuffer_setting={2: 3}, "
            '\"cube_nbuffer_setting\": {-1: 4, 0: 1, 1: 1})'
        )
        print("Please Enter:")
        print("")
        config_text = input().strip()
        selected_old_hash_orders, old_value_map_by_section = parse_filter_config(config_text)
        print(
            "[INFO] Filter parsed: "
            f"L1 reuse={selected_old_hash_orders['L1 reuse']}, "
            f"Cube nbuffer={selected_old_hash_orders['Cube nbuffer']}, "
            f"Vec nbuffer={selected_old_hash_orders['Vec nbuffer']}"
        )

    try:
        # Load valid hashOrders from swimlane JSON if provided
        valid_hash_orders: dict[str, set[int]] | None = None
        if args.swimlane_json:
            swimlane_path = Path(args.swimlane_json).resolve()
            if not swimlane_path.is_file():
                print(f"[ERROR] swimlane JSON not found: {swimlane_path}")
                return 1
            valid_hash_orders = extract_valid_hash_orders_from_json(swimlane_path)
            for section in ("L1 reuse", "Cube nbuffer", "Vec nbuffer"):
                print(
                    f"[INFO] Valid hashOrders from JSON [{section}]: "
                    f"{sorted(valid_hash_orders[section])}"
                )

        print(f"[STEP] Extract old folder: {old_dir}")
        run_cmd([sys.executable, str(EXTRACT_SCRIPT), str(old_dir)])

        print(f"[STEP] Extract new folder: {new_dir}")
        run_cmd([sys.executable, str(EXTRACT_SCRIPT), str(new_dir)])

        old_hash_order = old_dir / "HASH_ORDER.log"
        new_hash_order = new_dir / "HASH_ORDER.log"
        if not old_hash_order.is_file():
            print(f"[ERROR] Missing file: {old_hash_order}")
            return 1
        if not new_hash_order.is_file():
            print(f"[ERROR] Missing file: {new_hash_order}")
            return 1

        compare_output = (
            Path(args.output).resolve() if args.output else Path.cwd() / "HASH_ORDER_COMPARE.log"
        )

        print(f"[STEP] Compare: {old_hash_order} vs {new_hash_order}")
        run_cmd([
            sys.executable,
            str(COMPARE_SCRIPT),
            str(old_hash_order),
            str(new_hash_order),
            "-o",
            str(compare_output),
            "--aggregate",
        ])

        if args.filter:
            filter_compare_output(compare_output, selected_old_hash_orders)
            print(f"[OK] compare output filtered: {compare_output}")

        print(f"[OK] compare output: {compare_output}")
        summary_output = summarize_compare(
            compare_output,
            selected_old_hash_orders if args.filter else None,
            old_value_map_by_section if args.filter else None,
            valid_hash_orders,
        )
        print(f"[OK] summary output: {summary_output}")

        # # Cleanup log files in old and new directories
        # old_cleaned = cleanup_log_files(old_dir)
        # new_cleaned = cleanup_log_files(new_dir)
        # print(f"[CLEANUP] Removed {old_cleaned} log files from old_dir")
        # print(f"[CLEANUP] Removed {new_cleaned} log files from new_dir")

        return 0
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())