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
"""CLI entry for runtime dependency verification.

Usage:
    python tools/verify_dep_correctness.py <dump_dir>
"""
import argparse
import logging
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dep_verifier import get_registered_rules  # noqa: E402
from dep_verifier.data_loader import (  # noqa: E402
    infer_edge_seq_no,
    load_dyn_stitch_edges,
    load_dyn_topo,
    load_slot_access,
    load_slot_cell_table,
    load_slot_mapping,
    load_static_topo,
)
from dep_verifier.rule_base import RuleContext  # noqa: E402
from dep_verifier.report import ViolationReport  # noqa: E402


INPUT_FILES = {
    "static_topo":       "static_topo.csv",
    "dyn_topo":          "dyn_topo.txt",
    "stitch_edges":      "dyn_stitch_edges.csv",
    "slot_mapping":      "slot_mapping.csv",
    "slot_cell_table":   "slot_cell_table.csv",
    "slot_access":       "dyn_slot_access.csv",
}
REPORT_NAME = "dep_check_report.csv"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Runtime tensor data-flow verification. Detects three classes of "
            "operator-level problems: (1) concurrent write overlap, "
            "(2) missing producer/consumer dependency, "
            "(3) illegal read/write linkage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The directory should contain the runtime dump files "
            "(static_topo.csv, dyn_topo.txt, dyn_stitch_edges.csv, "
            "slot_mapping.csv, slot_cell_table.csv, dyn_slot_access.csv). "
            "On success, prints 'PASS'; otherwise prints a grouped summary "
            "and writes dep_check_report.csv to the same directory."
        ),
    )
    p.add_argument("dump_dir", help="Directory containing runtime dump files")
    return p.parse_args(argv)


def _resolve(dump_dir: str, key: str) -> Optional[str]:
    path = os.path.join(dump_dir, INPUT_FILES[key])
    return path if os.path.isfile(path) else None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    dump_dir = args.dump_dir
    if not os.path.isdir(dump_dir):
        print(f"ERROR: directory does not exist: {dump_dir}", file=sys.stderr)
        return 2

    static_topo = _resolve(dump_dir, "static_topo")
    dyn_topo = _resolve(dump_dir, "dyn_topo")
    stitch_edges_path = _resolve(dump_dir, "stitch_edges")
    slot_mapping_path = _resolve(dump_dir, "slot_mapping")
    slot_cell_table_path = _resolve(dump_dir, "slot_cell_table")
    slot_access_path = _resolve(dump_dir, "slot_access")

    if not (static_topo and dyn_topo and stitch_edges_path):
        missing = [INPUT_FILES[k] for k, p in [
            ("static_topo", static_topo),
            ("dyn_topo", dyn_topo),
            ("stitch_edges", stitch_edges_path),
        ] if not p]
        print(f"ERROR: missing required files in directory: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        static_fns = load_static_topo(static_topo)
        dyn_tasks = load_dyn_topo(dyn_topo)
        stitch_edges = load_dyn_stitch_edges(stitch_edges_path)
        slot_roles = {}
        slot_tensor_names = {}
        slot_func_names = {}
        slot_cell_tables = {}
        slot_accesses = []
        if slot_mapping_path:
            _, slot_roles, slot_tensor_names, slot_func_names = load_slot_mapping(slot_mapping_path)
        if slot_cell_table_path:
            slot_cell_tables = load_slot_cell_table(slot_cell_table_path)
        if slot_access_path:
            slot_accesses = load_slot_access(slot_access_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: failed to load input files: {exc}", file=sys.stderr)
        return 2

    infer_edge_seq_no(stitch_edges, dyn_tasks)

    ctx = RuleContext(
        static_functions=static_fns,
        dyn_tasks=dyn_tasks,
        stitch_edges=stitch_edges,
        slot_cell_tables=slot_cell_tables,
        slot_accesses=slot_accesses,
        slot_roles=slot_roles,
        slot_tensor_names=slot_tensor_names,
        slot_func_names=slot_func_names,
    )

    report = ViolationReport(
        slot_tensor_names=slot_tensor_names,
        slot_func_names=slot_func_names,
    )
    for cls in get_registered_rules():
        report.extend(cls().run(ctx))

    report.print_console()
    report.save_csv(os.path.join(dump_dir, REPORT_NAME))
    return 1 if report.has_failure() else 0


if __name__ == "__main__":
    sys.exit(main())
