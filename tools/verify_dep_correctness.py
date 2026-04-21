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
运行时依赖边建立正确性校验 —— CLI 入口。

用法:
    python tools/verify_dep_correctness.py <dump_dir>

dump_dir 目录下应包含运行时生成的以下文件（缺某个文件对应规则自动跳过）：
    static_topo.csv         规则 1/2 必需
    dyn_topo.txt            规则 1/2/3 必需
    dyn_stitch_edges.csv    规则 2/3 必需
    slot_mapping.csv        规则 4-β/γ 识别 INPUT/OUTPUT slot 所需
    slot_cell_table.csv     规则 4 所需
    dyn_slot_access.csv     规则 4 所需

输出：
    控制台：PASS 或 FAIL + 每 (rule, slot) 一行摘要
    文件  ：<dump_dir>/dep_check_report.csv
退出码：0 = PASS，1 = FAIL，2 = 参数或加载错误。
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
        description="运行时依赖边建立正确性校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="目录下需包含 static_topo.csv / dyn_topo.txt / dyn_stitch_edges.csv "
               "/ slot_mapping.csv / slot_cell_table.csv / dyn_slot_access.csv。",
    )
    p.add_argument("dump_dir", help="运行时 dump 文件所在目录")
    return p.parse_args(argv)


def _resolve(dump_dir: str, key: str) -> Optional[str]:
    path = os.path.join(dump_dir, INPUT_FILES[key])
    return path if os.path.isfile(path) else None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    dump_dir = args.dump_dir
    if not os.path.isdir(dump_dir):
        print(f"ERROR: 目录不存在: {dump_dir}", file=sys.stderr)
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
        print(f"ERROR: 目录中缺失必需文件: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        static_fns = load_static_topo(static_topo)
        dyn_tasks = load_dyn_topo(dyn_topo)
        stitch_edges = load_dyn_stitch_edges(stitch_edges_path)
        slot_roles = {}
        slot_cell_tables = {}
        slot_accesses = []
        if slot_mapping_path:
            _, slot_roles = load_slot_mapping(slot_mapping_path)
        if slot_cell_table_path:
            slot_cell_tables = load_slot_cell_table(slot_cell_table_path)
        if slot_access_path:
            slot_accesses = load_slot_access(slot_access_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: 加载输入文件失败: {exc}", file=sys.stderr)
        return 2

    infer_edge_seq_no(stitch_edges, dyn_tasks)

    ctx = RuleContext(
        static_functions=static_fns,
        dyn_tasks=dyn_tasks,
        stitch_edges=stitch_edges,
        slot_cell_tables=slot_cell_tables,
        slot_accesses=slot_accesses,
        slot_roles=slot_roles,
    )

    report = ViolationReport()
    for cls in get_registered_rules():
        report.extend(cls().run(ctx))

    report.print_console()
    report.save_csv(os.path.join(dump_dir, REPORT_NAME))
    return 1 if report.has_failure() else 0


if __name__ == "__main__":
    sys.exit(main())
