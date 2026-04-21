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

输入（必选）：
  --static-topo  <static_topo.csv>        编译期静态依赖
  --dyn-topo     <dyn_topo.txt>           运行时全量 topo
  --stitch-edges <dyn_stitch_edges.csv>   运行时 stitch 边详细记录

输入（可选，cell 粒度规则 4-α/β/γ 所需；不提供则自动跳过）：
  --slot-cell-table <slot_cell_table.csv> partial/fullcover slot cell 表元信息
  --slot-access     <dyn_slot_access.csv> 实例级 cell R/W 事件流

可选：
  --enable / --disable   启用/禁用指定规则（多次指定或逗号分隔）
  --list-rules           列出全部已注册规则并退出
  --input-slots          program 输入 slot 列表（规则 4-γ 豁免）
  --output-slots         program 输出 slot 列表（规则 4-β 豁免）
  --zero-init-slots      零初始化 slot 列表（规则 4-γ 豁免）
  --json-report          输出 JSON 报告路径
  --csv-report           输出 CSV 报告路径
  --max-per-rule         控制台每条规则最多打印多少违规（默认 50）

退出码：
  0  校验通过（无 ERROR / SUSPECT-HIGH）
  1  存在 ERROR 或 SUSPECT-HIGH 级别违规
  2  参数或文件加载错误
"""
import argparse
import logging
import sys
from typing import List, Optional, Set

# 允许 "python tools/verify_dep_correctness.py" 方式直接运行
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dep_verifier import get_registered_rules  # noqa: E402
from dep_verifier.data_loader import (  # noqa: E402
    infer_edge_seq_no,
    load_dyn_stitch_edges,
    load_dyn_topo,
    load_slot_access,
    load_slot_cell_table,
    load_static_topo,
)
from dep_verifier.rule_base import RuleContext  # noqa: E402
from dep_verifier.report import ViolationReport  # noqa: E402


def _parse_slot_option(values: List[str]) -> Set[int]:
    out: Set[int] = set()
    if not values:
        return out
    for v in values:
        for part in v.replace(",", " ").split():
            part = part.strip()
            if part:
                out.add(int(part))
    return out


def _parse_rule_filter(values: List[str]) -> Set[str]:
    out: Set[str] = set()
    if not values:
        return out
    for v in values:
        for part in v.replace(",", " ").split():
            part = part.strip()
            if part:
                out.add(part)
    return out


def _select_rules(enable: Set[str], disable: Set[str]):
    """根据 --enable/--disable 过滤规则。

    - 未指定 --enable：默认启用所有 DEFAULT_ENABLED=True 的规则
    - 指定 --enable：只启用显式列出的规则
    - --disable 总是被剔除
    """
    all_rules = get_registered_rules()
    if enable:
        selected = [r for r in all_rules if r.RULE_ID in enable]
        missing = enable - {r.RULE_ID for r in all_rules}
        if missing:
            logging.warning("以下规则 id 未注册，已忽略: %s", ", ".join(sorted(missing)))
    else:
        selected = [r for r in all_rules if r.DEFAULT_ENABLED]
    selected = [r for r in selected if r.RULE_ID not in disable]
    return selected


def _print_rule_list():
    rules = get_registered_rules()
    print(f"当前已注册规则共 {len(rules)} 条：")
    for cls in rules:
        flag = "on " if cls.DEFAULT_ENABLED else "off"
        print(f"  [{flag}] {cls.RULE_ID:<35} {cls.DESCRIPTION}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行时依赖边建立正确性校验（规则 1/2/3/4）",
    )
    parser.add_argument("--static-topo", help="static_topo.csv 路径")
    parser.add_argument("--dyn-topo", help="dyn_topo.txt 路径")
    parser.add_argument("--stitch-edges", help="dyn_stitch_edges.csv 路径")
    parser.add_argument("--slot-cell-table",
                        help="slot_cell_table.csv 路径（规则 4-α/β/γ 所需；不提供则相关规则自动跳过）")
    parser.add_argument("--slot-access",
                        help="dyn_slot_access.csv 路径（规则 4-α/β/γ 所需；不提供则相关规则自动跳过）")

    parser.add_argument("--enable", action="append", default=[],
                        help="启用的规则 id（可多次指定或逗号分隔）")
    parser.add_argument("--disable", action="append", default=[],
                        help="禁用的规则 id（可多次指定或逗号分隔）")
    parser.add_argument("--list-rules", action="store_true",
                        help="列出全部已注册规则并退出")

    parser.add_argument("--input-slots", action="append", default=[],
                        help="program 输入 slot（规则 4-B 豁免），逗号或空格分隔")
    parser.add_argument("--output-slots", action="append", default=[],
                        help="program 输出 slot（规则 4-A 豁免）")
    parser.add_argument("--zero-init-slots", action="append", default=[],
                        help="零初始化 slot（规则 4-B 豁免）")

    parser.add_argument("--json-report", help="输出 JSON 报告路径")
    parser.add_argument("--csv-report", help="输出 CSV 报告路径")
    parser.add_argument("--max-per-rule", type=int, default=50,
                        help="控制台每规则最多打印的违规条数（默认 50）")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG 级别日志")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.list_rules:
        _print_rule_list()
        return 0

    if not (args.static_topo and args.dyn_topo and args.stitch_edges):
        logging.error("必须同时提供 --static-topo / --dyn-topo / --stitch-edges，"
                      "或使用 --list-rules 查看可用规则")
        return 2

    try:
        static_fns = load_static_topo(args.static_topo)
        dyn_tasks = load_dyn_topo(args.dyn_topo)
        stitch_edges = load_dyn_stitch_edges(args.stitch_edges)
        slot_cell_tables = {}
        slot_accesses = []
        if args.slot_cell_table:
            slot_cell_tables = load_slot_cell_table(args.slot_cell_table)
        if args.slot_access:
            slot_accesses = load_slot_access(args.slot_access)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("加载输入文件失败: %s", exc)
        return 2

    infer_edge_seq_no(stitch_edges, dyn_tasks)

    if (args.slot_cell_table or args.slot_access) and not (args.slot_cell_table and args.slot_access):
        logging.warning(
            "--slot-cell-table 与 --slot-access 一般成对使用；"
            "当前只提供了一个，cell 级规则可能无法完整校验。")

    ctx = RuleContext(
        static_functions=static_fns,
        dyn_tasks=dyn_tasks,
        stitch_edges=stitch_edges,
        slot_cell_tables=slot_cell_tables,
        slot_accesses=slot_accesses,
        program_input_slots=_parse_slot_option(args.input_slots),
        program_output_slots=_parse_slot_option(args.output_slots),
        zero_init_slots=_parse_slot_option(args.zero_init_slots),
    )

    enable = _parse_rule_filter(args.enable)
    disable = _parse_rule_filter(args.disable)
    rule_classes = _select_rules(enable, disable)
    if not rule_classes:
        logging.error("没有启用任何规则，退出")
        return 2

    logging.info("将执行以下规则: %s",
                 ", ".join(r.RULE_ID for r in rule_classes))

    report = ViolationReport()
    for cls in rule_classes:
        rule = cls()
        violations = rule.run(ctx)
        report.extend(cls.RULE_ID, violations)

    report.print_console(max_per_rule=args.max_per_rule)
    if args.json_report:
        report.save_json(args.json_report)
    if args.csv_report:
        report.save_csv(args.csv_report)

    return 1 if report.has_failure() else 0


if __name__ == "__main__":
    sys.exit(main())
