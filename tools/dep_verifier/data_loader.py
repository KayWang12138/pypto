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
三个 dump 文件解析器。

输出一个 RuleContext（见 rule_base.py），所有规则共享同一份只读上下文。
"""
import csv
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    CellTableDesc,
    DynTask,
    SlotAccessEvent,
    StaticFunction,
    StaticOp,
    StitchEdge,
)

logger = logging.getLogger(__name__)


def _parse_slot_list(cell: str) -> List[int]:
    """解析形如 "[7;8]" / "[]" / "" 的 slot 列表字段。"""
    if cell is None:
        return []
    s = cell.strip()
    if not s or s == "[]":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    parts = s.replace(",", ";").split(";")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            logger.warning("无法解析 slot 值: %r", p)
    return out


def _parse_int_list(cells: List[str]) -> List[int]:
    out = []
    for c in cells:
        c = (c or "").strip()
        if not c:
            continue
        try:
            out.append(int(c))
        except ValueError:
            logger.warning("无法解析整数列表项: %r", c)
    return out


def _parse_task_id_list_semi(cell: str) -> List[int]:
    """解析 dyn_stitch_edges.csv 中的 "65539;65540" 格式。"""
    if not cell or not cell.strip():
        return []
    return [int(p) for p in cell.split(";") if p.strip()]


def load_static_topo(path: str) -> Dict[int, StaticFunction]:
    """
    解析 static_topo.csv。

    表头格式：
      funcKey,rootHash,rawName,opIdx,opmagic,leafHash,coreType,psgId,
      incastSlots,outcastSlots,staticSuccessors...

    staticSuccessors 之后的列都是后继 opIdx（同一 function 内）。
    """
    logger.info("加载 static_topo: %s", path)
    functions: Dict[int, StaticFunction] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"static_topo 文件为空: {path}")

        # 固定列索引
        col = {name: i for i, name in enumerate(header)}
        fixed_tail = col["staticSuccessors"]

        for row in reader:
            if not row or not row[0].strip():
                continue
            func_key = int(row[col["funcKey"]])
            root_hash = row[col["rootHash"]]
            raw_name = row[col["rawName"]]
            op_idx = int(row[col["opIdx"]])
            opmagic = int(row[col["opmagic"]])
            leaf_hash = row[col["leafHash"]]
            core_type = int(row[col["coreType"]])
            psg_id = int(row[col["psgId"]])
            incast = _parse_slot_list(row[col["incastSlots"]])
            outcast = _parse_slot_list(row[col["outcastSlots"]])
            static_succ = _parse_int_list(row[fixed_tail:])

            op = StaticOp(
                func_key=func_key,
                root_hash=root_hash,
                raw_name=raw_name,
                op_idx=op_idx,
                opmagic=opmagic,
                leaf_hash=leaf_hash,
                core_type=core_type,
                psg_id=psg_id,
                incast_slots=incast,
                outcast_slots=outcast,
                static_successors_op_idx=static_succ,
            )

            fn = functions.get(func_key)
            if fn is None:
                fn = StaticFunction(func_key=func_key, root_hash=root_hash, raw_name=raw_name)
                functions[func_key] = fn
            fn.ops[op_idx] = op

    logger.info("  加载完成: %d 个 function, %d 个 op",
                len(functions),
                sum(len(fn.ops) for fn in functions.values()))
    return functions


def load_dyn_topo(path: str) -> List[DynTask]:
    """
    解析 dyn_topo.txt。

    表头格式：
      seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,
      psgId,wrapId,staticSuccCount,successors...
    """
    logger.info("加载 dyn_topo: %s", path)
    tasks: List[DynTask] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"dyn_topo 文件为空: {path}")

        col = {name: i for i, name in enumerate(header)}
        succ_start = col["successors"]

        for row in reader:
            if not row or not row[0].strip():
                continue
            succ = _parse_int_list(row[succ_start:])
            tasks.append(DynTask(
                seq_no=int(row[col["seqNo"]]),
                task_id=int(row[col["taskId"]]),
                root_index=int(row[col["rootIndex"]]),
                root_hash=row[col["rootHash"]],
                opmagic=int(row[col["opmagic"]]),
                leaf_index=int(row[col["leafIndex"]]),
                leaf_hash=row[col["leafHash"]],
                core_type=int(row[col["coreType"]]),
                psg_id=int(row[col["psgId"]]),
                wrap_id=int(row[col["wrapId"]]),
                static_succ_count=int(row[col["staticSuccCount"]]),
                successors=succ,
            ))

    logger.info("  加载完成: %d 个 task 实例", len(tasks))
    return tasks


def load_dyn_stitch_edges(path: str) -> List[StitchEdge]:
    """
    解析 dyn_stitch_edges.csv。
    """
    logger.info("加载 dyn_stitch_edges: %s", path)
    edges: List[StitchEdge] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"dyn_stitch_edges 文件为空: {path}")
        col = {name: i for i, name in enumerate(header)}

        def cell(row, key, default=""):
            idx = col.get(key)
            if idx is None or idx >= len(row):
                return default
            return row[idx]

        for row in reader:
            if not row or not row[0].strip():
                continue
            edges.append(StitchEdge(
                stitch_kind=cell(row, "stitchKind"),
                slot_idx=int(cell(row, "slotIdx")),
                producer_root_hash=cell(row, "producerRootHash"),
                producer_func_key=int(cell(row, "producerFuncKey")),
                producer_func_idx=int(cell(row, "producerFuncIdx")),
                producer_op_idx=int(cell(row, "producerOpIdx")),
                producer_opmagic=int(cell(row, "producerOpmagic")),
                producer_task_id=int(cell(row, "producerTaskId")),
                producer_raw_name=cell(row, "producerRawName"),
                consumer_root_hash=cell(row, "consumerRootHash"),
                consumer_func_key=int(cell(row, "consumerFuncKey")),
                consumer_func_idx=int(cell(row, "consumerFuncIdx")),
                consumer_op_idx=int(cell(row, "consumerOpIdx")),
                consumer_opmagic=int(cell(row, "consumerOpmagic")),
                consumer_task_id=int(cell(row, "consumerTaskId")),
                consumer_raw_name=cell(row, "consumerRawName"),
                producer_static_succ_task_ids=_parse_task_id_list_semi(
                    cell(row, "producerStaticSuccTaskIds")),
                producer_stitch_succ_task_ids=_parse_task_id_list_semi(
                    cell(row, "producerStitchSuccTaskIds")),
            ))
    logger.info("  加载完成: %d 条 stitch 边", len(edges))
    return edges


def _parse_bracket_int_list(cell: str) -> List[int]:
    """解析 "[0,1,2]" / "[]" 形式的整数列表。"""
    if cell is None:
        return []
    s = cell.strip().strip('"')
    if not s or s == "[]":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            logger.warning("无法解析 cellIdxList 元素: %r", p)
    return out


def load_slot_cell_table(path: str) -> Dict[int, List[CellTableDesc]]:
    """
    解析 slot_cell_table.csv。

    新版 header（兼容旧版会自动忽略缺失列）：
      slotIdx,stitchPolicy,rootHash,funcKey,dim,cellShape,strideShape,cellCount,outcastCount

    Returns:
        Dict[slotIdx, List[CellTableDesc]]
          同一 slot 可能有多行（partial 1 行 + 多个 fullcover root 各 1 行）。
    """
    logger.info("加载 slot_cell_table: %s", path)
    out: Dict[int, List[CellTableDesc]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"slot_cell_table 文件为空: {path}")
        col = {name: i for i, name in enumerate(header)}

        def pick(row, key, default=""):
            idx = col.get(key)
            if idx is None or idx >= len(row):
                return default
            return row[idx]

        for row in reader:
            if not row or not row[0].strip():
                continue
            slot_idx = int(pick(row, "slotIdx"))
            stitch_policy = pick(row, "stitchPolicy", "partial")
            desc = CellTableDesc(
                slot_idx=slot_idx,
                stitch_policy=stitch_policy,
                root_hash=pick(row, "rootHash", "0"),
                func_key=int(pick(row, "funcKey", "-1")),
                dim=int(pick(row, "dim", "0")),
                cell_shape=_parse_bracket_int_list(pick(row, "cellShape")),
                stride_shape=_parse_bracket_int_list(pick(row, "strideShape")),
                cell_count=int(pick(row, "cellCount", "1")),
                outcast_count=int(pick(row, "outcastCount", "1")),
            )
            out[slot_idx].append(desc)

    rows = sum(len(v) for v in out.values())
    logger.info("  加载完成: %d 个 slot, 共 %d 行 cell 表元数据", len(out), rows)
    return dict(out)


def load_slot_access(path: str) -> List[SlotAccessEvent]:
    """
    解析 dyn_slot_access.csv。

    header：
      seqNo,slotIdx,rootHash,funcKey,funcIdx,opIdx,taskId,accessType,
      cellIdxList,allConcrete
    """
    logger.info("加载 dyn_slot_access: %s", path)
    events: List[SlotAccessEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"dyn_slot_access 文件为空: {path}")
        col = {name: i for i, name in enumerate(header)}

        def pick(row, key, default=""):
            idx = col.get(key)
            if idx is None or idx >= len(row):
                return default
            return row[idx]

        for row in reader:
            if not row or not row[0].strip():
                continue
            events.append(SlotAccessEvent(
                seq_no=int(pick(row, "seqNo")),
                slot_idx=int(pick(row, "slotIdx")),
                root_hash=pick(row, "rootHash", "0"),
                func_key=int(pick(row, "funcKey", "-1")),
                func_idx=int(pick(row, "funcIdx", "0")),
                op_idx=int(pick(row, "opIdx", "0")),
                task_id=int(pick(row, "taskId", "0")),
                access_type=pick(row, "accessType", "W"),
                cell_idx_list=_parse_bracket_int_list(pick(row, "cellIdxList")),
                all_concrete=bool(int(pick(row, "allConcrete", "1"))),
            ))
    logger.info("  加载完成: %d 条 slot 访问事件", len(events))
    return events


def infer_edge_seq_no(edges: List[StitchEdge], tasks: List[DynTask]) -> Tuple[int, int]:
    """
    为每条 stitch 边推断 seqNo（直接 dump 文件中不含 seqNo 列）。

    策略：在 dyn_topo.txt 中查找 (producerTaskId, consumerTaskId) 同时出现的 seqNo。
          若有唯一候选，则填充；若有多个/零个，则为 None，并累计 ambiguous 计数。

    Returns:
        (resolved_count, ambiguous_count)
    """
    # seqNo -> set(taskId)
    tasks_by_seq: Dict[int, Set[int]] = defaultdict(set)
    for t in tasks:
        tasks_by_seq[t.seq_no].add(t.task_id)

    resolved = 0
    ambiguous = 0
    for e in edges:
        candidates = [
            seq for seq, ids in tasks_by_seq.items()
            if e.producer_task_id in ids and e.consumer_task_id in ids
        ]
        if len(candidates) == 1:
            e.inferred_seq_no = candidates[0]
            resolved += 1
        else:
            e.inferred_seq_no = None
            ambiguous += 1
    if ambiguous:
        logger.warning("%d 条 stitch 边无法唯一确定 seqNo（%d/%d 已解析）",
                       ambiguous, resolved, len(edges))
    return resolved, ambiguous
