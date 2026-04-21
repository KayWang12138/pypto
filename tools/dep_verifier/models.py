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
"""共享数据模型。"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


TASK_ID_OP_BITS = 16
TASK_ID_OP_MASK = (1 << TASK_ID_OP_BITS) - 1


def decode_task_id(task_id: int) -> Tuple[int, int]:
    """taskId = (funcIdx << 16) | opIdx"""
    return task_id >> TASK_ID_OP_BITS, task_id & TASK_ID_OP_MASK


def encode_task_id(func_idx: int, op_idx: int) -> int:
    return (func_idx << TASK_ID_OP_BITS) | op_idx


class Severity(str, Enum):
    """校验违规的严重等级。"""
    ERROR = "ERROR"
    SUSPECT_HIGH = "SUSPECT-HIGH"
    SUSPECT_MEDIUM = "SUSPECT-MEDIUM"
    SUSPECT_LOW = "SUSPECT-LOW"
    INFO = "INFO"


@dataclass
class StaticOp:
    """static_topo.csv 中的一条 operation 记录（函数模板内的 op 信息）。"""
    func_key: int
    root_hash: str
    raw_name: str
    op_idx: int
    opmagic: int
    leaf_hash: str
    core_type: int
    psg_id: int
    incast_slots: List[int]
    outcast_slots: List[int]
    static_successors_op_idx: List[int]  # 本 function 内后继 op 的 opIdx 列表


@dataclass
class StaticFunction:
    """按 func_key 聚合的 StaticOp 集合，快速查询。"""
    func_key: int
    root_hash: str
    raw_name: str
    ops: Dict[int, StaticOp] = field(default_factory=dict)  # op_idx -> StaticOp

    def incast_union(self) -> Set[int]:
        s: Set[int] = set()
        for op in self.ops.values():
            s.update(op.incast_slots)
        return s

    def outcast_union(self) -> Set[int]:
        s: Set[int] = set()
        for op in self.ops.values():
            s.update(op.outcast_slots)
        return s


@dataclass
class DynTask:
    """dyn_topo.txt 中的一行：运行时 task 实例。"""
    seq_no: int
    task_id: int
    root_index: int      # == funcKey
    root_hash: str
    opmagic: int
    leaf_index: int
    leaf_hash: str
    core_type: int
    psg_id: int
    wrap_id: int
    static_succ_count: int
    successors: List[int]  # 全部后继 taskId 列表（前 static_succ_count 为静态，其余为 stitch）

    @property
    def func_idx(self) -> int:
        return self.task_id >> TASK_ID_OP_BITS

    @property
    def op_idx(self) -> int:
        return self.task_id & TASK_ID_OP_MASK

    @property
    def static_successors(self) -> List[int]:
        return self.successors[: self.static_succ_count]

    @property
    def stitch_successors(self) -> List[int]:
        return self.successors[self.static_succ_count:]


@dataclass
class StitchEdge:
    """dyn_stitch_edges.csv 中的一行：一条具体建立的 stitch 边。"""
    stitch_kind: str
    slot_idx: int
    producer_root_hash: str
    producer_func_key: int
    producer_func_idx: int
    producer_op_idx: int
    producer_opmagic: int
    producer_task_id: int
    producer_raw_name: str
    consumer_root_hash: str
    consumer_func_key: int
    consumer_func_idx: int
    consumer_op_idx: int
    consumer_opmagic: int
    consumer_task_id: int
    consumer_raw_name: str
    producer_static_succ_task_ids: List[int]
    producer_stitch_succ_task_ids: List[int]
    # 运行时根据 dyn_topo 反查出来的 seqNo（加载阶段填充，可能为 None 表示无法唯一确定）
    inferred_seq_no: Optional[int] = None


@dataclass
class CellTableDesc:
    """slot_cell_table.csv 中的一行：某 slot 的 CellMatchTable 元信息。

    新版 header：
      slotIdx, stitchPolicy, rootHash, funcKey, dim, cellShape, strideShape,
      cellCount, outcastCount

    - stitchPolicy == "partial": 聚合行，rootHash/funcKey 为 sentinel (0/-1)
    - stitchPolicy == "fullcover": 每个 (slot, root) 一行
    """
    slot_idx: int
    stitch_policy: str
    root_hash: str
    func_key: int
    dim: int
    cell_shape: List[int]
    stride_shape: List[int]
    cell_count: int
    outcast_count: int

    @property
    def is_partial(self) -> bool:
        return self.stitch_policy == "partial"

    @property
    def is_full_cover(self) -> bool:
        return self.stitch_policy == "fullcover"


@dataclass
class SlotAccessEvent:
    """dyn_slot_access.csv 中的一行：实例级 cell 读写事件。

    header：
      seqNo, slotIdx, rootHash, funcKey, funcIdx, opIdx, taskId,
      accessType, cellIdxList, allConcrete
    """
    seq_no: int
    slot_idx: int
    root_hash: str
    func_key: int
    func_idx: int
    op_idx: int
    task_id: int
    access_type: str                # "W" or "R"
    cell_idx_list: List[int]
    all_concrete: bool

    @property
    def is_writer(self) -> bool:
        return self.access_type == "W"

    @property
    def is_reader(self) -> bool:
        return self.access_type == "R"


@dataclass
class Violation:
    """违规记录，由各规则产生并汇总输出。"""
    rule_id: str
    severity: Severity
    message: str
    details: Dict[str, object] = field(default_factory=dict)

    def short(self) -> str:
        return f"[{self.severity.value}] {self.rule_id}: {self.message}"
