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
规则基类与运行时上下文。

设计原则：
  - 每个规则继承 Rule 类并实现 check(ctx) -> List[Violation]
  - 规则之间互不感知，通过 RuleContext 共享只读输入数据
  - RuleContext 提供一系列索引辅助方法，避免各规则重复建索引
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    CellTableDesc,
    DynTask,
    Severity,
    SlotAccessEvent,
    StaticFunction,
    StitchEdge,
    Violation,
    decode_task_id,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleContext:
    """
    规则共享的只读上下文。

    加载三个文件之后一次构建，供所有规则反复查询。
    也保存了用户 CLI 指定的 program 输入/输出 slot 集合（供规则 4-B 豁免用）。
    """
    static_functions: Dict[int, StaticFunction]
    dyn_tasks: List[DynTask]
    stitch_edges: List[StitchEdge]

    # cell 级 dump；可选，未提供时对应规则会自动跳过。
    slot_cell_tables: Dict[int, List[CellTableDesc]] = field(default_factory=dict)
    slot_accesses: List[SlotAccessEvent] = field(default_factory=list)

    program_input_slots: Set[int] = field(default_factory=set)
    program_output_slots: Set[int] = field(default_factory=set)
    zero_init_slots: Set[int] = field(default_factory=set)

    # --- 懒构建的索引（首次访问时构建） ---
    _task_by_id: Optional[Dict[Tuple[int, int], DynTask]] = None  # (seqNo, taskId)
    _tasks_by_seq: Optional[Dict[int, List[DynTask]]] = None       # seqNo -> tasks
    _tasks_by_root: Optional[Dict[int, List[DynTask]]] = None      # rootIndex -> tasks
    _predecessors: Optional[Dict[Tuple[int, int], List[int]]] = None  # (seqNo, taskId) -> 前驱 taskId 列表

    # cell 级索引
    _writers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _readers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _writer_seqs_by_slot_cell: Optional[Dict[Tuple[int, int], List[int]]] = None
    _reader_seqs_by_slot_cell: Optional[Dict[Tuple[int, int], List[int]]] = None
    _descendants_cache: Optional[Dict[Tuple[int, int], Set[int]]] = None

    # -------- 索引访问器 --------
    @property
    def task_by_id(self) -> Dict[Tuple[int, int], DynTask]:
        if self._task_by_id is None:
            self._task_by_id = {(t.seq_no, t.task_id): t for t in self.dyn_tasks}
        return self._task_by_id

    @property
    def tasks_by_seq(self) -> Dict[int, List[DynTask]]:
        if self._tasks_by_seq is None:
            d: Dict[int, List[DynTask]] = defaultdict(list)
            for t in self.dyn_tasks:
                d[t.seq_no].append(t)
            self._tasks_by_seq = dict(d)
        return self._tasks_by_seq

    @property
    def tasks_by_root(self) -> Dict[int, List[DynTask]]:
        if self._tasks_by_root is None:
            d: Dict[int, List[DynTask]] = defaultdict(list)
            for t in self.dyn_tasks:
                d[t.root_index].append(t)
            self._tasks_by_root = dict(d)
        return self._tasks_by_root

    @property
    def predecessors(self) -> Dict[Tuple[int, int], List[int]]:
        """以 dyn_topo 的全量 successors（含 stitch）构建的反向邻接表。"""
        if self._predecessors is None:
            d: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for t in self.dyn_tasks:
                for s in t.successors:
                    d[(t.seq_no, s)].append(t.task_id)
            self._predecessors = dict(d)
        return self._predecessors

    # -------- cell 级索引 --------
    @property
    def writers_of_cell(self) -> Dict[Tuple[int, int, int], List[SlotAccessEvent]]:
        """(seqNo, slotIdx, cellIdx) -> 所有 writer 事件。"""
        if self._writers_of_cell is None:
            d: Dict[Tuple[int, int, int], List[SlotAccessEvent]] = defaultdict(list)
            for e in self.slot_accesses:
                if not e.is_writer:
                    continue
                for c in e.cell_idx_list:
                    d[(e.seq_no, e.slot_idx, c)].append(e)
            self._writers_of_cell = dict(d)
        return self._writers_of_cell

    @property
    def readers_of_cell(self) -> Dict[Tuple[int, int, int], List[SlotAccessEvent]]:
        if self._readers_of_cell is None:
            d: Dict[Tuple[int, int, int], List[SlotAccessEvent]] = defaultdict(list)
            for e in self.slot_accesses:
                if not e.is_reader:
                    continue
                for c in e.cell_idx_list:
                    d[(e.seq_no, e.slot_idx, c)].append(e)
            self._readers_of_cell = dict(d)
        return self._readers_of_cell

    @property
    def writer_seqs_by_slot_cell(self) -> Dict[Tuple[int, int], List[int]]:
        """(slotIdx, cellIdx) -> 有 writer 的 seqNo 升序列表。"""
        if self._writer_seqs_by_slot_cell is None:
            s: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
            for (seq, slot, cell), _ in self.writers_of_cell.items():
                s[(slot, cell)].add(seq)
            self._writer_seqs_by_slot_cell = {k: sorted(v) for k, v in s.items()}
        return self._writer_seqs_by_slot_cell

    @property
    def reader_seqs_by_slot_cell(self) -> Dict[Tuple[int, int], List[int]]:
        if self._reader_seqs_by_slot_cell is None:
            s: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
            for (seq, slot, cell), _ in self.readers_of_cell.items():
                s[(slot, cell)].add(seq)
            self._reader_seqs_by_slot_cell = {k: sorted(v) for k, v in s.items()}
        return self._reader_seqs_by_slot_cell

    def cell_count_of(self, slot_idx: int) -> Optional[int]:
        """返回某 slot 的 cellCount（partial 行优先；否则取 fullcover 行最大值）。"""
        rows = self.slot_cell_tables.get(slot_idx)
        if not rows:
            return None
        for r in rows:
            if r.is_partial:
                return r.cell_count
        return max(r.cell_count for r in rows)

    # -------- 可达性（同 seqNo 内 DAG） --------
    def descendants(self, seq_no: int, task_id: int) -> Set[int]:
        """同 seqNo DAG 上 `task_id` 的所有后代（不含自身）。BFS + 记忆化。"""
        if self._descendants_cache is None:
            self._descendants_cache = {}
        key = (seq_no, task_id)
        cached = self._descendants_cache.get(key)
        if cached is not None:
            return cached
        task_by_id = self.task_by_id
        visited: Set[int] = set()
        stack: List[int] = [task_id]
        while stack:
            cur = stack.pop()
            t = task_by_id.get((seq_no, cur))
            if t is None:
                continue
            for s in t.successors:
                if s in visited:
                    continue
                visited.add(s)
                stack.append(s)
        self._descendants_cache[key] = visited
        return visited

    def reaches(self, seq_no: int, u: int, v: int) -> bool:
        """同 seqNo DAG 上 u 是否可达 v。"""
        if u == v:
            return True
        return v in self.descendants(seq_no, u)

    # -------- 便捷查询 --------
    def get_static_op(self, func_key: int, op_idx: int):
        fn = self.static_functions.get(func_key)
        if fn is None:
            return None
        return fn.ops.get(op_idx)

    def iter_writer_instances(self):
        """
        迭代所有 writer 实例：(seq_no, func_key, func_idx, slot_idx)

        一个 DynTask 表示 `(funcIdx, opIdx)` 实例；对该 op 的每个 outcast slot 产生一条记录。
        注意：一个 (seq_no, funcIdx, slotIdx) 可能由同 function 内多个 op 共同写入，
        此处逐 op 产出，去重交给调用方。
        """
        for t in self.dyn_tasks:
            op = self.get_static_op(t.root_index, t.op_idx)
            if op is None:
                continue
            for s in op.outcast_slots:
                yield (t.seq_no, t.root_index, t.func_idx, t.op_idx, s)

    def iter_reader_instances(self):
        """同 iter_writer_instances，但基于 incast_slots。"""
        for t in self.dyn_tasks:
            op = self.get_static_op(t.root_index, t.op_idx)
            if op is None:
                continue
            for s in op.incast_slots:
                yield (t.seq_no, t.root_index, t.func_idx, t.op_idx, s)


class Rule(ABC):
    """所有校验规则的抽象基类。"""

    #: 规则 id，在 --enable/--disable 命令行选项中使用；需全局唯一。
    RULE_ID: str = ""
    #: 人类可读的规则描述。
    DESCRIPTION: str = ""
    #: 默认是否启用。
    DEFAULT_ENABLED: bool = True

    def __init__(self):
        if not self.RULE_ID:
            raise ValueError(f"{type(self).__name__}.RULE_ID 未设置")

    @abstractmethod
    def check(self, ctx: RuleContext) -> List[Violation]:
        """执行校验并返回违规列表。"""
        raise NotImplementedError

    def run(self, ctx: RuleContext) -> List[Violation]:
        """执行入口，集中处理异常以避免单规则失败拉挂整体校验。"""
        logger.info("  -> 运行规则 %s", self.RULE_ID)
        try:
            return self.check(ctx) or []
        except Exception as exc:  # noqa: BLE001
            logger.exception("规则 %s 运行失败", self.RULE_ID)
            return [Violation(
                rule_id=self.RULE_ID,
                severity=Severity.ERROR,
                message=f"规则运行时异常: {exc}",
                details={"exception": type(exc).__name__},
            )]
