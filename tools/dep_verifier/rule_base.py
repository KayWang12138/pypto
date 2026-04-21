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
"""规则基类与只读上下文。"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    CellTableDesc,
    DynTask,
    SlotAccessEvent,
    StaticFunction,
    StitchEdge,
    Violation,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleContext:
    """所有规则共享的只读上下文。"""
    static_functions: Dict[int, StaticFunction]
    dyn_tasks: List[DynTask]
    stitch_edges: List[StitchEdge]
    slot_cell_tables: Dict[int, List[CellTableDesc]] = field(default_factory=dict)
    slot_accesses: List[SlotAccessEvent] = field(default_factory=list)
    slot_roles: Dict[int, str] = field(default_factory=dict)

    # 懒构建索引
    _task_by_id: Optional[Dict[Tuple[int, int], DynTask]] = None
    _writers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _readers_of_cell: Optional[Dict[Tuple[int, int, int], List[SlotAccessEvent]]] = None
    _writer_seqs_by_slot_cell: Optional[Dict[Tuple[int, int], List[int]]] = None
    _reader_seqs_by_slot_cell: Optional[Dict[Tuple[int, int], List[int]]] = None
    _descendants_cache: Optional[Dict[Tuple[int, int], Set[int]]] = None

    @property
    def task_by_id(self) -> Dict[Tuple[int, int], DynTask]:
        if self._task_by_id is None:
            self._task_by_id = {(t.seq_no, t.task_id): t for t in self.dyn_tasks}
        return self._task_by_id

    @property
    def writers_of_cell(self) -> Dict[Tuple[int, int, int], List[SlotAccessEvent]]:
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
        if self._writer_seqs_by_slot_cell is None:
            s: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
            for (seq, slot, cell) in self.writers_of_cell.keys():
                s[(slot, cell)].add(seq)
            self._writer_seqs_by_slot_cell = {k: sorted(v) for k, v in s.items()}
        return self._writer_seqs_by_slot_cell

    @property
    def reader_seqs_by_slot_cell(self) -> Dict[Tuple[int, int], List[int]]:
        if self._reader_seqs_by_slot_cell is None:
            s: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
            for (seq, slot, cell) in self.readers_of_cell.keys():
                s[(slot, cell)].add(seq)
            self._reader_seqs_by_slot_cell = {k: sorted(v) for k, v in s.items()}
        return self._reader_seqs_by_slot_cell

    def descendants(self, seq_no: int, task_id: int) -> Set[int]:
        """同 seqNo DAG 上 `task_id` 的所有后代（不含自身），BFS + 记忆化。"""
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
        if u == v:
            return True
        return v in self.descendants(seq_no, u)

    def get_static_op(self, func_key: int, op_idx: int):
        fn = self.static_functions.get(func_key)
        if fn is None:
            return None
        return fn.ops.get(op_idx)

    def is_program_input_slot(self, slot_idx: int) -> bool:
        r = self.slot_roles.get(slot_idx)
        return r in ("INPUT", "INOUT")

    def is_program_output_slot(self, slot_idx: int) -> bool:
        r = self.slot_roles.get(slot_idx)
        return r in ("OUTPUT", "INOUT")

    def is_partial_slot(self, slot_idx: int) -> bool:
        """slot 是否存在 partial cell table 行。"""
        rows = self.slot_cell_tables.get(slot_idx, [])
        for row in rows:
            if (row.stitch_policy or "").lower() == "partial":
                return True
        return False

    def slot_cell_count(self, slot_idx: int) -> Optional[int]:
        rows = self.slot_cell_tables.get(slot_idx)
        if not rows:
            return None
        return max(r.cell_count for r in rows)


class Rule(ABC):
    """校验规则抽象基类。"""
    RULE_ID: str = ""
    DESCRIPTION: str = ""

    def __init__(self):
        if not self.RULE_ID:
            raise ValueError(f"{type(self).__name__}.RULE_ID 未设置")

    @abstractmethod
    def check(self, ctx: RuleContext) -> List[Violation]:
        raise NotImplementedError

    def run(self, ctx: RuleContext) -> List[Violation]:
        try:
            return self.check(ctx) or []
        except Exception as exc:  # noqa: BLE001
            logger.exception("规则 %s 运行异常", self.RULE_ID)
            return [Violation(
                rule_id=self.RULE_ID,
                message=f"规则运行异常: {exc}",
            )]
