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
规则 4（核心）：实例级数据流完整性。

包含三个子规则，在同一 Rule 对象里完成以避免重复建索引：
  4-A  Producer 实例"有去向"（悬置 producer 检测）
  4-B  Consumer 实例"有来源"（野读检测）
  4-C  写入引用率统计（汇总视角）

豁免集合由 RuleContext 提供：
  program_input_slots   4-B 条件 4
  program_output_slots  4-A 条件 2
  zero_init_slots       4-B 的额外豁免（场景 B2 零初始化）

关于跨 task 传递（4-A 条件 3 / 4-B 条件 3）的实现：
  采用"后续 seqNo 内存在该 slot 的 reader"作为正向证据，但仅降级严重度而非直接豁免，
  以便仍能暴露 engram-tmp 这类典型 bug。
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..models import Severity, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class InstanceDataFlowRule(Rule):
    RULE_ID = "rule_instance_data_flow"
    DESCRIPTION = (
        "规则 4（legacy）：基于 stitch 边视角的实例级数据流校验。"
        "已被 rule_cell_write_conflict / rule_cell_write_reach / rule_cell_read_source 取代，"
        "默认关闭，可用 --enable rule_instance_data_flow 重新启用作为辅助观察。"
    )
    #: 旧规则存在大量跨 DeviceTask 切片写的假报，默认关闭；cell 粒度的新规则 4-α/β/γ 负责主责任。
    DEFAULT_ENABLED = False

    def check(self, ctx: RuleContext) -> List[Violation]:
        # 构建实例级索引，三子规则共享
        writers, readers = self._build_writer_reader_sets(ctx)
        ref_producers, ref_consumers = self._build_referenced_sets(ctx)

        # 每个 seqNo 内每个 slot 的最大 writer funcIdx（条件 2 判定用）
        max_writer_funcidx = self._max_writer_funcidx(writers)

        # slot 的下游 seqNo 是否存在 reader（条件 3 判定用）
        slot_future_reader = self._slot_future_reader(readers)

        violations: List[Violation] = []
        violations.extend(self._check_sub_rule_a(
            ctx, writers, ref_producers, max_writer_funcidx, slot_future_reader))
        violations.extend(self._check_sub_rule_b(
            ctx, readers, ref_consumers, writers))
        violations.extend(self._check_sub_rule_c(
            ctx, writers, ref_producers))
        return violations

    # ==================================================================
    # 索引构建
    # ==================================================================
    @staticmethod
    def _build_writer_reader_sets(ctx: RuleContext):
        """
        writers[(seqNo, slotIdx)] = set(funcIdx)
        readers[(seqNo, slotIdx)] = set(funcIdx)
        """
        writers: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        readers: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        for (seq_no, func_key, func_idx, op_idx, slot) in ctx.iter_writer_instances():
            writers[(seq_no, slot)].add(func_idx)
        for (seq_no, func_key, func_idx, op_idx, slot) in ctx.iter_reader_instances():
            readers[(seq_no, slot)].add(func_idx)

        return writers, readers

    @staticmethod
    def _build_referenced_sets(ctx: RuleContext):
        """从 dyn_stitch_edges 建立被引用的 producer/consumer 集合。"""
        ref_producers: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        ref_consumers: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for e in ctx.stitch_edges:
            if e.inferred_seq_no is None:
                continue
            ref_producers[(e.inferred_seq_no, e.slot_idx)].add(e.producer_func_idx)
            ref_consumers[(e.inferred_seq_no, e.slot_idx)].add(e.consumer_func_idx)
        return ref_producers, ref_consumers

    @staticmethod
    def _max_writer_funcidx(writers) -> Dict[Tuple[int, int], int]:
        return {k: max(v) for k, v in writers.items() if v}

    @staticmethod
    def _slot_future_reader(readers) -> Dict[int, List[int]]:
        """slot -> 所有存在该 reader 的 seqNo 有序列表"""
        by_slot: Dict[int, Set[int]] = defaultdict(set)
        for (seq_no, slot), fids in readers.items():
            if fids:
                by_slot[slot].add(seq_no)
        return {s: sorted(v) for s, v in by_slot.items()}

    # ==================================================================
    # 子规则 A：悬置 producer
    # ==================================================================
    def _check_sub_rule_a(self, ctx, writers, ref_producers,
                          max_writer_funcidx, slot_future_reader) -> List[Violation]:
        violations = []
        for (seq_no, slot), fids in writers.items():
            refs = ref_producers.get((seq_no, slot), set())
            for fidx in fids:
                if fidx in refs:
                    continue  # 条件 1 命中

                is_last_writer = (max_writer_funcidx.get((seq_no, slot)) == fidx)
                is_output_slot = slot in ctx.program_output_slots

                # 条件 2：task 内最后一次写 + program 输出 slot
                if is_last_writer and is_output_slot:
                    continue

                # 条件 3：跨 task 存在 reader
                future_seqs = [s for s in slot_future_reader.get(slot, [])
                               if s > seq_no]
                has_future_reader = bool(future_seqs)

                if is_last_writer and has_future_reader:
                    severity = Severity.SUSPECT_MEDIUM
                    reason = "task 内末轮写 + 后续 task 有 reader（跨 task 合法传递大概率成立）"
                elif has_future_reader:
                    severity = Severity.SUSPECT_HIGH
                    reason = "非末轮写但在同 task 内无下游 + 跨 task 依赖机制传递"
                else:
                    severity = Severity.SUSPECT_HIGH
                    reason = "既不是末轮写也没有跨 task reader，怀疑悬置"

                violations.append(Violation(
                    rule_id=f"{self.RULE_ID}.A",
                    severity=severity,
                    message="悬置 producer：实例级写入缺去向",
                    details={
                        "seqNo": seq_no,
                        "slotIdx": slot,
                        "funcIdx": fidx,
                        "isLastWriter": is_last_writer,
                        "isProgramOutputSlot": is_output_slot,
                        "hasFutureReader": has_future_reader,
                        "futureReaderSeqNos": future_seqs,
                        "reason": reason,
                    },
                ))
        return violations

    # ==================================================================
    # 子规则 B：野读 consumer
    # ==================================================================
    def _check_sub_rule_b(self, ctx, readers, ref_consumers, writers) -> List[Violation]:
        violations = []
        predecessors = ctx.predecessors  # (seqNo, taskId) -> 前驱 taskId 列表

        # 为了 O(1) 判断"同 task 内有 writer"，提前构建
        writer_func_set: Dict[Tuple[int, int], Set[int]] = writers

        for (seq_no, slot), fids in readers.items():
            # 条件 4：program 输入 slot
            if slot in ctx.program_input_slots:
                continue
            if slot in ctx.zero_init_slots:
                continue

            for fidx in fids:
                # 条件 1：同 task 内有 stitch 上游
                refs = ref_consumers.get((seq_no, slot), set())
                if fidx in refs:
                    continue

                # 条件 2：静态依赖上游——consumer 所在 function 实例的任一 op 有静态前驱
                #         简化判定：同 task 内同 funcIdx 有 writer 即可命中（同 function 内部依赖）
                same_func_writers = {w for w in writer_func_set.get((seq_no, slot), set())
                                     if w == fidx}
                if same_func_writers:
                    continue

                # 条件 3：跨 task 上游有效（上一个 seqNo 内存在该 slot 的 writer）
                prev_seq_writers = writer_func_set.get((seq_no - 1, slot), set())
                if prev_seq_writers:
                    severity = Severity.SUSPECT_MEDIUM
                    reason = "跨 task 上游存在 writer，依赖跨 task 传递机制"
                    violations.append(Violation(
                        rule_id=f"{self.RULE_ID}.B",
                        severity=severity,
                        message="consumer 读取依赖跨 task 传递（需人工确认）",
                        details={
                            "seqNo": seq_no,
                            "slotIdx": slot,
                            "funcIdx": fidx,
                            "previousSeqWriters": sorted(prev_seq_writers),
                            "reason": reason,
                        },
                    ))
                    continue

                # 全部条件未命中 → 真正的野读
                violations.append(Violation(
                    rule_id=f"{self.RULE_ID}.B",
                    severity=Severity.ERROR,
                    message="consumer 野读：无同 task stitch 上游 / 静态上游 / 跨 task writer / 输入 slot",
                    details={
                        "seqNo": seq_no,
                        "slotIdx": slot,
                        "funcIdx": fidx,
                    },
                ))
        return violations

    # ==================================================================
    # 子规则 C：写入引用率统计
    # ==================================================================
    def _check_sub_rule_c(self, ctx, writers, ref_producers) -> List[Violation]:
        violations = []
        for (seq_no, slot), fids in writers.items():
            if not fids:
                continue
            refs = ref_producers.get((seq_no, slot), set())
            ref_in_writers = refs & fids
            total = len(fids)
            referenced = len(ref_in_writers)
            coverage = referenced / total if total else 1.0

            if coverage >= 1.0:
                continue

            # 分级：overwrite 型（极小 coverage）最严重
            if total >= 2 and referenced <= 1:
                severity = Severity.SUSPECT_HIGH
                reason = "覆盖型疑点：多 writer 写同 slot 但仅 0~1 个 writer 被引用"
            elif coverage < 0.5:
                severity = Severity.SUSPECT_HIGH
                reason = f"低覆盖率 ({referenced}/{total})"
            else:
                severity = Severity.SUSPECT_MEDIUM
                reason = f"部分 writer 悬置 ({referenced}/{total})"

            violations.append(Violation(
                rule_id=f"{self.RULE_ID}.C",
                severity=severity,
                message="写入引用率异常",
                details={
                    "seqNo": seq_no,
                    "slotIdx": slot,
                    "totalWriters": total,
                    "referencedWriters": referenced,
                    "coverage": round(coverage, 4),
                    "reason": reason,
                },
            ))
        return violations
