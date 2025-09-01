#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from .util import get_curlybrace_list
from ..utils import CodeHelper, Instruction, Tensor, AggregationVec, TensorMap


def new_aggregation_vec(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.dst, AggregationVec):
        raise Exception()
    if len(inst.src) >= 1:
        value_str = get_curlybrace_list(inst.src[0])
        h(f'std::vector<std::pair<Tensor, std::vector<int>>> aggregationVec{inst.dst.idx} = {value_str};')
    else:
        h(f'std::vector<std::pair<Tensor, std::vector<int>>> aggregationVec{inst.dst.idx};')


def tensor2aggregationvec(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], TensorMap):
        raise Exception()
    if not isinstance(inst.dst, AggregationVec):
        raise Exception()
    h(f'for (auto &[offset, tensor] : mp{inst.src[0].idx})' + ' { ' + \
      f'aggregationVec{inst.dst.idx}.emplace_back(tensor, offset);' + ' }')


def assemble(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], AggregationVec):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Assemble(aggregationVec{inst.src[0].idx});')
