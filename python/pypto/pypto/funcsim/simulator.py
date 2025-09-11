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

from typing import Sequence, TYPE_CHECKING, Union
from . import executor
from ..utils import Tensor, Instruction


if TYPE_CHECKING:
    from ..module import AscppModule


class ExecGraph():
    inputs: list[Tensor]

    def __init__(self, inst_list: list['Instruction'], inputs: list[Tensor], outputs: Union[Tensor, Sequence[Tensor]], \
                 module: 'AscppModule'):
        self.inputs = []
        self.outputs = []
        self.inst_list = []
        self._tensors = {}
        self.module = module

        for i in inst_list:
            self.inst_list.append(self.copy_inst(i))
        for i in inputs:
            self.inputs.append(self.copy_tensor(i))
        if isinstance(outputs, Tensor):
            self.outputs.append(self.copy_tensor(outputs))
        else:
            if outputs is not None:
                for i in outputs:
                    self.outputs.append(self.copy_tensor(i))

    def run(self, inputs: list[Tensor]):
        for iself, ioutside in zip(self.inputs, inputs):
            iself.from_np(ioutside.data)
        executor.run_all(self.inst_list)
        return self.outputs
    
    def copy_tensor(self, x: Tensor):
        if x.idx in self._tensors:
            res = self._tensors[x.idx]
        else:
            res = Tensor()
            res.idx = x.idx
            self._tensors[x.idx] = res
        return res
    
    def copy_inst(self, x: 'Instruction'):
        from ..module import AscppModule

        def forward_res(v, x: 'Instruction', src):
            if isinstance(v, AscppModule):
                callstr = 'cls % d.forward' % v.set_idx
            if x.src[0] == callstr:
                src.append(v.__class__.__name__)
            return v, x, src
    
        src = []
        for i in x.src:
            if isinstance(i, Tensor):
                src.append(self.copy_tensor(i))
            elif x.inst == 'call_func':
                for _, v in vars(self.module).items():
                    v, x, src = forward_res(v, x, src)
            else:
                src.append(i)

        if isinstance(x.dst, Sequence):
            dst = []
            for i in x.dst:
                if isinstance(i, Tensor):
                    dst.append(self.copy_tensor(i))
                else:
                    dst.append(i)
        elif isinstance(x.dst, Tensor):
            dst = self.copy_tensor(x.dst)
        else:
            dst = x.dst
        
        res = Instruction(x.inst, src, dst)
        return res

    
class Simulator():
    graphs: dict[str, ExecGraph]
    
    def __init__(self):
        self.graphs = {}

    def record(self, name: str, inst_list: list['Instruction'], inputs: list[Tensor], outputs: list[Tensor], \
               module: 'AscppModule'):
        exe_graph = ExecGraph(inst_list, inputs, outputs, module)
        self.graphs[name] = exe_graph
    
    def run(self, name: str, inputs: list[Tensor]):
        res = self.graphs[name].run(inputs)
        if len(res) == 1:
            return res[0]
        else:
            return res