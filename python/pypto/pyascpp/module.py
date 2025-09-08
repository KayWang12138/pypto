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
from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Sequence
from pathlib import Path
import os

from . import context
from . import parser
from .utils import Tensor, Instruction, CustStruct, Tuple, Var, AggregationVec, Vector
from .gen_ast_testing_code import gen_ast_ut_code, gen_ast_st_code, gen_ast_golden_script, \
    gen_ast_prefix_code, parse_st_code, parse_ut_code


class AscppModule(ABC):
    idx: int = -1
    _args: list[Union[Tensor, CustStruct, Var, Vector]]
    args_counter: int
    _initialized: bool = False
    inst_list: list[Instruction]
    _current_endif: Optional[Instruction]
    _current_startif: Optional[Instruction]
    _n_returned_tensors: int
    _is_vector: bool = False
    code: str
    init_code: str
    _stage: str
    custstructs: list[CustStruct]
    _return_type: str

    def __init__(self, *args: Union[Tensor, CustStruct, Var, Vector]) -> None:

        self._args = []
        self.args_counter = 0
        self.inst_list = []
        self.init_code = ''
        self._current_endif = None
        self._current_startif = None
        self._stage = 'init'
        self.custstructs = []
        self._return_type = ''

        args = self.clean_args(args)
        for a in args:
            a.set_idx(self.args_counter)
            self.args_counter += 1
            self._args.append(a)

            if isinstance(a, CustStruct):
                self.custstructs.append(a)
        context.stack_in_module(self)
        self.init(*args)
        self._gen_init()
        self.debug()
        context.stack_out_module()

        if context.active_module is not None:
            self.create_module()
            if context.active_module._stage == 'init':
                context.active_module.add_inst(Instruction('make_mod_class', [type(self).__name__] + list(args), self))
            elif context.active_module._stage == 'call':
                context.active_module.add_inst(Instruction('new_class', [type(self).__name__] + list(args), self))
        else:
            ...

    def __call__(self, *args: Union[Tensor, CustStruct, Var, Vector]) -> Any:
        def forward_res(*args: Union[Tensor, CustStruct, Var, Vector]):
            res = self.forward(*args)
            self._return_type = parser.get_return_type(res)
            if res is None:
                self._n_returned_tensors = 0
            elif isinstance(res, Tensor):
                self._n_returned_tensors = 1
            elif isinstance(res, Sequence):
                self._n_returned_tensors = len(res)
            elif isinstance(res, Vector):
                self._is_vector = True
            else:
                raise NotImplementedError()
            self._gen_code(res)
            
            self.debug()
            context.stack_out_module()
            
            self._initialized = True

        if not self._initialized:
            self._args = []
            self.inst_list = []
            self._n_returned_tensors = 0
            self._current_endif = None
            self._current_startif = None
            self.code = ''
            self._stage = 'call'
            args = self.clean_args(args)
            for a in args:
                a.push()
                self._args.append(a)
                a.set_idx(self.args_counter)
                self.args_counter += 1

                if isinstance(a, CustStruct):
                    self.custstructs.append(a)
            context.stack_in_module(self)
            forward_res(*args)

        if context.active_module is not None:
            func_name = f'cls{self.idx}.forward'
            if self._is_vector:
                result = context.active_module.create_vector(dtypes=["Tensor"])
                context.active_module.add_inst(Instruction('call_func', [func_name] + list(args), result))
            elif self._n_returned_tensors == 1:
                result = context.active_module.create_tensor()
                context.active_module.add_inst(Instruction('call_func', [func_name] + list(args), result))
            elif self._n_returned_tensors > 1:
                result = [context.active_module.create_tensor() for _ in range(self._n_returned_tensors)]
                context.active_module.add_inst(Instruction('call_func', [func_name] + list(args), result))
            else:
                result = None
                context.active_module.add_inst(Instruction('call_func', [func_name] + list(args), None))
            return result
        else:
            return None

    @classmethod
    def clean_args(cls, arglist: Union[tuple, list]):
        arglist = list(arglist)
        for i, arg in enumerate(arglist):
            # Convert values into variables
            if isinstance(arg, int):
                dtype = 'int'
            elif isinstance(arg, float):
                dtype = 'float'
            elif isinstance(arg, bool):
                dtype = 'bool'
            else:
                continue

            if context.active_module is None:
                arglist[i] = Var(dtype, is_declare=False)
            else:
                value = arg
                arglist[i] = Var(dtype)
                arglist[i].eq(value)
        return tuple(arglist)

    @classmethod
    def debug(cls, *args: Any):
        ...
        
    @abstractmethod
    def init(self):
        ...
        
    @abstractmethod
    def forward(self) -> Union[
        None, Tensor, tuple[Tensor, ...], list[Tensor]]:
        return None

    def set_idx(self, idx: int):
        self.idx = idx

    def get_stage(self) -> str:
        return self._stage

    def create_module(self):
        if context.active_module is None:
            raise Exception()
        self.set_idx(context.active_module.args_counter)
        context.active_module.args_counter += 1

    def create_tensor(self):
        new_tsr = Tensor()
        new_tsr.set_idx(self.args_counter)
        self.args_counter += 1
        return new_tsr

    def create_var(self, dtype='int'):  # dtype TO BE CHANGED
        new_var = Var(dtype, is_declare=False)
        new_var.set_idx(self.args_counter)
        self.args_counter += 1
        return new_var

    def create_aggregationvec(self):  # dtype TO BE CHANGED
        new_aggregationvec = AggregationVec()
        new_aggregationvec.set_idx(self.args_counter)
        self.args_counter += 1
        return new_aggregationvec

    def create_tuple(self, dtypes: list[str]):
        new_tuple = Tuple(dtypes, is_declare=False)
        new_tuple.set_idx(self.args_counter)
        self.args_counter += 1
        return new_tuple

    def create_vector(self, dtypes: list[str]):
        new_vector = Vector(dtypes, is_declare=False)
        new_vector.set_idx(self.args_counter)
        self.args_counter += 1
        return new_vector

    def add_inst(self, inst: Instruction):
        self.inst_list.append(inst)

    def insert_inst(self, inst: Instruction, prev_inst: Instruction):
        idx = self.inst_list.index(prev_inst)
        if idx == -1:
            raise Exception(f'Cannot find instruction to insert: {prev_inst}')
        self.inst_list.insert(idx, inst)

    def start_if(self, cond: Union[Var, bool]):
        new_inst = Instruction('if', [cond], None)
        self.add_inst(new_inst)
        self._current_endif = None
        self._current_startif = new_inst
        return new_inst

    def end_if(self, inst: Instruction):
        new_inst = Instruction('close_bracket', [], None)
        self.add_inst(new_inst)
        self._current_endif = new_inst
        self._current_startif = inst

    def start_elseif(self, cond: Union[Var, bool]):
        if self._current_endif is None:
            raise Exception()
        # move intermediate vars before if
        while self._current_endif != self.inst_list[-1]:
            if self._current_startif is None:
                raise Exception()
            inst = self.inst_list.pop(-1)
            self.insert_inst(inst, self._current_startif)
        new_inst = Instruction('elif', [cond], None)
        self.add_inst(new_inst)
        self._current_endif = None
        return self._current_startif

    def start_else(self):
        self.add_inst(Instruction('else', [], None))
        self._current_endif = None
        self._current_startif = None

    def end_else(self):
        self.add_inst(Instruction('close_bracket', [], None))

    def gen_src_code(self, is_gen_ast: bool = False):
        def parse_children(mod: AscppModule, d: dict[AscppModule, int]):
            for _, v in vars(mod).items():
                if isinstance(v, AscppModule):
                    new_depth = d[self] + 1
                    old_depth = d.get(v, 0)
                    d[v] = max(new_depth, old_depth)
                    parse_children(v, d)

        def gen_result_code_sub(mod: AscppModule, result_code: list[str],
                                generated_structs_list: list[type[CustStruct]]):
            for custstruct_obj in mod.custstructs:
                custstruct_class = custstruct_obj.__class__
                if custstruct_class not in generated_structs_list:
                    generated_structs_list.append(custstruct_class)
                    result_code.append(parser.get_custstruct_code(custstruct_obj))

        def gen_result_code(traversed_class_list: list[AscppModule]):
            result_code: list[str] = []
            generated_class_list: list[type[AscppModule]] = []
            generated_structs_list: list[type[CustStruct]] = []
            for mod in traversed_class_list:
                if type(mod) not in generated_class_list:
                    gen_result_code_sub(mod, result_code, generated_structs_list)
                    if is_gen_ast:
                        result_code.append(mod.code.replace("::forward", ""))
                    else: 
                        result_code.append(mod.init_code)
                        result_code.append(mod.code)
                    generated_class_list.append(type(mod))

            return result_code

        # find the depth of each module
        depth_dict: dict[AscppModule, int] = {}
        depth_dict[self] = 0
        parse_children(self, depth_dict)

        traversed_class_list: list[AscppModule] = []
        current_depth = 0
        need_continue = True
        while need_continue:
            need_continue = False
            for mod, dep in depth_dict.items():
                if dep == current_depth and hasattr(mod, 'init_code'):
                    need_continue = True
                    traversed_class_list.append(mod)
            current_depth += 1
        traversed_class_list.reverse()
        return gen_result_code(traversed_class_list)
        
    def gen_code(self, target_file: str):
        result_code = self.gen_src_code()
        with open(target_file, 'w') as f:
            f.write('\n\n'.join(result_code))
        
    def gen_ast_code(self, directory_path: str):
        ast_include_file_list = [
            "interface/operation/operation_impl.h",
            "interface/operation/operation.h",
            "interface/function/function.h",
            "tilefwk/tensor.h",
            "interface/tensor/logical_tensor.h",
            "interface/tensor/raw_tensor.h",
            "tilefwk/tilefwk.h",
            "interface/inner/tilefwk.h",
            "interface/tensor/tensormap.h",
            "interface/configs/config_manager.h",
            "interface/configs/config_storage.h",
            "interface/utils/common.h",
            "interface/utils/id_gen.h",
            "interface/utils/log.h"
        ]
        ast_code_prefix = gen_ast_prefix_code(ast_include_file_list)
        ast_code_suffix = "} // namespace"
        result_code = self.gen_src_code(is_gen_ast=True)
        
        src_operator_folder = Path(directory_path, 'src/operator/custom')
        if not os.path.exists(src_operator_folder):
            os.mkdir(src_operator_folder)
        operator_name = type(self).__name__
        src_operator_file = Path(src_operator_folder, f'{operator_name}.cpp')
        with open(src_operator_file, 'w') as f:
            f.write(ast_code_prefix)
            f.write('\n\n'.join(result_code))
            f.write(ast_code_suffix)
        
        gen_ast_ut_code(directory_path, operator_name, self._args, self._return_type)
        gen_ast_st_code(directory_path, operator_name, self._args, self._return_type)
        gen_ast_golden_script(directory_path, operator_name, self._args)
        
    def _gen_init(self):
        # Get list of variables
        variables = []
        indexes_to_change = []
        for k, v in vars(self).items():
            if isinstance(v, Tensor):
                tsr = self.create_tensor()
                tsr.eq(v)
                variables.append(tsr)
                indexes_to_change.append((k, tsr))
            elif isinstance(v, Var):
                var = self.create_var(dtype=v.dtype)
                var.eq(v)
                variables.append(var)
                indexes_to_change.append((k, var))
            elif isinstance(v, (CustStruct, Vector, AscppModule)):
                variables.append(v)
                if isinstance(v, CustStruct):
                    self.custstructs.append(v)
        self.init_code = parser.parse_init(type(self).__name__, self.inst_list, self._args, variables)

        for name, var in indexes_to_change:
            setattr(self, name, var)

    def _gen_code(self, res: Union[None, Tensor, tuple[Tensor, ...], list[Tensor]] = None):
        self.code = parser.parse(type(self).__name__, self.inst_list, self._args, res)
        fwd_signature = parser.get_signature(self._args, res)
        lines = self.init_code.split('\n')
        lines.insert(-2, f"    {fwd_signature}")
        self.init_code = '\n'.join(lines)
