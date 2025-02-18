#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
from typing import Any, Union, Literal, Sequence
from .parser import parse_all
from ..utils import Var, GMTensor, Instruction, CodeHelper, DTYPE, DT, sizeof
from . import cmake_template, datautil_template, tensorutils_template, testmacros_template, workspace_template
from ..vec.module import VecModule
from ..cube.module import CubeModule
from .. import context

# rich imports
from rich.text import Text
from rich.style import Style
from rich.console import Console
from functools import partial
print = partial(Console(width=150).print, justify='center')


class KernelBase:
    _curr_phase: str
    _input_tensors: list[GMTensor]
    _input_vars: list[Var]
    _init_inst_list: list[Instruction]
    _compute_inst_list: list[Instruction]
    _workspace_list: list[GMTensor]
    _var_list: list[Var]
    _h: CodeHelper
    _used_classnames: dict[str, int]
    _modules: dict[Union[CubeModule, VecModule], str]

    def __init__(self, *args: Var):
        # some class variables
        self._curr_phase = 'init'
        self._init_inst_list = []
        self._compute_inst_list = []
        self._input_vars = []
        self._input_tensors = []
        self._var_list = []
        self._workspace_list = []
        self._used_classnames = {}
        self._modules = {}
        self._h = CodeHelper()

        for a in args:
            assert isinstance(a, Var), 'All input args should be Var'
            self._input_vars.append(a)
        context.active_kernel = self
        self.initialize(*args)
        context.active_kernel = None

    def initialize(self, *args: Var):
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        assert self._curr_phase=='init', 'Can only assign class variables in initialize'
        if isinstance(value, (CubeModule, VecModule)):
            clsname = value.__class__.__name__
            if clsname not in self._used_classnames:
                self._used_classnames[clsname] = 0
            else:
                self._used_classnames[clsname] += 1
                clsname += str(self._used_classnames[clsname])
            value._name = clsname
            self._modules[value] = name
        elif isinstance(value, GMTensor):
            value.name = name
            self._workspace_list.append(value)
        elif isinstance(value, Var):
            new_var = value.copy()
            new_var.name = f'm_{name}'
            new_var.varstr = f'm_{name}'
            self._var_list.append(new_var)
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
            value = new_var
        super().__setattr__(name, value)

    def split_workspace(self, dtype: DTYPE, shape: Sequence[Union[int, Var]]):
        assert self._curr_phase=='init', 'Can only split workspace in initialize'
        return GMTensor('', dtype, shape)

    def get_total_workspace_size(self):
        res = None
        for i in self._workspace_list:
            assert i.length is not None
            if res is None:
                res = i.length * sizeof(i.dtype)
            else:
                res = res + i.length * sizeof(i.dtype)
        if res is None:
            res = 0
        return res

    def __call__(self, *args: GMTensor):
        for a in args:
            assert isinstance(a, GMTensor), 'Forward inputs must be GMTensors'
        self.inner_forward(*args)

    def inner_forward(self, *args: Union[GMTensor, Var]):
        for a in args:
            if isinstance(a, GMTensor):
                self._input_tensors.append(a)
        self._curr_phase = 'computing'
        context.active_kernel = self
        if hasattr(self, 'forward'):
            self.forward(*args)  # type: ignore
        context.active_kernel = None
        self._curr_phase = 'finished'


    def append(self, inst: Instruction):
        if self._curr_phase=='init':
            self._init_inst_list.append(inst)
        elif self._curr_phase=='computing':
            self._compute_inst_list.append(inst)
        else:
            raise Exception(f'Can only call in initialize or forward. Not {self._curr_phase}')

    def run_module(self, mod: Union[CubeModule, VecModule]):
        if mod not in self._modules:
            self._modules[mod] = mod._name
        self.append(Instruction('RUN', name=self._modules[mod], mode='vec' if isinstance(mod, VecModule) else 'cube'))


    # code gen related
    def get_all_args(self):
        args: list[str] = []
        for i in self._input_tensors:
            args.append(f'GM_ADDR {i.name}')
        if len(self._workspace_list)>0:
            args.append('GM_ADDR workspace')
        for i in self._input_vars:
            args.append(f'{i.dtype} {i.name}')
        return ', '.join(args)

    def get_all_args_notype(self):
        args: list[str] = []
        for i in self._input_tensors:
            args.append(f'{i.name}')
        if len(self._workspace_list)>0:
            args.append('workspace')
        for i in self._input_vars:
            args.append(f'{i.name}')
        return ', '.join(args)

    def gen_tiling(self, mode: Literal['cube', 'vec']):
        assert mode in ['cube', 'vec']
        for m in self._modules:
            names = [i.name for i in m._input_vars]
            names.append(f'{self._modules[m]}shape')
            if mode=='cube' and isinstance(m, CubeModule):
                self._h(f'tilingShape{m._name}({", ".join(names)});')
            if mode=='vec' and isinstance(m, VecModule):
                self._h(f'tilingShape{m._name}({", ".join(names)});')

    def gen_init(self, mode: Literal['cube', 'vec']):
        assert mode in ['cube', 'vec']
        for m in self._modules:
            names = [i.name for i in m._input_tensors]
            names.append(f'{self._modules[m]}shape')
            if mode=='cube' and isinstance(m, CubeModule):
                self._h(f'{self._modules[m]}.Init({", ".join(names)});')
            if mode=='vec' and isinstance(m, VecModule):
                self._h(f'{self._modules[m]}.Init({", ".join(names)});')

    def gen_compute(self, mode: Literal['cube', 'vec']):
        assert mode in ['cube', 'vec']
        if mode=='cube':
            filtered_list = [i for i in self._compute_inst_list if i.mode!='vec']
            parse_all(filtered_list, self._h)
        if mode=='vec':
            filtered_list = [i for i in self._compute_inst_list if i.mode!='cube']
            parse_all(filtered_list, self._h)

    def gen_members(self, mode: Literal['cube', 'vec']):
        assert mode in ['cube', 'vec']
        # var list
        for v in self._var_list:
            self._h(f'{v.dtype} {v.name};')
        # modules and shapes
        for m in self._modules:
            if mode=='cube' and isinstance(m, CubeModule):
                self._h(f'{m._name} {self._modules[m]};')
                self._h(f'{m._name}ShapeInfo {self._modules[m]}shape;')
            if mode=='vec' and isinstance(m, VecModule):
                self._h(f'{m._name} {self._modules[m]};')
                self._h(f'{m._name}ShapeInfo {self._modules[m]}shape;')

    def gen_workspace(self):
        if len(self._workspace_list)>0:
            self._h('int offset = 0;')
        for tsr in self._workspace_list:
            self._h(f'auto {tsr.name} = shiftAddr<{tsr.dtype}>(workspace, {str(tsr.length)}, offset);')

    def gen_golden(self, target_file: str):
        type_mapping = {
            'half'      : 'np.float16',
            'float'     : 'np.float32',
            'int'       : 'np.int32',
            'int32_t'   : 'np.int32',
            'int16_t'   : 'np.int16',
            'int8_t'    : 'np.int8',
            'void'      : 'np.int8',
            'uint32_t'  : 'np.uint32',
            'uint16_t'  : 'np.uint16',
            'uint8_t'   : 'np.uint8'
        }

        h = CodeHelper()
        h('import os')
        h('import numpy as np')
        h()
        h('np.random.seed(43)')
        h()
        h()

        h('def pack_s8tos4(x):')
        h.ir()
        h('assert x.dtype==np.int8, "Must be type int8"')
        h('x = x.reshape([-1, 2])')
        h('x1 = x[:,0]')
        h('x2 = x[:,1]')
        h('x_pack = (x1 & 0xf) + (x2 << 4)')
        h('return x_pack')
        h.il()
        h()
        h()

        h('def gen_golden_data_simple():')
        h.ir()
        for i in self._input_vars:
            if i.value is None:
                raise Exception(f'Value of Var {i.name} is None. Cannot generate test entry.')
            h(f'{i.name} = {i.value}')
        h()
        for i in self._input_tensors:
            # if not i.is_output:
            if i.length is None:
                raise Exception(f'GMTensor {i.name} has no length. Cannot generate test entry.')
            if i.dtype.ctype=='void': # int4
                h(f'{i.name} = pack_s8tos4(np.random.random([{i.length}]).astype({type_mapping[i.dtype.ctype]}) * 0.5)')
            else:
                h(f'{i.name} = np.random.random([{i.length}]).astype({type_mapping[i.dtype.ctype]}) * 0.5')
        h()
        h('# # In case ones are needed ')
        for i in self._input_tensors:
            # if not i.is_output:
            if i.length is None:
                raise Exception(f'GMTensor {i.name} has no length. Cannot generate test entry.')
            if i.dtype.ctype=='void':
                h(f'# {i.name} = pack_s8tos4(np.ones([{i.length}]).astype({type_mapping[i.dtype.ctype]}))')
            else:
                h(f'# {i.name} = np.ones([{i.length}]).astype({type_mapping[i.dtype.ctype]})')
        h()
        h()
        for i in self._input_tensors:
            # if not i.is_output:
            if i.length is None:
                raise Exception(f'GMTensor {i.name} has no length. Cannot generate test entry.')
            if not i.is_output:
                h(f"{i.name}.tofile('./workspace/input/input_{i.name}.bin')")
            else:
                h(f"{i.name}.tofile('./workspace/output/output_{i.name}.bin')")
        h.il()
        h()
        h()
        h('if __name__=="__main__":')
        h.ir()
        h("os.makedirs('./workspace/input/', exist_ok=True)")
        h("os.makedirs('./workspace/output/', exist_ok=True)")
        h('gen_golden_data_simple()')
        h()

        if os.path.exists(target_file):
            print(Text('WARNING: [gen_golden ] File exists! Please manually delete the original file.', style=Style(color='yellow', bold=True)))
        else:
            fout = open(target_file, 'w')
            fout.write(str(h))
            fout.close()

    def gen_checker(self, target_file: str):
        type_mapping = {
            'half'      : 'np.float16',
            'float'     : 'np.float32',
            'int'       : 'np.int32',
            'int32_t'   : 'np.int32',
            'int16_t'   : 'np.int16',
            'int8_t'    : 'np.int8',
            'void'      : 'np.int8',
            'uint32_t'  : 'np.uint32',
            'uint16_t'  : 'np.uint16',
            'uint8_t'   : 'np.uint8'
        }
        h = CodeHelper()
        h('import numpy as np')
        h('np.set_printoptions(linewidth=200)')
        h('np.set_printoptions(precision=4, suppress=True)')
        h('from rich.text import Text')
        h('from rich.style import Style')
        h('from rich.console import Console')
        h('console = Console()')
        h('print = Console().print')
        h("print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))")
        h()
        h()

        h('def check_diff(x, y):')
        h.ir()
        h('diff = np.abs(x - y)')
        h('arg = np.argmax(diff.flatten())')
        h("print('Left:', x.flatten()[arg], 'Right:', y.flatten()[arg], 'Index:', arg)")
        h.il()
        h()
        h()

        h('def unpack_s4tos8(x):')
        h.ir()
        h('x1 = (((x & 0xf) << 4) >> 4)')
        h('x2 = (x & np.uint8(0xf0).view(np.int8)) >> 4')
        h('x_unpack = np.stack([x1, x2], axis=-1).flatten()')
        h('return x_unpack')
        h.il()
        h()
        h()

        h("if __name__ == '__main__':")
        h.ir()

        for i in self._input_vars:
            if i.value is None:
                raise Exception(f'Value of Var {i.name} is None. Cannot generate test entry.')
            h(f'{i.name} = {i.value}')
        h()

        for i in self._input_tensors:
            if i.length is None:
                    raise Exception(f'GMTensor {i.name} has no length. Cannot generate test entry.')
            if i.is_output:
                tmpstr = 'output'
            else:
                tmpstr = 'input'
            if i.dtype.ctype=='void':
                h(f'{i.name} = unpack_s4tos8(np.fromfile("./workspace/{tmpstr}/{tmpstr}_{i.name}.bin", dtype={type_mapping[i.dtype.ctype]}))')
            else:
                h(f'{i.name} = np.fromfile("./workspace/{tmpstr}/{tmpstr}_{i.name}.bin", dtype={type_mapping[i.dtype.ctype]})')

        h()
        if len(self._workspace_list)>0:
            h("workspace = np.fromfile('./workspace/output/output_workspace.bin', dtype=np.float16)")
            h()
        h('# Computation and comparison')
        h()

        if os.path.exists(target_file):
            print(Text('WARNING: [gen_checker] File exists! Please manually delete the original file.', style=Style(color='yellow', bold=True)))
        else:
            fout = open(target_file, 'w')
            fout.write(str(h))
            fout.close()

    def gen_code(self, target_file: str, kernel_name: str):
        # generate project template files
        os.makedirs('./codebase', exist_ok=True)
        os.makedirs('./.vscode', exist_ok=True)
        # if not os.path.exists('./codebase/tensorutils.h'):
        if True:
            with open('./codebase/tensorutils.h', 'w') as fout:
                fout.write(tensorutils_template.TENSORUTILS_TEMPLATE)
            with open('./codebase/data_utils.h', 'w') as fout:
                fout.write(datautil_template.DATAUTIL_TEMPLATE)
            with open('./codebase/testmacros.h', 'w') as fout:
                fout.write(testmacros_template.TESTMACROS_TEMPLATE)
        os.makedirs('./workspace/cmake', exist_ok=True)
        if not os.path.exists('./workspace/cmake/npu_lib.cmake'):
            with open('./workspace/cmake/npu_lib.cmake', 'w') as fout:
                fout.write(workspace_template.NPU_LIB_CMAKE_TEMPLATE)
        if not os.path.exists('./workspace/build.sh'):
            with open('./workspace/build.sh', 'w') as fout:
                fout.write(workspace_template.BUILD_SH_TEMPLATE)
        if not os.path.exists('./workspace/run.sh'):
            with open('./workspace/run.sh', 'w') as fout:
                fout.write(workspace_template.RUN_SH_TEMPALTE)
        if not os.path.exists('./workspace/clear.sh'):
            with open('./workspace/clear.sh', 'w') as fout:
                fout.write(workspace_template.CLEAR_SH_TEMPLATE)
        if not os.path.exists('./b.sh'):
            with open('./b.sh', 'w') as fout:
                fout.write(workspace_template.B_SH_TEMPLATE)
        if not os.path.exists('./r.sh'):
            with open('./r.sh', 'w') as fout:
                fout.write(workspace_template.R_SH_TEMPALTE)
        if not os.path.exists('./bb.sh'):
            with open('./bb.sh', 'w') as fout:
                fout.write(workspace_template.BB_SH_TEMPLATE)
        if not os.path.exists('./rr.sh'):
            with open('./rr.sh', 'w') as fout:
                fout.write(workspace_template.RR_SH_TEMPALTE)
        if not os.path.exists('./.vscode/c_cpp_properties.json'):
            with open('./.vscode/c_cpp_properties.json', 'w') as fout:
                fout.write(workspace_template.VSCODE_CPP_PROPERTIES_TEMPLATE)
        if not os.path.exists('./.vscode/settings.json'):
            with open('./.vscode/settings.json', 'w') as fout:
                fout.write(workspace_template.VSCODE_SETTINGS_TEMPLATE)

        # make some padding
        print()
        # generate all .h sub-kernels
        for mod in self._modules:
            mod.gen_code(f'codebase/{mod._name}.h')
        # generate kernels
        h = self._h
        # header
        # # generate kernels
        # header
        h('#include "tensorutils.h"')
        for m in self._modules:
            h(f'#include "{m._name}.h"')
        h()
        h()
        # cube
        h('class CubeHandler{')
        h('public:')
        h.ir()
        h('__aicore__ inline CubeHandler(){}')
        h(f'__aicore__ inline void Init({self.get_all_args()}){{')
        h.ir()
        # h('// scalar computations')
        # self.cube_handler.gen_init_compute()
        # h()
        if len(self._workspace_list)>0:
            h('// workspace allocation')
            self.gen_workspace()
            h()
        h('// kernel tiling')
        self.gen_tiling('cube')
        h()
        h('// kernel initialization')
        self.gen_init('cube')
        h.il()
        h('}')
        h()
        h('__aicore__ inline void Compute(){')
        h.ir()
        self.gen_compute('cube')
        h.il()
        h('}')
        h.il()
        h()
        h('private:')
        h.ir()
        h('TPipe pipe;')
        self.gen_members('cube')
        h.il()
        h('};')
        h()
        h()

        # vec
        h('class VecHandler{')
        h('public:')
        h.ir()
        h('__aicore__ inline VecHandler(){}')
        h(f'__aicore__ inline void Init({self.get_all_args()}){{')
        h.ir()
        # h('// scalar computations')
        # self.vec_handler.gen_init_compute()
        # h()
        if len(self._workspace_list)>0:
            h('// workspace allocation')
            self.gen_workspace()
            h()
        h('// kernel tiling')
        self.gen_tiling('vec')
        h()
        h('// kernel initialization')
        self.gen_init('vec')
        h.il()
        h('}')
        h()
        h('__aicore__ inline void Compute(){')
        h.ir()
        self.gen_compute('vec')
        h.il()
        h('}')
        h.il()
        h()
        h('private:')
        h.ir()
        h('TPipe pipe;')
        self.gen_members('vec')
        h.il()
        h('};')
        h()
        h()

        h(f'extern "C" __global__ __aicore__ void {kernel_name}({self.get_all_args()}){{')
        h.ir()
        h('if ASCEND_IS_AIC{')
        h.ir()
        h('CubeHandler cube;')
        h(f'cube.Init({self.get_all_args_notype()});')
        h(f'cube.Compute();')
        h('mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy')
        h.il()
        h('}')
        h('if ASCEND_IS_AIV{')
        h.ir()
        h('VecHandler vec;')
        h(f'vec.Init({self.get_all_args_notype()});')
        h(f'vec.Compute();')
        h('copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy')
        h.il()
        h('}')
        h.il()
        h('}')

        # make some padding
        print()

        fout = open(os.path.join('./codebase', target_file), 'w')
        fout.write(h.result)
        fout.close()

    def gen_test_entry(self, target_file: str, kernel_file: str, kernel_name: str, n_cores: int, test_speed: bool=True, force: bool=False):
        h = CodeHelper()
        h('#include "testmacros.h"')
        h(f'#include "aclrtlaunch_{kernel_name}.h"')
        h()
        h()
        h('int32_t main(int32_t argc, char *argv[])')
        h('{')
        h.ir()
        h(f'uint32_t blockDim = {n_cores};')
        h()
        h('// constants')
        for i in self._input_vars:
            if i.value is None:
                raise Exception(f'Value of Var {i.name} is None. Cannot generate test entry.')
            h(f'{i.dtype} {i.name} = {i.value};')
        h('')
        h('INIT_ACL();')
        h()
        h('// Define Tensors')
        for i in self._input_tensors:
            if i.length is None:
                raise Exception(f'GMTensor {i.name} has no length. Cannot generate test entry.')
            if i.dtype in [DT.half, DT.bf16]:
                sizestr = 'sizeof(uint16_t)'
            elif i.dtype==DT.int4:
                sizestr = 'sizeof(uint8_t)/2'
            else:
                sizestr = f'sizeof({i.dtype})'
            h(f'DEFINE_VARIABLE({i.name}, {i.length}*{sizestr});')
        h()
        if len(self._workspace_list)>0:
            h(f'DEFINE_VARIABLE(workspace, {self.get_total_workspace_size()});')
            h()
        h('// Copy input tensors')
        for i in self._input_tensors:
            if not i.is_output:
                h(f'COPY_VARIABLE({i.name});')
        h()
        h('// Run')
        h(f'RUN_KERNEL({kernel_name}, {self.get_all_args_notype()});')
        h()
        h.il()
        h('#ifndef NOSPEEDTEST')
        h.ir()
        if test_speed:
            h(f'SPEED_TEST({kernel_name}, {self.get_all_args_notype()});')
        else:
            h(f'// SPEED_TEST({kernel_name}, {self.get_all_args_notype()});')
        h.il()
        h('#endif')
        h.ir()
        h()
        h('// Write Tensors')
        h('SYNC_STREAM();')

        has_output = False
        for i in self._input_tensors:
            if i.is_output:
                h(f'WRITE_VARIABLE({i.name});')
                has_output = True
        if not has_output:
            print(Text('Reminder: No output GMTensor. May forgot to set is_output=True', style=Style(color='deep_sky_blue2', bold=True)))
        if len(self._workspace_list)>0:
            h('WRITE_VARIABLE(workspace);')
        h()
        h('// End')

        h('END_ACL();')
        h()
        h('return 0;')

        h.il()
        h('}')
        h('// Auto-generated code. Readability is not guaranteed')

        if os.path.exists(os.path.join('./codebase', target_file)) and not force:
            print(Text('WARNING: File exists! Manually delete the original file or set force=True', style=Style(color='yellow', bold=True)))
        else:
            fout = open(os.path.join('./codebase', target_file), 'w')
            fout.write(h.result)
            fout.close()
            fout = open('workspace/CMakeLists.txt', 'w')
            fout.write(cmake_template.cmake_template.replace('main.cpp', target_file).replace('kernel_custom.cpp', kernel_file))
            fout.close()


