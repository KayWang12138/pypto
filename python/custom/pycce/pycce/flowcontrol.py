# -----------------------------------------------------------------------------------------------------------
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
from .utils import Var, Instruction, DATATYPE
from typing import Union
from . import context


class Range():
    def __init__(self, start: Union[int, 'Var'], end: Union[int, 'Var'], step: Union[int, 'Var']=1):
        self.start = start
        self.end = end
        self.step = step
        self._cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._cnt>0:
            g_vec = context.active_vec
            g_cube = context.active_cube
            if g_cube is not None:
                g_cube.append(Instruction('ENDLOOP'))
            elif g_vec is not None:
                g_vec.append(Instruction('ENDLOOP'))
            else:
                raise Exception('Loop must be called in VecModule or CubeModule')
            raise StopIteration()

        self._cnt += 1
        g_vec = context.active_vec
        g_cube = context.active_cube

        if g_cube is not None:
            name = f'_loop_var_%d'%g_cube.fetch_tmp_idx()
            g_cube.append(Instruction('STARTLOOP', name=name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(name, DATATYPE.int, auto_declare=False)
            return loop_const
        elif g_vec is not None:
            name = f'_loop_var_%d'%g_vec.fetch_tmp_idx()
            g_vec.append(Instruction('STARTLOOP', name=name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(name, DATATYPE.int, auto_declare=False)
            return loop_const
        else:
            raise Exception('Loop must be called in VecModule or CubeModule')


class Loop():
    def __init__(self, name: str, start: Union[int, 'Var'], end: Union[int, 'Var'], step: Union[int, 'Var']=1) -> None:
        self.name = name
        self.start = start
        self.end = end
        self.step = step

    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel

        if g_cube is not None:
            g_cube.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int, auto_declare=False)
            return loop_const
        elif g_vec is not None:
            g_vec.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int, auto_declare=False)
            return loop_const
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int, auto_declare=False)
            loop_const.assign_module(g_kernel)
            return loop_const
        else:
            raise Exception('Loop must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('ENDLOOP'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDLOOP'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDLOOP'))
            return True
        else:
            raise Exception('Loop must be called in VecModule or CubeModule')


class If():
    def __init__(self, cond: Union[int, 'Var']) -> None:
        self.cond = cond

    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('STARTIF', cond=self.cond))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTIF', cond=self.cond))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTIF', cond=self.cond))
        else:
            raise Exception('If must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
            return True
        else:
            raise Exception('If must be called in VecModule or CubeModule')


class Elif():
    def __init__(self, cond: Union[int, 'Var']) -> None:
        self.cond = cond

    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('STARTELIF', cond=self.cond))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTELIF', cond=self.cond))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTELIF', cond=self.cond))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')


class Else():
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('STARTELSE'))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTELSE'))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTELSE'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_kernel = context.active_kernel
        if g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')
