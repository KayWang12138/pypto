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
from ..utils import CodeHelper, Var, Instruction, Tensor
from collections.abc import Callable

_select_cnt = 0

def set_flag(i: Instruction, h: CodeHelper):
    h(f'set_flag(PIPE_{i.src}, PIPE_{i.dst}, (event_t){i.idd});')

def wait_flag(i: Instruction, h: CodeHelper):
    h(f'wait_flag(PIPE_{i.src}, PIPE_{i.dst}, (event_t){i.idd});')

def bar(i: Instruction, h: CodeHelper):
    h(f'pipe_barrier(PIPE_{i.pipe});')

def assign_var(i: Instruction, h: CodeHelper):
    h(f'{i.v.name} = {i.src};')

def create_var(i: Instruction, h: CodeHelper):
    v: Var = i.v
    h(f'{v.dtype} {v.name} = {i.value};')

def get_val(i: Instruction, h: CodeHelper):
    h(f'{i.dst} = ({i.dst.dtype}) {i.src}.GetValue(0);')

def set_val(i: Instruction, h: CodeHelper):
    h(f'{i.src}.SetValue(0, ({i.dst.dtype})({i.dst}));')

def reinterpret(i: Instruction, h: CodeHelper):
    h(f'Tensor<{i.dst.dtype}, {i.dst.pos.cce_pos()}> {i.dst} = (Tensor<{i.dst.dtype}, {i.dst.pos.cce_pos()}>){i.src};')

# flow control
def start_loop(i: Instruction, h: CodeHelper):
    h(f'for (int {i.name}={i.start}; {i.name}<{i.end}; {i.name}+={i.step}){{')
    h.ir()

def end_loop(i: Instruction, h: CodeHelper):
    h.il()
    h('}')

def start_if(i: Instruction, h: CodeHelper):
    h(f'if ({i.cond}){{')
    h.ir()

def start_elif(i: Instruction, h: CodeHelper):
    h(f'else if({i.cond}){{')
    h.ir()

def start_else(i: Instruction, h: CodeHelper):
    h('else {')
    h.ir()

def end_if(i: Instruction, h: CodeHelper):
    h.il()
    h('}')

# event
def event_set(i: Instruction, h: CodeHelper):
    h(f'{i.name}.set();')

def event_wait(i: Instruction, h: CodeHelper):
    h(f'{i.name}.wait();')

def event_setall(i: Instruction, h: CodeHelper):
    h(f'{i.name}.setall();')

def event_release(i: Instruction, h: CodeHelper):
    h(f'{i.name}.release();')

# cross-core
def vec_ready(i: Instruction, h: CodeHelper):
    h(f'VEC_READY({i.flag}, PIPE_{i.pipe});')

def wait_cube(i: Instruction, h: CodeHelper):
    h(f'WAIT_CUBE({i.flag}, PIPE_{i.pipe});')

def allvec_ready(i: Instruction, h: CodeHelper):
    h(f'ALLVEC_READY({i.flag}, PIPE_{i.pipe});')

def allvec_wait(i: Instruction, h: CodeHelper):
    h(f'ALLVEC_WAIT({i.flag}, PIPE_{i.pipe});')

# mte2
def gm_to_ub(i: Instruction, h: CodeHelper):
    h(f'copy_gm_to_ubuf({i.dst}.vptr(), {i.src}.vptr(), 0, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

# V
def ub_to_ub(i: Instruction, h: CodeHelper):
    h(f'copy_ubuf_to_ubuf({i.dst}.vptr(), {i.src}.vptr(), 0, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

# mte3
def ub_to_gm(i: Instruction, h: CodeHelper):
    h(f'copy_ubuf_to_gm({i.dst}.vptr(), {i.src}.vptr(), 0, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

# vector masks
def set_mask(i: Instruction, h: CodeHelper):
    h(f'set_vector_mask({i.high}, {i.low});')

def reset_mask(i: Instruction, h: CodeHelper):
    h('set_vector_mask(-1, -1);')


## 910B computations
# unary
def exp(i: Instruction, h: CodeHelper):
    h(f'vexp({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def ln(i: Instruction, h: CodeHelper):
    h(f'vln({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def abs(i: Instruction, h: CodeHelper):
    h(f'vabs({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def rec(i: Instruction, h: CodeHelper):
    h(f'vrec({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def sqrt(i: Instruction, h: CodeHelper):
    h(f'vsqrt({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def rsqrt(i: Instruction, h: CodeHelper):
    h(f'vrsqrt({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def relu(i: Instruction, h: CodeHelper):
    h(f'vrelu({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

# binary
def add(i: Instruction, h: CodeHelper):
    h(f'vadd({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

def sub(i: Instruction, h: CodeHelper):
    h(f'vsub({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

def mul(i: Instruction, h: CodeHelper):
    h(f'vmul({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

def div(i: Instruction, h: CodeHelper):
    h(f'vdiv({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

def vmax(i: Instruction, h: CodeHelper):
    h(f'vmax({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

def vmin(i: Instruction, h: CodeHelper):
    h(f'vmin({i.dst}.ptr(), {i.src1}.ptr(), {i.src2}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride});')

# unaryscalar
def adds(i: Instruction, h: CodeHelper):
    h(f'vadds({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def muls(i: Instruction, h: CodeHelper):
    h(f'vmuls({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def vmaxs(i: Instruction, h: CodeHelper):
    h(f'vmaxs({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def vmins(i: Instruction, h: CodeHelper):
    h(f'vmins({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def lrelu(i: Instruction, h: CodeHelper):
    h(f'vlrelu({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

def axpy(i: Instruction, h: CodeHelper):
    h(f'vaxpy({i.dst}.ptr(), {i.src}.ptr(), ({i.dst.dtype}){i.val}, {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

# cast
def cast(i: Instruction, h: CodeHelper):
    h(f'vconv_{i.src.dtype.abbr}2{i.dst.dtype.abbr}{i.mode.postfix}({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride});')

# group functions
def vcadd(i: Instruction, h: CodeHelper):
    h(f'vcadd({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcgadd(i: Instruction, h: CodeHelper):
    h(f'vcgadd({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcpadd(i: Instruction, h: CodeHelper):
    h(f'vcpadd({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcmax(i: Instruction, h: CodeHelper):
    h(f'vcmax({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride}, {i.mode});')

def vcgmax(i: Instruction, h: CodeHelper):
    h(f'vcgmax({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcmin(i: Instruction, h: CodeHelper):
    h(f'vcmin({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride}, {i.mode});')

def vcgmin(i: Instruction, h: CodeHelper):
    h(f'vcgmin({i.dst}.ptr(), {i.src}.ptr(), {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

# dup brcb
def dup(i: Instruction, h: CodeHelper):
    h(f'vector_dup({i.dst}.ptr(), ({i.dst.dtype}){i.src}, {i.repeat}, {i.dst_blk_stride}, {i.dst_blk_stride}, {i.dst_rep_stride}, {i.dst_rep_stride});')

def brcb(i: Instruction, h: CodeHelper):
    h(f'vbrcb({i.dst}.ptr(), {i.src}.ptr(), {i.dst_blk_stride}, {i.dst_rep_stride}, {i.repeat});')


INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'BAR'                   : bar,
    'SETFLAG'               : set_flag,
    'WAITFLAG'              : wait_flag,
    'ASSIGNVAR'             : assign_var,
    'CREATEVAR'             : create_var,
    'GETVAL'                : get_val,
    'SETVAL'                : set_val,
    'REINTERPRET'           : reinterpret,
    # flowcontrol
    'STARTLOOP'             : start_loop,
    'ENDLOOP'               : end_loop,
    'STARTIF'               : start_if,
    'STARTELIF'             : start_elif,
    'STARTELSE'             : start_else,
    'ENDIF'                 : end_if,
    # events
    'EVENTSET'              : event_set,
    'EVENTWAIT'             : event_wait,
    'EVENTSETALL'           : event_setall,
    'EVENTRELEASE'          : event_release,
    # cross-core
    'VECREADY'              : vec_ready,
    'WAITCUBE'              : wait_cube,
    'ALLVECREADY'           : allvec_ready,
    'ALLVECWAIT'            : allvec_wait,
    # mte2
    'GM2UB'                 : gm_to_ub,
    # V
    'UB2UB'                 : ub_to_ub,
    # mte3
    'UB2GM'                 : ub_to_gm,
    # # 910B - Vector computations
    # vector masks
    'SETMASK'               : set_mask,
    'RESETMASK'             : reset_mask,
    # unary
    'EXP'                   : exp,
    'LN'                    : ln,
    'ABS'                   : abs,
    'REC'                   : rec,
    'SQRT'                  : sqrt,
    'RSQRT'                 : rsqrt,
    'RELU'                  : relu,
    # binary
    'ADD'                   : add,
    'SUB'                   : sub,
    'MUL'                   : mul,
    'DIV'                   : div,
    'MAX'                   : vmax,
    'MIN'                   : vmin,
    # unaryscalar
    'ADDS'                  : adds,
    'MULS'                  : muls,
    'MAXS'                  : vmaxs,
    'MINS'                  : vmins,
    'LRELU'                 : lrelu,
    'AXPY'                  : axpy,
    # cast
    'CAST'                  : cast,
    # group
    'VCADD'                 : vcadd,
    'VCGADD'                : vcgadd,
    'VCPADD'                : vcpadd,
    'VCMAX'                 : vcmax,
    'VCGMAX'                : vcgmax,
    'VCMIN'                 : vcmin,
    'VCGMIN'                : vcgmin,
    # dup brcb
    'DUP'                   : dup,
    'BRCB'                  : brcb,
}


def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)
