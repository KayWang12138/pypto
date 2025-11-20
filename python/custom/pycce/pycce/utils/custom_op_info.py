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
import string
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from enum import IntEnum


class CoreType(IntEnum):
    CORE_AIV = 0
    CORE_AIC = 1
    CORE_ANY = 2
    CORE_AICPU = 3
    CORE_HUB = 4
    CORE_GMATOMIC = 5

class OpCoreType(IntEnum):
    CORE_AIC = 0
    CORE_AIV = 1
    CORE_ANY = 2
    CORE_AICPU = 3
    CORE_HUB = 4
    CORE_GMATOMIC = 5

class CorePipeType(IntEnum):
    PIPE_S = 0
    PIPE_V = 1
    PIPE_M = 2
    PIPE_MTE1 = 3
    PIPE_MTE2 = 4
    PIPE_MTE3 = 5
    PIPE_ALL = 6
    PIPE_MTE4 = 7
    PIPE_MTE5 = 8
    PIPE_V2 = 9
    PIPE_FIX = 10


class MemoryType(IntEnum):
    MEM_UB = 0
    MEM_L1 = 1
    MEM_L0A = 2
    MEM_L0B = 3
    MEM_L0C = 4
    MEM_FIX = 5
    MEM_FIX_QUANT_PRE = 6
    MEM_FIX_RELU_PRE = 7
    MEM_FIX_RELU_POST = 8
    MEM_FIX_QUANT_POST = 9
    MEM_FIX_ELT_ANTIQ = 10
    MEM_FIX_MTE2_ANTIQ = 11
    MEM_BT = 12
    MEM_L2 = 13
    MEM_L3 = 14
    MEM_DEVICE_DDR = 15
    MEM_HOST1 = 16
    MEM_FAR1 = 17
    MEM_FAR2 = 18
    MEM_WORKSPACE = 19
    MEM_VECTOR_REG = 20
    MEM_UNKNOWN = 21

class DataType(IntEnum):
    DT_INT4 = 0
    DT_INT8 = 1
    DT_INT16 = 2
    DT_INT32 = 3
    DT_INT64 = 4
    DT_FP8 = 5
    DT_FP16 = 6
    DT_FP32 = 7
    DT_BF16 = 8
    DT_HF4 = 9
    DT_HF8 = 10
    DT_UINT8 = 11
    DT_UINT16 = 12
    DT_UINT32 = 13
    DT_UINT64 = 14
    DT_BOOL = 15
    DT_DOUBLE = 16
    DT_BOTTOM = 17

class CalcType(IntEnum):
    CALC_TYPE_ELTWISE = 0
    CALC_TYPE_CAST = 1
    CALC_TYPE_BROADCAST = 2
    CALC_TYPE_OTHER = 3
    CALC_TYPE_REDUCE = 4
    CALC_TYPE_MATMUL = 5
    CALC_TYPE_CONV = 6
    CALC_TYPE_MOVE_IN = 7
    CALC_TYPE_MOVE_OUT = 8
    CALC_TYPE_MOVE_LOCAL = 9
    CALC_TYPE_SYNC = 10
    CALC_TYPE_DISTRIBUTED = 11
    CALC_TYPE_SYS = 12
    CALC_TYPE_CUSTOM = 13
    CALC_TYPE_BOTTOM = 14


class TileOpFormat(IntEnum):
    TILE_OP_ND = 0
    TILE_OP_NZ = 1


class CastMode(IntEnum):
    CAST_NONE = 0
    CAST_RINT = 1
    CAST_ROUND = 2
    CAST_FLOOR = 3
    CAST_CEIL = 4
    CAST_TRUNC = 5
    CAST_ODD = 6


@dataclass
class TileOpCfg:
    name: string = "custom_tile_op"
    pipe_id_start: CorePipeType = CorePipeType.PIPE_S
    pipe_id_end: CorePipeType = CorePipeType.PIPE_S
    core_type: CoreType = CoreType.CORE_AIV


@dataclass
class CoreOpcodeInfo:
    opcode: int = 0 # hash code
    core_type: CoreType = CoreType.CORE_AIV
    name: string = "custom_op_code"
    input_memory_type: List[MemoryType] = field(default_factory = lambda : [MemoryType.MEM_UB])
    output_memory_type: List[MemoryType] = field(default_factory = lambda : [MemoryType.MEM_UB])
    tile_op_cfg: TileOpCfg = field(default_factory = TileOpCfg)
    op_calc_type: CalcType = CalcType.CALC_TYPE_ELTWISE
    attrs: List[int] = field(default_factory=list)


@dataclass
class OpIOInfo:
    name: string = "custom_op_code"
    dtype: DataType = DataType.DT_BF16
    shape: List[int] = field(default_factory=list)


@dataclass
class CustomOpInfo:
    op_name: string
    op_src_path: string = ""
    op_pybind_src_path: string = ""
    tile_op_src_path: string = ""
    code_op_code_info: CoreOpcodeInfo = ()
    input: Dict[str, OpIOInfo] = field(default_factory=dict)
    output: Dict[str, OpIOInfo] = field(default_factory=dict)

    def set_op_name(self, name):
        self.op_name = "Custom" + name;

    def set_op_src_path(self, path):
        self.op_src_path = path

    def add_op_input(self, new_in: OpIOInfo):
        if new_in.name in self.input:
            logging.error("Input " + new_in.name + " is exists!")
        else:
            self.input[new_in.name] = new_in

    def del_op_input(self, cur_in: OpIOInfo):
        try:
            self.input.pop(cur_in.name)
        except KeyError as error:
            logging.error("Input " + cur_in.name + " is not found!")

    def del_op_input(self, name: str):
        try:
            self.input.pop(name)
        except KeyError as error:
            logging.error("Input " + name + " is not found!")

    def add_op_output(self, new_out: OpIOInfo):
        if new_out.name in self.output:
            logging.error("Output " + new_out.name + " is exists!")
        else:
            self.output[new_out.name] = new_out

    def del_op_output(self, cur_out: OpIOInfo):
        try:
            self.output.pop(cur_out.name)
        except KeyError as error:
            logging.error("Outut " + cur_out.name + " is not found!")

    def del_op_output(self, name: str):
        try:
            self.output.pop(name)
        except KeyError as error:
            logging.error("Output " + name + " is not found!")

    def get_op_name(self):
        return self.op_name

    def get_op_src_path(self):
        return self.op_src_path

    def get_input_num(self):
        return len(self.input)

    def get_output_num(self):
        return len(self.output)
