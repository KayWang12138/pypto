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
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import logging
import string
import json
import os
from .custom_op_info import CustomOpInfo, CoreOpcodeInfo, OpIOInfo, DataType
from ..tileop import TileOpModule
from ..utils import CodeHelper

class CustomOpInfoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        return super().default(obj)


@dataclass
class CustomOpInfoGenerator:
    _custom_op_info_dict = {}
    _custom_op_file_name = "./custom_op_src/custom_op_info.json"

    def register_op(self, op_info: CustomOpInfo):
        if op_info.get_op_name() in self._custom_op_info_dict:
            logging.error("custom op name " +  op_info.get_op_name() + " is exists")
        else:
            self._custom_op_info_dict[op_info.get_op_name()] = op_info

    def get_custom_op_info_dict(self):
        return self._custom_op_info_dict

    def get_custom_op_info(self, op_name: string):
        try:
            return self._custom_op_info_dict[op_name]
        except KeyError as error:
            logging.error(op_name + " is not found!")

    def set_custom_op_info_file_name(self, name: string):
        _custom_op_file_name = name

    def dump_custom_op_info(self):
        for op_name in self._custom_op_info_dict:
            logging.info("CustomOP " + op_name + ": ")
            logging.info(self._custom_op_info_dict[op_name])

    def write_custom_op_info_file(self):
        with open(self._custom_op_file_name, "w") as json_file:
            json.dump(self._custom_op_info_dict, json_file, indent = 4, cls = CustomOpInfoJSONEncoder)

    def read_custom_op_info_file(self):
        try:
            with open(self._custom_op_file_name, "r") as file:
                data = json.loads(file.read())
                logging.debug(data)
                for key in data:
                    self.register_op(CustomOpInfo(**data[key]))

        except FileNotFoundError:
            logging.error(f"Error: The file '{self._custom_op_file_name}' was not found.")
        except json.JSONDecodeError:
            logging.error(f"Error: Could not decode JSON from '{self._custom_op_file_name}'. Check for valid JSON format.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

def snake_to_camel(s: str):
    result = []
    is_upper = True
    for c in s:
        if c == '_':
            is_upper = True
            continue
        elif is_upper:
            result.append(c.upper())
        else:
            result.append(c)
        is_upper = False
    return ''.join(result)


def generate_ascpp_interface(tileop: TileOpModule, opCodeInfo: CoreOpcodeInfo):
    h = CodeHelper()
    h("#include \"custom_operation.h\"\n")
    h("namespace npu::tile_fwk {\n")
    input_tensors = []
    output_tensors = []
    for tsr in tileop.get_input_tensors():
        if tsr.is_output:
            output_tensors.append(tsr.name)
        else:
            input_tensors.append(f"const Tensor &{tsr.name}")
    op_name = snake_to_camel(tileop.get_name())
    h(f"Tensor {op_name}({', '.join(input_tensors)});\n")
    h(f"Tensor {op_name}({', '.join(input_tensors)}) {{\n")
    h.ir()
    input_tensors = list(map(lambda x: x.replace("const Tensr &", ""), input_tensors))
    h(f"return TensorBinaryOperation(*Program::GetInstance().GetCurrentFunction(), {', '.join(input_tensors)});")
    h.il()
    h("}\n")

    h(f"class {op_name}Register {{")
    h("public:")
    h.ir()
    h(f"{op_name}Register() {{")
    h.ir()
    h(f"InferShapeRegistry::GetInstance().RegisterInferShapeFunc(\
static_cast<Opcode>(static_cast<int32_t>(Opcode::BUILT_IN_END) + \
{1 + opCodeInfo.opcode}), ElewiseInferFunc);")

    h.il()
    h("}")
    h.il()
    h("};")
    h(f"static {op_name}Register {op_name}_register;")
    h('} // namespace')
    return h.result


def generate_ascpp_pybind(tileop: TileOpModule):
    h = CodeHelper()
    h('#include "Python.h"')
    h('#include "pybind11/chrono.h"')
    h('#include "pybind11/complex.h"')
    h('#include "pybind11/functional.h"')
    h('#include "pybind11/operators.h"')
    h('#include "pybind11/stl.h"\n')
    h('#include "tilefwk/tilefwk_op.h"')
    h('#include "tilefwk/tensor.h"')
    h('#include "tilefwk/tile_shape.h"')
    h('#include "tilefwk/tilefwk.h"')
    h('#include "tilefwk/function.h"')
    h('#include "interface/inner/tilefwk.h"')
    h('#include "interface/configs/config_manager.h"')
    h('#include "interface/inner/tilefwk.h"')
    h(f'#include "custom_op_src/{tileop.get_name()}/{tileop.get_name()}.h"\n')
    h('namespace py = pybind11;')
    h('using namespace npu::tile_fwk;\n')
    h('namespace pypto {')
    h('void bind_operation(py::module &m) {')
    h.ir()
    h('m.def(')
    h.ir()
    input_tensors = []
    for tsr in tileop.get_input_tensors():
        input_tensors.append(f"const Tensor &{tsr.name}")
    input_tensors_str = ", ".join(input_tensors)
    h(f'"{tileop.get_name()}", []({input_tensors_str}) {{')
    h.ir()
    input_tensors_str = input_tensors_str.replace('const Tensor &','')
    op_name = snake_to_camel(tileop.get_name())
    h(f'return npu::tile_fwk::{op_name}({input_tensors_str});')
    h.il()
    h(f'}}, "Tensor {op_name}."')
    h.il()
    h(');')
    h.il()
    h('}')
    h('} // namespace pypto')
    return h.result


def udpate_custom_op_json(tileop: TileOpModule, opCodeInfo: CoreOpcodeInfo):
    r = CustomOpInfoGenerator()
    op_name = snake_to_camel(tileop.get_name())
    input_dict = {}
    for tsr in tileop.get_input_tensors():
        input_dict[tsr.name] = OpIOInfo(tsr.name, tsr.dtype.ctype, tsr.length)
    r.register_op(CustomOpInfo(op_name, f"{os.path.abspath(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}.h')}", \
        op_pybind_src_path = f"{os.path.abspath(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}_pybind.h')}", \
        tile_op_src_path = f"{os.path.abspath(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}_tile_op.h')}", \
        code_op_code_info = opCodeInfo, \
        input = input_dict, \
        output = {}))
    r.write_custom_op_info_file()

def generate_custom_op(tileop: TileOpModule, opCodeInfo: CoreOpcodeInfo):
    if not os.path.exists('./custom_op_src/'):
        os.mkdir(f'./custom_op_src/')

    if not os.path.exists(f'./custom_op_src/{tileop.get_name()}'):
        os.mkdir(f'./custom_op_src/{tileop.get_name()}')

    #tile op interface
    with open(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}_tile_op.h', 'w') as f:
        f.write(tileop.gen_code('test'))

    # ascendcpp interface
    with open(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}.h', 'w') as f:
        f.write(generate_ascpp_interface(tileop, opCodeInfo))

    # ascendcpp pybind
    with open(f'./custom_op_src/{tileop.get_name()}/{tileop.get_name()}_pybind.h', 'w') as f:
        f.write(generate_ascpp_pybind(tileop))

    # registration
    udpate_custom_op_json(tileop, opCodeInfo)




