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

from typing import Sequence, Union
from pathlib import Path
from .parser import get_args_str, get_return_type, CodeHelper
from .utils import Tensor, Instruction, CustStruct, Tuple, Var, Vector, ConfigMap


def parse_ut_code(func_name: str, args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
          return_type: str) -> str:
    
    args_str = get_args_str(args)
    helper = CodeHelper()
    helper(f'{return_type} {func_name}({args_str});\n\n')
    helper(f'TEST_F({func_name}Test, utest) {{')
    helper.ir()
    arg_list = []
    for arg in args:
        if isinstance(arg, Tensor):
            shape_str = str(arg.get_shape())[1: -1] if arg.get_shape() else ''
            arg_list.append(f'tsr{arg.idx}')
            helper(f'Tensor tsr{arg.idx}({arg.dtype}, {{{shape_str}}}, "tsr{arg.idx}");')
        elif isinstance(arg, Var):
            helper(f"{arg.dtype} v{arg.idx} = ({arg.dtype}){arg.value};")
            arg_list.append(f'v{arg.idx}')
        else:
            raise NotImplementedError
    
    helper("config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);\n")
    helper('FUNCTION("CUSTOM_FUNC") {')
    helper.ir()
    helper(f'{func_name}({", ".join(arg_list)});')
    helper.il()
    helper('}\n')
    helper('ALOG_INFO(Program::GetInstance().Dump());')
    helper.il()
    helper('}')
    
    return helper.res


def gen_st_prefix_code(helper: CodeHelper, return_type: str, func_name: str, args_str: str):
    helper('class CustomOperatorTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};\n')
    helper(f'{return_type} {func_name}({args_str});\n')
    helper(f'TEST_F({func_name}Test, stest) {{')
    helper.ir()
    helper(f'aclInit(nullptr);')
    helper(f'rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());\n')


def gen_st_suffix_code(helper, args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
    cpp_dtype_mapper: dict):
    for arg in args:
        if isinstance(arg, Tensor) and arg.is_output:
            shape_str = str(arg.get_shape())[1:-1] if arg.get_shape() else ''
            helper(f'std::vector<{cpp_dtype_mapper.get(arg.dtype, "ILLEGAL_TYPE")}> \
tsr{arg.idx}_golden({shape_str.replace(", ", " * ")});')
            helper(f'std::vector<{cpp_dtype_mapper.get(arg.dtype, "ILLEGAL_TYPE")}> \
tsr{arg.idx}_output({shape_str.replace(", ", " * ")});')
            helper(f'machine::GetRA()->CopyFromTensor((uint8_t *)tsr\
{arg.idx}_output.data(), (uint8_t *)tsr{arg.idx}_ptr, {shape_str.replace(", ", " * ")} * sizeof({arg.dtype}));')
            helper(f'readInput(GetGoldenDir() + "/tsr{arg.idx}_golden.bin", tsr{arg.idx}_golden);')
            helper(f'int ret{arg.idx} = resultCmp(tsr{arg.idx}_golden, tsr{arg.idx}_output, 0.001f);')
            helper(f'EXPECT_EQ(ret{arg.idx}, true);')
    helper.il()
    helper('}')
    

def parse_st_code(func_name: str, args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
          return_type: str) -> str:
    cpp_dtype_mapper = {
        'DataType::DT_INT8': 'int8_t',
        'DataType::DT_INT16': 'int16_t',
        'DataType::DT_INT32': 'int32_t',
        'DataType::DT_INT64': 'int64_t',
        'DataType::DT_FP16': 'npu::tile_fwk::float16',
        'DataType::DT_FP32': 'float',
        'DataType::DT_DOUBLE': 'double',
        'DataType::DT_BF16': 'npu::tile_fwk::bfloat16',
        'DataType::DT_UINT8': 'uint8_t',
        'DataType::DT_UINT16': 'uint16_t',
        'DataType::DT_UINT32': 'uint32_t',
        'DataType::DT_UINT64': 'uint64_t'
    }
    args_str = get_args_str(args)
    helper = CodeHelper()
    gen_st_prefix_code(helper, return_type, func_name, args_str)
    
    arg_list = []
    for arg in args:
        if isinstance(arg, Tensor):
            shape_str = str(arg.get_shape())[1:-1] if arg.get_shape() else ''
            if arg.is_output:
                helper(f'uint8_t *tsr{arg.idx}_ptr = allocDevAddr(\
{shape_str.replace(", ", " * ")} * sizeof({cpp_dtype_mapper.get(arg.dtype, "ILLEGAL_TYPE")}));')
            else:
                helper(f'void *tsr{arg.idx}_ptr = readToDev<\
{cpp_dtype_mapper.get(arg.dtype, "ILLEGAL_TYPE")}>(GetGoldenDir() + \
"/tsr{arg.idx}.bin", {shape_str.replace(", ", " * ")});')
            helper(f'Tensor tsr{arg.idx}({arg.dtype}, {{{shape_str}}}, \
(uint8_t *)tsr{arg.idx}_ptr, "tsr{arg.idx}");')
            arg_list.append(f'tsr{arg.idx}')
        elif isinstance(arg, Var):
            helper(f'{arg.dtype} v{arg.idx} = ({arg.dtype}){arg.value};')
            arg_list.append(f'v{arg.idx}')
        else:
            raise NotImplementedError
    
    func_arg_list = list(filter(lambda x: 'v' not in x, arg_list))
    helper(f'PROGRAM("CUSTOM") {{')
    helper.ir()
    helper('Program::GetInstance().GetConfig().Reset();')
    helper(f'FUNCTION("CUSTOM_FUNC", {{.funcType = FunctionType::STATIC}}, {{{", ".join(func_arg_list)}}}) {{')
    helper.ir()
    helper(f'{func_name}({", ".join(arg_list)});')
    helper.il()
    helper('}')
    helper.il()
    helper('}')
    helper('DevFuncRunner::Run(Program::GetInstance().GetLastFunction());\n')
    
    gen_st_suffix_code(helper, args, cpp_dtype_mapper)
    
    return helper.res


def gen_ast_prefix_code(include_file_list: list[str]):
    ast_code_prefix = ''
    for file in include_file_list:
        ast_code_prefix += ("#include " + "\"" + file + "\"\n")
    ast_code_prefix += "\nusing namespace npu::tile_fwk;\n\n"
    ast_code_prefix += "namespace npu::tile_fwk {\n"
    return ast_code_prefix


def gen_ast_ut_code(directory_path: str, operator_name: str, \
    args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
    return_type: str):
    ast_ut_include_file_list = [
        "gtest/gtest.h",
        "tilefwk/tilefwk_op.h",
        "tilefwk/tilefwk.h",
        "interface/inner/tilefwk.h",
        "interface/configs/config_storage.h",
        "interface/tensor/logical_tensor.h",
        "interface/tensor/raw_tensor.h",
        "interface/interpreter/raw_tensor_data.h",
        "interface/configs/config_manager.h",
        "interface/tensor/float.h"
    ]
    
    ast_ut_code_prefix = gen_ast_prefix_code(ast_ut_include_file_list)
    ast_ut_code_prefix += f'\nclass {operator_name}Test : public testing::Test {{\n'
    ast_ut_code_prefix += 'public:\n'
    ast_ut_code_prefix += 'static void SetUpTestCase() {}\n'
    ast_ut_code_prefix += 'static void TearDownTestCase() {}\n'
    ast_ut_code_prefix += 'void SetUp() override { Program::GetInstance().Reset(); }\n'
    ast_ut_code_prefix += 'void TearDown() override {}\n};\n\n'

    ut_code = parse_ut_code(operator_name, args, return_type)
    
    ast_ut_code_suffix = '} // namespace'
    
    with open(Path(directory_path, f'tests/ut/operator/src/test_{operator_name}.cpp'), 'w') as f:
        f.write(ast_ut_code_prefix)
        f.write(ut_code)       
        f.write(ast_ut_code_suffix)


def gen_ast_st_code(directory_path: str, operator_name: str, \
    args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
    return_type: str):
    ast_st_include_file_list = [
        "test_suite_stest_ops.h",
        "test_dev_func_runner.h"
    ]
    
    ast_st_code_prefix = gen_ast_prefix_code(ast_st_include_file_list)
    st_code = parse_st_code(operator_name, args, return_type)
    ast_st_code_suffix = '} // namespace'
    
    with open(Path(directory_path, f'tests/st/operator/src/test_{operator_name}.cpp'), 'w') as f:
        f.write(ast_st_code_prefix)
        f.write(st_code)        
        f.write(ast_st_code_suffix)
            

def gen_ast_golden_script(directory_path: str, operator_name: str, args: Sequence):
    numpy_dtype_mapper = {
        'DataType::DT_INT8': 'np.int8',
        'DataType::DT_INT16': 'np.int16',
        'DataType::DT_INT32': 'np.int32',
        'DataType::DT_INT64': 'np.int64',
        'DataType::DT_FP16': 'np.float16',
        'DataType::DT_FP32': 'np.float32',
        'DataType::DT_DOUBLE': 'np.float64',
        'DataType::DT_BF16': 'np.bfloat16',
        'DataType::DT_UINT8': 'np.uint8',
        'DataType::DT_UINT16': 'np.uint16',
        'DataType::DT_UINT32': 'np.uint32',
        'DataType::DT_UINT64': 'np.uint64'
    }
    
    golden_script_code = \
f'''import sys
import logging
from pathlib import Path
from typing import List

import numpy as np

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister
    
@GoldenRegister.reg_golden_func(
    case_names=[
        "{operator_name}Test.stest"
    ]
)
def run(case_name: str, output: Path) -> bool:
'''
    output_tsr = []
    for idx, arg in enumerate(args):
        if isinstance(arg, Tensor):
            golden_script_code += f"    tsr{idx} = np.random.uniform(-1, 1, {arg.get_shape()}).astype(\
{numpy_dtype_mapper.get(arg.dtype, 'ILLEGAL_TYPE')})\n"
            if not arg.is_output:
                golden_script_code += f"    tsr{idx}.tofile(Path(output, 'tsr{idx}.bin'))\n"
            else:
                output_tsr.append(f'tsr{idx}')
        elif isinstance(arg, Var):
            golden_script_code += f"    v{idx} = {arg.value}\n"
        else:
            raise NotImplementedError(f"dtype of {type(arg)} is not implemented")
    golden_script_code += "    ################ complete golden logic here ###################\n\n\n\n"
    golden_script_code += "    ################ golden logic finish ###################"
    for tsr in output_tsr:
        golden_script_code += f"\n    {tsr}.tofile(Path(output, '{tsr}_golden.bin'))\n"
        
        
    golden_script_code += \
f'''
    return True

def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "{operator_name}Test.stest",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = run(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
'''
    with open(Path(directory_path, f'tests/cmake/scripts/golden/op/test_{operator_name}.py'), 'w') as f:
        f.write(golden_script_code)
            
        
