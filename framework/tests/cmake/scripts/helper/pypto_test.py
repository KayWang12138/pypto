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

import abc
import os
import logging
from collections.abc import Iterable
import torch
import numpy as np
from numpy.testing import assert_allclose
import pto


class TestBuilder(abc.ABC):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__()
        
        self.params = params
        self.kernel = kernel
        self.kernel_golden = kernel_golden
        self.tiling = tiling
        
        self.input_pto_list = []
        self.output_pto_list = []
        self.input_data_list = []
        self.output_data_list = []
        self.tensor_list = []
        
        self.rtol_value = 1e-3
        self.atol_value = 1e-3
        
    def __call__(self, on_board: bool = True):
        self.run(on_board)
    
    def set_tol(self, rtol=1e-3, atol=1e-3):
        self.rtol_value = rtol
        self.atol_value = atol
    
    def setup_inputs(self, *args):
        for idx, item in enumerate(args):
            dtype = self.dtype_conversion(str(item.dtype))
            pto_tensor = pto.tensor(item.shape, dtype, f"PTO_TENSOR_{idx}")
            self.input_pto_list.append(pto_tensor)
            self.input_data_list.append(item)
    
    def get_input_list(self):
        return self.input_pto_list
    
    def get_output_list(self):
        return self.output_pto_list
    
    def get_input_data_list(self):
        return self.input_data_list
    
    def get_output_data_list(self):
        return self.output_data_list
        
    def init_output(self, goldens):
        for idx, golden in enumerate(goldens):
            dtype = self.dtype_conversion(str(golden.dtype))
            pto_tensor = pto.tensor(golden.shape, dtype, f"PTO_TENSOR_out_{idx}")
            output_data = torch.zeros(golden.shape)
            self.output_pto_list.append(pto_tensor)
            self.output_data_list.append(output_data)
        self.init_tensor_list()
    
    def init_tensor_list(self):
        self.tensor_list = self.input_pto_list + self.output_pto_list
    
    @abc.abstractmethod
    def get_input_from_param(self):
        pass
        
    def run_pto(self, kernel, tiling, on_board: bool = True):
        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)
        
        logging.info("Function compile ...")
        pto.set_vec_tile_shapes(tiling, tiling)
        with pto.function("MAIN", self.input_pto_list, self.output_pto_list):
            kernel(*self.input_pto_list, *self.output_pto_list, self.params)
        assert all(isinstance(x, pto.tensor) for x in self.output_pto_list)
        logging.info("Function compile done.")
        
        if on_board:
            logging.info("Kernel Launch ...")
            pto.runtime._device_run_once_data_from_host(self.input_data_list, self.output_data_list)
            logging.info("Kernel run finish.")
        
            for idx in range(len(self.golden_output)):
                assert_allclose(self.golden_output[idx], self.output_data_list[idx], 
                                rtol=self.rtol_value, atol=self.atol_value)

    def run(self, on_board: bool = True):
        if on_board:
            pto.runtime._device_init()
            pto.set_codegen_option("support_dynamic_unaligned", True)
        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.kernel_golden(*self.inputs))
        self.init_output(self.golden_output)
        logging.info("PTO run is called.")
        self.run_pto(self.kernel, self.tiling, on_board)
        logging.info("PTO run finished.")
        if on_board:
            pto.runtime._device_fini()
    
    def torch_convert(self, data_tuple: tuple):
        if not isinstance(data_tuple, tuple):
            data_tuple = (data_tuple, )

        def _convert(item):
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item)
            elif torch.is_tensor(item):
                return item
            elif isinstance(item, (int, float, bool)):
                return torch.tensor(item)
            elif isinstance(item, Iterable) and not isinstance(item, str):
                return type(item)(_convert(subitem) for subitem in item)
            else:
                return item
            
        result = tuple(_convert(item) for item in data_tuple)
        if not isinstance(data_tuple, tuple) and len(result) == 1:
            return result[0]
        return result
    
    def dtype_conversion(self, str_dtype):
        if str_dtype in ['int4']:
            return pto.DT_INT4
        elif str_dtype in ['int8', 'torch.int8', 'np.int8']:
            return pto.DT_INT8
        elif str_dtype in ['int16', 'torch.int16', 'np.int16']:
            return pto.DT_INT16
        elif str_dtype in ['int32', 'torch.int32', 'np.int32']:
            return pto.DT_INT32
        elif str_dtype in ['int64', 'torch.int64', 'np.int64']:
            return pto.DT_INT64
        elif str_dtype in ['float8', 'torch.float8', 'np.float8']:
            return pto.DT_FP8
        elif str_dtype in ['float16', 'half', 'torch.float16', 'np.float16']:
            return pto.DT_FP16
        elif str_dtype in ['float32', 'torch.float32', 'np.float32']:
            return pto.DT_FP32
        elif str_dtype in ['bfloat16', 'torch.bfloat16']:
            return pto.DT_BF16
        elif str_dtype in ['uint8', 'torch.uint8', 'np.uint8']:
            return pto.DT_UINT8
        elif str_dtype in ['uint16', 'torch.uint16', 'np.uint16']:
            return pto.DT_UINT16
        elif str_dtype in ['uint32', 'torch.uint32', 'np.uint32']:
            return pto.DT_UINT32
        elif str_dtype in ['uint64', 'torch.uint64', 'np.uint64']:
            return pto.DT_UINT64
        elif str_dtype in ['bool', 'torch.bool', 'np.bool_']:
            return pto.DT_BOOL
        elif str_dtype in ['torch.double', 'np.double']:
            return pto.DT_DOUBLE
        else:
            raise ValueError("undefined dtype")
