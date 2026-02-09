#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import json
import logging
import argparse
import torch
import ml_dtypes
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from tensor_diff import TensorComparator
from run_float_diff import DataDiffAnalyzer


class PassComparator:
    """Pass比较器类，封装所有比较逻辑"""
    
    def __init__(self, 
                 verify_path_pass1: str = "",
                 verify_path_pass2: str = "",
                 atol: float = 1e-3,
                 rtol: float = 1e-3,
                 topk: int = 50,
                 is_sort: bool = False):
        """
        初始化比较器
        
        参数:
            verify_path_pass1: 第一个pass的验证文件路径
            verify_path_pass2: 第二个pass的验证文件路径
            atol: 绝对容差
            rtol: 相对容差
            topk: 打印前k个差异
            is_sort: 是否排序数据
        """
        self.verify_path_pass1 = verify_path_pass1
        self.verify_path_pass2 = verify_path_pass2
        self.atol = atol
        self.rtol = rtol
        self.topk = topk
        self.is_sort = is_sort
        
        # 数据类型映射字典（可以作为类属性或实例属性）
        self.dtype_dict = {
            "DT_BF16": ml_dtypes.bfloat16,
            "DT_FP32": np.float32,
            "DT_FP16": np.float16,
            "DT_INT32": np.int32,
            "DT_INT8": np.int8,
            "DT_INT64": np.int64,
            "DT_INT16": np.int16
        }
        
        self.torch_dtype_dict = {
            ml_dtypes.bfloat16: torch.bfloat16,
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int8: torch.int8,
            np.int64: torch.int64,
            np.int16: torch.int16
        }
    
    def is_contain(self, a_offset: List[int], b_offset: List[int], 
                   a_shape: List[int], b_shape: List[int]) -> bool:
        """
        检查tensor a是否完全包含在tensor b中
        
        参数:
            a_offset: tensor a的偏移量
            b_offset: tensor b的偏移量
            a_shape: tensor a的形状
            b_shape: tensor b的形状
        
        返回:
            bool: True表示a包含在b中
        """
        for i in range(len(a_offset)):
            if (a_offset[i] < b_offset[i]) or \
               ((a_offset[i] + a_shape[i]) > (b_offset[i] + b_shape[i])):
                return False
        return True
    
    def compare_data(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        比较两个数据项
        
        参数:
            a: 第一个数据项的字典
            b: 第二个数据项的字典
        
        返回:
            bool: True表示数据一致，False表示不一致
        """
        # 获取数据类型
        dtype = a["outputDtype"]
        
        # 构建文件路径
        f_a = self._build_file_path(a, self.verify_path_pass1)
        f_b = self._build_file_path(b, self.verify_path_pass2)
        
        # 检查文件是否存在
        if not os.path.exists(f_a) or not os.path.exists(f_b):
            logging.info("有些文件不存在，直接跳过")
            return True
        
        # 解析偏移量和形状
        a_offset = json.loads(a["tensorOffset"])
        b_offset = json.loads(b["tensorOffset"])
        a_shape = json.loads(a["outputValidShape"])
        b_shape = json.loads(b["outputValidShape"])
        
        # 获取numpy数据类型
        np_dtype_a = self.dtype_dict.get(dtype)
        if np_dtype_a is None:
            logging.error(f"不支持的数据类型: {dtype}")
            return False
        
        # 读取数据
        data_a = np.fromfile(f_a, np_dtype_a)
        data_b = np.fromfile(f_b, np_dtype_a)
        data_b = data_b.reshape(b_shape)
        
        # 构建切片
        slices = []
        for dim in range(data_b.ndim):
            start = a_offset[dim] - b_offset[dim]
            stop = start + a_shape[dim]
            slices.append(slice(start, stop))
        
        # 提取对应切片
        b_slice = data_b[tuple(slices)]
        
        # 转换为torch tensor
        t_dtype_a = self.torch_dtype_dict.get(np_dtype_a)
        
        if dtype == "DT_BF16":
            tensor_a = torch.frombuffer(
                memoryview(data_a.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
            tensor_b = torch.frombuffer(
                memoryview(b_slice.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
        else:
            tensor_a = torch.from_numpy(data_a).to(dtype=t_dtype_a)
            tensor_b = torch.from_numpy(b_slice).to(dtype=t_dtype_a)
        
        # 比较数据
        comparator = TensorComparator()
        result_is_close, result_reason_str, result_info = comparator.check_isclose(
           tensor_a, tensor_b, self.rtol, self.atol, calc_dtype=torch.float32, is_ignore_bothzero=True, is_detail=True
        )
        
        # 打印日志
        self._log_comparison_info(a, b, a_shape, b_shape, a_offset, b_offset)
        
        if not result_is_close:
            comparator.print_isclose_info(result_is_close, result_reason_str, result_info)
            logging.error("数据对比失败")
            analyzer = DataDiffAnalyzer()
            analyzer.fix_input_and_compute(data_a, b_slice, [data_a.dtype, b_slice.dtype], self.is_sort)
            return False
        
        logging.info("数据对比通过")
        return True
    
    def loop_compare(self, pass_a: str, pass_b: str, df_loop: pd.DataFrame, 
                    rawTensorList: List[int] = None) -> bool:
        """
        比较一个循环内的所有数据
        
        参数:
            pass_a: 第一个pass名称
            pass_b: 第二个pass名称
            df_loop: 包含循环数据的DataFrame
            rawTensorList: 指定要比较的raw tensor列表
        
        返回:
            bool: 所有比较是否都通过
        """
        df_a = df_loop[df_loop["passName"].str.contains(pass_a)]
        df_b = df_loop[df_loop["passName"].str.contains(pass_b)]
        
        values_a = df_a["rawTensorMagic"].dropna().unique()
        values_b = df_b["rawTensorMagic"].dropna().unique()
        
        # 获取共同的raw tensor
        common_values_list = list(set(values_a) & set(values_b))
        if len(rawTensorList) != 0:
            common_values_list = rawTensorList
        
        for raw_magic in common_values_list:
            a_records = df_a[df_a["rawTensorMagic"] == raw_magic].to_dict(orient='records')
            b_records = df_b[df_b["rawTensorMagic"] == raw_magic].to_dict(orient='records')
            
            # 遍历数据多的一方
            if len(a_records) < len(b_records):
                a_records, b_records = b_records, a_records
            
            for ai in a_records:
                for bi in b_records:
                    a_offset = json.loads(ai["tensorOffset"])
                    b_offset = json.loads(bi["tensorOffset"])
                    a_shape = json.loads(ai["outputValidShape"])
                    b_shape = json.loads(bi["outputValidShape"])
                    
                    if self.is_contain(a_offset, b_offset, a_shape, b_shape):
                        is_right = self.compare_data(ai, bi)
                        if not is_right:
                            return False
        
        return True
    
    def pass_compare(self, pass_a: str, pass_b: str, 
                    paths: List[str] = None, 
                    rawTensorList: List[int] = None) -> None:
        """
        主比较函数
        
        参数:
            pass_a: 第一个pass名称
            pass_b: 第二个pass名称
            paths: 指定要比较的路径列表
            rawTensorList: 指定要比较的raw tensor列表
        """
        # 读取CSV文件
        csv_path = os.path.join(self.verify_path_pass1, "verify_result.csv")
        df = pd.read_csv(csv_path, encoding="utf-8", 
                        na_values=["", " ", "NaN", "NA"])
        
        # 筛选两个pass的数据
        df_pass = df[df["passName"].str.contains(f'{pass_a}|{pass_b}', 
                                                 na=False, regex=True)]
        
        # 确定要比较的路径
        if paths == []:
            paths = df_pass["verifyType"].dropna().unique()
        
        for path in paths:
            df_path = df_pass[df_pass["verifyType"] == path]
            loop_info_list = df_path["loopInfo"].dropna().unique()
            
            for loop_info in loop_info_list:
                df_loop = df_path[df_path["loopInfo"] == loop_info]
                res = self.loop_compare(pass_a, pass_b, df_loop, rawTensorList)
                if not res:
                    logging.error(f"比较失败: pass={pass_a}/{pass_b}, "
                                 f"path={path}, loop={loop_info}")
                    return
    
    def _build_file_path(self, data: Dict[str, Any], base_path: str) -> str:
        """构建文件路径"""
        if data["passName"] == "tensor_graph":
            return os.path.join(base_path, data["passName"], data["outputTensor"])
        else:
            return os.path.join(base_path, data["passName"], 
                               data["verifyType"], data["outputTensor"])
    
    def _log_comparison_info(self, a: Dict, b: Dict, 
                            a_shape: List, b_shape: List,
                            a_offset: List, b_offset: List):
        """记录比较信息"""
        logging.info("------" * 10)
        logging.info(f'functionName : {a["verifyType"]}')
        logging.info(f'rawTensorMagic : {a["rawTensorMagic"]}, path : {a["loopInfo"]}')
        logging.info(f'line : {a["No."]}, 数据a的shape: {a_shape}, '
                    f'offset: {a_offset}, dtype: {a["outputDtype"]}')
        logging.info(f'line : {b["No."]}, 数据b的shape: {b_shape}, '
                    f'offset: {b_offset}, dtype: {b["outputDtype"]}')


def main():
    """主函数：解析参数并运行比较"""
    parser = argparse.ArgumentParser(
        description="Pass Compare",
        epilog="示例: python script.py --p ExpandFunction RemoveUndrivenView"
    )
    
    parser.add_argument("--p", nargs='*', type=str, default=[], required=True,
                       help="要比较的两个pass名称，用空格分隔")
    parser.add_argument("--path", nargs='*', type=str, default=[],
                       help="要比较的路径名称，用空格分隔。留空则比较所有路径")
    parser.add_argument("--raw", nargs='*', type=int, default=[],
                       help="指定要比较的raw tensors，用空格分隔。留空则比较所有")
    parser.add_argument("--verify_path", nargs='*', type=str, default=[],
                       help="验证文件目录，以'/'结尾。如果提供2个值，分别代表两个pass的路径")
    parser.add_argument("--sort", action='store_true',
                       help="是否在绘图时排序数据（默认: False）")
    parser.add_argument("--atol", type=float, default=1e-3,
                       help="绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-3,
                       help="相对容差")
    parser.add_argument("--topk", type=int, default=50,
                       help="打印差异的行数")
    
    args = parser.parse_args()
    
    # 参数验证
    if len(args.p) != 2:
        logging.error("传入的pass个数不为2!")
        sys.exit(1)
    
    # 设置验证路径
    if len(args.verify_path) == 2:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[1]
    elif len(args.verify_path) == 1:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[0]
    else:
        logging.error("verify_path参数不正确!")
        sys.exit(1)
    
    # 创建比较器实例
    comparator = PassComparator(
        verify_path_pass1=verify_path_pass1,
        verify_path_pass2=verify_path_pass2,
        atol=args.atol,
        rtol=args.rtol,
        topk=args.topk,
        is_sort=args.sort
    )
    
    # 记录配置信息
    logging.info(f"对比的两个pass为 {args.p[0]}, {args.p[1]}")
    logging.info(f"rawTensorList: {args.raw}")
    logging.info(f"path: {args.path}")
    logging.info(f"verify_path_pass1: {verify_path_pass1}")
    logging.info(f"verify_path_pass2: {verify_path_pass2}")
    
    # 执行比较
    comparator.pass_compare(
        pass_a=args.p[0],
        pass_b=args.p[1],
        paths=args.path,
        rawTensorList=args.raw
    )


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式（含时间、级别、内容）
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler("app.log", encoding="utf-8")  # 输出到文件（持久化）
        ]
    )
    
    main()