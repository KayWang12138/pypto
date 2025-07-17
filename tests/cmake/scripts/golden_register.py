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
"""STest Golden 处理函数注册管理.
"""
import logging
from typing import Dict, Callable, Union, List, Optional, overload, Tuple


def match_gtest_filter(test_case_list: List[str], filter_pattern: str) -> int:
    has_fuzzy_match = False

    for test_case in test_case_list:
        if filter_pattern == test_case:
            return -2

        expected_prefix = test_case + '/'
        if filter_pattern.startswith(expected_prefix):
            index_part = filter_pattern[len(expected_prefix):]
            if index_part.isdigit():
                return int(index_part)

        if filter_pattern == test_case + '*':
            has_fuzzy_match = True

    return -1 if has_fuzzy_match else -3 


class GoldenRegister:
    # 全局回调函数注册表
    _REG_MAP: Dict[str, Callable] = {}

    @classmethod
    def reg_golden_func(cls, case_names: Union[str, List[str]]) -> Callable:
        # 注册回调函数的装饰器
        def decorator(func: Callable) -> Callable:
            case_name_list = [case_names] if isinstance(case_names, str) else case_names
            for name in case_name_list:
                ori_func = cls._REG_MAP.get(name, None)
                if ori_func:
                    logging.debug("Case(%s) update func %s -> %s to %s", name, ori_func, func, hex(id(cls._REG_MAP)))
                else:
                    logging.debug("Case(%s) register func %s to %s", name, func, hex(id(cls._REG_MAP)))
                cls._REG_MAP[name] = func
            return func
        return decorator

    @classmethod
    def get_golden_func(cls, case_name: str) -> Tuple[Optional[Callable], int]:
        """根据名称获取回调函数"""
        filter_ret = match_gtest_filter(list(cls._REG_MAP.keys()), case_name)
        if filter_ret == -3:
            return (None, filter_ret)
        elif filter_ret == -2:
            func = cls._REG_MAP[case_name]
        elif filter_ret == -1:
            func = cls._REG_MAP[case_name.rstrip('*')]
        else:
            func = cls._REG_MAP[case_name[:-len(f"\\{filter_ret}")]]
        logging.debug("Case(%s) get func %s from %s", case_name, func, hex(id(cls._REG_MAP)))
        return (func, filter_ret)

    @classmethod
    def get_golden_func_num(cls) -> int:
        return len(cls._REG_MAP)
