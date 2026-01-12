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
"""STest 用例并行执行.
"""
import argparse
import logging
from typing import List, Any, Optional, Dict

from stest_accelerate import STestAccelerate
from accelerate.gtest_accelerate import GTestAccelerate


class DistributedSTestAccelerate(STestAccelerate):
    """分布式STest执行加速

    支持多卡并行执行，通过设备分组实现分布式测试。
    继承自STestAccelerate，使用父类的device_list参数，按照rank_size进行设备分组。
    """

    @property
    def mark(self) -> str:
        return "Distributed-STest"

    @staticmethod
    def main() -> bool:
        """分布式主处理流程"""
        parser = argparse.ArgumentParser(description="Distributed STest Execute Accelerate", epilog="Best Regards!")
        DistributedSTestAccelerate.reg_args(parser=parser)
        
        # 继承父类的device参数
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                           help="Specific parallel accelerate device, "
                                "If this parameter is not specified, 0 device will be used by default.")
        # 只保留rank_size参数
        parser.add_argument("--rank_size", type=int, required=True,
                          help="Number of devices per test group")
        
        args = parser.parse_args()
        
        # 获取设备列表（继承父类的逻辑）
        device_list = [0]
        if args.device is not None:
            device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        
        # 设备分组处理（顺序分组）
        device_groups = DistributedSTestAccelerate._group_devices_by_rank_size(
            devices=device_list, 
            rank_size=args.rank_size
        )
        
        # 创建执行参数
        params = []
        for group_id, device_group in enumerate(device_groups):
            p = GTestAccelerate.ExecParam(
                cntr_id=group_id,
                envs_func=DistributedSTestAccelerate.set_distributed_device_envs,
                custom={
                    "device_group": device_group,
                    "rank_size": args.rank_size,
                    "group_id": group_id
                }
            )
            params.append(p)
        
        logging.info("Created %d device groups with rank_size %d from %d devices: %s", 
                    len(device_groups), args.rank_size, len(device_list), device_groups)
        
        ctrl = DistributedSTestAccelerate(args=args, params=params, cntr_name="DeviceGroup")
        ctrl.process()
        return ctrl.post()

    @staticmethod
    def _group_devices_by_rank_size(devices: List[int], rank_size: int) -> List[List[int]]:
        """按照rank_size对设备进行顺序分组
        
        :param devices: 设备列表（从父类继承）
        :param rank_size: 每组设备数量
        :return: 设备分组列表
        """
        if len(devices) < rank_size:
            raise ValueError(f"Available devices ({len(devices)}) are less than required rank_size ({rank_size})")
        
        # 顺序分组策略
        device_groups = []
        sorted_devices = sorted(devices)
        
        for i in range(0, len(sorted_devices), rank_size):
            group = sorted_devices[i:i + rank_size]
            if len(group) == rank_size:  # 只保留完整的分组
                device_groups.append(group)
        
        return device_groups

    @staticmethod
    def set_distributed_device_envs(p: Any) -> Optional[Dict[str, str]]:
        """设置分布式设备环境变量
        
        多卡用例通过TILE_FWK_DEVICE_ID_LIST环境变量指定使用的设备组
        """
        custom_data = p.custom
        device_group = custom_data["device_group"]
        
        # 将设备列表转换为逗号分隔的字符串
        device_list_str = ",".join(str(device_id) for device_id in device_group)
        
        return {
            "TILE_FWK_DEVICE_ID_LIST": device_list_str,  # 多卡设备列表
        }


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    exit(0 if DistributedSTestAccelerate.main() else 1)
