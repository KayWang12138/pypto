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
""" 自动拷贝 AICPU 二进制.

自动拷贝 AICPU 二进制.
"""
import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path


class CopyCtrl:

    def __init__(self, args):
        self.binary: Path = Path(args.binary[0])
        self.device_id: int = args.device
        self.cfg_json: Path = args.json if args.json else Path(Path.home(),
                                                               "tile_fwk_experiment_copy_aicpu.json").resolve()

    @staticmethod
    def main():
        """ 主处理流程 """
        parser = argparse.ArgumentParser(description=f"Aicpu Copy.", epilog="Best Regards!")
        # 参数注册
        parser.add_argument("-b", "--binary", nargs=1, type=Path, required=True,
                            help="Specific binary root path.")
        parser.add_argument("-d", "--device", nargs="?", type=int, default=0,
                            help="Device ID, default 0.")
        parser.add_argument("-j", "--json", nargs=1, type=Path, required=False, default=None,
                            help="Specific config json")
        # 流程处理
        ctrl = CopyCtrl(args=parser.parse_args())
        ctrl.process()

    def process(self):
        if not self.cfg_json.exists():
            logging.error("Auto copy aicpu binary config file(%s) not exist, won't copy.", self.cfg_json)
            return
        with open(str(self.cfg_json), 'r', encoding='utf-8') as fh:
            desc = json.load(fh)
            cfg = desc.get(f"device_{self.device_id}", {})
            ip = cfg.get("IP", "")
            dest = cfg.get("Path", "")
            if ip != "" and dest != "":
                d_path = Path(dest)
                cmd = f"scp {self.binary} HwHiAiUser@{ip}:{d_path}"
                logging.info("[BGN] Copy AICPU binary auto, Device[%s], cmd=%s", self.device_id, cmd)
                ret = subprocess.run(shlex.split(cmd),
                                     capture_output=False, check=True, text=True, encoding='utf-8')
                ret.check_returncode()
                logging.info("[END] Copy AICPU binary auto, Device[%s]", self.device_id)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    CopyCtrl.main()
