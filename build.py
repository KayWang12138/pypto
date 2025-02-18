#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" 构建总入口.

构建总入口.
"""
import argparse
import logging
import multiprocessing


class BuildCtrl:
    """ 构建过程控制.

    本类包含由命令行指定或解析出的控制标记/参数，以控制构建过程执行.
    """

    def __init__(self, args):
        pass

    @staticmethod
    def main():
        """ 主处理流程 """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"Build Ctrl.", epilog="Best Regards!")
        parser.add_argument("-t", "--targets", nargs="?", type=str, action="append",
                            help="targets, specific build targets, "
                                 "If you specify more than one, all targets within the specified range are built.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int,
                            default=min(int(multiprocessing.cpu_count()), 32),
                            help="job num, specific job num of build.")
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("-u", "--utest", nargs="?", type=str, default="",
                            help="utest, enable UTest scene, specific UTest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("-s", "--stest", nargs="?", type=str, default="",
                            help="stest, enable STest scene, specific STest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific STest golden path.")
        parser.add_argument("--stest_golden_path_clean", action="store_true", default=False,
                            help="Clean STest golden.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")
        parser.add_argument("-d", "--device", nargs="?", type=int, default=0,
                            help="Device ID, default 0.")
        parser.add_argument("--timeout", nargs="?", type=int,
                            default=0,
                            help="build task timeout.")
        # 流程处理
        ctrl = BuildCtrl(args=parser.parse_args())
        logging.info("%s", ctrl)
        ctrl.clean()
        ctrl.configure()
        ctrl.build()

    def clean(self):
        pass

    def configure(self):
        pass

    def build(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    BuildCtrl.main()
