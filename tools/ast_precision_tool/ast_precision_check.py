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
""" 精度工具总入口.

精度工具总入口.
"""
import os
import argparse
import logging
import multiprocessing
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import json
import time


class AstPrecisionToolCtrl:
    """ 工具执行过程控制.

    本类包含由命令行指定或解析出的控制标记/参数, 以控制工具过程执行.
    """

    def __init__(self, args):
        if (args.dfx in ['dumpallop', 'dumpcopyop']) and not args.passname:
            raise argparse.ArgumentError(None, "--dumpallop and --dumpcopyop require --passname to be set")
        self.check_mode: str = args.checkmode
        self.output_path: Path = args.outputpath
        self.pass_list: Optional[list] = None if args.passlist is None else args.passlist

        self.dump_print: str = ''
        self.is_genreport: str = 'noreport'
        if args.dfx == 'dumpallop':
            self.dump_print = 'DumpAllOp'
        elif args.dfx == 'dumpcopyop':
            self.dump_print = 'DumpCopysOp'
        elif args.dfx == 'dumpcopyop':
            self.dump_print = 'DumpCopysOp'
        elif args.dfx == 'printshape':
            self.dump_print = 'PrintShape'
        elif args.dfx == 'genreport':
            self.is_genreport = args.dfx

        self.golden_dir: Optional[Path] = None if args.goldendir is None else args.goldendir
        self.golden_bin_name: Optional[list] = None if args.golden is None else args.golden
        self.dump_tensor_pt_path = "./dump_tensor_pt"

    @staticmethod
    def main():
        """ 主处理流程 """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"Ast Precision Tool Ctrl.", epilog="Best Regards!")
        parser.add_argument("-m", "--checkmode", nargs="?", type=str, default="pass",
                            choices=["pass", "passgolden", "subgraph", "subgraphgolden", "cmppass"],
                            help="checkmode")
        parser.add_argument("-path", "--outputpath", nargs="?", type=Path, default="",
                            help="Test cast json output path")
        parser.add_argument("-pass", "--passlist", nargs="+", type=str, default="",
                            help="Pass name list")
        parser.add_argument("-program", action="store_true", default=False,
                            help="Check the precision of the subgraph.")
        parser.add_argument("-goldendir", nargs="?", type=Path, default=None,
                            help="Test cast golden bin file dir")
        parser.add_argument("-g", "--golden", nargs="+", type=str, default="",
                            help="golden Tensor name")
        parser.add_argument("-dfx", nargs="?", type=str, default="",
                            choices=["dumpallop", "dumpcopyop", "printshape", "genreport"],
                            help="dfx option")
        # 流程处理
        ctrl = AstPrecisionToolCtrl(args=parser.parse_args())
        logging.info("%s", ctrl)
        ctrl.running()

    def running(self):
        # 清空output目录下的数据文件
        for root, dirs, files in os.walk(self.output_path):
            for file in files:
                if file.endswith(".json.py"):
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                if dir.endswith("_golden_data") or dir.endswith("_torch_data"):
                    shutil.rmtree(os.path.join(root, dir))
        if self.check_mode == "pass":
            self.passcheckwithoutgolden()
        elif self.check_mode == "passgolden":
            self.passcheckwithgolden()
        elif self.check_mode == "subgraph":
            self.programcheck()
        elif self.check_mode == "subgraphgolden":
            self.programcheck()
        elif self.check_mode == "cmppass":
            if (len(self.pass_list) != 2):
                print(f"The number of pass lists must be 2. current pass list :{self.pass_list}")
                return
            self.comparepass()

    def passcheckwithoutgolden(self):
        sorted_files = sorted(os.listdir(self.output_path))
        for dir_name in sorted_files:
            dir_path = os.path.join(self.output_path, dir_name)
            pass_name = dir_path.split('_')[-1]
            if not os.path.isdir(dir_path):
                continue
            if self.pass_list and pass_name not in self.pass_list:
                continue
            pt_file_list = []
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.json') and 'DEBUG' not in file_name:
                    print(f"Testing {file_path}")
                    start_time = time.time()

                    # 调用Python脚本处理JSON文件
                    cmd = ['python3', 'ast_json_to_torch.py', file_path, self.dump_print]
                    py_path = subprocess.check_output(cmd).decode().strip()
                    end_time = time.time()
                    cost_time = end_time - start_time
                    print(f"py_path: {py_path}, json-to-torch cost time: {cost_time} s")

                    # 创建目录并执行Python脚本
                    pt_path = f"{file_path.rsplit('.', 1)[0]}_torch_data"
                    os.makedirs(pt_path, exist_ok=True)

                    subprocess.run(['python3', py_path], check=True)
                    end_time2 = time.time()
                    cost_time2 = end_time2 - end_time
                    print(f"pt_path: {pt_path}, execute torch.py cost time: {cost_time2} s")

                    pt_file_list.append(pt_path)
            if len(pt_file_list) >= 2:
                before_pass_torch_data = pt_file_list[1]
                after_pass_torch_data = pt_file_list[0]
                start_time3 = time.time()

                # 比较数据
                cmd = ['python3', 'data_diff.py', before_pass_torch_data, after_pass_torch_data,
                       self.is_genreport]
                diff_res = subprocess.check_output(cmd).decode().strip()
                end_time3 = time.time()
                cost_time3 = end_time3 - start_time3
                print(f"Data diff cost time: {cost_time3} s.")

                if "ok" in diff_res:
                    print(f"{dir_path} ok")
                else:
                    print(diff_res)
                    print(f"{dir_path} error")

    def passcheckwithgolden(self):
        sorted_files = sorted(os.listdir(self.output_path))
        for dir_name in sorted_files:
            dir_path = os.path.join(self.output_path, dir_name)
            pass_name = dir_path.split('_')[-1]
            if not os.path.isdir(dir_path):
                continue
            if self.pass_list and pass_name not in self.pass_list:
                continue
            pt_file_list = []
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.json') and 'DEBUG' not in file_name:
                    print(f"Testing {file_path}")
                    start_time = time.time()

                    # 调用Python脚本处理JSON文件
                    cmd = ['python3', 'ast_json_to_torch.py', file_path, str(self.golden_dir)] + self.golden_bin_name
                    py_path = ''
                    try:
                        py_path = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
                        print('py_path ', py_path)
                    except subprocess.CalledProcessError as e:
                        print("Command failed with return code", e.returncode)
                        print("Output:", e.output.decode())
                    end_time = time.time()
                    cost_time = end_time - start_time
                    print(f"py_path: {py_path}, json-to-torch cost time: {cost_time} s")

                    # 创建目录并执行Python脚本
                    pt_path = f"{file_path.rsplit('.', 1)[0]}_torch_data"
                    print('test pt_path ', pt_path)
                    os.makedirs(pt_path, exist_ok=True)
                    golden_path = f"{file_path.rsplit('.', 1)[0]}_golden_data"
                    print('test golden_path ', golden_path)
                    os.makedirs(golden_path, exist_ok=True)

                    subprocess.run(['python3', py_path], check=True)
                    end_time2 = time.time()
                    cost_time2 = end_time2 - end_time
                    print(f"pt_path: {pt_path}, execute torch.py cost time: {cost_time2} s")

                    pt_file_list.append(pt_path)

                    diff_res_cmd = ["python3", "data_diff.py", golden_path, pt_path, self.is_genreport]
                    diff_res = subprocess.check_output(diff_res_cmd).decode().strip()
                    if "ok" in diff_res:
                        print(f"{dir_path} ok")
                    else:
                        print(f"{dir_path} error")
                        print(diff_res)

            if len(pt_file_list) >= 2:
                before_pass_torch_data = pt_file_list[1]
                after_pass_torch_data = pt_file_list[0]
                start_time3 = time.time()
                diff_res = ""
                print(file_name)
                if "SubgraphToFunction" in file_name:
                    cmd = ['python3', 'data_diff.py', after_pass_torch_data, self.dump_tensor_pt_path,
                           self.is_genreport]
                    diff_res = subprocess.check_output(cmd).decode().strip()
                    print(f"check SubgraphToFunction with dump_tensor result:")
                    print(diff_res)

                # 比较数据
                cmd = ['python3', 'data_diff.py', before_pass_torch_data, after_pass_torch_data,
                       self.is_genreport]
                diff_res = subprocess.check_output(cmd).decode().strip()
                end_time3 = time.time()
                cost_time3 = end_time3 - start_time3
                print(f"Data diff cost time: {cost_time3} s.")

                if "ok" in diff_res:
                    print(f"{dir_path} ok")
                else:
                    print(diff_res)
                    print(f"{dir_path} error")

    def programcheck(self):
        start_time4 = time.time()
        file_path = ''
        cmd = []
        for file_name in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file_name)
            if os.path.isfile(file_path) and file_name == "program.json":
                break
        if file_path == '':
            print(f"The program.json file does not exist.")
            return
        if self.check_mode == "subgraph":
            cmd = ['python3', 'ast_json_to_torch.py', file_path]
        elif self.check_mode == "subgraphgolden":
            cmd = ['python3', 'ast_json_to_torch.py', file_path, str(self.golden_dir)] + self.golden_bin_name
        py_path = subprocess.check_output(cmd).decode().strip()
        end_time4 = time.time()
        cost_time4 = end_time4 - start_time4
        print(f"py_path: {py_path}, json-to-torch cost time: {cost_time4} s")

        file_name, _ = os.path.splitext(file_path)
        pt_path = f"{file_name}_torch_data"
        golden_path = f"{file_name}_golden_data"
        os.makedirs(pt_path, exist_ok=True)
        os.makedirs(golden_path, exist_ok=True)

        subprocess.run(['python3', py_path], check=True)
        end_time5 = time.time()
        cost_time5 = end_time5 - start_time4
        print(f"pt_path: {pt_path}, execute torch.py cost time: {cost_time5} s")

        start_time6 = time.time()
        cmd_diff = ['python3', 'data_diff.py', pt_path, self.dump_tensor_pt_path, self.is_genreport]
        diff_res = subprocess.check_output(cmd_diff).decode().strip()
        end_time6 = time.time()
        cost_time6 = end_time6 - start_time6
        print(f"Data diff cost time: {cost_time6} s")

        if "ok" in diff_res:
            print(f"{file_path} ok")
        else:
            print(f"{file_path} error")
            print(diff_res)

    def comparepass(self):
        sorted_files = sorted(os.listdir(self.output_path))
        check_list = []
        pt_file_list = []
        for dir_name in sorted_files:
            dir_path = os.path.join(self.output_path, dir_name)
            pass_name = dir_path.split('_')[-1]
            if not os.path.isdir(dir_path):
                continue
            if self.pass_list and pass_name not in self.pass_list:
                continue
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if not (os.path.isfile(file_path) and file_name.endswith('.json') and 'DEBUG' not in file_name):
                    continue
                should_check: bool = False
                if 'Before' in file_name and len(check_list) == 0:
                    should_check = True
                if 'After' in file_name and len(check_list) == 1:
                    should_check = True
                if should_check == True:
                    check_list.append(file_path)
                    start_time = time.time()

                    # 调用Python脚本处理JSON文件
                    cmd = ['python3', 'ast_json_to_torch.py', file_path, self.dump_print]
                    py_path = subprocess.check_output(cmd).decode().strip()
                    end_time = time.time()
                    cost_time = end_time - start_time
                    print(f"py_path: {py_path}, json-to-torch cost time: {cost_time} s")

                    # 创建目录并执行Python脚本
                    pt_path = f"{file_path.rsplit('.', 1)[0]}_torch_data"
                    os.makedirs(pt_path, exist_ok=True)

                    subprocess.run(['python3', py_path], check=True)
                    end_time2 = time.time()
                    cost_time2 = end_time2 - end_time
                    print(f"pt_path: {pt_path}, execute torch.py cost time: {cost_time2} s")

                    pt_file_list.append(pt_path)
                    break
        if len(pt_file_list) >= 2:
            before_pass_torch_data = pt_file_list[0]
            after_pass_torch_data = pt_file_list[1]
            start_time3 = time.time()

            # 比较数据
            cmd = ['python3', 'data_diff.py', before_pass_torch_data, after_pass_torch_data,
                   self.is_genreport]
            diff_res = subprocess.check_output(cmd).decode().strip()
            end_time3 = time.time()
            cost_time3 = end_time3 - start_time3

            check_pass1 = check_list[0].split('_')[-1].split('.')[0]
            check_pass2 = check_list[1].split('_')[-1].split('.')[0]
            print(f"Data diff cost time: {cost_time3} s.")
            print(f"Comparison Data is :")
            print(f"           {check_list[0]}")
            print(f"           {check_list[1]}")
            if "ok" in diff_res:
                print(f"The comparison between {check_pass1} and {check_pass2} is ok")
            else:
                print(diff_res)
                print(f"The comparison between {check_pass1} and {check_pass2} is error")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    AstPrecisionToolCtrl.main()
