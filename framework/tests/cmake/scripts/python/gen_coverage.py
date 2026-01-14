#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""生成覆盖率
"""
import argparse
import logging
import math
import os
import re
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class GenCoverage:

    class FilterPathAction(argparse.Action):
        """自定义 Action: 解析 filter 参数时校验路径并格式化"""
        def __call__(self, parser, namespace, values, option_string=None):
            # 获取当前已收集的列表(初始为 None)
            cur_values = getattr(namespace, self.dest, None) or []
            path = Path(values)
            # 仅保留存在的路径，并格式化(目录加 /*)
            if path.exists():
                if path.is_dir():
                    cur_values.append(f" {path}/*")
                else:
                    cur_values.append(f" {path}")
            # 更新命名空间的值
            setattr(namespace, self.dest, cur_values)

    def __init__(self, args):
        self.src_root: Optional[Path] = Path(args.source[0]).resolve() if args.source else None
        self.data_dir: Path = Path(args.data[0]).resolve()
        self.info_file = Path(args.info[0]).resolve() if args.info else Path(self.data_dir, 'cov_result/coverage.info')
        self.filter_str: str = " ".join(args.filter)
        self.html_report = Path(args.html[0]).resolve() if args.html else Path(self.info_file.parent, "html_report")
        self.job_num: int = self.get_job_num(args=args)
        # 合法性检查
        self.lcov_version: str = ""
        self.lcov_support_job_num: bool = False
        self.chk_env()
        if not self.data_dir.exists():
            raise ValueError(f"The dir({self.data_dir}) required to find the .da files not exist.")
        self.info_file.parent.mkdir(parents=True, exist_ok=True)
        self.html_report.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        desc = f"\nGenerateCoverage"
        desc += f"\n    SrcRoot    : {self.src_root}"
        desc += f"\n    DataDir    : {self.data_dir}"
        desc += f"\n    InfoFile   : {self.info_file}"
        desc += f"\n    Filter     : {self.filter_str}"
        desc += f"\n    HtmlReport : {self.html_report}"
        desc += f"\n    JobNum     : {self.job_num}"
        desc += f"\n    lcov       : {self.lcov_version} ({self.lcov_support_job_num})"
        desc += f"\n"
        return desc

    @classmethod
    def reg_args(cls, parser):
        parser.add_argument("-s", "--source",
                            required=False, nargs=1, type=Path,
                            help="Specify the source base directory.")
        parser.add_argument("-d", "--data",
                            required=True, nargs=1, type=Path,
                            help="Specify the *.da's base directory.")
        parser.add_argument("-i", "--info_file", dest="info",
                            required=False, nargs=1, type=Path,
                            help="Specify coverage info file path.")
        parser.add_argument("-f", "--filter",
                            required=False, action=cls.FilterPathAction, type=str,
                            help="Specify filter file/dir in coverage info.")
        parser.add_argument("--html_report", dest="html",
                            required=False, nargs=1, type=Path,
                            help="Specify coverage html report dir.")
        parser.add_argument("-j", "--job_num",
                            nargs="?", type=int, default=None,
                            help="Specify parallel job num.")

    @classmethod
    def get_job_num(cls, args):
        if args.job_num:
            job_num = args.job_num
        else:
            if os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", 0):
                job_num = int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"), 0)
            elif os.environ.get("PYPTO_UTEST_PARALLEL_NUM", 0):
                job_num = int(os.environ.get("PYPTO_UTEST_PARALLEL_NUM", 0))
            else:
                job_num = int(math.ceil(float(cpu_count()) * 0.8))    # use 0.8 cpu
        job_num = min(max(int(job_num), 1), cpu_count(), 48)
        return job_num

    @classmethod
    def main(cls):
        # 参数注册
        parser = argparse.ArgumentParser(description="Generate Coverage", epilog="Best Regards!")
        cls.reg_args(parser=parser)
        # 参数处理
        ctrl = GenCoverage(args=parser.parse_args())
        logging.info("%s", ctrl)
        # 流程处理
        ctrl.process()

    def chk_env(self):
        try:
            ret = subprocess.run('lcov --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
            # 提取版本号
            version_output = ret.stdout.strip()
            version_match = re.search(r'version (\d+\.\d+)', version_output)
            if not version_match:
                raise RuntimeError(f"Can't get version from {version_output}")
            self.lcov_version = version_match.group(1)
            self.lcov_support_job_num = False
            # 验证版本是否 ≥2.1(支持 -j 参数)
            version_parts = list(map(int, self.lcov_version.split('.')))
            if version_parts[0] > 2 or (version_parts[0] == 2 and version_parts[1] >= 1):
                self.lcov_support_job_num = True
        except FileNotFoundError as e:
            raise FileNotFoundError(f"lcov is required to generate coverage data, please install.") from e
        try:
            ret = subprocess.run('genhtml --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"genhtml is required to generate coverage html report, please install.") from e

    def process(self):
        """使用 lcov 生成覆盖率
        """
        # 生成覆盖率原始统计文件
        cmd = f"lcov -c -d {self.data_dir} -o {self.info_file}"
        cmd += f" -j {self.job_num}" if self.lcov_support_job_num else ""
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated origin coverage file %s, cmd: %s", self.info_file, cmd)
        # 滤掉某些文件/路径的覆盖率信息
        info_file_filtered = Path(self.info_file.parent, f"{self.info_file.stem}_filtered{self.info_file.suffix}")
        cmd = f"lcov --remove {self.info_file} {self.filter_str} -o {info_file_filtered}"
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated filtered coverage file %s, cmd: %s", info_file_filtered, cmd)
        # 生成 html 报告
        prefix = f"-p {self.src_root}" if self.src_root else ""
        cmd = f'genhtml {info_file_filtered} {prefix} -o {self.html_report}'
        cmd += f" -j {self.job_num}" if self.lcov_support_job_num else ""
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated filtered coverage html report in %s, cmd: %s", self.html_report, cmd)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    ts = datetime.now(tz=timezone.utc)
    GenCoverage.main()
    duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
    logging.info("Generate Coverage use %s secs.", duration)
