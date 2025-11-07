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
""" """
import logging
import os
import signal
import subprocess


class TestCaseShellActuator:
    @classmethod
    def run(cls, cmd):
        logging.info(f"Start exec : {cmd}")
        with subprocess.Popen(
            cmd,
            env={**os.environ},
            text=True,
            encoding="utf-8",
            start_new_session=True,
            shell=True,
        ) as process:
            try:
                stdout, stderr = process.communicate()
            except KeyboardInterrupt:
                _pgid = os.getpgid(process.pid)
                os.killpg(_pgid, signal.SIGINT)
            except Exception:
                process.kill()
                raise
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if ret_code:
                raise subprocess.CalledProcessError(
                    ret_code, process.args, output=stdout, stderr=stderr
                )
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr)
