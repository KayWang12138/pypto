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
"""STest执行加速
"""
import argparse
import logging
import os
import queue
import shlex
import signal
import subprocess
import time

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Any, Optional, Tuple, NoReturn, Dict
from multiprocessing import JoinableQueue, Lock, Event, Process, Value

from tabulate import tabulate


class STestAccelerate:
    """STest 加速

    支持多 Device 并行执行用例, 以提升 STest 在多 Device 环境的执行效率.
    """

    def __init__(self, args):
        # 用例执行参数
        self.target_file: Path = Path(args.target[0]).resolve()
        self.target_name: str = self.target_file.name
        self.working_dir: Path = self.target_file.parent
        self.xsan_options: str = args.xsan_options if args.xsan_options else ""
        self.timeout: Optional[int] = args.timeout

        # 用例管理
        self.case_list: List[str] = str(args.cases).split(":")
        self.case_queue: JoinableQueue = JoinableQueue()
        self.case_exception_queue: JoinableQueue = JoinableQueue()  # Case 执行失败时, 用于收集错误信息
        self.case_terminate_queue: JoinableQueue = JoinableQueue()  # Case 被终止执行时, 收集相关信息
        self.case_execution_queue: JoinableQueue = JoinableQueue()  # Case 正常执行结束时，收集相关信息
        self.case_execution_timeout: Optional[int] = None  # 单 Case 执行 Timeout

        # 执行控制(多 Device)
        self.halt_on_error: bool = args.halt_on_error  # 失败时终止后续 Case 执行
        self.device_list: List[int] = []
        self.device_brief_queue: JoinableQueue = JoinableQueue()  # Device 执行结果统计上报
        self.device_terminate_event: Event = Event()  # 用于通知其他 Device 进程结束运行
        self.init_device(args=args)

        # DFX
        self.dfx_output_lock: Lock = Lock()  # 执行输出日志保序控制
        self.dfx_case_finish_count: Value = Value('i', 0)  # DFX, 统计 Case 完成进度
        self.dfx_process_duration: Optional[timedelta] = None
        self.dfx_device_duration_dict: Dict[int, timedelta] = {}
        self.dfx_device_exit_count: Value = Value('i', 0)  # DFX, 统计 Device 退出进度

        logging.info("\n\nSTest Accelerate Args:\n%s", str(tabulate(self.brief, tablefmt="simple")))

    @property
    def asan_option(self) -> str:
        return "ON" if "ASAN_OPTIONS" in self.xsan_options else "OFF"

    @property
    def ubsan_option(self) -> str:
        return "ON" if "UBSAN_OPTIONS" in self.xsan_options else "OFF"

    @property
    def brief(self) -> List[Any]:
        return [["Target", self.target_file],
                ["ASan", self.asan_option],
                ["UbSan", self.ubsan_option],
                ["CaseNum", len(self.case_list)],
                ["Devices", self.device_list],
                ["Timeout", self.timeout],
                ["HaltOnError", self.halt_on_error]]

    @classmethod
    def main(cls) -> bool:
        """主处理流程
        """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"STest Execute Accelerate", epilog="Best Regards!")
        parser.add_argument("-t", "--target", nargs=1, type=str, required=True,
                            help="Specific target executable file path.")
        parser.add_argument("-c", "--cases", type=str, default="", required=True,
                            help="Test Cases, multiple test cases are separated by ':'")
        parser.add_argument("--xsan_options", nargs="?", type=str, default="",
                            help="Specific XSan(ASan/UbSan) option.")
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Specific parallel accelerate device, "
                                 "If this parameter is not specified, 0 device will be used by default.")
        parser.add_argument("--timeout", nargs="?", type=int, default=None,
                            help="Task execute timeout.")
        parser.add_argument("--halt_on_error", action="store_true", default=False,
                            help="If any case failed, subsequent cases are not executed.")
        # 流程处理
        ctrl = STestAccelerate(args=parser.parse_args())
        ret: bool = ctrl.process()
        ret = ret and ctrl.post()
        return ret

    def init_device(self, args):
        self.device_list = [0]
        if args.device is not None:
            self.device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]

    def process(self) -> bool:
        """用例执行, 管理执行状态(主进程)

        :return: 执行成功与否
        """
        ts = datetime.now(tz=timezone.utc)

        # 任务准备
        # 以同步方式将待执行用例插入待执行队列, 按 Device 数量插入终止信号
        for cs in self.case_list:
            self.case_queue.put(cs)
        for _ in range(len(self.device_list)):
            self.case_queue.put(None)

        # 创建并启动子进程, 进行任务处理
        processes = []
        try:
            # 创建并启动子进程, 进行任务处理
            for device_id in self.device_list:
                p = Process(name=f"Device[{device_id}]", target=self.device_process, args=(device_id,), daemon=False)
                processes.append(p)
                p.start()
            # 等待处理结束
            if not self.timeout:
                # 未设置 timeout, 等待所有子进程结束
                for p in processes:
                    p.join()
            else:
                # 设置 timeout, 监控并适时停止子进程
                s_time = time.time()
                for p in processes:
                    r_time = int(max(self.timeout - (time.time() - s_time), 1))
                    p.join(timeout=r_time)
                self.device_terminate_event.set()  # 停止所有子进程对新任务的处理
            # 向子进程发送终止信号
            for p in processes:
                if not p.is_alive():
                    continue
                os.kill(p.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
                p.join(timeout=1)  # 等待子进程退出
        except KeyboardInterrupt:
            self.device_terminate_event.set()  # 停止所有子进程对新任务的处理
            for p in processes:
                if p.is_alive():
                    p.join(timeout=1)

        self.dfx_process_duration = datetime.now(tz=timezone.utc) - ts
        return True

    def post(self) -> bool:
        """后处理, 主要用于收集执行信息.

        :return: bool 执行成功与否
        """
        # Job 执行信息收集汇总
        job_exec_brief, job_revenue_desc = self._post_device_exec_info()

        # Case 执行信息收集汇总
        case_exec_brief, case_exec_result = self._post_case_exec_info()

        out: str = f"STest, HaltOnError({self.halt_on_error}), {job_revenue_desc}"
        out += job_exec_brief
        out += case_exec_brief

        if not case_exec_result:
            logging.error("%s", out)
        else:
            logging.info("%s", out)
        return case_exec_result

    def device_process(self, device_id: int):
        """Device 进程

        1. Device 进程执行时, 不会产生 Exception, 用例执行异常信息会上报至异常信息队列;
        2. Device 进程在任务队列为空, 或异常终止事件被设置时退出;

        :param device_id: DeviceId
        """
        pass_cnt: int = 0
        fail_cnt: int = 0
        ts = datetime.now(tz=timezone.utc)
        cs_proc: Optional[Process] = None

        try:
            time.sleep(5)  # 多消费者模式, 各消费者启动时增加一定延迟, 等待所有消费者启动完成
            while not self.device_terminate_event.is_set():
                # 用例获取
                gtest_filter = self.case_queue.get()
                self.case_queue.task_done()
                if gtest_filter is None:
                    break  # 终止信号, 正常退出
                # 用例执行
                cs_proc = Process(name=f"Device[{device_id}] Case[{gtest_filter}]",
                                  target=self.case_process, args=(device_id, gtest_filter,))
                cs_proc.start()
                cs_proc.join()
                if cs_proc.exitcode != 0:
                    fail_cnt += 1
                else:
                    pass_cnt += 1
        except queue.Empty:
            pass  # 队列为空, 正常退出
        except KeyboardInterrupt:
            # 强制终止时, 杀停子进程, 上报处理进度
            if cs_proc and cs_proc.is_alive():
                os.kill(cs_proc.pid, signal.SIGINT)
            logging.info("Device[%s] receive terminate event.", device_id)
        finally:
            # Device 执行结果统计与上报
            brief: List[Any] = [device_id, pass_cnt + fail_cnt, pass_cnt, fail_cnt,
                                (datetime.now(tz=timezone.utc) - ts)]
            self.device_brief_queue.put(brief)
            with self.dfx_output_lock:
                logging.info("Device[%s] Exist %s %s",
                             device_id, self.dfx_device_progress(update=True), self.dfx_case_progress(update=False))

    def case_process(self, device_id: int, gtest_filter: Optional[str]) -> NoReturn:
        """具体用例执行进程

        通过子进程实现各 Case 执行上下文隔离, 避免 Case 间相互影响

        :param device_id: DeviceId
        :param gtest_filter: GTestFilter
        :exception RuntimeError: 本用例执行失败时, 抛出该类型异常
        """
        ts = datetime.now(tz=timezone.utc)
        try:
            cmd: str = f"{self.xsan_options} ./{self.target_name} "
            if gtest_filter:
                cmd += f" --gtest_filter={gtest_filter}"
            act_env = {**os.environ}
            act_env.update({"TILE_FWK_STEST_DEVICE_ID": f"{device_id}"})
            # 通过全局锁, 来确保各用例输出结果在的输出顺序
            with self.dfx_output_lock:
                logging.info("Device[%s] [BGN] Run GTest(%s) XSAN(ASAN:%s UBSAN:%s) GTestFilter(%s)",
                             device_id, self.target_name, self.asan_option, self.ubsan_option, gtest_filter)
            ret = subprocess.run(shlex.split(cmd), env=act_env, cwd=self.working_dir,
                                 capture_output=True, check=False, text=True, encoding='utf-8')
            if ret.returncode:
                msg = f"stdout:\n{ret.stdout}\n{ret.stderr}"
                err = RuntimeError(f"Cmd:{cmd}\nDevice:{device_id}\nDetails:\n{msg}")
                # 收集错误现场信息并上报
                self.case_exception_queue.put(err)
                # 异常后处理
                if self.halt_on_error:
                    self.device_terminate_event.set()
                raise err  # 触发 Device 执行进程感知 Case 执行异常
            msg = ret.stdout or ret.stderr
            # 通过全局锁, 来确保各用例输出结果在的输出顺序
            with self.dfx_output_lock:
                logging.info("Device[%s] [END] Run GTest(%s) XSAN(ASAN:%s UBSAN:%s) GTestFilter(%s) %s "
                             "Output Below:\n%s",
                             device_id, self.target_name, self.asan_option, self.ubsan_option, gtest_filter,
                             self.dfx_case_progress(update=True), msg)
                time_brief: List[Any] = [device_id, gtest_filter, (datetime.now(tz=timezone.utc) - ts)]
                self.case_execution_queue.put(time_brief)

        except KeyboardInterrupt:
            # 强制终止时, 主动退出执行, 上报已运行时长
            logging.info("Device[%s] Case[%s] receive terminate event.", device_id, gtest_filter)
            brief: List[Any] = [device_id, gtest_filter, (datetime.now(tz=timezone.utc) - ts)]
            self.case_terminate_queue.put(brief)

    def dfx_device_progress(self, update=True) -> str:
        """获取 Device 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            self.dfx_device_exit_count.value += 1
        cnt: int = int(self.dfx_device_exit_count.value)
        pgs: float = cnt / len(self.device_list) * 100
        return f"DeviceProgress[{cnt}/{len(self.device_list)} {pgs:.2f}%]"

    def dfx_case_progress(self, update=True) -> str:
        """获取 Case 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            self.dfx_case_finish_count.value += 1
        cnt: int = int(self.dfx_case_finish_count.value)
        pgs: float = cnt / len(self.case_list) * 100
        return f"CaseProgress[{cnt}/{len(self.case_list)} {pgs:.2f}%]"

    def _post_device_exec_info(self) -> Tuple[str, str]:
        """获取 Device 执行信息统计.

        :returns:
            Tuple[str, timedelta]:
                - Device 并行执行收益描述(str)
                - Device 执行信息统计表(str)
        """
        heads: List[str] = ["DeviceId", "Total", "Success", "Failed", "Duration"]
        datas: List[Any] = []
        duration_sum: timedelta = timedelta()
        while not self.device_brief_queue.empty():
            _brief = self.device_brief_queue.get()
            devs_id: int = int(_brief[0])
            case_total: int = int(_brief[1])
            case_pass: int = int(_brief[2])
            case_fail: int = int(_brief[3])
            devs_duration: timedelta = _brief[-1]
            self.dfx_device_duration_dict[devs_id] = devs_duration
            duration_sum += devs_duration
            datas.append([devs_id, case_total, case_pass, case_fail, f"{devs_duration.total_seconds():.2f}"])
            self.device_brief_queue.task_done()
        brief: str = " None"
        if len(datas) != 0:
            brief = str(tabulate(datas, headers=heads, tablefmt='grid'))
        # 并行执行收益计算
        rate: float = float((duration_sum - self.dfx_process_duration) / self.dfx_process_duration) * 100
        desc: str = f"Duration {self.dfx_process_duration.total_seconds():.2f} secs, Revenue(Act/Ori, "
        desc += f"{self.dfx_process_duration.total_seconds():.2f}/{duration_sum.total_seconds():.2f}) {rate:.2f}%"
        return f"\n\nDevice Execution Brief:{brief}", desc

    def _post_case_exec_info(self) -> Tuple[str, bool]:
        """获取 Case 执行信息.

        :returns:
            Tuple[str, bool]:
                - Case 执行情况信息
                - Case 执行成功与否判定结果
        """
        case_terminate_brief, case_terminate_count = self._post_case_exec_terminate_info()
        case_exception_brief, case_exception_count = self._post_case_exec_exception_info()
        case_duration_brief = self._post_case_exec_duration_info()

        # Case 执行总体情况汇总
        case_remaining_count: int = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                case_remaining_count += 1
            self.case_queue.task_done()
        case_execution_heads = ["Total", "Executed", "Terminate", "Exception", "Remaining"]
        case_execution_datas = [[len(self.case_list), len(self.case_list) - case_remaining_count,
                                 case_terminate_count, case_exception_count, case_remaining_count]]
        case_execution_brief = str(tabulate(case_execution_datas, headers=case_execution_heads, tablefmt='grid'))
        case_execution_brief = f"\n\nCase Execution Brief:{case_execution_brief}"

        rst: bool = (case_terminate_count + case_exception_count + case_remaining_count) == 0
        out: str = case_execution_brief + case_duration_brief + case_terminate_brief + case_exception_brief
        return out, rst

    def _post_case_exec_terminate_info(self) -> Tuple[str, int]:
        """获取 Case 执行终止信息.

        :returns:
            Tuple[str, int]:
                - Case 终止执行情况信息
                - Case 终止执行数量
        """
        heads: List[str] = ["Idx", "Job", "CaseName", "Duration"]
        datas: List[Any] = []
        while not self.case_terminate_queue.empty():
            _brief = self.case_terminate_queue.get()
            job_id: int = int(_brief[0])
            case_name: str = str(_brief[1])
            case_duration: timedelta = _brief[2]
            datas.append([job_id, case_name, f"{case_duration.total_seconds():.2f}"])
            self.case_terminate_queue.task_done()
        brief: str = " None"
        if len(datas) != 0:
            datas = [[f"{idx}/{len(datas)}"] + ele for idx, ele in enumerate(datas)]
            brief = str(tabulate(datas, headers=heads, tablefmt='grid'))
        return f"\n\nCase Terminate Brief({len(datas)}):{brief}", len(datas)

    def _post_case_exec_exception_info(self) -> Tuple[str, int]:
        """获取 Case 执行异常信息.

        :returns:
            Tuple[str, int]:
                - Case 异常执行情况信息
                - Case 异常执行数量
        """
        datas: List[str] = []
        while not self.case_exception_queue.empty():
            _brief = self.case_exception_queue.get()
            datas.append(str(_brief))
            self.case_exception_queue.task_done()
        brief = " None" if len(datas) == 0 else ""
        for idx, data in enumerate(datas, start=1):
            brief += f"\nIdx:{idx}/{len(datas)}\n{data}"
        return f"\n\nCase Exception Brief:{brief}", len(datas)

    def _post_case_exec_duration_info(self) -> str:
        """获取 Case 执行耗时统计信息.

        :return: Case 执行耗时统计信息.
        """
        heads: List[str] = ["Job", "CaseName", "Duration", "Ratio(Job)", "Ratio(Process)"]
        datas: List[Any] = []
        while not self.case_execution_queue.empty():
            _brief = self.case_execution_queue.get()
            job_idx: int = _brief[0]
            case_name: str = str(_brief[1])
            case_duration: timedelta = _brief[2]
            job_duration: timedelta = self.dfx_device_duration_dict[job_idx]
            ratio_job: float = float(case_duration / job_duration) * 100
            ratio_process: float = float(case_duration / self.dfx_process_duration) * 100
            datas.append(
                [job_idx, case_name, f"{case_duration.total_seconds():.2f}",
                 f"{case_duration.total_seconds():.2f}/{job_duration.total_seconds():.2f} {ratio_job:.2f}%",
                 f"{case_duration.total_seconds():.2f}/{self.dfx_process_duration.total_seconds():.2f} "
                 f"{ratio_process:.2f}%"])
            self.case_execution_queue.task_done()
        brief: str = " None"
        if len(datas) != 0:
            brief = str(tabulate(datas, headers=heads, tablefmt='grid'))
        return f"\n\nCase Duration Brief:{brief}"


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if STestAccelerate.main() else 1)
