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
"""GTest用例并行执行.
"""
import argparse
import logging
import os
import queue
import shlex
import signal
import subprocess
import time
import math

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Any, Optional, Tuple, NoReturn
from multiprocessing import JoinableQueue, Lock, Event, Process, Value, cpu_count


class UTestAccelerate:
    """
    UTest 加速

    通过多进程并行执行, 以提升 UTest 执行效率.
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
        self.case_fail_error_queue: JoinableQueue = JoinableQueue()  # Case 执行失败时, 用于收集错误信息
        self.case_terminate_queue: JoinableQueue = JoinableQueue()  # Case 被终止执行时, 收集相关信息
        self.case_exec_queue: JoinableQueue = JoinableQueue() # Case 正常执行结束时，收集相关信息
        self.case_exec_timeout: Optional[int] = None  # 单 Case 执行 Timeout

        # 执行控制(多进程)
        self.halt_on_error: bool = args.halt_on_error  # 失败时终止后续 Case 执行
        self.job_num: int = args.job_num if args.job_num else int(math.ceil(float(cpu_count()) * 0.8))  # use 0.8 cpu
        self.job_num: int = min(min(min(max(self.job_num, 1), cpu_count()), 64), len(self.case_list))
        self.job_brief_queue: JoinableQueue = JoinableQueue()  # Job 执行结果统计上报
        self.job_terminate_event: Event = Event()  # 用于通知其他 Device 进程结束运行

        # DFX
        self.process_timedelta: Optional[timedelta] = None
        self.dfx_output_lock: Lock = Lock()  # 执行输出日志保序控制
        self.dfx_case_finish_cnt: Value = Value('i', 0)  # DFX, 统计 Case 完成进度
        self.dfx_job_exit_cnt: Value = Value('i', 0)  # DFX, 统计 Job 退出进度

        logging.info("\n\nUTest Accelerate Args:%s", self.brief)

    @property
    def asan_option(self) -> str:
        return "ON" if "ASAN_OPTIONS" in self.xsan_options else "OFF"

    @property
    def ubsan_option(self) -> str:
        return "ON" if "UBSAN_OPTIONS" in self.xsan_options else "OFF"

    @property
    def brief(self) -> str:
        out: str = ""
        out += f"\n\tTarget      : {self.target_file}"
        out += f"\n\tASan        : {self.asan_option}"
        out += f"\n\tUbSan       : {self.ubsan_option}"
        out += f"\n\tCaseNum     : {len(self.case_list)}"
        out += f"\n\tJobNum      : {self.job_num}"
        out += f"\n\tTimeout     : {self.timeout}"
        out += f"\n\tHaltOnError : {self.halt_on_error}"
        return out

    @classmethod
    def main(cls) -> bool:
        """ 主处理流程 """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"UTest Execute Accelerate", epilog="Best Regards!")
        parser.add_argument("-t", "--target", nargs=1, type=str, required=True,
                            help="Specific target executable file path.")
        parser.add_argument("-c", "--cases", type=str, default="", required=True,
                            help="Test Cases, multiple test cases are separated by ':'")
        parser.add_argument("--xsan_options", nargs="?", type=str, default="",
                            help="Specific XSan(ASan/UbSan) option.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=None,
                            help="Specific parallel accelerate job num.")
        parser.add_argument("--timeout", nargs="?", type=int, default=None,
                            help="Task execute timeout.")
        parser.add_argument("--halt_on_error", action="store_true", default=False,
                            help="If any case failed, subsequent cases are not executed.")
        # 流程处理
        ctrl = UTestAccelerate(args=parser.parse_args())
        ret: bool = ctrl.process()
        ret = ret and ctrl.post()
        return ret

    def process(self) -> bool:
        """
        用例执行, 管理执行状态(主进程)
        """
        ts = datetime.now(tz=timezone.utc)

        # 任务准备
        # 以同步方式将待执行用例插入待执行队列, 按 Job 数量插入终止信号
        for cs in self.case_list:
            self.case_queue.put(cs)
        for _ in range(self.job_num):
            self.case_queue.put(None)

        # 创建并启动子进程, 进行任务处理
        processes = []
        try:
            # 创建并启动子进程, 进行任务处理
            for job_idx in range(self.job_num):
                p = Process(name=f"Job[{job_idx}]", target=self.job_process, args=(job_idx,), daemon=False)
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
                self.job_terminate_event.set()  # 停止所有子进程对新任务的处理
            # 向子进程发送终止信号
            for p in processes:
                if not p.is_alive():
                    continue
                os.kill(p.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
                p.join(timeout=1)  # 等待子进程退出
        except KeyboardInterrupt:
            self.job_terminate_event.set()  # 停止所有子进程对新任务的处理
            for p in processes:
                if p.is_alive():
                    p.join(timeout=1)
        finally:
            self.process_timedelta = datetime.now(tz=timezone.utc) - ts
        return True

    def post(self) -> bool:
        # Job 执行信息收集汇总
        job_exec_str, jobs_exec_ori_secs = self._post_job_exec_info()

        # Case 中断执行信息汇总
        case_terminate_str, case_terminate_cnt = self._post_case_terminate_info()

        # Case 执行信息收集汇总
        case_remaining_cnt = self._post_case_exec_info()
        case_exec_str: str = (f"\n\tTotal({len(self.case_list)}) "
                              f"Executed({len(self.case_list) - case_remaining_cnt}) "
                              f"Remaining({case_remaining_cnt}) Terminate({case_terminate_cnt})")

        # Case 运行时长收集汇总
        case_exectime_str = self._post_case_time_info()

        # Case 异常信息收集汇总
        case_fail_brief, case_fail_datas_len = self._post_case_fail_info()

        cost_ori: timedelta = jobs_exec_ori_secs
        cost_rate: float = float((cost_ori - self.process_timedelta) / self.process_timedelta) * 100

        out: str = f"STest, HaltOnError({self.halt_on_error}), Cost {self.process_timedelta.seconds} secs, "
        out += f"Revenue(Act/Ori/Rate, {self.process_timedelta.seconds}/{cost_ori.seconds}/{cost_rate:.2f}%)"
        out += f"\nJob Execution:{job_exec_str}"
        out += f"\nCase Terminate Brief({case_terminate_cnt}):{case_terminate_str}"
        out += f"\nCase Execution Brief:{case_exec_str}"
        out += f"\nCase Exception Brief({case_fail_datas_len}):\n{case_fail_brief}"
        out += f"\nCase Execution Brief:{case_exectime_str}"

        ret: bool = case_remaining_cnt == 0 and case_fail_datas_len == 0
        if not ret:
            logging.error("%s", out)
        else:
            logging.info("%s", out)
        if self.timeout and self.process_timedelta.seconds > self.timeout:
            raise TimeoutError(f"Timeout, Act({self.process_timedelta.seconds}) > Exp({self.timeout}) secs.]")
        return ret

    def job_process(self, job_idx: int):
        """
        Job 进程

        1. Job 进程执行时, 不会产生 Exception, 用例执行异常信息会上报至异常信息队列;
        2. Job 进程在任务队列为空, 或异常终止事件被设置时退出;

        :param job_idx: JobIdx
        """
        succ_cnt: int = 0
        fail_cnt: int = 0
        ts = datetime.now(tz=timezone.utc)
        cs_proc: Optional[Process] = None

        try:
            time.sleep(5)  # 多消费者模式, 各消费者启动时增加一定延迟, 等待所有消费者启动完成
            while not self.job_terminate_event.is_set():
                # 用例获取
                gtest_filter = self.case_queue.get()
                self.case_queue.task_done()
                if gtest_filter is None:
                    break  # 终止信号, 正常退出
                # 用例执行
                cs_proc = Process(name=f"Job[{job_idx}] Case[{gtest_filter}]",
                                  target=self.case_process, args=(job_idx, gtest_filter,))
                cs_proc.start()
                cs_proc.join()
                if cs_proc.exitcode != 0:
                    fail_cnt += 1
                else:
                    succ_cnt += 1
        except queue.Empty:
            pass  # 队列为空, 正常退出
        except KeyboardInterrupt:
            # 强制终止时, 上报处理进度
            if cs_proc and cs_proc.is_alive():
                os.kill(cs_proc.pid, signal.SIGINT)
            logging.info("Job[%s] receive terminate event.", job_idx)
        finally:
            # Job 执行结果统计与上报
            brief: List[Any] = [job_idx, succ_cnt + fail_cnt, succ_cnt, fail_cnt,
                                (datetime.now(tz=timezone.utc) - ts)]
            self.job_brief_queue.put(brief)
            with self.dfx_output_lock:
                logging.info("Job[%s] Exist %s %s",
                             job_idx, self.dfx_job_progress(update=True), self.dfx_case_progress(update=False))

    def case_process(self, job_idx: int, gtest_filter: Optional[str]) -> NoReturn:
        """
        具体用例执行进程

        通过子进程实现各 Case 执行上下文隔离, 避免 Case 间相互影响

        :param job_idx: JobIdx
        :param gtest_filter: GTestFilter
        :exception RuntimeError 本用例执行失败时, 抛出该类型异常
        """
        ts = datetime.now(tz=timezone.utc)
        try:
            cmd: str = f"{self.xsan_options} ./{self.target_name} "
            if gtest_filter:
                cmd += f" --gtest_filter={gtest_filter}"
            # 通过全局锁, 来确保各用例输出结果在的输出顺序
            with self.dfx_output_lock:
                logging.info("Job[%s] [BGN] Run GTest(%s) XSAN(ASAN:%s UBSAN:%s) GTestFilter(%s)",
                             job_idx, self.target_name, self.asan_option, self.ubsan_option, gtest_filter)
            ret = subprocess.run(shlex.split(cmd), cwd=self.working_dir,
                                 capture_output=True, check=False, text=True, encoding='utf-8')
            if ret.returncode:
                msg = f"stdout:\n{ret.stdout}\n{ret.stderr}"
                err = RuntimeError(f"Cmd:{cmd}\nJob:{job_idx}\nDetails:\n{msg}")
                # 收集错误现场信息并上报
                self.case_fail_error_queue.put(err)
                # 异常后处理
                if self.halt_on_error:
                    self.job_terminate_event.set()
                raise err  # 触发 Job 执行进程感知 Case 执行异常
            msg = ret.stdout or ret.stderr
            # 通过全局锁, 来确保各用例输出结果在的输出顺序
            with self.dfx_output_lock:
                logging.info("Job[%s] [END] Run GTest(%s) XSAN(ASAN:%s UBSAN:%s) GTestFilter(%s) %s "
                             "Output Below:\n%s",
                             job_idx, self.target_name, self.asan_option, self.ubsan_option, gtest_filter,
                             self.dfx_case_progress(update=True), msg)
                time_brief: List[Any] = [job_idx, gtest_filter, (datetime.now(tz=timezone.utc) - ts)]
                self.case_exec_queue.put(time_brief)

        except KeyboardInterrupt:
            # 强制终止时, 主动退出执行, 上报已运行时长
            logging.info("Job[%s] Case[%s] receive terminate event.", job_idx, gtest_filter)
            brief: List[Any] = [job_idx, gtest_filter, (datetime.now(tz=timezone.utc) - ts)]
            self.case_terminate_queue.put(brief)

    def dfx_job_progress(self, update=True) -> str:
        """
        获取 Job 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            self.dfx_job_exit_cnt.value += 1
        cnt: int = int(self.dfx_job_exit_cnt.value)
        pgs: float = cnt / self.job_num * 100
        return f"JobProgress[{cnt}/{self.job_num} {pgs:.2f}%]"

    def dfx_case_progress(self, update=True) -> str:
        """
        获取 Case 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            self.dfx_case_finish_cnt.value += 1
        cnt: int = int(self.dfx_case_finish_cnt.value)
        pgs: float = cnt / len(self.case_list) * 100
        return f"CaseProgress[{cnt}/{len(self.case_list)} {pgs:.2f}%]"

    def _post_job_exec_info(self) -> Tuple[str, timedelta]:
        job_exec_str: str = ""
        jobs_exec_ori_secs: timedelta = timedelta()
        while not self.job_brief_queue.empty():
            brief = self.job_brief_queue.get()
            job_id: int = int(brief[0])
            job_total: int = int(brief[1])
            job_succ: int = int(brief[2])
            job_fail: int = int(brief[3])
            job_cost: timedelta = brief[-1]
            jobs_exec_ori_secs += job_cost
            job_exec_str += (f"\n\tJob({job_id})\tCost({job_cost.seconds}) secs\t"
                             f"Progress[T/S/F {job_total}/{job_succ}/{job_fail}]")
            self.job_brief_queue.task_done()
        return job_exec_str, jobs_exec_ori_secs

    def _post_case_terminate_info(self) -> Tuple[str, int]:
        case_terminate_cnt: int = 0
        case_terminate_str: str = ""
        while not self.case_terminate_queue.empty():
            cs_bf = self.case_terminate_queue.get()
            job_id: int = int(cs_bf[0])
            cs_name: str = str(cs_bf[1])
            cs_cost: int = cs_bf[2].seconds
            case_terminate_cnt += 1
            case_terminate_str += f"\n\t{case_terminate_cnt}\tJob({job_id})\tCost({cs_cost}) secs\tCase({cs_name})"
            self.case_terminate_queue.task_done()
        return case_terminate_str, case_terminate_cnt

    def _post_case_time_info(self) -> str:
        """
        获取各用例执行时间
        """
        case_times = []
        while not self.case_exec_queue.empty():
            job_idx, case_name, duration = self.case_exec_queue.get()
            case_times.append((job_idx, case_name, duration))
            self.case_exec_queue.task_done()
        if not case_times:
            return "none case executed"
        
        time_info = ""
        col0_width = len("job_idx")
        col1_width = max(len("case_name"), max(len(name) for job_idx, name, duration in case_times))
        col2_width = max(
            len("time"),
            max(len(f"{duration.total_seconds():.2f} seconds") for job_idx, name, duration in case_times)
        )

        logging.info("\nCase Execution Report")
        logging.info("+" + "-" * col0_width + "+" + "-" * col1_width + "+" + "-" * col2_width + "+")
        logging.info("|" + "job_idx".ljust(col0_width) 
            + "|" + "case_name".ljust(col1_width) 
            + "|" + "time".rjust(col2_width) + "|"
        )
        logging.info("+" + "-" * col0_width + "+" + "-" * col1_width + "+" + "-" * col2_width + "+")
        for job_idx, name, duration in case_times:
            logging.info("|" + str(job_idx).ljust(col0_width) 
            + "|" + name.ljust(col1_width) + "|" 
            + f"{duration.total_seconds():.2f} seconds".rjust(col2_width) + "|")
            logging.info("+" + "-" * col0_width + "+" + "-" * col1_width + "+" + "-" * col2_width + "+")
        return time_info   

    def _post_case_exec_info(self) -> int:
        case_remaining_cnt: int = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                case_remaining_cnt += 1
            self.case_queue.task_done()
        return case_remaining_cnt

    def _post_case_fail_info(self) -> Tuple[str, int]:
        case_fail_datas: List[str] = []
        while not self.case_fail_error_queue.empty():
            brief = self.case_fail_error_queue.get()
            case_fail_datas.append(str(brief))
            self.case_fail_error_queue.task_done()
        case_fail_brief = "None" if len(case_fail_datas) == 0 else ""
        for idx, data in enumerate(case_fail_datas, start=1):
            case_fail_brief += f"\nIdx:{idx}/{len(case_fail_datas)}\n{data}"
        return case_fail_brief, len(case_fail_datas)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if UTestAccelerate.main() else 1)
