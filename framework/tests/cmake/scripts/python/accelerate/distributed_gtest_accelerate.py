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
"""GTest 执行加速 - 4卡用例在16卡上并行执行优化版
"""
import argparse
import dataclasses
import logging
import os
import queue
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from multiprocessing import JoinableQueue, Event, Process, Value
from typing import List, Any, Optional, Tuple, Dict, Callable

from utils.args_action import ArgsEnvDictAction, ArgsGTestFilterListAction
from utils.executable import Executable
from utils.table import Table


class DistriutedGTestAccelerate(ABC):
    """GTest 加速 - 支持4卡用例在16卡上并行执行
    """

    @dataclasses.dataclass
    class ExecParam:
        """执行参数 - 增强版支持设备组
        """
        cntr_id: Optional[int] = None
        envs_func: Optional[Callable] = None
        custom: Optional[Any] = None

        def __init__(self, cntr_id: int, envs_func: Optional[Callable] = None, custom: Optional[Any] = None):
            self.cntr_id = cntr_id
            self.envs_func = envs_func
            self.custom = custom

        def get_envs(self) -> Optional[Dict[str, str]]:
            """获取额外的环境变量配置
            """
            if self.envs_func:
                return self.envs_func(self)
            return None

    @dataclasses.dataclass
    class ExecResult:
        """执行结果
        """
        cntr_name: str = "Cntr"
        duration: Optional[timedelta] = None
        cntr_execution_details: JoinableQueue = JoinableQueue()
        cntr_duration_dict: Dict[int, timedelta] = dataclasses.field(default_factory=dict)
        case_execution_details: JoinableQueue = JoinableQueue()
        case_exception_details: JoinableQueue = JoinableQueue()
        case_terminate_details: JoinableQueue = JoinableQueue()

        def get_cntr_exec_info(self) -> Tuple[str, str]:
            """获取 Container 执行信息统计.

            :returns:
                Tuple[str, str]:
                    - Container 执行信息统计表(str)
                    - Container 并行执行收益描述(str)
            """
            heads = [self.cntr_name, "Total", "Success", "Failed", "Duration"]
            datas = []
            duration_sum = timedelta()
            while not self.cntr_execution_details.empty():
                _brief = self.cntr_execution_details.get()
                devs_id = int(_brief[0])
                case_total = int(_brief[1])
                case_pass = int(_brief[2])
                case_fail = int(_brief[3])
                devs_duration = _brief[-1]
                self.cntr_duration_dict[devs_id] = devs_duration
                duration_sum += devs_duration
                datas.append([devs_id, case_total, case_pass, case_fail, f"{devs_duration.total_seconds():.2f}"])
                self.cntr_execution_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                brief = Table.table(datas=datas, headers=heads)
            # 并行执行收益计算
            rate = (
                float((duration_sum - self.duration) / self.duration) * 100 
                if self.duration and duration_sum.total_seconds() > 0 
                else 0
            )
            desc = f"Duration {self.duration.total_seconds():.2f} secs, Revenue(Act/Ori, "
            desc += f"{self.duration.total_seconds():.2f}/{duration_sum.total_seconds():.2f}) {rate:.2f}%"
            return f"\n\n{self.cntr_name} Execution Brief:{brief}", desc

        def get_case_exec_terminate_info(self) -> Tuple[str, int]:
            """获取 Case 执行终止信息.

            :returns:
                Tuple[str, int]:
                    - Case 终止执行情况信息
                    - Case 终止执行数量
            """
            heads = ["Idx", self.cntr_name, "CaseName", "Duration"]
            datas = []
            while not self.case_terminate_details.empty():
                _brief = self.case_terminate_details.get()
                job_id = int(_brief[0])
                case_name = str(_brief[1])
                case_duration = _brief[2]
                datas.append([job_id, case_name, f"{case_duration.total_seconds():.2f}"])
                self.case_terminate_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                datas = [[f"{idx}/{len(datas)}"] + ele for idx, ele in enumerate(datas, start=1)]
                brief = Table.table(datas=datas, headers=heads)
            return f"\n\nCase Terminate Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_exception_info(self) -> Tuple[str, int]:
            """获取 Case 执行异常信息.

            :returns:
                Tuple[str, int]:
                    - Case 异常执行情况信息
                    - Case 异常执行数量
            """
            datas = []
            brief = ""
            while not self.case_exception_details.empty():
                chunk = self.case_exception_details.get()
                if len(chunk) != 0:
                    brief += chunk
                else:
                    datas.append(str(brief))
                    brief = ""
                self.case_exception_details.task_done()
            brief = "\nNone" if len(datas) == 0 else ""
            for idx, data in enumerate(datas, start=1):
                brief += f"\nIdx:{idx}/{len(datas)}\n{data}"
            return f"\n\nCase Exception Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_duration_info(self) -> str:
            """获取 Case 执行耗时统计信息.

            :return: Case 执行耗时统计信息.
            """
            heads = [self.cntr_name, "CaseName", "Duration", f"Ratio({self.cntr_name})", "Ratio(Total)"]
            datas = []
            while not self.case_execution_details.empty():
                _brief = self.case_execution_details.get()
                job_idx = _brief[0]
                case_name = str(_brief[1])
                case_duration = _brief[2]
                job_duration = self.cntr_duration_dict[job_idx]
                ratio_job = float(case_duration / job_duration) * 100 if job_duration.total_seconds() > 0 else 0
                ratio_process = (
                    float(case_duration / self.duration) * 100
                    if self.duration and self.duration.total_seconds() > 0 
                    else 0
                )
                datas.append(
                    [job_idx, case_name, case_duration.total_seconds(),
                     f"{case_duration.total_seconds():.2f}/{job_duration.total_seconds():.2f} {ratio_job:.2f}%",
                     f"{case_duration.total_seconds():.2f}/{self.duration.total_seconds():.2f} "
                     f"{ratio_process:.2f}%"])
                self.case_execution_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                # 把 data 按耗时降序重排, 重排后转换格式
                duration_idx = 2  # 2 is idx of duration
                datas = sorted(datas, key=lambda x: x[duration_idx], reverse=True)
                for item in datas:
                    item[duration_idx] = f"{item[duration_idx]:.2f}"
                brief = Table.table(datas=datas, headers=heads, auto_sort=False)
            return f"\n\nCase Duration Brief:{brief}"

    @dataclasses.dataclass
    class CntrContext:
        """Cntr处理上下文
        """
        cntr_id: int
        exec_param: Any
        success: int = 0
        failed: int = 0
        ts: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))
        exit_code: int = 0

        @property
        def total(self) -> int:
            return self.success + self.failed

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.total, self.success, self.failed, (datetime.now(timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class CaseContext:
        """Case处理上下文
        """
        cntr_id: int
        exec_param: Any
        gtest_filter: str
        ts: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.gtest_filter, (datetime.now(timezone.utc) - self.ts)]


    def __init__(self, args, params: List[ExecParam], cntr_name: str = "Cntr"):
        """
        :param args: 命令行参数
        :param params: 执行参数
        :param cntr_name: 容器名称, 用于回显内容
        """
        # 用例执行参数, 执行行为控制参数
        self.exe: Executable = Executable(file=args.target[0], envs=args.envs, timeout=args.timeout_case)
        
        # 自动创建设备组感知的执行参数
        self.exe_params: List[DistriutedGTestAccelerate.ExecParam] = self._create_4card_distributed_params(args)
        self.exe_result: DistriutedGTestAccelerate.ExecResult = (
            DistriutedGTestAccelerate.ExecResult(cntr_name=cntr_name)
        )
        self.exe_timeout: Optional[int] = args.timeout
        self.exe_halt_on_error: bool = args.halt_on_error

        # 用例管理
        self.case_list: List[str] = args.cases
        self.case_queue: JoinableQueue = JoinableQueue()
        self.case_execution_queue: JoinableQueue = JoinableQueue()
        self.case_exception_queue: JoinableQueue = JoinableQueue()
        self.case_terminate_queue: JoinableQueue = JoinableQueue()
        self.case_exec_count = Value('i', 0)

        # 容器管理
        self.cntr_name: str = cntr_name
        self.cntr_execution_queue: JoinableQueue = JoinableQueue()
        self.cntr_terminate_event = Event()
        self.cntr_exit_count = Value('i', 0)

        if len(self.exe_params) == 0:
            raise ValueError("No device groups created, cannot run any task")
        if len(self.exe_params) > len(self.case_list):
            logging.info("CaseNum(%s) less than device groups=%s, will use first %s groups",
                         len(self.case_list), len(self.exe_params), len(self.case_list))
            self.exe_params = self.exe_params[:len(self.case_list)]
        logging.info("\n\n%s 4-Card Distributed Execution Args:%s", self.mark, Table.table(datas=self.brief))


    @property
    def brief(self) -> List[Any]:
        """简要信息
        """
        return [
            ["Executable", self.exe.file],
            ["Total Timeout", self.exe_timeout],
            ["HaltOnError", self.exe_halt_on_error],
            ["Device Groups", len(self.exe_params)],
            ["Test Cases", len(self.case_list)],
            ["Case Timeout", self.exe.timeout],
            ["Execution Mode", "4-Card Distributed Only"]
        ]


    @property
    @abstractmethod
    def mark(self) -> str:
        pass


    @staticmethod
    def reg_args(parser: argparse.ArgumentParser):
        """注册命令行参数
        """
        parser.add_argument("-t", "--target", nargs=1, type=str, required=True,
                          help="Target executable file path")
        parser.add_argument("-e", "--env", nargs="+", action=ArgsEnvDictAction, 
                          default={}, dest="envs", help="Environment variables")
        parser.add_argument("--timeout", type=int, default=None, help="Total timeout")
        parser.add_argument("--timeout_case", type=int, default=None, help="Per-case timeout")
        parser.add_argument("--halt_on_error", action="store_true", default=False,
                          help="Stop on first failure")
        parser.add_argument("--gtest_filter", nargs="+", action=ArgsGTestFilterListAction,
                          default=[], required=True, dest="cases", help="Test cases to run")


    def process(self):
        """执行任务
        """
        ts = datetime.now(tz=timezone.utc)
        self._main()
        self.exe_result.duration = datetime.now(tz=timezone.utc) - ts


    def post(self) -> bool:
        """后处理, 获得执行结果汇总
        """
        cntr_exec_brief, cntr_revenue_desc = self.exe_result.get_cntr_exec_info()
        case_exec_brief, case_exec_result = self._post_case_exec_info()

        out = f"{self.mark}, HaltOnError({self.exe_halt_on_error}), {cntr_revenue_desc}"
        out += cntr_exec_brief
        out += case_exec_brief

        if case_exec_result:
            logging.info(out)
        else:
            logging.error(out)
        return case_exec_result


    def _create_4card_distributed_params(self, args) -> List[ExecParam]:
        """创建设备组参数 - 使用闭包避免可变默认参数
        """
        if hasattr(args, 'device') and args.device is not None:
            total_devices = len(args.device)
            logging.info(f"Have {total_devices} devices: {args.device}")
        else:
            total_devices = 0
            logging.error(f"Have {total_devices} device.")
        devices_per_group = 4
        num_groups = total_devices // devices_per_group
        
        if num_groups == 0:
            logging.error("Not enough devices for 4-card distributed execution")
            return []
        
        # 创建设备组
        device_groups = []
        for group_id in range(num_groups):
            start_device = group_id * devices_per_group
            device_list = list(range(start_device, start_device + devices_per_group))
            device_list_str = ','.join(map(str, device_list))
            
            # 使用闭包捕获当前值，避免默认参数问题
            def make_envs_func(devices, d_str, env_dict):
                def create_group_envs(param):
                    env_vars = {
                        "TILE_FWK_DEVICE_ID_LIST": d_str,
                        "CUDA_VISIBLE_DEVICES": d_str,
                        "ASCEND_DEVICE_IDS": d_str,
                    }
                    if env_dict:
                        env_vars.update(env_dict)
                    return env_vars
                return create_group_envs
            
            # 获取环境变量字典
            envs_dict = getattr(args, 'envs', {})
            
            # 创建新的ExecParam
            new_param = DistriutedGTestAccelerate.ExecParam(
                cntr_id=group_id,
                envs_func=make_envs_func(device_list, device_list_str, envs_dict),
                custom={
                    "devices": device_list,
                    "device_list_str": device_list_str,
                }
            )
            device_groups.append(new_param)
        
        logging.info("Created %d device groups for 4-card distributed execution:", len(device_groups))
        return device_groups


    def _post_case_exec_info(self) -> Tuple[str, bool]:
        """获取 Case 执行信息.

        :returns:
            Tuple[str, bool]:
                ▪ Case 执行情况信息

                ▪ Case 执行成功与否判定结果

        """
        terminate_brief, terminate_count = self.exe_result.get_case_exec_terminate_info()
        exception_brief, exception_count = self.exe_result.get_case_exec_exception_info()
        duration_brief = self.exe_result.get_case_exec_duration_info()

        # Case 执行总体情况汇总
        remaining_count = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                remaining_count += 1
            self.case_queue.task_done()
        success_count = len(self.case_list) - remaining_count - terminate_count - exception_count
        execution_heads = ["Total", "Success", "Failed", "Terminate", "Remaining"]
        execution_datas = [[len(self.case_list), success_count, exception_count, terminate_count, remaining_count]]
        execution_brief = Table.table(datas=execution_datas, headers=execution_heads)
        execution_brief = f"\n\nCase Execution Brief:{execution_brief}"

        rst = (terminate_count + exception_count + remaining_count) == 0
        out = execution_brief + duration_brief + terminate_brief + exception_brief
        return out, rst


    def _main(self):
        """主执行逻辑
        """
        cntr_process_group = []
        try:
            self._push_all_cases()
            cntr_process_group = self._start_cntr_process_group()
            self._wait_for_completion(cntr_process_group)
        except KeyboardInterrupt:
            logging.info("Main process received interrupt signal")
        finally:
            self._cleanup_processes(cntr_process_group)


    def _push_all_cases(self):
        """将用例加入队列
        """
        for case in self.case_list:
            self.case_queue.put(case)
        for _ in range(len(self.exe_params)):
            self.case_queue.put(None)


    def _start_cntr_process_group(self) -> List[Process]:
        """启动容器进程组
        """
        processes = []
        for param in self.exe_params:
            device_info = f"Devices{param.custom['devices']}"
            process = Process(
                name=f"{self.cntr_name}Process-{device_info}",
                target=self._cntr,
                args=(param.cntr_id, param, 0)
            )
            processes.append(process)
            process.start()
        return processes


    def _wait_for_completion(self, processes: List[Process], check_interval: int = 1):
        """等待进程完成
        """
        start_time = time.time()
        while True:
            time.sleep(check_interval)
            
            if self.exe_timeout and (time.time() - start_time) > self.exe_timeout:
                self.cntr_terminate_event.set()
                logging.warning("Execution timeout, terminating all processes")
                break
                
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count == 0:
                break
                
            if any(p.exitcode != 0 and self.exe_halt_on_error for p in processes if not p.is_alive()):
                self.cntr_terminate_event.set()
                break


    def _cleanup_processes(self, processes: List[Process], timeout: int = 5):
        """清理进程
        """
        self.cntr_terminate_event.set()
        
        for process in processes:
            if process.is_alive():
                process.join(timeout=timeout)
        
        for process in processes:
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGTERM)
                    process.join(timeout=1)
                except ProcessLookupError:
                    logging.warning(f"process {process.pid} stopped")
                except Exception as e:
                    logging.warning(f"process {process.pid} error: {e}")


    def _cntr(self, cntr_id: int, exec_param, delay: int):
        """容器进程 - 专用于4卡分布式执行
        """
        ctx = DistriutedGTestAccelerate.CntrContext(cntr_id=cntr_id, exec_param=exec_param)
        
        try:
            time.sleep(delay)
            while not self.cntr_terminate_event.is_set():
                case = self._get_next_case()
                if case is None:
                    break
                self._execute_case(cntr_id, exec_param, case, ctx)
        except KeyboardInterrupt:
            self.cntr_terminate_event.set()
            logging.info(f" {cntr_id} exit success")


    def _get_next_case(self) -> Optional[str]:
        """获取下一个用例
        """
        try:
            case = self.case_queue.get(timeout=1)
            self.case_queue.task_done()
            return case
        except queue.Empty:
            return None


    def _execute_case(self, cntr_id: int, exec_param, case: str, ctx: CntrContext):
        """执行单个用例
        """
        process = None
        try:
            process = Process(
                target=self._case,
                args=(cntr_id, exec_param, case)
            )
            process.start()
            process.join()
            
            if process.exitcode == 0:
                ctx.success += 1
            else:
                ctx.failed += 1
                if self.exe_halt_on_error:
                    self.cntr_terminate_event.set()
                    ctx.exit_code = process.exitcode
        except Exception as e:
            logging.error("Error executing case %s: %s", case, e)
            ctx.failed += 1


    def _case(self, cntr_id: int, param: ExecParam, gtest_filter: str):
        """执行GTest用例 - 实时输出4卡分布式测试结果
        """
        ctx = DistriutedGTestAccelerate.CaseContext(cntr_id=cntr_id, exec_param=param, 
                                         gtest_filter=gtest_filter)
        device_info = f"Device{param.custom['devices']}"
        logging.info("Executing %s on %s", gtest_filter, device_info)
        
        try:
            env_vars = os.environ.copy()
            if param.get_envs():
                env_vars.update(param.get_envs())
            
            command = [
                'mpirun', '-n', '4',
                str(self.exe.file),
                f'--gtest_filter={gtest_filter}'
            ]
            
            process = subprocess.Popen(
                command,
                env=env_vars,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            return_code = process.wait()
            
            if return_code == 0:
                logging.info("TestCase: %s success", gtest_filter)
                self.case_execution_queue.put(ctx.brief)
            else:
                logging.error("TestCase: %s failed, return: %s", gtest_filter, return_code)
                self._case_exception_exit(cntr_id, command, return_code, out="", err=f"return: {return_code}")
                
                
        except subprocess.TimeoutExpired as e:
            self._report_case_error(cntr_id, str(e), 1, "timesOut")
        except Exception as e:
            self._report_case_error(cntr_id, f"mpirun -n 4 {gtest_filter}", 1, str(e))


    def _case_exception_exit(self, cntr_id: int, cmd: str, ret_code: int,
                            out: Optional[str] = None, err: Optional[str] = None):
        """用例执行进程异常退出处理
        """
        # 收集错误现场信息并上报
        msg = (f"{self.cntr_name} : {cntr_id}\n"
               f"Cmd : {cmd}\n"
               f"RetCode : {ret_code}\n"
               f"stdout :\n{out}\n"
               f"stderr :\n{err}")
        self._put_case_exception_info(info=msg)
        # 异常后处理
        if self.exe_halt_on_error:
            self.cntr_terminate_event.set()
            logging.info("Send terminate event upload.")


    def _put_case_exception_info(self, info: str, chunk_size: int = 4096):
        for i in range(0, len(info), chunk_size):
            self.case_exception_queue.put(info[i:i + chunk_size])
        self.case_exception_queue.put("")  # 插入分隔符
