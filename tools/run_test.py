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
"""
"""
import subprocess
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import os
from queue import Queue
import shutil
import re


# 定义要运行的测试用例列表
test_cases = [
        "FunctionTest.TestAddTensorFunctionDim4",
        "FunctionTest.TestAddTensorFunctionDim2",
        "FunctionTest.TestOperationRopeV2Deepseekv3B32",
        "FunctionTest.test_fa_new",
        "FunctionTest.TestSubTensorFunctionDim2",
        "FunctionTest.TestMulTensorFunctionDim2",
        "FunctionTest.TestDivTensorFunctionDim2",
        "FunctionTest.TestAddScalarFunctionDim2",
        "FunctionTest.TestAddScalarFunctionDim3",
        "FunctionTest.TestSubScalarFunctionDim2",
        "FunctionTest.TestMulScalarFunctionDim2",
        "FunctionTest.TestDivScalarFunctionDim2",
        "FunctionTest.TestExpTensorFunctionDim2",
        "FunctionTest.TestSin",
        "FunctionTest.TestCos",
        "FunctionTest.TestGatherAxis0Indices2_1"
        "FunctionTest.TestGatherAxis1Indices2_1"
        "FunctionTest.TestGatherAxis3Indices4_2"
        "FunctionTest.TestGatherElementAxis1Indices2",
        "FunctionTest.TestGatherElementAxis0Indices2",
        "FunctionTest.TestScatter_",
        "FunctionTest.TestScatterUpdate2",
        "FunctionTest.testRowSumSingle",
        "FunctionTest.testRowMaxSingle",
        "FunctionTest.testSoftmax",
        "FunctionTest.TestRoPE",
        "FunctionTest.TestRoPEDeepseekV3",
        "FunctionTest.testRmsNormNewMultiDims",
        "FunctionTest.TestConcat",
        # "FunctionTest.TestAttention",
        "FunctionTest.TestAttentionPost",
        "FunctionTest.Test_qkvPre",
        "FunctionTest.Test_qkvPre2",
        "FunctionTest.Test_deepseekAttention_pre",
        "FunctionTest.TestBMMtest",
        "FunctionTest.TestBMMtest2",
        "FunctionTest.Test_deepseekMoEGate",
        "FunctionTest.Test_quant",
        "FunctionTest.Test_ScalarOp",
        "FunctionTest.TestPad",
        "FunctionTest.Test_quantMM",
        "FunctionTest.TestRmsNorm",
        "OperationImplTest.Test_MatmulWithSplitK",
        "OperationImplTest.Test_MatmulWithSplitKWithTrans",
        "OperationImplTest.test_BMMT_NZ_1_128_256_128_Batch",
]  # 示例测例名称

BIN_PATH = "/path_to_build/tests/ut/tile_fwk_utest"
output_dir = "test_case"
# 存储结果的字典
results = {}


def run_test(test_case, result_queue, delay):
    time.sleep(delay)
    """执行单个测试用例"""
    command = f"{BIN_PATH} --gtest_filter={test_case}"
    print("start execute:", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    log_file = open(os.path.join(output_dir, f"{test_case}.log"), 'w')
    # 获取子进程的输出和错误
    stdout_data, stderr_data = process.communicate()
    
    # 将输出和错误写入对应的文件
    log_file.write(stdout_data.decode())
    log_file.write(stderr_data.decode())

    found = False
    match_line = "CostModel Simulation Runtime"
    
    lines = stdout_data.decode('utf-8').splitlines()
    for line in lines:
        if match_line in line:
            found = True
            array = line.split(' ')
            t = array[-1]
            result_queue.put([test_case, True, t])
            break
    if not found:
        result_queue.put([test_case, False, 0])
    
    process.wait()
    print("end execute:", command)
    log_file.close()


def extract_output_filename(log_content):
    # 使用正则表达式匹配 output_xxx_xxx 格式的文件名
    pattern = r'output_\w+_\w+'
    match = re.search(pattern, log_content)
    if match:
        return match.group()
    return None


def ana_output(dir1, dir2, dir3):
    # 确保目标文件夹存在
    if not os.path.exists(dir3):
        os.makedirs(dir3)

    # 遍历 dir1 中的所有日志文件
    for log_file in os.listdir(dir1):
        log_path = os.path.join(dir1, log_file)

        # 只处理文件
        if os.path.isfile(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()

                # 提取输出文件名
                output_filename = extract_output_filename(log_content)
                if not output_filename:
                    print(f"未在 {log_file} 中找到输出文件名")
                    continue

                # 查找 dir2 中的输出文件
                output_file_path = os.path.join(dir2, output_filename)
                if os.path.exists(output_file_path):
                    # 复制到 dir3，并重命名为日志文件名
                    new_file_name = os.path.splitext(log_file)[0] + os.path.splitext(output_filename)[1]
                    new_file_path = os.path.join(dir3, new_file_name)
                    shutil.copytree(output_file_path, new_file_path)
                    print(f"成功复制 {output_filename} 到 {new_file_name}")
                else:
                    print(f"未在 dir2 中找到文件: {output_filename}")

            except Exception as e:
                print(f"处理文件 {log_file} 时出错: {e}")


def main():
    global output_dir
    current_datetime = datetime.now(ZoneInfo("Asia/Shanghai"))
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    print(formatted_datetime)
    output_dir = output_dir + "_" + formatted_datetime
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()
    result_queue = Queue()
    threads = []
    delay = 0
    for test_case in test_cases:
        thread = threading.Thread(target=run_test, args=(test_case, result_queue, delay))
        thread.start()
        threads.append(thread)
        delay += 10

    for thread in threads:
        thread.join()

    items = []
    while not result_queue.empty():
        items.append(result_queue.get())

    sorted_res = sorted(items, key=lambda x: (x[1] == True, x[0]))
    res_file = open(os.path.join(output_dir, f"total_res.csv"), 'w')
    for item in sorted_res:
        res = f"{item[0]}, {item[1]}, {item[2]}\n"
        res_file.write(res)
        print(item)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函数执行时间: {elapsed_time:.4f} 秒")
    ana_output(output_dir, './output', output_dir)


if __name__ == "__main__":
    main()