#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE for details.
"""pypto.integration.akg_bench — KernelBench 桥接层.

本子包提供"输入 KernelBench 用例 → pypto 7 阶段 agent 工作流生成算子
→ 内置 KernelVerifier 精度验证 + 性能测试"的端到端批处理脚手架.
全部代码自包含在本目录, 无外部源码依赖 (除 pypto 本身与 opencode CLI).

公共入口:
    - case_loader.load_case / case_loader.write_spec
    - pypto_runner.run_pypto_workflow
    - akg_verifier_runner.run_verifier
    - run_kernelbench.main (CLI)
"""

__all__ = [
    "case_loader",
    "pypto_runner",
    "akg_verifier_runner",
    "report",
]
