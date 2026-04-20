#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE for details.
"""pypto.integration.akg_bench.verifier — KernelBench 桥接层内置 KernelVerifier.

支持的组合 (其它输入直接 ``ValueError``):
    - dsl       = ``"pypto"``
    - backend   = ``"ascend"``
    - framework = ``"torch"``
    - bench     = ``"kernelbench"``
    - worker    = 本子包提供的 ``LocalWorker``

公共入口:
    - ``KernelVerifier``        — ``run`` / ``run_profile``
    - ``register_local_worker`` — 注册并发现 LocalWorker
    - ``get_worker_manager``    — 获取全局 WorkerManager 单例
    - ``load_config``           — 返回桥接层默认配置 dict
"""

from .config import load_config
from .kernel_verifier import KernelVerifier
from .manager import (
    WorkerManager,
    get_worker_manager,
    register_local_worker,
)
from .local_worker import LocalWorker

__all__ = [
    "KernelVerifier",
    "LocalWorker",
    "WorkerManager",
    "get_worker_manager",
    "load_config",
    "register_local_worker",
]
