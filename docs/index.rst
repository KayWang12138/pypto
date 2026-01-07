.. 
   Copyright (c) 2025 Huawei Technologies Co., Ltd.
   This program is free software, you can redistribute it and/or modify it under the terms and conditions of
   CANN Open Software License Agreement Version 2.0 (the "License").
   Please refer to the License for details. You may not use this file except in compliance with the License.
   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   See LICENSE in the root of the software repository for the full text of the License.
   
PyPTO文档中心
=====================

欢迎使用PyPTO文档。

PyPTO（发音:pai p-t-o）是CANN推出的一款面向AI加速器的高效编程框架，旨在简化算子开发流程，同时保持高性能计算能力。该框架采用创新的PTO（Parallel Tensor/Tile Operation）编程范式，以基于Tile的编程模型为核心设计理念，通过多层次的计算图表达，将用户通过API构建的AI模型从高层次的Tensor计算图逐步编译成硬件指令，最终生成可在目标平台上高效执行的代码，并由设备侧以MPMD（Multiple Program Multiple Data）方式调度执行。

.. PyPTO documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.

PyPTO文档中心
=============

.. toctree::
    :maxdepth: 2
    :caption: 目录

    context/index
    tutorials/index
    api/README
    tools/index