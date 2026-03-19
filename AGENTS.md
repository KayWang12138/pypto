# AGENTS.md

本文件为 OpenCode 在本代码仓库中进行 PyPTO 算子开发提供指导。

## 项目概述

本项目是华为 CANN PyPTO 算子开发项目，用于开发能在华为昇腾 AI 处理器上运行的自定义算子

### 核心功能

- 使用 PyPTO 编程语言开发昇腾 AI 处理器自定义算子
- 提供完整的开发、构建、测试及性能调优工作流支持
- 遵循官方开发规范和性能优化最佳实践

---

## 构建与测试命令

### 环境准备

```bash
# 设置NPU设备ID（必需）
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/

# 如果TILE_FWK_DEVICE_ID=0报"Invalid Device"，运行以下命令查看可用设备
npu-smi info
# 然后设置正确的设备号
export TILE_FWK_DEVICE_ID=<正确的设备号>
```

### 构建命令

```bash
# 编译whl包（推荐）
python3 build_ci.py -f python3 --disable_auto_execute

# 完整构建（包含C++后端）
python3 build_ci.py -f python3 -b npu --build_type Release

# 清理后重新构建
python3 build_ci.py -c -f python3

# 可编辑模式安装（开发时推荐）
python3 build_ci.py -f python3 --editable

# 使用cost_model后端（无NPU环境时）
python3 build_ci.py -f python3 -b cost_model
```

### 测试命令

```bash
# 运行所有UT测试
python3 build_ci.py -f python3 -u

# 运行所有ST测试
python3 build_ci.py -f python3 -s

# 运行特定UT测试模块
python3 build_ci.py -f python3 -u --utest_module test_operator

# 使用pytest运行单个测试文件
pytest python/tests/ut/operator/test_operator.py -v

# 运行单个测试用例
pytest python/tests/ut/operator/test_operator.py::test_function_name -v

# 使用特定设备运行ST测试
python3 build_ci.py -f python3 -s -d 0

# 运行examples验证
python3 build_ci.py -f python3 --example
```

### 运行单个示例

```bash
# NPU模式运行
python examples/03_advanced/advanced_nn/attention/attention.py --run_mode npu

# 模拟器模式运行（无NPU环境）
python examples/03_advanced/advanced_nn/attention/attention.py --run_mode sim
```

---

## 代码风格指南

### 文件头注释

所有Python文件必须包含标准版权头：

```python
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
```

### 导入顺序

```python
# 1. 标准库
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict

# 2. 第三方库
import torch
import numpy as np

# 3. 本地模块
import pypto
```

### 命名约定

| 类型 | 约定 | 示例 |
|-----|------|------|
| 模块/包 | 小写下划线 | `test_operator.py` |
| 类名 | 大驼峰 | `AttentionConfig` |
| 函数名 | 小写下划线 | `scaled_dot_product_attention` |
| 常量 | 大写下划线 | `BATCH_SIZE`, `NUM_HEADS` |
| 私有方法 | 单下划线前缀 | `_get_job_num()` |

### 类型注解

```python
def scaled_dot_product_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ...

@dataclass
class AttentionConfig:
    num_heads: int = 8
    head_dim: int = 64
    scale: Optional[float] = None
    dtype: pypto.DataType = pypto.DT_BF16
```

### 错误处理

```python
# 环境变量验证
if 'TILE_FWK_DEVICE_ID' not in os.environ:
    print("Please set TILE_FWK_DEVICE_ID before running")
    return None

try:
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
except ValueError:
    print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer")
    return None

# 参数验证
if run_mode not in ["npu", "sim"]:
    raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
```

### 文档字符串

```python
def get_device_id() -> Optional[int]:
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """

def test_attention_with_projection(device_id=None, run_mode: str = "npu") -> None:
    """Test complete attention with input/output projections."""
```

### PyPTO 特定规范

```python
# JIT装饰器定义kernel
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def attention_kernel(
    q: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    k: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
) -> pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16):
    ...

# Tiling配置
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)

# 数据类型使用常量
pypto.DT_BF16  # bfloat16
pypto.DT_FP16  # float16
pypto.DT_FP32  # float32
```

---

## 目录结构

```
pypto/
├── python/pypto/          # Python包源码
│   └── tests/             # 测试用例
│       ├── ut/            # 单元测试
│       └── st/            # 系统测试
├── examples/              # 示例代码
│   ├── 01_beginner/       # 初级示例
│   ├── 02_intermediate/   # 中级示例
│   └── 03_advanced/       # 高级示例
├── docs/                  # 文档资源
│   └── api/               # API参考文档
├── models/                # 模型实现示例
├── framework/             # C++源码
├── build_ci.py            # 构建入口脚本
├── pytest.ini             # pytest配置
└── pyproject.toml         # Python项目配置
```

---

## 核心开发原则

### 原则 1：遇问题直接定位修复

- 搜索 `docs/` 中的 API 文档
- 查阅 `examples/` 中的官方示例
- 定位问题点后修复，**禁止简化代码或推翻重写**

### 原则 2：基于官方文档实现

- 优先查阅本地 `docs/api/` API 文档
- 参考 `examples/` 中的类似实现
- 所有API/参数用法必须严格参照官方文档

### 原则 3：渐进式调试

```
Level 0: 8-16 元素    → 基础功能验证
Level 1: 1K 元素      → 典型场景验证
Level 2: 极值/零值    → 边界情况验证
Level 3: 大数据量     → 性能验证
```

---

## 测试文件命名

- 测试文件：`test_*.py`
- 特定模型测试：`glm_*.py`, `deepseekv32_*.py`, `qwen3_next_*.py`
- 测试类：继承 `TestBuilder` 或使用 pytest 风格

## pytest标记

```python
@pytest.mark.soc("910")  # 特定SOC版本
@pytest.mark.world_size(2)  # NPU卡数要求
```

## 常见问题

| 问题 | 解决方案 |
|-----|---------|
| "Invalid Device" | 运行 `npu-smi info` 确认设备号 |
| "If no NPU environment" | 设置 `export TILE_FWK_DEVICE_ID=0` |
| 导入 pypto 失败 | 先运行 `python3 build_ci.py -f python3` 编译安装 |
| 精度不满足 | 使用 `rtol=1e-3, atol=1e-3` 作为默认容差 |

---

## Skills目录

本项目使用的skills位于 `.opencode/skills/` 目录，可用skills：
- `pypto-operator-develop-workflow`: 算子开发流程
- `pypto-operator-perf-analyzer`: 算子性能分析
- `pypto-operator-perf-autotuner`: 算子性能自动化调优
