# 阶段五：性能测试详细指南（msprof）

## 1. 前置条件

1. **环境要求**
   - 已安装CANN工具包（8.0及以上版本）
   - 可访问NPU设备
   - 已配置msprof工具路径

2. **算子要求**
   - AscendC算子已实现并可运行
   - 算子有Python测试接口（通过torch_npu调用）
   - 算子支持指定的参数配置

## 2. PyPTO性能数据采集

**开启debug模式**：
```python
flash_attention_score_v2_kernel.debug_options = {"runtime_debug_mode": 1}
```

**数据文件**：`bubble_analysis.log` 中的 `Core Total Work Time`

## 3. AscendC性能数据采集（msprof）⭐

### 步骤1：准备性能测试脚本

**脚本要求**：
- 必须包含warmup阶段（5-10次，消除首次编译开销）
- 正式测试阶段建议迭代100次以上
- 每次迭代后调用`torch.npu.synchronize()`
- 使用`print`输出参数配置信息（便于后续分析）

**脚本模板**：
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AscendC算子性能测试脚本"""

import torch
import torch_npu
import numpy as np
import os
import sys

# 配置NPU设备
device_id = 0
device = torch.device(f"npu:{device_id}")


def run_operator(B, N, Sq, Skv, D, dtype=torch.bfloat16):
    """
    运行AscendC算子
    
    参数:
        B: Batch size
        N: Head num
        Sq: Query sequence length
        Skv: Key/Value sequence length
        D: Head dimension
        dtype: 数据类型
    """
    # 设置算子参数
    input_layout = "BNSD"
    scale = 1 / (D ** 0.5)
    
    # 生成输入数据（根据实际算子调整）
    q = torch.randn(B, N, Sq, D, dtype=dtype, device=device)
    k = torch.randn(B, N, Skv, D, dtype=dtype, device=device)
    v = torch.randn(B, N, Skv, D, dtype=dtype, device=device)
    
    # Warmup阶段
    for _ in range(5):
        # 调用算子接口
        out = torch_npu.npu_fusion_attention_v2(q, k, v, N, input_layout, scale=scale)
    torch.npu.synchronize()
    
    # 正式测试阶段（100次迭代）
    for i in range(100):
        # 调用算子接口
        out = torch_npu.npu_fusion_attention_v2(q, k, v, N, input_layout, scale=scale)
    torch.npu.synchronize()


if __name__ == "__main__":
    # 配置参数
    B = 2      # Batch size
    N = 8      # Head num
    Sq = 16    # Query sequence length
    Skv = 16   # Key/Value sequence length
    D = 64     # Head dimension
    
    print(f"Running operator with msprof...")
    print(f"Parameters: B={B}, N={N}, Sq={Sq}, Skv={Skv}, D={D}")
    
    run_operator(B, N, Sq, Skv, D)
    
    print("Benchmark completed")
```

### 步骤2：执行msprof性能采集

**基本命令格式**：
```bash
msprof --output=<输出目录> --ai-core=on --task-time=l2 python <benchmark脚本>
```

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--output` | 性能数据输出目录 | `/tmp/msprof_<算子名>` |
| `--ai-core` | AI Core性能采集开关 | `on` |
| `--task-time` | Task时间采集级别 | `l2`（详细级别） |
| `--ascendcl` | AscendCL接口采集 | `on`（可选） |
| `--runtime-api` | Runtime API采集 | `off`（减少开销） |

**完整执行命令示例**：
```bash
# 清理旧数据
rm -rf /tmp/msprof_fa_output

# 执行性能采集
msprof --output=/tmp/msprof_fa_output \
       --ai-core=on \
       --task-time=l2 \
       python benchmark_flash_attention.py
```

**执行过程检查**：

执行过程中会输出：
- `[INFO] Start profiling....` - 开始采集
- `[INFO] Start export data...` - 开始导出数据
- `[INFO] Profiling finished.` - 采集完成

### 步骤3：定位性能数据文件

**输出目录结构**：
```
/tmp/msprof_<name>/
└── PROF_<timestamp>_<id>/
    ├── device_0/              # 设备端数据
    │   ├── data/              # 原始数据
    │   ├── sqlite/            # SQLite数据库
    │   ├── sample.json        # 采样数据
    │   └── info.json.0        # 设备信息
    ├── host/                  # Host端数据
    └── mindstudio_profiler_output/  # 分析结果
        ├── op_summary_<timestamp>.csv      # 算子详细数据
        ├── op_statistic_<timestamp>.csv    # 算子统计信息
        ├── task_time_<timestamp>.csv       # Task时间数据
        └── api_statistic_<timestamp>.csv   # API统计信息
```

**关键文件说明**：

| 文件 | 用途 | 关键指标 |
|------|------|----------|
| `op_summary_*.csv` | 单次执行详细数据 | Task Duration, AI Core Time |
| `op_statistic_*.csv` | 统计汇总 | Count, Avg Time, Min/Max Time |
| `task_time_*.csv` | Task级时间数据 | Task Start Time, Duration |

### 步骤4：提取纯kernel性能数据

**查看算子统计信息**：
```bash
cat /tmp/msprof_<name>/PROF_<timestamp>/mindstudio_profiler_output/op_statistic_<timestamp>.csv
```

**输出格式**：
```csv
Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
0,FlashAttentionScore,MIX_AIC,105,3166.783,28.301,30.16,41.061,100.0
```

**关键指标说明**：
- `Count`: 执行次数
- `Total Time(us)`: 总耗时（微秒）
- `Min Time(us)`: 单次最小耗时
- `Avg Time(us)`: **单次平均耗时（纯kernel时间）**
- `Max Time(us)`: 单次最大耗时（通常包含首次编译开销）
- `Ratio(%)`: 该算子占总时间的比例

**查看算子详细执行数据**：
```bash
cat /tmp/msprof_<name>/PROF_<timestamp>/mindstudio_profiler_output/op_summary_<timestamp>.csv | head -20
```

**关键列说明**：
- `Task Duration(us)`: Task执行时间（包含调度开销）
- `aicore_time(us)`: AI Core实际工作时间
- `aiv_time(us)`: AI Vector实际工作时间
- `cube_utilization(%)`: Cube利用率

**纯kernel耗时 = `Avg Time(us)` 或 `aicore_time(us)`**

**注意**：
- 首次执行通常耗时较长（包含kernel编译和初始化）
- 稳定后的执行时间更能反映真实性能
- 建议关注`Avg Time(us)`作为基准指标

### 步骤5：性能数据分析

**核心指标计算**：
```
单次纯kernel耗时 = Avg Time(us)
总kernel耗时 = Count × Avg Time(us)
性能波动 = (Max Time - Min Time) / Avg Time × 100%
```

**性能评估标准**：

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|------|------|------|------|------|
| 平均耗时 | <30μs | 30-100μs | 100-500μs | >500μs |
| 性能波动 | <5% | 5-10% | 10-20% | >20% |
| Cube利用率 | >80% | 60-80% | 40-60% | <40% |

**常见问题分析**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 首次执行耗时过长（Max Time >> Avg Time） | kernel编译和初始化开销 | 增加warmup迭代次数 |
| 性能波动大（Max - Min > 20% Avg） | 系统负载不稳定、调度延迟 | 增加迭代次数、隔离测试环境 |
| Cube利用率低（<40%） | 内存访问瓶颈、任务粒度不合适 | 优化tiling策略、调整block配置 |

### 步骤6：生成性能报告

**报告模板**：
```markdown
## AscendC算子性能测试报告

### 测试配置
- **算子名称**: {operator_name}
- **参数配置**:
  - B (Batch): {B}
  - N (Heads): {N}
  - Sq (Query长度): {Sq}
  - Skv (KV长度): {Skv}
  - D (维度): {D}
  - 数据类型: {dtype}

### 性能指标

| 指标 | 数值 | 单位 |
|------|------|------|
| 执行次数 | {Count} | 次 |
| 平均耗时 | {Avg Time} | μs |
| 最小耗时 | {Min Time} | μs |
| 最大耗时 | {Max Time} | μs |
| 总耗时 | {Total Time} | μs |
| Cube利用率 | {cube_utilization} | % |

### 性能分析
- **纯kernel耗时**: {Avg Time} μs
- **性能波动**: {波动率}%
- **性能评级**: {评级}

### 数据文件位置
- 原始数据: {profiling_output_dir}
- 算子统计: {op_statistic_csv}
- 详细数据: {op_summary_csv}
```

**报告输出**：将报告保存至算子目录 `{operator_path}/KERNEL_PERFORMANCE_REPORT.md`

## 4. PyPTO与AscendC性能对比报告

```markdown
## Kernel性能测试报告

### 测试配置
- **算子名称**: {operator_name}
- **参数配置**: B={B}, N={N}, Sq={Sq}, Skv={Skv}, D={D}

### 性能对比

| 实现 | 平均耗时(μs) | 最小耗时(μs) | 最大耗时(μs) | 性能波动 |
|------|-------------|-------------|-------------|---------|
| PyPTO | {pypto_avg} | {pypto_min} | {pypto_max} | {pypto_var}% |
| AscendC | {ascendc_avg} | {ascendc_min} | {ascendc_max} | {ascendc_var}% |
| 性能比 | {ratio}x | - | - | - |

### 数据文件
- PyPTO: {pypto_log_path}
- AscendC: {msprof_output_path}
```

## 5. 最佳实践

**测试环境准备**：
- 确保NPU设备空闲，避免其他任务干扰
- 设置合适的环境变量
- 预热NPU设备（执行一次简单操作）

**参数配置建议**：
- 迭代次数：100-1000次（根据算子复杂度）
- warmup次数：5-10次
- 数据规模：覆盖典型使用场景

**数据采集优化**：
- 使用`--task-time=l2`获取详细task时间
- 关注`Avg Time(us)`而非单次执行时间
- 分析`op_summary`中的详细指标

**结果验证**：
- 对比多次测试结果，确保稳定性
- 检查性能波动是否在合理范围
- 验证Cube利用率是否符合预期

## 6. 常见问题

**Q1: msprof命令找不到**
```bash
# 检查CANN安装
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证msprof可用
which msprof
```

**Q2: 首次执行耗时过长**
- 增加warmup迭代次数（10次以上）
- 首次执行结果不计入统计

**Q3: 性能数据为空**
- 检查算子是否真正在NPU上执行
- 确认`torch.npu.synchronize()`已调用
- 检查msprof输出目录权限

**Q4: 如何获取更细粒度的kernel时间**
- 使用`--task-time=l3`获取更详细的task trace
- 查看`task_time_*.csv`文件
- 分析单个task的执行时间