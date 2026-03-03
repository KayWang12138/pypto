---
name: pypto-performance-autotuner
description: PyPTO 性能自动调优技能 — 搜索最优 NPU 性能配置。完整覆盖 tile/pass/runtime/stitch 全部调优旋钮，按 PR#1263 优先级排序（stitch → matmul → vector → scheduling），结合泳道图性能数据驱动分层搜索策略。触发词：性能调优、自动调优、autotune、set_pass_options、set_runtime_options、tile shapes、搜索最优配置、性能搜索
license: Apache-2.0
---

# PyPTO Performance Autotuner

搜索最优 NPU 性能调优配置的知识技能。覆盖 PyPTO 全部调优 API，按影响度排序，结合泳道图性能数据驱动分层搜索。

## 触发场景

- "自动调优" / "性能调优" / "autotune" / "搜索最优配置"
- "set_pass_options 调优" / "set_runtime_options 调优"
- "tile shapes 优化" / "stitch 调优" / "matmul 调优"
- "性能瓶颈分析" / "泳道图分析"

## 重要约束

⚠️ **所有调优必须在 NPU 模式执行** — SIM 仅用于结构验证，不可作为性能基准。

---

## 一、调优旋钮完整目录

> 来源: `pypto/python/pypto/config.py` + PR #1263 `performance.md`

### 1.1 Cube Tile Shapes（Matmul 切块）

```python
pypto.set_cube_tile_shapes(
    m: List[int],    # [L0, L1] — Matmul M 维度切分
    k: List[int],    # [L0, L1] 或 [L0, L1_A, L1_B] — K 维度切分
    n: List[int],    # [L0, L1] — N 维度切分
    enable_multi_data_load: bool = False,  # L1 大包搬运（提升数据搬运效率）
    enable_split_k: bool = False,          # 多核切 K（K 轴并行化）
)
```

**参数语义**:

| 参数 | 含义 | 推荐值 | 约束 |
|------|------|--------|------|
| `m` | M 维度 [L0 tile, L1 tile] | `[16, 128]`, `[16, 256]` | L0 通常固定 16；L1 为 16 的倍数 |
| `k` | K 维度 | `[16, 128]`, `[16, 64, 128]` | 2 元素=[L0,L1]；3 元素=[L0, L1_A, L1_B] 分别对应两个输入矩阵 |
| `n` | N 维度 | `[16, 128]`, `[16, 256]` | 同 m |
| `enable_multi_data_load` | L1 大包搬运 | `True` | 提升带宽利用率；推荐默认开启 |
| `enable_split_k` | 多核切 K | `False`/`True` | 当 K 轴很长且 M/N 核数不足时开启 |

**推荐 Tilesize 组合**（来自 PR #1263）:

```python
# 方阵场景
pypto.set_cube_tile_shapes(m=[16, 128], k=[16, 128], n=[16, 128])

# 大矩阵场景
pypto.set_cube_tile_shapes(m=[16, 256], k=[16, 64], n=[16, 256],
                           enable_multi_data_load=True)

# K 轴很长场景 — 开启 split_k
pypto.set_cube_tile_shapes(m=[16, 128], k=[16, 128], n=[16, 128],
                           enable_split_k=True)
```

### 1.2 Vector Tile Shapes（向量切块）

```python
pypto.set_vec_tile_shapes(*args: int)  # 变长参数，对应张量各维度的 tile size
```

**参数语义**:

| 参数 | 含义 | 推荐值 | 约束 |
|------|------|--------|------|
| `*args` | 各维度 tile size | 按 16-64 KB 原则设定 | 需与上下游算子 tile 对齐 |

**调优要点**（来自 PR #1263）:
- **16-64 KB 原则**: 单次搬运数据量在 16-64 KB 范围内效率最高
- **上下游对齐**: Vector tile 应与上游 Cube 输出 tile 或下游算子输入 tile 对齐
- **归约轴不切分**: 如果某轴是归约轴，tile size 应等于该轴完整大小
- **泳道图观察**: 看并行度是否充分，若核间负载不均则需调整 tile

### 1.3 Pass Options（编译优化选项）

```python
pypto.set_pass_options(
    # === 切图控制 ===
    pg_skip_partition: bool,           # 跳过子图切分（用于全手动控制）
    pg_upper_bound: int,               # 子图大小上界
    pg_lower_bound: int,               # 子图大小下界（低于此值会被合并）
    pg_parallel_lower_bound: int,      # 同结构子图最低并行度
    sg_set_scope: int,                 # 手动指定 scope（-1 重置）

    # === AIV (Vector) 融合策略 ===
    mg_vec_parallel_lb: int,           # AIV 子图最低并行度
    vec_nbuffer_mode: int,             # AIV 广度合并策略 (0/1/2)
    vec_nbuffer_setting: Dict[int,int], # AIV 合并数量 {scope_id: count}

    # === AIC (Cube) 融合策略 ===
    cube_l1_reuse_mode: int,           # L1 复用合并策略 (0/1)
    cube_l1_reuse_setting: Dict[int,int], # L1 复用合并数量
    cube_nbuffer_mode: int,            # AIC 广度合并策略
    cube_nbuffer_setting: Dict[int,int], # AIC 合并数量

    # === 搬运控制 ===
    mg_copyin_upper_bound: int,        # 合并子图最大搬入数据量
)
```

**参数详解**:

| 参数 | 类型 | 典型值 | 语义 |
|------|------|--------|------|
| `pg_skip_partition` | bool | `True` | 跳过自动切图，全手动控制子图 |
| `pg_upper_bound` | int | 200, 500, 1000 | 子图最大算子数；增大→更大子图→更多融合机会 |
| `pg_lower_bound` | int | 128, 256, 512 | 子图最小算子数；增大→更激进合并 |
| `pg_parallel_lower_bound` | int | 8, 16, 20, 24 | 同结构子图并行下界 |
| `sg_set_scope` | int | 0, 1, 2, ... (-1=reset) | **手动 scope 指派** — 相同 scope 值的算子融合在一起 |
| `mg_vec_parallel_lb` | int | 4, 8 | AIV 并行下界 |
| `vec_nbuffer_mode` | int | 0, 1, 2 | 0=关闭, 1=按深度, 2=按广度 |
| `vec_nbuffer_setting` | Dict | `{0: 2}`, `{0: 4}` | `{scope_id: 合并数}` |
| `cube_l1_reuse_mode` | int | 0, 1 | 0=关闭, 1=启用 L1 复用 |
| `cube_l1_reuse_setting` | Dict | `{0: 4}` | `{scope_id: 合并数}` |
| `cube_nbuffer_mode` | int | 0, 1, 2 | AIC 合并策略 |
| `cube_nbuffer_setting` | Dict | `{0: 2}` | AIC 合并数量 |
| `mg_copyin_upper_bound` | int | 1024, 2048 | 合并子图最大搬入数据 (bytes) |

**关键互斥/依赖关系**:
- `cube_l1_reuse_mode` 与 `cube_nbuffer_mode` 互斥 — **L1 Reuse 优先**（PR #1263 明确推荐）
- `sg_set_scope` 是最细粒度控制，设置后覆盖自动切图结果
- `pg_skip_partition=True` 时，`pg_upper_bound`/`pg_lower_bound` 无效

### 1.4 Runtime Options（运行时选项）

```python
pypto.set_runtime_options(
    # === 调度策略 ===
    device_sched_mode: int,            # 0=默认, 1=L2亲和, 2=公平调度

    # === Stitch 调优 ===
    stitch_function_inner_memory: int,     # 非 outcast 内存池大小
    stitch_function_outcast_memory: int,   # workspace 内存评估量
    stitch_function_num_initial: int,      # 初始 stitch task 计算量
    stitch_function_num_step: int,         # 非初始 stitch 循环计算量
    stitch_function_size: int,             # 每个 loop 最大 CallOp 计算量
    stitch_cfgcache_size: int,             # 控制流缓存大小 (bytes)

    # === 其他 ===
    triple_stream_sched: bool,         # 三流调度
    run_mode: int,                     # 运行模式
    valid_shape_optimize: bool,        # shape 优化开关
)
```

**Stitch 调优（最高影响度）**:

| 参数 | 典型值 | 语义 | 调优方向 |
|------|--------|------|----------|
| `stitch_function_num_initial` | 16, 32, 64, 128 | 初始并发 stitch task 数 | 增大→更多并发→可能争抢资源 |
| `stitch_function_outcast_memory` | 256, 512, 1024 | workspace 内存 | 增大→更多临时缓存 |
| `stitch_function_inner_memory` | 256, 512 | 非 outcast 内存 | 与 outcast_memory 配合 |
| `device_sched_mode` | 0, 1 | 调度模式 | 1=L2亲和调度，通常优于默认 |

> ⚠️ **已规划替换**: `stitch_function_inner_memory`、`stitch_function_outcast_memory`、`stitch_function_num_initial` 三个参数未来将统一为 `stitch_function_max_num`。当前两套 API 并存。

### 1.5 Loop Unroll（循环展开）

```python
pypto.loop_unroll(
    *args,                        # 循环变量
    name: str,                    # 循环名称
    idx_name: str,                # 索引变量名
    unroll_list: List[int],       # 展开值列表
    submit_before_loop: bool,     # 循环前提交（解决数据依赖顺序问题）
)
```

**关键用法**（来自 PR #1263）:
- 动态范围很宽时用 `loop_unroll` 替代 Python for + if-else
- `submit_before_loop=True` 用于存在数据依赖的场景，确保前序任务先提交
- 每个 unroll level 可以配不同的 tile shape

### 1.6 Debug Options

```python
pypto.set_debug_options(
    compile_debug_mode: int,       # 编译调试等级
    runtime_debug_mode: int,       # 0=关闭, 1=基础, 2=详细
)
```

### 1.7 Experimental APIs

```python
pypto.experimental.set_operation_options(
    combine_axis: bool,            # 轴合并优化
)
```

---

## 二、调优优先级（来自 PR #1263）

> **按影响度从高到低排序，优先搜索高影响旋钮。**

```
优先级 1: Stitch 调优 ← 影响最大，控制并行任务调度
    │
    ├── stitch_function_num_initial
    ├── stitch_function_outcast_memory
    ├── stitch_function_inner_memory
    └── submit_before_loop (loop_unroll)
    │
优先级 2: Matmul 调优 ← Cube 算子性能核心
    │
    ├── set_cube_tile_shapes (m, k, n)
    ├── cube_l1_reuse_mode / cube_l1_reuse_setting  ← L1 Reuse 优先于 CubeNBuffer
    ├── cube_nbuffer_mode / cube_nbuffer_setting
    ├── enable_multi_data_load ← L1 大包搬运
    └── enable_split_k ← 多核切 K
    │
优先级 3: Vector 调优 ← AIV 算子性能
    │
    ├── set_vec_tile_shapes ← 与上下游 tile 对齐
    ├── sg_set_scope ← 手动融合控制
    ├── pg_upper_bound / pg_lower_bound ← 切图粒度
    ├── vec_nbuffer_mode / vec_nbuffer_setting ← 广度合并
    └── mg_vec_parallel_lb
    │
优先级 4: 调度策略 ← 全局调度优化
    │
    └── device_sched_mode = 1 (L2 亲和调度)
```

---

## 三、泳道图性能数据获取

> 泳道图（Swimlane Diagram）是评估配置好坏的**唯一可靠依据**。

### 3.1 采集方式

**方式 A: torch_npu.profiler（推荐）**

```python
import torch_npu
from torch_npu import profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
    schedule=profiler.schedule(wait=1, warmup=5, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler("./profiler_output"),
    profile_memory=True,
) as prof:
    for step in range(total_steps):
        run_operator(input_data)
        prof.step()
```

**方式 B: MindSpore Profiler**

```python
from mindspore.profiler import Profiler, ProfilerActivity

with mindspore.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    on_trace_ready=mindspore.profiler.tensorboard_trace_handler("./data"),
) as prof:
    run_operator(input_data)
    prof.step()
```

**方式 C: PyPTO 内置 profiling**

```python
# 通过 machine 模块配置
# enable_prof_func=True        # Function 级别
# enable_prof_aicore_time=True  # AI Core 时间
# enable_prof_aicore_pmu=True   # AI Core PMU
```

### 3.2 输出数据结构

```
profiler_output/
├── trace_view.json              # 泳道图时间线数据 (Chrome Trace Format)
├── op_statistic.csv             # 算子执行时间统计
├── kernel_details.csv           # NPU 算子详情
├── step_trace_time.csv          # 迭代级时间统计
├── memory_record.csv            # 内存使用记录
└── ascend_profiler_*.db         # SQLite 完整数据
```

### 3.3 关键指标提取

Agent 需要实现或调用一个**性能数据提取脚本**，从泳道图数据中提取以下指标：

```python
def extract_metrics(profiler_output_dir: str) -> dict:
    """从 profiler 输出提取关键性能指标。

    Returns:
        {
            "total_latency_ms": float,      # 端到端延迟 (ms)
            "kernel_time_ms": float,        # NPU kernel 总执行时间
            "idle_ratio": float,            # NPU 空闲比例 (0-1)
            "memory_peak_mb": float,        # 内存峰值 (MB)
            "op_count": int,                # 算子数量
            "top_ops": [                    # 耗时 Top-N 算子
                {"name": str, "time_ms": float, "count": int}
            ],
            "parallelism_score": float,     # 并行度评分 (0-1)
        }
    """
```

**提取方法**:

| 数据源 | 提取方式 | 用途 |
|--------|----------|------|
| `trace_view.json` | JSON 解析 Chrome Trace Events | 时间线分析、空闲检测、并行度 |
| `op_statistic.csv` | pandas 读取 | 算子级耗时排序 |
| `kernel_details.csv` | pandas 读取 | kernel 级别详情 |
| `memory_record.csv` | pandas 读取 | 内存峰值分析 |

**trace_view.json 格式**:

```json
{
  "traceEvents": [
    {
      "pid": 1234,
      "tid": 1,
      "name": "MatMul-op42",
      "ph": "X",
      "ts": 1000000,
      "dur": 15000,
      "args": {"op_type": "MatMul", "stream_id": 5}
    }
  ]
}
```

**泳道图核心观察点**:
1. **NPU 空闲间隙** → Stitch 调优不够，增大 `stitch_function_num_initial`
2. **核间负载不均** → Tile size 不合理，调整切块使各核负载均衡
3. **数据搬运瓶颈** → 开启 `enable_multi_data_load`，调大 `mg_copyin_upper_bound`
4. **算子串行执行** → 检查 `sg_set_scope` 是否正确融合，调整 `vec_nbuffer_mode`

### 3.4 评估协议

每次配置评估必须遵循：

1. **Warmup**: 执行 N 次预热（推荐 N≥5），丢弃结果
2. **NPU Synchronize**: 每次测量前后调用 `torch_npu.npu.synchronize()`
3. **多次采样**: 至少测量 M 次（推荐 M≥10）
4. **Outlier Removal**: IQR 1.5× 规则剔除离群值
5. **Aggregation**: 取中位数作为最终指标
6. **显著性门槛**: 新配置必须优于当前 best 至少 2%（`new < best × 0.98`）

---

## 四、搜索策略

### 4.1 分层搜索框架

> 按优先级分层搜索，高影响旋钮先搜索，低影响旋钮在高层最优基础上微调。

```
┌─────────────────────────────────────────────────────┐
│ 第 1 层: Stitch 调优 (最高影响)                       │
│                                                       │
│   搜索空间:                                           │
│     stitch_function_num_initial: [16, 32, 64, 128]   │
│     stitch_function_outcast_memory: [256, 512, 1024] │
│     stitch_function_inner_memory: [256, 512]         │
│     ⚠️ 以上三参数将统一为 stitch_function_max_num      │
│   策略: Grid Search (空间小，≤24 组合)                │
│   评估: 泳道图 total_latency_ms + idle_ratio          │
│   输出: best_stitch_config                            │
├─────────────────────────────────────────────────────┤
│ 第 2 层: Matmul Tile + L1 策略 (次高影响)             │
│   固定: best_stitch_config                            │
│                                                       │
│   搜索空间:                                           │
│     cube_tile m: [[16,64], [16,128], [16,256]]       │
│     cube_tile k: [[16,64], [16,128]]                 │
│     cube_tile n: [[16,64], [16,128], [16,256]]       │
│     enable_multi_data_load: [True, False]            │
│     enable_split_k: [True, False]                    │
│     cube_l1_reuse_mode: [0, 1]                       │
│     cube_l1_reuse_setting: [{0:2}, {0:4}]            │
│                                                       │
│   策略: Random Search + 领域剪枝 (空间较大)           │
│          → 取 top-5 → Grid Search 精修               │
│   评估: kernel_time_ms                                │
│   输出: best_matmul_config                            │
├─────────────────────────────────────────────────────┤
│ 第 3 层: Vector Tile + 切图旋钮 (中等影响)            │
│   固定: best_stitch_config + best_matmul_config       │
│                                                       │
│   搜索空间:                                           │
│     vec_tile_shapes: [按 16-64KB 原则生成]            │
│     sg_set_scope: [0, 1, 2, ...]                     │
│     pg_upper_bound: [200, 500, 1000]                 │
│     pg_lower_bound: [128, 256, 512]                  │
│     vec_nbuffer_mode: [0, 1, 2]                      │
│                                                       │
│   策略: Bayesian Optimization (TPE)                   │
│   评估: total_latency_ms + parallelism_score          │
│   输出: best_vector_config                            │
├─────────────────────────────────────────────────────┤
│ 第 4 层: 调度策略 (最低影响)                          │
│   固定: 上述全部 best configs                         │
│                                                       │
│   搜索空间:                                           │
│     device_sched_mode: [0, 1, 2]                     │
│                                                       │
│   策略: Grid Search (仅 3 个值)                       │
│   评估: total_latency_ms                              │
│   输出: final_best_config                             │
└─────────────────────────────────────────────────────┘
```

### 4.2 搜索策略选择指南

| 策略 | 适用条件 | 优点 | 缺点 |
|------|----------|------|------|
| **Grid Search** | 参数空间 < 50 组合 | 保证找全局最优 | 维度灾难 |
| **Random Search** | 空间 50-500 组合 | 高维友好，易并行 | 无导向性 |
| **Bayesian (TPE)** | 空间 > 500 组合 | 样本效率最高 (4-7x) | 冷启动需 10+ 样本 |
| **分层 + 剪枝** | 多优先级旋钮 | 降维，避免组合爆炸 | 可能错过层间交互 |

**决策树**:

```
组合数
  ├── < 50 → Grid Search（穷举）
  ├── 50-500 → Random Search + 领域剪枝 → top-K Grid 精修
  └── > 500 → Bayesian (TPE/Optuna) + Early Stopping
```

### 4.3 领域剪枝规则

以下组合不合理，搜索时应剪枝：

1. `cube_l1_reuse_mode=1` 且 `cube_nbuffer_mode≠0` → **互斥**，L1 Reuse 优先
2. `enable_split_k=True` 且 K 轴 tile 很小 → **无意义**，split_k 需要 K 轴足够长
3. `pg_skip_partition=True` 且设置了 `pg_upper_bound/pg_lower_bound` → **冲突**，skip 后 bound 无效
4. `vec_nbuffer_mode=0` 且设置了 `vec_nbuffer_setting` → **无效**，mode=0 关闭合并
5. `stitch_function_num_initial` 过大（>256）→ **资源争抢**，通常 16-128 范围

### 4.4 Early Stopping

- 连续 **5 次试验** 无改进（未超过 2% 门槛）→ 提前结束当前层
- 当前层最优配置已达到理论性能上限的 95% → 跳过后续试验

---

## 五、搜索空间定义

### 5.1 搜索空间 JSON Schema

```json
{
  "name": "operator_name",
  "description": "搜索空间描述",
  "layers": [
    {
      "name": "stitch",
      "priority": 1,
      "strategy": "grid",
      "params": {
        "stitch_function_num_initial": {"type": "choice", "values": [16, 32, 64, 128]},
        "stitch_function_outcast_memory": {"type": "choice", "values": [256, 512, 1024]},
        "stitch_function_inner_memory": {"type": "choice", "values": [256, 512]}
      }
    },
    {
      "name": "matmul",
      "priority": 2,
      "strategy": "random+grid",
      "max_trials": 50,
      "params": {
        "cube_tile_m": {"type": "choice", "values": [[16,64], [16,128], [16,256]]},
        "cube_tile_k": {"type": "choice", "values": [[16,64], [16,128]]},
        "cube_tile_n": {"type": "choice", "values": [[16,64], [16,128], [16,256]]},
        "enable_multi_data_load": {"type": "bool"},
        "enable_split_k": {"type": "bool"},
        "cube_l1_reuse_mode": {"type": "choice", "values": [0, 1]},
        "cube_l1_reuse_setting": {"type": "choice", "values": [null, {"0": 2}, {"0": 4}]}
      },
      "pruning_rules": [
        "cube_l1_reuse_mode==1 → cube_nbuffer_mode must be 0"
      ]
    },
    {
      "name": "vector",
      "priority": 3,
      "strategy": "bayesian",
      "max_trials": 100,
      "params": {
        "sg_set_scope": {"type": "range", "min": -1, "max": 5},
        "pg_upper_bound": {"type": "choice", "values": [200, 500, 1000]},
        "pg_lower_bound": {"type": "choice", "values": [128, 256, 512]},
        "vec_nbuffer_mode": {"type": "choice", "values": [0, 1, 2]}
      }
    },
    {
      "name": "scheduling",
      "priority": 4,
      "strategy": "grid",
      "params": {
        "device_sched_mode": {"type": "choice", "values": [0, 1, 2]}
      }
    }
  ]
}
```

### 5.2 配置应用模板

```python
import pypto

def apply_config(config: dict):
    """将搜索到的配置应用到 PyPTO。"""

    # Stitch 层
    if "stitch" in config:
        sc = config["stitch"]
        pypto.set_runtime_options(
            stitch_function_num_initial=sc.get("stitch_function_num_initial"),
            stitch_function_outcast_memory=sc.get("stitch_function_outcast_memory"),
            stitch_function_inner_memory=sc.get("stitch_function_inner_memory"),
        )

    # Matmul 层
    if "matmul" in config:
        mc = config["matmul"]
        pypto.set_cube_tile_shapes(
            m=mc["cube_tile_m"],
            k=mc["cube_tile_k"],
            n=mc["cube_tile_n"],
            enable_multi_data_load=mc.get("enable_multi_data_load", False),
            enable_split_k=mc.get("enable_split_k", False),
        )
        if mc.get("cube_l1_reuse_mode"):
            pypto.set_pass_options(
                cube_l1_reuse_mode=mc["cube_l1_reuse_mode"],
                cube_l1_reuse_setting=mc.get("cube_l1_reuse_setting"),
            )

    # Vector 层
    if "vector" in config:
        vc = config["vector"]
        if vc.get("sg_set_scope") is not None:
            pypto.set_pass_options(sg_set_scope=vc["sg_set_scope"])
        if vc.get("pg_upper_bound"):
            pypto.set_pass_options(
                pg_upper_bound=vc["pg_upper_bound"],
                pg_lower_bound=vc.get("pg_lower_bound"),
            )
        if vc.get("vec_nbuffer_mode"):
            pypto.set_pass_options(vec_nbuffer_mode=vc["vec_nbuffer_mode"])

    # Scheduling 层
    if "scheduling" in config:
        pypto.set_runtime_options(
            device_sched_mode=config["scheduling"]["device_sched_mode"]
        )
```

---

## 六、实战工作流

### Step 1: 建立 Baseline

```python
# 不设任何调优参数，直接运行算子
baseline_metrics = run_and_profile(operator, input_data)
print(f"Baseline latency: {baseline_metrics['total_latency_ms']:.2f} ms")
```

### Step 2: 分析泳道图确定瓶颈

1. 用 Perfetto (https://ui.perfetto.dev/) 打开 `trace_view.json`
2. 观察泳道图，识别瓶颈类型:

| 现象 | 瓶颈类型 | 首选调优方向 |
|------|----------|-------------|
| 大段 NPU 空闲 | Stitch 不足 | 调大 `stitch_function_num_initial` |
| 核间负载不均 | Tile 不合理 | 调整 tile shapes |
| Cube 算子耗时长 | Matmul 效率低 | L1 Reuse + 大包搬运 |
| Vector 算子串行 | 融合不足 | `sg_set_scope` + `vec_nbuffer` |
| 数据搬运占比高 | 搬运瓶颈 | `enable_multi_data_load` + `mg_copyin_upper_bound` |

### Step 3: 定义搜索空间

根据瓶颈类型，只搜索相关层的旋钮，其他层保持默认或已知最优。

### Step 4: 执行分层搜索

按第四章的分层框架逐层搜索，每层固定上层最优结果。

### Step 5: 验证最终配置

```python
# 用最终 best config 多次运行验证
final_metrics = []
for _ in range(30):
    torch_npu.npu.synchronize()
    m = run_and_profile(operator, input_data, config=final_best_config)
    torch_npu.npu.synchronize()
    final_metrics.append(m)

# IQR 去除离群值后取中位数
verified_latency = median_with_iqr_filter(final_metrics)
speedup = baseline_metrics['total_latency_ms'] / verified_latency
print(f"Final: {verified_latency:.2f} ms, Speedup: {speedup:.2f}x")
```

### Step 6: 输出最优配置

```json
{
  "operator": "softmax_npu",
  "baseline_latency_ms": 12.5,
  "best_latency_ms": 8.3,
  "speedup": "1.51x",
  "best_config": {
    "stitch": {
      "stitch_function_num_initial": 64,
      "stitch_function_outcast_memory": 512
    },
    "matmul": {
      "cube_tile_m": [16, 128],
      "cube_tile_k": [16, 128],
      "cube_tile_n": [16, 128],
      "enable_multi_data_load": true,
      "cube_l1_reuse_mode": 1,
      "cube_l1_reuse_setting": {"0": 4}
    },
    "vector": {
      "sg_set_scope": 1,
      "pg_upper_bound": 500,
      "vec_nbuffer_mode": 1
    },
    "scheduling": {
      "device_sched_mode": 1
    }
  },
  "search_summary": {
    "total_trials": 87,
    "time_elapsed_min": 45,
    "layers_searched": ["stitch", "matmul", "vector", "scheduling"]
  }
}
```

---

## 七、实际代码示例

> 来自 PyPTO 仓库中的真实调优模式。

### 7.1 GLM Attention Fusion（sg_set_scope + L1 Reuse）

```python
# 来源: models/glm_v4_5/glm_attention_fusion.py
pypto.set_pass_options(
    sg_set_scope=0,                    # 手动 scope 0
    pg_upper_bound=200,
    cube_l1_reuse_setting={0: 4},       # scope 0 合并 4 个
)
```

### 7.2 DeepSeek MLA（大包搬运 + skip_partition）

```python
# 来源: models/deepseek_v4/hc_pre_impl.py
pypto.set_cube_tile_shapes(
    m=[16, 128], k=[16, 64], n=[16, 256],
    enable_multi_data_load=True,        # L1 大包搬运
)
pypto.set_pass_options(pg_skip_partition=True)  # 全手动切图
```

### 7.3 DeepSeek Compressor（vecNBuffer + copyin 控制）

```python
# 来源: models/deepseek_v32_exp/mla_indexer_prolog_quant_impl.py
pypto.set_pass_options(
    vec_nbuffer_mode=2,
    mg_copyin_upper_bound=2048,
    pg_upper_bound=1000,
    cube_l1_reuse_setting={0: 4},
)
```

---

## 八、参考资料

| 资源 | 说明 |
|------|------|
| PR #1263 `performance.md` | **主要参考** — PyPTO 性能调优权威指南 |
| `pypto/python/pypto/config.py` | 全部调优 API 签名定义 |
| Optuna (TPE Sampler) | 推荐的贝叶斯优化框架 |
| ytopt | HPC 专用贝叶斯优化框架 |
| Perfetto | 泳道图可视化工具 (https://ui.perfetto.dev/) |
| `references/search_space_schema.json` | 搜索空间 JSON Schema 参考 |
