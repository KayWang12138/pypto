# 性能参数规则

> **版本**: 1.0
> **最后更新**: 2026-03-15
> **说明**: 本文件用于”设计期可行性闸门”。默认值必须以 API 文档为准；调优建议必须标注为 HEURISTIC 并给出证据锚点。
> - **HARD**：不满足会直接导致编译失败或运行错误。
> - **HEURISTIC**：经验建议/调优起点，允许按实测调整。

---

## 目录

1. [开箱性能配置（HEURISTIC）](#1-开箱性能配置heuristic)
2. [pass_options 配置（HARD + HEURISTIC）](#2-pass_options-配置hard--heuristic)
3. [runtime_options 配置（HARD + HEURISTIC）](#3-runtime_options-配置hard--heuristic)
4. [误配风险库](#4-误配风险库)
5. [非可视化验证路径](#5-非可视化验证路径)
6. [证据索引](#6-证据索引)

---

## 1. 开箱性能配置（HEURISTIC）

### 1.1 Loop 写法选择

- **静态轴**：使用 Python `for` 循环，避免 `pypto.loop` 造成 root function 过碎。
- **动态轴**：使用 `pypto.loop`，并合理配置 `pypto.view(..., valid_shape=...)`。
- **动态轴范围较广（例如 1~64k）**：考虑使用 `pypto.loop_unroll(..., unroll_list=[...])`，并注意档位数量会显著增加编译耗时。

**证据**: `docs/tutorials/debug/performance.md`

### 1.2 TileShape 初值（起点）

**Vector 初值**：

```python
pypto.set_vec_tile_shapes(64, 512)
```

**Matmul 初值（以 DT_BF16 / DT_FP16 为例）**：

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

**证据**: `docs/tutorials/debug/performance.md`

---

## 2. pass_options 配置（HARD + HEURISTIC）

### 2.1 参数清单与默认值（以 API 文档为准）

以下默认值来自 `pypto.set_pass_options` 的参数说明：

| 参数 | 含义（摘要） | 默认值 |
|------|--------------|--------|
| `vec_nbuffer_setting` | AIV 同构子图合并粒度 | `{}` |
| `cube_l1_reuse_setting` | 重复 L1 搬运子图合并（L1Reuse） | `{}` |
| `cube_nbuffer_setting` | AIC 同构子图合并粒度（CubeNBuffer） | `{-1: 1}` |
| `sg_set_scope` | 手动控制合图 scopeId | `-1` |

**证据**: `docs/api/config/pypto-set_pass_options.md`

### 2.2 cube_l1_reuse_setting（手动覆盖，谨慎使用）

- **Level**: HEURISTIC
- **适用场景**: Cube 子图间存在冗余 L1 搬运，且希望手动控制合并粒度。
- **建议**: 通常考虑 `2/4/8` 等值，并结合泳道图实测择优。
- **示例**:

```python
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)
def kernel(...):
    ...
```

**证据**: `docs/tutorials/debug/performance.md`

### 2.3 cube_nbuffer_setting（替代策略）

- **Level**: HEURISTIC
- **适用场景**: 无法使能 L1Reuse 的少数场景（例如：Cube 子图间没有重复 L1 搬运；或 K 轴很长且未切 K）。
- **证据**: `docs/tutorials/debug/performance.md`

### 2.4 vec_nbuffer_setting（AIV 子图过碎时）

- **Level**: HEURISTIC
- **适用场景**: 泳道图里存在大量耗时很短的同构 AIV 子图（例如 10us 级），希望减少调度/头开销。
- **证据**: `docs/tutorials/debug/performance.md`

---

## 3. runtime_options 配置（HARD + HEURISTIC）

### 3.1 参数清单与默认值（以 API 文档为准）

以下默认值来自 `pypto.set_runtime_options` 的参数说明：

| 参数 | 含义（摘要） | 默认值 |
|------|--------------|--------|
| `device_sched_mode` | 调度模式（0/1/2/3） | `0` |
| `stitch_function_max_num` | 每次 stitch 处理的最大 loop 个数 | `128` |
| `run_mode` | 执行设备（0=NPU，1=SIM） | “根据是否设置 cann 环境变量决定” |
| `valid_shape_optimize` | validshape 编译优化（0/1） | `0` |

**证据**: `docs/api/config/pypto-set_runtime_options.md`

### 3.2 device_sched_mode（L2 亲和调度）

- **Level**: HEURISTIC
- **适用场景**: 上下游依赖较简单、或下游输入 Tensor 的 L2 命中率更关键时。
- **示例**:

```python
pypto.set_runtime_options(device_sched_mode=1)
```

**证据**: `docs/tutorials/debug/performance.md`

### 3.3 stitch_function_max_num（Stitch 调优旋钮）

- **Level**: HEURISTIC
- **建议**: 在内存资源允许的前提下，可逐步增大该值，结合泳道图与端到端耗时找到平衡点。
- **示例**:

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
def kernel(...):
    ...
```

**证据**: `docs/tutorials/debug/performance.md`

### 3.4 run_mode（NPU / SIM）

- **Level**: HARD
- **规则**:
  - `run_mode` 取值含义在 API 文档中定义为：0 表示 NPU，1 表示模拟器。
  - 运行时代码会将 `runtime_options["run_mode"]` 规范化为 int，并在请求 NPU 但未 source CANN 环境时抛出异常。

**证据**:
- `docs/api/config/pypto-set_runtime_options.md`
- `python/pypto/runtime.py`

**示例**（IntEnum 值可作为 int 使用）：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def kernel(...):
    ...
```

---

## 4. 误配风险库

常见性能参数误配场景，每条含触发条件、后果和规避措施。

| warning_code | trigger | impact | mitigation | evidence |
|--------------|---------|--------|------------|----------|
| PERF_RUN_MODE_ENV_MISMATCH | run_mode=0(NPU) 但未 source CANN 环境 | 运行时异常退出 | 确认环境配置后再设 run_mode | `docs/api/config/pypto-set_runtime_options.md`; `docs/install/prepare_environment.md` |
| PERF_PASS_OPTION_CONFLICT | cube_l1_reuse 与 cube_nbuffer 同时激进配置 | 编译失败或性能下降 | 回退冲突参数，最小集重试 | `docs/api/config/pypto-set_pass_options.md` |
| PERF_OVERTUNE_STABILITY_RISK | 多个 pass/runtime 参数同时偏离默认值 | 稳定性下降、结果不可复现 | 回退至 baseline 再逐项启用 | `docs/tutorials/debug/performance.md` |
| PERF_STITCH_WORKSPACE_RISK | stitch_function_max_num 过大 | workspace 内存占用显著增加 | 从 128 起步，按内存余量逐步增大 | `docs/api/config/pypto-set_runtime_options.md` |
| PERF_DEVICE_SCHED_MODE_MISUSE | 使用高级调度模式（2/3）但场景不匹配 | 调度管理开销反超收益 | 默认使用 0，仅 L2 亲和场景尝试 1 | `docs/api/config/pypto-set_runtime_options.md` |
| PERF_VALID_SHAPE_OPT_MISUSE | valid_shape_optimize=1 但动态 shape 主块占比低 | 编译复杂度增加但收益小 | 仅动态 shape 且主块占比高时开启 | `docs/api/config/pypto-set_runtime_options.md` |
| PERF_PASS_SCOPE_MISUSE | sg_set_scope 值设置不当 | 阻断合理合图 | 保持默认 -1，仅按需设置统一 scope | `docs/api/config/pypto-set_pass_options.md` |
| PERF_SUBSTITUTE_PRECISION_RISK | 多层 substitute 组合（如 gelu/tanh）精度累积 | 数值误差超出 atol/rtol | 优先使用 direct API；substitute 链>3 层时增加精度验证 | 参见 `api_mapping.md` §4 配方库 |
| PERF_SUBSTITUTE_PERF_RISK | substitute 组合调用链过长（>4 次 API 调用） | 性能显著低于 direct API | 评估是否可简化组合或寻找替代方案 | 参见 `api_mapping.md` §4 配方库 |

---

## 5. 非可视化验证路径

在不依赖泳道图的情况下，通过以下可执行检查验证性能参数配置的正确性：

| 检查项 | 方法 | 判定 |
|--------|------|------|
| 编译日志检查 | 编译后搜索 pass/runtime 相关错误签名 | 无错误签名 → PASS |
| 运行模式检查 | 确认 run_mode 与 CANN 环境变量一致 | 一致 → PASS |
| 数值检查 | 关键输出与 golden 对比，误差在 atol/rtol 内 | 在阈值内 → PASS |
| 参数范围检查 | 各参数值在文档 allowed_values 范围内 | 在范围内 → PASS |
| baseline 对比 | 修改参数前后端到端耗时对比 | 有改善 → PASS |

---

## 6. 证据索引

| 证据文件 | 内容 |
|----------|------|
| `docs/api/config/pypto-set_pass_options.md` | pass_options 参数定义与默认值 |
| `docs/api/config/pypto-set_runtime_options.md` | runtime_options 参数定义与默认值 |
| `docs/tutorials/debug/performance.md` | 性能调优教程、开箱配置建议 |
| `docs/install/prepare_environment.md` | 环境配置与设备准备 |
| `python/pypto/runtime.py` | run_mode 规范化逻辑代码级证据 |
