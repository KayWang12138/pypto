# PyPTO 性能调优 Knob 字典

## 1. 数据采集开关

### debug_options
- **API**: `pypto.set_debug_options(compile_debug_mode=1, runtime_debug_mode=1)`
- **文档**: `docs/api/config/pypto-set_debug_options.md`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `compile_debug_mode` | 0 | 1=启用编译阶段调试（生成计算图 JSON） |
| `runtime_debug_mode` | 0 | 1=启用执行阶段调试（生成泳道图 JSON） |

## 2. Tiling 配置（高优先级）

### set_cube_tile_shapes
- **API**: `pypto.set_cube_tile_shapes(m, k, n, enable_multi_data_load, enable_split_k)`
- **文档**: `docs/api/config/pypto-set_cube_tile_shapes.md`
- **推荐**: BF16/FP16 Matmul 使用 `[128,128], [128,128], [128,128]`

### set_vec_tile_shapes
- **API**: `pypto.set_vec_tile_shapes(*args)`
- **文档**: `docs/api/config/pypto-set_vec_tile_shapes.md`
- **推荐**: 16KB 原则，使用 `(64, 512)`

## 3. Pass 编译优化参数

### set_pass_options
- **API**: `pypto.set_pass_options(...)`
- **文档**: `docs/api/config/pypto-set_pass_options.md`

#### 切分控制
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `pg_skip_partition` | False | bool | True=跳过子图切分 |
| `pg_lower_bound` | 512 | int | 子图大小下界 |
| `pg_upper_bound` | 10000 | int | 子图大小上界 |

#### 合并策略
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `pg_parallel_lower_bound` | 20 | int | 相同结构子图最小并行度 |
| `mg_vec_parallel_lb` | 48 | 1-48 | AIV子图最小并行度 |
| `vec_nbuffer_mode` | 1 | 0/1/2 | AIV子图合并策略 |
| `cube_l1_reuse_mode` | 1 | 0/1 | 重复搬运子图合并 |
| `cube_nbuffer_mode` | 0 | 0/1 | 相同结构AIC子图合并 |

## 4. Runtime 调度参数

### set_runtime_options
- **API**: `pypto.set_runtime_options(...)`
- **文档**: `docs/api/config/pypto-set_runtime_options.md`

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `device_sched_mode` | 0 | 0/1/2/3 | 0=默认, 1=L2亲和, 2=公平, 3=同时开启 |
| `stitch_function_max_num` | 128 | 1-128 | 每次 stitch 最大 loop 数 |
| `valid_shape_optimize` | 0 | 0/1 | 动态 shape 优化 |

### 已废弃参数（不要使用）
- `stitch_function_inner_memory` → 使用 `stitch_function_max_num` 替代
- `stitch_function_outcast_memory` → 使用 `stitch_function_max_num` 替代
- `stitch_function_num_initial` → 使用 `stitch_function_max_num` 替代

## 5. 调优顺序建议

1. **Tiling 优先**: 先设置 `set_cube_tile_shapes` 和 `set_vec_tile_shapes`
2. **Pass 优化**: 调整 `cube_l1_reuse_mode`、`vec_nbuffer_mode`
3. **Runtime 调度**: 尝试不同 `device_sched_mode`
4. **精细控制**: 使用 `*_setting` 字典进行细粒度调整

## 6. MR 1263 变更

MR 1263 包含以下文档更新：
- `docs/tutorials/debug/performance.md` - 性能调优教程更新
- `docs/api/controlflow/pypto-loop_unroll.md` - 循环展开文档
