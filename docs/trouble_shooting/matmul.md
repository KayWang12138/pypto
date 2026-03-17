# MATMUL 组件错误码

- **范围**：FC3XXX - FC5XXX
- 本文档说明 MATMUL 子类 OP 的错误码定义、场景说明与排查建议。

## 错误码定义与使用说明

Matmul相关错误码的统一定义，参见 `/framework/src/interface/utils/matmul_error.h` 文件。

## 错误码定义和场景说明

### 1. PARAM（参数校验，`MatmulErrorCode::PARAM`，FC3xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `ERR_PARAM_INVALID` | **FC3000** | `matmul.pre.param.check` | 矩阵乘法输入参数无效（shape、dtype、format 等非法）。 |
| `ERR_PARAM_MISMATCH` | **FC3001** | `matmul.pre.param.match` | 输入参数维度/类型不匹配（K 轴长度不一致、数据类型不一致等）。 |
| `ERR_PARAM_UNSUPPORTED` | **FC3002** | `matmul.pre.param.support` | 输入参数为不支持的配置（非法数据格式、特殊维度组合等）。 |

### 2. CONFIG（配置校验，`MatmulErrorCode::CONFIG`，FC4xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `ERR_CONFIG_TILE` | **FC4000** | `matmul.pre.config.tile` | Tile 分块配置错误（分块大小非法、分块维度超出限制）。 |
| `ERR_CONFIG_ALIGNMENT` | **FC4001** | `matmul.pre.config.align` | 内存/数据对齐错误（未满足 16B/32B/64 元素对齐要求）。 |
| `ERR_CONFIG_UNSUPPORTED` | **FC4002** | `matmul.pre.config.combo` | 不支持的配置组合（tile+数据类型+格式组合非法）。 |

### 3. RUNTIME（运行时异常，`MatmulErrorCode::RUNTIME`，FC5xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `ERR_RUNTIME_NULLPTR` | **FC5000** | `matmul.runtime.ptr.check` | 空指针错误（张量地址、上下文、配置句柄为空）。 |
| `ERR_RUNTIME_STATE` | **FC5001** | `matmul.runtime.state.check` | 内部状态错误（执行流程状态异常、未初始化就运行）。 |
| `ERR_RUNTIME_LOGIC` | **FC5002** | `matmul.runtime.logic.check` | 逻辑不变量错误（计算流程分支异常、预期外执行路径）。 |

---

## 错误类型说明
### FC3000 ERR_PARAM_INVALID
1. **检查张量基础信息**：确认输入输出张量的维度、数据类型、格式均为合法有效值。
2. **检查维度合法性**：确认矩阵维度为正整数，无零维、负维度等非法情况。
3. **检查参数完整性**：确认 Matmul 接口必填参数均已传入，无缺失、无越界。
4. **查日志上下文**：通过 MATMUL_LOGE 日志查看参数详情，定位非法字段。

### FC3001 ERR_PARAM_MISMATCH
1. **检查输入矩阵维度**：确认 A 矩阵 K 维度与 B 矩阵 K 维度长度一致。
2. **检查数据类型**：A、B、C 矩阵数据类型与算力核要求类型匹配。
3. **检查转置配置**：确认 transposeA / transposeB 配置与实际内存排布一致。
4. **查日志上下文**：通过 MATMUL_LOGE 日志打印 shape 信息，定位不匹配项。

### FC3002 ERR_PARAM_UNSUPPORTED
1. **检查数据格式**：确认使用 NPU 支持的数据格式（如 ND、FRACTAL_Z 等）。
2. **检查数据类型**：确认不包含当前硬件不支持的低精度/高精度类型。
3. **检查维度组合**：确认 batch、M/N/K 维度未超出硬件支持上限。
4. **查日志上下文**：查看不支持的参数类型，切换为兼容配置重试。

### FC4000 ERR_CONFIG_TILE
1. **检查 Tile 分块参数**：确认 M/N/K 分块大小在硬件支持范围内。
2. **检查分块合理性**：确认分块大小可整除对应维度，无非法零值。
3. **检查分块策略**：确认使用官方推荐的分块组合，无自定义非法分块。
4. **查日志上下文**：通过日志获取非法 tile 参数，修正分块配置。

### FC4001 ERR_CONFIG_ALIGNMENT
1. **检查张量地址**：确认设备地址按 16B/32B/64B 对齐。
2. **检查分块大小**：确认 tile 大小满足硬件对齐约束。
3. **检查 workspace 内存**：确认工作区内存由统一内存管理器分配。
4. **查日志上下文**：查看未对齐地址/大小，使用内存分配接口重新申请。

### FC4002 ERR_CONFIG_UNSUPPORTED
1. **检查配置组合**：确认 Tile 配置、数据类型、格式为支持的组合。
2. **检查算子模式**：确认不混用不兼容的计算模式、精度模式。
3. **检查硬件适配**：确认当前配置与运行的 NPU 硬件型号匹配。
4. **查日志上下文**：根据日志提示，替换为支持的配置组合。

### FC5000 ERR_RUNTIME_NULLPTR
1. **检查输入输出张量**：确认 传入的Tensor 非空且已完成地址分配。
2. **检查上下文初始化**：确认 matmul 上下文、配置句柄已正常创建。
3. **检查函数入参**：确认调用层未传入空指针到 matmul 接口。
4. **查日志上下文**：定位空指针变量，检查上层初始化与赋值流程。

### FC5001 ERR_RUNTIME_STATE
1. **检查初始化流程**：确认 Matmul 上下文已完成初始化再执行计算。
2. **检查状态机流转**：确认按 初始化->配置->执行->释放 顺序调用。
3. **检查资源状态**：确认依赖的 NPU 资源、工作区未被提前释放。
4. **查日志上下文**：查看异常状态码，回溯流程调用顺序。

### FC5002 ERR_RUNTIME_LOGIC
1. **检查执行分支**：确认计算流程未进入未定义/异常分支。
2. **检查中间结果**：确认计算过程中临时数据、索引值符合预期。
3. **检查不变量约束**：确认核心计算逻辑的前置条件均满足。
4. **查日志上下文**：通过日志定位异常路径，核对计算逻辑。

## 排查建议

### 日志落盘
打开DEBUG日志，指定日志落盘路径
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

### 排查步骤
1. **检查入参**：查阅对应算子或 API 文档（如 `/docs/api/operation/pypto-matmul.md`、算子说明），确认输入及输出Tesnor是否符合Matmul约束规格，包括Shape，Dtype，Format等。

2. **检查切分设置**：调用Matmul算子前，会执行`set_cube_tile_shapes()`设置切分大小，检查设置的TileShape大小是否符合Tiling的切分约束，可通过以下方式查看

```py
pypto.set_cube_tile_shapes([32, 32], [16, 16], [32, 32])   #[mL0, mL1], [kL0, kL1], [nL0, nL1]
tile_shape_info = pypto.get_cube_tile_shapes()
print(tile_shape_info)
#输出：[32, 32], [16, 16], [32, 32]
```

3. **典型场景**：reshape/view、交换维度（转置场景）、会改变对应的维度约束，例如，当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，当输入矩阵input非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K，当输入矩阵input转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M

**关联 Skill**：[pypto-environment-setup](../../.opencode/skills/pypto-environment-setup/SKILL.md)（环境与 NPU 设备诊断、`npu-smi`、驱动与编译运行）