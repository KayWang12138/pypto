# API 映射规则（Torch → PyPTO）

> **版本**: 1.0
> **最后更新**: 2026-03-15
> **说明**: 本文件用于"设计期可行性闸门"。每条映射必须能回链到 `docs/api/operation/index.md` 中的具体条目。

---

## 1. 映射分级定义

- **`direct`**：`docs/api/operation/index.md` 的 toctree 中存在对应 API 条目，且硬约束可满足。
- **`substitute`**：无 1:1 API，但可由多个已存在 API 组合实现；每个基础 API 均需在 toctree 中可查。
- **`unsupported`**：无 API 或关键约束不可满足，设计期直接阻断。

---

## 2. 设计期前置闸门（Fail-Fast）

在开始映射前，先检查以下全局约束。任一闸门失败应立即报错，不进入后续设计阶段。

| gate_id | 规则 | 级别 | 失败后果 | warning_code | evidence |
|---------|------|------|----------|--------------|----------|
| MAP_GATE_01 | Torch 输入必须满足 from_torch 入口约束（如 contiguous） | HARD | 构图/转换失败 | MAP_NON_CONTIGUOUS_INPUT | `docs/api/others/pypto-from_torch.md`; `python/pypto/converter.py` |
| MAP_GATE_02 | 目标 API 的 dtype 必须在其文档支持列表内 | HARD | 编译或运行失败 | MAP_DTYPE_UNSUPPORTED | 各 `docs/api/operation/pypto-*.md` |
| MAP_GATE_03 | format/NZ/对齐约束必须满足 | HARD | 运行时错误或数值异常 | MAP_FORMAT_NZ_ALIGN | `docs/api/others/pypto-from_torch.md`; `python/pypto/converter.py` |
| MAP_GATE_04 | dynamic axis 定义与调用维度一致 | HARD | 运行不稳定/失败 | MAP_DYNAMIC_AXIS_MISMATCH | `docs/api/others/pypto-from_torch.md` |
| MAP_GATE_05 | view/assemble 不得形成图循环风险 | HARD | 图拓扑失败 | MAP_DAG_CYCLE_RISK | `docs/tutorials/appendix/issue.md` |
| MAP_GATE_06 | run_mode 与设备环境匹配 | HARD | 运行失败 | MAP_RUNTIME_ENV_MISMATCH | `docs/api/config/pypto-set_runtime_options.md`; `docs/install/prepare_environment.md` |

---

## 3. 映射矩阵

字段定义：`torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence`

### 3.1 Elementwise / Arithmetic

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.add | direct | pypto.add | dtype/shape/broadcast 满足约束 | 广播误用 | MAP_ADD_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-add.md` |
| torch.sub | direct | pypto.sub | dtype/shape/broadcast 满足约束 | 广播误用 | MAP_SUB_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-sub.md` |
| torch.mul | direct | pypto.mul | dtype/shape/broadcast 满足约束 | 广播误用 | MAP_MUL_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-mul.md` |
| torch.div | direct | pypto.div | dtype 支持且除数合法 | 除零/精度风险 | MAP_DIV_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-div.md` |
| torch.neg | direct | pypto.neg | dtype/shape 满足约束 | dtype 失配 | MAP_NEG_DTYPE_MISMATCH | `docs/api/operation/pypto-neg.md` |
| torch.abs | direct | pypto.abs | dtype/shape 满足约束 | dtype 失配 | MAP_ABS_DTYPE_MISMATCH | `docs/api/operation/pypto-abs.md` |
| torch.pow | direct | pypto.pow | 指数类型与输入类型满足约束 | 数值稳定性 | MAP_POW_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-pow.md` |
| torch.reciprocal | direct | pypto.reciprocal | 输入不为 0 | Inf 风险 | MAP_RECIPROCAL_DIVZERO | `docs/api/operation/pypto-reciprocal.md` |
| torch.remainder | direct | pypto.remainder | dtype 满足约束 | 符号语义差异 | MAP_REMAINDER_SEMANTIC_RISK | `docs/api/operation/pypto-remainder.md` |
| torch.fmod | direct | pypto.fmod | dtype 满足约束 | 与 remainder 语义混淆 | MAP_FMOD_SEMANTIC_RISK | `docs/api/operation/pypto-fmod.md` |
| torch.clamp | direct | pypto.clip | min/max 参数合法 | 边界值处理风险 | MAP_CLIP_BOUNDARY_RISK | `docs/api/operation/pypto-clip.md` |
| torch.maximum | direct | pypto.maximum | dtype/shape 满足约束 | 广播风险 | MAP_MAXIMUM_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-maximum.md` |
| torch.minimum | direct | pypto.minimum | dtype/shape 满足约束 | 广播风险 | MAP_MINIMUM_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-minimum.md` |
| torch.copysign | direct | pypto.copysign | dtype 满足约束 | dtype 失配 | MAP_COPYSIGN_DTYPE_MISMATCH | `docs/api/operation/pypto-copysign.md` |

### 3.2 Math Functions

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.exp | direct | pypto.exp | dtype 支持 | 上溢风险 | MAP_EXP_NUMERIC_RISK | `docs/api/operation/pypto-exp.md` |
| torch.exp2 | direct | pypto.exp2 | dtype 支持 | 上溢风险 | MAP_EXP2_NUMERIC_RISK | `docs/api/operation/pypto-exp2.md` |
| torch.expm1 | direct | pypto.expm1 | dtype 支持 | 小值区间误差 | MAP_EXPM1_NUMERIC_RISK | `docs/api/operation/pypto-expm1.md` |
| torch.log | direct | pypto.log | 输入域合法（>0） | 非法域 NaN/Inf | MAP_LOG_DOMAIN_INVALID | `docs/api/operation/pypto-log.md` |
| torch.log2 | direct | pypto.log2 | 输入域合法 | 非法域风险 | MAP_LOG2_DOMAIN_INVALID | `docs/api/operation/pypto-log2.md` |
| torch.log10 | direct | pypto.log10 | 输入域合法 | 非法域风险 | MAP_LOG10_DOMAIN_INVALID | `docs/api/operation/pypto-log10.md` |
| torch.log1p | direct | pypto.log1p | 输入域合法（>-1） | 非法域风险 | MAP_LOG1P_DOMAIN_INVALID | `docs/api/operation/pypto-log1p.md` |
| torch.sqrt | direct | pypto.sqrt | 输入域合法（>=0） | NaN 风险 | MAP_SQRT_DOMAIN_INVALID | `docs/api/operation/pypto-sqrt.md` |
| torch.rsqrt | direct | pypto.rsqrt | 输入域合法（>0） | NaN/Inf 风险 | MAP_RSQRT_DOMAIN_INVALID | `docs/api/operation/pypto-rsqrt.md` |
| torch.sin | direct | pypto.sin | dtype 支持 | 精度差异 | MAP_SIN_DTYPE_MISMATCH | `docs/api/operation/pypto-sin.md` |
| torch.cos | direct | pypto.cos | dtype 支持 | 精度差异 | MAP_COS_DTYPE_MISMATCH | `docs/api/operation/pypto-cos.md` |
| torch.ceil | direct | pypto.ceil | dtype 支持 | 类型转换风险 | MAP_CEIL_DTYPE_MISMATCH | `docs/api/operation/pypto-ceil.md` |
| torch.floor | direct | pypto.floor | dtype 支持 | 类型转换风险 | MAP_FLOOR_DTYPE_MISMATCH | `docs/api/operation/pypto-floor.md` |
| torch.round | direct | pypto.round | dtype 支持 | 舍入差异 | MAP_ROUND_SEMANTIC_RISK | `docs/api/operation/pypto-round.md` |
| torch.trunc | direct | pypto.trunc | dtype 支持 | 舍入差异 | MAP_TRUNC_SEMANTIC_RISK | `docs/api/operation/pypto-trunc.md` |
| torch.sign | direct | pypto.sign | dtype 支持 | 零值符号语义差异 | MAP_SIGN_SEMANTIC_RISK | `docs/api/operation/pypto-sign.md` |
| torch.cbrt | direct | pypto.cbrt | dtype 支持 | 数值风险 | MAP_CBRT_NUMERIC_RISK | `docs/api/operation/pypto-cbrt.md` |
| torch.hypot | direct | pypto.hypot | 输入 dtype 兼容 | 数值稳定性风险 | MAP_HYPOT_NUMERIC_RISK | `docs/api/operation/pypto-hypot.md` |
| torch.gcd | direct | pypto.gcd | 整数 dtype | dtype 失配 | MAP_GCD_DTYPE_MISMATCH | `docs/api/operation/pypto-gcd.md` |

### 3.3 Comparison / Logical

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.eq | direct | pypto.eq | dtype/shape 支持 | bool 输出链路风险 | MAP_EQ_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-eq.md` |
| torch.ne | direct | pypto.ne | dtype/shape 支持 | bool 输出链路风险 | MAP_NE_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-ne.md` |
| torch.lt | direct | pypto.lt | dtype/shape 支持 | bool 输出链路风险 | MAP_LT_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-lt.md` |
| torch.le | direct | pypto.le | dtype/shape 支持 | bool 输出链路风险 | MAP_LE_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-le.md` |
| torch.gt | direct | pypto.gt | dtype/shape 支持 | bool 输出链路风险 | MAP_GT_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-gt.md` |
| torch.ge | direct | pypto.ge | dtype/shape 支持 | bool 输出链路风险 | MAP_GE_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-ge.md` |
| torch.logical_and | direct | pypto.logical_and | bool/shape 兼容 | 广播风险 | MAP_LOGICAL_AND_INVALID | `docs/api/operation/pypto-logical_and.md` |
| torch.logical_not | direct | pypto.logical_not | bool/shape 兼容 | 语义误用 | MAP_LOGICAL_NOT_INVALID | `docs/api/operation/pypto-logical_not.md` |
| torch.isfinite | direct | pypto.isfinite | dtype 支持 | 类型兼容风险 | MAP_ISFINITE_DTYPE_MISMATCH | `docs/api/operation/pypto-isfinite.md` |
| torch.signbit | direct | pypto.signbit | dtype 支持 | 类型兼容风险 | MAP_SIGNBIT_DTYPE_MISMATCH | `docs/api/operation/pypto-signbit.md` |
| torch.where | direct | pypto.where | 条件与输入 shape 合法 | 广播风险 | MAP_WHERE_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-where.md` |

### 3.4 Bitwise

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.bitwise_and | direct | pypto.bitwise_and | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_AND_DTYPE_MISMATCH | `docs/api/operation/pypto-bitwise_and.md` |
| torch.bitwise_or | direct | pypto.bitwise_or | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_OR_DTYPE_MISMATCH | `docs/api/operation/pypto-bitwise_or.md` |
| torch.bitwise_xor | direct | pypto.bitwise_xor | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_XOR_DTYPE_MISMATCH | `docs/api/operation/pypto-bitwise_xor.md` |
| torch.bitwise_not | direct | pypto.bitwise_not | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_NOT_DTYPE_MISMATCH | `docs/api/operation/pypto-bitwise_not.md` |
| torch.bitwise_left_shift | direct | pypto.bitwise_left_shift | 整数 dtype 支持 | 溢出风险 | MAP_LSHIFT_OVERFLOW_RISK | `docs/api/operation/pypto-bitwise_left_shift.md` |
| torch.bitwise_right_shift | direct | pypto.bitwise_right_shift | 整数 dtype 支持 | 符号位语义风险 | MAP_RSHIFT_SEMANTIC_RISK | `docs/api/operation/pypto-bitwise_right_shift.md` |

### 3.5 Activation / Normalization

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.relu / F.relu | direct | pypto.relu | dtype 满足约束 | 精度差异 | MAP_RELU_DTYPE_MISMATCH | `docs/api/operation/pypto-relu.md` |
| torch.sigmoid | direct | pypto.sigmoid | dtype 满足约束 | 精度差异 | MAP_SIGMOID_DTYPE_MISMATCH | `docs/api/operation/pypto-sigmoid.md` |
| F.softmax | direct | pypto.softmax | dim 与 dtype 满足约束 | 数值稳定性 | MAP_SOFTMAX_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-softmax.md` |
| F.leaky_relu | direct | pypto.lrelu | 参数与 dtype 合法 | 参数失配 | MAP_LRELU_PARAM_INVALID | `docs/api/operation/pypto-lrelu.md` |
| F.prelu | direct | pypto.prelu | 权重形状与 dtype 合法 | 权重维度失配 | MAP_PRELU_WEIGHT_MISMATCH | `docs/api/operation/pypto-prelu.md` |
| F.rms_norm | direct | pypto.rms_norm | 归一化维度合法 | 维度失配 | MAP_RMSNORM_DIM_INVALID | `docs/api/operation/pypto-rms_norm.md` |
| F.silu / swish | substitute | pypto.mul(x, pypto.sigmoid(x)) | mul/sigmoid 约束均满足 | 数值精度差异 | MAP_SILU_SUBSTITUTE_RISK | `docs/api/operation/pypto-mul.md`; `docs/api/operation/pypto-sigmoid.md` |
| torch.tanh | substitute | (exp(x)-exp(-x))/(exp(x)+exp(-x)) | exp/neg/sub/add/div 均可用 | 数值稳定性风险（大值溢出） | MAP_TANH_SUBSTITUTE_RISK | `docs/api/operation/pypto-exp.md`; `docs/api/operation/pypto-neg.md`; `docs/api/operation/pypto-sub.md`; `docs/api/operation/pypto-add.md`; `docs/api/operation/pypto-div.md` |
| F.gelu | substitute | x*0.5*(1+tanh(sqrt(2/π)*(x+0.044715*x³))) | 需 tanh substitute + mul/add/pow | 多层 substitute 精度累积风险 | MAP_GELU_SUBSTITUTE_RISK | 基础 API 均在 `docs/api/operation/index.md` |
| F.layer_norm | substitute | mean→sub→var→rsqrt→mul→add | sum/div/sub/var/rsqrt/mul/add 均可用 | 语义差异、精度风险 | MAP_LAYERNORM_SUBSTITUTE_RISK | `docs/api/operation/pypto-sum.md`; `docs/api/operation/pypto-var.md`; `docs/api/operation/pypto-rsqrt.md` |

### 3.6 Reduction / Statistics

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.sum | direct | pypto.sum | reduction 轴/tiling 约束满足 | tile 上限风险 | MAP_SUM_REDUCTION_INVALID | `docs/api/operation/pypto-sum.md` |
| torch.amax / torch.max(dim) | direct | pypto.amax | reduction 约束满足 | 轴选择风险 | MAP_AMAX_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-amax.md` |
| torch.amin / torch.min(dim) | direct | pypto.amin | reduction 约束满足 | 轴选择风险 | MAP_AMIN_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-amin.md` |
| torch.prod | direct | pypto.prod | reduction 约束满足 | 溢出风险 | MAP_PROD_OVERFLOW_RISK | `docs/api/operation/pypto-prod.md` |
| torch.var | direct | pypto.var | reduction 约束满足 | 维度/精度风险 | MAP_VAR_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-var.md` |
| torch.cumsum | direct | pypto.cumsum | 轴/shape 约束满足 | 累积溢出风险 | MAP_CUMSUM_OVERFLOW_RISK | `docs/api/operation/pypto-cumsum.md` |
| torch.mean | substitute | pypto.sum + 元素计数 + pypto.div | sum/div 均可用且计数合法 | 数值偏差 | MAP_MEAN_SUBSTITUTE_RISK | `docs/api/operation/pypto-sum.md`; `docs/api/operation/pypto-div.md` |
| torch.std | substitute | pypto.var + pypto.sqrt | var/sqrt 均可用 | 精度与性能风险 | MAP_STD_SUBSTITUTE_RISK | `docs/api/operation/pypto-var.md`; `docs/api/operation/pypto-sqrt.md` |

### 3.7 Matrix / Linear Algebra

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.matmul | direct | pypto.matmul | cube tile/dtype/format 约束满足 | tile 失配 | MAP_MATMUL_TILE_INVALID | `docs/api/operation/pypto-matmul.md`; `docs/api/config/pypto-set_cube_tile_shapes.md` |
| torch.bmm | direct | pypto.matmul | 3D 维度合法 | batch 维失配 | MAP_BMM_DIM_MISMATCH | `docs/api/operation/pypto-matmul.md` |
| torch.mm | direct | pypto.matmul | 2D 维度合法 | 维度失配 | MAP_MM_DIM_MISMATCH | `docs/api/operation/pypto-matmul.md` |
| F.linear | substitute | pypto.matmul + pypto.add（bias） | weight 需转置、bias 维度合法 | 维度与布局风险 | MAP_LINEAR_SUBSTITUTE_RISK | `docs/api/operation/pypto-matmul.md`; `docs/api/operation/pypto-add.md` |
| torch.addmm | substitute | pypto.matmul + pypto.add | matmul/add 约束同时满足 | dtype 对齐风险 | MAP_ADDMM_SUBSTITUTE_RISK | `docs/api/operation/pypto-matmul.md`; `docs/api/operation/pypto-add.md` |
| torch.baddbmm | substitute | pypto.matmul + pypto.add | batch 维与 dtype 合法 | 维度失配 | MAP_BADDBMM_SUBSTITUTE_RISK | `docs/api/operation/pypto-matmul.md`; `docs/api/operation/pypto-add.md` |

### 3.8 Index / Scatter / Sort

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.gather | direct | pypto.gather | dim/view/tile 约束满足 | 维度切分非法 | MAP_GATHER_DIM_INVALID | `docs/api/operation/pypto-gather.md` |
| torch.scatter | direct | pypto.scatter / pypto.scatter_ | dim/index 约束满足 | in-place 语义风险 | MAP_SCATTER_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-scatter.md`; `docs/api/operation/pypto-scatter_.md` |
| torch.scatter_add | substitute | pypto.scatter_update 或 index_add_ 组合 | index 轴约束满足 | 语义偏差风险 | MAP_SCATTER_ADD_SUBSTITUTE_RISK | `docs/api/operation/pypto-scatter_update.md`; `docs/api/operation/pypto-index_add_.md` |
| torch.index_select | direct | pypto.index_select | index dtype/axis 约束满足 | 轴失配 | MAP_INDEX_SELECT_INVALID | `docs/api/operation/pypto-index_select.md` |
| torch.index_add | direct | pypto.index_add / pypto.index_add_ | index 轴与 dtype 合法 | in-place 风险 | MAP_INDEX_ADD_INVALID | `docs/api/operation/pypto-index_add.md`; `docs/api/operation/pypto-index_add_.md` |
| torch.index_put_ | direct | pypto.indexput_ | 索引语义满足约束 | in-place 覆盖风险 | MAP_INDEXPUT_INPLACE_RISK | `docs/api/operation/pypto-indexput_.md` |
| torch.topk | direct | pypto.topk | 轴/tiling 约束满足 | 轴受限风险 | MAP_TOPK_AXIS_INVALID | `docs/api/operation/pypto-topk.md` |
| torch.argsort | direct | pypto.argsort | dim/tiling 约束满足 | 维度限制风险 | MAP_ARGSORT_DIM_INVALID | `docs/api/operation/pypto-argsort.md` |
| F.one_hot | direct | pypto.one_hot | 类别维度与 dtype 合法 | 维度爆炸风险 | MAP_ONEHOT_DIM_RISK | `docs/api/operation/pypto-one_hot.md` |

### 3.9 Shape / Layout / Tensor Construction

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| torch.reshape | direct | pypto.reshape | 目标形状合法 | 形状失配 | MAP_RESHAPE_INVALID | `docs/api/operation/pypto-reshape.md` |
| torch.view | direct | pypto.view | view 约束与 valid_shape 合法 | 精度/拓扑风险 | MAP_VIEW_VALID_SHAPE_RISK | `docs/api/operation/pypto-view.md`; `docs/tutorials/appendix/faq.md` |
| torch.transpose | direct | pypto.transpose | 轴交换模式在支持范围内 | 模式不可达 | MAP_TRANSPOSE_MODE_UNSUPPORTED | `docs/api/operation/pypto-transpose.md` |
| torch.permute | substitute | 有限 transpose 组合 | 必须可分解为受支持 transpose 序列 | 组合不可达 | MAP_PERMUTE_LIMITED_SUPPORT | `docs/api/operation/pypto-transpose.md` |
| torch.unsqueeze | direct | pypto.unsqueeze | 维度插入合法 | 维度错误 | MAP_UNSQUEEZE_DIM_INVALID | `docs/api/operation/pypto-unsqueeze.md` |
| torch.cat | direct | pypto.concat | 连接轴与输入 shape 合法 | 维度不一致 | MAP_CONCAT_DIM_MISMATCH | `docs/api/operation/pypto-concat.md` |
| torch.clone | direct | pypto.clone | dtype/shape 合法 | 内存开销风险 | MAP_CLONE_MEMORY_RISK | `docs/api/operation/pypto-clone.md` |
| torch.pad | direct | pypto.pad | padding 参数合法 | 维度/边界风险 | MAP_PAD_PARAM_INVALID | `docs/api/operation/pypto-pad.md` |
| torch.tril | direct | pypto.tril | 输入维度合法 | 布局风险 | MAP_TRIL_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-tril.md` |
| torch.triu | direct | pypto.triu | 输入维度合法 | 布局风险 | MAP_TRIU_CONSTRAINT_VIOLATION | `docs/api/operation/pypto-triu.md` |
| torch.zeros | direct | pypto.zeros | shape/dtype 合法 | dtype 失配 | MAP_ZEROS_DTYPE_MISMATCH | `docs/api/operation/pypto-zeros.md` |
| torch.ones | direct | pypto.ones | shape/dtype 合法 | dtype 失配 | MAP_ONES_DTYPE_MISMATCH | `docs/api/operation/pypto-ones.md` |
| torch.full | direct | pypto.full | shape/dtype/value 合法 | dtype 失配 | MAP_FULL_DTYPE_MISMATCH | `docs/api/operation/pypto-full.md` |
| torch.arange | direct | pypto.arange | start/end/step 与 dtype 合法 | 步长非法 | MAP_ARANGE_STEP_INVALID | `docs/api/operation/pypto-arange.md` |
| torch.flatten | substitute | pypto.reshape | 展平维度可解析 | 维度错配 | MAP_FLATTEN_SUBSTITUTE_RISK | `docs/api/operation/pypto-reshape.md` |

### 3.10 Type Conversion

| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|----------|------|---------------------|---------------|------|--------------|----------|
| Tensor.to(dtype) | direct | pypto.cast | 目标 dtype 在支持列表内 | 精度损失 | MAP_CAST_PRECISION_LOSS | `docs/api/operation/pypto-cast.md` |

### 3.11 明确 unsupported（常见但当前无可靠映射）

| torch_op | tier | 原因 | warning_code | evidence |
|----------|------|------|--------------|----------|
| torch.roll | unsupported | 无 1:1 API | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |
| torch.flip | unsupported | 无 1:1 API | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |
| F.dropout | unsupported | 推理路径不提供训练态语义 | MAP_UNSUPPORTED_TRAINING_OP | `docs/api/operation/index.md` |
| F.batch_norm | unsupported | 无 1:1 API，无可靠 substitute | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |
| F.group_norm | unsupported | 无 1:1 API，无可靠 substitute | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |
| F.instance_norm | unsupported | 无 1:1 API，无可靠 substitute | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |
| F.conv1d/2d/3d | unsupported | 无 1:1 API（index.md 中不存在 pypto-conv） | MAP_UNSUPPORTED_API | `docs/api/operation/index.md` |

---

## 4. substitute 配方库

对于 substitute 映射，以下是关键组合配方的详细说明。

### 4.1 SiLU / Swish

- **目标 Torch op**: `F.silu(x)` / `x * torch.sigmoid(x)`
- **组合配方**: `pypto.mul(x, pypto.sigmoid(x))`
- **前置条件**: mul 和 sigmoid 的 dtype/shape 约束均满足
- **数值风险**: sigmoid 在极端值区域精度有限，但组合后风险低
- **性能风险**: 两次算子调用，性能略低于假设存在的 1:1 API
- **warning_code**: `MAP_SILU_SUBSTITUTE_RISK`
- **evidence**: `docs/api/operation/pypto-mul.md`; `docs/api/operation/pypto-sigmoid.md`

### 4.2 Tanh

- **目标 Torch op**: `torch.tanh(x)`
- **组合配方**: `pypto.div(pypto.sub(pypto.exp(x), pypto.exp(pypto.neg(x))), pypto.add(pypto.exp(x), pypto.exp(pypto.neg(x))))`
- **前置条件**: exp/neg/sub/add/div 均可用
- **数值风险**: 大值输入时 exp 溢出；建议对输入做 clamp 保护
- **性能风险**: 6 次算子调用，性能敏感场景需评估
- **warning_code**: `MAP_TANH_SUBSTITUTE_RISK`
- **evidence**: `docs/api/operation/pypto-exp.md`; `docs/api/operation/pypto-neg.md`; `docs/api/operation/pypto-sub.md`; `docs/api/operation/pypto-add.md`; `docs/api/operation/pypto-div.md`

### 4.3 GELU（近似）

- **目标 Torch op**: `F.gelu(x)`
- **组合配方**: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`，其中 tanh 本身也是 substitute
- **前置条件**: 所有基础 API（mul/add/pow/exp/neg/sub/div）均可用
- **数值风险**: 多层 substitute 组合，精度累积风险较高
- **性能风险**: 调用链长，性能敏感场景需重点评估
- **warning_code**: `MAP_GELU_SUBSTITUTE_RISK`
- **evidence**: 基础 API 均在 `docs/api/operation/index.md`

### 4.4 Mean

- **目标 Torch op**: `torch.mean(x, dim)`
- **组合配方**: `pypto.div(pypto.sum(x, dim, keepdim=True), N)`，N 为归约轴长度
- **前置条件**: sum/div 均可用，N 可获取
- **数值风险**: 大规模归约时浮点精度损失
- **warning_code**: `MAP_MEAN_SUBSTITUTE_RISK`
- **evidence**: `docs/api/operation/pypto-sum.md`; `docs/api/operation/pypto-div.md`

---

## 5. unsupported 前置告警模板

当映射结果为 unsupported 时，在设计阶段生成告警并阻断流程。

- **模板一（API 不存在）**
  - 触发：目标 Torch op 在 `docs/api/operation/index.md` 下无对应 API 且无可靠 substitute。
  - 告警：`[MAP_UNSUPPORTED_API] {torch_op} 当前无可落地映射，请在设计阶段调整算子路径。`

- **模板二（关键约束不可满足）**
  - 触发：存在候选 API，但 dtype/shape/format/runtime 任一硬约束不满足。
  - 告警：`[MAP_HARD_CONSTRAINT_FAILED] {torch_op} 映射失败：{constraint_name} 不满足。`

---

## 6. 证据索引

| 证据文件 | 用途 |
|----------|------|
| `docs/api/operation/index.md` | API 存在性判定（toctree 权威源） |
| `docs/api/operation/pypto-*.md` | 各 API 的参数、dtype、约束详情 |
| `docs/api/others/pypto-from_torch.md` | from_torch 入口约束（contiguous/format/dtype） |
| `docs/api/config/pypto-set_cube_tile_shapes.md` | Cube tiling 硬约束 |
| `docs/api/config/pypto-set_runtime_options.md` | 运行时参数约束 |
| `docs/install/prepare_environment.md` | 环境与设备配置约束 |
| `docs/tutorials/appendix/issue.md` | 已知问题与规避方案 |
| `docs/tutorials/appendix/faq.md` | view/reshape 常见问题 |
| `python/pypto/converter.py` | from_torch 约束链路代码级证据 |
