# pypto-op-design 四个 reference 文件超详细修改方案（仅方案，不执行代码改写）

## 0. 目标与边界（强约束）

### 0.1 本文档目标

将 `references/` 下 4 个现有 reference 文件，从”骨架+示例+TODO”升级为”可直接用于设计期可行性闸门”的规则文档。

### 0.2 修改范围（仅 4 个文件）

- `references/api_mapping.md`
- `references/tiling_rules.md`
- `references/loop_strategy.md`
- `references/performance_params.md`

### 0.3 明确不做

- 不新增 reference 文件。
- 明确不创建：
  - `testing_validation_rules.md`
  - `compatibility_rules.md`
- 不引入泳道图可视化依赖，全部采用非可视化验证路径（文本规则、配置检查、日志检查、数值检查）。

### 0.4 统一产出标准

- 每条规则都必须有“本地仓库证据锚点”。
- 每条规则必须标注级别：
  - `HARD`（硬约束）
  - `HEURISTIC`（经验建议）
- 无证据规则必须降级为 `HEURISTIC` 或删除。
- 文中不允许残留 `TODO` 占位语句。

---

## 1. 全局证据索引（所有文件可复用）

### 1.1 API/配置主证据

- `docs/api/operation/index.md`
- `docs/api/operation/*.md`
- `docs/api/controlflow/pypto-loop.md`
- `docs/api/controlflow/pypto-loop_unroll.md`
- `docs/api/config/pypto-set_vec_tile_shapes.md`
- `docs/api/config/pypto-set_cube_tile_shapes.md`
- `docs/api/config/pypto-set_pass_options.md`
- `docs/api/config/pypto-set_runtime_options.md`
- `docs/api/others/pypto-from_torch.md`

### 1.2 教程/行为证据

- `docs/tutorials/development/tiling.md`
- `docs/tutorials/development/loops.md`
- `docs/tutorials/debug/performance.md`
- `docs/tutorials/network_integration/pytorch_integration.md`
- `docs/tutorials/appendix/faq.md`
- `docs/tutorials/appendix/issue.md`
- `docs/install/prepare_environment.md`
- `AGENTS.md`

### 1.3 代码级证据

- `python/pypto/converter.py`（from_torch contiguous/format/dtype 约束链路）
- `examples/03_advanced/aclgraph/aclgraph.py`（组合式替代映射示例）

---

## 2. 四文件统一改造规范

### 2.1 规则行格式统一

所有文件统一使用规则行字段：

`rule_id | rule_text | level(HARD/HEURISTIC) | trigger | impact | warning_code | evidence`

### 2.2 告警码命名统一

- `MAP_*`：接口映射问题
- `TILE_*`：tiling 规则问题
- `LOOP_*`：循环/控制流问题
- `PERF_*`：pass/runtime 参数问题

### 2.3 术语统一

- 映射分级只允许：`direct` / `substitute` / `unsupported`
- 约束分级只允许：`HARD` / `HEURISTIC`

### 2.4 证据粒度统一

- 禁止“来源：docs/api/operation”这种笼统写法。
- 必须下钻到可定位路径（例如具体 API 文档文件）。

---

## 3. `api_mapping.md` 超详细修改方案

## 3.1 现存问题（需要完全修复）

- 存在未核实或不存在 API 的直接映射描述。
- 仅有粗粒度来源，缺少“规则->证据”逐条绑定。
- 缺少设计期 fail-fast 闸门，导致晚失败风险高。

## 3.2 目标结构（按顺序重写）

1. `适用范围与分级定义`
2. `Torch->PyPTO 映射矩阵`
3. `设计期前置闸门（Fail-Fast）`
4. `substitute 组合配方库`
5. `unsupported 清单与前置告警模板`
6. `晚失败高发模式前移`
7. `证据索引`

## 3.3 映射矩阵固定字段

每行必须包含：

`torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence`

其中：

- `tier` 只能是 `direct/substitute/unsupported`
- `preconditions` 至少覆盖 dtype/shape/format/runtime 中相关项
- `warning_code` 必须可复用、可机读

## 3.4 direct/substitute/unsupported 判定规则（必须写进文档）

- `direct`：`docs/api/operation/*.md` 存在明确 API 且约束可满足。
- `substitute`：无 1:1 API，但有可落证组合路径。
- `unsupported`：无 API 或关键约束不可满足，必须前置报错。

## 3.5 设计期前置闸门（新增硬段落）

至少新增以下闸门条目：

- `MAP_GATE_01`：输入 contiguous 检查（from_torch）
- `MAP_GATE_02`：dtype 支持检查（按目标 API 文档）
- `MAP_GATE_03`：format/NZ/对齐检查
- `MAP_GATE_04`：dynamic axis 一致性检查
- `MAP_GATE_05`：view/assemble 成环风险检查
- `MAP_GATE_06`：run_mode 与设备可用性检查

## 3.6 Torch 常用接口覆盖要求（新增）

按分类覆盖，不允许只举零散示例：

- Elementwise：`add/sub/mul/div/neg/abs/pow/where/clip`
- Math：`exp/log/sqrt/rsqrt/sin/cos/...`
- Activation：`relu/sigmoid/softmax/gelu/silu/tanh`
- Reduction：`sum/amax/amin/prod/var/mean/std`
- Matrix：`matmul/bmm/linear`
- Indexing：`gather/scatter/index_select/index_add_/topk/argsort`
- Shape：`reshape/view/transpose/permute/flatten/concat`
- Creation：`zeros/ones/full/arange`

每个 torch_op 至少输出一行映射矩阵记录。

## 3.7 substitute 配方模板（必须使用）

每条 substitute 映射按如下模板：

- `目标 Torch op`
- `组合配方（按执行顺序）`
- `前置条件`
- `数值风险`
- `性能风险`
- `形状/轴限制`
- `warning_code`
- `evidence`

## 3.8 unsupported 告警模板（必须新增）

每条 unsupported 必须包含：

- `判定条件`
- `阻断原因`
- `前置告警文案模板`
- `建议替代路径（若有）`

## 3.9 强制纠偏项

- 删除不存在 API 的“direct”描述。
- 所有笼统来源改为精确路径。
- 不可证实断言降级为 `HEURISTIC`。

---

## 4. `tiling_rules.md` 超详细修改方案

## 4.1 现存问题

- HARD 约束与 HEURISTIC 混写，读者无法区分。
- cube 约束不完整。
- 混合算子策略仍停留在 TODO。

## 4.2 目标结构

1. `算子分类与 tiling 决策入口`
2. `HARD 约束（必须满足）`
3. `HEURISTIC 建议（可调优）`
4. `混合算子策略（Cube+Vector）`
5. `非可视化校验清单`
6. `失败签名与定位`
7. `证据索引`

## 4.3 HARD 约束分组（必须补齐）

- `TILE_VEC_*`：vec 维度与对齐规则
- `TILE_RED_*`：reduction tile 上限/轴限制
- `TILE_CUBE_*`：cube 对齐、L0/L1 层级、buffer 上界、split-k 支持范围
- `TILE_FMT_*`：格式兼容边界

每条 HARD 规则必须写明失败后果（性能退化、编译失败或运行失败）。

## 4.4 HEURISTIC 建议（必须和 HARD 隔离）

- 初始 tile 选取建议
- 小 tile 与大 tile 的性能副作用
- loop 合并对 tile 的联动影响

并显式标注“建议不是硬约束”。

## 4.5 混合算子策略（必须从 TODO 升级为可执行规则）

固定流程：

1. 识别主算力阶段
2. 先定 cube tile（主干）
3. 后定 vector tile（尾部）
4. 做跨阶段 shape/format/dtype 连贯性校验
5. 失败触发 `TILE_MIXED_PIPELINE_INVALID`

## 4.6 非可视化校验清单（新增）

- 对齐检查 pass/fail
- buffer 预算 pass/fail
- 轴切分合法性 pass/fail
- 编译日志关键错误签名命中情况

---

## 5. `loop_strategy.md` 超详细修改方案

## 5.1 必修错误

- 把错误引用 `loop.md` 全量更正为 `docs/tutorials/development/loops.md`。

## 5.2 目标结构

1. `是否需要 loop（判定树）`
2. `静态轴 vs 动态轴（硬规则）`
3. `标准写法模板`
4. `loop_unroll 使用边界`
5. `submit_before_loop 触发条件`
6. `循环合并与尾块处理`
7. `失败签名与规避`
8. `证据索引`

## 5.3 必须新增硬规则

- 静态轴优先 Python `for`
- 动态轴才使用 `pypto.loop`
- `loop_unroll` 仅在受控条件开启
- 依赖链不满足时必须约束调度顺序

## 5.4 模板化示例要求

至少提供以下模板段落（非伪代码风格，需贴合 API 语义）：

- 动态轴 loop 模板
- loop_unroll 模板（含禁忌）
- 父子 loop 依赖模板
- 尾块有效区间处理模板

## 5.5 风险条目（必须可触发告警码）

- `LOOP_UNROLL_COMPILE_BLOWUP`
- `LOOP_DEPENDENCY_HAZARD`
- `LOOP_TAIL_VALIDITY_MISMATCH`
- `LOOP_MERGE_MISUSE`

每条写“触发条件+后果+规避动作+证据”。

---

## 6. `performance_params.md` 超详细修改方案

## 6.1 现存问题

- 默认值与示例值混淆。
- 参数定义粒度不够，无法做设计期 gating。
- TODO 残留。

## 6.2 目标结构

1. `参数分层（baseline/tuning）`
2. `pass_options 规范表`
3. `runtime_options 规范表`
4. `按算子类型的组合建议`
5. `误配风险库`
6. `非可视化验证路径`
7. `证据索引`

## 6.3 参数表固定字段

`option_key | layer | scope | default | allowed_values | recommended_values | side_effect | warning_code | evidence`

要求：

- `default` 仅能来自 API 文档
- `recommended_values` 明确标注“示例，不等于默认”
- 每项必须有误配风险描述

## 6.4 必须覆盖的参数类

- `set_pass_options` 关键项
- `set_runtime_options` 关键项
- run_mode 与设备/环境关系

## 6.5 非可视化验证路径（必须新增）

- 编译日志检查项
- runtime 生效检查项
- 数值检查项
- 约束命中诊断项

## 6.6 风险库告警码（示例）

- `PERF_RUN_MODE_ENV_MISMATCH`
- `PERF_PASS_OPTION_CONFLICT`
- `PERF_OVERTUNE_STABILITY_RISK`

---

## 7. 四文件交叉一致性修订清单

## 7.1 一致性检查项

- 映射分级术语一致（direct/substitute/unsupported）
- HARD/HEURISTIC 定义一致
- 告警码无重名异义
- 证据路径无失效
- 无 TODO 残留

## 7.2 交叉引用规则

- `api_mapping.md` 中的映射风险，需能在 `tiling/loop/performance` 找到约束支撑。
- `performance_params.md` 的误配风险，需能反向关联到 mapping/tiling/loop 的失败模式。

---

## 8. 执行顺序（后续真正落地改文件时）

1. 先改 `api_mapping.md`（建立全局可行性闸门）
2. 再改 `tiling_rules.md`（硬约束完整化）
3. 再改 `loop_strategy.md`（控制流正确性）
4. 最后改 `performance_params.md`（参数与误配风险）
5. 做一次四文件一致性清洗

---

## 9. 验收标准（用于最终确认）

- 仅 4 个现有文件在方案中被修改。
- 不包含新增文件实施项。
- Torch 常用接口已按三分级覆盖。
- 每条 HARD 规则可回链本地证据。
- 每条 HEURISTIC 明确标注。
- 全流程不依赖泳道图可视化。
- 不存在错误路径或不存在 API 声明。

---

## 10. 交付物定义

按本文方案落地后，这 4 个 reference 文件应具备以下能力：

- 在编码前完成映射可行性分级（direct/substitute/unsupported）。
- 对不可支持或高风险设计实现前置阻断。
- 通过非可视化路径完成规则验证与问题定位。
- 显著降低环境/shape/dtype/format/controlflow 失配导致的晚失败。

---

## 11. 具体修改内容（可直接粘贴到目标文件）

> 本节给的是“具体修改稿”，不是抽象说明。落地时按“替换块”执行。

## 11.1 `api_mapping.md` 具体修改稿

### A) 文件头替换为以下结构

```markdown
# API 映射规则（Torch -> PyPTO）

## 1. 映射分级定义

- `direct`：存在对应 PyPTO API，且硬约束可满足。
- `substitute`：无 1:1 API，但可由多个已存在 API 组合实现。
- `unsupported`：无 API 或关键约束不可满足，设计期直接阻断。

## 2. 设计期前置闸门（Fail-Fast）

| gate_id | 规则 | 级别 | 失败后果 | warning_code | evidence |
|---|---|---|---|---|---|
| MAP_GATE_01 | Torch 输入必须满足 from_torch 入口约束（如 contiguous） | HARD | 构图/转换失败 | MAP_NON_CONTIGUOUS_INPUT | docs/api/others/pypto-from_torch.md; python/pypto/converter.py |
| MAP_GATE_02 | 目标 API 的 dtype 必须在文档支持列表内 | HARD | 编译或运行失败 | MAP_DTYPE_UNSUPPORTED | docs/api/operation/*.md |
| MAP_GATE_03 | format/NZ/对齐约束必须满足 | HARD | 运行时错误或数值异常 | MAP_FORMAT_NZ_ALIGN | docs/api/others/pypto-from_torch.md; python/pypto/converter.py |
| MAP_GATE_04 | dynamic axis 定义与调用维度一致 | HARD | 运行不稳定/失败 | MAP_DYNAMIC_AXIS_MISMATCH | docs/api/others/pypto-from_torch.md; docs/tutorials/development/tensor_creation.md |
| MAP_GATE_05 | view/assemble 不得形成图循环风险 | HARD | 图拓扑失败 | MAP_DAG_CYCLE_RISK | docs/tutorials/appendix/issue.md |
| MAP_GATE_06 | run_mode 与设备环境匹配 | HARD | 运行失败 | MAP_RUNTIME_ENV_MISMATCH | docs/api/config/pypto-set_runtime_options.md; docs/install/prepare_environment.md; AGENTS.md |

## 3. 映射矩阵

字段定义：
`torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence`
```

### B) 在“映射矩阵”中插入以下完整覆盖行（按类别）

```markdown
### 3.1 Elementwise / Arithmetic
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.add | direct | pypto.add | dtype/shape/broadcast 满足约束 | 广播误用 | MAP_ADD_CONSTRAINT_VIOLATION | docs/api/operation/pypto-add.md |
| torch.sub | direct | pypto.sub | 同上 | 同上 | MAP_SUB_CONSTRAINT_VIOLATION | docs/api/operation/pypto-sub.md |
| torch.mul | direct | pypto.mul | 同上 | 同上 | MAP_MUL_CONSTRAINT_VIOLATION | docs/api/operation/pypto-mul.md |
| torch.div | direct | pypto.div | dtype 支持且除数合法 | 除零/精度风险 | MAP_DIV_CONSTRAINT_VIOLATION | docs/api/operation/pypto-div.md |
| torch.neg | direct | pypto.neg | dtype/shape 满足约束 | dtype 失配 | MAP_NEG_DTYPE_MISMATCH | docs/api/operation/pypto-neg.md |
| torch.abs | direct | pypto.abs | dtype/shape 满足约束 | dtype 失配 | MAP_ABS_DTYPE_MISMATCH | docs/api/operation/pypto-abs.md |
| torch.pow | direct | pypto.pow | 指数类型与输入类型满足约束 | 数值稳定性 | MAP_POW_CONSTRAINT_VIOLATION | docs/api/operation/pypto-pow.md |
| torch.remainder | direct | pypto-remainder | dtype 满足约束 | 符号语义差异 | MAP_REMAINDER_SEMANTIC_RISK | docs/api/operation/pypto-remainder.md |
| torch.fmod | direct | pypto.fmod | dtype 满足约束 | 与 remainder 语义混淆 | MAP_FMOD_SEMANTIC_RISK | docs/api/operation/pypto-fmod.md |
| torch.clamp | direct | pypto.clip | min/max 参数合法 | 边界值处理风险 | MAP_CLIP_BOUNDARY_RISK | docs/api/operation/pypto-clip.md |

### 3.2 Math Functions
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.exp | direct | pypto.exp | dtype 支持 | 上溢风险 | MAP_EXP_NUMERIC_RISK | docs/api/operation/pypto-exp.md |
| torch.exp2 | direct | pypto.exp2 | dtype 支持 | 上溢风险 | MAP_EXP2_NUMERIC_RISK | docs/api/operation/pypto-exp2.md |
| torch.expm1 | direct | pypto.expm1 | dtype 支持 | 小值区间误差 | MAP_EXPM1_NUMERIC_RISK | docs/api/operation/pypto-expm1.md |
| torch.log | direct | pypto.log | 输入域合法（>0） | 非法域 NaN/Inf | MAP_LOG_DOMAIN_INVALID | docs/api/operation/pypto-log.md |
| torch.log2 | direct | pypto.log2 | 输入域合法 | 非法域风险 | MAP_LOG2_DOMAIN_INVALID | docs/api/operation/pypto-log2.md |
| torch.log10 | direct | pypto.log10 | 输入域合法 | 非法域风险 | MAP_LOG10_DOMAIN_INVALID | docs/api/operation/pypto-log10.md |
| torch.log1p | direct | pypto.log1p | 输入域合法（>-1） | 非法域风险 | MAP_LOG1P_DOMAIN_INVALID | docs/api/operation/pypto-log1p.md |
| torch.sqrt | direct | pypto.sqrt | 输入域合法（>=0） | NaN 风险 | MAP_SQRT_DOMAIN_INVALID | docs/api/operation/pypto-sqrt.md |
| torch.rsqrt | direct | pypto.rsqrt | 输入域合法（>0） | NaN/Inf 风险 | MAP_RSQRT_DOMAIN_INVALID | docs/api/operation/pypto-rsqrt.md |
| torch.reciprocal | direct | pypto.reciprocal | 输入不为 0 | Inf 风险 | MAP_RECIPROCAL_DIVZERO | docs/api/operation/pypto-reciprocal.md |
| torch.sin | direct | pypto.sin | dtype 支持 | 精度差异 | MAP_SIN_DTYPE_MISMATCH | docs/api/operation/pypto-sin.md |
| torch.cos | direct | pypto.cos | dtype 支持 | 精度差异 | MAP_COS_DTYPE_MISMATCH | docs/api/operation/pypto-cos.md |
| torch.ceil | direct | pypto.ceil | dtype 支持 | 类型转换风险 | MAP_CEIL_DTYPE_MISMATCH | docs/api/operation/pypto-ceil.md |
| torch.floor | direct | pypto.floor | dtype 支持 | 类型转换风险 | MAP_FLOOR_DTYPE_MISMATCH | docs/api/operation/pypto-floor.md |
| torch.round | direct | pypto.round | dtype 支持 | 舍入差异 | MAP_ROUND_SEMANTIC_RISK | docs/api/operation/pypto-round.md |
| torch.trunc | direct | pypto.trunc | dtype 支持 | 舍入差异 | MAP_TRUNC_SEMANTIC_RISK | docs/api/operation/pypto-trunc.md |
| torch.sign | direct | pypto.sign | dtype 支持 | 零值符号语义差异 | MAP_SIGN_SEMANTIC_RISK | docs/api/operation/pypto-sign.md |
| torch.cbrt | direct | pypto.cbrt | dtype 支持 | 数值风险 | MAP_CBRT_NUMERIC_RISK | docs/api/operation/pypto-cbrt.md |
| torch.hypot | direct | pypto.hypot | 输入 dtype 兼容 | 数值稳定性风险 | MAP_HYPOT_NUMERIC_RISK | docs/api/operation/pypto-hypot.md |

### 3.3 Comparison / Logical
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.eq | direct | pypto.eq | dtype/shape 支持 | bool 输出链路风险 | MAP_EQ_CONSTRAINT_VIOLATION | docs/api/operation/pypto-eq.md |
| torch.ne | direct | pypto.ne | 同上 | 同上 | MAP_NE_CONSTRAINT_VIOLATION | docs/api/operation/pypto-ne.md |
| torch.lt | direct | pypto.lt | 同上 | 同上 | MAP_LT_CONSTRAINT_VIOLATION | docs/api/operation/pypto-lt.md |
| torch.le | direct | pypto.le | 同上 | 同上 | MAP_LE_CONSTRAINT_VIOLATION | docs/api/operation/pypto-le.md |
| torch.gt | direct | pypto.gt | 同上 | 同上 | MAP_GT_CONSTRAINT_VIOLATION | docs/api/operation/pypto-gt.md |
| torch.ge | direct | pypto.ge | 同上 | 同上 | MAP_GE_CONSTRAINT_VIOLATION | docs/api/operation/pypto-ge.md |
| torch.logical_and | direct | pypto.logical_and | bool/shape 兼容 | 广播风险 | MAP_LOGICAL_AND_INVALID | docs/api/operation/pypto-logical_and.md |
| torch.logical_not | direct | pypto.logical_not | bool/shape 兼容 | 语义误用 | MAP_LOGICAL_NOT_INVALID | docs/api/operation/pypto-logical_not.md |
| torch.isfinite | direct | pypto.isfinite | dtype 支持 | 类型兼容风险 | MAP_ISFINITE_DTYPE_MISMATCH | docs/api/operation/pypto-isfinite.md |
| torch.signbit | direct | pypto.signbit | dtype 支持 | 类型兼容风险 | MAP_SIGNBIT_DTYPE_MISMATCH | docs/api/operation/pypto-signbit.md |

### 3.4 Bitwise
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.bitwise_and | direct | pypto.bitwise_and | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_AND_DTYPE_MISMATCH | docs/api/operation/pypto-bitwise_and.md |
| torch.bitwise_or | direct | pypto.bitwise_or | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_OR_DTYPE_MISMATCH | docs/api/operation/pypto-bitwise_or.md |
| torch.bitwise_xor | direct | pypto.bitwise_xor | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_XOR_DTYPE_MISMATCH | docs/api/operation/pypto-bitwise_xor.md |
| torch.bitwise_not | direct | pypto.bitwise_not | 整数 dtype 支持 | dtype 失配 | MAP_BITWISE_NOT_DTYPE_MISMATCH | docs/api/operation/pypto-bitwise_not.md |
| torch.bitwise_left_shift | direct | pypto.bitwise_left_shift | 整数 dtype 支持 | 溢出风险 | MAP_LSHIFT_OVERFLOW_RISK | docs/api/operation/pypto-bitwise_left_shift.md |
| torch.bitwise_right_shift | direct | pypto.bitwise_right_shift | 整数 dtype 支持 | 符号位语义风险 | MAP_RSHIFT_SEMANTIC_RISK | docs/api/operation/pypto-bitwise_right_shift.md |

### 3.5 Activation / Normalization
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.relu / F.relu | direct | pypto.relu | dtype 满足约束 | 精度差异 | MAP_RELU_DTYPE_MISMATCH | docs/api/operation/pypto-relu.md |
| torch.sigmoid | direct | pypto.sigmoid | dtype 满足约束 | 精度差异 | MAP_SIGMOID_DTYPE_MISMATCH | docs/api/operation/pypto-sigmoid.md |
| torch.softmax | direct | pypto.softmax | dim 与 dtype 满足约束 | 数值稳定性 | MAP_SOFTMAX_CONSTRAINT_VIOLATION | docs/api/operation/pypto-softmax.md |
| F.leaky_relu | direct | pypto.lrelu | 参数与 dtype 合法 | 参数失配 | MAP_LRELU_PARAM_INVALID | docs/api/operation/pypto-lrelu.md |
| F.prelu | direct | pypto.prelu | 权重形状与 dtype 合法 | 权重维度失配 | MAP_PRELU_WEIGHT_MISMATCH | docs/api/operation/pypto-prelu.md |
| F.rms_norm | direct | pypto.rms_norm | 归一化维度合法 | 维度失配 | MAP_RMSNORM_DIM_INVALID | docs/api/operation/pypto-rms_norm.md |
| F.layer_norm | substitute | pypto.rms_norm + 补偿项（若业务允许） | 需确认语义可接受 | 语义差异 | MAP_LAYERNORM_SUBSTITUTE_RISK | docs/api/operation/pypto-rms_norm.md |
| F.gelu | unsupported | 无文档 1:1 API | 无稳定 direct/substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| F.silu | unsupported | 无文档 1:1 API | 无稳定 direct/substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| torch.tanh | unsupported | 无文档 1:1 API | 无稳定 direct/substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |

### 3.6 Reduction / Statistics
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.sum | direct | pypto.sum | reduction 轴/tiling 约束满足 | tile 上限风险 | MAP_SUM_REDUCTION_INVALID | docs/api/operation/pypto-sum.md |
| torch.amax / torch.max(dim) | direct | pypto.amax | reduction 约束满足 | 轴选择风险 | MAP_AMAX_CONSTRAINT_VIOLATION | docs/api/operation/pypto-amax.md |
| torch.amin / torch.min(dim) | direct | pypto.amin | reduction 约束满足 | 轴选择风险 | MAP_AMIN_CONSTRAINT_VIOLATION | docs/api/operation/pypto-amin.md |
| torch.prod | direct | pypto.prod | reduction 约束满足 | 溢出风险 | MAP_PROD_OVERFLOW_RISK | docs/api/operation/pypto-prod.md |
| torch.var | direct | pypto.var | reduction 约束满足 | 维度/精度风险 | MAP_VAR_CONSTRAINT_VIOLATION | docs/api/operation/pypto-var.md |
| torch.cumsum | direct | pypto.cumsum | 轴/shape 约束满足 | 累积溢出风险 | MAP_CUMSUM_OVERFLOW_RISK | docs/api/operation/pypto-cumsum.md |
| torch.mean | substitute | pypto.sum + 元素计数 + pypto.div | sum/div 均可用且计数合法 | 数值偏差 | MAP_MEAN_SUBSTITUTE_RISK | docs/api/operation/pypto-sum.md; docs/api/operation/pypto-div.md |
| torch.std | substitute | pypto.var + pypto.sqrt | var/sqrt 均可用 | 精度与性能风险 | MAP_STD_SUBSTITUTE_RISK | docs/api/operation/pypto-var.md; docs/api/operation/pypto-sqrt.md |

### 3.7 Matrix / Linear Algebra
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.matmul | direct | pypto.matmul | cube tile/dtype/format 约束满足 | tile 失配 | MAP_MATMUL_TILE_INVALID | docs/api/operation/pypto-matmul.md; docs/api/config/pypto-set_cube_tile_shapes.md |
| torch.bmm | direct | pypto.matmul | 3D 维度合法 | batch 维失配 | MAP_BMM_DIM_MISMATCH | docs/api/operation/pypto-matmul.md |
| F.linear | direct | pypto.matmul (+ bias add) | weight/bias 维度合法 | 维度与布局风险 | MAP_LINEAR_DIM_INVALID | docs/api/operation/pypto-matmul.md; docs/api/operation/pypto-add.md |
| torch.mm | direct | pypto.matmul | 2D 维度合法 | 维度失配 | MAP_MM_DIM_MISMATCH | docs/api/operation/pypto-matmul.md |
| torch.addmm | substitute | pypto.matmul + pypto.add | matmul/add 约束同时满足 | dtype 对齐风险 | MAP_ADDMM_SUBSTITUTE_RISK | docs/api/operation/pypto-matmul.md; docs/api/operation/pypto-add.md |
| torch.baddbmm | substitute | pypto.matmul + pypto.add | batch 维与 dtype 合法 | 维度失配 | MAP_BADDBMM_SUBSTITUTE_RISK | docs/api/operation/pypto-matmul.md; docs/api/operation/pypto-add.md |
| F.conv1d/2d/3d | direct | pypto.conv | 平台/形状/参数满足 conv 约束 | 平台不支持风险 | MAP_CONV_PLATFORM_LIMIT | docs/api/operation/pypto-conv.md |

### 3.8 Index / Scatter / Sort
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.gather | direct | pypto.gather | dim/view/tile 约束满足 | 维度切分非法 | MAP_GATHER_DIM_INVALID | docs/api/operation/pypto-gather.md |
| torch.scatter | direct | pypto.scatter / pypto.scatter_ | dim/index 约束满足 | in-place 语义风险 | MAP_SCATTER_CONSTRAINT_VIOLATION | docs/api/operation/pypto-scatter.md; docs/api/operation/pypto-scatter_.md |
| torch.scatter_add | substitute | pypto.scatter_update 或 index_add_ 组合 | index 轴约束满足 | 语义偏差风险 | MAP_SCATTER_ADD_SUBSTITUTE_RISK | docs/api/operation/pypto-scatter_update.md; docs/api/operation/pypto-index_add_.md |
| torch.index_select | direct | pypto.index_select | index dtype/axis 约束满足 | 轴失配 | MAP_INDEX_SELECT_INVALID | docs/api/operation/pypto-index_select.md |
| torch.index_add | direct | pypto.index_add / pypto.index_add_ | index 轴与 dtype 合法 | in-place 风险 | MAP_INDEX_ADD_INVALID | docs/api/operation/pypto-index_add.md; docs/api/operation/pypto-index_add_.md |
| torch.index_put_ | direct | pypto.indexput_ | 索引语义满足约束 | in-place 覆盖风险 | MAP_INDEXPUT_INPLACE_RISK | docs/api/operation/pypto-indexput_.md |
| torch.where | direct | pypto.where | 条件与输入 shape 合法 | 广播风险 | MAP_WHERE_CONSTRAINT_VIOLATION | docs/api/operation/pypto-where.md |
| torch.topk | direct | pypto.topk | 轴/tiling 约束满足 | 轴受限风险 | MAP_TOPK_AXIS_INVALID | docs/api/operation/pypto-topk.md |
| torch.argsort | direct | pypto.argsort | dim/tiling 约束满足 | 维度限制风险 | MAP_ARGSORT_DIM_INVALID | docs/api/operation/pypto-argsort.md |
| F.one_hot | direct | pypto.one_hot | 类别维度与 dtype 合法 | 维度爆炸风险 | MAP_ONEHOT_DIM_RISK | docs/api/operation/pypto-one_hot.md |

### 3.9 Shape / Layout / Tensor Construction
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.reshape | direct | pypto.reshape | 目标形状合法 | 形状失配 | MAP_RESHAPE_INVALID | docs/api/operation/pypto-reshape.md |
| torch.view | direct | pypto.view | view 约束与 valid_shape 合法 | 精度/拓扑风险 | MAP_VIEW_VALID_SHAPE_RISK | docs/api/operation/pypto-view.md; docs/tutorials/appendix/faq.md |
| torch.transpose | direct | pypto.transpose | 轴交换模式在支持范围内 | 模式不可达 | MAP_TRANSPOSE_MODE_UNSUPPORTED | docs/api/operation/pypto-transpose.md |
| torch.permute | substitute | 有限 transpose 组合 | 必须可分解为受支持 transpose 序列 | 组合不可达 | MAP_PERMUTE_LIMITED_SUPPORT | docs/api/operation/pypto-transpose.md |
| torch.unsqueeze | direct | pypto.unsqueeze | 维度插入合法 | 维度错误 | MAP_UNSQUEEZE_DIM_INVALID | docs/api/operation/pypto-unsqueeze.md |
| torch.cat | direct | pypto.concat | 连接轴与输入 shape 合法 | 维度不一致 | MAP_CONCAT_DIM_MISMATCH | docs/api/operation/pypto-concat.md |
| torch.clone | direct | pypto.clone | dtype/shape 合法 | 内存开销风险 | MAP_CLONE_MEMORY_RISK | docs/api/operation/pypto-clone.md |
| torch.pad | direct | pypto.pad | padding 参数合法 | 维度/边界风险 | MAP_PAD_PARAM_INVALID | docs/api/operation/pypto-pad.md |
| torch.tril | direct | pypto.tril | 输入维度合法 | 布局风险 | MAP_TRIL_CONSTRAINT_VIOLATION | docs/api/operation/pypto-tril.md |
| torch.triu | direct | pypto.triu | 输入维度合法 | 布局风险 | MAP_TRIU_CONSTRAINT_VIOLATION | docs/api/operation/pypto-triu.md |
| torch.zeros | direct | pypto.zeros | shape/dtype 合法 | dtype 失配 | MAP_ZEROS_DTYPE_MISMATCH | docs/api/operation/pypto-zeros.md |
| torch.ones | direct | pypto.ones | shape/dtype 合法 | dtype 失配 | MAP_ONES_DTYPE_MISMATCH | docs/api/operation/pypto-ones.md |
| torch.full | direct | pypto.full | shape/dtype/value 合法 | dtype 失配 | MAP_FULL_DTYPE_MISMATCH | docs/api/operation/pypto-full.md |
| torch.arange | direct | pypto.arange | start/end/step 与 dtype 合法 | 步长非法 | MAP_ARANGE_STEP_INVALID | docs/api/operation/pypto-arange.md |
| torch.flatten | substitute | pypto.reshape | 展平维度可解析 | 维度错配 | MAP_FLATTEN_SUBSTITUTE_RISK | docs/api/operation/pypto-reshape.md |

### 3.10 明确 unsupported（常见但当前不建议映射）
| torch_op | tier | pypto_api_or_recipe | preconditions | risk | warning_code | evidence |
|---|---|---|---|---|---|---|
| torch.roll | unsupported | 无稳定 1:1 API | 无文档 API | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| torch.flip | unsupported | 无稳定 1:1 API | 无文档 API | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| F.dropout | unsupported | 推理路径不提供训练态 dropout 语义 | 语义不一致 | 设计期阻断 | MAP_UNSUPPORTED_TRAINING_OP | docs/api/operation/index.md |
| F.batch_norm | unsupported | 无文档 1:1 API | 无稳定 substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| F.group_norm | unsupported | 无文档 1:1 API | 无稳定 substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
| F.instance_norm | unsupported | 无文档 1:1 API | 无稳定 substitute 语义 | 设计期阻断 | MAP_UNSUPPORTED_API | docs/api/operation/index.md |
```

### C) 新增“unsupported 前置告警模板”段落（原样加入）

```markdown
## 4. unsupported 前置告警模板

- 模板一（API 不存在）
  - 触发：目标 Torch op 在 docs/api/operation 下无对应 API 且无可靠 substitute。
  - 告警：`[MAP_UNSUPPORTED_API] {torch_op} 当前无可落地映射，请在设计阶段调整算子路径。`

- 模板二（关键约束不可满足）
  - 触发：存在候选 API，但 dtype/shape/format/runtime 任一硬约束不满足。
  - 告警：`[MAP_HARD_CONSTRAINT_FAILED] {torch_op} 映射失败：{constraint_name} 不满足。`
```

### D) 删除项（明确执行）

- 删除所有“仅写来源目录、不写具体文档”的规则行。
- 删除所有引用不存在 API 文件名的直接映射条目。

## 11.2 `tiling_rules.md` 具体修改稿

### A) 顶层结构替换为以下 6 段

```markdown
# Tiling 规则

## 1. 算子分类与决策入口
## 2. HARD 约束
## 3. HEURISTIC 建议
## 4. 混合算子（Cube+Vector）
## 5. 非可视化校验清单
## 6. 失败签名与定位
```

### B) 在“HARD 约束”中插入以下具体规则行

```markdown
| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---|---|---|---|---|---|---|
| TILE_VEC_01 | set_vec_tile_shapes 维度数与目标张量维度一致 | HARD | vec 算子配置 tiling | 编译失败 | TILE_VEC_DIM_MISMATCH | docs/api/config/pypto-set_vec_tile_shapes.md |
| TILE_VEC_01A | set_vec_tile_shapes 参数个数不超过 4 且每维 > 0 | HARD | vec 算子配置 tiling | 编译失败 | TILE_VEC_ARG_INVALID | docs/api/config/pypto-set_vec_tile_shapes.md; docs/tutorials/development/tiling.md |
| TILE_VEC_02 | vec 尾轴满足对齐约束 | HARD | vec tiling 配置 | 运行失败/性能异常 | TILE_VEC_ALIGN_INVALID | docs/tutorials/development/tiling.md |
| TILE_EXPR_01 | 控制 (TensorShape/TileShape)*(1+算子输入个数) < 18000 | HARD | tile 过小导致切分过多 | 表达式表编译失败 | TILE_EXPR_TABLE_OVERFLOW | docs/tutorials/development/tiling.md |
| TILE_RED_01 | reduction tile 满足容量与轴限制 | HARD | sum/amax/amin/prod/var | 编译失败或结果异常 | TILE_REDUCTION_INVALID | docs/api/operation/pypto-sum.md; docs/api/operation/pypto-var.md |
| TILE_CUBE_01 | cube tile 满足 32-byte 对齐与层级关系 | HARD | matmul 场景 | 编译/运行失败 | TILE_CUBE_ALIGN_INVALID | docs/api/config/pypto-set_cube_tile_shapes.md |
| TILE_CUBE_02 | L0/L1 与 buffer 预算满足上界 | HARD | matmul 场景 | 运行失败 | TILE_CUBE_BUFFER_OVERFLOW | docs/api/config/pypto-set_cube_tile_shapes.md |
| TILE_CUBE_03 | split-k 仅在支持条件启用 | HARD | enable_split_k=true | 性能退化或错误 | TILE_SPLITK_UNSUPPORTED | docs/api/config/pypto-set_cube_tile_shapes.md |
| TILE_CUBE_04 | 满足 0 < mL0<=mL1,kL0<=kL1,nL0<=nL1 且 L1%L0==0 | HARD | cube tile 配置 | 编译失败 | TILE_CUBE_DIVISIBILITY_INVALID | docs/api/config/pypto-set_cube_tile_shapes.md |
| TILE_CUBE_05 | Bias/FixPipe 场景满足 nL0*4<=1KB 与 nL0*8<=2KB | HARD | bias/fixpipe 场景 | 运行失败 | TILE_CUBE_SPECIAL_BUFFER_OVERFLOW | docs/api/config/pypto-set_cube_tile_shapes.md |
| TILE_CUBE_06 | 输入为3D/4D时 enable_split_k 只能为 False | HARD | 高维 matmul 场景 | 编译或运行失败 | TILE_SPLITK_DIM_UNSUPPORTED | docs/api/config/pypto-set_cube_tile_shapes.md |
```

### C) 在“HEURISTIC 建议”中加入具体建议行（明确非硬约束）

```markdown
| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---|---|---|---|---|---|---|
| TILE_HEUR_01 | 优先从中等 tile 起步，避免过小导致循环开销过高 | HEURISTIC | 初次调优 | 性能不稳定 | TILE_HEUR_SMALL_TILE | docs/tutorials/debug/performance.md |
| TILE_HEUR_02 | 混合链路先定 Cube 主干再定 Vector 尾部 | HEURISTIC | matmul+elementwise | 局部瓶颈 | TILE_HEUR_MIXED_ORDER | docs/tutorials/development/tiling.md |
```

### D) 将“混合算子策略 TODO”替换为以下可执行段落

```markdown
## 4. 混合算子（Cube+Vector）

执行顺序：
1. 识别主算力阶段（通常为 matmul）。
2. 对主阶段配置 cube tile 并验证硬约束。
3. 对尾部逐元素/归约阶段配置 vec tile。
4. 校验跨阶段 shape/dtype/format 连贯性。
5. 任一步失败触发：`TILE_MIXED_PIPELINE_INVALID`。
```

### E) “非可视化校验清单”插入以下内容

```markdown
- 对齐检查：PASS/FAIL
- Buffer 预算：PASS/FAIL
- 轴切分合法性：PASS/FAIL
- 编译日志关键错误签名：是否命中
```

## 11.3 `loop_strategy.md` 具体修改稿

### A) 全文先执行一个确定性替换

- 将所有 `docs/tutorials/development/loop.md` 替换为 `docs/tutorials/development/loops.md`。

### B) 顶层结构替换为以下 8 段

```markdown
# Loop 策略

## 1. 是否需要 loop（判定树）
## 2. 静态轴 vs 动态轴
## 3. 标准写法模板
## 4. loop_unroll 使用边界
## 5. submit_before_loop 触发条件
## 6. 循环合并与尾块处理
## 7. 失败签名与规避
## 8. 证据索引
```

### C) “静态轴 vs 动态轴”段落写成以下明确规则

```markdown
| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---|---|---|---|---|---|---|
| LOOP_AXIS_01 | 静态轴优先 Python for，不优先 pypto.loop | HARD | 轴长度编译期可知 | 编译复杂度增加 | LOOP_STATIC_AXIS_MISUSE | docs/tutorials/debug/performance.md |
| LOOP_AXIS_02 | 动态轴使用 pypto.loop 并补齐边界控制 | HARD | 轴长度运行期确定 | 运行错误/结果异常 | LOOP_DYNAMIC_AXIS_INVALID | docs/tutorials/development/loops.md |
| LOOP_UNROLL_01 | 多层嵌套时仅最内层 loop_unroll 可使用 unroll_list | HARD | 嵌套循环场景 | 展开策略失效/性能退化 | LOOP_UNROLL_NESTED_INVALID | docs/tutorials/debug/performance.md |
| LOOP_UNROLL_02 | unroll 因子遵循排序去重且包含 1 的约束语义 | HARD | 配置 unroll_list | 展开档位不可控 | LOOP_UNROLL_FACTOR_INVALID | docs/api/controlflow/pypto-loop_unroll.md |
| LOOP_VIEW_01 | 动态轴场景 view 的 shape 范围不可过小 | HARD | 动态 shape + loop | 限制 TileShape 导致性能劣化 | LOOP_VIEW_TILE_TOO_SMALL | docs/tutorials/debug/performance.md |
```

### D) 增加“模板化写法”段落（原样加入）

```markdown
### 动态轴 loop 模板（语义模板）
- 使用 `for i in pypto.loop(start, end, step)` 管理动态区间。
- 在循环体内先做 view，再做 compute，最后写回。

### loop_unroll 模板
- 仅在热点且分支简单场景启用。
- 若编译时长显著上升，回退 unroll。

### 依赖链模板
- 当后续循环依赖前序结果时，显式约束调度顺序。
- 必要时启用 `submit_before_loop`。
```

### E) “失败签名与规避”插入以下条目

```markdown
| warning_code | trigger | impact | mitigation | evidence |
|---|---|---|---|---|
| LOOP_UNROLL_COMPILE_BLOWUP | 过度 unroll | 编译时间爆炸 | 降低或关闭 unroll | docs/tutorials/debug/performance.md |
| LOOP_DEPENDENCY_HAZARD | 循环间读写依赖未约束 | 结果错误 | 强制串行依赖 | docs/tutorials/development/loops.md |
| LOOP_TAIL_VALIDITY_MISMATCH | 尾块有效区间处理不当 | 数值错误 | 增加有效区间校验 | docs/tutorials/development/loops.md |
| LOOP_MERGE_MISUSE | 不当合并循环 | 性能或正确性问题 | 回退为独立循环 | docs/tutorials/debug/performance.md |
```

## 11.4 `performance_params.md` 具体修改稿

### A) 顶层结构替换为以下 7 段

```markdown
# 性能参数规则

## 1. 参数分层（baseline / tuning）
## 2. pass_options 规范表
## 3. runtime_options 规范表
## 4. 按算子类型组合建议
## 5. 误配风险库
## 6. 非可视化验证路径
## 7. 证据索引
```

### B) 参数表字段固定（原样加入）

```markdown
`option_key | layer | scope | default | allowed_values | recommended_values | side_effect | warning_code | evidence`
```

### C) 插入 pass/runtime 具体条目（替换占位符后可直接使用）

```markdown
| option_key | layer | scope | default | allowed_values | recommended_values | side_effect | warning_code | evidence |
|---|---|---|---|---|---|---|---|---|
| set_pass_options.vec_nbuffer_setting | baseline | 编译期 | {} | {-1:1} / {} / {-1:N,0:N2,...} | {}（先自动） | 误配会导致子图合并异常或性能波动 | PERF_PASS_OPTION_CONFLICT | docs/api/config/pypto-set_pass_options.md |
| set_pass_options.cube_l1_reuse_setting | baseline | 编译期 | {} | {-1:1} / {} / {-1:N,0:N1,...} | {}（先自动） | 误配会导致L1复用失效或收益不稳 | PERF_PASS_OPTION_CONFLICT | docs/api/config/pypto-set_pass_options.md |
| set_pass_options.cube_nbuffer_setting | baseline | 编译期 | {-1:1} | {-1:1} / {} / {-1:N,0:N1,...} | {-1:1}（先保守） | 误配会导致AIC子图合并不当 | PERF_PASS_OPTION_CONFLICT | docs/api/config/pypto-set_pass_options.md |
| set_pass_options.sg_set_scope | tuning | 编译期 | -1 | [-1,2147483647] | -1（默认），按需设置统一scope | scope误设可能阻断合理合图 | PERF_PASS_SCOPE_MISUSE | docs/api/config/pypto-set_pass_options.md |
| set_runtime_options.device_sched_mode | baseline | 运行期 | 0 | 0/1/2/3 | 0（默认）或1（L2亲和） | 高级模式可能引入调度管理开销 | PERF_DEVICE_SCHED_MODE_MISUSE | docs/api/config/pypto-set_runtime_options.md |
| set_runtime_options.stitch_function_max_num | baseline | 运行期 | 128 | [1,1024] | 128 起步，按内存和并行度调优 | 值过大提升并行但增加workspace占用 | PERF_STITCH_WORKSPACE_RISK | docs/api/config/pypto-set_runtime_options.md |
| set_runtime_options.run_mode | baseline | 运行期 | 有 CANN 环境变量→0(NPU)，否则1(模拟器) | 0/1 | NPU可用时取0 | 与设备/环境不匹配会运行失败 | PERF_RUN_MODE_ENV_MISMATCH | docs/api/config/pypto-set_runtime_options.md; docs/install/prepare_environment.md; AGENTS.md |
| set_runtime_options.valid_shape_optimize | tuning | 运行期 | 0 | 0/1 | 动态shape主块占比高时考虑1 | 不当开启可能增加编译复杂度 | PERF_VALID_SHAPE_OPT_MISUSE | docs/api/config/pypto-set_runtime_options.md |
```

### D) 在“误配风险库”中加入以下固定条目

```markdown
| warning_code | trigger | impact | mitigation | evidence |
|---|---|---|---|---|
| PERF_RUN_MODE_ENV_MISMATCH | run_mode 与设备/环境不匹配 | 运行失败 | 修正 run_mode 与设备配置 | docs/api/config/pypto-set_runtime_options.md; docs/install/prepare_environment.md |
| PERF_PASS_OPTION_CONFLICT | pass 参数组合冲突 | 编译失败/性能下降 | 回退冲突参数，最小集重试 | docs/api/config/pypto-set_pass_options.md |
| PERF_OVERTUNE_STABILITY_RISK | 过激调优参数 | 稳定性下降 | 回退至 baseline 再逐步启用 | docs/tutorials/debug/performance.md |
```

### E) “非可视化验证路径”插入以下可执行清单

```markdown
1. 编译日志检查：是否出现 pass/runtime 相关错误签名。
2. 运行模式检查：run_mode 是否与环境一致。
3. 数值检查：关键输出与 golden 对比是否在阈值内。
4. 约束检查：参数是否命中文档允许范围。
```

## 11.5 四文件联动的具体替换动作清单

按顺序执行：

1. 在 `api_mapping.md` 写入第 11.1 的结构、闸门、映射矩阵样例、unsupported 模板。
2. 在 `tiling_rules.md` 写入第 11.2 的 HARD/HEURISTIC 分层与混合策略替换块。
3. 在 `loop_strategy.md` 先做路径纠错，再写入第 11.3 的规则表与失败签名表。
4. 在 `performance_params.md` 写入第 11.4 的参数字段、风险库、非可视化检查。

完成后再统一检查：

- 无 TODO。
- 无不存在 API 声明。
- 所有 HARD 规则都有证据路径。
- 告警码不重名异义。
