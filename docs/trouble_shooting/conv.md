# CONV 组件错误码

（待补充）

- **范围**：FC6XXX - FC8XXX
- 本文档说明 CONV 子类 OP 的错误码定义、场景说明与排查建议。

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/interface/utils/conv_error.h` 文件。
---

## 错误码定义和场景说明

### 1. Operation（Operation非法拦截类报错，`ConvError::Operation`，F61xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `ATTR_SPEC_INVALID` | **F6101** | `conv.operation.checkfmap` | Operation校验输入规格参数不合法（input,weight,output的维度，shape，数据类型等）。 |
| `L1_TILING_INVALID` | **F6102** | `conv.operation.checkweight` | Operation校验输入Tiling参数不合法（L1层级Tiling不合法）。 |
| `L0_TILING_INVALID` | **F6103** | `conv.operation.checkbias` | Operation校验输入Tiling参数不合法（L0层级Tiling不合法）。 |
| `UNKNOWN` | **F6199** | `conv.operation.reserved` | Operation阶段未知报错预留错误码。 |

### 2. Tile切分（Tile图切分，`ConvError::ExpandFunction`，F62xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `EXPANDFUNC_TENSOR_OP_NULLPTR` | **F6201** | `conv.expandfunc.tensor_nullptr` | Tile图切分，tensor图处理节点空指针报错。 |
| `EXPANDFUNC_TENSOR_ATTR_GET_FAILED` | **F6202** | `conv.expandfunc.get_attr` | Tile图切分，tensor图节点属性获取失败。 |
| `EXPANDFUNC_TILE_OP_NULLPTR` | **F6203** | `conv.expandfunc.tile_nullptr` | Tile图切分，tile图新生成节点空指针报错。 |
| `EXPANDFUNC_PARAMS_INVALID` | **F6204** | `conv.expandfunc.params_check` | Tile图切分，参数不匹配错误（维度，类型，Tile块配置）。 |
| `EXPANDFUNC_INNER_STATUS_FAILED` | **F6205** | `conv.expandfunc.check_status` | Tile图切分，内部功能函数返回值异常错误。 |
| `UNKNOWN` | **F6299** | `conv.operation.reserved` | ExpandFunc Tile图切分阶段未知报错预留错误码。 |

### 3. CodenGen
### 4. TileOp

---

## 排查建议

### Operation shape/TileShape 拦截编译报错
可根据报错参考约束说明：`docs/api/config/pypto-set_conv_tile_shapes.md`


### Pass 图阶段 拦截编译报错
1. 打开编译debug模式，dump pass阶段图，配置 `debug_options={"compile_debug_mode": 1}`
```python
@pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
def conv_kernel()
```

2. 复跑问题用例，在output下生成对应时间戳的dump结果，根据报错日志所示图阶段，使用pto-toolkit打开，查看执行图阶段之前的dump图，对conv operation 切成的 Tile子图进行排查；

