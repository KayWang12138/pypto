# pypto.set\_pass\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[Union[int, str], int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[Union[int, str], int]] = None,
                     cube_nbuffer_setting: Optional[Dict[Union[int, str], int]] = None,
                     sg_set_scope: Optional[int] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | 输入      | 含义：合图参数，用于配置相同结构AIV子图的合并数量。 <br> 说明：该参数适用于结构相同的AIV子图合并。 <br> 类型：dict[Union[int, str], int]，支持整数 key（hashorder）和字符串 key（semantic_label）混合使用。 <br> 取值：<br> {-1: 1}：跳过AIV子图合并 <br> {} （空字典）：自动合并，根据AIV核心数自动计算合并粒度<br> {-1: N, 0: N2, ...}：手动合并，默认粒度为N <br> {"label": N}：按语义标签设置合并粒度，字符串 key 的值会直接替换对应同构子图组的合并粒度（详见下方语义标签配置说明） <br> 默认值：{} 空字典 <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_setting | 输入 | 含义：合图参数，用于配置重复搬运同一GM数据的子图合并数量。<br> 说明：该参数适用于含有CUBE计算的子图合并。 <br> 类型：dict[Union[int, str], int]，支持整数 key（hashorder）和字符串 key（semantic_label）混合使用。 <br> 取值：<br>{-1: 1}：跳过L1Reuse合并 <br> {} （空字典）：自动合并，根据AIC核心数自动计算合并粒度<br> {-1: N, 0: N1, ...}：手动合并，默认合并粒度为N。 <br> {"label": N}：按语义标签设置合并粒度，字符串 key 的值会直接替换对应子图的合并粒度（详见下方语义标签配置说明） <br> 默认值：{} 空字典 <br> 影响Pass范围：L1ReuseMerge |
| cube_nbuffer_setting    | 输入      | 含义：合图参数，用于配置相同结构AIC子图的合并数量。 <br> 说明：该参数适用于结构相同的AIC子图合并。 <br> 类型：dict[Union[int, str], int]，支持整数 key（hashorder）和字符串 key（semantic_label）混合使用。 <br> 取值：<br>{-1: 1}：跳过AIC子图合并 <br> {} （空字典）：自动合并，根据AIC核心数自动计算合并粒度<br> {-1: N, 0: N1, ...}：手动合并，默认合并粒度为N <br> {"label": N}：按语义标签设置合并粒度，字符串 key 的值会直接替换对应同构子图组的合并粒度（详见下方语义标签配置说明） <br>默认值：{-1: 1} <br> 影响Pass范围： L1ReuseMerge |
| sg_set_scope            | 输入      | 含义：手动控制合图参数。 <br> 说明：将operation赋予特定的scopeId，若相邻的operation具有相同的非-1的scopeId，则会被强制合并在一个子图之中，并且这个子图不会与其他子图合并。该参数仅对存在上下游连接通路的operation生效，例如operation A的输出作为operation B的输入即构成此类连接通路。 <br> 类型：int <br> 取值范围：-1~2147483647 <br> 默认值：-1 <br> 影响Pass范围：GraphPartition <br> 配置建议：1）视图类Operation与其对应的计算类Operation应配置相同的scopeId。2）Reshape Operation较为特殊，部分场景会单独成子图，手动控制合图行为可能失效。|

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。
- 语义标签key：字符串 key 必须与至少一个 operation 通过 `pypto.set_semantic_label` 设置的 semantic_label 完全匹配，否则编译时报错。

## 调用示例

```python
   # 纯整数 key 配置
   pypto.set_pass_options(
                       vec_nbuffer_setting={-1: 2},
                       cube_l1_reuse_setting={},
                       cube_nbuffer_setting={-1: 1, 1: 2})

   # 混合整数 key 和语义标签 key 配置
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={-2: 1, -1: 2, "V1": 1})

   # 纯语义标签 key 配置
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```

### dict类型配置说明
#### 整数键值对含义
Key (hashorder): 同构子图组id。<br>
- 值 M: 匹配 hashorder 为 M 的特定子图组。<br>
- 值 -1: 匹配所有未显式指定的子图组。<br>

Value (N): 表示合并粒度。即：同构子图组内每N个子图合并为一个新子图执行。<br>
#### 配置行为
Pass 在处理子图合并时，遵循 "精确匹配 > 默认配置 > 自动处理" 的逻辑：<br>
- 精确匹配: 若 hashorder 命中字典中的特定 Key，则按其对应的 Value N 进行合并。<br>
- 默认配置: 若未精确命中，但字典中存在 -1，则按 -1 对应的 Value 执行合并。<br>
- 自动处理: 若既未精确命中也无 -1 配置，则自动计算合并粒度进行合并优化。<br>
#### 配置示例
| 配置                  | 说明                                                                 |
|---------------------- |----------------------------------------------------------------------|
|{-1: 1}|跳过子图合并。合并粒度为1，即所有同构子图组内的子图不进行合并。|
|{0: 5}|对于hashorder为0的同构子图组，每5个子图合并为一个子图；<br>其他同构子图组，根据硬件核心数自动计算合并粒度并进行合并。|
|{0: 5, 2: 8, -1: 2}    |hashorder为0的同构子图组，每5个子图合并为一个子图；<br>hashorder为2的同构子图组，每8张子图合并为一个子图；<br>其他的同构子图组使用-1对应的默认合并粒度，即每2张子图合并为一个子图。<br> |
|{0: 5, -1: 1}    |hashorder为0的同构子图组，每5个子图合并为一个子图；<br>其他同构子图组不做处理。 |

### 语义标签（semantic_label）key 配置说明

#### 功能概述

除整数 key（hashorder）外，`vec_nbuffer_setting`、`cube_l1_reuse_setting` 和 `cube_nbuffer_setting` 还支持使用字符串 key，即通过 `pypto.set_semantic_label` 设置的语义标签名称。字符串 key 允许用户精确控制特定 operation 所在子图（允许多个）的合并粒度，无需关心其 hashorder 编号。

#### 字符串键值对含义

Key (label): 语义标签名称，必须与至少一个 operation 的 `semantic_label` 完全匹配。<br>
Value (N): 表示合并粒度。<br>

#### 优先级机制

字符串 key 的优先级**高于**整数 key。处理流程为：<br>
1. 首先根据整数 key（hashorder）确定各同构子图组的合并粒度。<br>
2. 然后字符串 key 的值**直接替换**（而非取 max）对应子图组的合并粒度。<br>
3. 当多个不同的字符串 label 指向同一个同构子图组时，取这些 label 值中的最大值。<br>

#### vec_nbuffer_setting / cube_nbuffer_setting 的语义标签行为

字符串 key 覆盖其所在 operation 对应的**整个同构子图组**的合并粒度。

#### cube_l1_reuse_setting 的语义标签行为

与 vec_nbuffer_setting 和 cube_nbuffer_setting 不同，cube_l1_reuse_setting 的字符串 key **仅作用于包含对应标签 operation 的子图**，不展开到整个同构组。即同构组内可能只有部分子图被字符串 key 覆盖，其他子图保持整数 key 的值。

#### 语义标签配置示例

| 配置                                | 说明                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
|{-1: 2, "V1": 1}|所有同构子图组默认合并粒度为2；但 V1 标签所在的同构子图组合并粒度被替换为1。|
|{"V1": 3}|V1 标签所在的同构子图组合并粒度为3；其他同构子图组自动计算合并粒度。|
|{-1: 2, "V1": 1, "V2": 3}|默认合并粒度为2；V1 所在组替换为1；V2 所在组替换为3。若某一组有同时有 V1 和 V2 两种OP，则取 max(1, 3) = 3。|
