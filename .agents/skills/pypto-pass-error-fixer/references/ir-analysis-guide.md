# PyPTO IR分析指导文档

## 文档目的

本文档旨在指导AI Agent理解和分析PyPTO的.tifwkgr中间表达（IR）文件，用于：
- 辅助问题定位和调试
- 检查Pass模块是否存在逻辑错误
- 验证数据流和内存访问的正确性
- 评估IR转换的合理性

## IR概述

### 什么是IR

IR（Intermediate Representation）是PyPTO编译器在各个Pass阶段生成的中间表达，用于表示计算图、数据流和内存布局。每个Pass都会生成Before和After两个IR文件，用于对比Pass执行前后的变化。

### IR的作用

1. **问题定位**：通过IR变化定位Pass执行中的错误
2. **逻辑验证**：检查数据流、内存访问、依赖关系的正确性
3. **性能分析**：评估优化Pass的效果
4. **调试辅助**：理解编译器内部状态和转换过程

## IR文件结构

### 文件格式

.tifwkgr文件采用文本格式，包含以下主要部分：

```
-------------
Function {函数名}[索引] {hash} {图类型} {
  RAWTENSOR声明
  INCAST/OUTCAST声明
  
  操作节点定义
}
```

### 文件头

```
Function TENSOR_TENSOR_update_kernel_loop_Unroll1_PATH0_hiddenfunc0_5[5] 311644491877735055 DYNAMIC_LOOP_PATH TILE_GRAPH {
```

**字段说明：**
- `Function`: 固定关键字
- `函数名`: 当前处理的函数名称
- `[索引]`: 索引
- `{hash}`: 唯一标识符
- `{图类型}`: TENSOR_GRAPH 或 TILE_GRAPH
- `{图属性}``:` 图的额外属性（如DYNAMIC_LOOP_PATH）

**图类型说明：**
- `TENSOR_GRAPH`: 高层张量图，表示原始的计算逻辑，未进行tiling优化
- `TILE_GRAPH`: 分块图，已进行tiling优化，包含内存访问细节

### 注释

```
/* /mnt/workspace/gitCode/cann/pypto/test_scatter_update.py:20 */
```

注释标记了原始Python代码的行号，用于IR与源代码的对应关系。

## IR元素详解

### 1. RAWTENSOR（原始张量）

#### 语法格式

```
RAWTENSOR[索引] <shape> @{编号}"{名称}"
```

#### 示例

```
RAWTENSOR[  0] <1 x 16 x DT_INT_INT32> @10"TENSOR_3"
RAWTENSOR[  1] <8 x 128 x DT_FP32> @12"TENSOR_2"
RAWTENSOR[  2] <16 x 128 x DT_FP32> @14"TENSOR_1"
```

#### 字段说明

- **索引**: 张量的索引（0, 1, 2, ...）
- **shape**: 张量的形状和数据类型
  - 格式：`dim1 x dim2 x ... x DT_数据类型`
  - 数据类型：DT_INT32, DT_FP32, DT_FP16等
- **编号**: 张量的编号，唯一标识@10, @12, @14等）
- **名称**: 张量的名称，可能为空

#### 分析要点

1. **索引唯一性**: 确保每个RAWTENSOR索引唯一
2. **形状合理性**: 检查shape的维度和大小是否合理
3. **数据类型**: 验证数据类型是否匹配计算需求

### 2. INCAST/OUTCAST（输入tensor/输出tensor）

#### 语法格式

```
INCAST[索引] <shape / valid_shape> %{logic tensor编号或名称}@{raw tensor编号或名称}#(xx) fromSlot[槽位]
OUTCAST[索引] <shape / valid_shape> %{logic tensor编号或名称}@{raw tensor编号或名称}#(xx) toSlot[槽位]
```

#### 示例

```
INCAST[  0]  <1 x 16 x DT_INT32 / 1 x 16 x DT_INT32> %6@10#(-1) fromSlot[2]
OUTCAST[  0]  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %14@16#(-1) toSlot[3]
```

#### 字段说明

- **索引**: 索引位置
- **shape**: 张量的形状
- **valid_shape**: 张量的有效形状
- **logic tensor编号或名称**: 标识符（%6, %9, %aaa等）
- **raw tensor编号或名称**: 标识符（@6, @9, @aaa等）

### 3. 操作节点

#### 语法格式

输出tensor列表 = op_id? opcode 输入参数列表？属性列表？ // 输入参数列表和属性列表可以为空

```
<shape / valid_shape> %{输出logic tensor编号或名称}@{输出raw tensor编号或名称}#(xx){内存类型} = !{op_id} {opcode}(参数) %{输入logic tensor编号或名称}@{输入raw tensor编号或名称}#(xx){内存类型} #{属性}
```

#### 示例

```
<1 x 16 x DT_INT32 / 1 x 16 x DT_INT32> %7@11#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10005 VIEW(g:-1, s:-1) %6@10#(-1)MEM_UNKNOWN::MEM_UNKNOWN from offset:[  0,  0] dynoffset:[  0,  0] to MEM_UNKNOWN dynvalidshape:[  1, 16] #IS_GLOBAL_INPUT{1}

<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %1@6#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10001 INDEX_OUTCAST(g:-1, s:-1) %10@13#(-1)MEM_UNKNOWN::MEM_UNKNOWN, %0@5#(-1)MEM_UNKNOWN::MEM_UNKNOWN, %13@15#(-1)MEM_UNKNOWN::MEM_UNKNOWN #CACHE_MODE{PA_BSND} #PA_NZ_BLOCK_SIZE{1} #axis{0}

<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %76@17#(0)MEM_UB::MEM_UB = !10015 TILE_ADDS(g:0, s:-1) %87@52#(0)MEM_UB::MEM_UB #IS_CUBE{0} #SCALAR{1.000000} #last_use{[0, 1]} #op_attr_reverseOperand{0}
```

#### 字段说明

- **shape**: 张量的形状
- **valid_shape**: 张量的有效形状
- **logic tensor编号或名称**: 标识符（%6, %9, %aaa等）
- **raw tensor编号或名称**: 标识符（@6, @9, @aaa等）
- **内存类型**: 变量的内存类型
- **op_id**: 操作的唯一标识符（!10005, !10001等）
- **opcode**: 操作的名称
- **参数**: 操作的输入参数（变量列表、常量等）
- **属性**: 操作的属性配置

#### 运行时参数

| 参数名 | 说明 | 示例 |
|-------|------|------|
| RUNTIME_COA_GET_PARAM_OFFSET(param_index, dim_index) | 获取参数偏移 | RUNTIME_COA_GET_PARAM_OFFSET(2,1,0) |
| RUNTIME_COA_GET_PARAM_VALID_SHAPE(param_index, dim_index) | 获取参数有效shape | RUNTIME_COA_GET_PARAM_VALID_SHAPE(2,1,0) |

#### 分析要点

1. **变量定义**: 确保输出变量只被定义一次
2. **变量引用**: 确保输入变量都已被定义
3. **shape匹配**: 验证输入输出shape的合理性
4. **内存类型**: 检查内存类型转换的正确性
5. **操作参数**: 验证操作参数的完整性和正确性
6. **属性一致性**: 检查操作属性的合理性

## IR分析方法

### 1. 文件完整性检查

#### 检查清单

- [ ] 文件头格式正确
- [ ] 所有RAWTENSOR索引唯一
- [ ] 所有INCAST/OUTCAST索引唯一
- [ ] 所有操作节点op_id唯一
- [ ] 所有变量定义和使用匹配

#### 检查方法

```python
# 伪代码示例
def check_file_completeness(ir_file):
    # 1. 检查文件头
    if not ir_file.has_valid_header():
        return False, "Invalid file header"
    
    # 2. 检查RAWTENSOR索引唯一性
    rawtensor_indices = ir_file.get_rawtensor_indices()
    if len(set(rawtensor_indices)) != len(rawtensor_indices):
        return False, "Duplicate RAWTENSOR indices"
    
    # 3. 检查变量定义和使用
    defined_vars = ir_file.get_defined_variables()
    used_vars = ir_file.get_used_variables()
    undefined_vars = used_vars - defined_vars
    if undefined_vars:
        return False, f"Undefined variables: {undefined_vars}"
    
    return True, "File is complete"
```

### 2. 数据流分析

#### 分析目标

- 追踪数据从输入到输出的完整路径
- 验证数据依赖关系的正确性
- 检查是否存在悬空数据或数据泄露

#### 分析步骤

1. **构建数据流图**
   - 以变量为节点
   - 以数据依赖为边

2. **追踪数据路径**
   - 从INCAST开始追踪
   - 沿着操作链追踪到OUTCAST

3. **验证数据完整性**
   - 确保所有输入数据都被使用
   - 确保所有输出数据都有来源

#### 示例分析

```
# 数据流示例
INCAST[0] → %6 → VIEW → %7 → ... → OUTCAST[0]
```

### 3. 内存访问分析

#### 分析目标

- 检查内存访问的合法性
- 验证内存类型转换的正确性
- 检查是否存在内存冲突

#### 分析要点

1. **内存类型转换**
   - MEM_UNKNOWN → MEM_DEVICE_DDR
   - MEM_DEVICE_DDR → MEM_UB
   - MEM_UB → MEM_DEVICE_DDR

2. **内存访问模式**
   - TILE_COPY_IN: DDR → UB
   - TILE_COPY_OUT: UB → DDR
   - 计算操作: UB → UB

3. **内存冲突检查**
   - 同一内存区域不能同时被读写
   - 需要同步机制保证数据一致性

#### 示例分析

```
# 内存访问模式示例
%0 (DDR) → TILE_COPY_IN → %84 (UB) → TILE_ADDS → %76 (UB) → TILE_COPY_OUT → %14 (DDR)
```

### 4. 依赖关系分析

#### 分析目标

- 检查操作之间的依赖关系
- 验证调度顺序的合理性
- 检查是否存在循环依赖

#### 分析要点

1. **数据依赖**
   - 读取-写入依赖（RAW）
   - 写入-读取依赖（WAR）
   - 写入-写入依赖（WAW）

2. **控制依赖**
   - 条件分支
   - 循环结构

3. **同步依赖**
   - 内存屏障
   - 同步操作

### 5. Shape一致性检查

#### 检查方法

1. **操作输入输出shape匹配**
   - 每个操作的输入shape应该符合操作要求
   - 输出shape应该与操作结果一致

2. **张量shape传播**
   - 沿着数据流追踪shape变化
   - 验证shape变化的合理性

3. **动态shape处理**
   - 检查动态shape的处理是否正确
   - 验证运行时参数的使用

## 问题定位技巧

### 1. 对比Before和After IR

#### 对比维度

| 维度 | 说明 | 检查方法 |
|-----|------|---------|
| 操作数量 | 操作节点的增减 | 统计操作数量变化 |
| 变量数量 | 变量的增减 | 统计变量数量变化 |
| 内存类型 | 内存类型的变化 | 对比内存类型 |
| 数据流 | 数据流路径的变化 | 追踪数据流 |
| shape | shape的变化 | 对比shape |

#### 常见问题模式

1. **操作丢失**
   - 症状：After中缺少某些操作
   - 可能原因：Pass错误删除了必要操作
   - 定位方法：对比操作ID和操作名

2. **变量未定义**
   - 症状：After中使用了未定义的变量
   - 可能原因：Pass删除了变量定义但保留了使用
   - 定位方法：检查变量定义和使用

3. **shape不匹配**
   - 症状：操作输入输出shape不一致
   - 可能原因：Pass错误修改了shape
   - 定位方法：检查操作节点的shape

4. **内存类型错误**
   - 症状：内存类型转换不正确
   - 可能原因：Pass错误分配了内存类型
   - 定位方法：检查内存类型转换链

### 2. 错误模式识别

#### 语法错误

```
# 错误示例：缺少等号
<16 x 128 x DT_FP32> %1@6#(-1)MEM_UNKNOWN::MEM_UNKNOWN !10001 ADDS(g:-1, s:-1) %10@13#(-1)MEM_UNKNOWN::MEM_UNKNOWN
```

**定位方法**：
- 检查操作节点格式
- 验证等号存在性

#### 语义错误

```
# 错误示例：使用了未定义的变量
<16 x 128 x DT_FP32> %1@6#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10001 ADDS(g:-1, s:-1) %999@13#(-1)MEM_UNKNOWN::MEM_UNKNOWN
# %999 未定义
```

**定位方法**：
- 检查变量定义和使用
- 验证变量索引范围

#### 逻辑错误

```
# 错误示例：数据流断裂
INCAST[0] → %6 → VIEW → %7
# %7 未被任何操作使用
```

**定位方法**：
- 追踪数据流
- 检查悬空变量

### 3. 性能问题定位

#### 常见性能问题

1. **冗余操作**
   - 症状：存在重复或无用的操作
   - 定位方法：分析操作冗余性

2. **数据拷贝过多**
   - 症状：TILE_COPY_IN/TILE_COPY_OUT过多
   - 定位方法：统计数据拷贝次数

3. **内存访问不连续**
   - 症状：offset不连续，影响性能
   - 定位方法：分析offset模式

4. **并行度不足**
   - 症状：操作串行化，未充分利用并行
   - 定位方法：分析依赖关系

## 逻辑错误检查清单

### 变量相关

- [ ] 所有变量都有唯一定义
- [ ] 所有使用的变量都已定义
- [ ] 变量索引在有效范围内
- [ ] 变量地址标识符一致

### 操作相关

- [ ] 所有操作ID唯一
- [ ] 操作参数数量正确
- [ ] 操作输入输出shape匹配
- [ ] 操作属性合理

### 数据流相关

- [ ] 所有INCAST数据都被使用
- [ ] 所有OUTCAST数据都有来源
- [ ] 数据流路径完整
- [ ] 不存在悬空数据

### 内存相关

- [ ] 内存类型转换正确
- [ ] 内存访问合法
- [ ] 不存在内存冲突
- [ ] 内存复用合理

### 依赖关系相关

- [ ] 不存在循环依赖
- [ ] 数据依赖正确
- [ ] 控制依赖正确
- [ ] 同步依赖正确

### Shape相关

- [ ] 所有shape维度合理
- [ ] shape传播正确
- [ ] 动态shape处理正确
- [ ] shape边界检查正确

## IR分析流程

### 标准分析流程

```
1. 读取IR文件
   ↓
2. 检查文件完整性
   ↓
3. 解析IR元素
   ↓
4. 构建数据流图
   ↓
5. 执行一致性检查
   ↓
6. 分析数据流
   ↓
7. 检查内存访问
   ↓
8. 验证依赖关系
   ↓
9. 生成分析报告
```

### 详细分析步骤

#### 步骤1：读取和解析

```python
def parse_ir_file(file_path):
    """解析IR文件"""
    ir = IRFile()
    
    # 1. 读取文件头
    ir.parse_header()
    
    # 2. 解析RAWTENSOR
    ir.parse_rawtensors()
    
    # 3. 解析INCAST/OUTCAST
    ir.parse_casts()
    
    # 4. 解析操作节点
    ir.parse_operations()
    
    return ir
```

#### 步骤2：完整性检查

```python
def check_completeness(ir):
    """检查IR完整性"""
    checks = [
        check_header(ir),
        check_rawtensor_indices(ir),
        check_variable_definitions(ir),
        check_operation_ids(ir),
    ]
    
    return all(checks)
```

#### 步骤3：数据流分析

```python
def analyze_dataflow(ir):
    """分析数据流"""
    # 1. 构建数据流图
    graph = build_dataflow_graph(ir)
    
    # 2. 追踪数据路径
    paths = trace_data_paths(graph)
    
    # 3. 检查数据完整性
    integrity = check_data_integrity(paths)
    
    return integrity
```

#### 步骤4：内存分析

```python
def analyze_memory(ir):
    """分析内存访问"""
    # 1. 构建内存访问图
    mem_graph = build_memory_graph(ir)
    
    # 2. 检查内存类型转换
    type_checks = check_memory_types(mem_graph)
    
    # 3. 检查内存冲突
    conflicts = check_memory_conflicts(mem_graph)
    
    return type_checks, conflicts
```

#### 步骤5：生成报告

```python
def generate_analysis_report(ir, checks):
    """生成分析报告"""
    report = {
        'file': ir.file_path,
        'completeness': checks['completeness'],
        'dataflow': checks['dataflow'],
        'memory': checks['memory'],
        'dependencies': checks['dependencies'],
        'issues': checks['issues'],
        'recommendations': checks['recommendations'],
    }
    
    return report
```

## 最佳实践

### 1. 分析策略

#### 自顶向下分析

1. 先检查文件结构和完整性
2. 再分析数据流和依赖关系
3. 最后深入细节检查

#### 关键路径优先

1. 优先分析关键数据流路径
2. 重点检查核心操作节点
3. 验证关键变量的正确性

### 2. 调试技巧

#### 分段验证

1. 将IR分段分析
2. 逐步验证每个部分
3. 定位问题到具体位置

#### 对比验证

1. 对比Before和After IR
2. 识别变化点
3. 验证变化的合理性

### 3. 记录和报告

#### 详细记录

1. 记录分析过程和发现
2. 保存关键IR片段
3. 记录问题和解决方案

#### 结构化报告

1. 使用标准报告格式
2. 包含问题、原因、建议
3. 提供可操作的改进建议

## 常见问题和解决方案

### 问题1：变量未定义

**症状**：IR中使用了未定义的变量

**原因**：Pass删除了变量定义但保留了使用

**解决方案**：
1. 检查变量定义和使用
2. 确保所有变量都有定义
3. 修复Pass逻辑

### 问题2：数据流断裂

**症状**：数据流路径不完整

**原因**：Pass错误删除了操作

**解决方案**：
1. 追踪数据流路径
2. 识别断裂点
3. 恢复必要操作

### 问题3：shape不匹配

**症状**：操作输入输出shape不一致

**原因**：Pass错误修改了shape

**解决方案**：
1. 检查shape传播
2. 验证操作shape要求
3. 修复shape计算

### 问题4：内存类型错误

**症状**：内存类型转换不正确

**原因**：Pass错误分配了内存类型

**解决方案**：
1. 检查内存类型转换链
2. 验证内存访问模式
3. 修复内存类型分配

## 附录

### IR语法总结

```
IR文件 ::= 文件头 RAWTENSOR* INCAST* OUTCAST* operation*

文件头 ::= "Function" 函数名 "[" 参数数量 "]" hash 图属性 "{" 

RAWTENSOR ::= "RAWTENSOR[" 索引 "] <" shape "> @" 编号 "\"" 名称 "\""

INCAST ::= "INCAST[" 索引 "] <" shape "/" valid_shape "> %" logic_tensor "@" raw_tensor "#(" 组号 ") fromSlot[" 槽位 "]"

OUTCAST ::= "OUTCAST[" 索引 "] <" shape "/" valid_shape "> %" logic_tensor "@" raw_tensor "#(" 组号 ") toSlot[" 槽位 "]"

operation ::= "<" shape / valid_shape "> %" 输出logic_tensor "@" 输出raw_tensor "#(" 组号 ")" 内存类型 " = !" operation_id opcode 参数 属性*

logic_tensor ::= "%" 名称 或 "%" 编号

raw_tensor ::= "@" 编号 或 "@" 名称

shape ::= 维度 " x " 维度 " x " ... " x " 数据类型

属性 ::= "#" 属性名 "{" 属性值 "}"
```