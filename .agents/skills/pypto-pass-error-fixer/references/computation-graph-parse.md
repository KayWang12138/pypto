# PyPTO计算图JSON分析指导文档

## 文档概述

本文档旨在指导AI Agent理解PyPTO编译过程中生成的计算图JSON文件，帮助Agent通过分析JSON结构来理解计算图的演变过程和各Pass阶段的优化效果。

## 一、计算图JSON文件结构

### 1.1 根节点结构

```json
{
  "entryhash": "311644491877735055",    // 计算图唯一标识
  "version": "2.0",                      // JSON格式版本
  "functions": [...]                    // 函数数组，通常包含一个主函数
}
```

### 1.2 Function节点结构

```json
{
  "func_magicname": "TENSOR_TENSOR_update_kernel_loop_Unroll1_PATH0_hiddenfunc0_5",
  "funcmagic": 5,                        // 函数魔数ID
  "graphtype": 1,                        // 图类型: 1=TensorGraph, 2=TileGraph, 3=BlockGraph, 4=ExecuteGraph
  "incasts": [...],                     // 输入节点列表
  "outcasts": [...],                    // 输出节点列表
  "operations": [...],                  // 操作节点列表
  "tensors": [...],                     // Tensor节点列表
  "rawtensors": [...],                  // 原始Tensor信息
  "_total_subgraph_count": 0,           // 子图总数（BlockGraph/ExecuteGraph阶段有效）
  "file": "...",                        // 源文件路径
  "line": 156                           // 源代码行号
}
```

### 1.3 图类型说明

| graphtype | 类型名称 | 说明 |
|-----------|---------|------|
| 1 | Tensor Graph | 用户定义的原始计算图，未经过Tile展开 |
| 2 | Tile Graph | 经过Tile展开后的计算图，包含Tile级别的操作 |
| 3 | Block Graph | 调度运行在单个AI Core上的子图 |
| 4 | Execute Graph | 包含子图调度的执行图 |

## 二、核心节点类型详解

### 2.1 Tensor节点

```json
{
  "magic": 0,                           // Tensor唯一标识ID
  "shape": [1, 8],                      // Tensor形状
  "validshape": [1, 8],                 // 有效形状
  "dynvalidshape": [[0, 1], [0, 8]],    // 动态有效形状（运行时计算）
  "offset": [0, 0],                     // 在rawtensor中的偏移量
  "rawtensor": 5,                       // 所属rawtensor的ID
  "nodetype": 0,                        // 节点类型: 0=普通Tensor, 1=Incast, 2=Outcast
  "mem_id": -1,                         // 内存ID（分配后有效）
  "mem_range": [0, 0],                  // 内存范围
  "mem_type": {                         // 内存类型信息
    "asis": 0,                          // 源内存层级
    "tobe": 0                           // 目标内存层级
  },
  "subgraphid": -1,                     // 所属子图ID（切图后有效）
  "subgraph_boundary": false,           // 是否为子图边界
  "life_range": [-1, -1],               // 生命周期范围
  "kind": 1                             // Tensor类型
}
```

**关键分析点：**
- `magic`: 用于追踪Tensor在计算图中的流转
- `shape`: 观察Tensor形状在各Pass阶段的变化
- `mem_type.asis`/`mem_type.tobe`: 分析数据在不同内存层级间的流动
- `subgraphid`: 观察Tensor被分配到哪个子图

### 2.2 Operation节点

```json
{
  "opmagic": 10010,                     // Operation唯一标识ID
  "opcode": "VIEW",                     // 操作类型
  "ioperands": [6],                    // 输入Tensor的magic ID列表
  "ooperands": [7],                     // 输出Tensor的magic ID列表
  "attr": [...],                        // 操作属性（包含shape信息）
  "tile": {                             // Tile配置信息
    "vec": [16, 128],                   // Vector配置
    "cube": [...],                      // Cube配置
    "comm": [...]                       // Common配置
  },
  "subgraphid": -1,                     // 所属子图ID
  "latency": 10,                        // 延迟周期
  "kind": 2,                            // 操作类型分类
  "op_attr": {...},                     // 操作特定属性
  "file": "...",                        // 源文件路径
  "line": 20,                           // 源代码行号
  "backtrace": ""                       // 回溯信息
}
```

**关键分析点：**
- `opcode`: 识别操作类型（VIEW, ADDS, MATMUL, INDEX_OUTCAST等）
- `ioperands`/`ooperands`: 追踪数据依赖关系
- `tile`: 分析Tile配置和优化策略
- `subgraphid`: 观察操作被分配到哪个子图

### 2.3 Incast/Outcast节点

```json
// incasts数组格式
[
  [6, [2]],  // [Tensor的magic ID, [参数索引]]
  [9, [1]],
  [12, [0]]
]

// outcasts数组格式
[
  [14, [3]],  // [Tensor的magic ID, [参数索引]]
  [18, [4]]
]
```

**关键分析点：**
- `incasts`: 标识计算图的输入Tensor
- `outcasts`: 标识计算图的输出Tensor
- 用于追踪数据从外部输入到最终输出的完整路径

### 2.4 RawTensor节点

```json
{
  "rawmagic": 5,                        // RawTensor唯一标识ID
  "rawshape": [1, 8],                   // RawTensor形状
  "ori_rawshape": [],                   // 原始形状
  "datatype": 3,                        // 数据类型
  "format": 0,                          // 数据格式
  "kind": 0,                            // RawTensor类型
  "symbol": "View_TENSOR_3"             // 符号名称（如果有）
}
```

**关键分析点：**
- `rawmagic`: 多个Tensor可以共享同一个RawTensor
- `rawshape`: 实际分配的内存形状
- `datatype`: 数据类型（3=int32, 7=float32等）

## 三、各Pass阶段分析要点

### 3.1 Tensor Graph阶段（graphtype=1）

**代表Pass：** Pass_04_ExpandFunction

**分析重点：**
1. **原始计算结构**
   - 观察`operations`中用户定义的操作（VIEW, ADDS, MATMUL等）
   - 检查`incasts`和`outcasts`确认输入输出

2. **Tensor形状**
   - 所有Tensor的`shape`应与用户代码定义一致
   - `subgraphid`通常为-1（未切图）

3. **内存分配**
   - `mem_id`通常为-1（未分配）
   - `mem_type.asis`和`mem_type.tobe`可能为0或未设置

**示例分析：**
```json
{
  "graphtype": 1,
  "operations": [
    {
      "opcode": "VIEW",
      "ioperands": [6],
      "ooperands": [7],
      "attr": [23, 2, 0, 0, 2, [0, 0], [0, 0], 2, [0, 1], [0, 16]]
    },
    {
      "opcode": "ADDS",
      "ioperands": [1],
      "ooperands": [76],
      "op_attr": {
        "SCALAR": {"data_type": 7, "value": 1.0}
      }
    }
  ]
}
```

### 3.2 Tile Graph阶段（graphtype=2）

**代表Pass：** Pass_17_L1CopyInReuseMerge

**分析重点：**
1. **Tile展开**
   - 对比Tensor Graph，观察节点数量显著增加
   - 原始大Tensor被切分为多个小Tile

2. **内存层级分配**
   - 检查`mem_type.asis`和`mem_type.tobe`的值
   - 常见值：0=UB, 1=L1, 15=GM

3. **Tile操作**
   - 出现TILE_COPY_IN, TILE_COPY_OUT等操作
   - `tile`字段包含详细的Tile配置

4. **子图分配**
   - `subgraphid`可能仍为-1（未切图）或已分配
   - `_total_subgraph_count`指示子图总数

**示例分析：**
```json
{
  "graphtype": 2,
  "_total_subgraph_count": 2,
  "operations": [
    {
      "opcode": "VIEW",
      "subgraphid": 1,
      "attr": [0, 2, 0, 0, 2, [0, 0], [0, 0], 2, [0, 1], [0, 8]]
    }
  ],
  "tensors": [
    {
      "magic": 0,
      "mem_type": {"asis": 0, "tobe": 0},
      "subgraphid": -1
    }
  ]
}
```

### 3.3 Block Graph阶段（graphtype=3）

**代表Pass：** Pass_36_CodegenPreproc (LEAF文件)

**分析重点：**
1. **子图隔离**
   - 每个Block Graph只包含一个子图的节点
   - `subgraphid`固定为某个值

2. **内存分配完成**
   - `mem_id`已分配具体值
   - `mem_range`指示内存占用范围

3. **操作类型**
   - 包含COPY_IN, COPY_OUT等数据搬运操作
   - 计算操作与内存操作明确分离

4. **子图参数**
   - `global_tensors`标识全局Tensor
   - `subfunc_param`包含子图参数信息

**示例分析：**
```json
{
  "graphtype": 3,
  "func_magicname": "..._leaf0_8",
  "global_tensors": [6, 9, 12, 18],
  "operations": [
    {
      "opcode": "COPY_IN",
      "subgraphid": 0,
      "in_param_loc": [0],
      "op_attr": {"IS_CUBE": false}
    }
  ],
  "tensors": [
    {
      "magic": 3,
      "mem_id": 8,
      "mem_range": [0, 0],
      "mem_type": {"asis": 0, "tobe": 0},
      "subgraphid": 0
    }
  ]
}
```

### 3.4 Execute Graph阶段（graphtype=4）

**代表Pass：** Pass_36_CodegenPreproc (ROOT文件)

**分析重点：**
1. **子图调用**
   - 包含CALL操作，调用Block Graph
   - 不包含具体的计算操作

2. **执行流程**
   - 通过CALL操作的顺序定义执行流程
   - 支持子图间的数据依赖

3. **全局视角**
   - 展示所有子图的调度关系
   - `_total_subgraph_count`指示子图总数

### 子图调用分析

**CALL操作结构：**
```json
{
  "opcode": "CALL",
  "opmagic": 10021,
  "ioperands": [],           // 输入参数（如果有）
  "ooperands": [],           // 输出参数（如果有）
  "attr": [
    // 子图调用信息
    0,                       // 子图ID
    1,                       // 调用类型
    // ... 其他属性
  ],
  "subgraphid": 0            // 所属子图ID
}
```

**分析方法：**
1. 从 `attr` 字段提取子图ID
2. 在Block Graph文件中查找对应的子图
3. 分析子图的输入输出
4. 追踪数据在子图间的流转

**示例：**
```
Execute Graph (ROOT)
  └─ CALL → Block Graph (subgraph_id=0)
       ├─ COPY_IN
       ├─ 计算操作
       └─ COPY_OUT
  └─ CALL → Block Graph (subgraph_id=1)
       ├─ COPY_IN
       ├─ 计算操作
       └─ COPY_OUT
```

**示例分析：**
```json
{
  "graphtype": 4,
  "func_magicname": "..._ROOT",
  "_total_subgraph_count": 2,
  "operations": [
    {
      "opcode": "CALL",
      "opmagic": 10021,
      "attr": [...],  // 包含子图调用信息
      "subgraphid": 0
    }
  ]
}
```

## 四、关键分析场景

### 4.1 追踪数据流转

**步骤：**
1. 从`incasts`找到输入Tensor的magic ID
2. 通过`operations`中的`ioperands`和`ooperands`追踪依赖链
3. 最终到达`outcasts`中的输出Tensor

**示例：**
```python
# 输入Tensor magic: 6
# Operation 1: ioperands=[6], ooperands=[7]  (VIEW)
# Operation 2: ioperands=[7], ooperands=[0]  (VIEW)
# 输出Tensor magic: 0
```

### 4.2 对比Pass前后差异

**方法：**
1. 读取Pass_xx的Before_xxx.json和After_xxx.json
2. 对比`operations`数组长度和内容
3. 对比`tensors`数组长度和内容
4. 观察新增、删除、修改的节点

**常见差异：**
- 节点数量变化：Pass优化导致节点增删
- Tensor形状变化：Tile展开或reshape优化
- 内存分配变化：内存策略调整

### 4.3 识别性能瓶颈

**指标：**
1. **节点数量**：过多节点可能影响性能
2. **子图数量**：`_total_subgraph_count`过多可能增加调度开销
3. **内存使用**：通过`mem_range`分析内存占用
4. **数据搬运**：统计COPY_IN/COPY_OUT操作数量

### 4.4 验证优化效果

**对比维度：**
1. **Tensor Graph vs Tile Graph**
   - 节点数量大幅增加（Tile展开）
   - Tensor形状变小（Tile切分）

2. **Tile Graph vs Block Graph**
   - 子图数量确定
   - 内存分配完成

3. **Block Graph vs Execute Graph**
   - 操作类型变化（计算→调度）
   - 数据依赖关系明确

## 五、常见操作类型说明

| opcode | 说明 | 常见属性 |
|--------|------|---------|
| VIEW | 视图操作，不改变数据 | shape, offset |
| ADDS | 标量加法 | SCALAR |
| ADD | 张量加法 | - |
| MATMUL | 矩阵乘法 | IS_CUBE |
| INDEX_OUTCAST | 索引输出 | axis, CACHE_MODE |
| ASSEMBLE | 数据组装 | - |
| COPY_IN | 数据拷入 | in_param_loc |
| COPY_OUT | 数据拷出 | out_param_loc |
| CALL | 子图调用 | subgraph_id |
| RESHAPE | 张量重塑 | shape |
| TRANSPOSE | 张量转置 | axes |
| BROADCAST | 张量广播 | shape |

## 六、内存层级说明

| 值 | 内存层级 | 说明 |
|----|---------|------|
| 0 | MEM_UB | 统一缓冲区（Unified Buffer）|
| 1 | MEM_L1 | L1缓存 |
| 2 | MEM_L0A | L0A缓存（矩阵A输入）|
| 3 | MEM_L0B | L0B缓存（矩阵B输入）|
| 4 | MEM_L0C | L0C缓存（矩阵C输出）|
| 15 | MEM_DEVICE_DDR | 设备DDR内存（Global Memory）|

## 七、分析工具建议

### 7.1 JSON解析脚本

```python
import json

def load_graph(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_graph_type(graph):
    graphtype = graph['functions'][0]['graphtype']
    types = {1: 'TensorGraph', 2: 'TileGraph', 3: 'BlockGraph', 4: 'ExecuteGraph'}
    return types.get(graphtype, 'Unknown')

def count_nodes(graph):
    func = graph['functions'][0]
    return {
        'operations': len(func['operations']),
        'tensors': len(func['tensors']),
        'rawtensors': len(func['rawtensors'])
    }

def trace_data_flow(graph, start_magic):
    """追踪从给定Tensor开始的数据流"""
    func = graph['functions'][0]
    flow = [start_magic]
    current = start_magic

    while True:
        # 找到以current为输入的operation
        found = False
        for op in func['operations']:
            if current in op['ioperands']:
                for output in op['ooperands']:
                    if output not in flow:
                        flow.append(output)
                        current = output
                        found = True
                        break
                if found:
                    break
        if not found:
            break

    return flow
```

### 7.2 对比工具

```python
def compare_graphs(before_graph, after_graph):
    before_func = before_graph['functions'][0]
    after_func = after_graph['functions'][0]

    return {
        'operations_diff': len(after_func['operations']) - len(before_func['operations']),
        'tensors_diff': len(after_func['tensors']) - len(before_func['tensors']),
        'subgraph_count': after_func.get('_total_subgraph_count', 0)
    }
```

## 八、分析报告模板

### 8.1 单Pass分析报告

```markdown
## Pass_xx_PassName 分析报告

### 图类型
- Before: [图类型]
- After: [图类型]

### 节点统计
- Operations: [before数量] → [after数量] ([差值])
- Tensors: [before数量] → [after数量] ([差值])
- RawTensors: [before数量] → [after数量] ([差值])

### 关键变化
1. [变化1描述]
2. [变化2描述]
3. [变化3描述]

### 性能影响
- [正面/负面]影响
- 原因：[分析原因]
```

### 8.2 全流程分析报告

```markdown
## 计算图全流程分析报告

### 阶段概览
1. Tensor Graph: [节点数] nodes
2. Tile Graph: [节点数] nodes
3. Block Graph: [子图数] subgraphs, [平均节点数] nodes/subgraph
4. Execute Graph: [子图数] subgraph

### 优化效果
- 节点总数变化: [初始] → [最终] ([倍数]x)
- 子图划分: [子图数]个子图
- 内存分配: [内存占用] bytes

### 潜在优化点
1. [优化点1]
2. [优化点2]
```

## 九、注意事项

1. **版本兼容性**
   - JSON格式版本为"2.0"，注意版本差异
   - 不同版本的字段可能有所不同

2. **动态形状**
   - `dynvalidshape`包含运行时计算的形状信息
   - 分析时需要区分静态shape和动态shape

3. **子图边界**
   - `subgraph_boundary`标识Tensor是否跨越子图边界
   - 跨边界Tensor需要特殊处理

4. **内存复用**
   - 多个Tensor可能共享同一个RawTensor
   - 通过`rawtensor`字段追踪内存复用关系

5. **调试信息**
   - `file`和`line`字段可用于定位源代码
   - `backtrace`包含调用栈信息

## 十、参考资料

- [PyPTO计算图官方文档目录] (docs/tools/computation_graph/)
