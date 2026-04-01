# PyPTO 计算图 JSON 分析指导文档

## 文档概述

本文档指导 AI Agent 使用 `scripts/computation_graph_analyzer.py` 工具分析 PyPTO 计算图 JSON 文件，理解计算图演变过程和各 Pass 阶段的优化效果。

## 工具引入

### 命令行使用

```bash
# 查看计算图摘要
python3 computation_graph_analyzer.py <json_file_path>

# 查看特定 Tensor 的详细信息
python3 computation_graph_analyzer.py <json_file_path> <tensor_magic>

# 示例
python3 computation_graph_analyzer.py output/output_xxx/Pass_04_ExpandFunction/Before_xxx.json
python3 computation_graph_analyzer.py output/output_xxx/Pass_04_ExpandFunction/Before_xxx.json 28
```

### Python 模块使用

```python
from computation_graph_analyzer import ComputationGraphAnalyzer, compare_graphs

# 创建分析器
analyzer = ComputationGraphAnalyzer()

# 加载计算图
graph = analyzer.load_graph('path/to/graph.json')

# 获取摘要信息
summary = analyzer.get_summary()
```

## 一、计算图基础信息

### 1.1 获取图类型

```python
func = analyzer.graph.get_main_function()
graph_type = func.get_graph_type_str()  # 'TensorGraph', 'TileGraph', 'BlockGraph', 'ExecuteGraph'
```

**图类型说明：**
| graphtype | 类型名称 | 说明 |
|-----------|---------|------|
| 1 | Tensor Graph | 用户定义的原始计算图，未经过 Tile 展开 |
| 2 | Tile Graph | 经过 Tile 展开后的计算图，包含 Tile 级别的操作 |
| 3 | Block Graph | 调度运行在单个 AI Core 上的子图 |
| 4 | Execute Graph | 包含子图调度的执行图 |

### 1.2 获取节点统计

```python
counts = analyzer.count_nodes()
# {
#   'operations': 13,
#   'tensors': 16,
#   'rawtensors': 16,
#   'subgraphs': 0
# }
```

### 1.3 获取输入输出 Tensor

```python
func = analyzer.graph.get_main_function()
input_tensors = func.get_input_tensor_magics()   # [21, 24, 27]
output_tensors = func.get_output_tensor_magics()  # [39, 49]
```

## 二、核心数据获取与分析

### 2.1 Tensor 查询与分析

```python
# 根据 magic ID 查找 Tensor
tensor = analyzer.find_tensor_by_magic(28)

if tensor:
    print(f"Tensor[{tensor.magic}]")
    print(f"  Shape: {tensor.shape}")
    print(f"  ValidShape: {tensor.validshape}")
    print(f"  Offset: {tensor.offset}")
    print(f"  RawTensor: {tensor.rawtensor}")
    print(f"  NodeType: {tensor.nodetype}")  # 0=普通, 1=输入, 2=输出
    print(f"  MemID: {tensor.mem_id}")
    print(f"  MemRange: {tensor.mem_range}")
    mem_asis, mem_tobe = tensor.get_memory_type_str()
    print(f"  MemType: {mem_asis} -> {mem_tobe}")
    print(f"  SubgraphID: {tensor.subgraphid}")
    print(f"  SubgraphBoundary: {tensor.subgraph_boundary}")
```

**关键分析点：**
- `magic`: 追踪 Tensor 在计算图中的流转
- `shape`: 观察 Tensor 形状在各 Pass 阶段的变化
- `mem_type`: 分析数据在不同内存层级间的流动
- `subgraphid`: 观察 Tensor 被分配到哪个子图

### 2.2 Operation 查询与分析

```python
# 根据 opmagic ID 查找 Operation
op = analyzer.find_operation_by_magic(10001)

# 根据 opcode 查找所有 Operation
index_outcast_ops = analyzer.find_operations_by_opcode('INDEX_OUTCAST')

if op:
    print(f"Operation[{op.opmagic}]")
    print(f"  Opcode: {op.opcode}")
    print(f"  Inputs: {op.ioperands}")
    print(f"  Outputs: {op.ooperands}")
    print(f"  Attr: {op.attr}")
    print(f"  SubgraphID: {op.subgraphid}")
    print(f"  Latency: {op.latency}")
    if op.file:
        print(f"  Source: {op.file}:{op.line}")
```

**关键分析点：**
- `opcode`: 识别操作类型（VIEW, ADDS, MATMUL, INDEX_OUTCAST 等）
- `ioperands`/`ooperands`: 追踪数据依赖关系
- `subgraphid`: 观察操作被分配到哪个子图

### 2.3 生产者-消费者关系分析

```python
# 查找 Tensor 的生产者
producer = analyzer.find_producer_of_tensor(28)
if producer:
    print(f"Producer: Operation[{producer.opmagic}] {producer.opcode}")

# 查找 Tensor 的消费者
consumers = analyzer.find_consumers_of_tensor(28)
for consumer in consumers:
    print(f"Consumer: Operation[{consumer.opmagic}] {consumer.opcode}")
```

### 2.4 数据流追踪

```python
# 正向追踪（从输入到输出）
flow = analyzer.trace_data_flow(28)
# 返回: [28, 37, 42, 39]

# 反向追踪（从输出到输入）
flow_backward = analyzer.trace_data_flow_backward(39)
# 返回: [39, 42, 37, 28, ...]
```

### 2.5 按类型统计 Operation

```python
op_types = analyzer.get_operations_by_type()
# {
#   'VIEW': 2,
#   'INDEX_OUTCAST': 2,
#   'ADDS': 1,
#   ...
# }
```

## 三、各 Pass 阶段分析要点

### 3.1 Tensor Graph 阶段（graphtype=1）

**代表 Pass：** Pass_04_ExpandFunction

**分析重点：**
1. **原始计算结构**
   ```python
   # 观察用户定义的操作
   view_ops = analyzer.find_operations_by_opcode('VIEW')
   add_ops = analyzer.find_operations_by_opcode('ADDS')
   ```

2. **Tensor 形状**
   ```python
   # 所有 Tensor 的 shape 应与用户代码定义一致
   for tensor_magic in func.get_input_tensor_magics():
       tensor = analyzer.find_tensor_by_magic(tensor_magic)
       print(f"Input Tensor[{tensor.magic}] shape: {tensor.shape}")
   ```

3. **内存分配**
   ```python
   # mem_id 通常为 -1（未分配）
   # subgraphid 通常为 -1（未切图）
   ```

### 3.2 Tile Graph 阶段（graphtype=2）

**代表 Pass：** Pass_17_L1CopyInReuseMerge

**分析重点：**
1. **Tile 展开**
   ```python
   # 对比 Tensor Graph，观察节点数量显著增加
   counts = analyzer.count_nodes()
   print(f"Operations: {counts['operations']}")
   ```

2. **内存层级分配**
   ```python
   # 检查 mem_type.asis 和 mem_type.tobe 的值
   memory_stats = analyzer.analyze_memory_usage()
   print(f"Memory by type: {memory_stats['memory_by_type']}")
   ```

3. **子图分配**
   ```python
   func = analyzer.graph.get_main_function()
   print(f"Subgraph count: {func.total_subgraph_count}")
   ```

### 3.3 Block Graph 阶段（graphtype=3）

**代表 Pass：** Pass_36_CodegenPreproc (LEAF 文件)

**分析重点：**
1. **子图隔离**
   ```python
   # 每个 Block Graph 只包含一个子图的节点
   subgraph_ops = analyzer.get_subgraph_operations(0)
   subgraph_tensors = analyzer.get_subgraph_tensors(0)
   ```

2. **内存分配完成**
   ```python
   # mem_id 已分配具体值
   # mem_range 指示内存占用范围
   ```

3. **全局 Tensor**
   ```python
   func = analyzer.graph.get_main_function()
   global_tensors = func.global_tensors
   ```

### 3.4 Execute Graph 阶段（graphtype=4）

**代表 Pass：** Pass_36_CodegenPreproc (ROOT 文件)

**分析重点：**
1. **子图调用**
   ```python
   # 包含 CALL 操作，调用 Block Graph
   call_ops = analyzer.find_operations_by_opcode('CALL')
   ```

2. **执行流程**
   ```python
   # 通过 CALL 操作的顺序定义执行流程
   for call_op in call_ops:
       print(f"CALL to subgraph: {call_op.attr}")
   ```

## 四、关键分析场景

### 4.1 追踪数据流转完整路径

```python
# 从输入到输出的完整数据流
func = analyzer.graph.get_main_function()
input_tensors = func.get_input_tensor_magics()

for input_magic in input_tensors:
    print(f"\n追踪输入 Tensor[{input_magic}] 的数据流:")
    flow = analyzer.trace_data_flow(input_magic)
    
    for tensor_magic in flow:
        tensor = analyzer.find_tensor_by_magic(tensor_magic)
        if tensor:
            print(f"  Tensor[{tensor.magic}] shape: {tensor.shape}")
            
            producer = analyzer.find_producer_of_tensor(tensor_magic)
            if producer:
                print(f"    <- {producer.opcode} (line {producer.line})")
```

### 4.2 对比 Pass 前后差异

```python
# 加载 Pass 前后的计算图
before_analyzer = ComputationGraphAnalyzer()
before_analyzer.load_graph('Pass_xx/Before_xxx.json')

after_analyzer = ComputationGraphAnalyzer()
after_analyzer.load_graph('Pass_xx/After_xxx.json')

# 对比差异
diff = compare_graphs(before_analyzer, after_analyzer)
print(f"Operations: {diff['before_counts']['operations']} -> {diff['after']}")
print(f"Tensors: {diff['before_counts']['tensors']} -> {diff['after_counts']['tensors']}")

# 分析操作类型变化
before_ops = before_analyzer.get_operations_by_type()
after_ops = after_analyzer.get_operations_by_type()

all_op_types = set(before_ops.keys()) | set(after_ops.keys())
for op_type in sorted(all_op_types):
    before_count = before_ops.get(op_type, 0)
    after_count = after_ops.get(op_type, 0)
    if before_count != after_count:
        print(f"{op{op_type}}: {before_count} -> {after_count}")
```

### 4.3 识别性能瓶颈

```python
# 节点数量
counts = analyzer.count_nodes()
print(f"Operations: {counts['operations']}")
print(f"Subgraphs: {counts['subgraphs']}")

# 内存使用
memory_stats = analyzer.analyze_memory_usage()
print(f"Allocated tensors: {memory_stats['allocated_tensors']}")
print(f"Memory by type: {memory_stats['memory_by_type']}")

# 数据搬运操作
copy_in_ops = analyzer.find_operations_by_opcode('COPY_IN')
copy_out_ops = analyzer.find_operations_by_opcode('COPY_OUT')
print(f"COPY_IN: {len(copy_in_ops)}, COPY_OUT: {len(copy_out_ops)}")
```

### 4.4 查找有多个消费者的 Tensor

```python
# 查找有多个消费者的 Tensor
multi_consumer_tensors = analyzer.find_tensor_with_multiple_consumers()

for tensor, consumers in multi_consumer_tensors:
    print(f"Tensor[{tensor.magic}] has {len(consumers)} consumers")
    for consumer in consumers:
        print(f"  - {consumer.opcode}")

# 只查找特定类型的消费者
multi_consumer_tensors = analyzer.find_tensor_with_multiple_consumers('INDEX_OUTCAST')
```

## 五、子图分析

### 5.1 获取子图信息

```python
func = analyzer.graph.get_main_function()
print(f"Total subgraphs: {func.total_subgraph_count}")

# 遍历所有子图
for subgraph_id in range(func.total_subgraph_count):
    ops = analyzer.get_subgraph_operations(subgraph_id)
    tensors = analyzer.get_subgraph_tensors(subgraph_id)
    
    print(f"\nSubgraph {subgraph_id}:")
    print(f"  Operations: {len(ops)}")
    print(f"  Tensors: {len(tensors)}")
```

### 5.2 分析子图间数据流转

```python
# 查找跨子图边界的 Tensor
func = analyzer.graph.get_main_function()
for tensor in func.tensors:
    if tensor.subgraph_boundary:
        print(f"Tensor[{tensor.magic}] crosses subgraph boundary")
        print(f"  SubgraphID: {tensor.subgraphid}")
        
        # 查找生产者和消费者
        producer = analyzer.find_producer_of_tensor(tensor.magic)
        consumers = analyzer.find_consumers_of_tensor(tensor.magic)
        
        if producer:
            print(f"  Producer: {producer.opcode} (subgraph {producer.subgraphid})")
        
        for consumer in consumers:
            print(f"  Consumer: {consumer.opcode} (subgraph {consumer.subgraphid})")
```

## 六、常见操作类型

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

## 七、内存层级

| 值 | 内存层级 | 说明 |
|----|---------|------|
| 0 | MEM_UB | 统一缓冲区（Unified Buffer）|
| 1 | MEM_L1 | L1 缓存 |
| 2 | MEM_L0A | L0A 缓存（矩阵 A 输入）|
| 3 | MEM_L0B | L0B 缓存（矩阵 B 输入）|
| 4 | MEM_L0C | L0C 缓存（矩阵 C 输出）|
| 15 | MEM_DEVICE_DDR | 设备 DDR 内存（Global Memory）|

## 八、注意事项

1. **版本兼容性**
   - JSON 格式版本为 "2.0"
   - 不同版本的字段可能有所不同

2. **动态形状**
   - `dynvalidshape` 包含运行时计算的形状信息
   - 分析时需要区分静态 shape 和动态 shape

3. **子图边界**
   - `subgraph_boundary` 标识 Tensor 是否跨越子图边界
   - 跨边界 Tensor 需要特殊处理

4. **内存复用**
   - 多个 Tensor 可能共享同一个 RawTensor
   - 通过 `rawtensor` 字段追踪内存复用关系

5. **调试信息**
   - `file` 和 `line` 字段可用于定位源代码
   - 使用这些信息快速定位问题代码

## 九、数据结构说明

### TensorInfo

```python
@dataclass
class TensorInfo:
    magic: int                          # Tensor 唯一标识
    shape: List[int]                    # Tensor 形状
    validshape: List[int]               # 有效形状
    dynvalidshape: List[List[int]]      # 动态有效形状
    offset: List[int]                   # 在 rawtensor 中的偏移量
    rawtensor: int                      # 所属 rawtensor 的 ID
    nodetype: int                       # 节点类型 (0=普通, 1=输入, 2=输出)
    mem_id: int                         # 内存 ID
    mem_range: List[int]                # 内存范围
    mem_type_asis: int                  # 源内存层级
    mem_type_tobe: int                  # 目标内存层级
    subgraphid: int                     # 所属子图 ID
    subgraph_boundary: bool             # 是否为子图边界
    life_range: List[int]                # 生命周期范围
    kind: int                           # Tensor 类型
```

### OperationInfo

```python
@dataclass
class OperationInfo:
    opmagic: int                        # Operation 唯一标识
    opcode: str                         # 操作类型
    ioperands: List[int]                 # 输入 Tensor 的 magic ID 列表
    ooperands: List[int]                 # 输出 Tensor 的 magic ID 列表
    attr: List[Any]                     # 操作属性
    subgraphid: int                     # 所属子图 ID
    latency: int                        # 延迟周期
    kind: int                           # 操作类型分类
    tile: Optional[Dict[str, Any]]      # Tile 配置信息
    op_attr: Dict[str, Any]             # 操作特定属性
    file: Optional[str]                 # 源文件路径
    line: Optional[int]                 # 源代码行号
```

### FunctionInfo

```python
@dataclass
class FunctionInfo:
    func_magicname: str                 # 函数名称
    funcmagic: int                      # 函数魔数 ID
    graphtype: int                      # 图类型 (1=Tensor, 2=Tile, 3=Block, 4=Execute)
    incasts: List[List[Any]]            # 输入节点列表
    outcasts: List[List[Any]]           # 输出节点列表
    operations: List[OperationInfo]      # 操作节点列表
    tensors: List[TensorInfo]            # Tensor 节点列表
    rawtensors: List[RawTensorInfo]      # 原始 Tensor 信息
    total_subgraph_count: int           # 子图总数
    file: Optional[str]                 # 源文件路径
    line: Optional[int]                 # 源代码行号
    global_tensors: List[int]           # 全局 Tensor 列表
```

## 十、扩展开发

脚本采用面向对象设计，易于扩展：

```python
class CustomAnalyzer(ComputationGraphAnalyzer):
    def custom_analysis(self):
        """自定义分析方法"""
        # 实现自定义逻辑
        pass
```

## 十一、注意事项

1. **路径问题**: 使用绝对路径或相对于脚本所在目录的路径
2. **JSON 格式**: 确保计算图 JSON 文件格式正确（版本 2.0）
3. **内存管理**: 大型计算图可能占用较多内存
4. **错误处理**: 建议使用 try-except 处理文件加载和解析错误

## 十二、参考资料

- [计算图分析脚本](scripts/computation_graph_analyzer.py)
- [PyPTO 计算图官方文档目录](docs/tools/computation_graph/)
