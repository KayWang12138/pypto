# program.json 文件详细说明

## 文件概述

`program.json` 是 PyPTO 编译后生成的程序描述文件，包含了完整的函数定义、操作序列、张量信息、内存布局等编译结果。

**文件位置**：输出目录下的 `program.json`（例如：`output/output_<timestamp>_<pid>/program.json`）

**文件大小**：1.3MB，85603 行

**格式**：JSON

## 文件结构

```json
{
  "curr_funcmagic": 2,
  "enable_cvfuse": false,
  "entryhash": "10193433060527433398",
  "functions": [...]
}
```

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `curr_funcmagic` | int | 当前函数魔法数 |
| `enable_cvfuse` | bool | 是否启用 CV 融合优化 |
| `entryhash` | string | 入口函数哈希值 |
| `functions` | array | 函数列表 |

## 函数结构

每个函数包含以下主要字段：

### 1. 函数标识

```json
{
  "func_magicname": "PROGRAM_ENTRY",
  "funcmagic": 1,
  "functype": 0,
  "graphtype": 5,
  "hash": "0",
  "rawname": "PROGRAM_ENTRY"
}
```

**字段说明**：
- `func_magicname`：函数魔法名称
- `funcmagic`：函数魔法数
- `functype`：函数类型（0=普通函数）
- `graphtype`：图类型（5=执行图）
- `hash`：函数哈希值
- `rawname`：原始函数名

### 2. 优化参数

```json
{
  "_cube_nbuffer_mode": 0,
  "_funcid": 20,
  "_l1_reuse_num": 0,
  "_mg_vec_parallel_lb": 48,
  "_ooo_preschedule_method": "PriorDFS",
  "_opseed": 10001,
  "_sg_cube_parallel_num": 24,
  "_sg_mg_copyin_upper_bound": 2097152,
  "_sg_parallel_num": 1,
  "_vec_nbuffer_mode": 1
}
```

**字段说明**：
- `_cube_nbuffer_mode`：Cube 操作的 NBuffer 模式
- `_mg_vec_parallel_lb`：向量并行下界
- `_ooo_preschedule_method`：乱序调度预调度方法
- `_sg_cube_parallel_num`：子图 Cube 并行数
- `_sg_parallel_num`：子图并行数
- `_vec_nbuffer_mode`：向量操作的 NBuffer 模式

### 3. 操作序列 (operations)

```json
{
  "operations": [
    {
      "opcode": "CALL",
      "kind": 2,
      "calleehash": "10193433060527433398",
      "ioperands": [87],
      "ooperands": [100],
      "latency": 9,
      "tile": {...},
      "sync_queue": {...}
    }
  ]
}
```

**操作类型** (`opcode`)：
- `CALL`：函数调用
- `ASSEMBLE`：组装操作
- `VIEW`：视图操作
- `DIV`、`EXP`、`SUB`、`ADD`：数学运算
- `REDUCE`：归约操作
- `COPY`：拷贝操作

**字段说明**：
- `opcode`：操作码
- `kind`：操作类型
- `calleehash`：被调用函数哈希值（CALL 操作）
- `ioperands`：输入操作数索引列表
- `ooperands`：输出操作数索引列表
- `latency`：操作延迟
- `tile`：分块配置
- `sync_queue`：同步队列配置

### 4. 张量信息 (tensors)

```json
{
  "tensors": [
    {
      "magic": 87,
      "kind": 1,
      "rawtensor": 0,
      "shape": [-1, 32, 1, 256],
      "validshape": [-1, 32, 1, 256],
      "offset": [0, 0, 0, 0],
      "life_range": [-1, -1],
      "mem_id": -1,
      "mem_range": [0, 0],
      "nodetype": 0,
      "subgraph_boundary": false
    }
  ]
}
```

**字段说明**：
- `magic`：张量魔法数（唯一标识）
- `kind`：张量类型（1=中间张量，0=输入/输出）
- `rawtensor`：原始张量索引
- `shape`：张量形状（-1 表示动态维度）
- `validshape`：有效形状
- `offset`：内存偏移量
- `life_range`：生命周期范围 [开始操作, 结束操作]
- `mem_id`：内存 ID
- `mem_range`：内存范围
- `nodetype`：节点类型
- `subgraph_boundary`：是否为子图边界

### 5. 原始张量 (rawtensors)

```json
{
  "rawtensors": [
    {
      "symbol": "TENSOR_1",
      "datatype": 7,
      "format": 0,
      "kind": 0,
      "rawshape": [-1, 32, 1, 256],
      "ori_rawshape": [],
      "rawmagic": 0
    }
  ]
}
```

**字段说明**：
- `symbol`：张量符号名
- `datatype`：数据类型（7=FP32）
- `format`：数据格式
- `rawshape`：原始形状
- `rawmagic`：原始张量魔法数

### 6. Cast 操作

```json
{
  "incasts": [...],
  "outcasts": [...]
}
```

**说明**：
- `incasts`：输入 Cast 操作列表
- `outcasts`：输出 Cast 操作列表

## 数据类型映射

| datatype 值 | 类型 |
|------------|------|
| 7 | FP32 (float32) |
| 6 | FP16 (float16) |
| 5 | INT32 |
| 4 | INT16 |
| 3 | INT8 |

## 图类型 (graphtype)

| graphtype 值 | 类型 |
|-------------|------|
| 0 | STATIC_GRAPH |
| 1 | DYNAMIC_GRAPH |
| 5 | EXECUTE_GRAPH |

## 使用场景

1. **运行时加载**：运行时根据 `program.json` 构建执行图
2. **调试分析**：分析编译结果和优化效果
3. **性能分析**：通过 `latency` 和 `life_range` 分析性能
4. **内存分析**：通过 `mem_id` 和 `mem_range` 分析内存布局

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [run.log 文件详细说明](./run-log.md)
- [topo.json 文件详细说明](./topo-json.md)
- [输出目录与产物总览](./README.md)

