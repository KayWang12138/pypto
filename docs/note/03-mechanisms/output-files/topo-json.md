# topo.json 文件详细说明

## 文件概述

`topo.json` 是 PyPTO 编译后生成的拓扑描述文件，用于描述程序的拓扑结构。

**文件位置**：输出目录下的 `topo.json`（例如：`output/output_<timestamp>_<pid>/topo.json`）

**文件大小**：5.0B，2 行

**格式**：JSON

## 文件内容

```json
null
```

## 说明

在当前的 softmax 示例中，`topo.json` 文件内容为 `null`，表示：

1. **简单拓扑**：对于简单的单函数程序，不需要额外的拓扑描述
2. **动态生成**：拓扑信息可能在运行时动态生成
3. **可选字段**：拓扑信息是可选的，不是所有程序都需要

## 如何定位某个 kernel

对于复杂的多函数程序，`topo.json` 包含函数间的依赖关系：

1. **查找函数节点**：在 `nodes` 数组中查找 `function_hash` 对应的函数
2. **查找依赖关系**：在 `edges` 数组中查找以该函数为 `source` 或 `target` 的边
3. **关联 program.json**：使用 `function_hash` 在 `program.json` 中查找对应的函数定义
4. **关联 run.log**：在 `run.log` 中搜索函数名或哈希值，定位编译/执行日志

## 预期格式（字段示例片段）

对于复杂的多函数程序，`topo.json` 可能包含以下结构：

```json
{
  "nodes": [
    {
      "id": "node_id",
      "type": "function",
      "function_hash": "hash_value"
    }
  ],
  "edges": [
    {
      "source": "source_node_id",
      "target": "target_node_id",
      "tensor_id": "tensor_magic"
    }
  ]
}
```

**关键字段说明：**
- `nodes[].id`：节点标识符
- `nodes[].function_hash`：函数哈希值（用于关联 `program.json`）
- `edges[].source` / `edges[].target`：边的源节点和目标节点
- `edges[].tensor_id`：传递的张量标识符

**注意：** 以上为示例格式，实际字段可能因版本不同而变化，请以实际产物为准。

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [run.log 文件详细说明](./run-log.md)
- [program.json 文件详细说明](./program-json.md)
- [输出目录与产物总览](./README.md)

