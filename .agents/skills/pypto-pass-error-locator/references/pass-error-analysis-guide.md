# PyPTO Pass 异常分析流程指导

本文件提供 PyPTO Pass 模块常见异常类型的分析流程，指导AI Agent 针对不同类型的异常，采用不同策略进行分析。

---

## 通用分析总流程

### 步骤 1：定位日志

1. 从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录获取最新 `pypto-*.log`。
2. 先找 `[ERROR]`，再找 `[WARN]`。

```bash
ls "$ASCEND_PROCESS_LOG_PATH/debug/plog"
```

```bash
grep -n "\[ERROR\]\|\[WARN\]" "$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-*.log"
```

### 步骤 2：提取关键信息

从日志中提取以下信息：

- `op_magic` 或 `tensor_magic`
- Pass 名称
- Before / After 阶段
- `subgraph_id`
- 报错文本
- 文件路径和行号

### 步骤 3：选择分析入口

1. 如果日志里有 `op_magic` 或 `tensor_magic`，优先走计算图 JSON。
2. 如果拿到 `.tifwkgr`，走 IR 分析。
3. 如果两者都有，先用 JSON 定位，再用 IR 验证变化。

### 步骤 4：定位计算图节点

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --tensor-magic <tensor_magic>
```

建议先在 `output/` 下找最近的计算图文件：

```bash
ls -lt output/*/Pass_*/*.json
```

如果需要确认具体文件名，可先找对应 pass 目录：

```bash
ls -lt output/*/Pass_* | head
```

### 步骤 5：定位源码

1. 取 `OperationInfo.file` 和 `OperationInfo.line`。
2. 打开源码上下文，建议看前后各 20 行。

```bash
sed -n '<line-20>,<line+20>p' <source_file>
```

如果已经通过脚本输出了 `file` 和 `line`，可直接回看用户代码位置。

### 步骤 6：查官方文档

1. 算子类问题，先查 `docs/api/operation/`。
2. 参数类问题，先查 `docs/api/config/pypto-set_pass_options.md`。
3. 计算图解析类问题，直接使用本文件的“通用分析总流程”和 `scripts/computation_graph_analyzer.py`。

```bash
grep -R "<opcode>" docs/api/operation/
```

```bash
grep -n "<param_name>" docs/api/config/pypto-set_pass_options.md
```

### 步骤 7：对比 Before / After

1. 对比同一节点的 `shape`、`validshape`、`mem_type`、`ioperands`、`ooperands`。
2. 确认 Pass 是否产生了非预期变更。

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <after_json_path> \
  --op-magic <op_magic>
```

### 步骤 8：输出根因链

按以下格式输出：

`现象(日志) -> 触发位置(源码行) -> 状态异常(计算图 Before/After) -> 违反规则(文档约束)`

---

## 一、参数配置异常

### 日志特征

- `vec_nbuffer_setting`
- `cube_l1_reuse_setting`
- `cube_nbuffer_setting`
- `sg_set_scope`
- 包含 pass 配置参数名关键字

### 分析流程

#### 步骤 1：获取参数约束

```bash
grep -n "<param_name>" docs/api/config/pypto-set_pass_options.md
```

#### 步骤 2：检查用户参数配置

1. 定位用户代码中的参数设置位置。
2. 对照文档确认参数范围、类型和默认值。

#### 步骤 3：检查 pass 逻辑

1. 回看日志里的源码位置。
2. 检查 pass 是否正确处理了该参数。

#### 步骤 4：确定根因

1. 如果是用户配置不满足约束，明确指出参数值和不满足的规则。
2. 如果是 pass 逻辑问题，明确指出源码位置和错误分支。

---

## 二、算子约束违反

### 日志特征

- `opmagic`
- `tensor_magic`

### 分析流程

#### 步骤 1：提取关键信息

从日志中提取 `op_magic` 或 `tensor_magic`。

#### 步骤 2：加载计算图

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --tensor-magic <tensor_magic>
```

#### 步骤 3：获取算子信息

1. 如果是 `op_magic`，直接定位 Operation。
2. 如果是 `tensor_magic`，先查生产者，再查消费者。

#### 步骤 4：定位用户代码

1. 使用脚本输出的 `file` 和 `line`。
2. 查看源码上下文。

```bash
sed -n '<line-20>,<line+20>p' <source_file>
```

#### 步骤 5：检查算子约束

```bash
grep -R "<opcode>" docs/api/operation/
```

#### 步骤 6：对比 Pass 前后变化

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <after_json_path> \
  --op-magic <op_magic>
```

#### 步骤 7：确定根因

1. 如果是用户代码不满足约束，说明违反了哪条文档规则。
2. 如果是 pass 处理导致约束破坏，说明具体变更点。

---

## 输出要求

最终分析结果至少包含：

- 日志证据
- 计算图证据
- 源码位置
- 官方文档约束
- 可执行修复建议
