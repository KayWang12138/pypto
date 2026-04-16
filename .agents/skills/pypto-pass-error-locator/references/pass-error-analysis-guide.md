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

## 三、计算图成环

### 日志特征

当日志中出现以下关键字时，应优先按“计算图成环”场景分析：

- `cycle detected`
- `graph cycle`
- `topological sort failed`
- `circular dependency`
- `Loop Detected`
- `OperationLoopCheck`
- `Cycle Detection`

说明：

- 当前报错 pass 不一定是成环的根因 pass。
- 某些 pass 只是首次执行拓扑检查、排序或子图依赖检查，因此会最先暴露问题，但环可能由更早的 pass 引入。

### 输入材料

分析该场景时，至少需要准备以下材料：

- 当前报错日志
- 当前报错 pass 对应的计算图 JSON
- 上游相邻 pass 的计算图 JSON
- 若可获得，当前 pass 的 Before / After 两份图
- 日志中对应的源码文件与行号

常见计算图目录位置：

- `build/output/pass/`
- `pypto/output/pass/`

pass 中间目录通常类似：

- `Pass_{PASS_SEQ}_{PASS_NAME}`

例如：

```bash
ls -lt build/output/pass/Pass_*/*.json
```

```bash
ls -lt pypto/output/pass/Pass_*/*.json
```

### 分析目标

本场景的分析目标包括：

1. 确认当前图中是否存在 op 之间的循环依赖
2. 确认当前图中是否存在 subgraph 之间的循环依赖
3. 判断当前报错 pass 是否为根因 pass
4. 沿 pass 链向前回溯，定位首次引入环的 pass 模块
5. 结合源码分析形成根因链，并给出修复建议

### 分析流程

#### 步骤 1：定位当前报错 pass 和对应图文件

1. 从日志中提取以下信息：
   - pass 名称
   - Before / After 阶段
   - 报错文件和行号
   - 相关 `op_magic`、`tensor_magic`
2. 在 pass 输出目录中定位当前报错 pass 的图文件
3. 同时收集上一个 pass 的输出图，作为回溯起点

#### 步骤 2：对当前图执行 op 级成环检测

op 级成环检测用于确认计算图中是否存在 operation 之间的循环依赖。

分析口径参考：

- `framework/src/interface/function/function.cpp`
- `Function::OperationLoopCheck(const std::string&)`

其核心逻辑是：

- 遍历所有 operation
- 根据 op 输出 tensor 到 consumer op 的关系建立依赖边
- 对 consumer 链执行 DFS
- 若遍历过程中再次进入 `IN_STACK` 状态的 op，则说明存在回边，图中成环

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --detect-op-cycle
```

需要记录的结果：

- 是否存在 op 级环
- 成环路径上的 `opmagic` 列表
- 每条依赖边对应的 `tensor magic`
- 每个 op 的 `opcode`、`file`、`line`

#### 步骤 3：对当前图执行 subgraph 级成环检测

subgraph 级成环检测用于确认不同 `subgraphid` 之间是否通过 tensor 形成闭环。

分析口径参考：

- `framework/src/interface/function/function.cpp`
- `Function::LoopCheck()`

其核心逻辑是：

- 按 `subgraphid` 聚合 operation
- 根据跨 subgraph 的 tensor 生产消费关系建立依赖边
- 对 subgraph 图执行 DFS
- 若遍历过程中再次进入 `IN_STACK` 状态的 subgraph，则说明 subgraph 之间存在环

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --detect-subgraph-cycle
```

需要记录的结果：

- 是否存在 subgraph 级环
- 成环的 `subgraph id` 路径
- 触发该路径的跨 subgraph tensor
- 对应 producer op 和 consumer op

#### 步骤 4：判断当前图成环是否能被脚本证实

根据脚本结果分情况处理：

1. 如果脚本检测到显式环：
   - 继续沿 pass 链向前回溯，定位首次引入环的 pass
2. 如果脚本未检测到显式环，但日志提示 `cycle detected` 或 `topological sort failed`：
   - 不得直接认定“图中无环”
   - 需要进一步怀疑以下隐式依赖：
     - `dependOperand`
     - operation group 顺序约束
     - 调度级或拓扑级附加依赖

说明：

- 当前图 JSON 中已稳定导出 `ioperands`、`ooperands`
- 当前图 JSON 中未直接导出 `dependOperand`
- 因此脚本第一阶段只能稳定分析“显式 tensor 数据流”上的成环问题
- 若日志来自拓扑排序或调度阶段，则可能存在 JSON 未表达的隐式依赖，需要回到源码侧确认

#### 步骤 5：沿 pass 链向前回溯，定位首次引入环的 pass

当前报错 pass 不一定是根因 pass，需要逐步回溯。

回溯方法：

1. 对当前报错 pass 的输入图执行成环检测
2. 对上一个 pass 的输出图执行成环检测
3. 按 pass 顺序持续向前回溯
4. 找到“前一阶段无环、当前阶段有环”的边界
5. 将该 pass 标记为“首次引入环的根因候选 pass”

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <curr_graph> \
  --detect-op-cycle
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <prev_graph> \
  --detect-op-cycle
```

若脚本支持对比模式，也可执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --before-json <prev_graph> \
  --after-json <curr_graph> \
  --compare-cycle
```

回溯结论必须区分：

- 报错 pass：首次抛出异常的 pass
- 根因候选 pass：首次引入环的 pass
- 根因未定：图文件不完整或隐式依赖无法从 JSON 还原时的结论

#### 步骤 6：结合源码确认根因修改点

对“首次引入环”的 pass，需要重点检查其是否进行了以下操作：

- 新增了错误的 producer-consumer 依赖
- 复用了旧 tensor，导致边关系串错
- 修改 op 输入输出时遗留旧依赖
- 跨 subgraph 迁移 op 后未同步修正边界 tensor
- 引入额外 depend 依赖但未同步维护一致性

需要输出的证据包括：

- 成环路径上的关键 op
- 成环路径上的关键 tensor
- pass 中建立或修改依赖的源码位置
- Before / After 图差异

### 输出要求

最终输出必须明确区分以下概念：

- **报错 pass**：第一个抛出成环异常的 pass
- **当前图证据**：当前图中是否真的检测到显式环
- **根因候选 pass**：首次引入环的 pass
- **能力边界**：是否存在 JSON 未表达的隐式依赖

建议按如下根因链格式输出：

`现象(某 pass 报 cycle) -> 图证据(某 JSON 中的 op/subgraph 环路) -> 首次引入位置(某上游 pass) -> 触发源码(某文件某行的依赖修改逻辑)`

### 注意事项

- 当前报错 pass 不一定是根因 pass
- 脚本未检测到显式环，不等于图中一定无环
- 拓扑排序失败还可能来自 `dependOperand` 或调度顺序约束
- 图文件不完整时，只能给出“根因候选”，不能伪造成确定结论

---

## 输出要求

最终分析结果至少包含：

- 日志证据
- 计算图证据
- 源码位置
- 官方文档约束
- 可执行修复建议
