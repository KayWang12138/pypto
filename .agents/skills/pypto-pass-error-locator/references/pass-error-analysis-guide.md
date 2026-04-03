# PyPTO Pass 异常分析流程指导

本文档针对 PyPTO Pass 模块常见异常类型，提供不同异常类型的分析流程指导。

---

## 一、参数配置异常

### 日志特征
- `vec_nbuffer_setting`
- `cube_l1_reuse_setting`
- `cube_nbuffer_setting`
- `sg_set_scope`
- 包含pass配置参数名关键字

### 分析流程

#### 步骤 1：获取参数约束
1. 从 `docs/api/config/pypto-set_pass_options.md` 获取指定配置参数的使用说明

#### 步骤 2：检查用户参数配置
1. 检查用户提供的代码，定位到参数配置代码行，获取用户参数配置
2. 检查用户参数配置，对比参数使用说明，检查是否满足参数约束

#### 步骤 3：检查pass业务逻辑
1. 分析pass异常位置代码逻辑，检查约束实现是否正确，约束是否有例外场景

#### 步骤 4：确定根因
1. 如果用户配置参数不满足约束：给出异常的参数配置信息，代码位置，不满足的约束点
2. 如果是pass逻辑问题：给出pass逻辑异常代码位置，不满足的约束点

---

## 二、算子约束违反

### 分析流程

#### 步骤 1：提取关键信息
1. 从日志中提取 opmagic 或 tensor magic
2. 识别 magic 类型（op_magic 或 tensor_magic）

#### 步骤 2：加载计算图
1. 从 `output` 目录下，按时间戳找到最新的计算图输出目录，例如：`output/output_20260324_162232_912961_1605290_C0A8451A`
2. 在该目录下查找异常pass模块的 `.json` 计算图文件，获取文件路径，例如：`output/output_20260402_211910_059229_787054_C0A8451A/Pass_01_AutoCast/After_001_AutoCast_TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5.json`
3. 使用 `ComputationGraphAnalyzer.load_graph(json_path)` 加载计算图
4. 获取分析器实例

#### 步骤 3：获取算子信息
1. **如果是 op_magic**：
   - 调用 `find_operation_by_magic(opmagic)` 获取 OperationInfo
2. **如果是 tensor_magic**：
   - 调用 `find_producer_of_tensor(tensor_magic)` 获取生产者 op
   - 调用 `find_consumers_of_tensor(tensor_magic)` 获取消费者 op 列表
   - 对生产者和每个消费者分别执行后续分析步骤

#### 步骤 4：定位用户代码
1. 从 OperationInfo 获取 `file` 和 `line` 属性
2. 定位到用户代码的具体位置，确定该位置的算子类型

#### 步骤 6：检查算子约束
1. 从 `docs/api/operation/` 目录下获取对应算子的使用说明文档
2. 检查用户代码中该算子的使用是否满足文档约束
3. 检查pass异常位置对该算子的处理逻辑，是否可能导致pass处理后不满足算子约束

#### 步骤 7：确定根因
1. 如果用户代码中算子使用不满足约束：给出异常的参数配置信息，代码位置，不满足的约束点
2. 如果是pass逻辑问题：给出pass逻辑异常代码位置，代码逻辑的解释说明，经过pass处理后不满足约束的原因

---

## 通用分析原则

### 1. 自底向上追溯
- 从错误点开始，向上追溯数据流
- 检查每个 Pass 的输入输出
- 确认数据在哪个环节出现问题

### 2. 结合业务场景
- 分析用户的实际业务需求
- 确认错误场景是否在预期范围内
- 考虑边界情况和特殊输入

### 3. 文档化分析过程
- 记录分析过程中的关键发现
- 记录修复方案的依据
- 为后续类似问题提供参考

