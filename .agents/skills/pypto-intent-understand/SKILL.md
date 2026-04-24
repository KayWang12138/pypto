---
name: pypto-intent-understand
description: "PyPTO 算子需求意图理解。将用户的自然语言算子描述转化为结构化需求文档。当用户描述要开发、实现、创建某个算子时触发，例如：'开发一个 sinh 算子'、'实现 GELU'、'参考 PyTorch 的 F.scaled_dot_product_attention'、'根据论文实现算子'、'创建自定义算子'"
---

# PyPTO 算子需求意图理解

将用户的自然语言描述转化为**结构化、完备**的 PyPTO 算子开发需求文档（SPEC.md）。

## 核心原则

> 严格遵循以下核心原则，确保需求理解准确完整

### 原则 1：基于事实，不猜测

- ✅ 明确标注信息来源和置信度（✓ 高 / ⚠ 中 / ❓ 低）
- ✅ 不确定时主动询问，而非自行推断
- ❌ 禁止凭空猜测公式、规格或参数含义

### 原则 2：功能优先级明确

- ✅ 对所有功能（包括可选参数）标注优先级（P0/P1/P2/P3）
- ✅ P0 和 P1 功能必须在第一个版本实现
- ✅ 明确标注哪些功能暂不支持及原因

### 原则 3：可选参数深度分析

- ✅ 从代码位置、计算逻辑、实现复杂度、依赖关系四个维度分析可选参数
- ✅ 明确参数间的约束和互斥关系
- ❌ 禁止遗漏常用可选参数的分析

### 原则 4：信息完整性

- ✅ 必须信息（算子名称、公式、输入输出规格、动态轴）不可缺失
- ✅ 复杂算子必须提供算法描述
- ✅ 建议提供典型配置，便于后续 golden 生成和设计方案

### 原则 5：至少确认一次

- ✅ 除非用户强烈要求无需确认，否则至少确认一次
- ✅ 最多 2 次确认，整体确认而非逐项确认

### 原则 6：默认值必须显式披露

- ✅ 允许使用默认值的字段，必须在确认环节展示
- ✅ 默认值只能补齐非阻塞信息，不得静默替代关键需求
- ❌ 禁止用户未感知的默认值注入

## 核心工作流

```
阶段 1: 快速解析 → 阶段 2: 可视化确认 → 阶段 3: 可选补充 → 输出文件
```

当用户的首次输入已包含全部必须信息（算子名称、公式/算法、输入输出规格），可将阶段 1 和阶段 2 合并为一步：直接展示数据流图 + 规格确认清单，等待用户确认后生成文件。不需要机械地走完每个阶段。

---

## 阶段 1：快速解析

### 接收用户输入

用户可以用任意方式描述算子需求。接收后立即进行输入分类。

### 输入分类

先判断输入属于以下 4 类中的哪一类：

1. **标准参考类**
   - 适用：常见算子、框架 API 参考、可搜索到标准定义的算子
   - 识别特征：用户直接给出算子名，或明确说“参考 PyTorch/NumPy/TensorFlow/JAX 的 ...”

2. **外部材料类**
   - 适用：论文/文档链接、代码片段参考
   - 识别特征：输入包含 URL、论文链接、文档片段、函数定义、import 语句或明显代码语法

3. **自定义描述类**
   - 适用：自定义算子、无标准定义、无法定位可信外部资料的算子
   - 识别特征：名称明显是自定义，或搜索无法定位可信定义

4. **直接规格类**
   - 适用：用户首次输入已经包含算子名称、公式/算法、输入输出规格等关键信息
   - 识别特征：无需依赖外部资料即可直接进入规格确认

### 复杂度判定与关键特性识别

完成输入分类后，判断是否属于复杂算子并识别关键特性。

**复杂算子判定标准**：
- 涉及多步骤、分块、循环、在线更新、状态维护或多个子算子组合
- 标记后追加以下要求：
  - 必须提供算法描述或可恢复算法描述的材料
  - 必须展示分解后的计算图
  - 必须提高确认强度，确保流程理解无歧义

**关键特性识别**（复杂算子必须执行）：
复杂算子通常包含可独立配置的关键特性，这些特性会显著影响实现复杂度和性能。必须主动识别并拆解，让用户确认。

**常见关键特性清单**：

| 特性类别 | 典型特性 | 适用算子类型 | 影响范围 |
|----------|----------|--------------|----------|
| **数值稳定性** | safe_softmax, logsumexp, numerical_stable | Softmax/归一化/指数运算 | 计算精度、溢出处理 |
| **分块策略** | tiling_strategy, block_size | 大规模矩阵/注意力/卷积 | 内存布局、循环结构 |
| **Mask 机制** | causal_mask, padding_mask, custom_mask, sliding_window | 注意力/序列处理 | 数据流、内存访问、边界处理 |
| **在线算法** | online_softmax, online_reduction, streaming_update | Reduction/Softmax | 计算顺序、中间状态 |
| **量化支持** | int8_quant, fp8_quant, blockwise_quant, dequantize | 量化算子 | 数据类型、缩放因子 |
| **融合模式** | bias_add, residual_add, activation_fused | 通用 | 算子边界、中间结果复用 |
| **动态 Shape** | variable_length, ragged_tensor, dynamic_axis | 序列处理/变长输入 | 内存分配、循环展开 |
| **并行策略** | split_k, split_n, parallel_reduction | 矩阵乘法/Reduction | 并行度、同步开销 |
| **内存优化** | in_place, workspace_reuse, page_attention | 通用 | 内存占用、数据依赖、内存优化 |
| **边界处理** | padding_mode, boundary_condition, edge_handling | 卷积/池化/插值 | 边界精度、计算逻辑 |
| **精度控制** | mixed_precision, accumulator_dtype, rounding_mode | 通用 | 计算精度、数值范围 |
| **随机性** | deterministic, seed_control, dropout_rate | Dropout/随机采样 | 可复现性、随机数生成 |
| **稀疏性** | sparse_pattern, masked_compute, structured_sparsity | 稀疏算子 | 数据布局、计算跳过 |

**特性识别流程**：
1. 从用户描述中提取已提及的特性
2. 从参考材料（论文/文档/代码）中识别隐含特性
3. 生成特性清单，标注置信度和来源
4. 在确认环节让用户逐一确认或修改

### 各类输入的详细处理

#### 类别 1：标准参考类（通常为 ✓ 高置信度）

优先从已有知识或可信参考中恢复标准定义。

已知的常见算子包括：

| 类别 | 算子列表 |
|------|----------|
| 激活函数 | relu, gelu, silu, swish, mish, sigmoid, tanh, softmax, hardswish, hardsigmoid |
| 数学运算 | sin, cos, sinh, cosh, tanh, exp, log, log2, log10, sqrt, rsqrt, pow, abs, sign |
| 归一化 | layer_norm, rms_norm, batch_norm, instance_norm, group_norm |
| 注意力 | scaled_dot_product_attention, multi_head_attention |
| 矩阵运算 | matmul, addmm, bmm, linear, einsum |
| Reduction | sum, mean, max, min, amax, amin, argmax, argmin, prod |
| 逐元素 | add, sub, mul, div, maximum, minimum, clamp |

**处理方式**：
1. 从自身知识库或可信参考中获取公式和标准规格
2. 标记为 ✓ 高置信度，展示并建议确认
3. 若属于多步骤/复杂算子（如 Flash Attention、Scan），追加算法描述要求

**交互示例**：
```text
用户: 开发一个 GELU 算子

助手: 我知道 GELU 算子：

      GELU(x) = x * Φ(x)
            ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715*x³)))

      标准规格：
      - 输入: x [...], float32
      - 输出: y [...], float32 (shape与输入相同)

      ✓ 算子名称: gelu
      ✓ 公式: GELU(x) = x * Φ(x)
      ⚠ 有两个版本：
         1. 精确版 (使用erf)
         2. 近似版 (使用tanh)
      ❓ 你需要哪个版本？
```

#### 类别 2：外部材料类（通常为 ⚠ 中置信度）

**识别特征**：输入包含 URL、论文链接、文档片段、代码片段或函数定义。

**处理方式**：
1. 若是论文/文档链接，使用 WebFetch 获取内容并提取规格信息
2. 若是代码片段，分析代码结构，提取输入输出与计算逻辑
3. 统一标记为 ⚠ 中置信度
4. **必须**让用户确认提取结果是否正确

**交互示例**：
```text
用户: 参考 PyTorch 的 F.scaled_dot_product_attention

助手: 我提取到以下规格信息：

      公式: Attention(Q,K,V) = softmax(QK^T/√d) @ V

      标准规格：
      - Q: [batch, num_heads, seq_len, head_dim]
      - K: [batch, num_heads, seq_len, head_dim]
      - V: [batch, num_heads, seq_len, head_dim]
      - Output: [batch, num_heads, seq_len, head_dim]

      可选参数：
      - attn_mask: 注意力掩码
      - dropout_p: dropout 概率
      - is_causal: 是否因果注意力
      - scale: 缩放因子（默认 1/√d）

      ❓ 你需要支持哪些可选参数？
      ❓ 输入的dtype是什么？(float32/bfloat16)
```

#### 类别 3：自定义描述类（通常为 ❓ 需完整描述）

**识别特征**：名称明显是自定义（如 my_op, custom_fusion）、无法定位可信标准定义、或用户明确说是自定义。

**处理方式**：
1. 明确告知这是自定义算子
2. 引导用户提供必须信息（数学公式、输入输出规格）
3. 提供多种输入方式：公式描述、伪代码、NumPy/PyTorch 代码、流程图描述

```text
助手: 我没有找到 "my_fusion_op" 的标准定义，这似乎是一个自定义算子。

      请提供以下信息：

      必须提供：
      1. 数学公式（例如: y = f(x) = ... 或用文字描述计算逻辑）
      2. 输入输出规格（输入: x [shape] dtype / 输出: y [shape] dtype）

      你可以用以下任一方式提供：
      - 公式描述: "y = relu(x) * sigmoid(gate), x和gate都是[b,s,d]的float32"
      - 伪代码: h1 = relu(x); h2 = sigmoid(gate); y = h1 * h2
      - NumPy/PyTorch 代码: def my_op(x, gate): return torch.relu(x) * torch.sigmoid(gate)
```

#### 类别 4：直接规格类（信息完整度决定置信度）

**识别特征**：用户首次输入已经包含算子名称、公式/算法、输入输出规格等关键信息。

**处理方式**：
1. 直接整理已有信息，生成结构化规格草稿
2. 若信息完整且一致，可直接进入可视化确认
3. 若信息存在空缺，仅补问缺失项，不重复追问已有信息

### 置信度标记规范

| 标记 | 置信度 | 信息来源 | 处理方式 |
|------|--------|----------|----------|
| ✓ | 高 | 自身知识库/框架知识 | 展示并建议确认 |
| ⚠ | 中 | WebSearch/WebFetch/代码分析 | **必须**让用户确认 |
| ❓ | 低 | 推断/猜测 | 主动询问用户 |

---

## 阶段 2：可视化确认

这是**最关键的一步**，确保理解与用户意图对齐。

### 展示内容

按以下顺序展示：

#### 0. 关键特性拆解（复杂算子专属）

对于复杂算子，在展示数据流图之前，**必须**先展示识别出的关键特性清单：

```
🔍 识别到的关键特性:

| 特性 | 是否需要 | 置信度 | 来源 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| causal_mask | ✓ 需要 | ✓ 高 | 用户描述 | 需要上三角 mask 逻辑 | P0 |
| online_softmax | ✓ 需要 | ⚠ 中 | 论文推断 | 需要分块 + 数值稳定更新 | P0 |
| multi_query_attention | ✗ 不需要 | ❓ 低 | 默认假设 | - | - |
| paged_attention | ? 待确认 | ❓ 低 | 网络搜索 | 需要KV cache管理 | P1 |
| dropout | ? 待确认 | ❓ 低 | 框架默认 | 需要随机数生成 | P2 |

❓ 请确认以上特性是否符合你的需求：
  - 需要调整哪些特性的"是否需要"状态？
  - 是否有遗漏的特性？
```

**特性确认交互**：
- 使用 `AskUserQuestion` 让用户确认特性清单
- 提供选项：全部确认 / 修改特性 / 添加新特性 / 我不确定，帮我解释
- 若用户不确定，提供每个特性的简要说明和典型使用场景

#### 1. ASCII 数据流图

根据算子类型选择合适的数据流图模板：

**简单 Element-wise 算子**（单输入，shape 不变）：
```
    输入 x                      输出 y
┌────────────────┐         ┌────────────────┐
│  [b, s, n, d]  │ ──────▶ │  [b, s, n, d]  │
│   float32      │  sinh   │   float32      │
└────────────────┘         └────────────────┘

公式: y = sinh(x) = (e^x - e^(-x)) / 2
动态轴: b, s
```

**多输入 Matmul 类算子**：
```
    输入 a           输入 b           输入 c (可选)
┌──────────┐    ┌──────────┐    ┌──────────┐
│  [m, k]  │    │  [n, k]  │    │  [m, n]  │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │    ┌──────────┘               │
     ▼    ▼                          │
  ┌────────────┐                     │
  │  a @ b^T   │                     │
  └─────┬──────┘                     │
        │                            │
        ▼                            ▼
     ┌──────────────────────────────────┐
     │              +                   │
     └───────────────┬──────────────────┘
                   ▼
             ┌──────────┐
             │ 输出 y   │
             │  [m, n]  │
             └──────────┘
```

**其他算子**：参考上述模板，按实际数据流绘制。组合算子展示子算子的分解与合并关系。

#### 2. 规格确认清单

对于简单算子（公式足以描述）：
```
✅ 必须信息:
  [✓] 算子名称: {name}
  [✓] 数学公式: {formula}
  [✓] 输入: {input_name}[{shape}] {dtype}, 动态轴: {dynamic_axes}
  [✓] 输出: {output_name}[{shape}] {dtype}, 动态轴: {dynamic_axes}

⚙️ 自动推断（使用默认值，可修改）:
  • 支持数据类型: float32 (atol=0.001, rtol=0.001)
  • 性能目标: 无特殊要求
  • 边界处理: 正常计算
```

对于复杂算子（公式无法完整表述计算流程），增加算法部分和关键特性部分：
```
✅ 必须信息:
  [✓] 算子名称: {name}
  [✓] 数学公式: {formula}
  [✓] 关键特性:
      ┌─────────────────────┬────────┬──────────────────────────┐
      │ 特性                │ 状态   │ 实现说明                 │
      ├─────────────────────┼────────┼──────────────────────────┤
      │ causal_mask         │ ✓ 需要 │ 上三角 mask 逻辑         │
      │ online_softmax      │ ✓ 需要 │ 分块 + 数值稳定更新      │
      │ paged_attention     │ ✗ 不需 │ -                        │
      │ dropout             │ ✗ 不需 │ -                        │
      └─────────────────────┴────────┴──────────────────────────┘
  [✓] 算法描述:
      Algorithm: {algorithm_name}
      ────────────────────────────
      {带编号的伪代码步骤}
  [✓] 输入: ...
  [✓] 输出: ...
```

**何时需要算法描述**：当计算涉及分块、循环、在线更新、状态维护等流程性逻辑时，公式只能描述数学语义但无法描述实现策略。此时需要算法描述来说明"怎么算"。典型例子：Flash Attention 的分块计算 + Online Softmax、Scan 类算子的递推过程。

**算法描述格式**：使用带编号的伪代码步骤，清晰展示循环结构、分块策略、状态更新等流程。示例：

```
Algorithm: Flash Attention (Forward)
────────────────────────────────────
输入: Q, K, V ∈ R^{N×d}, 分块大小 Br, Bc
输出: O ∈ R^{N×d}

1. 将 Q 分为 Tr = ⌈N/Br⌉ 块, K/V 分为 Tc = ⌈N/Bc⌉ 块
2. 初始化 O = 0, l = 0, m = -∞
3. for j = 1 to Tc:                          // 外层循环: K/V 块
     3.1 从 HBM 加载 K_j, V_j 到 SRAM
     3.2 for i = 1 to Tr:                    // 内层循环: Q 块
           3.2.1 从 HBM 加载 Q_i, O_i, l_i, m_i 到 SRAM
           3.2.2 S_ij = Q_i @ K_j^T  ∈ R^{Br×Bc}
           3.2.3 m̃_ij = rowmax(S_ij)
           3.2.4 P̃_ij = exp(S_ij - m̃_ij)
           3.2.5 l̃_ij = rowsum(P̃_ij)
           3.2.6 m_new = max(m_i, m̃_ij)
           3.2.7 l_new = exp(m_i - m_new) * l_i + exp(m̃_ij - m_new) * l̃_ij
           3.2.8 O_i ← diag(l_new)^{-1} * (diag(l_i) * exp(m_i - m_new) * O_i
                        + exp(m̃_ij - m_new) * P̃_ij @ V_j)
           3.2.9 写回 O_i, l_new, m_new 到 HBM
4. return O
```

#### 3. 操作选项

使用 `AskUserQuestion` 工具向用户提供以下选择：
- **修改某项** — 修改以上任何信息
- **添加更多细节** — 进入阶段 3
- **确认无误，继续** — 跳过阶段 3 直接生成文件

### 确认策略

- **高置信度 (✓)**：展示但不强制确认，用户可以直接继续
- **中置信度 (⚠)**：**必须**确认 1 次
- **确认次数**：最多 2 次（分类确认（仅中置信度时）+ 最终确认）
- **确认方式**：整体确认，不逐项确认
- **多步骤/复杂算子**：分解展示，整体确认

---

## 2.5 Golden 参考实现策略指导

在确认必须信息之后、进入可选补充前，如果算子属于**分块算子**（即涉及 loop、分块计算、累加等流程性逻辑），应向用户指导 golden 参考实现的两种可选策略：

### 全量 vs 分块实现策略

**Golden 函数可以采用两种等价的实现策略**：

#### 1. 全量实现（默认）
一次性对整个输入 tensor 计算，适用于简单、逐元素的算子。
```python
def relu_golden(x):
    return torch.relu(x)
```

#### 2. 分块实现（可选，推荐用于复杂算子）
使用 `for` loop 将输入分切成小 tile，逐 tile 计算，最后用 `torch.cat` 或类似方式组合结果。
这种实现方式更贴近 PyPTO kernel 的实际执行方式（PyPTO kernel 也是 tile-by-tile 处理），能更好地：
- 验证 PyPTO kernel 的边界处理（padding、越界访问）
- 验证累积逻辑（多个 tile 的部分结果如何合并）
- 验证 tile size 选择对数值精度的影响

```python
def attention_golden(q, k, v, window_size=None):
    """分块 attention（演示如何将计算分解为多个 tile）。"""
    outputs = []
    for b in range(q.shape[0]):
        q_tile = q[b:b+1, ...]  # 切出单个 batch
        k_tile = k[b:b+1, ...]
        v_tile = v[b:b+1, ...]
        # 计算该 tile 的结果
        out_tile = _compute_attention_single_tile(q_tile, k_tile, v_tile, window_size)
        outputs.append(out_tile)
    return torch.cat(outputs, dim=0)
```

**何时应该选择分块实现**：
- 算子规格中包含 loop 或分块策略描述
- 计算涉及多步骤分块累加（如 FlashAttention、分块矩阵乘法）
- 需要提前验证 PyPTO kernel 的分块边界处理

**两种实现的要求**：
- 必须数值等价（允许浮点精度偏差 1e-5）
- 都应该在测试中验证
- Golden 函数选定后，PyPTO kernel 会对照该实现进行精度验证

---

## 阶段 3：可选补充

如果用户选择"添加更多细节"，通过 `AskUserQuestion` 展示以下可选配置项：

1. **动态轴范围** — 动态轴的取值范围（默认: 无限制）
2. **性能目标** — 是否有特殊性能指标要求（默认: 无特殊要求）
3. **边界条件处理** — 零值/极值/NaN/Inf 的处理方式（默认: 正常计算）
4. **参考实现** — PyTorch/NumPy 参考实现链接或代码
5. **应用场景与典型配置** — 目标模型、使用位置、典型配置表格（强烈建议提供）

   典型配置用于后续 golden 验证和设计方案生成，采用 7 列格式：

   | 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
   |----------|------|--------|------|------------|------------|------|
   | 性能_P0 | 性能 | P0 | {关键参数} | {各输入shape} | {输出shape} | 核心性能场景 |
   | 功能_P0 | 功能 | P0 | {关键参数} | {各输入shape} | {输出shape} | 核心功能验证 |

   **引导流程**：若用户选择提供应用场景，主动询问是否提供典型配置。若用户不提供典型配置，根据动态轴范围推荐默认配置供确认。

用户可以只填写关心的项，其余使用默认值。

---


---

## Output Generation and Field Reference

Read `references/spec-examples.md` for:
- Output file generation rules
- File conflict handling
- Field definitions and default values
- Optional parameter deep analysis
- Feature priority system
- Requirements understanding checklist
