# PyPTO 融合算子整网集成 Skill

## Skill 基本信息

**名称：** PyPTO 融合算子整网集成

**描述：** 将 PyPTO 融合算子替换到整网中，替代多个小算子组合的完整工作流程。包含理解验证、算子开发、模型集成、精度验证与性能调优的全流程指导。

**触发词：** 融合算子替换、整网集成、算子融合、替换小算子、模型算子替换、融合算子集成、fused op integration、replace small ops、model integration

**适用场景：**
- 将多个小算子融合为一个 PyPTO 算子
- 将 PyPTO 算子集成到 vllm 等推理框架
- GLM、LLaMA 等大模型的算子优化
- MoE、Attention 等复杂结构的算子融合

**前置条件：**
- 已有 PyPTO 开发环境
- 已有目标模型源码（如 vllm 工程）
- 已了解要替换的算子位置和上下文

---

## 工作流程概览

```
阶段一：前置准备
  ├─ 步骤 1：环境与工程验证
  └─ 步骤 2：算子需求分析

阶段二：理解验证（关键环节）⚠️
  ├─ 步骤 3.1：分析原始小算子组合
  ├─ 步骤 3.2：编写 Golden 脚本
  ├─ 步骤 3.3：验证理解正确性
  └─ 步骤 3.4：决策与迭代

阶段三：PyPTO 算子开发（如已有可跳过）
  ├─ 步骤 4：设计方案
  ├─ 步骤 5：算子实现
  └─ 步骤 6：单算子验证

阶段四：模型集成
  ├─ 步骤 7：调整目录结构
  ├─ 步骤 8：配置适配层
  └─ 步骤 9：修改模型调用逻辑

阶段五：整网验证与调优
  ├─ 步骤 10：端到端精度验证
  ├─ 步骤 11：性能分析与调优
  └─ 步骤 12：问题排查与修复

阶段六：提交与文档
  ├─ 步骤 13：创建 Issue 跟踪
  └─ 步骤 14：提交 PR
```

---

## 详细步骤指南

### 阶段一：前置准备

#### 步骤 1：环境与工程验证

**目标：** 确保开发环境和目标工程就绪

**操作：**

1. **验证 PyPTO 环境**
   ```bash
   # 检查 NPU 环境
   npu-smi info
   
   # 检查 PyPTO 安装
   python -c "import pypto; print(pypto.__version__)"
   
   # 检查 torch_npu
   python -c "import torch_npu; print(torch_npu.__version__)"
   ```

2. **验证目标工程**
   ```bash
   # 检查目标模型工程结构
   # 例如 vllm 工程
   ls -la vllm/
   ls -la vllm_ascend/
   
   # 运行原始模型，确认基准正常
   python examples/run_glm4.py  # 示例命令
   ```

3. **记录基准性能**
   ```bash
   # 记录原始实现的性能数据
   python examples/run_glm4.py --benchmark > baseline_perf.txt
   ```

**输出物：**
- 环境检查报告
- 原始模型运行日志
- 基准性能数据

**推荐 Skill：** `pypto-environment-setup`

---

#### 步骤 2：算子需求分析

**目标：** 明确要替换的算子组合和融合目标

**操作：**

1. **定位目标算子**
   - 在模型代码中找到要优化的性能热点
   - 识别可以融合的小算子组合
   - 记录算子的调用位置和上下文

   **示例：**
   ```python
   # 目标文件：vllm/model_executor/models/glm4_moe.py
   # 目标函数：Glm4MoeDecoderLayer.forward()
   
   # 原始小算子组合：
   # layernorm → quantization → matmul → dequantization → normalize → rotary_emb
   ```

2. **分析输入输出规格**
   ```python
   # 输入规格
   - hidden_states: [bs, hidden_dim], dtype=float16
   - residual: [bs, hidden_dim], dtype=float16 or None
   - layer weights: multiple tensors
   
   # 输出规格
   - q: [bs, num_heads, head_dim]
   - k: [bs, num_kv_heads, head_dim]
   - v: [bs, num_kv_heads, head_dim]
   - residual: [bs, hidden_dim]
   ```

3. **确定融合收益**
   - 减少内存访问次数
   - 减少算子启动开销
   - 提高数据局部性
   - 预估性能提升空间

**输出物：**
- 算子需求分析文档（记录目标文件、函数、输入输出规格）
- 融合收益分析报告

**推荐 Skill：** `pypto-intent-understanding`（如需开发新算子）

---

### 阶段二：理解验证（关键环节）⚠️

> **为什么需要这个阶段？**
> 
> 在分析整网中的小算子组合时，我们往往无法直接确定其精确实现细节。只能根据算子名称、前后依赖关系，人为推测计算逻辑。这个推测是否正确，需要通过 Golden 验证来确认。

#### 步骤 3.1：分析原始小算子组合

**目标：** 深入理解原始实现的计算逻辑

**操作：**

1. **阅读原始代码**
   ```python
   # 示例：GLM-4.5 的 attention_pre 计算
   # 文件：vllm/model_executor/models/glm4_moe.py
   # 位置：Glm4MoeDecoderLayer.forward()
   
   # Step 1: LayerNorm with residual
   if residual is None:
       residual = hidden_states
       hidden_states = self.input_layernorm(hidden_states)
   else:
       hidden_states, residual = self.input_layernorm(
           hidden_states, residual)
   
   # Step 2: QKV projection (with quantization)
   qkv, _ = self.qkv_proj(hidden_states)
   
   # Step 3: Split Q, K, V
   q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
   
   # Step 4: Q/K normalization
   if self.use_qk_norm:
       q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q.shape)
       k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(k.shape)
   
   # Step 5: Rotary embedding
   q, k = self.rotary_emb(positions, q, k)
   ```

2. **梳理数据流**
   ```
   hidden_states (input)
       ↓
   [LayerNorm] → normalized_hidden
       ↓
   [Quantization] → quant_hidden (int8)
       ↓
   [MatMul: gate_up_proj] → qkv_quant
       ↓
   [Dequantization] → qkv (float16)
       ↓
   [Split] → q, k, v
       ↓
   [Q/K Norm] → q_norm, k_norm
       ↓
   [Rotary Emb] → q_rot, k_rot, v
       ↓
   q, k, v, residual (output)
   ```

3. **分析隐藏细节**
   - 检查是否有隐式的 reshape、transpose、contiguous
   - 注意 dtype 转换（如 float32 → float16）
   - 检查是否有边界条件处理
   - 记录所有中间变量的形状

**输出物：**
- 原始代码片段注释版
- 数据流图
- 隐藏细节清单

---

#### 步骤 3.2：编写 Golden 脚本

**目标：** 基于理解编写 PyTorch 参考实现

**操作：**

1. **创建 Golden 文件**
   ```bash
   mkdir -p models/glm_v4_5
   touch models/glm_v4_5/glm_attention_pre_quant_golden.py
   ```

2. **编写 Golden 实现**
   ```python
   # models/glm_v4_5/glm_attention_pre_quant_golden.py
   import torch
   import torch.nn.functional as F
   
   def attention_pre_quant_golden(
       hidden_states: torch.Tensor,
       residual: torch.Tensor,
       ln_weight: torch.Tensor,
       ln_bias: torch.Tensor,
       qkv_input_scale: torch.Tensor,
       qkv_input_offset: torch.Tensor,
       qkv_weight: torch.Tensor,
       qkv_quant_bias: torch.Tensor,
       qkv_deq_scale: torch.Tensor,
       q_norm_weight: torch.Tensor,
       q_norm_bias: torch.Tensor,
       k_norm_weight: torch.Tensor,
       k_norm_bias: torch.Tensor,
       cos: torch.Tensor,
       sin: torch.Tensor,
   ):
       """
       基于 PyTorch 的 Golden 参考实现
       
       计算流程：
       1. LayerNorm with residual connection
       2. Quantization (int8)
       3. QKV projection (matmul)
       4. Dequantization (fp16)
       5. Split Q, K, V
       6. Q/K normalization
       7. Rotary position embedding
       
       Args:
           hidden_states: [bs, hidden_dim]
           residual: [bs, hidden_dim] or None
           ... (其他参数说明)
       
       Returns:
           q: [bs, num_heads, head_dim]
           k: [bs, num_kv_heads, head_dim]
           v: [bs, num_kv_heads, head_dim]
           residual_out: [bs, hidden_dim]
       """
       
       # ========== Step 1: LayerNorm with residual ==========
       if residual is None:
           residual_out = hidden_states
           normalized = F.layer_norm(
               hidden_states, 
               [hidden_states.shape[-1]], 
               ln_weight, 
               ln_bias
           )
       else:
           # Fused layernorm with residual
           normalized = F.layer_norm(
               hidden_states, 
               [hidden_states.shape[-1]], 
               ln_weight, 
               ln_bias
           )
           residual_out = hidden_states  # 原始 residual 作为输出
       
       # ========== Step 2: Quantization ==========
       # Scale and offset for int8 quantization
       quant_input = (normalized * qkv_input_scale + qkv_input_offset).to(torch.int8)
       
       # ========== Step 3: QKV projection ==========
       qkv = F.linear(quant_input.float(), qkv_weight, qkv_quant_bias)
       
       # ========== Step 4: Dequantization ==========
       qkv = qkv * qkv_deq_scale
       
       # ========== Step 5: Split Q, K, V ==========
       q_size = qkv_weight.shape[0] * 2 // 3  # 假设 q_size = 2 * kv_size
       kv_size = qkv_weight.shape[0] // 3
       q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
       
       # ========== Step 6: Q/K normalization ==========
       bs = hidden_states.shape[0]
       num_heads = q_size // 64  # 假设 head_dim = 64
       num_kv_heads = kv_size // 64
       head_dim = 64
       
       # Reshape for layer norm
       q = q.view(bs, num_heads, head_dim)
       k = k.view(bs, num_kv_heads, head_dim)
       
       # Apply layer norm
       q = F.layer_norm(q, [head_dim], q_norm_weight, q_norm_bias)
       k = F.layer_norm(k, [head_dim], k_norm_weight, k_norm_bias)
       
       # ========== Step 7: Rotary embedding ==========
       # Apply rotary position embedding
       q_rot = apply_rotary_pos_emb(q, cos, sin)
       k_rot = apply_rotary_pos_emb(k, cos, sin)
       
       return q_rot, k_rot, v, residual_out
   
   
   def apply_rotary_pos_emb(x, cos, sin):
       """应用旋转位置编码"""
       # x: [bs, num_heads, head_dim]
       # cos, sin: [bs, 1, head_dim//2]
       
       head_dim = x.shape[-1]
       x1 = x[..., :head_dim//2]
       x2 = x[..., head_dim//2:]
       
       # Rotary embedding formula
       out1 = x1 * cos - x2 * sin
       out2 = x1 * sin + x2 * cos
       
       return torch.cat([out1, out2], dim=-1)
   ```

3. **编写独立测试**
   ```python
   # test_golden.py
   import torch
   
   def test_attention_pre_quant_golden():
       # 构造测试输入
       bs = 2
       hidden_dim = 4096
       num_heads = 32
       num_kv_heads = 8
       head_dim = 128
       
       hidden_states = torch.randn(bs, hidden_dim, dtype=torch.float16, device='npu:0')
       residual = torch.randn(bs, hidden_dim, dtype=torch.float16, device='npu:0')
       
       # 构造权重（简化版）
       ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='npu:0')
       ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='npu:0')
       
       # ... (其他权重)
       
       # 调用 golden 实现
       q, k, v, residual_out = attention_pre_quant_golden(
           hidden_states, residual,
           ln_weight, ln_bias,
           # ... (其他参数)
       )
       
       # 检查输出形状
       assert q.shape == (bs, num_heads, head_dim)
       assert k.shape == (bs, num_kv_heads, head_dim)
       assert v.shape == (bs, num_kv_heads, head_dim)
       assert residual_out.shape == (bs, hidden_dim)
       
       print("Golden test passed!")
   
   if __name__ == "__main__":
       test_attention_pre_quant_golden()
   ```

**输出物：**
- Golden 实现文件（`*_golden.py`）
- 独立测试脚本

**推荐 Skill：** `pypto-golden-generator`

---

#### 步骤 3.3：验证理解正确性

**目标：** 将 Golden 脚本集成到整网中，验证理解是否正确

**操作：**

1. **创建适配层临时版本**
   ```python
   # glm_pto_kernels/__init__.py (临时版本，用于验证)
   
   def attention_pre(hidden_states, residual, layer, attention, positions):
       """
       [验证阶段] 使用 Golden 实现验证理解正确性
       """
       # 导入 golden 实现
       from glm_pto_kernels.glm_attention_pre_quant_golden import attention_pre_quant_golden
       
       # 准备参数
       cos, sin = attention.rotary_emb.cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
       cos = cos.unsqueeze(1).contiguous()
       sin = sin.unsqueeze(1).contiguous()
       
       # 调用 golden 实现
       q, k, v, residual_res = attention_pre_quant_golden(
           hidden_states, residual,
           layer.input_layernorm.weight.data,
           layer.input_layernorm.bias.data,
           attention.qkv_proj.aclnn_input_scale_reciprocal.data,
           attention.qkv_proj.aclnn_input_offset.data,
           attention.qkv_proj.weight.data,
           attention.qkv_proj.quant_bias.data,
           attention.qkv_proj.deq_scale.data,
           attention.q_norm.weight.data,
           attention.q_norm.bias.data,
           attention.k_norm.weight.data,
           attention.k_norm.bias.data,
           cos, sin
       )
       
       return q, k, v, residual_res
   ```

2. **修改模型调用逻辑**
   ```python
   # vllm/model_executor/models/glm4_moe.py
   
   # 在文件顶部添加导入
   import glm_pto_kernels
   
   # 在 Glm4MoeAttention.forward() 中修改
   def forward(
       self,
       layer,  # 增加 layer 输入
       positions: torch.Tensor,
       hidden_states: torch.Tensor,
       residual: torch.Tensor,  # 增加 residual 输入
   ) -> torch.Tensor:
       # 使用 golden 实现
       q, k, v, residual = glm_pto_kernels.attention_pre(
           hidden_states, residual, layer, self, positions
       )
       
       # 后续使用 q, k, v, residual
       attn_output = self.attn(q, k, v)
       output, _ = self.o_proj(attn_output)
       return output, residual
   ```

3. **运行整网验证**
   ```bash
   # 运行模型，检查是否能正常运行
   python examples/run_glm4.py --input "测试输入"
   
   # 检查输出是否正常
   # 检查中间激活值是否合理
   # 检查是否有 NaN/Inf
   ```

4. **端到端精度对比**
   ```python
   # compare_outputs.py
   import torch
   
   # 加载原始模型输出
   original_output = torch.load("original_model_output.pt")
   
   # 加载 golden 替换后的输出
   golden_output = torch.load("golden_model_output.pt")
   
   # 对比精度
   diff = torch.abs(original_output - golden_output)
   max_diff = torch.max(diff).item()
   mean_diff = torch.mean(diff).item()
   
   print(f"Max diff: {max_diff}")
   print(f"Mean diff: {mean_diff}")
   
   # 判断是否在可接受范围内
   if max_diff < 1e-3:
       print("✅ Golden 验证通过！理解正确。")
   else:
       print("❌ Golden 验证失败！需要重新审视理解。")
   ```

**验证检查点：**
- ✅ 模型能正常启动和运行
- ✅ 中间激活值形状、dtype 正确
- ✅ 无 NaN/Inf 等异常值
- ✅ 下游算子能正确接收输出
- ✅ 端到端输出与原始实现对齐（diff < 1e-3）

---

#### 步骤 3.4：决策与迭代

**目标：** 根据验证结果决定下一步行动

**场景 1：验证通过 ✅**
- Golden 脚本可作为 PyPTO 算子的精度标杆
- 进入阶段三：PyPTO 算子开发
- Golden 文件后续用于精度对比

**场景 2：验证失败 ❌**

**常见问题排查：**

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| 运行时报错（shape mismatch） | 参数传递错误 | 检查 tensor shape、dtype 是否匹配 |
| 输出全为 NaN | 数值溢出或除零 | 检查 normalization、scale 是否合理 |
| 输出与原始差异大 | 计算逻辑理解错误 | 重新阅读原始代码，检查隐藏操作 |
| 下游算子报错 | 输出格式不对 | 检查输出 tensor 的 memory format |
| 性能异常慢 | Golden 实现过于低效 | Golden 只关注正确性，性能不影响验证 |

**排查流程：**
```python
# 逐层打印中间结果
def attention_pre_quant_golden_debug(...):
    # Step 1
    normalized = F.layer_norm(...)
    print(f"After LayerNorm: shape={normalized.shape}, mean={normalized.mean()}, std={normalized.std()}")
    
    # Step 2
    quant_input = ...
    print(f"After Quant: shape={quant_input.shape}, min={quant_input.min()}, max={quant_input.max()}")
    
    # Step 3
    qkv = F.linear(...)
    print(f"After QKV: shape={qkv.shape}, mean={qkv.mean()}, std={qkv.std()}")
    
    # ... 继续每一步
```

**迭代策略：**
1. 定位问题步骤
2. 对比原始实现的中间结果
3. 修正理解或参数传递
4. 重新运行验证
5. 重复直到通过

---

### 阶段三：PyPTO 算子开发（如已有可跳过）

> **前提条件：** 步骤 3.3 验证通过，Golden 实现正确

#### 步骤 4：设计方案

**目标：** 设计 PyPTO 算子的实现方案

**操作：**

1. **分析计算特性**
   - 计算密集型 vs 访存密集型
   - 数据依赖关系
   - 并行化机会

2. **设计 Tiling 策略**
   ```python
   # design.md 示例
   ## Tiling Strategy
   
   ### 输入分块
   - hidden_states: [BS, HIDDEN_DIM] → 分块为 [TILE_BS, TILE_HIDDEN]
   - weights: 根据 MatMul 算子分块策略
   
   ### 计算流程
   1. LayerNorm: 按 TILE_BS 分块
   2. Quantization: 逐元素，无需分块
   3. MatMul: 使用 CUBE 算子，按标准 Tiling
   4. Dequantization: 逐元素，无需分块
   5. Split: 逐元素，无需分块
   6. LayerNorm: 按 TILE_BS 分块
   7. Rotary Emb: 按 TILE_BS 分块
   ```

3. **API 映射分析**
   ```python
   # API 映射表
   | 原始操作 | PyPTO API | 备注 |
   |---------|-----------|------|
   | LayerNorm | pto.layer_norm | 支持 fused residual |
   | Quantization | pto.quantize | int8 量化 |
   | MatMul | pto.matmul | CUBE 算子 |
   | Dequantization | pto.dequantize | 反量化 |
   | Split | pto.split | 张量切分 |
   | LayerNorm | pto.layer_norm | Q/K normalization |
   | Rotary Emb | 自定义实现 | 需要编写 TILING 实现 |
   ```

**输出物：**
- design.md 设计文档

**推荐 Skill：** `pypto-op-design`

---

#### 步骤 5：算子实现

**目标：** 编写 PyPTO 算子实现代码

**操作：**

1. **创建算子文件**
   ```bash
   mkdir -p models/glm_v4_5
   touch models/glm_v4_5/glm_attention_pre_quant.py
   ```

2. **编写算子实现**
   ```python
   # models/glm_v4_5/glm_attention_pre_quant.py
   import torch
   import pypto as pto
   
   def attention_pre_quant(
       hidden_states, residual,
       ln_weight, ln_bias,
       qkv_input_scale, qkv_input_offset,
       qkv_weight, qkv_quant_bias, qkv_deq_scale,
       q_norm_weight, q_norm_bias,
       k_norm_weight, k_norm_bias,
       cos, sin,
       q_out, k_out, v_out, residual_out
   ):
       """
       PyPTO 实现的 attention_pre_quant 算子
       
       Args:
           hidden_states: [bs, hidden_dim], input FP16
           residual: [bs, hidden_dim] or None
           ... (其他参数)
       
       Returns:
           q_out: [bs, num_heads, head_dim], output tensor
           k_out: [bs, num_kv_heads, head_dim], output tensor
           v_out: [bs, num_kv_heads, head_dim], output tensor
           residual_out: [bs, hidden_dim], output tensor
       """
       
       bs = hidden_states.shape[0]
       hidden_dim = hidden_states.shape[1]
       num_heads = q_norm_weight.shape[0]
       num_kv_heads = k_norm_weight.shape[0]
       head_dim = q_norm_weight.shape[1]
       
       # ========== PyPTO 实现 ==========
       # 使用 PyPTO API 实现融合算子
       # ... (具体实现)
       
       pass
   ```

3. **编写测试脚本**
   ```python
   # test_attention_pre_quant.py
   import torch
   import pypto as pto
   from models.glm_v4_5.glm_attention_pre_quant import attention_pre_quant
   from models.glm_v4_5.glm_attention_pre_quant_golden import attention_pre_quant_golden
   
   def test_precision():
       # 构造测试输入
       bs = 2
       hidden_dim = 4096
       # ... (构造输入)
       
       # 调用 PyPTO 实现
       q_pto, k_pto, v_pto, residual_pto = attention_pre_quant(...)
       
       # 调用 Golden 实现
       q_golden, k_golden, v_golden, residual_golden = attention_pre_quant_golden(...)
       
       # 对比精度
       def compare_tensor(name, pto_tensor, golden_tensor):
           diff = torch.abs(pto_tensor - golden_tensor)
           max_diff = torch.max(diff).item()
           mean_diff = torch.mean(diff).item()
           print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
           assert max_diff < 1e-3, f"{name} precision check failed"
       
       compare_tensor("q", q_pto, q_golden)
       compare_tensor("k", k_pto, k_golden)
       compare_tensor("v", v_pto, v_golden)
       compare_tensor("residual", residual_pto, residual_golden)
       
       print("✅ Precision test passed!")
   
   if __name__ == "__main__":
       test_precision()
   ```

**输出物：**
- 算子实现文件（`*.py`）
- 测试脚本

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 6：单算子验证

**目标：** 验证 PyPTO 算子实现正确性

**操作：**

1. **运行单算子测试**
   ```bash
   # 编译算子
   python -c "import models.glm_v4_5.glm_attention_pre_quant"
   
   # 运行精度测试
   python test_attention_pre_quant.py
   ```

2. **精度对比**
   - 与 Golden 实现对比
   - 允许误差范围：max_diff < 1e-3

3. **性能测试**
   ```bash
   # 运行性能测试
   python benchmark_attention_pre_quant.py
   
   # 记录性能数据
   # - 执行时间
   # - 内存占用
   # - AI Core 利用率
   ```

**验证检查点：**
- ✅ 精度测试通过
- ✅ 无运行时错误
- ✅ 性能满足预期

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 7：调整目录结构

**目标：** 创建 PyPTO 算子库目录结构

**操作：**

```bash
# 在模型工程根目录下创建
cd glm-net/

# 创建 PyPTO 算子库目录
mkdir -p glm_pto_kernels

# 复制已验证的 PyPTO 算子文件
cp models/glm_v4_5/glm_attention_pre_quant.py glm_pto_kernels/

# 创建包初始化文件
touch glm_pto_kernels/__init__.py

# 调整后的目录结构
# glm-net/
# ├── glm_pto_kernels/
# │   ├── __init__.py
# │   ├── glm_attention_pre_quant.py
# │   └── ... (其他算子)
# ├── vllm/
# │   └── model_executor/models/glm4_moe.py
# └── vllm_ascend/
#     └── ...
```

**输出物：**
- 标准化的目录结构

---

#### 步骤 8：配置适配层

**目标：** 在适配层中封装 PyPTO 算子调用

**操作：**

1. **编写适配层函数**
   ```python
   # glm_pto_kernels/__init__.py
   
   import torch
   
   # 配置算子开关，可灵活切换
   USE_PTO_ATTENTION_PRE = True
   
   def attention_pre(hidden_states, residual, layer, attention, positions):
       """
       [适配层] 调用 PyPTO 实现的 attention_pre 算子
       
       Args:
           hidden_states: [bs, hidden_dim], 输入隐藏状态
           residual: [bs, hidden_dim] or None, 残差连接
           layer: 包含 layernorm 权重的层对象
           attention: 包含 qkv_proj、q_norm、k_norm 的注意力对象
           positions: 位置索引
       
       Returns:
           q: [bs, num_heads, head_dim]
           k: [bs, num_kv_heads, head_dim]
           v: [bs, num_kv_heads, head_dim]
           residual_out: [bs, hidden_dim]
       """
       if not USE_PTO_ATTENTION_PRE:
           # 如果未开启，返回 None，由模型使用原始实现
           return None
       
       # 导入 PyPTO 实现
       from glm_pto_kernels.glm_attention_pre_quant import attention_pre_quant as attention_pre_quant_pto
       
       # 准备 rotary embedding 的 cos/sin
       cos, sin = attention.rotary_emb.cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
       cos = cos.unsqueeze(1).contiguous()
       sin = sin.unsqueeze(1).contiguous()
       
       # 获取 batch size 和维度信息
       bs = hidden_states.shape[0]
       q_size = attention.q_size
       kv_size = attention.kv_size
       
       # 分配输出张量
       q = torch.empty((bs, q_size), dtype=hidden_states.dtype, device=hidden_states.device)
       k = torch.empty((bs, kv_size), dtype=hidden_states.dtype, device=hidden_states.device)
       v = torch.empty((bs, kv_size), dtype=hidden_states.dtype, device=hidden_states.device)
       residual_res = torch.empty((bs, hidden_states.shape[1]), 
                                   dtype=hidden_states.dtype, device=hidden_states.device)
       
       # 处理 residual 为 None 的情况
       if residual is None:
           residual = torch.zeros((bs, hidden_states.shape[1]), 
                                   dtype=hidden_states.dtype, device=hidden_states.device)
       
       # 调用 PyPTO 算子
       attention_pre_quant_pto(
           hidden_states, residual,
           layer.input_layernorm.weight.data,
           layer.input_layernorm.bias.data,
           attention.qkv_proj.aclnn_input_scale_reciprocal.data,
           attention.qkv_proj.aclnn_input_offset.data,
           attention.qkv_proj.weight.data,
           attention.qkv_proj.quant_bias.data,
           attention.qkv_proj.deq_scale.data,
           attention.q_norm.weight.data,
           attention.q_norm.bias.data,
           attention.k_norm.weight.data,
           attention.k_norm.bias.data,
           cos, sin,
           q, k, v, residual_res
       )
       
       return q, k, v, residual_res
   ```

2. **配置其他算子适配层**
   ```python
   # glm_pto_kernels/__init__.py
   
   # Gate 算子
   USE_PTO_GATE = True
   
   def gate(gate_layer, hidden_states):
       """Gate 算子适配层"""
       if not USE_PTO_GATE:
           return None
       
       from glm_pto_kernels.glm_gate import gate as gate_pto
       
       bs = hidden_states.shape[0]
       ne = gate_layer.weight.shape[0]
       
       router_logits_res = torch.empty(
           (bs, ne),
           dtype=gate_layer.weight.dtype,
           device=hidden_states.device
       )
       
       gate_pto(gate_layer.weight, hidden_states, router_logits_res)
       
       return router_logits_res
   
   
   # Paged Attention 算子
   USE_PTO_FA = True
   
   def paged_attention(query, key_cache, value_cache, block_tables, actual_seqs_cpu, output):
       """Paged Attention 算子适配层"""
       if not USE_PTO_FA:
           return None
       
       from glm_pto_kernels.glm_attention import attention
       
       attention(query, key_cache, value_cache, block_tables, actual_seqs_cpu, output)
       
       return output
   ```

**输出物：**
- 完整的适配层代码

---

#### 步骤 9：修改模型调用逻辑

**目标：** 在模型代码中替换原始算子调用

**操作：**

1. **定位目标文件和函数**
   ```python
   # 目标文件：vllm/model_executor/models/glm4_moe.py
   # 目标函数：Glm4MoeDecoderLayer.forward()
   ```

2. **添加导入**
   ```python
   # 在文件顶部添加
   import glm_pto_kernels
   ```

3. **替换算子调用**

   **示例 1：Gate 算子**
   ```python
   # 原始代码
   router_logits = self.gate(hidden_states.to(dtype=torch.float32))
   
   # 替换后代码
   if glm_pto_kernels.USE_PTO_GATE:
       router_logits = glm_pto_kernels.gate(self.gate, hidden_states.to(dtype=torch.float32))
   else:
       router_logits = self.gate(hidden_states.to(dtype=torch.float32))
   ```

   **示例 2：Attention Pre 算子**
   ```python
   # 修改 Glm4MoeAttention.forward()
   def forward(
       self,
       layer,  # 增加 layer 输入
       positions: torch.Tensor,
       hidden_states: torch.Tensor,
       residual: torch.Tensor,  # 增加 residual 输入
   ) -> torch.Tensor:
       if glm_pto_kernels.USE_PTO_ATTENTION_PRE:
           # PTO 内核实现
           q, k, v, residual = glm_pto_kernels.attention_pre(
               hidden_states, residual, layer, self, positions
           )
       else:
           # 原始实现
           qkv, _ = self.qkv_proj(hidden_states)
           q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
           if self.use_qk_norm:
               q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q.shape)
               k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(k.shape)
           q, k = self.rotary_emb(positions, q, k)
       
       attn_output = self.attn(q, k, v)
       output, _ = self.o_proj(attn_output)
       return output, residual  # 增加 residual 输出
   
   
   # 修改 Glm4MoeDecoderLayer.forward()
   # 将 input_layernorm() 与 self_attn() 合并
   if glm_pto_kernels.USE_PTO_ATTENTION_PRE:
       # PTO 内核实现
       hidden_states, residual = self.self_attn(
           self, positions=positions, hidden_states=hidden_states, residual=residual
       )
   else:
       # 原始实现
       if residual is None:
           residual = hidden_states
           hidden_states = self.input_layernorm(hidden_states)
       else:
           hidden_states, residual = self.input_layernorm(hidden_states, residual)
       hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
   ```

   **示例 3：Paged Attention 算子**
   ```python
   # 目标文件：vllm_ascend/attention/attention_v1.py
   # 目标函数：_forward_decode_only()
   
   # 原始代码
   torch_npu._npu_paged_attention(
       query=query,
       key_cache=self.key_cache,
       value_cache=self.value_cache,
       num_kv_heads=self.num_kv_heads,
       num_heads=self.num_heads,
       scale_value=self.scale,
       block_table=attn_metadata.block_tables,
       context_lens=attn_metadata.seq_lens,
       out=output
   )
   
   # 替换后代码
   if glm_pto_kernels.USE_PTO_FA:
       # PTO 内核实现
       glm_pto_kernels.paged_attention(
           query, self.key_cache, self.value_cache,
           attn_metadata.block_tables, attn_metadata.seq_lens, output
       )
   else:
       # 原始实现
       torch_npu._npu_paged_attention(
           query=query,
           key_cache=self.key_cache,
           value_cache=self.value_cache,
           num_kv_heads=self.num_kv_heads,
           num_heads=self.num_heads,
           scale_value=self.scale,
           block_table=attn_metadata.block_tables,
           context_lens=attn_metadata.seq_lens,
           out=output
       )
   ```

4. **保存修改**
   ```bash
   # 保存所有修改的文件
   git add .
   git commit -m "feat: integrate PyPTO fused operators"
   ```

**输出物：**
- 修改后的模型代码
- Git commit 记录

**注意事项：**
- 严格匹配目标文件和目标函数
- 注意代码缩进对齐
- 使用开关变量保证可回滚

---

### 阶段五：整网验证与调优

#### 步骤 10：端到端精度验证

**目标：** 验证融合算子在整网中的精度正确性

**操作：**

1. **运行整网推理**
   ```bash
   # 运行完整的模型推理
   python examples/run_glm4.py \
       --model-path /path/to/model \
       --input "你好，请介绍一下你自己。" \
       --output golden_output.txt
   ```

2. **对比原始输出**
   ```bash
   # 保存原始模型输出
   python examples/run_glm4.py \
       --model-path /path/to/model \
       --input "你好，请介绍一下你自己。" \
       --output original_output.txt \
       --use-original-impl
   
   # 对比输出
   diff original_output.txt golden_output.txt
   ```

3. **中间激活值检查**
   ```python
   # 在关键位置打印中间结果
   def forward_with_debug(self, ...):
       # 打印输入
       print(f"Input: shape={hidden_states.shape}, mean={hidden_states.mean()}, std={hidden_states.std()}")
       
       # 调用 PyPTO 算子
       q, k, v, residual = glm_pto_kernels.attention_pre(...)
       
       # 打印输出
       print(f"Q: shape={q.shape}, mean={q.mean()}, std={q.std()}")
       print(f"K: shape={k.shape}, mean={k.mean()}, std={k.std()}")
       print(f"V: shape={v.shape}, mean={v.mean()}, std={v.std()}")
       print(f"Residual: shape={residual.shape}, mean={residual.mean()}, std={residual.std()}")
       
       # 检查 NaN/Inf
       assert not torch.isnan(q).any(), "Q contains NaN"
       assert not torch.isinf(q).any(), "Q contains Inf"
       # ... 检查其他输出
   ```

4. **端到端精度测试**
   ```python
   # test_e2e_precision.py
   import torch
   import glm_pto_kernels
   
   def test_e2e_precision():
       # 构造测试输入
       test_input = "测试输入文本"
       
       # 原始实现输出
       original_output = run_model_with_original_ops(test_input)
       
       # PyPTO 实现输出
       pto_output = run_model_with_pto_ops(test_input)
       
       # 对比 logits
       logits_diff = torch.abs(original_output.logits - pto_output.logits)
       max_diff = torch.max(logits_diff).item()
       mean_diff = torch.mean(logits_diff).item()
       
       print(f"Logits max diff: {max_diff:.6e}")
       print(f"Logits mean diff: {mean_diff:.6e}")
       
       # 判断精度是否可接受
       if max_diff < 1e-2:  # 端到端精度允许更大误差
           print("✅ E2E precision test passed!")
       else:
           print("❌ E2E precision test failed!")
           # 进一步定位问题
           debug_precision_diff(original_output, pto_output)
   
   if __name__ == "__main__":
       test_e2e_precision()
   ```

**验证检查点：**
- ✅ 模型能正常启动和运行
- ✅ 无 NaN/Inf 等异常值
- ✅ 中间激活值在合理范围内
- ✅ 端到端输出与原始实现对齐（logits diff < 1e-2）
- ✅ 生成文本质量无明显下降

**推荐 Skill：** `pypto-precision-compare`、`pypto-precision-debugger`

---

#### 步骤 11：性能分析与调优

**目标：** 分析融合算子性能并进行优化

**操作：**

1. **采集性能数据**
   ```bash
   # 运行性能分析
   python examples/run_glm4.py \
       --model-path /path/to/model \
       --benchmark \
       --profile \
       --profile-output profile_data/
   ```

2. **分析性能数据**
   ```bash
   # 使用 PyPTO 性能分析工具
   python -m pypto.tools.profile_analyzer \
       --profile-dir profile_data/ \
       --output performance_report.html
   ```

3. **查看性能报告**
   - 打开 `performance_report.html`
   - 查看 AI Core 利用率
   - 查看内存带宽利用率
   - 查看算子执行时间分布

4. **识别性能瓶颈**
   ```python
   # 常见性能问题
   - AI Core 利用率低 → 可能 Tiling 策略不佳
   - 内存带宽饱和 → 可能数据访问模式不佳
   - 算子执行时间长 → 可能计算逻辑可优化
   ```

5. **性能调优**
   - **Tiling 策略优化**：调整 TileShape
   - **内存访问优化**：优化数据布局
   - **计算逻辑优化**：减少冗余计算

**输出物：**
- 性能分析报告
- 优化后的性能数据

**推荐 Skill：** `pypto-operator-auto-tuner`、`tune-frontend`、`tune-incore`、`tune-swimlane`

---

#### 步骤 12：问题排查与修复

**目标：** 排查集成过程中的问题

**常见问题排查：**

**问题 1：运行时报错 - Shape Mismatch**
```python
# 错误信息
RuntimeError: shape '[2, 4096]' is invalid for input of size 8192

# 排查方法
# 1. 检查输入 tensor 的实际 shape
print(f"hidden_states.shape: {hidden_states.shape}")
print(f"expected: [bs, hidden_dim]")

# 2. 检查中间变量的 shape
# 3. 检查权重 tensor 的 shape
# 4. 检查输出 tensor 的分配
```

**问题 2：精度异常 - NaN/Inf**
```python
# 错误信息
RuntimeError: NaN detected in output

# 排查方法
# 1. 检查输入数据是否正常
assert not torch.isnan(hidden_states).any(), "Input contains NaN"
assert not torch.isinf(hidden_states).any(), "Input contains Inf"

# 2. 检查权重是否正常
assert not torch.isnan(weight).any(), "Weight contains NaN"

# 3. 逐层检查中间结果
# 4. 检查 scale/offset 是否合理
```

**问题 3：性能不达预期**
```python
# 现象：性能比原始实现还慢

# 排查方法
# 1. 检查 Tiling 策略是否合理
# 2. 检查是否有不必要的内存拷贝
# 3. 检查是否充分利用了 AI Core
# 4. 使用 PyPTO 性能分析工具定位瓶颈
```

**问题 4：端到端精度差异大**
```python
# 现象：logits diff > 1e-2

# 排查方法
# 1. 检查 Golden 验证是否通过
# 2. 逐层对比中间结果
# 3. 检查是否有数值精度问题（fp16 累加误差）
# 4. 检查是否有 dtype 转换问题
```

**推荐 Skill：**
- `pypto-aicore-error-locator`（定位 aicore error）
- `pypto-host-stacktrace-analyzer`（分析堆栈信息）
- `pypto-precision-debugger`（排查精度问题）

---

### 阶段六：提交与文档

#### 步骤 13：创建 Issue 跟踪

**目标：** 创建 GitCode Issue 记录本次工作

**操作：**

```bash
# 使用 pypto-issue-creator skill
# 提供上下文信息，自动生成 Issue
```

**Issue 内容模板：**
```markdown
## 背景
将 GLM-4.5 模型的 attention_pre 算子融合为 PyPTO 实现，替代原有的 layernorm + quantization + matmul + dequantization + normalize + rotary_emb 组合。

## 工作内容
- [x] 分析原始小算子组合
- [x] 编写 Golden 参考实现
- [x] 验证理解正确性
- [x] 设计 PyPTO 算子实现方案
- [x] 实现 PyPTO 算子
- [x] 集成到整网
- [x] 精度验证
- [x] 性能测试

## 性能数据
- 原始实现延迟：XX ms
- PyPTO 实现延迟：XX ms
- 性能提升：XX%

## 精度数据
- Logits max diff: XX
- Logits mean diff: XX
- 端到端验证：✅ 通过

## 相关文件
- models/glm_v4_5/glm_attention_pre_quant.py
- models/glm_v4_5/glm_attention_pre_quant_golden.py
- glm_pto_kernels/__init__.py
```

**推荐 Skill：** `pypto-issue-creator`

---

#### 步骤 14：提交 PR

**目标：** 将修改提交到 GitCode 仓库

**操作：**

1. **检查修改**
   ```bash
   git status
   git diff
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/glm-attention-pre-quant
   ```

3. **提交修改**
   ```bash
   git add .
   git commit -m "feat(glm): integrate PyPTO fused attention_pre_quant operator

- Add PyPTO implementation of attention_pre_quant fused operator
- Add golden reference for verification
- Add adapter layer for model integration
- Update Glm4MoeDecoderLayer to use fused operator
- Verified precision and performance

Performance: XX% improvement
Precision: max_diff < 1e-3
"
   ```

4. **推送到远程**
   ```bash
   git push origin feature/glm-attention-pre-quant
   ```

5. **创建 PR**
   ```bash
   # 使用 pypto-pr-creator skill
   # 自动创建符合规范的 PR
   ```

**推荐 Skill：** `pypto-pr-creator`

---

## 常见问题 FAQ

### Q1: Golden 验证失败怎么办？

**A:** 按以下步骤排查：

1. **逐层对比中间结果**
   ```python
   # 在 Golden 实现中打印每层的输出
   def attention_pre_quant_golden_debug(...):
       normalized = F.layer_norm(...)
       print(f"LayerNorm output: mean={normalized.mean()}, std={normalized.std()}")
       
       quant_input = ...
       print(f"Quantization output: min={quant_input.min()}, max={quant_input.max()}")
       
       # ... 继续每一步
   ```

2. **对比原始实现的中间结果**
   - 在原始模型代码中也打印相同位置的中间结果
   - 逐层对比差异

3. **检查常见错误**
   - Shape 不匹配
   - Dtype 转换错误
   - 参数传递顺序错误
   - 隐藏的 reshape/transpose 操作

4. **使用二分法定位**
   - 注释掉后半部分计算
   - 只对比前半部分
   - 逐步缩小范围

### Q2: 端到端精度差异大怎么办？

**A:** 可能原因和解决方法：

1. **Golden 验证不充分**
   - Golden 只在单个算子层面验证
   - 需要在整网中验证 Golden 是否正确

2. **数值精度问题**
   - FP16 累加误差
   - 需要在关键位置使用 FP32 累加

3. **隐藏的 dtype 转换**
   - 检查是否有隐式的 float16 → float32 转换
   - 确保所有 dtype 转换都是显式的

4. **内存布局问题**
   - 检查 tensor 的 stride 和 contiguous 性
   - 某些操作可能依赖特定的内存布局

### Q3: 性能不如预期怎么办？

**A:** 性能调优策略：

1. **分析性能瓶颈**
   - 使用 `pypto-operator-auto-tuner`
   - 查看 AI Core 利用率、内存带宽

2. **优化 Tiling 策略**
   - 调整 TileShape
   - 优化数据切分方式

3. **优化内存访问**
   - 减少 memory copy
   - 优化 tensor layout

4. **使用调优 Skill**
   - `tune-frontend`：前端代码级优化
   - `tune-incore`：核内指令级优化
   - `tune-swimlane`：深度性能调优

### Q4: 如何回滚到原始实现？

**A:** 使用开关变量：

```python
# 在 glm_pto_kernels/__init__.py 中
USE_PTO_ATTENTION_PRE = False  # 改为 False

# 模型代码中的逻辑
if glm_pto_kernels.USE_PTO_ATTENTION_PRE:
    # PyPTO 实现
    q, k, v, residual = glm_pto_kernels.attention_pre(...)
else:
    # 原始实现
    qkv, _ = self.qkv_proj(hidden_states)
    # ...
```

---

## 最佳实践总结

### 1. 理解验证是关键

- ⚠️ 不要跳过 Golden 验证环节
- ⚠️ Golden 验证通过后再开发 PyPTO 算子
- ⚠️ Golden 脚本后续用于精度对比

### 2. 适配层设计原则

- ✅ 保持原始代码结构不变
- ✅ 使用开关变量灵活切换
- ✅ 适配层只负责参数转换和调用桥接
- ✅ 不在适配层中添加业务逻辑

### 3. 算子替换位置精准匹配

- ✅ 严格按照文档标注的目标文件和目标函数
- ✅ 注意代码缩进对齐
- ✅ 修改后立即测试

### 4. 验证流程分层进行

- ✅ 单算子验证 → Golden 对比
- ✅ 层级验证 → 中间激活值检查
- ✅ 端到端验证 → 最终输出对比

### 5. 性能优化循序渐进

- ✅ 先保证正确性
- ✅ 再优化性能
- ✅ 使用专业工具分析瓶颈
- ✅ 避免盲目优化

---

## 相关资源

### 官方文档
- PyPTO 官方文档：`docs/`
- 算子开发指南：`docs/operator_development.md`
- 性能调优指南：`docs/performance_tuning.md`

### 示例代码
- GLM-4.5 融合算子示例：`models/glm_v4_5/`
- 其他模型示例：`models/`

### 调试工具
- 精度对比工具：`pypto.tools.precision_compare`
- 性能分析工具：`pypto.tools.profile_analyzer`
- 堆栈分析工具：`pypto.tools.stacktrace_analyzer`

### 相关 Skill
- `pypto-golden-generator`：生成 Golden 参考实现
- `pypto-op-design`：算子设计方案
- `pypto-op-develop`：算子实现
- `pypto-precision-compare`：精度对比
- `pypto-precision-debugger`：精度调试
- `pypto-operator-auto-tuner`：性能调优
- `pypto-aicore-error-locator`：定位 aicore error
- `pypto-host-stacktrace-analyzer`：分析堆栈信息

---

## 附录：完整示例

### GLM-4.5 Attention Pre Quant 算子集成示例

**完整文件列表：**
```
models/glm_v4_5/
├── glm_attention_pre_quant_golden.py    # Golden 参考实现
├── glm_attention_pre_quant.py           # PyPTO 实现
├── test_attention_pre_quant.py          # 单算子测试
└── benchmark_attention_pre_quant.py     # 性能测试

glm_pto_kernels/
├── __init__.py                          # 适配层
└── glm_attention_pre_quant.py           # PyPTO 算子副本

vllm/model_executor/models/
└── glm4_moe.py                          # 修改后的模型文件
```

**完整工作流程：**
```bash
# 1. 环境验证
npu-smi info
python -c "import pypto; import torch_npu"

# 2. 编写 Golden 实现
vim models/glm_v4_5/glm_attention_pre_quant_golden.py

# 3. 验证 Golden 正确性
python models/glm_v4_5/test_attention_pre_quant_golden.py

# 4. Golden 集成到整网
vim glm_pto_kernels/__init__.py
vim vllm/model_executor/models/glm4_moe.py
python examples/run_glm4.py --test

# 5. 开发 PyPTO 实现
vim models/glm_v4_5/glm_attention_pre_quant.py
python models/glm_v4_5/test_attention_pre_quant.py

# 6. 替换到整网
vim glm_pto_kernels/__init__.py
python examples/run_glm4.py --test

# 7. 性能测试
python examples/run_glm4.py --benchmark

# 8. 提交代码
git add .
git commit -m "feat(glm): integrate PyPTO attention_pre_quant"
git push origin feature/glm-attention-pre-quant
```

---

## 结语

融合算子替换整网小算子是一个系统工程，需要严格遵循"理解验证 → 开发实现 → 集成验证"的流程。Golden 验证环节是确保成功的关键，切勿跳过。

遵循本 Skill 的指导，可以系统化地完成融合算子集成工作，避免常见陷阱，提高开发效率和成功率。

---

**Skill 版本：** v1.0
**最后更新：** 2026-04-10
**维护者：** PyPTO Team