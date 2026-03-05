# PFA 算子性能优化设计文档

**日期：** 2026-02-28  
**目标：** 优化 `glm_attention_ifa_pfa.py` 中 `pfa_func` 函数性能，目标提升 50%+  
**当前性能：** 996us  
**策略：** 方案1+2并行实施

---

## 1. 优化方案概览

### 1.1 方案1：稳健优化版（V1）

**预期提升：** 40-50%  
**风险等级：** 低  
**实施周期：** 1-2小时

#### 核心优化点

| 优化项 | 原配置 | 优化配置 | 预期收益 |
|--------|--------|----------|----------|
| s2_tile | 128 | 256 | 10-20% |
| c1_tile K轴 | [128, 128] | [64, 256] | 5-10% |
| v1_tile | [128, 128] | [128, 256] | 5-10% |
| cube_l1_reuse_setting | {0: 4} | 动态计算 | 5-15% |
| 并行循环 | range() | pypto.loop() | 10-30% |
| L2调度 | 默认 | L2亲和调度 | 5-10% |
| stitch_function内存 | 1024 | 4096 | 5-10% |

#### 关键实现细节

**1. Tile配置调整**
```python
def get_pfa_config_opt_v1(device="cpu"):
    """PFA 稳健优化版配置"""
    b = 8
    s1 = 128
    s2 = s1
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s1 + block_size - 1) // block_size)

    actual_seq_values = [s1] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(
        b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale,
        kv_layout=kv_layout, q_d=q_d, kv_d=q_d, block_size=block_size,
        block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
        actual_seq=actual_seq_tensor
    )
    atten_cfg.max_num_blocks_per_query = (s1 + block_size - 1) // block_size
    
    # 优化后的 Tile 配置
    cube_tile = 128
    m_tile = 128
    s2_tile = 256  # 增大 s2_tile
    
    tile_cfg = AttentionTileConfig(
        g_tile=nq,
        s2_tile=s2_tile,
        # QK^T: [g_tile, d] @ [s2_tile, d]^T = [g_tile, s2_tile]
        # M维度=L0:128, L1:128, K维度=L0:64, L1:256, N维度=L0:128, L1:128
        c1_tile_shape=[[128, 128], [64, 256], [128, 128]],  # 优化 K 轴
        v1_tile_shape=[128, s2_tile],  # [128, 256]
        # PV: [g_tile, s2_tile] @ [s2_tile, d] = [g_tile, d]
        c2_tile_shape=[[128, 128], [64, 256], [128, 128]],
        v2_tile_shape=[128, 128]
    )
    
    return atten_cfg, tile_cfg
```

**2. JIT配置动态化**
```python
def get_pfa_jit_config_v1(s1, s2_tile):
    """根据实际参数动态计算 JIT 配置"""
    # s2_loop = sum((i + s2_tile) // s2_tile for i in range(s1))
    # 对于因果注意力，平均 s2_loop 约为 s1 / (2 * s2_tile)
    avg_s2_loop = (s1 + s2_tile - 1) // (2 * s2_tile)
    
    return {
        "runtime_options": {
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 4096,  # 增大
            "stitch_function_inner_memory": 4096,    # 增大
            "device_sched_mode": 1  # L2 亲和调度
        },
        "pass_options": {
            "pg_upper_bound": 2048,  # 增大
            "cube_l1_reuse_setting": {0: max(2, avg_s2_loop)},
            "cube_l1_reuse_mode": 1  # 开启全局 L1 复用
        },
        "debug_options": {
            "runtime_debug_mode": 1,
            "compile_debug_mode": 0
        }
    }
```

**3. 并行循环优化**
```python
# 原代码使用静态 range() 循环
for n2_idx in range(n2_sym):    # KV Head (nkv=1)
    for g_idx in range(g_loop):  # Query Head 组 (12/1=12)

# 优化后：使用 pypto.loop 实现多核并行
for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
    for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
        # 核心计算...
```

---

### 1.2 方案2：激进优化版（V2）

**预期提升：** 50-70%  
**风险等级：** 中高  
**实施周期：** 2-3小时

#### 包含方案1的所有优化，并增加：

| 额外优化项 | 说明 | 预期收益 |
|-----------|------|----------|
| KV组装预取 | 预取下一轮block索引 | 3-5% |
| Online Softmax融合 | 减少scope切换 | 5-15% |
| 极限内存配置 | stitch_function内存增大到8192 | 5-10% |
| 优化循环展开 | unroll_list=[16, 8, 4, 2, 1] | 5-10% |

#### 关键实现细节

**1. KV组装优化 - 预取机制**
```python
for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", 
                         unroll_list=[16, 8, 4, 2, 1]):
    block_num = s2_tile // block_size
    idx = s2_idx * block_num
    
    # K组装
    kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
    
    # 预取下一轮 block 索引（减少循环内的间接访存延迟）
    next_idx = s2_idx + 1
    if next_idx < s2_loop:
        # 可以在这里添加预取逻辑
        pass
```

**2. Online Softmax融合优化**
```python
# 优化：减少 scope 切换，增加操作融合
if pypto.is_loop_begin(s2_idx):
    # 融合：scale + amax + sub + exp + sum
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij
    
    # V组装和matmul
    vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
    vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], 
                              valid_shape=[actual_s2_tile, dn])
    
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
    
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    oi_update[:] = oi_tmp
else:
    pypto.set_pass_options(sg_set_scope=1)
    # 融合计算
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    max_new = pypto.maximum(max_update, tilda_mij)
    tsub = pypto.sub(sij_scale, max_new)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    pypto.set_pass_options(sg_set_scope=-1)
    
    pypto.set_pass_options(sg_set_scope=2)
    tsub2 = pypto.sub(max_update, max_new)
    max_update[:] = max_new
    update_mul = pypto.exp(tsub2)
    sum_update[:] = sum_update * update_mul + sum_local
    pypto.set_pass_options(sg_set_scope=-1)
    
    # V组装和更新
    vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
    vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], 
                              valid_shape=[actual_s2_tile, dn])
    
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
    
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    oi_update[:] = oi_update * update_mul + oi_tmp
```

**3. 极限内存配置**
```python
def get_pfa_jit_config_v2(s1, s2_tile):
    """激进优化版 JIT 配置"""
    avg_s2_loop = (s1 + s2_tile - 1) // (2 * s2_tile)
    
    return {
        "runtime_options": {
            "stitch_function_num_initial": 256,      # 进一步增大
            "stitch_function_outcast_memory": 8192,  # 极限配置
            "stitch_function_inner_memory": 8192,    # 极限配置
            "device_sched_mode": 1
        },
        "pass_options": {
            "pg_upper_bound": 3072,  # 极限配置
            "cube_l1_reuse_setting": {0: max(4, avg_s2_loop)},
            "cube_l1_reuse_mode": 1
        },
        "debug_options": {
            "runtime_debug_mode": 1,
            "compile_debug_mode": 0
        }
    }
```

---

## 2. 实施计划

### 2.1 阶段1：基础优化（1-2小时）

**任务清单：**
- [ ] 创建 `glm_attention_ifa_pfa_opt_v1.py` 文件
- [ ] 实现 Tile 配置优化
- [ ] 实现 JIT 配置动态化
- [ ] 实现并行循环优化
- [ ] 运行基础测试验证功能正确性
- [ ] 采集性能数据，对比基线

### 2.2 阶段2：激进优化（2-3小时）

**任务清单：**
- [ ] 创建 `glm_attention_ifa_pfa_opt_v2.py` 文件
- [ ] 实现 KV 组装预取优化
- [ ] 实现 Online Softmax 融合
- [ ] 调整极限内存配置
- [ ] 优化循环展开策略
- [ ] 运行完整测试验证功能和精度
- [ ] 采集性能数据，对比基线和V1版本

### 2.3 阶段3：性能对比（1小时）

**任务清单：**
- [ ] 采集三个版本（baseline、V1、V2）的性能数据
- [ ] 生成泳道图并对比分析
- [ ] 识别性能瓶颈变化
- [ ] 总结优化效果和经验

### 2.4 阶段4：文档完善（1-2小时）

**任务清单：**
- [ ] 编写优化记录文档
- [ ] 创建性能对比工具
- [ ] 完善参数配置指南
- [ ] 总结优化工作流程
- [ ] 记录易用性和文档改进点

---

## 3. 测试验证策略

### 3.1 功能验证

**测试用例：**
- 基础场景：b=8, s1=128, nq=12, nkv=1, d=128
- 精度要求：rtol=0.05, atol=0.005（与原测试一致）

**验证方法：**
```python
# 对比优化前后输出
assert_allclose(
    optimized_output.flatten().tolist(),
    baseline_output.flatten().tolist(),
    rtol=0.05, atol=0.005
)
```

### 3.2 性能验证

**性能指标：**
1. **总执行时间**：泳道图末尾时间戳
2. **GM访问量**：泳道图 MTE2 统计
3. **L2命中率**：性能计数器
4. **核利用率**：泳道图核分布
5. **子图融合率**：计算图节点数

**性能对比方法：**
```python
def benchmark_pfa(config_name, config_file):
    # 1. 运行测试
    run_time = measure_execution_time(config_file)
    
    # 2. 生成泳道图
    generate_swimlane(config_file)
    
    # 3. 采集性能指标
    metrics = collect_performance_metrics(config_file)
    
    # 4. 返回结果
    return {
        "config": config_name,
        "run_time": run_time,
        "metrics": metrics
    }
```

---

## 4. 风险评估

### 4.1 技术风险

| 风险项 | 风险等级 | 缓解措施 |
|--------|----------|----------|
| Tile配置调整导致精度损失 | 中 | 严格对比测试，确保误差在容忍范围内 |
| 内存配置过大导致OOM | 中 | 逐步增大配置，监控内存使用 |
| 并行循环导致竞争条件 | 低 | 使用pypto.loop保证正确性 |
| 极限配置不稳定 | 高 | V2版本单独测试，提供回退方案 |

### 4.2 回退策略

- **V1版本失败**：回退到baseline版本
- **V2版本失败**：回退到V1版本
- **性能未达预期**：保留最稳定的优化版本，继续迭代

---

## 5. 预期成果

### 5.1 代码文件

- `models/glm_v4_5/glm_attention_ifa_pfa_opt_v1.py` - 稳健优化版
- `models/glm_v4_5/glm_attention_ifa_pfa_opt_v2.py` - 激进优化版
- `models/glm_v4_5/test_pfa_performance.py` - 性能测试脚本

### 5.2 文档

- `docs/mylearning/pfa_optimization_record.md` - 优化记录文档
- `docs/mylearning/pfa_config_guide.md` - 参数配置指南
- `docs/mylearning/optimization_workflow.md` - 优化工作流程

### 5.3 工具

- `tools/compare_swimlane.py` - 性能对比工具

### 5.4 性能提升目标

- **V1版本**：预期提升 40-50%（约 500-600us）
- **V2版本**：预期提升 50-70%（约 300-500us）

---

## 6. 易用性和文档改进点

### 6.1 框架易用性

1. **参数配置简化**
   - 提供 `get_pfa_jit_config()` 等辅助函数
   - 自动计算最优配置参数

2. **错误提示改进**
   - 更清晰的Tile配置错误提示
   - 内存配置不足时的友好提示

3. **调试支持**
   - 提供中间结果查看接口
   - 性能瓶颈自动识别

### 6.2 文档完善

1. **优化指南完善**
   - 补充PFA算子特有的优化点
   - 添加因果注意力的优化注意事项
   - 提供更多实际案例

2. **最佳实践总结**
   - Tile配置选择决策树
   - JIT参数调优流程
   - 性能问题排查清单

3. **工作流程文档**
   - 标准化的性能优化流程
   - 验证检查清单
   - 常见问题解决方案

### 6.3 工具优化

1. **性能分析工具**
   - 自动生成性能对比报告
   - 可视化性能瓶颈
   - 优化建议生成

2. **配置生成工具**
   - 根据场景自动推荐配置
   - 配置参数在线验证

---

## 附录

### A. 参考文档

- `docs/mylearning/pfa_optimization_guide.md` - PFA 优化指南
- `docs/tutorials/debug/performance.md` - 性能调优指南
- `docs/tutorials/debug/matmul-performance-guide.md` - Matmul 高性能编程指导

### B. 性能基线

**当前性能：** 996us（b=8, s1=128, nq=12, nkv=1, d=128）

**性能分解：**
- QK^T 计算：约 30%
- Softmax 计算：约 20%
- PV 计算：约 30%
- KV 组装：约 10%
- 其他：约 10%

### C. 优化验证检查清单

- [ ] 功能正确性验证通过
- [ ] 精度测试通过（rtol=0.05, atol=0.005）
- [ ] 性能提升达到预期
- [ ] 无内存泄漏
- [ ] 无编译警告
- [ ] 代码符合规范
- [ ] 文档更新完整
