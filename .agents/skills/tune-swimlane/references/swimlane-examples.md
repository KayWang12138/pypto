## 性能优化建议库

### 建议 1：气泡率过高

**症状**：气泡率 > 10%

**可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**优化建议**：
1. **Stitch 调优（优先级高）**
   ```python
   @pypto.frontend.jit(
       runtime_options={"stitch_function_max_num": 128}
   )
   ```

2. **Loop Unroll（优先级高）**
   ```python
   for s2_idx in pypto.loop(s2_loop, unroll_list=[8, 4, 2, 1], name="LOOP_s2", idx_name="s2_idx"):
       # 计算逻辑
   ```

3. **L1Reuse 优化**

4. **调整任务粒度**
- 增大 loop 的 tile size
- 减少 loop 层级

5. **合图调优**


### 建议 2：核心利用率低

**症状**：核心利用率 < 50%

**可能原因：**
- 等待时间过长
- 任务调度不均衡
- 内存访问冲突

**优化建议**：

1. **L2 亲和调度**
   ```python
   @pypto.jit(runtime_options={"device_sched_mode": 1})
   ```

2. **调整 TileSize**
   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
   ```

3. **启用 CubeNBuffer 合并同构子图**
   ```python
   pypto.set_pass_options(cube_nbuffer_setting={-1: 4})
   ```

### 建议 3：核心负载不均衡

**症状**：AicoreTime 差异 > 20%

**可能原因：**
- 任务分配不均
- 任务执行时间差异大

**优化建议**：
1. **调整任务分配策略**
   - 使用更均匀的任务切分
   - 避免某些核心任务过多

2. **优化任务粒度**
   - 调整 tile size 使任务更均匀

3. **调整任务执行顺序**
   - 使用 sg_set_scope 合并子图
   ```python
   pypto.set_pass_options(sg_set_scope=1)
   # ... 操作 ...
   pypto.set_pass_options(sg_set_scope=-1)
   ```


## 调优流程

```
┌────────────────────────────────────────────────┐
│                深度性能调优流程                │
├────────────────────────────────────────────────┤
│                                                │
│  1. 采集泳道图数据                             │
│     └─ debug_options={"runtime_debug_mode": 1} │
│                                                │
│  2. 分析泳道图                                 │
│     ├─ 查看任务执行顺序                        │
│     ├─ 识别气泡（等待调度时间）                │
│     └─ 分析核心利用率                          │
│                                                │
│  3. 选择调优方向                               │
│     ├─ 气泡率高 → Stitch/Loop Unroll           │
│     ├─ 利用率低 → 调度策略/TileShape           │
│     └─ 负载不均 → 合图优化                     │
│                                                │
│  4. 应用优化                                   │
│     └─ 每次只修改一个参数                      │
│                                                │
│  5. 验证                                       │
│     ├─ 重新编译运行                            │
│     ├─ 检查精度                                │
│     └─ 对比性能数据                            │
│                                                │
│  6. 迭代直到达到目标性能                       │
│                                                │
└────────────────────────────────────────────────┘
```


## 常见问题

### Q1: 泳道图文件在哪里？

A: 泳道图文件在 `output/output_*/` 目录下，其中 `*` 是时间戳。

### Q2: 如何查看性能统计？

A: 使用 PyPTO Toolkit 打开 `merged_swimlane.json` 文件，然后点击 "查看性能报告" 按钮。

### Q3: 气泡是什么？

A: 气泡是指线程等待调度的时间，表示线程空闲的时间段。气泡率越低，说明调度效率越高。

### Q4: 控制开销占比过高怎么办？

A: 对于小数据量，控制开销占比高是正常现象。可以通过增加数据规模来降低控制开销占比。

### Q5: 如何选择合适的 Tilesize？

A:

* 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
* 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
* 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑


## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
- [GLM Attention 案例](../../../../models/glm_v4_5/glm_attention.py)
- [性能优化案例](../../../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
