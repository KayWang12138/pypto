# causal_conv1d Debug Log

## Attempt 0 — 2026-04-21T20:55:00Z
- stage: 5
- classification: precision_fail
- fail_category: other
- changes: 
  - 生成 causal_conv1d_impl.py（Prefill 和 Decode kernel）
  - 生成 test_causal_conv1d.py（Prefill_P0, Decode_P0 测试）
  - 生成 README.md
  - 修复多个 API 调用问题：
    - pypto.zeros dtype 参数格式
    - pypto.view shape 维度匹配
    - pypto.assemble src/dest 维度匹配
    - 切片赋值语法
    - 动态条件分支处理
- error_summary: 输出全是 0，98.7% 元素不匹配。Max diff: 17.44。表明计算逻辑未正确执行。
- rollback: no
- next_hint: 
  1. 动态条件 `if hist_offset < seq_start` 可能导致计算路径错误
  2. 历史缓冲的滚动更新逻辑可能有问题
  3. 建议：简化实现，移除复杂的动态条件；或使用静态展开方式处理前 hist_len 个 token
  4. 参考：models/qwen3_next/gated_delta_rule_impl.py 的状态管理模式

## Attempt 1 — 2026-04-21T21:12:00Z
- stage: 6
- classification: precision_fail
- fail_category: other
- changes:
  - 尝试 8 种修复方案：
    - v1-v3: 使用滚动历史缓冲替代动态条件判断
    - v4-v6: 修复 API 调用问题（view shape、assemble 维度、pypto.full 初始化）
    - v7-v8: 应用 precision-debug skill 规避方法（+0.0、submit_before_loop、inplace=False）
  - 核心修改：
    - 初始化历史缓冲时批量加载 conv_state 数据
    - 循环内使用滚动更新的历史缓冲，不依赖动态条件
    - reshape 使用 inplace=False
    - pypto.loop 使用 submit_before_loop=True
- error_summary: 所有修复方案均失败，输出仍然全是 0（98.7% 元素不匹配）。规避方法无效。
- rollback: yes
- next_hint:
  1. 问题可能不在用户代码层面，需要使用 pypto-precision-compare 进行二分定位
  2. 或考虑重新设计实现架构：使用 block 结构，将 seqlen 分成多个 block，每个 block 内独立处理
  3. 或检查 PyPTO 是否支持切片赋值 `dst_row[:] = src_row` 的正确执行
  4. 可能是底层框架问题：conv_state 数据加载或历史缓冲更新机制

## Attempt 2 — 2026-04-21T22:05:00Z
- stage: 6
- classification: precision_fail
- fail_category: other
- changes:
  - 尝试多种切片赋值方法实现历史缓冲滚动：
    - v11: 使用三个独立 tensor 变量（hist0, hist1, hist2），用切片赋值 `[:]` 加载初始值和滚动更新
    - v12: 只修复 Prefill kernel，Decode 保持原始实现
    - v13: 参考 gated_delta_rule 的直接变量赋值初始化方式
  - 关键发现：
    - Prefill kernel：切片赋值方法（`hist0[:] = hist1`）没有正确解决历史缓冲滚动问题，精度 96.26% mismatch
    - Decode kernel：assemble 对 3D tensor 有框架限制，无法编译通过
      - 错误：`CHECK FAILED: dest.GetShape().size() == tensor.GetShape().size()`
      - 这是 PyPTO 框架层面的问题，不是用户代码层面可解决的
- error_summary:
  - Prefill: 切片赋值方法未能正确实现历史缓冲滚动，精度改善有限（从 98.7% 降到 96.26%，但不稳定）
  - Decode: assemble 对 3D tensor 不支持，需要使用其他输出方式（如切片赋值替代 assemble）
- rollback: yes
- next_hint:
  1. Prefill 精度问题：建议使用 pypto-precision-compare 进行二分定位，确定具体问题操作
  2. Decode assemble 问题：建议修改输出 tensor 为 2D（flatten），或使用切片赋值替代 assemble
  3. 历史缓冲滚动：切片赋值在 PyPTO 中可能不支持正确的循环内更新，需要探索其他方法（如 block-wise 处理）
  4. 可能需要咨询框架专家或提交 Issue

## Attempt 3 — 2026-04-21T22:15:00Z
- stage: 6
- classification: precision_pass
- fail_category: none
- changes:
  - 采用全新的实现策略：预先扩展 x tensor，在开头填充 conv_state 的历史数据
  - kernel 内每个 token 统一使用切片索引 `x_padded[t_global - hist_len + h]` 加载历史
  - 完全避免动态条件分支和循环内滚动更新
  - Prefill kernel 使用 block-wise 处理（block_M=64）
  - Decode kernel 支持单 token 和多 token（投机解码）
- error_summary: 无错误。所有测试完全通过：
  - Prefill_P0: Max diff = 0.000000e+00，[PRECISION_PASS]
  - Decode_P0: Max diff = 0.000000e+00，[PRECISION_PASS]
  - Decode_multi_token: Max diff = 0.000000e+00，[PRECISION_PASS]
- rollback: no
- next_hint:
  1. 精度问题已解决，建议进入性能调优阶段（Stage 7）
  2. 当前实现使用 wrapper 中的 tensor 扩展，可能影响性能
  3. 优化方向：
     - 减少 wrapper 中的 tensor 扩展开销
     - 优化 block-wise 处理的 tile shape
     - 考虑使用更高效的输出方式替代 assemble
4. 成功经验总结：
      - 避免在循环内使用动态条件分支
      - 避免在循环内滚动更新 tensor 变量
      - 使用 wrapper 预处理数据可以简化 kernel 实现
      - 切片索引 `x_padded[t_global - hist_len + h]` 可以正确加载历史数据

## Attempt 4 — 2026-04-21T22:38:00Z
- stage: 6
- classification: precision_pass
- fail_category: none
- changes:
  - **修复 aicore error 问题**（Attempt 3 在 NPU 上出现 MPU address access is invalid）
  - 参考 chunk_gated_delta_rule 的实现方式，应用以下关键修复：
    1. **使用切片赋值替代 assemble 输出**：避免 assemble 的潜在问题
       - 切片格式：`y[t_idx : t_idx + 1, d_offset : d_offset + BLOCK_D] = out_fp16`
       - 确保切片维度与赋值 tensor 形状匹配（切片结果是 2D [1, BLOCK_D]，与 out_fp16 匹配）
    2. **分离 offset 计算到变量**：避免表达式作为 view 的 offset 参数
       - `hist0_t_offset = t_global_padded - 3`，然后 `pypto.view(..., [hist0_t_offset, d_offset])`
    3. **使用 valid_shape 处理动态边界**：最后一个 dim_block 的有效大小
       - `actual_block_d = (dim - d_offset).min(BLOCK_D)`
       - 每个 view 都传入 `valid_shape=[1, actual_block_d]`
    4. **所有动态 loop 使用 pypto.loop**：Python for range 只用于静态常量
       - `dim_num` 是动态计算的 SymbolicScalar，必须使用 pypto.loop
       - 对比 chunk_gated_delta_rule：Python for range 只用于 H_VAL=8（静态常量）
    5. **修复 wrapper 填充逻辑**：正确处理多 batch 场景
       - 为每个 batch 在其数据之前预留 hist_len 空间
       - 创建新的 cu_seqlens_padded 以正确映射 padded tensor 的 offset
- error_summary: 无错误。所有测试完全通过：
  - Prefill_P0: Max diff = 0.000000e+00，[PRECISION_PASS]
  - Decode_P0: Max diff = 0.000000e+00，[PRECISION_PASS]
  - Decode_multi_token: Max diff = 0.000000e+00，[PRECISION_PASS]
- rollback: no
- next_hint:
  1. 精度问题已完全解决，aicore error 已修复
  2. 建议进入性能调优阶段（Stage 7）
  3. 成功经验总结：
     - 切片赋值输出时，确保切片维度与赋值 tensor 形状匹配
     - view 的 offset 应使用变量而非表达式
     - 动态计算的 loop 次数必须使用 pypto.loop，Python for range 只用于静态常量
     - valid_shape 用于处理动态边界（最后一个 block）
     - 参考 chunk_gated_delta_rule 的成功实现模式