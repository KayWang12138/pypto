[Bug-Report|缺陷反馈]: submit_before_loop=True on outer loop does not guarantee inner parallel loop completion before next iteration

## 问题描述 / Problem Description

When implementing FlashAttentionScoreGrad in a single-pass loop structure, using `submit_before_loop=True` on the outer S1 loop does not guarantee that all inner S2 parallel tasks have completed before the next S1 iteration starts. This causes incorrect results when using the "double-pointer" pattern (dk_in/dk_out pointing to the same device memory) for cross-iteration accumulation.

## 环境信息 / Environment

| Item | Value |
|------|-------|
| PyPTO Commit | e8c1db49 (2026-03-31) |
| CANN | 8.5.0 |
| NPU | Ascend910C (A3) |
| Python | 3.10.20 |
| torch | 2.7.1 |
| torch_npu | 2.7.1.post2 |

## 重现步骤 / Steps to Reproduce

1. Create a kernel with outer S1 loop (`submit_before_loop=True`) and inner S2 loop (parallel, with `unroll_list=[8,4,2,1]`)
2. Pass `dk_in` and `dk_out` as separate kernel parameters, but point them to the same device memory on host side
3. In the inner S2 loop body: `view(dk_in, offset) → add(contribution) → assemble(dk_out, offset)`
4. Run with S=256 (2 S tiles), so there are 2 outer S1 iterations

Minimal reproduction pattern:
```python
@pypto.frontend.jit
def kernel(dk_in: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
           dk_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16), ...):
    for s1_idx in pypto.loop(s_loop, submit_before_loop=True):
        for s2_idx in pypto.loop(s_loop, unroll_list=[8, 4, 2, 1]):
            # Each s2 block writes to different s2_off
            dk_prev = pypto.view(dk_in, [TILE, D], [s2_off, 0])
            dk_new = pypto.add(dk_prev, contribution)
            pypto.assemble(dk_new, [s2_off, 0], dk_out)

# Host side: dk_in and dk_out point to same memory
dk = torch.zeros(...)
kernel(dk, dk, ...)  # same tensor passed twice
```

## 预期结果 / Expected Behavior

`submit_before_loop=True` on the outer loop should ensure ALL tasks (including inner parallel tasks) from S1 iteration i complete before S1 iteration i+1 begins. The `view(dk_in)` in iteration i+1 should see the values written by `assemble(dk_out)` in iteration i.

## 实际结果 / Actual Behavior

- S=128 (1 tile, no cross-iteration dependency): **PASS** ✓
- S=256 (2 tiles, cross-iteration dependency): **FAIL** ✗ (dK max diff: 0.965)
- Adding `submit_before_loop=True` to inner loop: **PASS** ✓ but 3x slower (serializes everything)

The inner S2 parallel tasks from iteration i may still be in-flight when iteration i+1's inner tasks start reading dk_in.

## 分析 / Analysis

Tested with precision-debugger skill:
- `unroll_list=[1]` on inner loop: still fails → not an unroll issue
- `+0.0` trick: still fails → not a compiler optimization issue
- `submit_before_loop=True` on BOTH loops: passes but 3x slower → confirms root cause is insufficient synchronization

`submit_before_loop=True` appears to guarantee the **submission order** of loop body tasks, but not the **completion** of inner parallel tasks before the next outer iteration.

## 影响 / Impact

This prevents implementing efficient single-pass gradient operators (like FlashAttentionScoreGrad) that need cross-dimension accumulation. The workaround requires either:
1. Two-pass computation (P and dS computed twice), or
2. Both loops serialized (3x performance loss)

A full synchronization barrier at outer loop boundaries would enable single-pass implementations.
