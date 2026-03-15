# Final Report Template

orchestrator 在流程结束时（无论成功或阻塞）输出以下结构化摘要：

```markdown
## 开发结果
- 算子: {op}
- spec: custom/{op}/spec.md
- design: custom/{op}/design.md
- golden: custom/{op}/{op}_golden.py
- test_entry: custom/{op}/test_{op}.py
- kernel: custom/{op}/{op}_impl.py

## 精度结果
- 状态: PASS / FAIL
- 容差: rtol / atol
- 若失败: 当前定位路径
- 精度修复循环次数: N

## 性能结果
- output_dir: custom/{op}/output/output_xxx/
- core_utilization: ...
- bubble_rate: ...
- load_balance: ...
- max_work_time: ...

## 已知问题
- 环境限制 / 仅 sim 跑通 / NPU 未验证 / 数据缺失
```

## 使用说明

- `状态` 使用统一结束态：SUCCESS / BLOCKED_CONTRACT / BLOCKED_ACCURACY / BLOCKED_PERFORMANCE / BLOCKED_ENVIRONMENT
- 性能结果仅在精度 PASS 且完成 Stage 5-6 后填充
- 已知问题如实列出，不掩盖未验证项
