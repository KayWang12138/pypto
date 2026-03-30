# 断裂点识别报告

## 1. 摘要

- **会话时间**: 2026-03-29
- **断裂点总数**: 1 个
- **按优先级分布**: 致命: 1 个

## 2. 断裂点详情

### [FP-001] A2-API 行为异常: pypto.matmul

- **类型**: A2-API 行为异常
- **优先级**: 致命
- **置信度**: high

#### 问题描述

batch_matmul 实现调用 `pypto.matmul()` 时触发 aicore 异常 (error code: 507015)。

#### 证据片段

```
[ERROR] Exception Type: exception aicore error
retcode: 507015
The MPU address access is invalid
```

#### 优化建议

1. 参考已验证的 matmul 实现
2. 检查动态轴声明方式
3. 验证 tiling 配置
