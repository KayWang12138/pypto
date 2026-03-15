# Overwrite Policy

## 需要覆盖确认的工件

以下工件在覆盖前必须向用户确认：

| 工件 | Owner | 原因 |
|------|-------|------|
| `spec.md` | `pypto-intent-understanding` | 用户需求文档，覆盖可能丢失用户确认内容 |
| `design.md` | `pypto-op-design` | 设计方案，覆盖可能丢失手动调整 |
| `{op}_golden.py` | `pypto-golden-generator` | 参考实现，覆盖可能丢失精度调试修改 |
| `test_{op}.py` | `pypto-op-develop` | 测试入口，覆盖可能丢失自定义测试用例 |
| `{op}_impl.py` | `pypto-op-develop` | 核心实现，覆盖可能丢失调优修改 |
| `README.md` | `pypto-op-develop` | 文档，覆盖可能丢失手动补充说明 |

## 无需覆盖确认的工件

| 工件 | 原因 |
|------|------|
| `output/output_*` | 运行时产出，可随时重新生成 |
| `.orchestrator_state.json` | orchestrator 自身状态，每次 stage 转换自动更新 |

## 续跑场景处理

当 orchestrator 检测到 `custom/{op}/` 目录已存在且包含历史工件时：

1. 读取 `.orchestrator_state.json` 确定上次中断位置
2. 列出已有工件，通知用户
3. 从最近失败或未完成的 stage 继续
4. 不主动覆盖已完成 stage 的工件
