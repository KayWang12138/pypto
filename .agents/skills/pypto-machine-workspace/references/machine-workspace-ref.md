# Workspace 内存异常参考

本文档提取自 `docs/trouble_shooting/machine.md` 的 Workspace 章节，用于快速参考。

## Workspace 内存预算结构

```
workspaceSize = memBudget.Total()
             = tensor.Total() + aicoreSpilled + debug.dumpTensor + debug.leafDump + metadata.Total()
```

详细诊断流程请参考主 SKILL.md。
