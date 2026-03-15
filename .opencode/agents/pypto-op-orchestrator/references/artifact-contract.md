# Artifact Contract

## 标准目录结构

```text
custom/{op}/
├── spec.md
├── design.md
├── {op}_golden.py
├── test_{op}.py
├── {op}_impl.py
├── README.md
├── .orchestrator_state.json
└── output/
    └── output_*/
```

## 工件 Owner / Consumer 表

| 工件 | Owner | 主要消费者 | 导出函数 |
|------|-------|------------|----------|
| `spec.md` | `pypto-intent-understanding` | orchestrator / golden / design | — |
| `design.md` | `pypto-op-design` | orchestrator / op-develop | — |
| `{op}_golden.py` | `pypto-golden-generator` | test / accuracy / debug | `{op}_golden()` |
| `test_{op}.py` | `pypto-op-develop` | orchestrator / perf-autotuner | — |
| `{op}_impl.py` | `pypto-op-develop` | test / accuracy / debug / perf | `{op}_wrapper()` |
| `README.md` | `pypto-op-develop` | 用户 / 审查者 | — |
| `.orchestrator_state.json` | orchestrator | orchestrator | — |
| `output/output_*` | runtime | verify / perf tools | — |

## 三文件分离与导入约定

三个 Python 文件职责严格分离：

| 文件 | 职责 | 导出 |
|------|------|------|
| `{op}_golden.py` | 纯 torch 参考实现 | `{op}_golden()` |
| `{op}_impl.py` | PyPTO kernel 实现 | `{op}_wrapper()`（内含 `@pypto.frontend.jit` kernel） |
| `test_{op}.py` | 测试入口，只做 import + 调用 + 精度对比 | 不导出，作为入口脚本运行 |

`test_{op}.py` 不包含 golden 或 kernel 的实现代码，只通过 import 引用：

```python
from {op}_golden import {op}_golden
from {op}_impl import {op}_wrapper
```

## Overwrite Policy

| 工件 | 覆盖前需确认 |
|------|-------------|
| `spec.md` / `design.md` / `{op}_golden.py` / `test_{op}.py` / `{op}_impl.py` / `README.md` | 是 |
| `output/output_*` / `.orchestrator_state.json` | 否 |

## 幂等与重入

- orchestrator 根据 `.orchestrator_state.json` 判断从哪个 stage 继续
- child skill 不应假设流程从头开始
- 上游工件有效时，下游失败从最近失败 stage 重启
