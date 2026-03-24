---
name: pypto-op-workflow
description: PyPTO 算子开发工作流程。用于开发华为昇腾 AI 处理器自定义算子。在接到算子开发任务时使用，确保开发过程规范、高效、符合官方最佳实践。Triggers: 开发算子、算子开发流程、全流程开发、算子开发工作流、operator workflow。
tag: [PyPTO, 算子开发]
---

# PyPTO 算子开发工作流程

本技能提供 PyPTO 算子开发的完整工作流程指导。正式端到端开发优先使用 `pypto-op-orchestrator`；本 Skill 适合手动串联相关 Skills。

## 定位

| 入口 | 适用场景 | 特点 |
|------|----------|------|
| `pypto-op-workflow` | 手动串联完整开发流程 | 无状态、轻量、适合直接对话触发 |
| `pypto-op-orchestrator` | 正式端到端开发 | 有状态、带阶段门禁、支持恢复与重试 |

## 推荐阶段顺序

```
需求理解 → API 探索 → Golden 参考实现 → 设计方案 → 代码实现 → 精度修复（按需）→ 性能调优（按需）
```

## 对应 Skill 串联

1. **需求理解**：`pypto-intent-understanding`
2. **API 探索**：`pypto-api-explorer`
3. **Golden 参考实现**：`pypto-golden-generator`
4. **设计方案**：`pypto-op-design`
5. **代码实现**：`pypto-op-implement`
6. **精度修复（按需）**：`pypto-precision-debugger`
7. **性能分析与调优（按需）**：`pypto-op-perf-analyzer` → `pypto-op-perf-autotuner`

## 适用边界

### 本 Skill 负责什么

- 识别完整开发任务需要经过哪些阶段。
- 指导用户或上层 Agent 以正确顺序调用相关 Skills。
- 强调工件依赖关系与推荐执行顺序。

### 本 Skill 不负责什么

- 不维护 `.orchestrator_state.json`。
- 不定义全局重试策略、恢复入口或 BLOCKED / SUCCESS 结束态。
- 不替代 `pypto-op-implement`、`pypto-precision-debugger`、`pypto-op-perf-autotuner` 等阶段型 Skills 的细节职责。

## 核心原则

1. **先确认工件，再进入下游阶段**
   - `spec.md` 不完整，不进入 API 探索。
   - `api_report.md` 不完整，不进入 Golden / 设计阶段。
   - `design.md` 不完整，不进入代码实现。

2. **环境问题单独处理，不把环境准备当成主流程阶段**
   - 遇到环境阻塞时，调用 `pypto-environment-setup`。
   - 环境检查是前置条件，不单独占用主阶段编号。

3. **实现、精度修复、性能调优职责分离**
   - `pypto-op-implement` 只负责代码实现与测试入口生成。
   - `pypto-precision-debugger` 只负责精度问题定位与修复。
   - `pypto-op-perf-autotuner` 只在精度通过后进入。

4. **始终以真实验证结果推进**
   - 不得凭经验宣布精度通过或性能达标。
   - 任何结论都应有对应工件或命令输出支撑。

## 交付检查清单

- [ ] `spec.md` 已确认需求完整
- [ ] `api_report.md` 已确认 API 可行性
- [ ] `{op}_golden.py` 已生成并可作为精度基线
- [ ] `design.md` 已能指导实现
- [ ] `{op}_impl.py`、`test_{op}.py`、`README.md` 已生成
- [ ] 已完成首次真实验证，并明确是精度通过、精度失败还是运行失败
- [ ] 若精度通过，已完成性能分析；若进入调优，已有调优前后实测对比

## 使用建议

1. **想要完整但轻量地手动推进**：使用本 Skill。
2. **想要正式的端到端状态机**：切换到 `pypto-op-orchestrator`。
3. **只做单阶段工作**：直接调用对应的阶段型 Skill，不必经过本 Skill。
