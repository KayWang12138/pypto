# `.orchestrator_state.json` Schema v2 设计与实现

状态：已实施（2026-04-21）
适用范围：`.opencode/plugins/lib/state-transition-core.ts`、`.opencode/plugins/pypto-state-transition.ts` 及 orchestrator agent 文档。`.agents/hooks/pypto-op-lint/` 下的 lint 逻辑保持不变。

---

## 1. 动机

v1 `.orchestrator_state.json` 只能回答"现在到第几阶段"，当 stage 6 失败三次后无从得知：
- 每次为什么失败？失败模式一样吗？
- 当前状态是哪一个 Subagent 写的？
- Stage 4 完成时落盘的 Golden 文件，Stage 5 开始前是不是还是同一份？

v2 在**保留所有现有字段**的前提下，额外维护三类证据：
1. 产物指纹（`artifacts`）——用于跨阶段产物一致性校验
2. 事件流水（`history`）——用于复盘与重试决策
3. 最近一次失败（`last_error`）——用于恢复时直接读取

以及一个用于后续迁移的 `schema_version` 标签。

## 2. Schema v2 全貌

```ts
interface OrchestratorState {
  // ── v1 字段（保持不变） ────────────────────────────
  operator_name?: string;
  current_stage: number;
  stage_status: Record<string, string>;         // "1".."7" → pending/in_progress/completed/failed
  stage_retry_count?: Record<string, number>;
  last_updated?: string;
  // plugin 写入的旁字段（保持原样）
  spec_md_hash?: string;
  perf_iteration?: { count: number; last_improvement: number; consecutive_no_improvement: number };

  // ── v2 新增（均可选，读取老文件时自动补默认） ─────
  schema_version?: number;                              // 当前固定为 2
  history?: HistoryEntry[];                             // 最多 50 条（HISTORY_CAP）
  artifacts?: Record<string, ArtifactFingerprint>;      // key = 文件名
  last_error?: StageErrorRecord & { stage: number; at: string };
}

type HistoryEntry = {
  at: string;                               // ISO 时间戳
  action: "init" | "start_stage" | "complete_stage" | "fail_stage";
  stage: number;
  attempt: number;                          // 本次动作前的 retry_count + 1
  reason?: string;
  gate?: { warnCount: number; failCount: number; failRuleIds?: string[] };
  error?: StageErrorRecord;
};

type ArtifactFingerprint = {
  sha256: string;
  stage: number;                            // 该指纹由第几阶段的 complete_stage 写入
  recorded_at: string;
};

type StageErrorRecord = {
  message: string;
  kind?: string;
  rule_ids?: string[];
};
```

## 3. 行为规约（由 `applyTransition` 保证）

| 动作 | 变更 |
|---|---|
| `init` | `current_stage=1`, `stage_status["1"]="in_progress"`；写 history |
| `start_stage N` | 前置校验→ `stage_status[N]="in_progress"`；写 history；**不**清除 `last_error` |
| `complete_stage N` | `stage_status[N]="completed"`，`current_stage = min(N+1, 7)`；若 `last_error.stage===N` 则清空；写 history（gate 字段可带） |
| `fail_stage N` | `stage_retry_count[N]++`，`stage_status[N]="failed"`；设置 `last_error`；写 history |

其它不变语义：
- `init` 要求 stage=1 且无其它阶段 in_progress
- `start_stage` 要求上一阶段已完成（>=2 时）
- `complete_stage` 要求当前阶段必须是 in_progress
- 所有动作都刷新 `schema_version=2` 与 `last_updated=now`
- history 用 FIFO 裁到 50 条

`pypto-state-transition.ts` 在 `state_transition` 工具层附加以下职责：
- `complete_stage` 成功后，用 `captureStageArtifacts(opDir, stage)` 计算并写入当 stage 的 artifacts SHA256
- `fail_stage` 透传 `errorKind` / `errorMessage` 给 core
- SPEC.md 的 hash 仍由 `spec_md_hash` 字段兼容记录（未下沉到 `artifacts` 以避免 D5 checks 变化）

## 变更范围

| 文件 | 类型 | 说明 |
|---|---|---|
| `.opencode/plugins/lib/state-transition-core.ts` | 新增类型 + 逻辑 | `SCHEMA_VERSION`、`HISTORY_CAP`、`HistoryEntry`、`ArtifactFingerprint`、`StageErrorRecord`、`migrateState`、attempt 快照、history 追加、last_error 管理 |
| `.opencode/plugins/pypto-state-transition.ts` | 追加参数 | 工具 args 增加 `errorKind` / `errorMessage`；`complete_stage` 后调用 `captureStageArtifacts` 落盘 sha |
| `.opencode/plugins/__tests__/state-transition-core.test.ts` | 追加用例 | 保留 9 条原用例；新增 4 条覆盖 schema_version、history、last_error、history cap、migrateState |
| `.opencode/agents/pypto-op-orchestrator.md` | 追加段落 | 在"状态文件结构"段末尾新增 v2 字段说明与用途 |

**不改动**：`pypto_op_lint/infer.py`、`pypto_op_lint/hooks.py`、`rules.json`、D5 checks、现有 `custom/<op>/.orchestrator_state.json`（读到旧文件自动迁移）。

## 4. 迁移与兼容

**读取路径**：`pypto-state-transition.ts` 调用 `applyTransition`，内部 `migrateState(state)` 将缺失的 v2 字段补齐（`schema_version=2`、`history=[]`、`artifacts={}`）。写回时已是 v2 形态。v1 文件在第一次 `state_transition` 后被自动升级，无需手动迁移。

**反向兼容**：Python 端 lint / infer 只读 `operator_name`、`current_stage`、`stage_status`、`spec_md_hash`，这些在 v2 中保持不变；现存 `.orchestrator_state.json` 文件无需删除即可继续工作。

## 4. 失败恢复模型

Orchestrator 在 crash 后恢复时：
1. 读 `.orchestrator_state.json`。
2. 若 `last_error` 存在，直接据此展示"上次失败原因"而不需要回溯 history。
3. 若需要判定"是否该继续重试"，查看 `stage_retry_count[N]` 与重试上限。
4. 若需要审计最近的动作流，扫描 `stage_history`（最新 50 条）。
5. 若需要判定"阶段工件是否被后续动作篡改"，对比 `artifacts[name].sha256` 与当前文件的 sha256。

## 5. 实施清单

| 文件 | 变更 | 状态 |
|---|---|---|
| `.opencode/plugins/lib/state-transition-core.ts` | 扩展类型 + applyTransition + migrateState | ✅ |
| `.opencode/plugins/pypto-state-transition.ts` | 增加 `errorKind/errorMessage` 参数、`captureStageArtifacts` | ✅ |
| `.opencode/plugins/__tests__/state-transition-core.test.ts` | 新增 4 组用例，13 项全部通过 | ✅ |
| `.opencode/agents/pypto-op-orchestrator.md` | 在"状态持久化"节追加 v2 字段说明表 | ✅ |
| Python 侧 `pypto_op_lint` | 仅读 `current_stage/operator_name/stage_retry_count`，无需修改 | ✅（验证无影响） |

## 6. 测试结果

```
bun test plugins/__tests__/
 21 pass  21 expected   ran in 26ms

pytest .agents/hooks/pypto-op-lint/tests/
 81 passed in 5.44s
```

## 7. 不做的事（scope-out）

- **不**做 schema v3 前置设计（`environment` 字段、跨算子聚合）
- **不**动 `rules.json`（新增 v2 字段一致性检查规则要另起 PR）
- **不**迁移已存的 4 个 `custom/<op>/.orchestrator_state.json`（读时自动迁移，不主动改盘）
