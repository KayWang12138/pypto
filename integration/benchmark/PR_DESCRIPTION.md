# PR Description

## 背景

这个 PR 交付的是一套面向 PyPTO 的 KernelBench benchmark / verifier 工作流。
它把上游 KernelBench 用例、PyPTO 算子产物、反作弊检查、精度验证、性能验证和汇总报告
串成了一条可复现的端到端链路。

代码迁移自 akg。

## 主要改动

### 1. 新增自包含的 benchmark 桥接层

- 新增 `integration/benchmark/` 目录，作为 KernelBench 与 PyPTO 的桥接层
- 支持从上游 KernelBench case 自动生成 `SPEC.md` 和 `task_desc.py`
- 接入 PyPTO 7 阶段 agent 工作流，生成 `custom/<op>/` 下的算子产物
- 接入 verifier 工作流，统一输出单 case 结果和 batch 汇总报告

### 2. 新增 verifier / 反作弊 / 报告能力

- 新增统一入口 `integration/benchmark/verifier_runner.py`
- 支持 `direct` 和 `opencode` 两种验证模式
- 增加脚本机械层反作弊检测和 skill 语义层反作弊审阅
- 支持 `correctness`、`performance`、`full` 三种验证模式
- 输出 `result.json`、`summary.json`、`summary.md`、`skill_report.json`

### 3. 新增本地复现脚本和 fixture

- 新增 `integration/benchmark/scripts/download_kernelbench.sh`
- 新增 `integration/benchmark/scripts/smoke_test.sh`
- 新增 `integration/benchmark/scripts/local/test-unit.sh`
- 新增 `integration/benchmark/scripts/local/test-direct.sh`
- 新增 `integration/benchmark/scripts/local/test-skill-legit.sh`
- 新增 `integration/benchmark/scripts/local/test-skill-cheat.sh`
- 新增 `integration/benchmark/scripts/local/test-integration.sh`
- 新增反作弊 fixture 与 `ReLUCheat` 样例

### 4. 目录与接口统一

- 桥接层目录统一为 `integration/benchmark`
- Python 模块入口统一为 `integration.benchmark.*`
- 验证入口统一为 `verifier_runner.py`
- 日志名、任务名前缀、报告标题统一使用 `benchmark`

### 5. 文档与 skill 同步

- 更新 `integration/benchmark/README.md`
- 增加 step-by-step 复现方式
- 增加 reviewer 可直接照做的本地回归脚本说明
- 修正和清理不准确的运行依赖表述
- `.opencode/skills/pypto-kernel-validate/SKILL.md` 中的调用路径已同步更新

## 如何复现

下面按步骤给出下载、冒烟、回归和集成验证命令。

### Step 0: 下载数据

```bash
cd pypto
bash integration/benchmark/scripts/download_kernelbench.sh
```

### Step 1: 跑离线冒烟

```bash
cd pypto
bash integration/benchmark/scripts/smoke_test.sh
```

预期结果：

- `integration.benchmark.*` 关键模块可 import
- `case_loader` 能从 `19_ReLU.py` 生成 `SPEC.md` 和 `task_desc.py`
- `run_kernelbench --help` 正常输出

### Step 2: 跑无 NPU / 无 LLM 的 unit 回归

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/unit \
  bash integration/benchmark/scripts/local/test-unit.sh
```

预期结果：

- `cheat_detector` 对 5 类 fixture 的 verdict 与预期一致
- `verifier verify` 对 `multi_jit` fixture 触发 cheat-gate
- `verdict_machine=failed_cheat`
- `correctness.status=skipped`

### Step 3: 跑 direct 模式最小正例

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/direct \
  bash integration/benchmark/scripts/local/test-direct.sh
```

预期结果：

- 在真实 NPU 上完成 correctness 验证
- 在真实 NPU 上完成 performance 验证
- `verdict_machine=pass`
- `cheat_multi_kernel=False`

### Step 4: 跑 opencode skill 正例

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/skill_legit \
  bash integration/benchmark/scripts/local/test-skill-legit.sh
```

预期结果：

- `pypto-kernel-validator` 成功拉起
- `custom/ReLU/.skill_validate/skill_report.json` 成功生成
- `final_verdict=PASS`
- `cheat_check_script=pass`
- `cheat_check_semantic=pass`
- `correctness=passed`

### Step 5: 跑 opencode skill 反作弊负例

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/skill_cheat \
  bash integration/benchmark/scripts/local/test-skill-cheat.sh
```

预期结果：

- `ReLUCheat` fixture 被 skill 判定为作弊
- `skill_report.json` 中 `final_verdict=FAIL_CHEAT`

### Step 6: 跑 batch 集成链路

开发期可先跑 cheap 集成，复用现有 `custom/<op>/` 产物，只验证
`run_kernelbench -> verifier -> report` 这一段：

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/integration \
  FULL=0 bash integration/benchmark/scripts/local/test-integration.sh
```

预期结果：

- `summary.json` / `summary.md` 成功生成
- `totals.success == totals.total`
- `correctness.fail == 0`
- `mode=performance`

如需验证真正的 7 阶段开发链路，再跑 FULL：

```bash
cd pypto
BENCHMARK_LOG_DIR=/tmp/benchmark_pr_steps/integration_full \
  bash integration/benchmark/scripts/local/test-integration.sh
```

说明：

- `FULL=1` 会真正触发 `pypto-op-orchestrator` 的 Stage 1-7
- 单 case 耗时较长，适合在 reviewer 需要确认全链路时再执行

## 其他命令

下面这些命令用于单独执行 direct / opencode case 和查看结果。

### 手工跑单 case direct

```bash
cd pypto
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode direct \
  --report-dir /tmp/benchmark_relu_direct/report
```

### 手工跑单 case opencode

```bash
cd pypto
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode opencode \
  --report-dir /tmp/benchmark_relu_skill/report
```

### 手工查看结果

```bash
cat /tmp/benchmark_relu_direct/report/summary.md
cat /tmp/benchmark_relu_direct/report/ReLU/result.json

cat /tmp/benchmark_relu_skill/report/summary.md
cat /tmp/benchmark_relu_skill/report/ReLU/result.json
cat custom/ReLU/.skill_validate/skill_report.json
```

## 建议截图

- `smoke_test.sh` 通过时的终端输出
- `test-direct.sh` 中 correctness / performance 都通过的终端输出
- `test-skill-legit.sh` 中 `final_verdict=PASS` 的终端输出
- `test-skill-cheat.sh` 中 `final_verdict=FAIL_CHEAT` 的终端输出
- `test-integration.sh` 生成的 `summary.md` 总览部分
- `custom/ReLU/.skill_validate/skill_report.json` 或 `<report-dir>/<op>/result.json`

## Reviewer 关注点

- direct / opencode 两条验证路径是否都能独立工作
- 反作弊正例和负例是否都能稳定给出明确 verdict
- batch 报告是否能稳定落盘并给出可读汇总
