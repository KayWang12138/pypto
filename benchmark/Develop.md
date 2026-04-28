# pypto KernelBench 桥接层开发说明

本文档记录 `benchmark` 的内部设计、目录结构和调试信息。运行
`test-integration.sh` 的日常用法见 [README.md](README.md)。

## 概述

桥接层把 KernelBench 用例、PyPTO 7 阶段算子生成、验证和报告汇总串起来:

1. **输入**: 上游 KernelBench 用例 (`Model(nn.Module) + get_inputs + get_init_inputs`)
2. **算子生成**: pypto 自带的 7 阶段 agent 工作流 (`pypto-op-orchestrator`)
3. **反作弊 + 精度 + 性能**: opencode 加载 `pypto-kernel-validate` skill
4. **报告**: skill 落 `skill_report.json`, 桥接层汇总 batch summary

代码迁移自 akg。

## 数据流

```text
KernelBench/<level>/{N}_{name}.py
  -> case_loader.load_case             AST + 子进程探针 -> CaseSpec
  -> case_loader.write_spec            写 custom/<op>/SPEC.md
  -> pypto_runner.run_pypto_workflow   opencode run --agent pypto-op-orchestrator
                                       prompt 内嵌 ModelNew 文件契约
                                       对齐 get_init_inputs/get_inputs 拆分
                                       可选跳过 Stage 7 perf-tune
                                       -> custom/<op>/<op>_impl.py
                                       -> custom/<op>/<op>_pypto_impl.py
                                       -> custom/<op>/<op>_golden.py
  -> verifier_runner.run_verifier      按 verifier_mode 分流
        opencode 模式:                 opencode run --agent pypto-kernel-validator
                                       加载 .opencode/skills/pypto-kernel-validate
                                       Step1 cheat_detector 脚本机械检测
                                       Step2 LLM 语义审阅隐性作弊
                                       Step3 KernelVerifier 精度 + 性能
                                       Step4 写 skill_report.json
        direct 模式:                   跳过 LLM, 直调 KernelVerifier
  -> report.write_summary              JSON + Markdown
```

## 数据集

仅支持上游 KernelBench PyTorch 原版。

- 来源: <https://github.com/ScalingIntelligence/KernelBench>
- 固定 commit: `21fbe5a642898cd60b8f60c7aefb43d475e11f33`
- 布局: `KernelBench/<level>/{N}_{name}.py`

下载到桥接层 `.cache/KernelBench/`:

```bash
bash benchmark/scripts/download_kernelbench.sh
```

自定义下载位置:

```bash
KERNELBENCH_DIR=/data/KernelBench \
  bash benchmark/scripts/download_kernelbench.sh
```

升级数据集 commit 时, 同步修改 `scripts/download_kernelbench.sh` 顶部的
`KERNELBENCH_COMMIT` 常量和 `case_loader.py` docstring 里的 commit 引用。

新增仓内自维护 case 时, 参考 [新增 KernelBench Case 指南](ADD_NEW_CASE.md),
按 KernelBench 标准入口补充 `Model` / `get_inputs` / `get_init_inputs`。

## 反作弊设计

反作弊分三层:

- **脚本机械层** (`verifier/cheat_detector.py`): AST 检 `import pypto`,
  `@pypto.frontend.jit` / `pypto.frontend.jit(...)`; 字符串检
  `for testing only` / `TODO use pypto` / `fallback` / `workaround` 等可疑文本。
- **LLM 语义层** (`.opencode/skills/pypto-kernel-validate/SKILL.md`):
  识别空壳 jit kernel、forward 双路径、try/except fallback、绕过 jit 的预后处理、
  mock kernel、tile config 关闭核心算子、可疑注释等脚本难以覆盖的问题。
- **运行时层** (`verifier/pypto_adapter.py:get_swimlane_benchmark_body`):
  swimlane trace 抓到多个 kernel 时输出 `CHEAT_MULTI_KERNEL`, `perf=inf`。
  multi-kernel 判定统一以 runtime profile 为准。

## 工件契约

- **外层 prompt** (`pypto_runner.py:_PROMPT_TEMPLATE`): 显式声明
  `{op}_pypto_impl.py` 文件契约, 包含 ModelNew 模板、自检命令, 并对齐
  `get_init_inputs()` / `get_inputs()` 两段式调用约定。
- **verifier 兜底** (`verifier/pypto_adapter.py:get_modelnew_loader`):
  当 `{op}_pypto_impl.py` 缺失或缺 `ModelNew` 类时, 自动从 `{op}_impl.py`
  读取 `{op}_wrapper` 并动态包装出 ModelNew。
- **PyPTO workflow 成功判定**: runner 以 `.orchestrator_state.json` 为准,
  只有 7 个 stage 全部 `completed` 才允许进入 verifier。未完成且无明确
  `failed/blocked/cancelled` 时可触发 incomplete workflow retry。

## 常用开发命令

离线冒烟:

```bash
bash benchmark/scripts/download_kernelbench.sh
bash benchmark/scripts/smoke_test.sh
```

最小 direct case:

```bash
export BENCHMARK_LOG_DIR=/tmp/benchmark_relu_direct
python -m benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode direct \
  --report-dir /tmp/benchmark_relu_direct/report
```

完整 opencode verifier 路径:

```bash
export BENCHMARK_LOG_DIR=/tmp/benchmark_relu_skill
python -m benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode opencode \
  --report-dir /tmp/benchmark_relu_skill/report
```

手动跑统一 verifier CLI:

```bash
python -m benchmark.verifier cheat-check ./custom/relu --op-name relu
python -m benchmark.verifier verify ./custom/relu \
  --op-name relu \
  --task-desc ./custom/relu/task_desc.py \
  --mode performance \
  --json-out ./custom/relu/verify_run.json
```

断点续跑:

```bash
python -m benchmark.run_kernelbench --cases 19_ReLU --skip-pypto-gen
python -m benchmark.run_kernelbench --cases 19_ReLU --skip-stage7-perf-tune
```

## scripts/local

`scripts/local/` 覆盖 4 类测试:

1. **test-unit**: 不需要 NPU, 不烧 LLM。
2. **test-direct**: 需 NPU, 不烧 LLM。
3. **test-skill (legit / cheat)**: 需 NPU + LLM, 走 `pypto-kernel-validator`。
4. **test-integration**: 真 7 阶段 + skill。用户用法见 [README.md](README.md)。

## 目录结构

```text
pypto/benchmark/
├── __init__.py
├── case_loader.py              # KernelBench .py -> SPEC.md + task_desc
├── pypto_runner.py             # opencode run --agent pypto-op-orchestrator
├── verifier_runner.py          # opencode skill / direct KernelVerifier 调度
├── verifier/                   # 自包含验证子包
│   ├── kernel_verifier.py      # 精度 + 性能验证
│   ├── cheat_detector.py       # 脚本机械层反作弊
│   ├── pypto_adapter.py        # PyPTO 代码片段与 runtime 检测
│   ├── __main__.py             # verifier CLI
│   └── ...
├── run_kernelbench.py          # CLI 入口 + 批处理调度
├── report.py                   # JSON / Markdown 汇总
├── configs/default.yaml        # 默认配置
├── scripts/
│   ├── download_kernelbench.sh
│   ├── smoke_test.sh
│   └── local/
│       ├── lib.sh
│       ├── test-all.sh
│       ├── test-unit.sh
│       ├── test-direct.sh
│       ├── test-skill-legit.sh
│       ├── test-skill-cheat.sh
│       ├── test-integration.sh
│       └── fixtures/
├── .cache/KernelBench/
├── ADD_NEW_CASE.md
├── Develop.md
└── README.md

pypto/.opencode/
├── agents/
│   ├── pypto-op-orchestrator.md
│   └── pypto-kernel-validator.md
└── skills/
    └── pypto-kernel-validate/
        └── SKILL.md
```

`scripts/dev/` 不作为仓内共享内容维护。如需跨机器联调, 可在本地自行建立,
该目录由 `.gitignore` 屏蔽。

## 产物布局

```text
<pypto_repo>/
└── custom/<op>/
    ├── SPEC.md
    ├── API_REPORT.md
    ├── DESIGN.md
    ├── <op>_golden.py
    ├── <op>_impl.py
    ├── <op>_pypto_impl.py
    ├── test_<op>.py
    └── README.md

<report-dir>/
├── summary.json
├── summary.md
└── <op>/
    ├── pypto_run.log
    ├── pypto_session.md
    ├── verifier.log
    ├── verifier_session.md
    ├── skill_report.json
    ├── <op>_task_desc.py
    └── result.json
```

## 配置

所有 CLI 选项都可通过 `configs/default.yaml` 设置默认值。

`verifier` 子包默认把工作目录写到 `~/pypto_bench_logs/Task_<rand>/<op>/`。
可通过 CLI `--log-dir` 或环境变量 `PYPTO_BENCH_LOG_DIR` 覆盖。调试时可传
`--keep-verifier-artifacts` / `--keep-artifacts`, 或设置
`PYPTO_BENCH_KEEP_ARTIFACTS=1` 保留临时脚本和源码副本。

## 已知约束

- pypto agent 工作流真跑 LLM, 单 case 平均耗时数分钟到 30 分钟以上。
- 仅支持 NPU/Ascend 后端。
- 仅支持 PyTorch 框架。
- `bench_dir` 需要指向包含 `KernelBench/<level>/{N}_{name}.py` 的目录。
- 多卡并发依赖 `TILE_FWK_DEVICE_ID` 隔离, 同一时刻每卡仅 1 个 case。

## 失败排查

- **`level dir 不存在`**: 先跑 `bash benchmark/scripts/download_kernelbench.sh`,
  或用 `--bench-dir` / `--level` 指向已有路径。
- **`level_dir 下找不到任何 .py 用例`**: 检查 `--bench-dir` 是否指到
  `KernelBench/KernelBench/` 这一层。
- **opencode 子进程超时**: 加大 `--timeout-sec` 或 `--skill-timeout`,
  或检查 LLM / opencode 配置。
- **pypto 产物缺失**: 看 `<report-dir>/<op>/pypto_run.log`。
- **`opencode 可执行未找到`**: 安装 opencode CLI, 或临时使用
  `--verifier-mode direct`。
- **`skill_report.json 未产出`**: 看 `<report-dir>/<op>/verifier.log`。
- **`final_verdict=FAIL_CHEAT`**: 看 `skill_report.json` 的
  `performance.cheat_multi_kernel` 和 `cheat_check_semantic` evidence 字段。
- **KernelVerifier 失败**: 看 `<report-dir>/<op>/verifier.log`。
