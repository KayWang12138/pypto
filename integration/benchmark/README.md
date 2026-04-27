# pypto KernelBench 桥接层

> **状态**: MVP. 上游 PyTorch KernelBench 数据集, NPU/Ascend 后端.

## 概述

外层包装层, 把四件事串起来:

> 代码迁移自 akg。

1. **输入**: 上游 KernelBench 用例 (`Model(nn.Module) + get_inputs + get_init_inputs`)
2. **算子生成**: pypto 自带的 7 阶段 agent 工作流 (`pypto-op-orchestrator`)
3. **反作弊 + 精度 + 性能**: opencode 加载 `pypto-kernel-validate` skill, agent 在 skill 引导下
   做"脚本机械检测 + LLM 语义审阅 + KernelVerifier 精度 + 性能"
4. **报告**: skill 落 `skill_report.json`, 桥接层汇总成 batch summary

数据流:

```
KernelBench/<level>/{N}_{name}.py   (上游 PyTorch case 文件)
  -> case_loader.load_case             AST + 子进程探针 -> CaseSpec
  -> case_loader.write_spec            写 custom/<op>/SPEC.md
  -> pypto_runner.run_pypto_workflow   子进程 opencode run --agent
                                       pypto-op-orchestrator
                                       prompt 内嵌 ModelNew 文件契约
                                       (对齐 get_init_inputs/get_inputs 拆分,
                                       可选跳过 Stage 7 perf-tune)
                                       -> custom/<op>/<op>_impl.py
                                       -> custom/<op>/<op>_pypto_impl.py
                                       -> custom/<op>/<op>_golden.py
  -> verifier_runner.run_verifier      按 verifier_mode 分流:
        opencode 模式 (默认):           子进程 opencode run --agent pypto-kernel-validator
                                       agent 加载 .opencode/skills/pypto-kernel-validate
                                       Step1 cheat_detector (脚本机械层)
                                       Step2 LLM 亲自语义审阅 S1-S9 隐性作弊
                                       Step3 KernelVerifier 精度 + 性能
                                       Step4 写 skill_report.json
        direct 模式:                    跳过 LLM, 直调 KernelVerifier (CI / 离线 dev)
  -> report.write_summary              JSON + Markdown
```

反作弊设计 (脚本 + LLM 双层):

- **脚本机械层** (`verifier/cheat_detector.py`):
  AST 检 `import pypto`, 是否存在 `@pypto.frontend.jit` / `pypto.frontend.jit(...)`;
  字符串检 `for testing only` / `TODO use pypto` / `fallback` / `workaround` 等可疑文本.
  确定性, 无 LLM 开销, 抓最显眼的作弊. 注意 multi-kernel 不再由 AST 中 jit
  数量判定, 统一以运行时 profile 结果为准.
- **LLM 语义层** (`.opencode/skills/pypto-kernel-validate/SKILL.md`):
  S1 空壳 jit kernel, S2 forward 双路径, S3 try/except fallback, S4 绕过 jit 的预/后处理,
  S5 多 kernel 拆分 (以 runtime profile 为唯一真相源), S6 mock kernel, S7 shape 硬编码,
  S8 tile config 关闭核心算子, S9 可疑注释.
  非确定性但能识别脚本抓不到的隐性作弊形态. 这是本桥接层向外团队贡献 skill 的核心价值.
- **运行时层** (`verifier/pypto_adapter.py:get_swimlane_benchmark_body`):
  swimlane trace 抓到 >1 个 kernel 时直接 stdout 打 `CHEAT_MULTI_KERNEL`, perf=inf,
  不再做"多 trace span 累加"作弊兜底. 这是 multi-kernel 判定的唯一真相源.

工件契约 (与 pypto 内部工作流解耦):

- **外层 prompt** (`pypto_runner.py:_PROMPT_TEMPLATE`): 显式声明
  `{op}_pypto_impl.py` 文件契约 (含完整 ModelNew 模板、自检命令, 并明确
  对齐 `get_init_inputs()` / `get_inputs()` 的两段式调用约定).
  pypto 内部 SKILL/agent 不动, agent 像消费普通用户需求一样消费这段 prompt.
- **verifier 兜底** (`verifier/pypto_adapter.py:get_modelnew_loader`):
  生成的 ModelNew 加载代码内置 fallback — 当 `{op}_pypto_impl.py` 缺失或缺
  `ModelNew` 类时, 自动从 `{op}_impl.py` 中读取 `{op}_wrapper` 并动态包装
  出 ModelNew. 这层兜底纯属验证脚本自身的鲁棒性增强, 与 pypto 模板无关.

## 数据集

仅支持上游 KernelBench (PyTorch 原版).

- **来源**: <https://github.com/ScalingIntelligence/KernelBench>
- **固定 commit**: `21fbe5a642898cd60b8f60c7aefb43d475e11f33`
- **布局**: `KernelBench/<level>/{N}_{name}.py`,
  每个 .py 内含 `class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`.

下载到桥接层 `.cache/KernelBench/` (gitignored):

```bash
bash pypto/integration/benchmark/scripts/download_kernelbench.sh
```

自定义下载位置:

```bash
KERNELBENCH_DIR=/data/KernelBench \
  bash pypto/integration/benchmark/scripts/download_kernelbench.sh
# 然后在 CLI 用 --bench-dir /data/KernelBench/KernelBench
```

升级数据集 commit: 修改 `scripts/download_kernelbench.sh` 顶部的
`KERNELBENCH_COMMIT` 常量 + `case_loader.py` docstring 里的 commit 引用.

新增仓内自维护 case: 参考 [新增 KernelBench Case 指南](ADD_NEW_CASE.md),
按 KernelBench 标准入口补充 `Model` / `get_inputs` / `get_init_inputs`,
新规范 case 可在顶层声明 `FORMULA` 和 `DYNAMIC_AXIS`.

## 快速开始

### 依赖

```bash
# 1. 已安装 pypto (可正常 import)
# 2. 已安装 opencode CLI (which opencode)
# 3. 已配置可用的 LLM / opencode 运行环境, 因为 pypto agent 真跑 LLM
# 4. 已设置 NPU 环境 (CANN, torch_npu, TILE_FWK_DEVICE_ID 等)
# 5. 已下载上游 KernelBench (固定在 commit 21fbe5a):
bash pypto/integration/benchmark/scripts/download_kernelbench.sh
```

> 桥接层及内置 verifier 完全在 `pypto/integration/benchmark/` 下, 不需要安装任何额外包.

## 一步步复现单 case

下面这组命令适合直接写进 PR 说明, 也适合你后面自己复现并截图:

### Step 1: 拉数据并跑离线冒烟

```bash
cd pypto
bash integration/benchmark/scripts/download_kernelbench.sh
bash integration/benchmark/scripts/smoke_test.sh
```

### Step 2: 跑一个最小 direct case

```bash
cd pypto
export BENCHMARK_LOG_DIR=/tmp/benchmark_relu_direct
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode direct \
  --report-dir /tmp/benchmark_relu_direct/report
```

预期你会看到:

- CLI 正常结束, `summary.json` / `summary.md` / `result.json` 被写出.
- `result.json` 中 `pypto_status`、`verifier_status`、`correctness` 都有明确结果.

### Step 3: 查看报告产物

```bash
cat /tmp/benchmark_relu_direct/report/summary.md
cat /tmp/benchmark_relu_direct/report/ReLU/result.json
```

### Step 4: 如需验证完整 skill 路径, 再跑 opencode 模式

```bash
cd pypto
export BENCHMARK_LOG_DIR=/tmp/benchmark_relu_skill
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance \
  --verifier-mode opencode \
  --report-dir /tmp/benchmark_relu_skill/report
```

### PR 截图建议

- 终端里 `run_kernelbench` 跑 `19_ReLU` 的实时输出.
- `<report-dir>/summary.md` 的总览段落.
- `<report-dir>/ReLU/result.json` 里 `correctness` / `perf` / `verifier_status` 字段.
- 如果走 `opencode` 模式, 再补一张 `skill_report.json` 或 `verifier.log` 的截图.

### MVP: 单用例端到端 (默认走 opencode skill, 含 LLM 反作弊语义审阅)

```bash
cd pypto
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --level level1 \
  --devices 0 \
  --arch ascend910b4 \
  --mode performance
```

> `--cases` 接受完整 stem (`19_ReLU`)、仅序号前缀 (`19`) 或闭区间
> (`1:21,31,41:50`).
> 不传 `--bench-dir` 时默认指向 `.cache/KernelBench/KernelBench/`.

### CI / 离线 dev: 跳过 LLM 语义层 (--verifier-mode direct)

```bash
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU --mode performance --verifier-mode direct
```

> direct 模式仍会跑脚本机械层反作弊 (cheat_detector) 和运行时多 kernel 检测,
> 但跳过 SKILL.md 中要求的 LLM 语义审阅. 仅用于不愿付 LLM 调用成本的场景.

### correctness / 快速生成: 跳过 Stage 7 迭代性能调优

```bash
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --mode correctness \
  --skip-stage7-perf-tune
```

> 该参数只跳过 pypto 工作流第七步的迭代性能优化, 不会跳过 Stage 1-6 的实现与精度验证.

### 批处理多用例 (含性能, 多卡)

```bash
python -m integration.benchmark.run_kernelbench \
  --cases 19:21 \
  --level level1 \
  --devices 0,1,2 \
  --concurrency 3 \
  --mode performance \
  --report-dir benchmark_report
```

### 批处理多 level

KernelBench 每个 level 都从 `1` 开始编号, 因此多 level 的 level 信息
必须直接合并到 `--cases` / `CASE` 中.

CLI 写法:

```bash
python -m integration.benchmark.run_kernelbench \
  --cases 'level1=1:21;level2=31,41:50' \
  --devices 0,1 \
  --concurrency 2 \
  --mode performance
```

环境变量入口遵循同一规则:

```bash
CASE='level1=1:21;level2=31,41:50' \
python -m integration.benchmark.run_kernelbench \
  --devices 0,1 \
  --concurrency 2 \
  --mode performance
```

`integration/benchmark/scripts/local/test-integration.sh` 和 monitor 入口也读取
`CASE`/`CASES`. 多 level 不使用 `LEVELS`/`LEVEL`; 写成
`CASE='level1=...;level2=...'` 才无歧义.

裸 selector 仍可用于单 level:

```bash
python -m integration.benchmark.run_kernelbench \
  --level level1 \
  --cases 1:21,31,41:50 \
  --devices 0,1 \
  --concurrency 2 \
  --mode performance
```

多 level 运行时, pypto 产物与报告按 `level/op` 分目录, 避免不同 level
出现同名 op 时互相覆盖. `summary.md` 会同时输出聚合总览和按 level 汇总.

### 仅手动跑统一 verifier CLI (跳过 pypto 生成阶段)

```bash
python -m integration.benchmark.verifier cheat-check ./custom/relu --op-name relu
python -m integration.benchmark.verifier verify ./custom/relu \
  --op-name relu \
  --task-desc ./custom/relu/task_desc.py \
  --mode performance \
  --json-out ./custom/relu/verify_run.json
```

### 跑前 N 个

```bash
python -m integration.benchmark.run_kernelbench --level level1 --limit 5 ...
```

### 断点续跑

外层会检测 `custom/{op}/{op}_pypto_impl.py` 是否存在, 已存在的算子默认跳过 pypto 生成阶段, 直接进入验证. 强制重跑请加 `--force-regen`.

```bash
# 只重跑验证 (不重新生成算子)
python -m integration.benchmark.run_kernelbench --cases 19_ReLU --skip-pypto-gen
```

若希望重跑生成但跳过性能调优, 可改用:

```bash
python -m integration.benchmark.run_kernelbench \
  --cases 19_ReLU \
  --skip-stage7-perf-tune
```

## 测试脚本 (scripts/local)

仓内只保留 `scripts/local/` 这一套可共享脚本, 用于单机和 CI 回归.
如果你需要跨机器联调, 可以在本地自行维护 `scripts/dev/`;
该目录已加入 `.gitignore`, 适合放 ssh/rsync/nohup 等个人环境脚本, 不会进入 PR.

`scripts/local/` 覆盖 4 类测试 (按依赖从轻到重):

1. **test-unit** — 不需要 NPU, 不烧 LLM (~10 s).
   `cheat_detector` 5 类 fixture + `verifier verify` 的 cheat-gate.
2. **test-direct** — 需 NPU, 不烧 LLM (~30 s).
   走 `--verifier-mode direct`, 跑 `KernelVerifier` 的 correctness + performance.
3. **test-skill (legit / cheat)** — 需 NPU + 烧 LLM (~3-5 min/case).
   走 `--verifier-mode opencode`, spawn `pypto-kernel-validator` subagent + 加载 `pypto-kernel-validate` skill;
   `legit` 用现成 `custom/ReLU/`, `cheat` 自动部署 `relu_cheat` fixture (runtime 多 kernel + mock kernel).
4. **test-integration** — 真 7 阶段 + skill (~50 min/case, FULL 模式).
   不带 `--skip-pypto-gen`, 让 `pypto-op-orchestrator` 真跑 Stage 1-7 算子开发, 再走 verifier.
   开发期可 `FULL=0` 走 cheap 模式 (~3 min) 复用产物.

### 单机一键 (scripts/local/)

```bash
cd pypto

# 一键跑全 (4 步, ~60 min, 含 FULL 集成)
bash integration/benchmark/scripts/local/test-all.sh

# 快速回归 (跳 50 min 的 test-integration)
BUDGET=fast bash integration/benchmark/scripts/local/test-all.sh

# CI 离线 (只跑 test-unit, 不需 NPU 不烧 LLM)
SKIP_NPU=1 bash integration/benchmark/scripts/local/test-all.sh

# 单跑某一步
bash integration/benchmark/scripts/local/test-unit.sh
bash integration/benchmark/scripts/local/test-direct.sh
bash integration/benchmark/scripts/local/test-skill-legit.sh       # 复用 custom/ReLU/
bash integration/benchmark/scripts/local/test-skill-cheat.sh       # 自动部署 relu_cheat fixture
FULL=0 bash integration/benchmark/scripts/local/test-integration.sh # cheap 集成
```

各步骤的 log 默认落在 `${BENCHMARK_LOG_DIR:-/tmp/benchmark_test_<ts>}/`,
可设 `BENCHMARK_LOG_DIR=/path` 自定义.

## 目录结构

```
pypto/integration/benchmark/
├── __init__.py
├── case_loader.py              # KernelBench .py -> SPEC.md + task_desc
├── pypto_runner.py             # 子进程跑 opencode run --agent pypto-op-orchestrator
├── verifier_runner.py      # 双模式调度: opencode skill / direct KernelVerifier
├── verifier/                   # 自包含验证子包
│   ├── kernel_verifier.py      # 精度 + 性能验证 (ascend + torch + pypto)
│   ├── cheat_detector.py       # 脚本机械层反作弊 (AST + 文件系统 + 字符串)
│   ├── pypto_adapter.py        # PyPTO 代码片段; 含运行时多 kernel CHEAT 标记
│   ├── __main__.py             # 统一 verifier CLI: cheat-check / verify
│   └── ...
├── run_kernelbench.py          # CLI 入口 + 批处理调度
├── report.py                   # JSON / Markdown 汇总
├── configs/default.yaml        # 默认配置
├── scripts/
│   ├── download_kernelbench.sh # 拉取上游 KernelBench 至 .cache/
│   ├── smoke_test.sh           # 离线冒烟 (不烧 LLM 不占 NPU)
│   ├── local/                  # 单机一键测试套件
│   │   ├── lib.sh              # 内部 helper (路径推断 + conda 激活)
│   │   ├── test-all.sh         # 一键跑全 (4 步)
│   │   ├── test-unit.sh        # cheat_detector + verify cheat-gate
│   │   ├── test-direct.sh      # NPU correctness + perf (不烧 LLM)
│   │   ├── test-skill-legit.sh # opencode skill 合法 ReLU
│   │   ├── test-skill-cheat.sh # opencode skill 反作弊 relu_cheat
│   │   ├── test-integration.sh # batch 端到端 (FULL 7 阶段 / cheap)
│   │   └── fixtures/           # cheat_detector 5 类 fixture + relu_cheat 算子
├── .cache/KernelBench/         # gitignored, 下载产物
├── .gitignore
└── README.md

pypto/.opencode/
├── agents/
│   ├── pypto-op-orchestrator.md       # 7 阶段算子开发 agent
│   └── pypto-kernel-validator.md      # 算子产物校验 subagent (本桥接层用)
└── skills/
    └── pypto-kernel-validate/
        └── SKILL.md                    # 反作弊 + 精度 + 性能验证 skill
```

> `scripts/dev/` 不再作为仓内共享内容维护; 如需跨机器联调, 请在本地自行建立
> 同名目录, `.gitignore` 已默认屏蔽该目录.

## 产物布局

```
<pypto_repo>/
└── custom/<op>/                    # pypto agent 工作流自动产物
    ├── SPEC.md                     # 由 case_loader 写入
    ├── API_REPORT.md               # pypto Stage 2 产物
    ├── DESIGN.md                   # pypto Stage 4 产物
    ├── <op>_golden.py              # pypto Stage 3
    ├── <op>_impl.py                # pypto Stage 5 (核心)
    ├── <op>_pypto_impl.py          # 给 verifier 用的 ModelNew 包装
    ├── test_<op>.py                # pypto Stage 5
    └── README.md

<report-dir>/
├── summary.json                    # 全量 JSON 结果 (含 kernelbench_commit)
├── summary.md                      # Markdown 汇总
└── <op>/
    ├── pypto_run.log               # pypto agent (opencode) 子进程 stdout/stderr
    ├── pypto_session.md            # pypto agent 完成后的 opencode 会话 Markdown 导出
    ├── verifier.log                # 验证日志 (opencode 模式: validator agent stdout;
    │                               #   direct 模式: KernelVerifier 子进程 stdout)
    ├── verifier_session.md         # validator agent 完成后的 opencode 会话 Markdown 导出
    ├── skill_report.json           # opencode 模式产物: 反作弊 + 精度 + 性能综合报告
    ├── <op>_task_desc.py           # opencode 模式 skill 输入 (task_desc 副本)
    └── result.json                 # 单 case 结构化结果 (含 skill_report 摘要)
```

## 配置

所有 CLI 选项都可通过 `configs/default.yaml` 设置默认值. 完整字段见配置文件注释.

`verifier` 子包默认把工作目录写到 `~/pypto_bench_logs/Task_<rand>/<op>/`,
可通过 CLI `--log-dir` 或环境变量 `PYPTO_BENCH_LOG_DIR` 覆盖. `verify` /
`profile` 临时脚本和源码副本默认在运行后清理; 调试时可传
`--keep-verifier-artifacts` (批处理) / `--keep-artifacts` (verifier CLI),
或设置 `PYPTO_BENCH_KEEP_ARTIFACTS=1` 保留.

## 已知约束

- pypto agent 工作流真跑 LLM, 单 case 平均耗时数分钟到 30 分钟; 默认超时 1800s.
- 仅支持 NPU/Ascend 后端 (pypto 本身只支持 NPU).
- 仅支持 PyTorch 框架 (内置 verifier 不接受 numpy / mindspore).
- `bench_dir` 需要指向包含 `KernelBench/<level>/{N}_{name}.py` 的上游数据集目录.
- 多卡并发依赖 `TILE_FWK_DEVICE_ID` 环境变量隔离, 同一时刻每卡仅 1 个 case.

## 失败排查

- **`level dir 不存在`**: 先跑 `bash scripts/download_kernelbench.sh`,
  或用 `--bench-dir` / `--level` 指向已有路径.
- **`level_dir 下找不到任何 .py 用例`**: 检查 `--bench-dir` 是否指到了
  `KernelBench/KernelBench/` 这一层 (而不是外层 `KernelBench/` 仓根).
- **opencode 子进程超时**: 加大 `--timeout-sec` (pypto 生成) 或 `--skill-timeout` (validator);
  或检查当前机器上的 LLM / opencode 配置是否可用.
- **pypto 产物缺失**: 看 `<report-dir>/<op>/pypto_run.log`, 通常是 SPEC 推导有误或 pypto 编译环境异常.
- **`opencode 可执行未找到`** (verifier_mode=opencode): 装 opencode CLI, 或临时
  `--verifier-mode direct` 跳过 LLM 语义层.
- **`skill_report.json 未产出`**: 看 `<report-dir>/<op>/verifier.log` 排查 validator agent
  的执行情况; 子进程超时 / LLM 调用失败 / SKILL 未被 discover 都会触发.
- **`final_verdict=FAIL_CHEAT`**: 看 `skill_report.json` 的 `performance.cheat_multi_kernel`
  和 `cheat_check_semantic` evidence 字段. multi-kernel 只以 runtime profile 为准;
  其它 cheat 原因包括没真正用 PyPTO / 用了占位实现 / forward 走 fallback 路径等.
- **KernelVerifier 失败 (direct 模式)**: 看 `<report-dir>/<op>/verifier.log`; 若 `ModelNew` 找不到,
  说明产物没有正确生成 `{op}_pypto_impl.py`, 由 verifier 子包的 wrapper-only fallback 兜底.
