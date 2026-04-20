---
name: pypto-kernel-validate
description: 校验一个声称由 PyPTO 开发的算子产物 — 反作弊 (脚本机械检测 + LLM 语义审阅) 加精度与性能验证, 输出统一 JSON 报告. 调用方: 任何在 cwd 为 PyPTO 仓的 opencode agent.
---

# PyPTO Kernel Validate Skill

> 给定一个算子产物目录, 判定该实现是否 (1) 真正使用 PyPTO 而非作弊, (2) 精度通过, (3) 性能符合要求.
> 整套流程以 **JSON 报告** 结束, 不在对话中堆叠人类总结.

## 适用场景

任意 agent 工作流生成 PyPTO 算子之后, 在交付前调用本 skill 做最终把关. 典型调用方:

- KernelBench 桥接层 (`pypto/integration/akg_bench/`) 在 verify 阶段.
- 外部团队的算子 agent, 把产物丢给本 skill 做合规校验.

## 输入约定 (由调用方在 prompt 中传入)

| 字段 | 必需 | 说明 |
|---|---|---|
| `op_name` | 是 | 算子名 (与产物文件 `<op>_impl.py` / `<op>_pypto_impl.py` 前缀一致) |
| `op_dir` | 是 | 算子产物目录绝对路径 |
| `task_desc_file` | 是 | KernelBench 风格 task_desc.py 绝对路径 (含 `Model` / `get_inputs` / `get_init_inputs`) |
| `output_dir` | 是 | 报告输出目录, skill 会写出 `skill_report.json` |
| `mode` | 否 | `correctness` (默认) / `performance` / `full` |
| `device_id` | 否 | NPU 卡号, 默认 0 |
| `arch` | 否 | 默认 `ascend910b4` |
| `verify_timeout` | 否 | 单次验证子进程超时, 秒, 默认 300 |

## 工作流 (必须严格按以下 4 步执行, 不得跳步)

### Step 1: 脚本机械检测 (cheat_detector)

调用统一 verifier CLI 的 `cheat-check` 子命令:

```bash
python -m pypto.integration.akg_bench.verifier cheat-check \
  "<op_dir>" \
  --op-name "<op_name>" \
  --json-out "<output_dir>/cheat_check_script.json"
```

读 `cheat_check_script.json`, 关注 `verdict` 与 `checks[*]`:

- `verdict == "cheat"` —— 已有铁证, 仍需要继续 Step 2 (你要给出语义层独立佐证), 但 Step 3 必须 `--no-cheat-gate` 才会跑, 默认跳过.
- `verdict == "suspicious"` —— 软警告, 你要在 Step 2 重点核查这些项.
- `verdict == "pass"` —— 机械层 OK, Step 2 可正常进行.

机械检测覆盖范围 (你不必在 Step 2 重复同样的工作, 但要交叉印证):

- `import_pypto`: 是否存在 `import pypto` / `from pypto`.
- `has_jit`: 是否至少存在一个 `@pypto.jit` 装饰器或 `pypto.jit(...)` 调用.
- `jit_def_count`: 是否只装饰了 1 个 jit 函数 (要求 ≤1; >1 为铁证).
- `jit_func_call_count`: 函数式 `pypto.jit(fn)` 是否只出现 ≤1 次.
- `forward_jit_calls`: ModelNew.forward 中调 jit 函数次数 (启发).
- `swimlane_kernel_dirs`: 若已跑过 profile, kernel trace 子目录数 ≤1.
- `forbidden_text_patterns`: 是否出现 testing-only / fallback / workaround / TODO use pypto / bypass pypto 等可疑文本.

### Step 2: 语义审阅 (你必须亲自完成 — 这是本 skill 的核心价值)

机械检测能抓最显眼的作弊形态, 但很多隐性作弊依赖**语义理解**, 必须由你逐行读源码后判断.

**强制要求**: 用 `read` 工具打开以下文件, 至少各读完一遍:

1. `<op_dir>/<op>_impl.py`
2. `<op_dir>/<op>_pypto_impl.py` (如存在)

然后**逐项**对照下表做判定. 对每项给出 PASS / FAIL / SUSPICIOUS, 并指明依据 (引用具体行号或代码片段).

| # | 隐性作弊形态 | 识别要点 |
|---|---|---|
| S1 | **空壳 jit kernel** —— `@pypto.jit` 装饰器装在一个 pass-through 函数上, 函数体只 `return torch.xxx(x)` 或直接 `return x` | 看被装饰函数的 body, 是否真有 `pypto.dsl.*` / pypto 算子调用; 全 `torch.*` 即作弊 |
| S2 | **forward 双路径** —— `forward` 中 `if cond: return pypto_path(...) else: return torch_path(...)` 评测时走 pypto, 默认走 torch | 找 forward 中的 if/else / mode flag / 环境变量分支 |
| S3 | **try/except fallback** —— 用 try 包 jit 调用, 失败直接 fallback 到 torch native | 找 try/except 包 jit 调用的代码块 |
| S4 | **绕过 jit 的 pre/post-process** —— 把核心计算放在 jit 前后的 numpy/torch 代码里, jit 只做无关紧要的一小步 | 看 forward / wrapper 在 jit 调用前后还做了哪些计算; 只允许 to-device / reshape / dtype cast / 简单 broadcast |
| S5 | **多 kernel 拆分** —— 即便 `jit_def_count == 1`, 如果代码里 `for ... range(n): kernel(x)` 循环调用 jit, 也是作弊 (一个算子应一次 forward 一次 jit) | 看 forward 中是否在循环里反复调 jit kernel |
| S6 | **mock kernel** —— jit 函数只写 `pass` / `return None` / 返回 dummy tensor; 实际计算在 wrapper 里 torch 完成 | 看 jit 函数体是否真有意义的运算 |
| S7 | **shape 硬编码作弊** —— jit 函数硬编码 input shape, 只能跑 KernelBench 的特定 case, 实质是把 baseline 复制了一遍 | 看 jit 函数对 shape 是参数化还是写死;  写死也算作弊 |
| S8 | **tile config 关闭核心算子** —— 通过 PyPTO config 把核心算子改成 noop / identity, 让评测路径不实际算 | 看 wrapper / __init__ / forward 是否对 pypto 全局配置做特殊修改 |
| S9 | **可疑注释** —— 代码注释提示 "for testing", "use pypto later", "fallback", "workaround", "demo only" 等 | 通读注释; 与 Step 1 的 `forbidden_text_patterns` 交叉印证 |

把审阅结论组织成 `cheat_check_semantic` 字段, 至少包括:

```json
{
  "verdict": "pass | suspicious | cheat",
  "items": [
    {"id": "S1", "status": "pass", "evidence": "..."},
    {"id": "S2", "status": "fail", "evidence": "<op>_impl.py L45-L50: forward 中存在 USE_PYPTO env flag 分支..."},
    ...
  ],
  "reasoning": "<总结性论述, 200-500 字>"
}
```

> **底线原则**: 拿不准的项一律标 `suspicious` 而非 `pass`, 把不确定性写在 `evidence` / `reasoning` 里. 宁可让调用方多核一遍, 不要错放作弊产物过关.

### Step 3: 精度 + 性能验证 (调统一 verifier CLI)

仅在 Step 1 与 Step 2 综合判定不是 `cheat` 时执行. 调:

```bash
python -m pypto.integration.akg_bench.verifier verify \
  "<op_dir>" \
  --op-name "<op_name>" \
  --task-desc "<task_desc_file>" \
  --mode "<mode>" \
  --arch "<arch>" \
  --device-id <device_id> \
  --verify-timeout <verify_timeout> \
  --json-out "<output_dir>/verify_run.json"
```

注意:

- 该 CLI 内部会再跑一次 `cheat_detector` 作为 gate, 这是设计冗余, 不影响 Step 1/2 已得出的结论.
- 若 `mode` 为 `performance` 或 `full`, 同时检查 `verify_run.json.performance.cheat_multi_kernel`: 该字段为 true 表示运行时 swimlane 抓到了多 kernel, 你必须把综合 verdict 升级为 `cheat`.
- CLI 退出码: `0` 通过, 非零失败. 但**不要依赖**退出码做判定, 一律以 JSON 内的 `verdict_machine` / `correctness.status` / `performance.status` 为准.

### Step 4: 综合报告 (写入 skill_report.json)

把三方结论合并写到 `<output_dir>/skill_report.json`. **必须** 用 `write` 工具落盘, 不要只在 chat 里输出.

报告结构:

```json
{
  "op_name": "...",
  "op_dir": "...",
  "task_desc_file": "...",
  "mode": "...",
  "skill_version": "1.0",
  "cheat_check_script": { /* Step 1 cheat_check_script.json 全文 */ },
  "cheat_check_semantic": {
    "verdict": "pass | suspicious | cheat",
    "items": [...],
    "reasoning": "..."
  },
  "correctness": { /* verify_run.json.correctness, status 之外补 reasoning */ },
  "performance": { /* verify_run.json.performance, 同上 */ },
  "final_verdict": "PASS | FAIL_CHEAT | FAIL_CORRECTNESS | FAIL_PERFORMANCE | ERROR",
  "final_reasoning": "<60-200 字的综合判定论述, 引用以上字段做依据>"
}
```

`final_verdict` 决定规则 (按优先级从上到下):

1. `cheat_check_script.verdict == "cheat"` 或 `cheat_check_semantic.verdict == "cheat"` 或 `performance.cheat_multi_kernel == true` → `FAIL_CHEAT`
2. `correctness.status != "passed"` → `FAIL_CORRECTNESS`
3. mode 含性能且 `performance.status == "failed"` 或 `"error"` → `FAIL_PERFORMANCE`
4. `cheat_check_script` 或 `cheat_check_semantic` 任一为 `suspicious` → 仍 `PASS`, 但 `final_reasoning` 必须明确列出未消的 suspicious 项, 让调用方自行评估
5. 其余 → `PASS`
6. 任何 step 因技术性原因失败 (CLI 异常, 文件不可读, etc.) → `ERROR`

## 输出契约

- 唯一刚性产物: `<output_dir>/skill_report.json`
- 中间产物可选保留: `cheat_check_script.json`, `verify_run.json` (推荐落盘, 便于排错)
- chat 中只回一行: `skill_report.json written: <path>; final_verdict=<verdict>`. 不重复 JSON 内容, 不堆叠人类总结.

## 失败处理

| 现象 | 处置 |
|---|---|
| Step 1 CLI 非零退出但 JSON 已写 | 继续 Step 2, JSON 内的 verdict 已是 `cheat` |
| Step 3 CLI 异常崩溃, 没写 JSON | `final_verdict = ERROR`, 在 `final_reasoning` 中粘贴异常文本 (≤500 字) |
| `op_dir` 缺源文件 | 直接 `final_verdict = ERROR`, 不进入 Step 2/3 |
| Step 2 你拿不准任何一项 | 该项标 `suspicious`, 综合 verdict 取 `suspicious`; final_verdict 仍可 PASS, 但要在 reasoning 里点名 |

## 你绝对不能做的事

- 不能跳过 Step 2 直接采信 Step 1 的脚本结论 (脚本只能抓机械层, 这正是你存在的理由).
- 不能修改 `<op_dir>` 下的任何文件 (你是审阅者, 不是修复者).
- 不能在 chat 里给"通过"的口头结论而不写 `skill_report.json`.
- 不能为了让 verdict 看起来"友好"而把 cheat 降级为 suspicious / pass; 严格遵守上面 final_verdict 的优先级规则.
