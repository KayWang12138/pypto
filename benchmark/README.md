# test-integration.sh 使用说明

`benchmark/scripts/local/test-integration.sh` 用于批量运行
KernelBench -> PyPTO 7 阶段算子生成 -> verifier -> report 的端到端集成测试。

内部设计、目录结构和调试细节见 [Develop.md](Develop.md)。

## 前置条件

运行前需要确认:

1. 已安装并可 import `pypto`。
2. 已安装 opencode CLI, 且 `which opencode` 可找到。
3. 已配置可用的 LLM / opencode 运行环境。
4. 已设置 NPU / CANN / `torch_npu` 环境。
5. 已下载 KernelBench 数据集:

```bash
bash benchmark/scripts/download_kernelbench.sh
```

## 基本用法

在 pypto 仓根执行:

```bash
bash benchmark/scripts/local/test-integration.sh
```

默认行为:

- `FULL=1`: 真跑 PyPTO 7 阶段生成和 verifier。
- `CASES=19_ReLU`: 默认只跑一个 ReLU case。
- `VERIFIER_MODE=opencode`: verifier 走 opencode skill。
- `MODE=performance`: 跑 correctness + performance。
- `CONCURRENCY=1`: 单 case 串行。
- `BENCHMARK_LOG_DIR=/tmp/benchmark_test_<timestamp>`: 自动创建日志目录。

快速复用已有 PyPTO 产物, 只测 verifier:

```bash
FULL=0 bash benchmark/scripts/local/test-integration.sh
```

## 常用环境变量

`test-integration.sh` 通过环境变量配置运行参数:

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `CASES` | case 选择器。支持单 level 或多 level | `19_ReLU`, `1:21`, `level1=1:21;level2=31,41:50` |
| `FULL` | `1` 真跑 PyPTO 生成; `0` 复用已有产物 | `FULL=1` |
| `DEVICE` | 单卡设备号, 脚本会映射为 `TILE_FWK_DEVICE_ID` | `DEVICE=3` |
| `CONCURRENCY` | 并发 case 数 | `CONCURRENCY=5` |
| `BENCHMARK_LOG_DIR` | report、batch log、monitor state 输出目录 | `/data/x00952168/pypto` |
| `OPENCODE_MODEL` | opencode 使用的模型 | `alibaba-cn/glm-5` |
| `PYPTO_TIMEOUT` | PyPTO 生成阶段超时时间, 单位秒 | `10800` |
| `SKILL_TIMEOUT` | verifier opencode skill 超时时间, 单位秒 | `1500` |
| `SKILL_RETRY` | verifier skill 重试次数 | `2` |
| `SKILL_RETRY_INTERVAL` | verifier skill 重试间隔, 单位秒 | `600` |
| `VERIFY_RTOL` | verifier 精度相对误差阈值 | `1e-2` |
| `VERIFY_ATOL` | verifier 精度绝对误差阈值 | `2.5e-2` |
| `VERIFIER_MODE` | `opencode` 或 `direct` | `opencode` |
| `SKIP_STAGE7_PERF_TUNE` | 是否跳过 PyPTO Stage 7 性能调优 | `1` |

多 level 时, 不使用 `LEVELS` / `LEVEL`; 直接把 level 写进 `CASES`:

```bash
CASES='level1=1:21;level2=31,41:50' \
bash benchmark/scripts/local/test-integration.sh
```

## 前台运行示例

下面示例按指定 case 集合在设备 3 上运行, 使用 `alibaba-cn/glm-5`, 并把报告写到
`/data/x00952168/pypto`:

```bash
CASES='level1=3,16,19,20,23,26,36,40,46,49,51,90,92,95,100;level2=9,12,18,28,33,37,45,51,56,62,66,75,81,88,98;level3=1,6,21,33,43' \
FULL=1 \
DEVICE=3 \
OPENCODE_MODEL=alibaba-cn/glm-5 \
VERIFY_RTOL=1e-2 \
VERIFY_ATOL=2.5e-2 \
CONCURRENCY=5 \
PYPTO_TIMEOUT=10800 \
BENCHMARK_LOG_DIR=/data/x00952168/pypto \
bash benchmark/scripts/local/test-integration.sh
```

## 后台运行 (nohup)

长批量任务建议用固定 `BENCHMARK_LOG_DIR` 后台运行:

```bash
RUN_DIR=/data/x00952168/pypto
mkdir -p "${RUN_DIR}"

nohup env \
  CASES='level1=3,16,19,20,23,26,36,40,46,49,51,90,92,95,100;level2=9,12,18,28,33,37,45,51,56,62,66,75,81,88,98;level3=1,6,21,33,43' \
  FULL=1 \
  DEVICE=3 \
  OPENCODE_MODEL=alibaba-cn/glm-5 \
  VERIFY_RTOL=1e-2 \
  VERIFY_ATOL=2.5e-2 \
  CONCURRENCY=5 \
  PYPTO_TIMEOUT=10800 \
  BENCHMARK_LOG_DIR="${RUN_DIR}" \
  bash benchmark/scripts/local/test-integration.sh \
  >"${RUN_DIR}/nohup.log" 2>&1 &
```

查看后台日志:

```bash
tail -f /data/x00952168/pypto/nohup.log
tail -f /data/x00952168/pypto/batch.log
```

如果需要在同一个后台命令里激活 conda 或 source CANN 环境, 用 `bash -lc`:

```bash
RUN_DIR=/data/x00952168/pypto
mkdir -p "${RUN_DIR}"

nohup bash -lc '
  source /path/to/cann/set_env.sh
  CASES="level1=3,16,19,20,23,26,36,40,46,49,51,90,92,95,100;level2=9,12,18,28,33,37,45,51,56,62,66,75,81,88,98;level3=1,6,21,33,43" \
  FULL=1 \
  DEVICE=3 \
  OPENCODE_MODEL=alibaba-cn/glm-5 \
  VERIFY_RTOL=1e-2 \
  VERIFY_ATOL=2.5e-2 \
  CONCURRENCY=5 \
  PYPTO_TIMEOUT=10800 \
  BENCHMARK_LOG_DIR="'"${RUN_DIR}"'" \
  bash benchmark/scripts/local/test-integration.sh
' >"${RUN_DIR}/nohup.log" 2>&1 &
```

环境变量位置要注意:

- `nohup FULL=1 bash ...` 是错误写法, `nohup` 会把 `FULL=1` 当成命令名。
- 可以写 `FULL=1 nohup bash benchmark/scripts/local/test-integration.sh ...`。
- 更推荐写 `nohup env FULL=1 ... bash benchmark/scripts/local/test-integration.sh ...`。
- 使用 `bash -lc` 时, 变量要放进引号内那条真正执行的命令中。

## 输出位置

脚本会在 `BENCHMARK_LOG_DIR` 下写:

```text
<BENCHMARK_LOG_DIR>/
├── batch.log
├── logs/
├── monitor_state/
└── report/
    ├── summary.json
    ├── summary.md
    └── <level>/<op>/result.json
```

常用查看命令:

```bash
cat "${BENCHMARK_LOG_DIR}/report/summary.md"
cat "${BENCHMARK_LOG_DIR}/report/summary.json"
```

单 case 的详细日志在:

- `report/<level>/<op>/pypto_run.log`
- `report/<level>/<op>/pypto_session.md`
- `report/<level>/<op>/verifier.log`
- `report/<level>/<op>/verifier_session.md`
- `report/<level>/<op>/result.json`

## 常见问题

- **想快速检查 verifier**: 用 `FULL=0` 复用已有 `custom/<op>/` 产物。
- **不想走 LLM verifier**: 设置 `VERIFIER_MODE=direct`。
- **PyPTO 生成超时**: 增大 `PYPTO_TIMEOUT`。
- **validator 超时**: 增大 `SKILL_TIMEOUT`, 或调整 `SKILL_RETRY` / `SKILL_RETRY_INTERVAL`。
- **多 level case 选择错误**: 检查 `CASES` 是否写成
  `level1=...;level2=...` 形式。
- **NPU 设备不对**: 优先设置 `DEVICE=<id>`; 脚本会导出
  `TILE_FWK_DEVICE_ID=<id>`。
