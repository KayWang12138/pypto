# Stage Gates

## Gate 原则

- 没有 `spec.md` → 不能进入 Stage 2
- 没有 `design.md` 和 `{op}_golden.py` → 不能进入 Stage 3
- Stage 3 未产出 3 文件（`test_{op}.py` + `{op}_impl.py` + `README.md`） → 不能判定精度
- 精度失败 → 不能进入性能阶段（Stage 5）

## Stage 3 判定逻辑

Stage 3（功能实现）关心两件事：实现能跑 + 精度是否正确。

### 前提约束

`test_{op}.py` 必须使用 `numpy.testing.assert_allclose` 进行精度对比，不得使用手写 `assert max_diff < tolerance`。这是 orchestrator 能通过错误类型区分"运行失败"和"精度失败"的基础。

### 判定方式

| 条件 | 检测方式 |
|------|----------|
| `test_{op}.py` 存在 | 文件检查 |
| `{op}_impl.py` 存在 | 文件检查 |
| `README.md` 存在 | 文件检查 |
| 脚本执行结果 | 检查退出码和输出内容 |

### 判定结果

- 脚本 exit 0 → **Stage 3 PASS**（精度通过 → 进入 Stage 5 性能采集）
- 脚本输出包含 `Not equal to tolerance` 关键字 → **Stage 3 部分通过**（实现能跑但精度不通过 → 进入 Stage 4 精度定位）
- 脚本因其他错误退出（无 `Not equal to tolerance`） → **Stage 3 FAIL**（实现有问题 → Stage 3 内重试修复）

### 关键字检测说明

`numpy.testing.assert_allclose` 抛出的 `AssertionError` 包含 `Not equal to tolerance` 关键字。orchestrator 检查脚本的 stderr/stdout 中是否包含此关键字来区分"运行失败"和"精度失败"。

## Stage 2 串行约束

Stage 2A（golden-generator）必须在 Stage 2B（op-design）之前完成。design 生成时参考 golden 代码结构和函数签名。
