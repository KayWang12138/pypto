# 动态 Cell Match 精度问题根修复盘

## 1. 问题根因（Root Cause）

本次精度问题（`tmp` 相关链路在 `L_STATIC` 增大、跨 `dev_task` 后出现误差/NaN）的根因不是 stitch 规则本身，而是 **workspace 预算与实际运行时需求不一致**，具体表现为：

1. `BOUNDARY_OUTCAST` 的 `slot_bytes` 由 `MaxOutcastMem()` 决定；
2. 某些动态场景下，`maxDynamicAssembleOutcastMem` 在 launch 前没有被正确更新到足够大；
3. 导致 `slot_bytes` 实际远小于真实 `required bytes`（例如日志中 `slot_bytes=1024/4096`，`required=8256`）；
4. outcast 写入越界后覆盖了后续池（尤其是动态 cell match 池）；
5. 最终体现为 cell match 表被污染、依赖建立异常、精度错乱/NaN。

简化链路：

`预算低估 -> boundary slot 太小 -> 越界写 -> 覆盖 dynamic cell table -> stitch 命中异常 -> 精度失败`

---

## 2. 修复方式（What Was Fixed）

### 2.1 运行时预算“按表达式求值 + 实际张量上界兜底”

在运行时 workspace 计算链路中，改为：

- 分别评估：
  - `maxDynamicAssembleOutcastMem`
  - `maxDynamicCellMatchTableMem`
- 使用本次真实输入/输出张量大小（`runtime_tensor_upper_bound`）做下界兜底：
  - `maxDynamicAssembleOutcastMem = max(expr_eval, runtime_tensor_upper_bound)`

涉及文件：

- `python/src/bindings/runtime.cpp`
- `framework/src/machine/runtime/device_launcher_binding.h`

### 2.2 运行时保护：`SlotMemGuard`

在分配/绑定 outcast 时增加运行时检测：若 `required > slot_bytes`，立即打印并断言，防止静默越界。

涉及文件：

- `framework/src/machine/utils/dynamic/dev_workspace.h`

### 2.3 稳定兜底（防止表达式链路失效时回退到 0）

当存在动态 cell match slot 且动态预算仍为 0 时，保留保守兜底（避免再次出现零预算）。

涉及文件：

- `framework/src/machine/runtime/device_launcher.h`

---

## 3. 修复前后变化（Before vs After）

## 修复前

- 常见日志特征：
  - `slot_bytes=1024` 或 `slot_bytes=4096`
  - `required=8256`
  - 高频 `SlotMemGuard` 告警
- 行为：
  - boundary outcast 槽不足，发生越界与池覆盖
  - dynamic cell match 表被破坏
  - 跨 task 场景出现 mismatch/NaN

## 修复后

- 动态预算在运行时按真实输入求值（并有兜底）；
- boundary slot 按可用上界放大，避免 outcast 越界写；
- dynamic cell match 表内存不再被边界 outcast 覆盖；
- `minimal_608_embedding_tmp_sum_case.py`、`only.py` 精度恢复为 0 mismatch。

---

## 4. 关键日志证据（本地回归）

### 4.1 `issue/minimal_608_embedding_tmp_sum_case.py`

- `Allocator/BoundaryOutcast ... slot_bytes=32768`
- `[DIFF] d_emb: mismatch=0/2064`
- `[Workspace/Fallback] runtime_tensor_upper_bound=8256 ...`

### 4.2 `issue/only.py`

- `Allocator/BoundaryOutcast ... slot_bytes=32768`
- `tmp->d_emb: d_emb=OK`
- `[DIFF] d_emb: mismatch=0/2064`

### 4.3 `issue/608.py`

- 运行时预算有动态上推：
  - `runtime_tensor_upper_bound` 从 `2MB -> 4MB -> 8MB` 逐步提升
- 当前日志末尾仍出现 `exception invalid error`（非本次精度根因链路本身，建议单独作为稳定性问题继续跟踪）。

---

## 5. 结论

本次问题的“根本修复点”是：**把 boundary outcast 的预算来源从“可能失效的静态/未更新值”，改为“运行时表达式求值 + 实际输入输出上界”**，并以 `SlotMemGuard` 做硬保护，阻断越界写导致的池污染。

精度层面（`tmp` 动态链路）已恢复；后续建议对 `608.py` 的运行时异常单独开稳定性问题，避免与本次精度根因混淆。

