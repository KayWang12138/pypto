# 常见问题与已知问题库

> **用法**：当你遇到"跑不起来/结果不对/性能不对/编译报错"时，先按本页的**快速检查清单**做一次收敛，然后按具体条目定位。

**说明：** 本页包含**常见问题**（用户可能遇到的问题及解决方案）。如需查看**已知问题**（框架已知的限制/待修复问题），请参考项目 Issue 跟踪系统或联系维护团队。

---

## 0. 快速检查清单（90% 问题先靠它收敛）

- **环境与依赖**
  - Python 版本是否 >= 3.9
  - PyTorch/torch_npu/PyPTO 的 Python 版本是否一致
  - 真实 NPU 环境是否已 `source .../set_env.sh` 或 `setenv.bash`
- **设备与运行模式**
  - `TILE_FWK_DEVICE_ID` 是否设置（真实环境）
  - 你现在是在 NPU 还是 SIM 下跑（`run_mode`）
- **输入与数据**
  - 输入是否 contiguous（很多算子/权重不支持非连续）
  - dtype 是否符合预期（BF16/FP16/FP32/int8）
  - 动态轴是否标注正确（dynamic_axis）
- **产物与日志**
  - 是否能在输出目录找到 `run.log`
  - 是否固定了输出目录（便于对比与复现）

### 0.1 快速验证与 grep 关键字（建议从这里开始）

下面给出一组“先把问题收敛到阶段/模块”的常用做法；不同版本日志关键字可能略有差异，但思路通用。

- **确认输出目录**
  - 现象：找不到 `run.log` / 不知道产物落在哪
  - 建议：在 `run.log` 或终端输出中搜索 `output`、`output_`、`result_dir`
- **确认运行模式（NPU / SIM）**
  - 建议：在 `run.log` 中搜索 `run_mode`、`SIM`、`NPU`
- **确认编译/执行大致阶段**
  - 建议：优先搜索 `OperatorBegin`、`OperatorEnd`、`GetWorkSpaceSize`、`OperatorDeviceRunOnce`
- **确认是否生成了图/IR 产物**
  - 建议：在输出目录查找 `topo.json`、`program.json`、`kernel_aicore/`、`kernel_aicpu/`
  - 产物字段与结构说明见：[输出目录与产物总览](../03-mechanisms/output-files/README.md)

---

## 1. 运行类问题（能不能跑起来）

### 1.1 未设置设备 id（NPU 场景）

- **现象**：NPU 运行报错或跑到默认设备导致不可预期
- **原因**：未设置 `TILE_FWK_DEVICE_ID` 或未设置 `torch.npu.set_device`
- **处理**：

```bash
export TILE_FWK_DEVICE_ID=0
```

以及在 Python 侧需要时：
- `torch.npu.set_device(0)`

### 1.2 已安装 torch_npu 但未安装/未配置 CANN（真实环境）

- **现象**：导入/运行异常；提示找不到 CANN 相关库或路径
- **原因**：CANN 未安装或 `set_env.sh` 未生效
- **验证方法**：检查环境变量 `LD_LIBRARY_PATH` 是否包含 CANN 库路径；在 `run.log` 中搜索 `CANN` 或 `set_env`
- **处理**：按 [环境准备与安装](../00-getting-started/01-environment-setup.md) 完成 CANN 安装并 `source set_env.sh`
- **复现最小化**：在干净环境中重新安装 CANN 并验证

### 1.3 CANN 包/ops 包版本不兼容

- **现象**：编译/运行阶段报错，表现为某些算子不可用或运行时异常
- **原因**：toolkit/ops/驱动固件版本组合不匹配
- **处理**：把版本组合回退到项目验证过的组合，并确保同一台机器上版本一致

---

## 2. 编译类问题（为什么编不过）

### 2.1 `set_vec_tile_shapes / set_cube_tile_shapes` 维度不匹配

- **现象**：tile shape 与 tensor 维度不匹配报错
- **原因**：vector tiling 维度数与张量维度数不一致，或 cube tiling 的 M/K/N 与 matmul 形状不一致
- **处理**：先用最小 shape 复现，再逐一核对每一维的含义与顺序

### 2.2 `set_xxx_tile_shapes` 最后一维 32B 对齐不满足

- **现象**：末维对齐校验报错
- **原因**：按 dtype 计算的 byte 对齐不满足 32Byte
- **处理**：调整末维 tile，使 `last_dim * bytes(dtype)` 满足 32B 对齐；必要时对输入做 padding

### 2.3 kernel 出参未写回（结果看起来“没算”）

- **现象**：输出张量保持初始值或全零
- **原因**：没有对输出 tensor 做 `out[:] = ...` 或 `y[:] = ...` 之类的写回
- **处理**：确认 kernel 内对出参显式赋值（特别是 view/assemble 路径）

### 2.4 循环中使用 Python `print()` 误解（只在编译期执行）

- **现象**：以为 print 能看到运行期张量值，但只能看到编译期信息
- **原因**：`pypto.loop` 内的 print 常为编译期行为
- **处理**：用 `run.log`/产物定位；必要时改为把关键中间结果写到输出 tensor 进行对比

---

## 3. 精度类问题（为什么结果不对）

### 3.1 view 未传入 valid_shape 导致边界精度问题

- **现象**：最后一个 tile 或边界位置误差大
- **原因**：valid_shape 未正确表达有效区域，导致边界 padding 区域参与计算/覆盖写回
- **验证方法**：对比边界 tile 的中间张量；在 `run.log` 中搜索 `valid_shape` 相关日志
- **处理**：补全 valid_shape；在最小 case 下逐步对比边界 tile
- **详见**：[精度调试（流程与方法）](07-precision-debugging.md)、[Hello World 调试文件详解](../01-examples/02-hello-world-debug.md)

### 3.2 同一个 Tensor 进行 View 和 Assemble 导致图成环

- **现象**：编译阶段提示图成环或依赖不合法
- **原因**：同一逻辑张量既被切 view 又被 assemble 回写，形成依赖环
- **处理**：拆分中间张量；避免在同一张量上形成“读写回环”

### 3.3 使用未初始化的 Tensor

- **现象**：结果随机、不稳定；同样输入多次结果不同
- **原因**：输出/中间 tensor 未初始化就参与计算
- **验证方法**：固定 seed 后多次运行，对比结果是否一致；检查 kernel 内是否对输出 tensor 显式初始化
- **处理**：确保创建后写满；必要时用 `zeros/ones/full` 初始化做定位
- **详见**：[精度调试（流程与方法）](07-precision-debugging.md)

### 3.4 量化/反量化路径精度偏差

- **现象**：与 FP16/BF16 baseline 有稳定偏差
- **原因**：int8 权重/激活量化尺度、累加精度、rounding 策略差异
- **处理**：先在 FP32 路径对齐，再逐步引入量化；对比每一步的中间张量

---

## 4. 动态形状与控制流问题

### 4.1 静态轴传入不同运行时值（或动态轴缺少标注）

- **现象**：同一算子多次执行时在第二次/某次报错或结果异常
- **原因**：编译期把某个轴当成静态，但运行时给了不同值；或应当 dynamic_axis 却没标
- **处理**：统一静态轴；必要时标注 dynamic_axis 并固定最小复现

### 4.2 父循环内跨多个子循环的 Tensor 内存不支持每次父循环迭代分配

- **现象**：复杂循环嵌套场景运行/编译异常
- **原因**：生命周期/内存分配策略不支持该模式
- **处理**：把该 tensor 的分配移动到父循环外；或改写为更明确的 buffer 复用

### 4.3 SymbolicScalar 不支持循环内自增

- **现象**：编译时报错或行为不符合预期
- **原因**：符号化标量在某些控制流形态下不支持运行时自增
- **处理**：改写循环索引策略（用 loop 的 idx 或将自增表达式移出不支持的范围）

---

## 5. 性能类问题（为什么慢）

### 5.1 Tiling 不合理（vector/cube 单元未填满）

- **现象**：吞吐低、泳道图空洞明显
- **原因**：tile 太小/太碎，导致并行度不足或搬运开销过高
- **验证方法**：查看泳道图确认空洞位置；在 `run.log` 中搜索 `tile` 或 `parallelism` 相关日志
- **处理**：先保证功能正确，再做渐进式 tiling 调参；参考 [性能优化指南](../07-features/01-performance-optimization.md)
- **详见**：[性能优化](07-features/01-performance-optimization.md)

### 5.2 过度同步/串行化

- **现象**：泳道图出现大量同步点，设备空闲
- **原因**：不必要的同步、调度策略不当、Host 侧开销过大
- **处理**：先看 `run.log` 与泳道图，确认瓶颈位置；必要时引入图捕获等策略

---

## 6. 建议：如何把问题“变小”

当你卡住时，建议按顺序做：
- 把输入缩到最小（能复现即可）
- 关掉动态轴（固定 shape）
- 关掉复杂控制流（先直线逻辑）
- 固定 dtype（先 FP32 对齐，再回到 BF16/FP16）
- 固定输出目录，把两次运行产物对齐对比


