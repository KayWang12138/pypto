# Interpreter FP4/FP8 改造说明

本文档介绍本次 `interpreter/calculator` 目录中关于 FP4/FP8 的改造内容、设计意图与影响范围。

## 1. 改造背景

在 interpreter 的 torch 计算路径中：

- 底层存储对 FP4 使用 **packed 表示**（1 byte 含 2 个 FP4）。
- 计算阶段实际需要 **逻辑 FP4 元素视图**（float32 展开后最后一维是 packed 的 2 倍）。

此前实现存在两类问题风险：

- FP4 参与运算时，`shape/stride/storageOffset` 在“逻辑视图”和“packed 存储”之间容易混用。
- FP4/FP8 转换逻辑分散在 `fp4_convert.*` 与 `fp8_convert.*`，维护成本较高。

## 2. 本次改动总览

### 2.1 转换文件合并与改名

- 新增：
  - `fp_convert.h`
  - `fp_convert.cpp`
- 删除：
  - `fp8_convert.h/.cpp`
  - `fp4_convert.h/.cpp`
- `calc_torch.cpp` 统一包含 `fp_convert.h`。

### 2.2 TensorData 语义统一（FP4）

在 `calc::Trans` 中，针对 `DT_FP4_E2M1X2/DT_FP4_E1M2X2`：

- 对 `rawShape/shape/stride/storageOffset` 做逻辑维度展开（最后一维或偏移按 FP4 逻辑元素语义处理）。
- 使 calculator 内看到的 `TensorData` 语义为“**逻辑 FP4 维度**”。

### 2.3 calc_torch 读写桥接

在 `calc_torch.cpp` 的 `From(const TensorData&)` 中：

- FP4 分支先把逻辑维度映射回 packed 视图（仅在访问原始内存时）。
- 再调用 `Fp4PackedToFloat32` 得到运算使用的 float32 逻辑视图。
- 修复了 FP4 分支被 `UInt8 -> float` 兜底分支覆盖的问题。

`ToOperand` 中：

- FP8 走 `Float32ToFp8`
- FP4 走 `Float32ToFp4Packed`
- 其他类型走直接 copy。

### 2.4 FP4 解码方式优化

FP4 解码采用 16 项映射表（nibble -> float）：

- E2M1：显式查表
- E1M2：显式查表

好处：

- 可读性高
- 便于核对规格
- 行为稳定（避免公式细节分歧）

## 3. 关键函数与职责

- `IsFp4PackedDtype(DataType)`：判定 FP4 packed 类型
- `Fp8ToFloat32 / Float32ToFp8`：FP8 编解码
- `Fp4PackedToFloat32 / Float32ToFp4Packed`：FP4 packed 编解码
- `calc::Trans`：将 FP4 的 TensorData 暴露为逻辑 shape 语义
- `calc_torch::From`：逻辑语义 -> packed 内存视图 -> float32 运算视图
- `calc_torch::ToOperand`：运算结果回写为目标 dtype（含 FP4/FP8 量化）

## 4. 测试变更

在 `test_interp_type_convert.cpp` 中新增了 FP4 用例（模仿 FP8 思路）：

- `Fp4E2M1PackedDecodeToFp32`
- `Fp4E1M2PackedDecodeToFp32`
- `Fp4E2M1AddRoundTripViaFp32`（端到端：解码参与运算 + 回写编码）

## 5. 兼容性与注意事项

- 对非 FP4 类型，shape 语义与已有逻辑保持一致。
- FP4 的内存仍是 packed 存储；“逻辑 shape”仅用于 calculator 计算语义。
- 若后续要把更上层（如 RawTensorData/Tensor 构造）也统一为逻辑 FP4 shape，需要进一步联动调整容量/对齐/stride 约束。

## 6. 建议回归项

- 类型转换：FP4/FP8 `Cast`（双向）
- 一元/二元基础算子：`Add/Sub/Mul/Neg/Sqrt`
- 形状相关算子：`Reshape/Expand/FormatND2NZ/FormatNZ2ND`
- 边界值：`+0/-0`、最小非零、最大可表示值、符号位
