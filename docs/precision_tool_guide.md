# PyPTO 精度工具使用说明

## 1. 工具概述

PyPTO 精度工具是一个用于调试和验证算子计算精度的强大工具，通过在主机 CPU 上模拟计算过程，并与基准数据进行对比，帮助开发者快速定位精度问题。

**核心功能：**
- **Tensor Graph验证**：验证用户编写的计算逻辑是否正确（pass变换前的graph验证）
- **Pass阶段验证**：验证各 Pass 处理是否正确，检测异常计算节点
- **中间结果分析**：保存和打印Tensor Graph阶段的模拟结果，支持人工分析
- **上板Dump**：获取NPU实际执行的leaf function输入输出数据，与CPU模拟结果对比

## 2. 核心组件架构

### 2.1 FlowVerifier（流程验证器）
- **位置**：`framework/src/interface/interpreter/flow_verifier.h/cpp`
- **主要功能**：
  - `VerifyTensorGraph()` - 验证前端计算图
  - `VerifyPass()` - 验证各个 Pass 阶段
  - `CompareData()` - 逐元素数据比较
  - `CompareResult` - 比较结果管理

### 2.2 Calc（计算操作层）
- **位置**：`framework/src/interface/interpreter/calc.h`
- **主要功能**：
  - 提供所有算术操作的统一接口（Add, Sub, Mul, Div, Exp, Sqrt等）
  - 数据类型转换和内存管理
  - 支持向量化、立方体操作

### 2.3 FunctionInterpreter（函数解释器）
- **位置**：`framework/src/interface/interpreter/function.h`
- **主要功能**：
  - 执行计算图的 CPU 模拟
  - 管理数据视图和内存布局
  - 控制流执行和数据捕获

### 2.4 Calculator（计算器后端）
- **位置**：`framework/src/interface/interpreter/calculator/`
- **主要功能**：
  - 使用 PyTorch 作为计算后端
  - 支持所有数据类型（FP32, FP16, BF16, INT8等）
  - 提供高精度数值计算

## 3. 使用方法

### 3.1 Tensor Graph验证模式

**适用场景**：验证用户编写的计算逻辑是否正确，基于golden数据验证整体计算正确性。

**步骤**：

1. **设置 golden 数据**
```python
verify_options = {
    "enable_pass_verify": True,
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(...):
    # 算子实现
    ...

def test():
    # 准备输入数据
    input_data = torch.rand(shape, dtype=torch.float)
    
    # 计算 golden 结果
    golden_output = reference_implementation(input_data)
    
    # 设置 golden 数据
    pypto.set_verify_golden_data(goldens=[None, None, golden_output])
    
    # 执行算子
    output = your_kernel(input_data)
```

2. **执行并查看结果**
```bash
python3 your_test.py
```

输出示例：
```
2025-12-17 22:18:49.013 V | tensor_graph Verify for 3 data view list index 0 result NO_COMPARE
2025-12-17 22:18:49.014 V | tensor_graph Verify for 3 data view list index 1 result NO_COMPARE
2025-12-17 22:18:49.015 V | tensor_graph Verify for 3 data view list index 2 result PASS
```

3. **分析结果文件**
执行后在 `{work_path}/output/output_*/verify_*/` 目录下生成：
- `tensor_graph/` - 前端初始计算图数据
- `verify_result.csv` - 详细验证报告

### 3.2 Pass阶段验证模式

**适用场景**：算子精度出现问题但没有明确方向时，排除 Pass 处理阶段是否引入错误。

**步骤**：

1. **开启精度调试开关**
```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_error_tol": [1e-3, 1e-6],  # [rtol, atol]
    "pass_verify_filter": ["all"]  # 或指定具体Pass名称
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(...):
    # 算子实现
    ...
```

2. **执行用例**
```bash
python3 your_test.py
```

3. **查看输出结果**
```
2025-12-17 22:18:49.013 V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_00_RemoveRedundantReshape Verify result PASS
2025-12-17 22:18:49.014 V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_01_AutoCast Verify result PASS
...
2025-12-17 22:18:49.020 V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_15_CodegenPreproc Verify result FAILED
```

4. **分析结果文件**
执行后在 `{work_path}/output/output_*/verify_*/` 目录下生成：
- `tensor_graph/` - 前端初始计算图数据
- `verify_result.csv` - 详细详细验证报告
- `{FUNC_NAME}.pass_{SEQ}_{NAME}/` - 各 Pass 的中间数据

### 3.3 中间结果分析模式

**适用场景**：需要检查特定中间结果或进行深度分析时。

**重要说明**：
- `pass_verify_print` 和 `pass_verify_save` 保存的是 **tensor graph 验证阶段模拟计算的结果**
- 这些结果是在主机 CPU 上通过 FunctionInterpreter 模拟执行计算图得到的
- **与实际在 NPU 上板执行的结果可能存在差异，主要用于算法逻辑验证**
- 这是pass变换前的graph验证，用于验证用户编写的计算逻辑是否正确

**步骤**：

1. **在代码中插入调测点**
```python
verify_options = {
    "enable_pass_verify": True,
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(input0, input1):
    # 保存中间结果到文件（保存的是Tensor Graph阶段CPU模拟计算结果）
    pypto.pass_verify_save(input0, "input0_debug")
    
    # 打印中间结果到控制台（打印的是Tensor Graph阶段CPU模拟计算结果）
    pypto.pass_verify_print(input1)
    
    result = input0 + input1
    pypto.pass_verify_save(result, "result_debug")
    
    return result
```

2. **执行用例**
```bash
python3 your_test.py
```

3. **查看输出**
控制台输出：
```
input1:<64x64xFP16/64x64xFP16>
[[0.03955 0.6094 0.1519 ... 0.7339 0.8789 0.8662]
 [0.6284 0.01465 0.6333 ... 0.2422 0.03516 0.8423]
 ...]
```

文件输出：
```
{work_path}/output/output_*/tensor/
├── input0_debug.data     # 二进制数据（Tensor Graph阶段CPU模拟计算结果）
├── input0_debug.csv      # 元数据（shape, dtype等）
├── result_debug.data
└── result_debug.csv
```

### 3.4 上板Leaf Func输入输出Dump

**适用场景**：需要获取实际在 NPU 上板执行的 leaf function 的输入输出数据时，用于与 CPU 模拟结果对比分析精度问题。

**重要说明**：
- 上板 dump 获取的是 **实际在 NPU AI Core 上执行的真实数据**
- 与 `pass_verify_save` 的 CPU 模拟结果不同，上板 dump 反映的是真实的硬件执行结果
- 主要用于对比 CPU 模拟结果与 NPU 实际执行结果的差异

**关键组件**：
- **Dump实现**：`framework/src/machine/device/dump/aicore_dump.h`
- **解析工具**：`tools/verifier/parse_dump_tensors.py`

**Dump数据结构**：
```cpp
struct DumpTensorInfo {
    uint32_t headSize;        // 头部大小
    uint32_t funcId;          // 函数ID
    uint32_t taskId;          // 任务ID
    uint32_t callopMagic;     // CallOp魔术字
    int32_t coreId;          // AI Core ID
    int32_t dataType;        // 数据类型
    int32_t rawMagic;        // Raw Tensor魔术字
    int32_t dims;            // 维度数量
    int64_t exeStart;        // 执行开始时间
    int64_t exeEnd;          // 执行结束时间
    uint64_t rootHash;       // Root函数哈希
    uint64_t funcHash;        // 函数哈希
    uint64_t timeStamp;       // 时间戳
    uint64_t shape[5];       // 形状信息
    uint64_t offset[5];      // 偏移信息
    uint64_t rawShape[5];    // 原始形状
    uint64_t tensorAddr;      // Tensor地址
};
```

**使用步骤**：

1. **启用上板Dump**
   通过环境变量或配置启用dump功能（具体配置方式请参考相关文档）

2. **执行算子**
   ```bash
   python3 your_test.py
   ```

3. **查看Dump文件**
   执行后在 `output/dump_tensor/device_X/` 目录下生成：
   ```
   output/dump_tensor/device_0/
   ├── taskId_seqNo_callopMagic_rootHash_funcHash_rawMagic_timeStamp_dataType_input0.tdump
   ├── taskId_seqNo_callopMagic_rootHash_funcHash_rawMagic_timeStamp_dataType_input1.tdump
   ├── taskId_seqNo_callopMagic_rootHash_funcHash_rawMagic_timeStamp_dataType_output0.tdump
   └── ...
   ```

4. **解析Dump数据**
   ```bash
   python3 tools/verifier/parse_dump_tensors.py \
       --dump_tensor_path output/dump_tensor/device_0 \
       --verify_path output/output_*/verify_*/
   ```

5. **查看解析结果**
   解析工具会生成 `tensor_info.csv` 文件，包含所有dump张量的详细信息：
   - 张量基本信息（shape, dtype, offset等）
   - 执行时间信息（exeStart, exeEnd, exeDuration）
   - 与verify结果的对比（如果提供了verify_path）
   - 合并的raw tensor数据

**解析工具功能**：
- **解析.tdump文件**：解析二进制的dump数据结构
- **数据提取**：提取tensor元数据和实际数据
- **自动对比**：与verify结果自动对比，标记精度差异
- **数据合并**：将分片的tensor数据合并为完整的raw tensor
- **多进程处理**：支持并行处理大量dump文件

**输出字段说明**：
| 字段 | 说明 |
|------|------|
| taskId | 任务ID |
| callopMagic | CallOp唯一标识 |
| funcId | 函数ID |
| coreId | AI Core ID |
| dataType | 数据类型（数值和字符串） |
| shape | 有效形状 |
| offset | 在raw tensor中的偏移 |
| rawShape | 原始形状 |
| exeStart | 执行开始时间 |
| exeEnd | 执行结束时间 |
| exeDuration | 执行持续时间 |
| bin_file | 数据文件路径 |
| verify_tensor_file | 对应的verify数据文件（如果存在） |
| cmp_res | 对比结果（True/False/NO_CMP） |

**与CPU模拟结果对比**：
```python
# 读取上板dump数据
import numpy as np
dump_data = np.fromfile("output/dump_tensor/device_0/xxx_output0.data", dtype=np.float32)

# 读取CPU模拟结果
cpu_data = np.fromfile("output/output_*/verify_*/tensor_graph/xxx.data", dtype=np.float32)

# 对比差异
diff = np.abs(dump_data - cpu_data)
max_diff = np.max(diff)
print(f"最大差异: {max_diff}")
```

## 4. 配置选项详解

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_pass_verify` | bool | False | 启用精度验证 |
| `pass_verify_save_tensor` | bool | False | 保存中间张量数据 |
| `pass_verify_error_tol` | list | [1e-3, 1e-6] | [相对误差, 绝对误差] |
| `pass_verify_filter` | list | ["all"] | 要验证的Pass列表 |
| `pass_verify_print` | - | - | 打印张量数据 |
| `pass_verify_save` | - | - | 保存张量到文件 |

## 5. 结果分析

### 5.1 verify_result.csv 文件结构

| 字段 | 说明 |
|------|------|
| No. | Pass内数据顺序编号 |
| rootFuncID | Root Function唯一标识 |
| funcID | Function节点唯一标识 |
| verifyType | Pass名称 |
| LoopInfo | 控制流信息 |
| opCode | Tile算子名称 |
| tensorMagic | Tensor唯一标识 |
| inputShape/outputShape | 输入/输出形状 |
| inputDtype/outputDtype | 输入/输出数据类型 |
| verifyResult | 验证结果（PASS/FAILED/NO_COMPARE） |
| maxAbsDiff | 最大绝对误差 |
| maxRelDiff | 最大相对误差 |
| errorCount | 错误数量 |
| errorRatio | 错误比例 |

### 5.2 误差判定规则

工具使用以下规则判定精度是否合格：

```cpp
//1. 统计接近零的数据
N_zero = count(|output| <= 1e-6 && |golden| > 1e-6)

//2. 计算相对误差
relDiff = |output - golden| / (|output| + |golden|) * 2

//3. 判定阈值
tol_attn = (routput| + |golden|) * rtol / 2 + atol

//4. 通过条件
通过 = (N_zero <= 1000) && (errorRatio <= rtol)
```

### 5.3 CPU模拟结果与上板Dump结果对比

**对比目的**：
- CPU模拟结果：通过 `pass_verify_save` 保存，反映算法逻辑正确性
- 上板Dump结果：通过 `aicore_dump` 机制获取，反映NPU硬件执行结果
- 对比两者差异，定位精度问题来源（算法逻辑 vs 硬件执行）

**对比步骤**：

1. **获取CPU模拟结果**
   ```python
   # 在代码中使用pass_verify_save保存
   pypto.pass_verify_save(intermediate_result, "cpu_result")
   
   # 读取保存的数据
   import numpy as np
   cpu_data = np.fromfile("output/output_*/tensor/cpu_result.data", dtype=np.float32)
   ```

2. **获取上板Dump结果**
   ```bash
   # 启用上板dump并执行算子
   python3 your_test.py
   
   # 解析dump数据
   python3 tools/verifier/parse_dump_tensors.py \
       --dump_tensor_path output/dump_tensor/device_0
   ```

3. **对比分析**
   ```python
   # 读取dump数据
   dump_data = np.fromfile("output/dump_tensor/device_0/xxx_output0.data", dtype=np.float32)
   
   # 对比差异
   diff = np.abs(cpu_data - dump_data)
   max_diff = np.max(diff)
   mean_diff = np.mean(diff)
   
   print(f"CPU模拟 vs 上板Dump:")
   print(f"  最大差异: {max_diff}")
   print(f"  平均差异: {mean_diff}")
   print(f"  差异比例: {np.mean(diff > 1e-3) * 100:.2f}%")
   ```

**差异分析**：
- **差异很小（<1e-6）**：算法逻辑和硬件执行一致，问题可能在其他地方
- **差异中等（1e-6 ~ 1e-3）**：可能存在数据类型转换或精度损失
- **差异较大（>1e-3）**：可能存在算法实现错误或硬件特性差异

**注意事项**：
- 确保对比的是相同计算节点的数据
- 注意数据类型可能不同（FP32 vs FP16）
- 考虑硬件优化带来的精度损失（如向量化、融合等）

## 6. 数据类型支持

精度工具支持以下数据类型：
- **浮点类型**：FP32, FP16, BF16, FP8, FP8E4M3, FP8E5M2, FP8E8M0
- **整数类型**：INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64
- **特殊类型**：BOOL, INT4, HF4, HF8

## 7. 使用限制

⚠️ **重要限制**：

1. **不支持上板执行的中间数据检查**，仅支持前端及 Pass 检查
2. **不支持集合通信场景**
3. **不支持特定 Pass**（如 SubgraphToFunction）
4. **不支持 Pass 间自动对比校验**，需人工进行数据对比
5. **必须在算子编译期间执行**，无法在程序退出后构造模拟计算
6. **不支持基于昇腾AI处理器或GPU构造模拟计算**

## 8. 环境要求

### 8.1 编译工具要求（最新版本）
- cmake >= 3.16.3
- make
- g++ >= 9.4.0

### 8.2 早期版本环境要求
需要重新编译安装 PyPTO：
```bash
python3 -m pip install . --verbose --no-build-isolation
```

## 9. 最佳实践

1. **从小规模用例开始**：先使用小数据量验证功能正确性
2. **合理设置误差阈值**：
   - FP32: rtol=1e-3, atol=1e-6
   - FP16/BF16: rtol=1e-2, atol=1e-3
   - 低精度类型：适当放宽阈值

3. **分阶段调试**：
   - 先用Tensor Graph验证模式快速定位问题范围
   - 再用Pass阶段验证模式精确到具体 Pass
   - 最后用中间结果分析模式深入分析

4. **保存中间结果**：设置 `pass_verify_save_tensor=True` 保存所有中间数据

5. **分析误差模式**：关注误差的分布规律和特征值

## 10. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 编译失败 | cmake/g++版本过低 | 升级编译工具链 |
| 执行无输出 | enable_pass_verify未设置 | 检查配置选项 |
| 内存不足 | 数据规模过大 | 缩小测试用例规模 |
| 精度误报 | 误差阈值设置不合理 | 调整 rtol/atol 参数 |
| Pass跳过 | Pass不在filter列表中 | 添加Pass名称到filter |

## 11. 完整示例

### 11.1 Tensor Graph验证模式完整示例

```python
import torch
import pypto

# 配置精度验证选项
verify_options = {
    "enable_pass_verify": True,
}

@pypto.frontend.jit(verify_options=verify_options)
def add_kernel(
    input0: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
) -> pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 16, 1, 64)
    return input0 + input1

def test_add_with_golden():
    shape = (1, 16, 1, 64)
    
    # 准备使用CPU的数据用于计算golden
    input_data0_cpu = torch.rand(shape, dtype=torch.float)
    input_data1_cpu = torch.rand(shape, dtype=torch.float)
    
    # 计算golden结果
    torch_add = torch.add(input_data0_cpu, input_data1_cpu)
    
    # 设置golden数据
    pypto.set_verify_golden_data(goldens=[None, None, torch_add])
    
    # 准备NPU数据
    input_data0 = input_data0_cpu.to('npu')
    input_data1 = input_data1_cpu.to('npu')
    
    # 执行算子
    output_data = add_kernel(input_data0, input_data1)
    
    print("Tensor Graph验证模式测试完成，请查看日志输出")

if __name__ == "__main__":
    test_add_with_golden()
```

### 11.2 Pass阶段验证模式完整示例

```python
import torch
import pypto

# 配置精度验证选项
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_error_tol": [1e-3, 1e-6],
    "pass_verify_filter": ["all"]
}

@pypto.frontend.jit(verify_options=verify_options)
def add_kernel(
    input0: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
) -> pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 16, 1, 64)
    return input0 + input1

def test_add():
    shape = (1, 16, 1, 64)
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    
    output_data = add_kernel(input_data0, input_data1)
    
    # 检查结果
    torch_output = torch.add(input_data0.cpu(), input_data1.cpu())
    if torch.allclose(output_data.cpu(), torch_output, rtol=1e-3, atol=1e-6):
        print("精度验证通过！")
    else:
        print("精度验证失败！")

if __name__ == "__main__":
    test_add()
```

### 11.3 中间结果分析模式完整示例

```python
import torch
import pypto

# 配置精度验证选项
verify_options = {
    "enable_pass_verify": True,
}

@pypto.frontend.jit(verify_options=verify_options)
def complex_kernel(
    input0: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32),
) -> pypto.Tensor((1, 16, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 16, 1, 64)
    
    # 保存输入数据
    pypto.pass_verify_save(input0, "input0_debug")
    pypto.pass_verify_save(input1, "input1_debug")
    
    # 打印输入数据
    pypto.pass_verify_print(input0)
    
    # 中间计算
    temp = input0 + input1
    pypto.pass_verify_save(temp, "temp_add_result")
    
    result = temp * 2.0
    pypto.pass_verify_save(result, "final_result")
    
    return result

def test_complex():
    shape = (1, 16, 1, 64)
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    
    output_data = complex_kernel(input_data0, input_data1)
    
    print("中间结果分析模式测试完成")
    print("请查看 {work_path}/output/output_*/tensor/ 目录下的保存文件")

if __name__ == "__main__":
    test_complex()
```

## 12. 进阶技巧

### 12.1 选择性验证特定Pass

如果只想验证特定的Pass，可以设置filter（用于Pass阶段验证模式）：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_filter": ["AutoCast", "InferMemoryConflict", "Codegenation"]
}
```

### 12.2 自定义误差阈值

针对不同的数据类型设置不同的误差阈值：

```python
# FP32 精度验证
verify_options_fp32 = {
    "enable_pass_verify": True,
    "pass_verify_error_tol": [1e-3, 1e-6]
}

# FP16 精度验证
verify_options_fp16 = {
    "enable_pass_verify": True,
    "pass_verify_error_tol": [1e-2, 1e-3]
}
```

### 12.3 结合控制流调试

对于包含控制流的算子，可以结合控制流信息进行调试：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_filter": ["all"]
}

# 在控制流中插入调测点
@pypto.frontend.jit(verify_options=verify_options)
def kernel_with_control_flow(...):
    for i in range(10):
        temp = input * i
        pypto.pass_verify_save(temp, f"temp_iter_{i}")
    return result
```

---

**文档版本**：1.0  
**最后更新**：2025-03-16  
**适用PyPTO版本**：0.1.1+  
**维护者**：PyPTO开发团队
