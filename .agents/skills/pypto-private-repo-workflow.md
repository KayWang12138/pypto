# PyPTO 私仓开发工作流 Skill

## 概述

本 Skill 定义了在私仓 `https://gitcode.com/parsifal_wang/pypto.git` 中开发 PyPTO 算子的标准工作流程。

## 仓库信息

- **仓库地址**: https://gitcode.com/parsifal_wang/pypto.git
- **本地路径**: N:\pypto_private
- **开发目录**: examples/
- **AGENTS.md**: 包含完整的开发规范和指导

## 工作流程

### 1. 克隆私仓

```bash
git clone https://gitcode.com/parsifal_wang/pypto.git pypto_private
cd pypto_private
```

### 2. 创建示例目录

在 `examples/` 目录下创建新的示例目录：

```bash
mkdir -p examples/<example_name>
```

### 3. 开发算子

参考 `examples/00_hello_world/` 的代码结构，开发新的算子：

#### 必需文件

1. **主实现文件** (`<example_name>.py`)
   - 包含算子实现
   - 包含 golden 参考实现
   - 包含测试函数
   - 遵循 hello_world.py 的代码结构

2. **README.md**
   - 使用中文编写
   - 包含算子概述
   - 包含运行方法
   - 包含注意事项

#### 代码结构参考

```python
#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ... (license header)

"""
Example description
"""
import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    """Get and validate TILE_FWK_DEVICE_ID from environment variable."""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def create_<operator>_kernel(shape: tuple, run_mode: str = "npu"):
    """Create operator kernel."""
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def <operator>_kernel(
        # Define input/output tensors with PyPTO types
        x: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(...)
        # Operator implementation
        out[:] = ...

    return <operator>_kernel


def golden_<operator>(x: torch.Tensor) -> torch.Tensor:
    """Golden reference implementation."""
    # Implement golden on CPU
    return result


def test_<operator>(device_id=None, run_mode: str = "npu") -> None:
    """Test operator."""
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (...)
    
    # Prepare data
    input_data = torch.randn(shape, dtype=torch.float32, device=device)
    
    # Golden computation
    golden_result = golden_<operator>(input_data)
    
    # NPU computation
    output_data = torch.empty(shape, dtype=torch.float32, device=device)
    create_<operator>_kernel(shape, run_mode)(input_data, output_data)
    
    # Verify results
    max_diff = torch.max(torch.abs(output_data.cpu() - golden_result.cpu())).item()
    print(f"Max difference: {max_diff:.6f}")
    
    if run_mode == "npu":
        assert_allclose(output_data.cpu().numpy(), golden_result.cpu().numpy(), rtol=3e-3, atol=3e-3)
    
    print("✓ <operator> example passed")
    print()


def main():
    """Run example."""
    parser = argparse.ArgumentParser(
        description="PyPTO <operator> Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s <example>::test_<operator>
            Run the <operator> example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run. If not specified, all examples will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )

    args = parser.parse_args()

    examples = {
        "<example>::test_<operator>": {
            'name': '<operator>',
            'description': '<operator> implementation',
            'function': test_<operator>
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO <operator> Example")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All <operator> tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
```

### 4. 运行测试

#### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

#### 运行示例

```bash
# 运行示例
python3 examples/<example_name>/<example_name>.py

# 运行特定测试
python3 examples/<example_name>/<example_name>.py <example>::test_<operator>

# 以仿真模式运行
python3 examples/<example_name>/<example_name>.py --run_mode sim

# 列出所有可用示例
python3 examples/<example_name>/<example_name>.py --list
```

### 5. 提交代码

```bash
cd pypto_private

# 添加文件
git add examples/<example_name>/

# 提交
git commit -m "Add <example_name> example: <description>"

# 推送
git push origin main
```

## 开发规范

### 核心原则

1. **参考官方示例**: 严格参考 `examples/00_hello_world/` 的代码结构
2. **包含 Golden 实现**: 每个算子都必须有 CPU 上的 golden 参考实现
3. **精度验证**: 使用 `assert_allclose` 验证 NPU 和 CPU 结果
4. **中文文档**: README.md 必须使用中文编写

### 测试用例

- **Level 0**: 小规模测试（8-16 元素）- 基础功能验证
- **Level 1**: 典型规模测试（1K 元素）- 典型场景验证
- **Level 2**: 边界测试（极值、零值）- 边界情况验证
- **Level 3**: 大规模测试 - 性能验证

### 精度标准

- 默认相对误差: 3e-3
- 默认绝对误差: 3e-3
- 可根据具体算子调整

## 常见问题

### 环境变量未设置

错误: "If no NPU environment is available"

解决:
```bash
export TILE_FWK_DEVICE_ID=0
```

### 设备 ID 无效

错误: "Invalid Device"

解决:
```bash
npu-smi info
export TILE_FWK_DEVICE_ID=<actual_device_id>
```

### 编译错误

1. 检查 PyPTO API 使用是否正确
2. 参考 `AGENTS.md` 中的 API 文档
3. 对比官方示例

### 精度错误

1. 检查 golden 实现是否正确
2. 检查数据类型是否匹配
3. 调整精度容忍度

## 示例列表

### 已实现示例

- `examples/00_hello_world/`: Hello World 示例（张量加法）
- `examples/quant/`: FP8E4M3 Per-Token 量化示例

### 示例模板

使用 `examples/00_hello_world/` 作为新示例的模板。

## 参考文档

- `AGENTS.md`: 完整的开发规范和指导
- `examples/00_hello_world/README.md`: Hello World 示例文档
- `examples/quant/README.md`: 量化示例文档

## 注意事项

1. **代码版权**: 必须包含华为版权声明
2. **文件编码**: 使用 UTF-8 编码
3. **Python 版本**: 使用 Python 3
4. **代码风格**: 遵循 PEP 8 规范
5. **文档语言**: README.md 必须使用中文

## 重要经验教训

### 输入输出 Tensor 的数据类型和 Shape 确认

在开发算子之前，**必须**明确以下信息，如果有任何不明确的地方，必须向用户确认：

#### 必须确认的信息

1. **输入 Tensor 的 Shape**
   - 是二维 (m, n)？
   - 是三维 (batch, seq_len, hidden_size)？
   - 还是四维？
   - 每个维度的含义是什么？

2. **输入 Tensor 的数据类型**
   - FP32 (pypto.DT_FP32, torch.float32)
   - BF16 (pypto.DT_BF16, torch.bfloat16)
   - FP16 (pypto.DT_FP16, torch.float16)
   - 其他？

3. **输出 Tensor 的 Shape**
   - 与输入相同？
   - 有变化？如何变化？

4. **输出 Tensor 的数据类型**
   - FP8E4M3 (pypto.DT_FP8_E4M3, torch.float8_e4m3fn)
   - FP8E8M0 (pypto.DT_FP8_E8M0, torch.float8_e8m0fnu)
   - 其他？

5. **规约操作的规约轴**
   - Per-token 量化：规约轴为 -1（最后一个维度）
   - Per-channel 量化：规约轴为 -2（倒数第二个维度）
   - 其他？

6. **中间 Tensor 的 Shape 和数据类型**
   - 如果有中间计算结果，它们的 shape 和 dtype 是什么？

#### 常见的数据类型对应关系

| PyPTO 类型 | Torch 类型 | 说明 |
|-----------|-----------|------|
| pypto.DT_FP32 | torch.float32 | 32位浮点 |
| pypto.DT_BF16 | torch.bfloat16 | 16位脑浮点 |
| pypto.DT_FP16 | torch.float16 | 16位浮点 |
| pypto.DT_FP8_E4M3 | torch.float8_e4m3fn | FP8 E4M3 格式 |
| pypto.DT_FP8_E8M0 | torch.float8_e8m0fnu | FP8 E8M0 格式 |

#### 错误示例

❌ **错误做法**：猜测输入输出的数据类型和 shape
```python
# 错误：假设输入是 FP32，输出是 FP8E4M3
def quant_kernel(
    x: pypto.Tensor([m, n], pypto.DT_FP32),  # 假设是 FP32
    out: pypto.Tensor([m, n], pypto.DT_FP8_E4M3),  # 假设输出是 FP8E4M3
):
    ...
```

✅ **正确做法**：向用户确认后再实现
```
请确认以下信息：
1. 输入 tensor 的数据类型是什么？（FP32/BF16/FP16/其他）
2. 输入 tensor 的 shape 是什么？
3. 输出 tensor 的数据类型是什么？
4. 输出 tensor 的 shape 是什么？
5. 如果有规约操作，规约轴是哪个？
```

#### 实际案例

在 FP8E4M3 Per-Token 量化示例中，最初实现有错误：
- ❌ 输入使用了 FP32，但实际应该是 BF16
- ❌ scale 使用了 FP32，但实际应该是 FP8E8M0
- ❌ set_vec_tile_shapes 使用了四维，但实际 tensor 是二维

修正后：
- ✅ 输入：BF16 (torch.bfloat16, pypto.DT_BF16)
- ✅ 输出量化：FP8E4M3 (torch.float8_e4m3fn, pypto.DT_FP8_E4M3)
- ✅ 输出 scale：FP8E8M0 (torch.float8_e8m0fnu, pypto.DT_FP8_E8M0)
- ✅ set_vec_tile_shapes 使用二维 shape (m, n, 1, 1)

#### 检查清单

在开始实现算子之前，确保已经明确：

- [ ] 输入 tensor 的 shape
- [ ] 输入 tensor 的数据类型（PyPTO 和 Torch）
- [ ] 输出 tensor 的 shape
- [ ] 输出 tensor 的数据类型（PyPTO 和 Torch）
- [ ] 如果有规约操作，规约轴是哪个
- [ ] 中间 tensor 的 shape 和数据类型（如果有）
- [ ] set_vec_tile_shapes 的参数是否与 tensor shape 匹配
