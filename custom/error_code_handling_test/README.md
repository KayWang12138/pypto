# DUPLICATE_OPERATION Error Test

## Overview

This test demonstrates the `DUPLICATE_OPERATION (0x10003)` error code in PyPTO.

## Error Description

**Error Code:** `DUPLICATE_OPERATION (0x10003)`

**Error Description:** 重复操作错误

**出现原因：**
- 尝试添加已经存在的操作
- 操作名称或标识符重复

**解决办法：**
- 检查操作是否已存在
- 使用唯一的操作名称
- 在添加操作前进行去重检查

## Test Cases

### 1. Error Example

```python
@pypto.frontend.jit
def duplicate_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.add(x, x)  # 可能导致重复操作
    return op1
```

### 2. Correct Example

```python
@pypto.frontend.jit
def correct_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.mul(x, 2)  # 使用不同的操作
    return op1
```

## Running the Test

### Pattern Demonstration Mode (Default)

Run the test to demonstrate error patterns without actual execution:

```bash
python custom/error_code_handling_test/test_duplicate_operation.py
```

This mode will:
- Display the error example code
- Display the correct example code
- Show best practices for avoiding the error

### Run in NPU Mode

```bash
export TILE_FWK_DEVICE_ID=0
python custom/error_code_handling_test/test_duplicate_operation.py --run_mode npu
```

### Run in Simulation Mode

```bash
python custom/error_code_handling_test/test_duplicate_operation.py --run_mode sim
```

**Note:** For NPU or simulation mode, ensure:
1. NPU device ID is set: `export TILE_FWK_DEVICE_ID=0`
2. PTO tile library path is set (if required): `export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa/`

## Expected Output

The test will:
1. Attempt to create duplicate operations (may be optimized away by PyPTO)
2. Demonstrate the correct way to avoid duplicate operations
3. Verify the correct implementation works as expected

## Notes

- PyPTO may automatically optimize away duplicate operations in the computational graph
- The error is more likely to occur in complex graph construction scenarios
- Always use unique operations or reuse existing operation results when possible
