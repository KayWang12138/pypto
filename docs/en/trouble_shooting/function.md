# FUNCTION Component Error Codes

- **Range**: F2-F3XXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the FUNCTION component.
---

## Error Code Definitions and Usage

The unified definitions for related error codes can be found in the `framework/src/interface/utils/function_error.h` file.

This file defines the following error codes (FError):

### General Error Codes (0x21001U - 0x21008U)

- **EINTERNAL (0x21001U)**: Internal error

- **INVALID_OPERATION (0x21002U)**: Disallowed operation

- **INVALID_TYPE (0x21003U)**: Incorrect type

- **INVALID_VAL (0x21004U)**: Invalid value

- **INVALID_PTR (0x21005U)**: Invalid pointer

- **OUT_OF_RANGE (0x21006U)**: Parameter out of range

- **IS_EXIST (0x21007U)**: Parameter/operation already exists

- **NOT_EXIST (0x21008U)**: Parameter/operation does not exist

### File Error Codes (0x29001U - 0x29002U)

- **BAD_FD (0x29001U)**: Bad file descriptor state

- **INVALID_FILE (0x29002U)**: Invalid file content

### Unknown Error Code

- **UNKNOWN (0x3FFFFU)**: Unknown error

---

## Troubleshooting Recommendations

### General Troubleshooting Recommendations

#### 1. Enable Detailed Logging

When encountering FUNCTION component errors, enable detailed logging for more information:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0 # Debug level logging
export ASCEND_PROCESS_LOG_PATH=./debug_logs # Specify log persistence path
```

#### 2. Enable Graph Compilation Stage Debug Mode Switch

The Function, as the frontend, needs to summarize context based on the developer's usage/syntax, providing it to subsequent components such as the computation graph. When the developer's computation graph has issues, use this debug switch to view whether the program.json dumped by Function meets expectations.

How to enable: [View Computation Graph.md](../../docs/tools/computation_graph/查看计算图.md)

---

## Error Code Related Examples

### EINTERNAL (0x21001U)

**Error Description:** Internal error

**Causes:**
- An unexpected error occurred internally in the system
- Internal state inconsistency

**Solution:**
- Check system status
- Contact technical support

---

### INVALID_OPERATION (0x21002U)

**Error Description:** Disallowed operation

**Causes:**
- Attempting to perform an operation that is not allowed
- Such as writing different data to a Tensor a second time
- Incorrect operation context

**Solution:**
- Check that the operation is performed in the correct context
- Ensure the operation complies with system constraints

**Error Example:**

```cpp
// Error example - Assigning a different Tensor to a Tensor that already has actual data is prohibited
Tensor lhs(DT_FP32, tshape, "lhs");
Tensor rhs(DT_FP32, tshape, "rhs");

auto ptr1 = std::make_unique<uint8_t>(0);
auto ptr2 = std::make_unique<uint8_t>(0);

lhs.SetData(ptr1.get());
rhs.SetData(ptr2.get());

lhs = rhs;  // Incorrect usage: lhs already contains actual Data

// Correct example
Tensor lhs(DT_FP32, tshape, "lhs");
Tensor rhs(DT_FP32, tshape, "rhs");

auto ptr1 = std::make_unique<uint8_t>(0);
auto ptr2 = std::make_unique<uint8_t>(0);

rhs.SetData(ptr2.get());
lhs = rhs;

```

```python
# Error example - Not inside a dynamic function
def not_under_dynamic_function_example():
    x = pypto.tensor([4, 4], pypto.DT_INT32)
    y = x[0, 0]  # GetTensorData must be executed inside a dynamic function
    return y

# Correct example
@pypto.frontend.jit
def correct_dynamic_function_example(x):
    y = x[0, 0]  # GetTensorData executed inside a dynamic function
    return y
```

---

### INVALID_TYPE (0x21003U)

**Error Description:** Incorrect type

**Causes:**
- Type mismatch
- Using an unsupported data type

**Solution:**
- Check the data type
- Use the correct type

**Example:**

```python
# Error example - Data type mismatch
a = pypto.tensor((4, 4), pypto.DT_INT32)
b = pypto.tensor((4, 4), pypto.DT_FP32)
a[0, 0] = 1.3 # SetTensorData, supports only DT_INT32 tensors
data = b[0, 0] # GetTensorData, supports only DT_INT32 tensors

# Correct example
a = pypto.tensor((4, 4), pypto.DT_INT32)
b = pypto.tensor((4, 4), pypto.DT_INT32)
a[0, 0] = 1
data = b[0, 0]
```

---

### INVALID_VAL (0x21004U)

**Error Description:** Invalid value

**Causes:**
- Parameter values (shape, offset, etc.) do not match
- Parameter value format is incorrect

**Solution:**
- Check the parameter format
- Use valid parameter values

**Example:**

```python
# Error example - Invalid shape
x = pypto.tensor([-2, 4], pypto.DT_FP32)

# Correct example
x = pypto.tensor([-1, 4], pypto.DT_FP32)
y = pypto.tensor([4, 4], pypto.DT_FP32)
```

```python
# Error example - Inconsistent shape and offset dimensions
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset and shape dimensions are inconsistent
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])

# Correct example
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

```python
# Error example - Inconsistent shape and offset dimensions
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset and shape dimensions are inconsistent
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])
# Also, the view target shape must maintain the same number of dimensions as tensor(x)
z = pypto.view(x, [16, 16, 64], [0, 0, 0, 0], [])

# Correct example
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

```python
# Error example - offset
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset and shape dimensions are inconsistent
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])

# Correct example
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

---

### INVALID_PTR (0x21005U)

**Error Description:** Invalid pointer

**Causes:**
- Pointer is null
- Pointer is not properly initialized

**Solution:**
- Ensure the pointer is properly initialized
- Check pointer validity

**Example:**

```cpp
// Error example - storage_ is nullptr in the Tensor constructor
Tensor tensor(nullptr);  // Triggers FError::INVALID_PTR error
```

---

### OUT_OF_RANGE (0x21006U)

**Error Description:** Parameter out of range

**Causes:**
- Index out of range
- Parameter value exceeds valid range

**Solution:**
- Check the index range and use valid index values
- Use valid parameter values

**Example:**

```python
# Error example - Axis index out of range
@pypto.frontend.jit
def axis_out_of_range_example(x):
    # x shape is [4, 4], but trying to access axis 2
    return pypto.sum(x, axis=2)  # Axis 2 is out of range (valid axes are 0, 1)

# Correct example
@pypto.frontend.jit
def correct_axis_example(x):
    # Access valid axis 0 or 1
    return pypto.sum(x, axis=0)  # Axis 0 is within range
```

```python
# Error example - View offset mismatch
@pypto.frontend.jit
def view_offset_mismatch_example(x):
    # x shape is [8, 8], but offset [10, 10] is out of range
    return pypto.view(x, [4, 4], offset=[10, 10])  # Offset out of range

# Correct example
@pypto.frontend.jit
def correct_view_offset_example(x):
    # Offset is within valid range
    return pypto.view(x, [4, 4], offset=[2, 2])  # Correct offset
```

```json
// Error - Configuration value overflow
// C++ code calls SetOptionsNg
config::SetOptionsNg("runtime.device_sched_mode", 4);  // Out of range [0, 3], will error
config::SetOptionsNg("runtime.stitch_function_num_initial", 129);  // Out of range [1, 128], will error

// Range is defined in tile_fwk_config_schema.json
{
    "properties": {
        "runtime": {
            "properties": {
                "device_sched_mode": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3
                },
                "stitch_function_num_initial": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 128
                }
            }
        }
    }
}
```

---

### IS_EXIST (0x21007U)

**Error Description:** Parameter/operation already exists

**Causes:**
- Attempting to create an object that already exists
- Duplicate object names

**Solution:**
- Check if the object already exists
- Use unique object names

**Example:**

```python
# Error example - Duplicate idx_name in loop
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="b_idx"):
       ...

# Correct example
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
       ...
```

```cpp
// Error example - Duplicate function name
// Using the same name as the current function in a sub-function of a static function
auto &program = npu::tile_fwk::Program::GetInstance();
// Create a static function
std::string funcName = "test_function";
program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC,
                        npu::tile_fwk::GraphGraphType::TENSOR_GRAPH, {}, false);
// Executing again will trigger a CHECK assertion due to duplicate funcName
// program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC,
//                      npu::tile_fwk::GraphType::TENSOR_GRAPH, {}, false);
```

---

### NOT_EXIST (0x21008U)

**Error Description:** Parameter/operation does not exist

**Causes:**
- Accessing an object that does not exist
- Object not properly registered

**Solution:**
- Check if the object exists
- Ensure the object is properly registered

**Example:**

```cpp
// C++ code calls GetAnyConfig
// Error example - Will cause "key[xx.no_exist] has been not loaded form tile_fwk_config_schema.json." error
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
auto value = AnyCast<int64_t>(scope->GetAnyConfig("xx.no_exist"));

// Correct example
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
// The key being retrieved must exist in tile_fwk_config.json
auto value = AnyCast<int64_t>(scope->GetAnyConfig("pass.pg_parallel_lower_bound"));
```

```json
// Error example - Missing configuration field
// tile_fwk_config_schema.json is missing 'type' or 'properties' field
{
    "properties": {
        "pg_parallel_lower_bound": {
            // "type": "integer",
            "label": "...",
        }
    }
}

// Correct example
{
    "properties": {
        "pg_parallel_lower_bound": {
            "type": "integer",
            "label": "...",
        }
    }
}
```

---

### BAD_FD (0x29001U)

**Error Description:** Bad file descriptor state

**Causes:**
- File descriptor state error
- File not properly opened or closed
- File does not exist
- File is currently in use

**Solution:**
- Check the file descriptor state
- Ensure the file is properly opened and closed

---

### INVALID_FILE (0x29002U)

**Error Description:** Invalid file content

**Causes:**
- File content format error
- File content does not meet expectations

**Solution:**
- Check the file content format
- Use correct file content
