# 15. Debug Dialect

The `pto.debug` dialect provides debugging and source correlation metadata. It enables mapping between IR operations and original source code, and supports debugging and profiling.

Debug objects can be attached to other IR objects as attributes or annotations, and can also exist as independent operations in the IR.

---

## 15.1 Purpose

The debug dialect serves to:

* Link IR operations to source code locations
* Provide debugging annotations
* Enable execution tracing
* Support profiling and performance analysis

---

## 15.2 Core Operations

### 15.2.1 Source Location

#### `debug.semantic_label`

Add semantic label to operation. Semantic labels can be displayed on computation graphs and swimlane diagrams.

**Syntax:**
```
debug.semantic_label {label = "attention", file = "example.py", line = 42}
```

**Attributes:**
- `label`: semantic label name
- `file`: source file name
- `line`: line number (1-based)

#### `debug.source_loc`

Attach source location to operation.

**Syntax:**
```
debug.source_loc {file = "example.py", line = 42, column = 7}
```

**Attributes:**
- `file`: source file name
- `line`: line number (1-based)
- `column`: column number (1-based)
- `function`: function name (optional)

---

### 15.2.2 Dump

#### `debug.dump`

Dump the value of specified tensor for precision debugging.

**Syntax:**
```
debug.dump %tensor {file = "dump_output.bin", format = "binary"}
    : tensor<1024x512xf16>
```

**Attributes:**
- `file`: output file path
- `format`: dump format (binary, text, numpy) (optional)
- `name`: tensor name for identification (optional)

### 15.2.3 Tracing（TODO）

#### `debug.trace`

Insert trace point.

**Syntax:**
```
debug.trace {event_id = 123, message = "entering loop"}
```

**Attributes:**
- `event_id`: unique event identifier
- `message`: trace message
- `level`: trace level (debug, info, warning, error)

---

### 15.2.4 Metadata（TODO）

#### `debug.meta`

Attach arbitrary metadata.

**Syntax:**
```
debug.meta {key = "optimization_pass", value = "fusion", version = 1}
```

**Use cases:**
- Optimization pass information
- Performance hints
- Custom annotations

---

## 15.3 Usage

Debug operations can be attached to any operation:

```mlir
%result = pto.tensor.matmul %A, %B 
    {pto.debug.source_loc = {file = "model.py", line = 10}}
    : tensor<...>, tensor<...> -> tensor<...>
```

---

## 15.4 Lowering Behavior

Debug operations:

* **Preserved**: Propagated through lowering passes
* **Optional removal**: Can be stripped in release builds
* **No semantic impact**: Do not affect program execution

---

## 15.5 Examples

### Source Correlation

```mlir
pto.func.func @matmul(%A: tensor<1024x512xf16>, %B: tensor<512x256xf16>)
    -> (tensor<1024x256xf16>) {
  pto.debug.source_loc {file = "model.py", line = 15}
  %C = pto.tensor.matmul %A, %B : tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>
  pto.debug.source_loc {file = "model.py", line = 16}
  pto.func.return %C
}
```

### Execution Tracing

```
statement.for %i = 0 to N {
  debug.trace {event_id = 100, message = "loop iteration", level = "debug"}
  // loop body
}
```

---

## 15.6 Summary

The debug dialect provides:

* Source code correlation
* Execution tracing support
* Metadata attachment
* Debugging and profiling capabilities

It enables effective debugging and performance analysis of PTO-IR programs.

---

