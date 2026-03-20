# FUNCTION 组件错误码

- **范围**：F2-F3XXXX
- 本文档说明 FUNCTION 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/interface/utils/function_error.h` 文件。

该文件中定义了以下错误码（FError）：

### 通用错误码（0x21001U - 0x21008U）

- **EINTERNAL (0x21001U)**：内部错误

- **INVALID_OPERATION (0x21002U)**：不允许的操作

- **INVALID_TYPE (0x21003U)**：错误的类型

- **INVALID_VAL (0x21004U)**：无效的值

- **INVALID_PTR (0x21005U)**：无效的指针

- **OUT_OF_RANGE (0x21006U)**：参数超出范围

- **IS_EXIST (0x21007U)**：参数/操作已存在

- **NOT_EXIST (0x21008U)**：参数/操作不存在

### 文件错误码（0x29001U - 0x29002U）

- **BAD_FD (0x29001U)**：错误的文件描述符状态

- **INVALID_FILE (0x29002U)**：无效的文件内容

### 未知错误码

- **UNKNOWN (0x3FFFFU)**：未知错误

---

## 排查建议

### 通用排查建议

#### 1. 启用详细日志

在遇到 FUNCTION 组件错误时，可以启用详细日志获取更多信息：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0 # Debug级别日志
export ASCEND_PROCESS_LOG_PATH=./debug_logs # 指定日志落盘路径
```

#### 2. 开启图编译阶段调试模式开关

Function作为前端，需要根据开发者用法/语法总结出上下文，提供给后续组件使用，比如计算图，当开发者的计算图出问题时，使用该调试开关，可查看Function Dump出来的program.json是否符合预期。

开启方法: [查看计算图.md](../../docs/tools/computation_graph/查看计算图.md)
