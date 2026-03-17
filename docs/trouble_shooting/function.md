# FUNCTION 组件错误码

- **范围**：F2-F3XXXX
- 本文档说明 FUNCTION 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/interface/inner/function_error.h` 文件。

该文件中定义了以下错误分类（ErrorCategory）：

- **PROGRAM (F2xxxx)**：程序相关错误（20000U）

- **CONFIG (F2xxxx)**：配置相关错误（21000U）

- **FUNCTION (F2xxxx)**：函数相关错误（22000U）

- **OPERATION (F2xxxx)**：操作相关错误（23000U）

- **TENSOR (F2xxxx)**：张量相关错误（24000U）

- **TENSORSLOT (F2xxxx)**：张量槽相关错误（25000U）

- **SYMBOLIC_SCALAR (F2xxxx)**：符号化标量相关错误（26000U）

- **FILE (F2xxxx)**：文件操作相关错误（27000U）

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

#### 3. 使用错误码处理 Skill

加载 `pypto-function-error-handling` Skill 获取详细的错误码说明和解决方案：

**关联 Skill**：[pypto-function-error-handling](../../.agents/skills/pypto-function-error-handling/SKILL.md)
