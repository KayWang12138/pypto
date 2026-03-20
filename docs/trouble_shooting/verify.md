# VERIFY 组件错误码


- **范围**：FBXXXX
- 本文档说明 VERIFY 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义

相关错误码的统一定义，参见 ` framework/src/interface/interpreter/verify_error.h`与`framework/src/interface/interpreter/calculator/calc_error.h`文件。

## 排查建议

### 典型错误码场景
该场景涉及的错误码属于精度工具组件功能的一部分，出现这些错误码，精度工具仍然在正常发挥作用，用户可结合一下排查建议自行开展精度问题定位：

#### 错误码：0xB4001U：VERIFY_RESULT_MISMATCH
该场景为精度校验失败场景，同时会伴随如下输出：
#### 错误码：


### 非典型错误码场景
