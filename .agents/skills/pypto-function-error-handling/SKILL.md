# PyPTO 错误码处理指南

本技能文档了 PyPTO 开发中常见错误码的详细说明，包括错误原因、解决办法和示例代码。

## 目录

- [程序错误码 (ProgErr)](#程序错误码-progerr)
- [配置错误码 (ConfErr)](#配置错误码-conferr)
- [函数错误码 (FuncErr)](#函数错误码-funcerr)
- [操作错误码 (OpErr)](#操作错误码-operr)
- [张量错误码 (TensorErr)](#张量​错误码-tensorerr)
- [张量槽错误码 (TensorSlotErr)](#张量槽错误码-tensorsloterr)
- [符号标量错误码 (SymScalarErr)](#符号标量错误码-symscalarerr)
- [文件错误码 (FileErr)](#文件错误码-fileerr)

---

## 程序错误码 (ProgErr)

### FUNCTION_NOT_FOUND (F20001)

**错误描述：** 函数未找到

**出现原因：**
- 通过 magic 查找函数时未找到
- 函数未正确注册

**解决办法：**
- 检查函数 magic
- 确保函数已正确注册

### FUNCTION_ALREADY_EXISTS (F20002)

**错误描述：** 函数已存在

**出现原因：**
- 尝试创建已存在的函数
- 函数 magic 名称重复

**解决办法：**
- 检查函数是否已存在
- 使用唯一的函数名称

### FUNCTION_NAME_DUPLICATE (F20003)

**错误描述：** 函数名称不匹配

**出现原因：**
- 静态函数的子函数名称与当前函数名称相同

**解决办法：**
- 避免静态函数的子函数与当前函数同名

**错误用例：**

```cpp
// 错误示例 - 函数名称重复

TEST(ProgramErrorTest, FunctionNameMismatch) {
    auto &program = npu::tile_fwk::Program::GetInstance();
    // 创建一个静态函数
    std::string funcName = "test_function";
    program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC, 
                         npu::tile_fwk::GraphType::TENSOR_GRAPH, {}, false);
    // 再次执行会触发CHECK断言，由于funcName重复
    // program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC, 
    //                      npu::tile_fwk::GraphType::TENSOR_GRAPH, {}, false);
}
```

### NO_ACTIVE_FUNCTION (F20004)

**错误描述：** 无活动函数

**出现原因：**
- 没有活动的函数上下文
- 在函数外执行操作

**解决办法：**
- 确保在函数上下文中执行操作

### FUNCTION_STACK_NULL (F20005)

**错误描述：** 函数栈为空

**出现原因：**
- 函数栈为空
- 没有活动的函数上下文
- 在 FinishCurrentFunction 中，functionMagicNameStack_ 为空

**解决办法：**
- 确保函数栈不为空
- 在正确的上下文中执行操作
- 确保在调用 FinishCurrentFunction 前有函数在栈中

### CURRENT_MAGIC_NOT_FOUND (F20006)

**错误描述：** 当前 magic 未找到

**出现原因：**
- 当前 magic 不存在
- magic 未正确设置
- 当前函数的 magic 名称与期望的 funcMagicName 不匹配
- 在 DumpJson 中无法找到当前函数的 magic

**解决办法：**
- 检查当前 magic
- 确保正确设置 magic
- 确保当前函数的 magic 名称正确

### PROGRAM_ENTRY_NOT_FOUND (F20007)

**错误描述：** 程序入口未找到

**出现原因：**
- 程序入口函数未找到
- 入口函数名称不正确

**解决办法：**
- 检查程序入口函数
- 使用正确的入口函数名称

### CURRENT_FUNCTION_NOT_FOUND (F20008)

**错误描述：** 当前函数未找到

**出现原因：**
- 当前函数未找到
- 函数指针丢失
- 在 LoadJson 中无法找到当前函数
- currentFunctionPtr_ 为 nullptr

**解决办法：**
- 检查当前函数
- 确保函数指针有效
- 确保在 LoadJson 中正确设置了 currentFunctionPtr_

**测试用例：**

```cpp
// C++ 测试用例 - 当前函数未找到
// 测试场景：在 LoadJson 中，无法找到当前函数，currentFunctionPtr_ 为 nullptr

#include <gtest/gtest.h>
#include "interface/program/program.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/function/function.h"
#include "nlohmann/json.hpp"

TEST(ProgramErrorTest, CurrentFunctionNotFound_LoadJson) {
    auto &program = npu::tile_fwk::Program::GetInstance();
    program.Reset();
    
    // 创建一个空的 JSON，不包含任何函数
    nlohmann::json programJson;
    programJson["version"] = "1.0";
    programJson["curr_funcmagic"] = 0;
    programJson["functions"] = nlohmann::json::array();
    
    // 尝试 LoadJson，但 JSON 中没有函数
    // 这会导致 currentFunctionPtr_ 为 nullptr
    // 在 LoadJson 结束时会触发 ASSERT(ProgErr::CURRENT_FUNCTION_NOT_FOUND, ...)
    // 错误信息： "loss of pointer"
    
    // 由于 ASSERT 宏会终止程序，这里需要使用 EXPECT_DEATH 或类似机制
    // 在实际测试中，可能需要 mock 或修改测试框架
    
    // 正确的做法：确保 JSON 中至少包含一个函数
    programJson["functions"].push_back(nlohmann::json::object());
    programJson["functions"][0]["magic_name"] = "test_function_0";
    programJson["functions"][0]["func_magic"] = 0;
    programJson["functions"][0]["raw_name"] = "test_function";
    programJson["functions"][0]["func_type"] = 0;  // EAGER
    programJson["functions"][0]["graph_type"] = 1;  // TENSOR_GRAPH
    
    // 现在可以安全地调用 LoadJson
    program.LoadJson(programJson);
}
```

### MUST_HAVE_UNROLL_ONE (F20011)

**错误描述：** 循环必须展开一次

**出现原因：**
- 循环展开次数必须为 1
- 展开次数不满足要求

**解决办法：**
- 确保展开次数为 1
- 使用正确的展开次数

**示例：**

```python
# 错误示例 - 展开次数不为 1
# 会导致循环展开错误

# 正确示例
# 确保展开次数为 1
```

### LOOP_UNROLL_TIMES_INVALID (F20012)

**错误描述：** 循环展开次数无效

**出现原因：**
- 循环展开次数无效
- 展开次数为负数或零

**解决办法：**
- 确保展开次数为正整数
- 使用有效的展开次数

**示例：**

```python
# 错误示例 - 展开次数无效
# 会导致 "unrollTimes must larger than zero!" 错误

# 正确示例
for i in pypto.unroll(4):  # 有效的展开次数
    x = pypto.add(x, 1)
```

### LOOP_UNROLL_TIMES_EXISTS (F20013)

**错误描述：** 循环展开次数已存在

**出现原因：**
- 展开次数已存在
- 重复添加相同的展开次数

**解决办法：**
- 检查展开次数是否已存在
- 避免重复添加

**示例：**

```python
# 错误示例 - 展开次数已存在
# 会导致 "unrollTimes already exists in visited" 错误

# 正确示例
# 避免重复添加展开次数
```

### LOOP_UNROLL_TIMES_EMPTY (F20014)

**错误描述：** 循环展开次数列表为空

**出现原因：**
- 展开次数列表为空
- 未设置展开次数

**解决办法：**
- 确保展开次数列表不为空
- 设置有效的展开次数

**示例：**

```python
# 错误示例 - 展开次数列表为空
# 会导致 "unrollTimes_ is empty" 错误

# 正确示例
# 确保展开次数列表不为空
```

### LOOP_INDEX_NAME_DUPLICATE (F20015)

**错误描述：** 循环索引名称重复

**出现原因：**
- 嵌套循环使用相同的索引变量名
- 循环变量名冲突

**解决办法：**
- 使用不同的循环索引变量名
- 避免变量名冲突

**示例：**

```python
# 错误示例 - 循环索引名称重复
@pypto.frontend.jit
def loop_index_duplicate_example(x):
    for i in pypto.unroll(4):
        for i in pypto.unroll(4):  # 重复的索引名
            x = pypto.add(x, 1)
    return x

# 正确示例
@pypto.frontend.jit
def correct_loop_index_example(x):
    for i in pypto.unroll(4):
        for j in pypto.unroll(4):  # 不同的索引名
            x = pypto.add(x, 1)
    return x
```

---

## 配置错误码 (ConfErr)

### INVALID_TYPE (F21001)

**错误描述：** 配置类型无效

**出现原因：**
- `tile_fwk_config_schema.json` 中使用了不支持的类型
- 当前仅支持：integer、boolean、array、object

**解决办法：**
- 检查 `tile_fwk_config_schema.json` 中的 `type` 字段
- 使用支持的类型

**示例：**

```json
// 错误 - 会导致 "invalid type: short at ..." 错误
{
    "properties": {
        "pg_parallel_lower_bound": {
            "type": "short",
            "label": "...",
        }
    }
}

// 正确
{
    "properties": {
        "pg_parallel_lower_bound": {
            "type": "integer",
            "label": "...",
        }
    }
}
```

### FIELD_MISSING (F21002)

**错误描述：** 配置字段缺失

**出现原因：**
- `tile_fwk_config_schema.json` 中缺少 'type' 或 'properties' 字段

**解决办法：**
- 确保每个配置项包含必需字段
- 对象类型需包含 'type' 和 'properties' 字段

**示例：**

```json
// 错误 - 会导致 "field['type', 'properties'] not found in tile_fwk_config_schema.json" 错误
{
    "properties": {
        "pg_parallel_lower_bound": {
            // "type": "integer",
            "label": "...",
        }
    }
}

// 正确
{
    "properties": {
        "pg_parallel_lower_bound": {
            "type": "integer",
            "label": "...",
        }
    }
}
```

### KEY_NOT_LOADED (F21003)

**错误描述：** 配置键未加载

**出现原因：**
- C++ 代码调用 `GetAnyConfig` 接口访问配置键，但该键在 `tile_fwk_config.json` 中未定义

**解决办法：**
- 在 `tile_fwk_config.json` 中添加缺失的配置项

**示例：**

```cpp
// C++ 代码调用 GetAnyConfig
// 错误示例 - 会导致 "key[xx.no_exist] has been not loaded form tile_fwk_config_schema.json." 错误
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
auto value = AnyCast<int64_t>(scope->GetAnyConfig("xx.no_exist"));

// 正确示例
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
// 获取的键值必须在tile_fwk_config.json文件中存在
auto value = AnyCast<int64_t>(scope->GetAnyConfig("pass.pg_parallel_lower_bound"));
```

### VALUE_OVERFLOW (F21004)

**错误描述：** 配置值溢出

**出现原因：**
- 配置值超出 schema 中定义的 minimum 或 maximum 范围

**解决办法：**
- 检查 `tile_fwk_config_schema.json` 中配置项的 minimum 和 maximum 范围
- 使用在范围内的配置值

**示例：**

```cpp
// C++ 代码调用 SetOptionsNg
config::SetOptionsNg("runtime.device_sched_mode", 4);  // 超出范围 [0, 3]，会报错
config::SetOptionsNg("runtime.stitch_function_num_initial", 129);  // 超出范围 [1, 128]，会报错
```

```json
// tile_fwk_config_schema.json 中定义了范围
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

### SET_FAILED (F21005)

**错误描述：** 配置设置失败

**出现原因：**
- 设置配置项失败
- 配置键不存在或类型不匹配
- 在 SetOptionsNg 中设置选项时发生异常

**解决办法：**
- 检查配置键是否在 schema 中定义
- 确保配置值类型正确
- 检查配置值范围

**示例：**

```cpp
// C++ 代码调用 SetGlobalConfig
// 错误示例 - 设置不存在的键或类型不匹配
ConfigManagerNg::SetGlobalConfig("nonexistent.key", "", 0);  // 键不存在，会报错

// 正确示例
ConfigManagerNg::SetGlobalConfig("runtime.device_sched_mode", "", 0);  // 设置正确的键和值
```

### READ_FAILED (F21006)

**错误描述：** tile_fwk_config.json配置读取失败

**出现原因：**
- 读取配置文件失败
- 配置文件不存在或路径错误
- 配置文件格式错误
- 在 ReadJsonFile 中无法读取配置文件

**解决办法：**
- 检查配置文件路径
- 确保配置文件存在
- 验证配置文件格式正确

---

## 函数错误码 (FuncErr)

### FUNCTION_TYPE_MISMATCH (F22001)

**错误描述：** 函数类型不匹配

**出现原因：**
- 函数类型(Static/Dynamic/...)不匹配
- 期望的函数类型与实际类型不符

**解决办法：**
- 检查函数声明类型
- 使用正确的函数装饰器
- 确保函数返回类型正确

### GRAPH_TYPE_MISMATCH (F22002)

**错误描述：** 函数图类型不匹配

**出现原因：**
- 函数图类型不匹配
- 图类型不一致

**解决办法：**
- 检查函数图类型
- 确保图类型一致
- 使用正确的图类型

**示例：**

```python
# 错误示例 - 图类型不匹配
# 会导致图类型不匹配错误

# 正确示例
# 确保图类型一致
```

### GRAPH_CYCLE_DETECTED (F22003)

**错误描述：** 检测到循环

**出现原因：**
- 函数调用图中存在循环
- 存在递归调用或循环依赖

**解决办法：**
- 检查函数调用关系
- 消除循环依赖
- 使用正确的函数调用结构

**示例：**

```python
# 错误示例 - 循环依赖
# 会导致循环检测错误

# 正确示例
# 消除循环依赖
```

### SLOT_SCOPE_NULL (F22011)

**错误描述：** 槽位作用域为空

**出现原因：**
- 槽位作用域为空
- 槽位未正确初始化

**解决办法：**
- 确保槽位作用域不为空
- 检查槽位初始化

**示例：**

```python
# 错误示例 - 槽位作用域为空
# 会导致槽位作用域错误

# 正确示例
# 确保槽位作用域不为空
```

### ORIGIN_INCAST_NOT_FOUND (F22012)

**错误描述：** 原始输入转换未找到

**出现原因：**
- 原始输入转换不存在
- 转换未正确注册

**解决办法：**
- 检查原始输入转换
- 确保转换已正确注册

**示例：**

```python
# 错误示例 - 原始输入转换未找到
# 会导致转换未找到错误

# 正确示例
# 确保原始输入转换正确注册
```

### ORIGIN_OUTCAST_NOT_FOUND (F22013)

**错误描述：** 原始输出转换未找到

**出现原因：**
- 原始输出转换不存在
- 转换未正确注册

**解决办法：**
- 检查原始输出转换
- 确保转换已正确注册

**示例：**

```python
# 错误示例 - 原始输出转换未找到
# 会导致转换未找到错误

# 正确示例
# 确保原始输出转换正确注册
```

### SLOT_SET_INDEX_SIZE_INVALID (F22021)

**错误描述：** 槽位集合索引大小无效

**出现原因：**
- 槽位集合索引大小无效
- 索引超出范围

**解决办法：**
- 检查槽位集合索引大小
- 使用有效的索引值

**示例：**

```python
# 错误示例 - 槽位集合索引大小无效
# 会导致索引大小错误

# 正确示例
# 确保槽位集合索引大小有效
```

### INCAST_HAS_NO_CONSUMER (F22022)

**错误描述：** 输入转换无消费者

**出现原因：**
- 输入转换没有消费者
- 转换结果未被使用

**解决办法：**
- 检查输入转换消费者
- 确保转换结果被正确使用

**示例：**

```python
# 错误示例 - 输入转换无消费者
# 会导致转换无消费者错误

# 正确示例
# 确保输入转换有消费者
```

### OUTCAST_HAS_NO_PRODUCER (F22023)

**错误描述：** 输出转换无生产者

**出现原因：**
- 输出转换没有生产者
- 转换输入未定义

**解决办法：**
- 检查输出转换生产者
- 确保转换输入已定义

**示例：**

```python
# 错误示例 - 输出转换无生产者
# 会导致转换无生产者错误

# 正确示例
# 确保输出转换有生产者
```

### PARAM_INDEX_OUT_OF_BOUNDS (F22031)

**错误描述：** 参数索引越界

**出现原因：**
- 参数索引超出范围
- 访问无效的函数参数

**解决办法：**
- 检查参数索引范围
- 使用有效的索引值
- 验证函数参数数量

**示例：**

```python
# 错误示例 - 参数索引越界
# 会导致参数索引越界错误

# 正确示例
# 确保参数索引在有效范围内
```

### PARAM_ADDRESS_NOT_STORED (F22032)

**错误描述：** 参数地址未存储

**出现原因：**
- 参数地址未存储
- 参数地址丢失

**解决办法：**
- 检查参数地址存储
- 确保参数地址正确

**示例：**

```python
# 错误示例 - 参数地址未存储
# 会导致参数地址错误

# 正确示例
# 确保参数地址正确存储
```

### PARAM_INFO_MISMATCH (F22033)

**错误描述：** 参数信息不匹配

**出现原因：**
- 参数信息不匹配
- 参数类型或数量不匹配

**解决办法：**
- 检查参数信息
- 确保参数类型和数量正确

**示例：**

```python
# 错误示例 - 参数信息不匹配
# 会导致参数信息错误

# 正确示例
# 确保参数信息正确
```

### CALLEE_FUNCTION_NULL (F22041)

**错误描述：** 被调用函数为空

**出现原因：**
- 被调用函数为空
- 函数指针为空

**解决办法：**
- 确保被调用函数不为空
- 检查函数指针

**示例：**

```python
# 错误示例 - 被调用函数为空
# 会导致函数为空错误

# 正确示例
# 确保被调用函数不为空
```

### CALLEE_NOT_IN_FUNCTION_MAP (F22042)

**错误描述：** 被调用函数不在函数映射中

**出现原因：**
- 被调用函数未注册
- 函数映射中找不到该函数

**解决办法：**
- 检查被调用函数是否已注册
- 确保函数在映射中

**示例：**

```python
# 错误示例 - 被调用函数不在函数映射中
@pypto.frontend.jit
def callee_not_in_map_example(x):
    y = pypto.call_function("nonexistent_function", x)
    return y

# 正确示例
@pypto.frontend.jit
def my_function(x):
    return pypto.add(x, 1)

@pypto.frontend.jit
def correct_callee_example(x):
    y = pypto.call_function("my_function", x)
    return y
```

### TARGET_FUNCTION_NULL (F22043)

**错误描述：** 目标函数为空

**出现原因：**
- 目标函数为空
- 函数指针为空

**解决办法：**
- 确保目标函数不为空
- 检查函数指针

**示例：**

```python
# 错误示例 - 目标函数为空
# 会导致函数为空错误

# 正确示例
# 确保目标函数不为空
```

### FUNCTION_NO_PARENT (F22044)

**错误描述：** 函数无父函数

**出现原因：**
- 函数没有父函数
- 函数层次结构不正确
- 在 FinishCurrentFunction 中，当前函数没有父函数

**解决办法：**
- 确保函数有正确的父函数
- 检查函数层次结构
- 确保在调用 FinishCurrentFunction 时 generateCall 为 false，或者函数有父函数

### GET_TENSOR_DATA_INDEX_INVALID (F22051)

**错误描述：** 获取张量数据索引无效

**出现原因：**
- 获取张量数据时索引无效
- 索引超出范围

**解决办法：**
- 检查张量数据索引
- 使用有效的索引值

**示例：**

```python
# 错误示例 - 获取张量数据索引无效
# 会导致索引错误

# 正确示例
# 确保张量数据索引有效
```

### FUNCTION_NOT_IN_USAGE_DICT (F22052)

**错误描述：** 函数不在使用字典中

**出现原因：**
- 函数未在使用字典中注册
- 函数使用信息丢失

**解决办法：**
- 检查函数使用字典
- 确保函数已注册

**示例：**

```python
# 错误示例 - 函数不在使用字典中
# 会导致函数未注册错误

# 正确示例
# 确保函数在使用字典中
```

### IMPORT_INDEX_NOT_FOUND (F22053)

**错误描述：** 导入索引未找到

**出现原因：**
- 导入索引不存在
- 导入项未正确注册

**解决办法：**
- 检查导入索引
- 确保导入项已注册

**示例：**

```python
# 错误示例 - 导入索引未找到
# 会导致导入未找到错误

# 正确示例
# 确保导入索引正确
```

### INCAST_OUTCAST_INDEX_INVALID (F22054)

**错误描述：** 输入输出转换索引无效

**出现原因：**
- 输入输出转换索引无效
- 索引超出范围

**解决办法：**
- 检查输入输出转换索引
- 使用有效的索引值

**示例：：**

```python
# 错误示例 - 输入输出转换索引无效
# 会导致索引错误

# 正确示例
# 确保输入输出转换索引有效
```

### OUTCAST_INDEX_NOT_FOUND (F22055)

**错误描述：** 输出转换索引未找到

**出现原因：**
- 输出转换索引不存在
- 转换未正确注册

**解决办法：**
- 检查输出转换索引
- 确保转换已注册

**示例：**

```python
# 错误示例 - 输出转换索引未找到
# 会导致转换未找到错误

# 正确示例
# 确保输出转换索引正确
```

### JSON_FUNCTION_KIND_INVALID (F22071)

**错误描述：** JSON 函数类型无效

**出现原因：**
- JSON 反序列化时函数类型无效
- kind 字段值不正确

**解决办法：**
- 检查 JSON 函数类型
- 使用正确的 kind 值

**示例：**

```python
# 错误示例 - JSON 函数类型无效
# 会导致函数类型错误

# 正确示例
# 确保JSON 函数类型正确
```

### OPERAND_NOT_BELONGS_TO_CURRENT_FUNCTION (F22101)

**错误描述：** 操作数不属于当前函数

**出现原因：**
- 操作数不在当前函数中
- 操作数跨函数使用

**解决办法：**
- 检查操作数所属函数
- 确保操作数在当前函数中

**示例：**

```python
# 错误示例 - 操作数不属于当前函数
# 会导致操作数错误

# 正确示例
# 确保操作数属于当前函数
```

### PRODUCER_NOT_FOUND (F22102)

**错误描述：** 生产者未找到

**出现原因：**
- 操作的生产者未找到
- 生产者关系不正确

**解决办法：**
- 检查操作的生产者
- 确保生产者关系正确

**示例：**

```python
# 错误示例 - 生产者未找到
# 会导致生产者错误

# 正确示例
# 确保生产者关系正确
```

### OPERATION_ALREADY_IN_MAP (F22103)

**错误描述：** 操作已在映射中

**出现原因：**
- 操作已存在
- 重复添加相同的操作

**解决办法：**
- 检查操作是否已存在
- 避免重复添加

**示例：**

```python
# 错误示例 - 操作已在映射中
# 会导致操作重复错误

# 正确示例
# 避免重复添加操作
```

### INCAST_HAS_CONFLICT_TENSORS (F22111)

**错误描述：** 输入转换有冲突的张量

**出现原因：**
- 输入转换中有冲突的张量
- 张量类型或形状不匹配

**解决办法：**
- 检查输入转换中的张量
- 确保张量类型和形状匹配

**示例：**

```python
# 错误示例 - 输入转换有冲突的张量
# 会导致张量冲突错误

# 正确示例
# 确保输入转换中的张量无冲突
```

### OP_LIST_MISMATCH (F22121)

**错误描述：** 操作列表不匹配

**出现原因：**
- 操作列表不匹配
- 操作数量或顺序不正确

**解决办法：**
- 检查操作列表
- 确保操作数量和顺序正确

**示例：**

```python
# 错误示例 - 操作列表不匹配
# 会导致操作列表错误

# 正确示例
# 确保操作列表正确
```

### OP_NOT_FOUND_IN_POSITION (F22122)

**错误描述：** 在指定位置未找到操作

**出现原因：**
- 在指定位置找不到操作
- 操作位置不正确

**解决办法：**
- 检查操作位置
- 确保操作在正确位置

**示例：**

```python
# 错误示例 - 在指定位置未找到操作
# 会导致操作位置错误

# 正确示例
# 确保操作在正确位置
```

### OP_ALREADY_IN_GROUP (F22123)

**错误描述：** 操作已在组中

**出现原因：**
- 操作已存在于组中
- 重复添加操作到组

**解决办法：**
- 检查操作是否已在组中
- 避免重复添加

**示例：**

```python
# 错误示例 - 操作已在组中
# 会导致操作重复错误

# 正确示例
# 避免重复添加操作到组
```

### OP_GROUP_MISMATCH (F22124)

**错误描述：** 操作组不匹配

**出现原因：**
- 操作组不匹配
- 操作不属于指定组

**解决办法：**
- 检查操作组
- 确保操作属于正确的组

**示例：**

```python
# 错误示例 - 操作组不匹配
# 会导致操作组错误

# 正确示例
# 确保操作组正确
```

### OP_ALREADY_IN_MAP (F22125)

**错误描述：** 操作已在映射中

**出现原因：**
- 操作已存在
- 重复添加相同的操作

**解决办法：**
- 检查操作是否已存在
- 避免重复添加

**示例：**

```python
# 错误示例 - 操作已在映射中
# 会导致操作重复错误

# 正确示例
# 避免重复添加操作
```

---

## 操作错误码 (OpErr)

### OP_ATTRIBUTE_NULL (F23001)

**错误描述：** 操作属性为空

**出现原因：**
- 操作属性为空
- 属性未正确设置

**解决办法：**
- 确保操作属性已正确设置
- 检查属性指针

**示例：**

```python
# 错误示例 - 操作属性为空
# 会导致 "Operation doesn't have attribute" 错误

# 正确示例
# 确保属性已正确设置
```

### ATTRIBUTE_VALUE_INVALID (F23002)

**错误描述：** 操作属性值无效

**出现原因：**
- 属性值超出有效范围
- 属性值类型不匹配
- CastMode 值不在 [CAST_NONE, CAST_ODD] 范围内

**解决办法：**
- 检查属性值范围
- 使用有效的属性值
- 确保属性值类型正确

**示例：**

```python
# 错误示例 - 属性值无效
# 会导致属性值无效错误

# 正确示例
# 使用有效的属性值
```

### SYMBOLIC_SCALAR_INVALID (F23003)

**错误描述：** 符号标量无效

**出现原因：**
- 符号标量未初始化
- 符号标量值无效
- SymbolicScalar.IsValid() 返回 false

**解决办法：**
- 确保符号标量已正确初始化
- 使用有效的符号标量值
- 检查符号标量有效性

**示例：**

```python
# 错误示例 - 符号标量无效
# 会导致符号标量无效错误

# 正确示例
# 确保符号标量有效
```

### OPERAND_NULL (F23004)

**错误描述：** 操作数为空

**出现原因：**
- 操作数指针为空
- 操作数未正确初始化

**解决办法：**
- 确保操作数已正确初始化
- 检查操作数指针
- 使用有效的操作数

**示例：**

```python
# 错误示例 - 操作数为空
# 会导致操作数为空错误

# 正确示例
# 确保操作数已正确初始化
```

### OPERAND_INDEX_OUT_OF_BOUNDS (F23005)

**错误描述：** 操作数索引越界

**出现原因：**
- 操作数索引超出范围
- 访问无效的操作数

**解决办法：**
- 检查操作数索引范围
- 使用有效的索引值
- 验证操作数数量

**示例：**

```python
# 错误示例 - 操作数索引越界
# 会导致操作数索引越界错误

# 正确示例
# 确保索引在有效范围内
```

### TENSOR_NOT_IN_DICT (F23006)

**错误描述：** 张量不在字典中

**出现原因：**
- 张量未在 tensorDict 中找到
- 张量 magic 不存在
- 在 LoadJson 中无法找到张量

**解决办法：**
- 检查张量 magic
- 确保张量已正确注册
- 使用有效的张量

**示例：**

```python
# 错误示例 - 张量不在字典中
# 会导致张量不在字典中错误

# 正确示例
# 确保张量已正确注册
```

### OP_TYPE_MISMATCH (F23007)

**错误描述：** 操作类型不匹配

**出现原因：**
- 操作类型不匹配
- 期望的操作类型与实际类型不符
- 在 LoadJson 中，kind 字段不是 T_KIND_OPERATION

**解决办法：**
- 检查操作类型
- 确保操作类型正确
- 使用正确的操作类型

**示例：**

```python
# 错误示例 - 操作类型不匹配
# 会导致操作类型不匹配错误

# 正确示例
# 确保操作类型正确
```

### OP_ATTR_NULLPTR (F23008)

**错误描述：** 操作属性为空指针

**出现原因：**
- 操作属性指针为空
- CallOpAttribute 未正确设置

**解决办法：**
- 确保操作属性已正确设置
- 检查操作属性指针
- 使用有效的操作属性

**示例：**

```python
# 错误示例 - 操作属性为空
# 会导致操作属性为空错误

# 正确示例
# 确保操作属性已正确设置
```

### OP_NOT_REGISTER (F23009)

**错误描述：** 操作未注册

**出现原因：**
- 操作未注册
- 操作类型不支持

**解决办法：**
- 检查操作是否已注册
- 使用支持的操作类型

**示例：**

```python
# 错误示例 - 操作未注册
# 会导致操作未注册错误

# 正确示例
# 确保操作已注册
```

### OP_NOT_MARKED_DELETED (F2300A)

**错误描述：** 操作未标记为已删除

**出现原因：**
- 操作删除状态不正确
- 操作未被标记删除

**解决办法：**
- 检查操作删除状态
- 确保操作正确标记

**示例：**

```python
# 错误示例 - 操作未标记为已删除
# 会导致操作状态错误

# 正确示例
# 确保操作删除状态正确
```

### HASH_DUPLICATE (F2300B)

**错误描述：** 哈希值重复

**出现原因：**
- 操作哈希值重复
- 操作标识冲突

**解决办法：**
- 检查操作哈希值
- 确保操作标识唯一

**示例：**

```python
# 错误示例 - 哈希值重复
# 会导致哈希值重复错误

# 正确示例
# 确保操作哈希值唯一
```

### OP_DUPLICATE (F23011)

**错误描述：** 操作重复

**出现原因：**
- 操作重复
- 操作标识冲突

**解决办法：**
- 检查操作是否重复
- 使用唯一的操作标识

**示例：**

```python
# 错误示例 - 操作重复
# 会导致操作重复错误

# 正确示例
# 避免操作重复
```

### OP_MAGIC_OUT_OF_BOUNDS (F23012)

**错误描述：** 操作 magic 越界

**出现原因：**
- 操作 magic 值超出范围
- magic 值无效

**解决办法：**
- 检查操作 magic 值
- 使用有效的 magic 值

**示例：**

```python
# 错误示例 - 操作 magic 越界
# 会导致 magic 值错误

# 正确示例
# 确保操作 magic 值有效
```

### OP_MAGIC_RANGE_INVALID (F23013)

**错误描述：** 操作 magic 范围无效

**出现原因：**
- 操作 magic 范围无效
- magic 范围设置错误

**解决办法：**
- 检查操作 magic 范围
- 使用有效的 magic 范围

**示例：**

```python
# 错误示例 - 操作 magic 范围无效
# 会导致 magic 范围错误

# 正确示例
# 确保操作 magic 范围有效
```

### OP_CYCLE_DETECTED (F23021)

**错误描述：** 检测到操作循环

**出现原因：**
- 操作图中存在循环
- 存在循环依赖

**解决办法：**
- 检查操作依赖关系
- 消除循环依赖
- 使用正确的操作结构

**示例：**

```python
# 错误示例 - 操作循环依赖
# 会导致循环检测错误

# 正确示例
# 消除循环依赖
```

### CONSUMER_NOT_FOUND (F23051)

**错误描述：** 消费者未找到

**出现原因：**
- 操作的消费者未找到
- 消费者关系不正确

**解决办法：**
- 检查操作的消费者
- 确保消费者关系正确

**示例：**

```python
# 错误示例 - 消费者未找到
# 会导致消费者错误

# 正确示例
# 确保消费者关系正确
```

### MATCHES_SHOULD_NOT_BE_EMPTY (F23081)

**错误描述：** 匹配不应为空

**出现原因：**
- 匹配结果为空
- 连接操作失败

**解决办法：**
- 检查匹配结果
- 确保连接操作正确

**示例：**

```python
# 错误示例 - 匹配为空
# 会导致匹配错误

# 正确示例
# 确保匹配结果不为空
```

### OOP_ATTR_OFFSET_NOT_EMPTY (F23091)

**错误描述：** 操作属性偏移不为空

**出现原因：**
- 操作属性偏移不为空
- 偏移设置错误

**解决办法：**
- 检查操作属性偏移
- 确保偏移设置正确

**示例：**

```python
# 错误示例 - 操作属性偏移不为空
# 会导致偏移错误

# 正确示例
# 确保操作属性偏移正确
```

### ARG_LIST_NOT_EMPTY (F23092)

**错误描述：** 参数列表不为空

**出现原因：**
- 参数列表不为空
- 参数清理失败

**解决办法：**
- 检查参数列表
- 确保参数清理正确

**示例：**

```python
# 错误示例 - 参数列表不为空
# 会导致参数列表错误

# 正确示例
# 确保参数列表正确
```

### OUTCAST_INDEX_TO_EXPR_NOT_EMPTY (F23093)

**错误描述：** 输出转换索引到表达式不为空

**出现原因：**
- 输出转换索引到表达式不为空
- 表达式清理失败

**解决办法：**
- 检查输出转换索引到表达式
- 确保表达式清理正确

**示例：**

```python
# 错误示例 - 输出转换索引到表达式不为空
# 会导致表达式错误

# 正确示例
# 确保输出转换索引到表达式正确
```

---

## 张量错误码 (TensorErr)

### NO_DIMENSIONS (F24001)

**错误描述：** Tensor 无维度

**出现原因：**
- Tensor 维度为空（shape 为空列表）
- Tensor 未正确初始化（storage_ 为 nullptr）
- 在调用 GetShape(int axis) 时，Tensor 的维度数量为 0

**解决办法：**
- 检查 Tensor 维度是否为空
- 确保正确初始化 Tensor，提供有效的 shape 参数
- 使用有效的维度参数（非空列表）
- 在访问 tensor.shape 属性前检查 tensor.dim > 0
- 使用 tensor.IsEmpty() 检查 Tensor 是否为空

**示例：**

```cpp
// 错误示例 - 创建空维度 Tensor 并调用 GetShape
Tensor x(DT_FP32, {}, "x");  // 创建空维度 Tensor
x.GetShape(0);  // 错误！触发 ASSERT(dimCount > 0)

// 正正确示例 1 - 使用有效的维度
Tensor x(DT_FP32, {32, 32}, "x");  // 正确：有 2 个维度
int32_t dim = x.GetShape(0);  // 返回 32
```

### AXIS_OUT_OF_RANGE (F24002)

**错误描述：** Tensor 轴超出范围

**出现原因：**
- 访问的轴索引超出 Tensor 维度
- 轴参数为负数或过大
- 在 reduce 操作中使用了无效的轴参数

**解决办法：**
- 检查 Tensor 维度数量
- 确保轴索引在 [0, dim-1] 范围内
- 使用正确的轴索引或使用默认值

**示例：**

```python
# 错误示例 - 轴索引超出范围
@pypto.frontend.jit
def axis_out_of_range_example(x):
    # x 形状为 [4, 4]，但尝试访问轴 2
    return pypto.sum(x, axis=2)  # 轴 2 超出范围（有效轴为 0, 1）

# 正确示例
@pypto.frontend.jit
def correct_axis_example(x):
    # 访问有效的轴 0 或 1
    return pypto.sum(x, axis=0)  # 轴 0 在范围内
```

### DIMS_INCONSISTENT (F24003)

**错误描述：** Tensor 维度不一致

**出现原因：**
- 两个 Tensor 维度不一致
- 维度数量不匹配
- 在操作中维度参数不匹配

**解决办法：**
- 检查 Tensor 维度
- 使用 reshape 或 broadcast 调整维度
- 确保操作前维度匹配

**示例：**

```python
# 错误示例 - 维度不一致
@pypto.frontend.jit
def dims_inconsistent_example(x, y):
    # x 和 y 维度不一致时可能导致错误
    return pypto.add(x, y)

# 正确示例
@pypto.frontend.jit
def correct_dims_example(x, y):
    # 确保维度一致
    if x.dim != y.dim:
        y = pypto.reshape(y, x.shape)
    return pypto.add(x, y)
```

### VIEW_DIMENSION_MISMATCH (F24004)

**错误描述：** Tensor 视图维度不匹配

**出现原因：**
- View 操作的维度与源 Tensor 不匹配
- 视图形状参数错误
- 在 cube 操作中，形状维度与 offset 维度不匹配

**解决办法：**
- 检查 View 操作的维度参数
- 确保视图形状与源 Tensor 兼容
- 使用正确的视图参数
- 确保形状维度与 offset 维度一致

**示例：**

```python
# 错误示例 - 视图维度不匹配
@pypto.frontend.jit
def view_dimension_mismatch_example(x):
    # x 形状为 [8, 8]，但视图形状为 [4, 4, 4]
    return pypto.view(x, [4, 4, 4])  # 维度不匹配

# 正确示例
@pypto.frontend.jit
def correct_view_example(x):
    # 视图形状与源 Tensor 兼容
    return pypto.view(x, [4, 4])  # 正确的视图形状
```

### OFFSET_MISMATCH (F24005)

**错误描述：** Tensor 视图偏移不匹配

**出现原因：**
- View 操作的偏移参数错误
- 偏移超出源 Tensor 范围

**解决办法：**
- 检查 View 操作的偏移参数
- 确保偏移在有效范围内
- 使用正确的偏移值

**示例：**

```python
# 错误示例 - 视图偏移不匹配
@pypto.frontend.jit
def view_offset_mismatch_example(x):
    # x 形状为 [8, 8]，但偏移 [10, 10] 超出范围
    return pypto.view(x, [4, 4], offset=[10, 10])  # 偏移超出范围

# 正确示例
@pypto.frontend.jit
def correct_view_offset_example(x):
    # 偏移在有效范围内
    return pypto.view(x, [4, 4], offset=[2, 2])  # 正确的偏移
```

### SHAPE_OUT_OF_BOUNDS (F24006)

**错误描述：** Tensor 形状越界

**出现原因：**
- Tensor 形状值超出允许范围
- 形状维度过多或过大

**解决办法：**
- 检查 Tensor 形状值
- 使用合理的形状参数
- 参考 API 文档中的形状限制

**示例：**

```python
# 错误示例 - 形状越界界
@pypto.frontend.jit
def shape_out_of_bounds_example():
    # 形状过大
    x = pypto.tensor([100000, 100000], pypto.DT_FP32)
    return x

# 正确示例
@pypto.frontend.jit
def correct_shape_example():
    # 合理的形状
    x = pypto.tensor([1024, 1024], pypto.DT_FP32)
    return x
```

### INVALID_SHAPE (F24007)

**错误描述：** 无效的 Tensor 形状

**出现原因：**
- Tensor 形状包含负数或零
- 形状参数格式错误

**解决办法：**
- 检查 Tensor 形状参数
- 确保形状值为正整数
- 使用有效的形状格式

**示例：**

```python
# 错误示例 - 无效形状
@pypto.frontend.jit
def invalid_shape_example():
    # 形状包含负数
    x = pypto.tensor([-1, 4], pypto.DT_FP32)
    return x

# 正确则示例
@pypto.frontend.jit
def correct_shape_example():
    # 形状为正整数
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    return x
```

### SHAPE_MISMATCH (F24008)

**错误描述：** Tensor 形状不匹配

**出现原因：**
- 两个 Tensor 形状不匹配
- 形状维度或大小不一致

**解决办法：**
- 检查 Tensor 形状
- 使用 reshape 调整形状
- 确保操作前形状匹配

**示例：**

```python
# 错误示例 - 形状不匹配
@pypto.frontend.jit
def shape_mismatch_example(x, y):
    # x 和 y 形状不匹配时可能导致错误
    return pypto.add(x, y)

# 正确示例
@pypto.frontend.jit
def correct_shape_example(x, y):
    # 确保形状一致
    if x.shape != y.shape:
        y = pypto.reshape(y, x.shape)
    return pypto.add(x, y)
```

### DATATYPE_MISMATCH (F2400A)

**错误描述：** Tensor 数据类型不匹配

**出现原因：**
- 两个 Tensor 数据类型不匹配
- 操作要求特定数据类型但未满足

**解决办法：**
- 检查 Tensor 数据类型
- 使用类型转换函数
- 确保操作前数据类型匹配

**示例：**

```python
# 错误示例 - 数据类型不匹配
@pypto.frontend.jit
def datatype_mismatch_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = pypto.tensor([4, 4], pypto.DT_INT32)
    return pypto.add(x, y)  # 数据类型不匹配

# 正确示例
@pypto.frontend.jit
def correct_datatype_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = pypto.tensor([4, 4], pypto.DT_FP32)
    return pypto.add(x, y)  # 数据类型匹配
```

### STORAGE_NULL (F2400B)

**错误描述：** Tensor 存储为空

**出现原因：**
- Tensor 存储未分配
- Tensor 存储指针为空

**解决办法：**
- 检查 Tensor 存储
- 确保正确分配存储
- 使用有效的存储指针

**示例：**

```cpp
// 错误示例 - Tensor 构造函数中 storage_ 为 nullptr
// 用 nullptr 作为 LogicalTensor 创建 Tensor 对象
// 构造函数会检查 storage_ != nullptr (tensor.cpp:46)
// 这里会触发 STORAGE_NULL 错误
Tensor tensor(nullptr);  // 触发 F2400B 错误
```

### SELF_ASSIGNMENT (F2400C)

**错误描述：** 自赋值禁止

**出现原因：**
- 尝试将 Tensor 赋值给自己
- 自赋值操作被禁止

**解决办法：**
- 避免自赋值操作
- 使用正确的赋值目标
- 检查赋值操作

**示例：**

```python
# 错误示例 - 自赋值
# 会导致 "Prohibit self-assignment" 错误

# 正确示例
# 避免自赋值操作
```

### MAGIC_NOT_FOUND (F2400E)

**错误描述：** Tensor magic 未找到

**出现原因：**
- Tensor magic 不存在
- Tensor 未正确注册

**解决办法：**
- 检查 Tensor magic
- 确保正确注册 Tensor
- 使用有效的 magic 值

**示例：**

```python
# 错误示例 - Tensor magic 未找到
# 会导致 "rawTensorDict doesn't have magic" 错误

# 正确示例
# 确保 Tensor magic 正确注册
```

### NOT_UNDER_DYNAMIC_FUNCTION (F2400F)

**错误描述：** Tensor 不在动态函数中

**出现原因：**
- 在动态函数外使用需要动态上下文的操作
- Tensor 操作需要在 @pypto.frontend.jit 装饰的函数内执行
- 使用了 GetTensorData/GetTensorDataInt32 等 API 但不在动态函数上下文中

**解决办法：**
- 将 Tensor 操作放在 @pypto.frontend.jit 装饰的函数内
- 确保操作在正确的上下文中执行
- GetTensorData 操作必须在动态函数中调用

**示例：**

```python
# 错误示例 - 不在动态函数中
def not_under_dynamic_function_example():
    x = pypto.tensor([4, 4], pypto.DT_INT32)
    y = x[0, 0]  # GetTensorData 需要在动态函数中执行
    return y

# 正确示例
@pypto.frontend.jit
def correct_dynamic_function_example(x):
    y = x[0, 0]  # GetTensorData 在动态函数中执行
    return y
```

**重要提示：**
- `x[0, 0]` 索引操作（GetTensorData）**仅支持 DT_INT32 类型的 Tensor**
- 对于其他数据类型（DT_FP32、DT_FP16、DT_INT64、DT_BOOL 等），使用索引操作会失败
- 如果需要对非 DT_INT32 类型 Tensor 进行元素访问，请使用其他操作如 reshape、view 等

### NOT_IN_INPUT_LIST (F24010)

**错误描述：** Tensor 不在输入列表中

**出现原因：**
- Tensor 不在输入列表中
- 输入参数不匹配

**解决办法：**
- 检查输入参数
- 确保所有 Tensor 都在输入中
- 验证函数签名

**示例：**

```python
# 错误示例 - Tensor 不在输入列表中
@pypto.frontend.jit
def tensorslot_not_in_input_example():
    # 使用未在输入中的 Tensor
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    return x  # x 不在输入中

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # x 在输入中
    y = pypto.add(x, 1)
    return y
```

### INVALID_DESC_INDEX (F24012)

**错误描述：** 无效的描述索引

**出现原因：**
- 描述索引无效
- 索引超出范围

**解决办法：**
- 检查描述索引
- 使用有效的索引值

**示例：**

```python
# 错误示例 - 无效的描述索引
# 会导致索引错误

# 正确示例
# 确保描述索引有效
```

### OPERAND_NOT_CONSUMER (F24013)

**错误描述：** 操作数不是消费者

**出现原因：**
- 操作数不是该张量的消费者
- 消费者关系不正确

**解决办法：**
- 检查操作数消费者关系
- 确保消费者关系正确

**示例：**

```python
# 错误示例 - 操作数不是消费者
# 会导致消费者关系错误

# 正确示例
# 确保操作数是消费者
```

### OPERAND_NOT_PRODUCER (F24014)

**错误描述：** 操作数不是生产者

**出现原因：**
- 操作数不是该张量的生产者
- 生产者关系不正确

**解决办法：**
- 检查操作数生产者关系
- 确保生产者关系正确

**示例：**

```python
# 错误示例 - 操作数不是生产者
# 会导致生产者关系错误

# 正确示例
# 确保操作数是生产者
```

### TENSOR_HAS_PRODUCERS (F24015)

**错误描述：** 张量已有生产者

**出现原因：**
- 张量已有生产者
- 重复设置生产者

**解决办法：**
- 检查张量生产者
- 避免重复设置生产者

**示例：**

```python
# 错误示例 - 张量已有生产者
# 会导致生产者重复错误

# 正确示例
# 避免重复设置生产者
```

---

## 张量槽错误码 (TensorSlotErr)

### LOGICAL_TENSOR_NOT_FOUND_IN_INPUT (F25001)

**错误描述：** TensorSlot 逻辑张量在输入中未找到

**出现原因：**
- 逻辑张量不在输入列表中
- 输入参数不匹配

**解决办法：**
- 检查输入参数
- 确保所有逻辑张量都在输入中
- 验证函数签名

**示例：**

```python
# 错误示例 - 逻辑张量不在输入中
@pypto.frontend.jit
def tensorslot_not_in_input_example():
    # 使用未在输入中的逻辑张量
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    return x  # x 不在输入中

# 正确示例
@pypto.frontend.jit
def correct_tensorslotlot_example(x):
    # x 在输入中
    y = pypto.add(x, 1)
    return y
```

### LOGICAL_TENSOR_NOT_FOUND_IN_OUTPUT (F25002)

**错误描述：** TensorSlot 逻辑张量在输出中未找到

**出现原因：**
- 逻辑张量不在输出列表中
- 输出参数不匹配

**解决办法：**
- 检查输出参数
- 确保所有逻辑张量都在输出中
- 验证函数签名

**示例：**

```python
# 错误示例 - 逻辑张量不在输出中
# 会导致逻辑张量未找到错误

# 正确示例
# 确保逻辑张量在输出中
```

### NOT_FOUND_IN_INDEX_DICT (F25003)

**错误描述：** TensorSlot 在索引字典中未找到

**出现原因：**
- TensorSlot 未正确注册
- 索引查找失败

**解决办法：**
- 检查 TensorSlot 注册
- 确保索引正确
- 验证 TensorSlot 存在性

**示例：**

```python
# 错误示例 - TensorSlot 未找到
@pypto.frontend.jit
def tensorslot_not_found_example(x):
    # 尝试访问未注册的 TensorSlot
    y = pypto.get_tensorslot(999)  # 无效的索引
    return y

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # 使用有效的 TensorSlot
    y = pypto.add(x, 1)
    return y
```

### SLOT_SCOPE_NULL (F2500C)

**错误描述：** 槽位作用域为空

**出现原因：**
- 槽位作用域为空
- 槽位未正确初始化

**解决办法：**
- 确保槽位作用域不为空
- 检查槽位初始化

**示例：**

```python
# 错误示例 - 槽位作用域为空
# 会导致槽位作用域错误

# 正确示例
# 确保槽位作用域不为空
```

---

## 符号标量错误码 (SymScalarErr)

### OPERAND_SIZE_INVALID (F26022)

**错误描述：** 符号操作数大小无效

**出现原因：**
- 操作数大小无效
- 操作数维度不正确

**解决办法：**
- 检查操作数大小
- 使用正确的操作数大小
- 验证操作数维度

**示例：**

```python
# 错误示例 - 操作数大小无效
# 会导致操作数大小错误

# 正确示例
# 使用正确的操作数大小
```

### ELEMENT_KEY_MISMATCH (F26035)

**错误描述：** 元素键不匹配

**出现原因：**
- 元素键不匹配
- 键值对不正确

**解决办法：**
- 检查元素键
- 确保键值匹配

**示例：**

```python
# 错误示例 - 元素键不匹配
# 会导致键不匹配错误

# 正确示例
# 确保元素键正确
```

### TITLE_MISMATCH (F26036)

**错误描述：** 标题不匹配

**出现原因：**
- 标题不匹配
- 标题字符串不正确

**解决办法：**
- 检查标题
- 确保标题正确

**示例：**

```python
# 错误示例 - 标题不匹配
# 会导致标题错误

# 正确示例
# 确保标题正确
```

### SYMBOL_NOT_FOUND_IN_VALUE_DICT (F26037)

**错误描述：** 符号在值字典中未找到

**出现原因：**
- 符号未在值字典中注册
- 符号查找失败

**解决办法：**
- 检查符号值字典
- 确保符号已注册

**示例：**

```python
# 错误示例 - 符号在值字典中未找到
# 会导致符号未找到错误

# 正确示例
# 确保符号在值字典中
```

### TYPE_MISMATCH (F26041)

**错误描述：** 类型不匹配

**出现原因：**
- 符号标量类型不匹配
- 期望的类型与实际类型不符

**解决办法：**
- 检查符号标量类型
- 使用正确的类型
- 确保类型匹配

**示例：**

```python
# 错误示例 - 类型不匹配
# 会导致类型不匹配错误

# 正确示例
# 确保类型正确
```

---

## 文件错误码 (FileErr)

### FILE_OPEN_FAILED (F27001)

**错误描述：** 文件打开失败

**出现原因：**
- 文件不存在
- 文件权限不足
- 文件路径错误
- 在 DumpJsonFile 中无法打开文件

**解决办法：**
- 检查文件是否存在
- 确保文件权限正确
- 使用正确的文件路径

**测试用例：**

```cpp
// C++ 测试用例 - 文件打开失败
// 测试场景：在 DumpJsonFile 中无法打开文件

#include <gtest/gtest.h>
#include "interface/program/program.h"
#include "interface/tensor/raw_tensor.h"
#include <fstream>

TEST(ProgramErrorTest, FileOpenFailed) {
    auto &program = npu::tile_fwk::Program::GetInstance();
    program.Reset();
    
    // 创建一个函数
    std::string funcName = "test_function";
    program.BeginFunction(funcName, npu::tile_fwk::FunctionType::EAGER,
                         npu::tile_fwk::GraphType::TENSOR_GRAPH, {}, false);
    
    auto currentFunc = program.GetCurrentFunction();
    ASSERT_NE(currentFunc, nullptr);
    
    // 尝试 DumpJsonFile 到一个不存在的路径或只读路径
    std::string invalidPath = "/nonexistent_directory/test_program.json";
    
    // 尝试调用 DumpJsonFile，但文件无法打开
    // 这会触发 ASSERT(ProgErr::FILE_OPEN_FAILED, ...)
    // 错误信息： "Failed to open file: /nonexistent_directory/test_program.json"
    
    // 由于 ASSERT 宏会终止程序，这里需要使用 EXPECT_DEATH 或类似机制
    // 在实际测试中，可能需要 mock 或修改测试框架
    
    // 正确的做法：确保文件路径正确且有写入权限
    std::string validPath = "./test_program.json";
    program.DumpJsonFile(validPath, currentFunc);
    
    program.EndFunction(funcName, true);
    
    // 清理测试文件
    std::remove(validPath.c_str());
}
```
