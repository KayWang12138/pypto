# Backtrace 优化前后对比测试指南

## 概述

本文档说明如何使用 `test_backtrace_compare.cpp` 测试用例，在两个分支上运行相同的测试，直观对比优化前后的backtrace输出格式差异。

## 分支说明

- **origin_cmp** - 原始主线代码（优化前）
  - 传统C++格式：`libname(function+offset)`
  - 无文件/行号信息
  - 无源代码显示

- **modify_backtrack** - 带优化的代码（优化后）
  - Python风格格式：`File "path", line X`
  - 清晰的traceback标题
  - 源代码行显示
  - 堆栈反转（最近的调用在最后）

---

## 快速对比测试

### 方法1: 分支切换对比

```bash
# 1. 切换到原始分支
git checkout origin_cmp

# 2. 运行测试，保存输出
python3 build_ci.py -u=BacktraceCompareTest.SimpleErrorTest -d=11 -f cpp 2>&1 | tee output_before.log

# 3. 切换到优化分支
git checkout modify_backtrack

# 4. 运行相同测试，保存输出
python3 build_ci.py -u=BacktraceCompareTest.SimpleErrorTest -d=11 -f cpp 2>&1 | tee output_after.log

# 5. 对比输出
echo "========================================="
echo "优化前后输出对比："
echo "========================================="
diff -y --width=160 output_before.log output_after.log | less
```

### 方法2: 并排查看

```bash
# 在origin_cmp分支运行
git checkout origin_cmp
python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp 2>&1 > before.txt

# 在modify_backtrack分支运行
git checkout modify_backtrack
python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp 2>&1 > after.txt

# 使用vimdiff并排查看
vimdiff before.txt after.txt
```

---

## 测试用例说明

### 1. SimpleErrorTest（推荐首选）

**用途：** 最简单直观的对比测试

**运行命令：**
```bash
python3 build_ci.py -u=BacktraceCompareTest.SimpleErrorTest -d=11 -f cpp
```

**预期输出差异：**

**优化前（origin_cmp）：**
```
Test error: Invalid operation, func SimpleErrorTest, file test_backtrace_compare.cpp, line 65
./libpypto_impl.so(+0x123456) [0x7fff12345678]
./libpypto_impl.so(+0x234567) [0x7fff23456789]
```

**优化后（modify_backtrack）：**
```
Test error: Invalid operation, func SimpleErrorTest, file test_backtrace_compare.cpp, line 65

C++ Traceback (most recent call last):
 File "/path/to/test_backtrace_compare.cpp", line 58
   TEST_F(BacktraceCompareTest, SimpleErrorTest) {
 File "/path/to/test_backtrace_compare.cpp", line 65
   throw Error(__func__, __FILE__, __LINE__,
```

---

### 2. NestedCallTest（推荐）

**用途：** 展示多层函数调用的堆栈效果

**运行命令：**
```bash
python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp
```

**特点：**
- 3层函数嵌套调用（Level1 → Level2 → Level3）
- 清晰展示调用链
- 优化后可以看到每一层的源代码

---

### 3. DeepCallStackTest

**用途：** 展示更深层次的调用栈（5层）

**运行命令：**
```bash
python3 build_ci.py -u=BacktraceCompareTest.DeepCallStackTest -d=11 -f cpp
```

**特点：**
- 5层深度嵌套
- 测试堆栈反转效果
- 验证符号缓存优化

---

### 4. MultipleErrorsTest

**用途：** 测试多次错误的一致性

**运行命令：**
```bash
python3 build_ci.py -u=BacktraceCompareTest.MultipleErrorsTest -d=11 -f cpp
```

**特点：**
- 连续抛出多个错误
- 验证格式化一致性
- 测试符号缓存效果

---

### 5. AssertMacroTest

**用途：** 测试ASSERT宏的backtrace

**运行命令：**
```bash
python3 build_ci.py -u=BacktraceCompareTest.AssertMacroTest -d=11 -f cpp
```

**特点：**
- 使用ASSERT宏触发
- 包含断言表达式
- 展示断言失败的上下文

---

## 预期输出格式对比

### 优化前（origin_cmp分支）

```
========================================
BACKTRACE OUTPUT TEST
========================================

Test: Simple Error Exception
----------------------------------------

Caught Error Exception:
Test error: Invalid operation, func SimpleErrorTest, file test_backtrace_compare.cpp, line 65
./libpypto_impl.so(+0x123456) [0x7fff12345678]
./libpypto_impl.so(_ZN3npu8tile_fwk5ErrorC1EPKcS3_mNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_10__weak_ptrINS_9LazyValueIS9_EEEE+0x234) [0x7fff23456789]
/lib64/libc.so.6(__libc_start_main+0xf5) [0x7ffff7a05555]

========================================
```

**特点：**
- ❌ 难以阅读的地址和符号
- ❌ 没有文件路径和行号
- ❌ 没有源代码显示
- ❌ mangled的C++函数名

---

### 优化后（modify_backtrack分支）

```
========================================
BACKTRACE OUTPUT TEST
========================================

Test: Simple Error Exception
----------------------------------------

Caught Error Exception:
Test error: Invalid operation, func SimpleErrorTest, file test_backtrace_compare.cpp, line 65

C++ Traceback (most recent call last):
 File "/data/g00655722/new-ir/pypto_open/framework/tests/ut/interface/src/ir/test_backtrace_compare.cpp", line 58
   TEST_F(BacktraceCompareTest, SimpleErrorTest) {
 File "/data/g00655722/new-ir/pypto_open/framework/tests/ut/interface/src/ir/test_backtrace_compare.cpp", line 65
   throw Error(__func__, __FILE__, __LINE__,

========================================
```

**特点：**
- ✅ 清晰的Python风格标题
- ✅ 完整的文件路径和行号
- ✅ 源代码行显示（缩进3个空格）
- ✅ 堆栈从上到下（最近的调用在最后）
- ✅ 易于定位问题

---

## 关键改进点

| 特性 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| **标题** | 无 | `C++ Traceback (most recent call last):` | ✅ 清晰标识 |
| **格式** | `lib(func+offset)` | `File "path", line X` | ✅ Python风格 |
| **源码** | 无 | 显示实际代码行 | ✅ 易于定位 |
| **顺序** | 最近的在前 | 最近的在后 | ✅ 符合习惯 |
| **可读性** | 低 | 高 | ✅ 显著提升 |

---

## 完整测试流程示例

```bash
#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Backtrace 优化前后对比测试"
echo "=============================================="

# 测试用例列表
TESTS=(
    "BacktraceCompareTest.SimpleErrorTest"
    "BacktraceCompareTest.NestedCallTest"
    "BacktraceCompareTest.DeepCallStackTest"
)

for test in "${TESTS[@]}"; do
    echo ""
    echo -e "${YELLOW}运行测试: $test${NC}"
    echo "----------------------------------------------"

    # 在 origin_cmp 分支运行
    echo -e "${RED}[优化前 - origin_cmp]${NC}"
    git checkout origin_cmp > /dev/null 2>&1
    python3 build_ci.py -u=$test -d=11 -f cpp 2>&1 | grep -A 20 "Caught"

    echo ""
    echo -e "${GREEN}[优化后 - modify_backtrack]${NC}"
    git checkout modify_backtrack > /dev/null 2>&1
    python3 build_ci.py -u=$test -d=11 -f cpp 2>&1 | grep -A 20 "Caught"

    echo "=============================================="
done

echo ""
echo "测试完成！"
```

保存为 `run_comparison_tests.sh`，然后运行：

```bash
chmod +x run_comparison_tests.sh
./run_comparison_tests.sh
```

---

## 注意事项

1. **确保两个分支都已编译**
   ```bash
   git checkout origin_cmp && python3 build_ci.py -b
   git checkout modify_backtrack && python3 build_ci.py -b
   ```

2. **使用 -d=11 选项**
   - 确保详细输出，能看到完整的backtrace信息

3. **符号信息依赖**
   - 优化后的版本需要系统中有 `addr2line` 工具
   - 需要编译时包含调试信息（`-g` 选项）

4. **输出重定向**
   - 建议将输出重定向到文件以便详细对比
   - 使用 `2>&1` 捕获所有输出

---

## 故障排查

### 问题1: 看不到源代码行

**原因：** 编译时未包含调试信息或二进制被strip

**解决：**
```bash
# 确保使用Debug模式编译
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### 问题2: addr2line not found

**原因：** 系统中没有安装 binutils

**解决：**
```bash
# CentOS/RHEL
sudo yum install binutils

# Ubuntu/Debian
sudo apt-get install binutils
```

### 问题3: 符号缓存不生效

**原因：** 每次测试都是新进程，缓存在进程内

**说明：** 这是正常的，符号缓存只在同一进程的多次调用时生效

---

## 总结

通过在两个分支上运行相同的测试用例，可以直观地看到backtrace优化带来的改进：

1. ✅ **格式清晰** - Python风格更易读
2. ✅ **信息完整** - 文件名、行号、源代码都有
3. ✅ **定位快速** - 直接看到出错的代码
4. ✅ **习惯友好** - 符合Python用户习惯

推荐从 `SimpleErrorTest` 开始测试，这是最直观的对比。
