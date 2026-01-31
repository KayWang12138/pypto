# 如何使能主线的addr2line功能

## 问题分析

当前backtrace优化代码已经成功实现：
- ✅ Python风格标题 `C++ Traceback (most recent call last):`
- ✅ 堆栈反转（最近的调用在最后）

但是File/Line格式和源代码显示功能未生效，仍显示传统格式：
```
tile_fwk_utest(Level3Function()+0x60) [0x698450]
```

**原因：** 编译时未包含调试符号（-g选项），导致addr2line无法解析文件名和行号。

---

## 解决方案

### ✅ 方案1：使用Debug构建模式（推荐）

当前主线默认使用Release模式，需要切换到Debug模式：

```bash
cd /data/g00655722/new-ir/pypto_open
rm -rf build
mkdir build && cd build

# 使用Debug模式，自动包含-g选项
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8

# 重新运行测试
python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp
```

**原理：**
根据 `/data/g00655722/new-ir/pypto_open/cmake/intf.cmake:25`，编译选项配置为：
```cmake
$<$<CONFIG:Debug>:-g>
```
这意味着Debug模式会自动添加`-g`选项生成调试符号。

---

### 方案2：修改默认构建类型

如果想让默认构建包含调试符号，修改 `/data/g00655722/new-ir/pypto_open/cmake/config.cmake:125`：

```cmake
# 修改前
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type(default Release)" FORCE)

# 修改后
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type(default Debug)" FORCE)
```

然后重新编译：
```bash
cd /data/g00655722/new-ir/pypto_open
rm -rf build
mkdir build && cd build
cmake ..
make -j8
```

---

### 方案3：在Release模式添加-g选项

修改 `/data/g00655722/new-ir/pypto_open/cmake/intf.cmake`，在编译选项中添加：

```cmake
target_compile_options(tile_fwk_intf_pub
    INTERFACE
        # ... 其他选项 ...
        -g  # 添加这一行，在所有模式下都生成调试符号
)
```

**优点：** 不影响优化级别，Release模式仍然有优化
**缺点：** 增加二进制文件大小

---

## 验证方法

### 1. 检查编译选项

编译后检查是否包含-g选项：

```bash
cd /data/g00655722/new-ir/pypto_open/build
grep -- "-g" compile_commands.json
```

### 2. 检查二进制文件

检查生成的二进制文件是否包含调试符号：

```bash
# 查找测试二进制文件
find /data/g00655722/new-ir/pypto_open -name "*tile_fwk_utest*" -type f

# 检查是否被strip
file <path_to_binary>

# 期望输出包含: "not stripped"
# 如果看到 "stripped"，说明调试符号被移除了
```

### 3. 测试addr2line工具

手动测试addr2line是否能解析符号：

```bash
BINARY=<path_to_tile_fwk_utest>

# 获取一个函数地址
nm $BINARY | grep " main$"

# 测试addr2line（替换0xADDRESS为实际地址）
addr2line -e $BINARY -f -C -p 0xADDRESS
```

**期望输出：**
```
function_name at /path/to/file.cpp:123
```

**如果输出：**
```
??:?
```
说明没有调试符号。

---

## 快速验证脚本

创建并运行以下脚本进行完整验证：

```bash
#!/bin/bash

echo "==================================="
echo "Addr2line功能验证脚本"
echo "==================================="
echo ""

# 1. 检查构建类型
echo "1. 检查CMAKE_BUILD_TYPE..."
if [ -f "/data/g00655722/new-ir/pypto_open/build/CMakeCache.txt" ]; then
    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE:" /data/g00655722/new-ir/pypto_open/build/CMakeCache.txt | cut -d'=' -f2)
    echo "   当前构建类型: $BUILD_TYPE"

    if [ "$BUILD_TYPE" = "Debug" ]; then
        echo "   ✓ 使用Debug模式，应该包含调试符号"
    else
        echo "   ✗ 使用$BUILD_TYPE模式，可能没有调试符号"
        echo "   建议: cmake -DCMAKE_BUILD_TYPE=Debug .."
    fi
else
    echo "   ✗ 找不到build目录，请先编译"
fi
echo ""

# 2. 检查addr2line工具
echo "2. 检查addr2line工具..."
if command -v addr2line &> /dev/null; then
    echo "   ✓ addr2line可用: $(which addr2line)"
    addr2line --version | head -1 | sed 's/^/   版本: /'
else
    echo "   ✗ addr2line不可用"
    echo "   安装方法: yum install binutils (或 apt-get install binutils)"
fi
echo ""

# 3. 检查测试二进制
echo "3. 检查测试二进制文件..."
BINARY=$(find /data/g00655722/new-ir/pypto_open -name "*tile_fwk_utest*" -type f 2>/dev/null | head -1)
if [ -n "$BINARY" ]; then
    echo "   找到二进制: $BINARY"

    FILE_INFO=$(file "$BINARY")
    if echo "$FILE_INFO" | grep -q "not stripped"; then
        echo "   ✓ 二进制未被strip，包含调试符号"
    else
        echo "   ✗ 二进制已被strip，调试符号被移除"
        echo "   建议: 重新用Debug模式编译"
    fi
else
    echo "   ✗ 找不到测试二进制文件"
fi
echo ""

# 4. 测试addr2line
if [ -n "$BINARY" ] && command -v addr2line &> /dev/null && command -v nm &> /dev/null; then
    echo "4. 测试addr2line解析..."
    MAIN_ADDR=$(nm "$BINARY" 2>/dev/null | grep " main$" | awk '{print $1}')
    if [ -n "$MAIN_ADDR" ]; then
        echo "   找到main函数地址: 0x$MAIN_ADDR"
        echo "   运行: addr2line -e $BINARY -f -C -p 0x$MAIN_ADDR"
        RESULT=$(addr2line -e "$BINARY" -f -C -p "0x$MAIN_ADDR" 2>&1)
        echo "   结果: $RESULT"

        if echo "$RESULT" | grep -q "??:?"; then
            echo "   ✗ 无法解析符号，需要调试信息"
        else
            echo "   ✓ 成功解析符号"
        fi
    else
        echo "   ✗ 无法找到main函数符号"
    fi
fi
echo ""

echo "==================================="
echo "验证完成"
echo "==================================="
```

保存为 `check_addr2line.sh` 并运行：

```bash
chmod +x check_addr2line.sh
./check_addr2line.sh
```

---

## 预期效果对比

### 当前（无调试符号）

```
C++ Traceback (most recent call last):
tile_fwk_utest(_start+0x30) [0x83f4b0]
...
tile_fwk_utest(Level3Function()+0x60) [0x698450]
```

### Debug模式后（有调试符号）

```
C++ Traceback (most recent call last):
 File "/data/g00655722/new-ir/pypto_open/framework/tests/ut/interface/src/ir/test_backtrace_compare.cpp", line 92
   Level1Function();
 File "/data/g00655722/new-ir/pypto_open/framework/tests/ut/interface/src/ir/test_backtrace_compare.cpp", line 89
   Level2Function();
 File "/data/g00655722/new-ir/pypto_open/framework/tests/ut/interface/src/ir/test_backtrace_compare.cpp", line 85
   throw Error(__func__, __FILE__, __LINE__,
```

---

## 总结

**核心问题：** 主线默认使用Release模式，不包含调试符号（-g）

**最简单解决方案：** 使用Debug模式编译

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8
python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp
```

这将完全展示backtrace优化的所有功能：
1. ✅ Python风格标题
2. ✅ 堆栈反转
3. ✅ File/Line格式（需要调试符号）
4. ✅ 源代码显示（需要调试符号）
