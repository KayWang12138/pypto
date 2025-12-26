#!/bin/bash
# PyPTO 构建和运行脚本
set -e

echo "======================================"
echo "PyPTO 构建和运行脚本"
echo "======================================"

cd /Users/hengliao/Documents/PyPTO

# 清理旧的构建缓存
echo "清理构建缓存..."
rm -rf build third_party_path

# 设置环境变量
export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
export PATH="/Users/hengliao/Library/Python/3.9/bin:/opt/homebrew/bin:$PATH"

echo "使用编译器: $CC"
echo "CMake 路径: $(which cmake)"

# 构建
echo ""
echo "开始构建..."
python3 -m pip install . --no-build-isolation --force-reinstall

# 验证安装
echo ""
echo "======================================"
echo "验证安装..."
python3 -c "import pypto; print('PyPTO 导入成功!')"

# 运行示例
echo ""
echo "======================================"
echo "运行 Hello World 示例..."
cd /Users/hengliao/Documents/PyPTO/examples/hello_world
python3 hello_world.py

echo ""
echo "======================================"
echo "构建和测试完成!"

