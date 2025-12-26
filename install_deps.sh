#!/bin/bash
# PyPTO 依赖安装脚本 - macOS
# 安装 Homebrew 和 GCC

set -e

echo "====================================="
echo "PyPTO 依赖安装脚本"
echo "====================================="
echo ""

# 安装 Homebrew
if ! command -v brew &> /dev/null; then
    echo "正在安装 Homebrew..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # 配置 Homebrew 环境变量
    if [ -f /opt/homebrew/bin/brew ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    echo "Homebrew 安装完成!"
else
    echo "Homebrew 已安装: $(which brew)"
fi

echo ""

# 安装 GCC
echo "正在安装 GCC..."
brew install gcc

echo ""
echo "====================================="
echo "安装完成!"
echo "====================================="
echo ""

# 查找安装的 GCC 版本
GCC_PATH=""
GXX_PATH=""
for path in /opt/homebrew/bin /usr/local/bin; do
    if [ -d "$path" ]; then
        for gcc in $(ls "$path"/gcc-* 2>/dev/null | grep -E 'gcc-[0-9]+$' | sort -rV | head -1); do
            if [ -x "$gcc" ]; then
                ver=$(basename "$gcc" | sed 's/gcc-//')
                GCC_PATH="$gcc"
                GXX_PATH="$path/g++-$ver"
                break 2
            fi
        done
    fi
done

if [ -n "$GCC_PATH" ]; then
    echo "GCC 已安装: $GCC_PATH"
    echo ""
    echo "现在可以构建 PyPTO:"
    echo ""
    echo "  export CC=$GCC_PATH"
    echo "  export CXX=$GXX_PATH"
    echo "  cd /Users/hengliao/Documents/PyPTO"
    echo "  python3 -m pip install . --no-build-isolation"
else
    echo "错误: 未找到 GCC"
    exit 1
fi

