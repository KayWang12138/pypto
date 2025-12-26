#!/bin/bash
# PyPTO macOS 构建脚本
# 自动检测和配置构建环境

set -e

echo "====================================="
echo "PyPTO macOS 构建脚本"
echo "====================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测 Homebrew
check_homebrew() {
    if command -v brew &> /dev/null; then
        echo -e "${GREEN}✓ Homebrew 已安装${NC}"
        return 0
    else
        echo -e "${RED}✗ Homebrew 未安装${NC}"
        echo -e "${YELLOW}请先安装 Homebrew:${NC}"
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        echo ""
        echo "安装完成后，请根据提示添加 Homebrew 到 PATH，然后重新运行此脚本。"
        return 1
    fi
}

# 检测 GCC
check_gcc() {
    local gcc_found=0
    local gcc_path=""
    local gxx_path=""
    
    # 检查常见路径
    for path in /opt/homebrew/bin /usr/local/bin; do
        if [ -d "$path" ]; then
            for gcc in $(ls "$path"/gcc-* 2>/dev/null | grep -E 'gcc-[0-9]+$' | sort -rV); do
                local ver=$(basename "$gcc" | sed 's/gcc-//')
                local gxx="$path/g++-$ver"
                if [ -x "$gcc" ] && [ -x "$gxx" ]; then
                    gcc_path="$gcc"
                    gxx_path="$gxx"
                    gcc_found=1
                    break 2
                fi
            done
        fi
    done
    
    if [ $gcc_found -eq 1 ]; then
        echo -e "${GREEN}✓ GCC 已安装: $gcc_path${NC}"
        export CC="$gcc_path"
        export CXX="$gxx_path"
        return 0
    else
        echo -e "${RED}✗ GCC 未安装${NC}"
        echo -e "${YELLOW}请安装 GCC:${NC}"
        echo "  brew install gcc"
        return 1
    fi
}

# 检测 cmake
check_cmake() {
    # 检查 pip 安装的 cmake
    local user_bin=$(python3 -m site --user-base 2>/dev/null)/bin
    if [ -x "$user_bin/cmake" ]; then
        echo -e "${GREEN}✓ CMake 已安装 (pip): $user_bin/cmake${NC}"
        export PATH="$user_bin:$PATH"
        return 0
    fi
    
    # 检查 Homebrew 安装的 cmake
    if command -v cmake &> /dev/null; then
        echo -e "${GREEN}✓ CMake 已安装: $(which cmake)${NC}"
        return 0
    fi
    
    echo -e "${RED}✗ CMake 未安装${NC}"
    echo -e "${YELLOW}请安装 CMake:${NC}"
    echo "  pip3 install cmake"
    echo "  或"
    echo "  brew install cmake"
    return 1
}

# 主流程
main() {
    echo ""
    echo "检查构建环境..."
    echo ""
    
    local all_ok=1
    
    # 检查 Homebrew
    if ! check_homebrew; then
        all_ok=0
    fi
    
    # 检查 GCC
    if ! check_gcc; then
        all_ok=0
    fi
    
    # 检查 cmake
    if ! check_cmake; then
        all_ok=0
    fi
    
    echo ""
    
    if [ $all_ok -eq 0 ]; then
        echo -e "${RED}环境检查未通过，请安装缺失的组件后重试。${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}环境检查通过！${NC}"
    echo ""
    echo "当前配置:"
    echo "  CC=$CC"
    echo "  CXX=$CXX"
    echo "  cmake=$(which cmake)"
    echo ""
    
    # 询问是否开始构建
    read -p "是否开始构建 PyPTO? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "已取消构建。"
        echo ""
        echo "您可以手动运行以下命令进行构建:"
        echo "  export CC=$CC"
        echo "  export CXX=$CXX"
        echo "  export PATH=\"$(dirname $(which cmake)):\$PATH\""
        echo "  cd $(dirname "$0")"
        echo "  python3 -m pip install . --no-build-isolation"
        exit 0
    fi
    
    # 开始构建
    echo ""
    echo "开始构建 PyPTO..."
    cd "$(dirname "$0")"
    python3 -m pip install . --no-build-isolation --force-reinstall
    
    echo ""
    echo -e "${GREEN}构建完成！${NC}"
    echo ""
    echo "验证安装:"
    python3 -c "import pypto; print('PyPTO 导入成功!')"
}

main "$@"

