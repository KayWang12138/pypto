#!/bin/bash

# 获取当前工作目录或从环境变量获取 OpenCode 工作区根目录
WORKSPACE_ROOT="${OPENCODE_WORKSPACE_ROOT:-$(pwd)}"
CURRENT_DIR="$WORKSPACE_ROOT"
# source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash
# export PTO_TILE_LIB_CODE_PATH="/home/d00899108/pto-isa"
usage() {
    echo "用法: st [-c] [-s SCRIPT_PATH]"
    echo "  -c          编译安装后再运行"
    echo "  -s SCRIPT   指定要运行的 Python 脚本路径 (默认: issue.py)"
    exit 1
}

# 默认设置
NEED_COMPILE=false
SCRIPT_PATH="examples/00_hello_world/hello_world.py"

# 解析参数 (OpenCode 会将参数透传给此脚本)
while getopts "cs:" opt; do
  case $opt in
    c) NEED_COMPILE=true ;;
    s) SCRIPT_PATH="$OPTARG" ;;
    \?) usage ;;
  esac
done

# 转换脚本路径为绝对路径
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="${CURRENT_DIR}/${SCRIPT_PATH}"
fi

# 1. 编译流程
if [ "$NEED_COMPILE" = true ]; then
    echo "--- 开始执行编译流程 ---"
    cd "$CURRENT_DIR" || exit 1
    rm -rf build_out
    # 确保 python3 环境可用
    python3 build_ci.py -f=python3
    
    if [ ! -d "build_out" ]; then
        echo "错误: 编译失败，build_out 目录未生成。"
        exit 1
    fi

    cd build_out
    # 使用通配符安装 whl
    pip3 install pypto-*.whl --target="$(pwd)/build_out"
    cd ..
    echo "--- 编译安装完成 ---"
fi

# 2. 设置环境变量
install_path="${CURRENT_DIR}/build_out/build_out"

if [ ! -d "$install_path" ]; then
    echo "错误: 找不到路径 $install_path。如果是首次运行，请添加 -c 参数。"
    exit 1
fi

# 关键：在当前子进程设置环境变量
export LD_LIBRARY_PATH="${install_path}/pto/lib/:${LD_LIBRARY_PATH}"
export PYTHONPATH="${install_path}:${PYTHONPATH}"

# 3. 执行脚本
if [ -f "$SCRIPT_PATH" ]; then
    echo "正在启动脚本: $SCRIPT_PATH"
    python3 "$SCRIPT_PATH"
else
    echo "错误: 脚本文件不存在: $SCRIPT_PATH"
    exit 1
fi