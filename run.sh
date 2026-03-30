#!/bin/bash

usage() {
    echo "用法: $0 [-c] [-s SCRIPT_PATH]"
    echo "  -c          编译安装后再运行"
    echo "  -s SCRIPT   指定要运行的 Python 脚本路径 (默认: issue.py)"
    exit 1
}

# 默认不编译
NEED_COMPILE=false
SCRIPT_PATH="${CURRENT_DIR:-$(pwd)}/issue.py"

# 2. 解析参数
while getopts "cs:" opt; do
  case $opt in
    c)
      NEED_COMPILE=true
      ;;
    s)
      SCRIPT_PATH="$OPTARG"
      ;;
    \?)
      echo "无效参数: -$OPTARG"
      usage
      ;;
  esac
done


# 4. 如果指定了 -c 参数，则执行编译安装流程
if [ "$NEED_COMPILE" = true ]; then
    echo "检测到 -c 参数，开始执行编译流程..."
    
    # 清理并构建
    rm -rf build_out
    python3 build_ci.py -f=python3 --no_isolation --generator Ninja
    
    # 进入目录并安装
    cd build_out
    # 注意：确保该 whl 文件在 build_out 目录下存在
    pip3 install pypto-*.whl --target="$(pwd)/build_out"
    cd ..
    
    echo "编译安装完成。"
else
    echo "未指定 -c 参数，跳过编译步骤，直接运行。"
fi

# 5. 设置环境变量 (运行前必须设置)
# 获取当前绝对路径，确保变量拼接准确
CURRENT_DIR=$(pwd)
install_path="${CURRENT_DIR}/build_out/build_out"

# 检查路径是否存在，防止没编译过就直接运行导致报错
if [ ! -d "$install_path" ]; then
    echo "错误: 找不到路径 $install_path。如果是首次运行，请使用 -c 参数进行编译。"
    exit 1
fi

export LD_LIBRARY_PATH=${install_path}/pto/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=${install_path}:${PYTHONPATH}

# 6. 执行 Python 脚本
echo "正在启动脚本: $SCRIPT_PATH"
python3 "$SCRIPT_PATH"
