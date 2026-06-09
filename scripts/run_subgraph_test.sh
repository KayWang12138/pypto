#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# 子图 CostModel 一键运行脚本
#
# 用法:
#   bash run_subgraph_test.sh
#
# 功能:
#   1. 检查环境变量
#   2. cmake + make 编译
#   3. 查找编译产物
#   4. 运行子图 costmodel 测试
# -----------------------------------------------------------------------------------------------------------

set -e

# ─── 颜色 ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── 项目根目录（假设脚本在项目根目录或 scripts/ 下）───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$(dirname "$SCRIPT_DIR")/CMakeLists.txt" ]; then
    PROJECT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"
else
    echo -e "${RED}找不到项目根目录，请在项目根目录或 scripts/ 下运行${NC}"
    exit 1
fi
cd "$PROJECT_DIR"

echo ""
echo "============================================================"
echo "  子图 CostModel 一键运行"
echo "  项目目录: $PROJECT_DIR"
echo "============================================================"
echo ""

# ─── Step 0: 环境变量检查 ───
echo -e "${CYAN}[Step 0] 检查环境变量 ...${NC}"

if [ -z "$ASCEND_PATH" ]; then
    echo -e "${YELLOW}  ASCEND_PATH 未设置，尝试自动检测 ...${NC}"
    for p in /usr/local/Ascend/ascend-toolkit /usr/local/Ascend/cann /home/developer/Ascend/cann; do
        if [ -d "$p" ]; then
            export ASCEND_PATH="$p"
            echo -e "${GREEN}  自动设置 ASCEND_PATH=$p${NC}"
            break
        fi
    done
fi

if [ -z "$ASCEND_PATH" ]; then
    echo -e "${RED}  [FAIL] ASCEND_PATH 未设置且无法自动检测${NC}"
    echo "  请执行: export ASCEND_PATH=/path/to/ascend-toolkit"
    exit 1
fi

export TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID:-0}
echo -e "  ASCEND_PATH=$ASCEND_PATH"
echo -e "  TILE_FWK_DEVICE_ID=$TILE_FWK_DEVICE_ID"

# ─── Step 1: cmake 配置 ───
echo ""
echo -e "${CYAN}[Step 1] cmake 配置 ...${NC}"

BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "  首次 cmake ..."
    cmake .. -DENABLE_TESTS_STEST=ON
else
    echo "  CMakeCache 已存在，跳过 cmake（如需重新配置请删除 build/ 目录）"
fi

# ─── Step 2: make 编译 ───
echo ""
echo -e "${CYAN}[Step 2] make 编译 (make -j$(nproc)) ...${NC}"

cd "$BUILD_DIR"
make -j$(nproc)

echo -e "${GREEN}  编译完成${NC}"

# ─── Step 3: 查找编译产物 ───
echo ""
echo -e "${CYAN}[Step 3] 查找编译产物 ...${NC}"

cd "$PROJECT_DIR"

# 创建 output 目录
mkdir -p ./output/libs

# 复制 .so 文件
for so in \
    framework/src/interface/libtile_fwk_interface.so \
    framework/src/codegen/libtile_fwk_codegen.so \
    framework/src/machine/runtime/libtile_fwk_runtime.so \
    framework/src/machine/device/libtilefwk_backend_server.so \
    framework/src/passes/libtile_fwk_passes.so \
    framework/src/cost_model/simulation/libtile_fwk_simulation.so \
    framework/src/cost_model/simulation_ca/libtile_fwk_simulation_ca.so \
    framework/src/operator/libtile_fwk_operator.so \
    framework/src/machine/libtile_fwk_compiler.so; do
    if [ -f "$BUILD_DIR/$so" ]; then
        cp "$BUILD_DIR/$so" ./output/libs/
        echo -e "  ${GREEN}✓${NC} $(basename $so)"
    fi
done

# 查找 pto_impl 绑定模块
PTO_IMPL_SO=$(find "$BUILD_DIR" -name "pto_impl*.so" 2>/dev/null | head -1)
if [ -z "$PTO_IMPL_SO" ]; then
    # 可能在 python/ 目录下
    PTO_IMPL_SO=$(find "$PROJECT_DIR/python" -name "pto_impl*.so" 2>/dev/null | head -1)
fi

if [ -n "$PTO_IMPL_SO" ]; then
    PTO_IMPL_DIR=$(dirname "$PTO_IMPL_SO")
    echo -e "  ${GREEN}✓${NC} pto_impl: $PTO_IMPL_SO"
else
    echo -e "  ${YELLOW}⚠ 未找到 pto_impl*.so，可能需要单独编译 Python 绑定${NC}"
    echo -e "  尝试: cd build && make pto_impl -j"
    PTO_IMPL_DIR=""
fi

# ─── Step 4: 设置运行环境 ───
echo ""
echo -e "${CYAN}[Step 4] 设置运行环境 ...${NC}"

cd "$PROJECT_DIR"

export LD_LIBRARY_PATH="$PROJECT_DIR/output/libs:${ASCEND_PATH}/lib64:${LD_LIBRARY_PATH:-}"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

if [ -n "$PTO_IMPL_DIR" ]; then
    export PYTHONPATH="$PTO_IMPL_DIR:$PROJECT_DIR/python:${PYTHONPATH:-}"
    echo "  PYTHONPATH 包含 pto_impl 路径"
else
    export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"
fi

# ─── Step 5: 运行子图 costmodel 测试 ───
echo ""
echo "============================================================"
echo -e "${CYAN}  [Step 5] 运行子图 CostModel 测试${NC}"
echo "============================================================"
echo ""

TEST_FILE="python/tests/st/test_costmodel_subgraph.py"

if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}  [FAIL] 测试文件不存在: $TEST_FILE${NC}"
    exit 1
fi

python3 "$TEST_FILE"
TEST_EXIT=$?

# ─── 汇总 ───
echo ""
echo "============================================================"
if [ $TEST_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}>>> 子图 CostModel 测试通过！<<<${NC}"
else
    echo -e "  ${RED}>>> 子图 CostModel 测试失败 (exit: $TEST_EXIT) <<<${NC}"
fi
echo "============================================================"

exit $TEST_EXIT
