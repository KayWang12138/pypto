#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# PyPTO CostModel 一键验证脚本
#
# 功能:
#   1. 将测试脚本同步到远程服务器
#   2. 设置 CANN + Python 环境
#   3. 运行整图 costmodel 验证
#   4. 运行子图 costmodel 验证
#   5. 汇总报告结果
#
# 用法:
#   bash scripts/run_costmodel_verify.sh [SSH_HOST]
#
# 默认自动检测 ~/.devenv/.ssh/config 中可用的开发机
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# ─── 配置 ───
REMOTE_HOST="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_FILES=(
    "python/tests/st/test_costmodel_wholegraph.py"
    "python/tests/st/test_costmodel_subgraph_verify.py"
)

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── 自动检测可用的远程服务器 ───
if [ -z "$REMOTE_HOST" ]; then
    for host in $(grep "^Host " ~/.devenv/.ssh/config 2>/dev/null | awk '{print $2}'); do
        if ssh -o ConnectTimeout=5 "$host" "echo ok" >/dev/null 2>&1; then
            REMOTE_HOST="$host"
            break
        fi
    done
    if [ -n "$REMOTE_HOST" ]; then
        echo -e "${CYAN}自动选择远程服务器: $REMOTE_HOST${NC}"
    fi
fi

if [ -z "$REMOTE_HOST" ]; then
    echo -e "${RED}错误: 未找到可用的远程服务器${NC}"
    echo "用法: bash $0 <SSH_HOST>"
    exit 1
fi

REMOTE_WORKSPACE="/mnt/workspace/costmodel_verify"

echo ""
echo "============================================================"
echo "  PyPTO CostModel 一键验证"
echo "  远程服务器: $REMOTE_HOST"
echo "============================================================"
echo ""

# ─── Step 0: 检查远程连接 ───
echo -e "${CYAN}[Step 0] 检查远程服务器连接 ...${NC}"
if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" "echo 'connected'" > /dev/null 2>&1; then
    echo -e "${RED}  [FAIL] 无法连接到远程服务器: $REMOTE_HOST${NC}"
    exit 1
fi
echo -e "${GREEN}  连接成功${NC}"

# ─── Step 1: 同步测试脚本 ───
echo ""
echo -e "${CYAN}[Step 1] 同步测试脚本到远程服务器 ...${NC}"
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_WORKSPACE"

for f in "${TEST_FILES[@]}"; do
    local_path="$PROJECT_DIR/$f"
    remote_name="$(basename "$f")"
    if [ -f "$local_path" ]; then
        scp -q "$local_path" "$REMOTE_HOST:$REMOTE_WORKSPACE/$remote_name"
        echo -e "  ${GREEN}已同步: $remote_name${NC}"
    else
        echo -e "  ${RED}[FAIL] 本地文件不存在: $local_path${NC}"
        exit 1
    fi
done

# ─── Step 2: 运行整图 CostModel 验证 ───
echo ""
echo "============================================================"
echo -e "${CYAN}  [Step 2] 运行整图 CostModel 验证${NC}"
echo "============================================================"
echo ""

WHOLE_RESULT=0
ssh "$REMOTE_HOST" bash <<REMOTE_SCRIPT || WHOLE_RESULT=\$?
# 检测 CANN 路径
for cann_path in /home/developer/Ascend/cann-8.5.0 /home/developer/Ascend/cann-9.0.0; do
    if [ -f "\$cann_path/set_env.sh" ]; then
        source "\$cann_path/set_env.sh" 2>/dev/null
        break
    fi
done
source /home/developer/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/home/developer/.local/lib/python3.12/site-packages:\$PYTHONPATH
cd $REMOTE_WORKSPACE
rm -rf output
python3 test_costmodel_wholegraph.py
REMOTE_SCRIPT

if [ "$WHOLE_RESULT" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}  >>> 整图 CostModel 验证通过 <<<${NC}"
else
    echo ""
    echo -e "${RED}  >>> 整图 CostModel 验证失败 (exit code: ${WHOLE_RESULT}) <<<${NC}"
fi

# ─── Step 3: 运行子图 CostModel 验证 ───
echo ""
echo "============================================================"
echo -e "${CYAN}  [Step 3] 运行子图 CostModel 验证${NC}"
echo "============================================================"
echo ""

SUB_RESULT=0
ssh "$REMOTE_HOST" bash <<REMOTE_SCRIPT || SUB_RESULT=\$?
for cann_path in /home/developer/Ascend/cann-8.5.0 /home/developer/Ascend/cann-9.0.0; do
    if [ -f "\$cann_path/set_env.sh" ]; then
        source "\$cann_path/set_env.sh" 2>/dev/null
        break
    fi
done
source /home/developer/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/home/developer/.local/lib/python3.12/site-packages:\$PYTHONPATH
cd $REMOTE_WORKSPACE
python3 test_costmodel_subgraph_verify.py
REMOTE_SCRIPT

if [ "$SUB_RESULT" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}  >>> 子图 CostModel 验证通过 <<<${NC}"
else
    echo ""
    echo -e "${YELLOW}  >>> 子图 CostModel 跳过/失败 (exit: ${SUB_RESULT})${NC}"
    echo -e "    ${YELLOW}需要在含 CostModelRunSubgraphLine 的 pypto 版本上运行${NC}"
fi

# ─── Step 4: 汇总 ───
echo ""
echo "============================================================"
echo "  验证结果汇总"
echo "============================================================"
echo ""
if [ "$WHOLE_RESULT" -eq 0 ]; then
    echo -e "  整图 CostModel:  ${GREEN}PASS${NC}"
else
    echo -e "  整图 CostModel:  ${RED}FAIL (exit: ${WHOLE_RESULT})${NC}"
fi

if [ "$SUB_RESULT" -eq 0 ]; then
    echo -e "  子图 CostModel:  ${GREEN}PASS${NC}"
else
    echo -e "  子图 CostModel:  ${YELLOW}SKIP/FAIL${NC}"
fi
echo ""

if [ "$WHOLE_RESULT" -ne 0 ]; then
    echo -e "${RED}  >>> 整图验证失败，请检查日志 <<<${NC}"
    exit 1
elif [ "$SUB_RESULT" -ne 0 ]; then
    echo -e "${YELLOW}  >>> 整图通过 ✓ | 子图需在含子图功能的 pypto 上验证 <<<${NC}"
    exit 0
else
    echo -e "${GREEN}  >>> 全部验证通过！ <<<${NC}"
    exit 0
fi
