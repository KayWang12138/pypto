#!/bin/bash
# SumLstm Test Run Script
# Compiles C++ test and runs Python comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== Environment Setup ====================
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
elif [ -n "$ASCEND_HOME" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

echo "=============================================="
echo "SumLstm Custom Operator Test"
echo "=============================================="
echo "ASCEND_INSTALL_PATH: $_ASCEND_INSTALL_PATH"

# Source environment (ignore errors from setenv.bash)
if [ -f "$_ASCEND_INSTALL_PATH/bin/setenv.bash" ]; then
    echo "Sourcing setenv.bash..."
    set +e
    source "$_ASCEND_INSTALL_PATH/bin/setenv.bash" 2>/dev/null
    set -e
    echo "Environment sourced."
fi

export ASCEND_HOME=$_ASCEND_INSTALL_PATH
export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/runtime/lib64:$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH configured."

# ==================== Build ====================
echo ""
echo "[Step 1] Building C++ test executable..."
rm -rf build
mkdir -p build
cd build
echo "Running cmake..."
cmake .. -DCMAKE_BUILD_TYPE=Release
echo "Running make..."
make -j$(nproc) VERBOSE=1
cd ..
echo "Build completed."

# ==================== Run Test ====================
echo ""
echo "[Step 2] Running test..."
python3 test.py
EXIT_CODE=$?

# ==================== Cleanup ====================
echo ""
echo "[Step 3] Cleaning up temporary files..."
rm -f input_*.bin output_*.bin

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST PASSED"
else
    echo "TEST FAILED"
fi
echo "=============================================="

exit $EXIT_CODE
