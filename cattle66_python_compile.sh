#!/bin/bash

export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/GBowen666/pto-isa

cd /mnt/workspace/gitCode/GBowen666/pypto/

pip3 uninstall pypto -y

python3 -m pip install . --verbose

TEST_DIR="/mnt/workspace/gitCode/GBowen666/pypto/python/tests/ut/litenpu_codegen"

cd $TEST_DIR

rm -r output
rm -r simulator

show_usage() {
    echo "Usage: $0 <operator> [test_case]"
    echo ""
    echo "Operators:"
    echo "  amax      - Run amax tests"
    echo "  amin      - Run amin tests"
    echo "  sum       - Run sum tests"
    echo "  transpose - Run transpose tests"
    echo "  view      - Run view tests"
    echo "  unsqueeze - Run unsqueeze tests"
    echo "  reshape   - Run reshape tests"
    echo "  assemble  - Run assemble tests"
    echo "  all       - Run all tests"
    echo ""
    echo "Examples:"
    echo "  $0 amax                    # Run all amax tests"
    echo "  $0 amax test_amax_fp16_001  # Run single amax test case"
    echo "  $0 all                    # Run all operator tests"
    exit 1
}

if [ $# -lt 1 ]; then
    show_usage
fi

OPERATOR=$1
TEST_CASE=$2

case $OPERATOR in
    amax)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all amax tests..."
            python -m unittest -v test_amax
        else
            echo "Running amax test: $TEST_CASE"
            python -m unittest -v test_amax.$TEST_CASE
        fi
        ;;
    amin)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all amin tests..."
            python -m unittest -v test_amin
        else
            echo "Running amin test: $TEST_CASE"
            python -m unittest -v test_amin.$TEST_CASE
        fi
        ;;
    sum)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all sum tests..."
            python -m unittest -v test_sum
        else
            echo "Running sum test: $TEST_CASE"
            python -m unittest -v test_sum.$TEST_CASE
        fi
        ;;
    transpose)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all transpose tests..."
            python -m unittest -v test_transpose
        else
            echo "Running transpose test: $TEST_CASE"
            python -m unittest -v test_transpose.$TEST_CASE
        fi
        ;;
    view)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all view tests..."
            python -m unittest -v test_view
        else
            echo "Running view test: $TEST_CASE"
            python -m unittest -v test_view.$TEST_CASE
        fi
        ;;
    unsqueeze)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all unsqueeze tests..."
            python -m unittest -v test_unsqueeze
        else
            echo "Running unsqueeze test: $TEST_CASE"
            python -m unittest -v test_unsqueeze.$TEST_CASE
        fi
        ;;
    reshape)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all reshape tests..."
            python -m unittest -v test_reshape
        else
            echo "Running reshape test: $TEST_CASE"
            python -m unittest -v test_reshape.$TEST_CASE
        fi
        ;;
    assemble)
        if [ -z "$TEST_CASE" ]; then
            echo "Running all assemble tests..."
            python -m unittest -v test_assemble
        else
            echo "Running assemble test: $TEST_CASE"
            python -m unittest -v test_assemble.$TEST_CASE
        fi
        ;;
    all)
        echo "Running all operator tests..."
        echo ""
        echo "=========================================="
        echo "Running amax tests..."
        echo "=========================================="
        python -m unittest -v test_amax
        echo ""
        echo "=========================================="
        echo "Running amin tests..."
        echo "=========================================="
        python -m unittest -v test_amin
        echo ""
        echo "=========================================="
        echo "Running sum tests..."
        echo "=========================================="
        python -m unittest -v test_sum
        echo ""
        echo "=========================================="
        echo "Running transpose tests..."
        echo "=========================================="
        python -m unittest -v test_transpose
        echo ""
        echo "=========================================="
        echo "Running view tests..."
        echo "=========================================="
        python -m unittest -v test_view
        echo ""
        echo "=========================================="
        echo "Running unsqueeze tests..."
        echo "=========================================="
        python -m unittest -v test_unsqueeze
        echo ""
        echo "=========================================="
        echo "Running reshape tests..."
        echo "=========================================="
        python -m unittest -v test_reshape
        echo ""
        echo "=========================================="
        echo "Running assemble tests..."
        echo "=========================================="
        python -m unittest -v test_assemble
        ;;
    *)
        echo "Unknown operator: $OPERATOR"
        show_usage
        ;;
esac