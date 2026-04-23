#!/bin/bash

export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/GBowen666/pto-isa

cd /mnt/workspace/gitCode/GBowen666/pypto/

pip3 uninstall pypto -y

python3 -m pip install . --verbose

TEST_DIR="/home/l00613242/pto_src_litenpu/pto_20260421/pypto/python/tests/ut/litenpu_codegen"

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

run_test_with_stats() {
    local test_module=$1
    local test_case=$2
    
    if [ -z "$test_case" ]; then
        echo "Running all $test_module tests..."
        output=$(python -m unittest -v $test_module 2>&1)
    else
        echo "Running $test_module test: $test_case"
        output=$(python -m unittest -v $test_module.$test_case 2>&1)
    fi
    
    echo "$output"
    
    total_tests=$(echo "$output" | grep -E "test_.*\.\.\." | wc -l)
    passed_tests=$(echo "$output" | grep -E "test_.*\.\.\. ok" | wc -l)
    failed_tests=$(echo "$output" | grep -E "test_.*\.\.\. FAIL" | wc -l)
    error_tests=$(echo "$output" | grep -E "test_.*\.\.\. ERROR" | wc -l)
    
    echo ""
    echo "=========================================="
    echo "Test Statistics for $test_module"
    echo "=========================================="
    echo "Total tests:    $total_tests"
    echo "Passed:         $passed_tests"
    echo "Failed:         $failed_tests"
    echo "Errors:         $error_tests"
    
    if [ $failed_tests -gt 0 ] || [ $error_tests -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        echo "$output" | grep -E "test_.*\.\.\. (FAIL|ERROR)"
    fi
    echo "=========================================="
    echo ""
}

if [ $# -lt 1 ]; then
    show_usage
fi

OPERATOR=$1
TEST_CASE=$2

case $OPERATOR in
    amax)
        run_test_with_stats "test_amax" "$TEST_CASE"
        ;;
    amin)
        run_test_with_stats "test_amin" "$TEST_CASE"
        ;;
    sum)
        run_test_with_stats "test_sum" "$TEST_CASE"
        ;;
    transpose)
        run_test_with_stats "test_transpose" "$TEST_CASE"
        ;;
    view)
        run_test_with_stats "test_view" "$TEST_CASE"
        ;;
    unsqueeze)
        run_test_with_stats "test_unsqueeze" "$TEST_CASE"
        ;;
    reshape)
        run_test_with_stats "test_reshape" "$TEST_CASE"
        ;;
    assemble)
        run_test_with_stats "test_assemble" "$TEST_CASE"
        ;;
    all)
        echo "Running all operator tests..."
        echo ""
        
        run_test_with_stats "test_amax"
        run_test_with_stats "test_amin"
        run_test_with_stats "test_sum"
        run_test_with_stats "test_transpose"
        run_test_with_stats "test_view"
        run_test_with_stats "test_unsqueeze"
        run_test_with_stats "test_reshape"
        run_test_with_stats "test_assemble"
        
        echo "=========================================="
        echo "ALL OPERATORS TEST SUMMARY"
        echo "=========================================="
        ;;
    *)
        echo "Unknown operator: $OPERATOR"
        show_usage
        ;;
esac