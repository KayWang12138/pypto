#!/usr/bin/env bash
# Note: We don't use 'set -e' here because we want to continue running tests even if one fails

BUILD_DIR=build
MODE=${1:-run}   # 默认 run，传 test 则跑 gtest
PYTHON_TEST_PATTERN=${2:-}  # 可选的正则表达式，用于过滤 Python 测试文件

# Arrays to track test results
declare -a TEST_FILES=()
declare -a TEST_RESULTS=()  # "PASS" or "FAIL"
declare -a SKIPPED_FILES=()  # Files that were skipped

echo "========================================"
echo " Build mode: $MODE"
echo " Build dir : $BUILD_DIR"
if [ -n "$PYTHON_TEST_PATTERN" ]; then
    echo " Python test pattern: $PYTHON_TEST_PATTERN"
fi
echo "========================================"

if [[ "$MODE" == "test" ]]; then
    echo "[INFO] Configure with GoogleTest enabled"

    cmake -S . -B ${BUILD_DIR} \
        -DPTO_ENABLE_TESTS=ON \
        -DCMAKE_BUILD_TYPE=Debug

    cmake --build ${BUILD_DIR} -j

    echo "[INFO] Running ctest"
    ctest --test-dir ${BUILD_DIR} --verbose

else
    echo "[INFO] Configure without GoogleTest (run main)"

    cmake -S . -B ${BUILD_DIR} \
        -DPTO_ENABLE_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Debug

    cmake --build ${BUILD_DIR} -j

    echo "[INFO] Running Python tests under python/tests"
    PYTHON_TEST_DIR="python/tests"
    # Set PYTHONPATH so tests can directly import pypto
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)/python"
    if [ -d "${PYTHON_TEST_DIR}" ]; then
        # Run Python test files matching test_*.py recursively in subdirectories
        # Use find to recursively search for test files
        while IFS= read -r test_file; do
            if [ -f "${test_file}" ]; then
                # Extract just the filename without path
                test_filename=$(basename "${test_file}")
                
                # If pattern is provided, check if filename matches the regex
                if [ -n "$PYTHON_TEST_PATTERN" ]; then
                    if [[ ! "$test_filename" =~ $PYTHON_TEST_PATTERN ]]; then
                        SKIPPED_FILES+=("${test_filename}")
                        continue
                    fi
                fi
                
                echo "[INFO] Running Python test: ${test_file}"
                TEST_FILES+=("${test_filename}")
                if python3 "${test_file}"; then
                    TEST_RESULTS+=("PASS")
                    echo "[PASS] ${test_filename}"
                else
                    TEST_RESULTS+=("FAIL")
                    echo "[FAIL] ${test_filename}"
                fi
            fi
        done < <(find "${PYTHON_TEST_DIR}" -type f -name "test_*.py" | sort)
        
        # Print summary
        echo ""
        echo "========================================"
        echo " Python Test Summary"
        echo "========================================"
        total_tests=$((${#TEST_FILES[@]} + ${#SKIPPED_FILES[@]}))
        if [ ${total_tests} -eq 0 ]; then
            echo "No tests found."
        else
            passed=0
            failed=0
            skipped=0
            
            # Print passed tests
            for i in "${!TEST_FILES[@]}"; do
                status="${TEST_RESULTS[$i]}"
                file="${TEST_FILES[$i]}"
                if [ "$status" == "PASS" ]; then
                    echo "  ✓ PASS: ${file}"
                    ((passed++))
                else
                    echo "  ✗ FAIL: ${file}"
                    ((failed++))
                fi
            done
            
            # Print skipped tests
            for skipped_file in "${SKIPPED_FILES[@]}"; do
                echo "  ⊘ SKIP: ${skipped_file}"
                ((skipped++))
            done
            
            echo ""
            echo "Total: ${total_tests} | Passed: ${passed} | Failed: ${failed} | Skipped: ${skipped}"
            if [ ${failed} -gt 0 ]; then
                echo "========================================"
                exit 1
            fi
        fi
        echo "========================================"
    else
        echo "[WARN] Python test directory '${PYTHON_TEST_DIR}' not found, skipping Python tests"
    fi
fi
