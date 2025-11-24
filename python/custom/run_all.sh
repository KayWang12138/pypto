#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# Directory containing test cases
set -e
TEST_DIR="testcases"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory '$TEST_DIR' does not exist."
    exit 1
fi

rm -rf generatedcpp
mkdir generatedcpp

# Loop through all .py files in the directory, and any subdirectory
for test_file in $(find "$TEST_DIR" -maxdepth 2 -name "*.py"); do
    # Check if there are any .py files
    if [ ! -f "$test_file" ]; then
        echo "No .py test files found in '$TEST_DIR'"
        break
    fi

    echo "Running test: $test_file"
    python "$test_file"
    echo "Done running $test_file ........................................................................................................"


    # Check if the test succeeded
    if [ $? -ne 0 ]; then
        echo "Test failed: $test_file"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
done

echo "All tests completed"