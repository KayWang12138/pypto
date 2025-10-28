#!/bin/bash

# Directory containing test cases
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