# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

#!/bin/bash
function generate_test_case_data() {
    python3 $OP_PY_TOOLS_PATH/convert_test_case_data_to_json.py $case_data_set $op $start $end $test_case_data_path
}

function run_test_case() {
    test_case_files=$(find $test_case_data_path -maxdepth 1 -type f -name "*.json")
    export ASCEND_PROCESS_LOG_PATH=$plog_path
    index=0
    for file in $test_case_files; do
        file_name=$(basename $file .json)
        IFS='_' read -ra strs <<< "$file_name"
        case_op=${strs[0]}
        case_index=${strs[${#strs[@]}-1]}
        # run test
        test_case="Test$case_op/${case_op}OperationTest.Test$case_op/$index"
        test_case_path=$(dirname $file)/$test_case
        if [ ! -e $test_case_path ]; then
            mkdir -p $test_case_path
        fi
        CMD="python3 build.py -s='$test_case' -d=$device_id"
        LOG_FILE="${case_op}_test_case_$case_index.log"
        echo "Start exec : $CMD"
        eval "$CMD | tee $LOG_FILE"
        # clear golden data
        find build/tests/st/golden/ -name "*.bin" -type f -exec rm -f {} \;
        # generate test report
        generate_test_report $case_index $case_op $LOG_FILE $test_case_result $file
        index=$(($index+1))
    done
    rm -rf $test_case_data_path
    unset ASCEND_PROCESS_LOG_PATH
}

function generate_test_report() {
    # test case index
    case_index=$1
    # operation
    case_op=$2
    # log file
    LOG_FILE=$3
    # test report(excel file)
    test_case_result=$4
    test_case_data=$5
    log_path="$test_case_log_path/$case_op/$case_index"
    if [ ! -e $log_path ]; then
        mkdir -p $log_path
    fi
    python3 $OP_PY_TOOLS_PATH/analyze_test_case_log.py $LOG_FILE $test_case_result
    if [ $? -ne 0 ]; then
        cp -f $test_case_data $log_path
        cp -rf $plog_path/* $log_path
    fi
    mv $LOG_FILE $log_path
    rm -rf $plog_path/*
}

CASE_DATA_HOME=`pwd`
OP_PY_TOOLS_PATH="tests/st/machine/src/ops/operation_test/python_tools"
# define var
start=$1
end=$2
op=$3
device_id=${4:-6}
case_data_set=${5:-"$OP_PY_TOOLS_PATH/../test_case/Add_st_test_cases.csv"}
test_case_result=${6:-"$CASE_DATA_HOME/test_case_result.xlsx"}
echo "cmd is : $0 $1 $2 $3 $4 $5"
echo "cmd args dump:"
echo "    start index : $start"
echo "    end index : $end"
echo "    op : $op"
echo "    using device : $device_id"
echo "    test case input data file : $case_data_set"
echo "    test case report file : $test_case_result"

# clear test report report file
rm -f $test_case_result

test_case_log_path="$(dirname '$test_case_result')/test_case_log"
# clear test case log files
rm -rf $test_case_log_path
# clear test case data
test_case_data_path="build/tests/st/golden/test_cases"
if [ ! -e $test_case_data_path ]; then
    mkdir -p $test_case_data_path
fi
rm -rf $test_case_data_path/*.json
plog_path=$CASE_DATA_HOME/plog
if [ ! -e $plog_path ]; then
    mkdir -p $plog_path
fi
rm -rf $plog_path/*

generate_test_case_data
run_test_case
