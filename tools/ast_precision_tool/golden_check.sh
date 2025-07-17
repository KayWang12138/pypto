#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

start_time0=$(date +%s)
dump_tensor_pt_path="./dump_tensor_pt"
find $1 -name *_golden_data | xargs rm -r
find $1 -name *_torch_data | xargs rm -r
genreport="noreport"
for dir in "$1/"*; do
    key_pass_name=$2
    if [ -z "$key_pass_name" ]; then
        echo ""
    else
        if [[ $dir =~ $key_pass_name ]]; then
                echo ""
        else
            continue
        fi
    fi
    if [ -d "$dir" ]; then
        pt_file_list=()
        pt_file_list=()
        for file in "$dir/"*; do
            if [ -f "$file" ] && [[ "$file" == *.json ]] ; then
                start_time=$(date +%s)
                py_path=$(python3 ast_json_to_torch.py $file $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17})
                end_time=$(date +%s)
                cost_time=$[$end_time - $start_time]
                echo "py_path: $py_path, json-to-torch cost time: $cost_time s"

                file_name="${file%.*}"
                pt_path="${file_name}_torch_data"
                mkdir -p "$pt_path"
                golden_path="${file_name}_golden_data"
                mkdir -p "$golden_path"

                python3 $py_path
                end_time2=$(date +%s)
                cost_time2=$[$end_time2 - $end_time2]
                echo "pt_path: $pt_path, excute torch.py cost time: $cost_time2 s"
                pt_file_list+=($pt_path)

                diff_res=$(python3 data_diff.py $golden_path $pt_path $genreport)
                if [[ $diff_res =~ "ok" ]];then
                    echo "$dir ok"
                else
                    echo "$dir error"
                    echo "$diff_res"
                fi
            fi
        done
        len_pt_file_list=${#pt_file_list[@]}
        if [ $len_pt_file_list -eq 0 ]; then
            continue
        fi
        before_pass_torch_data=${pt_file_list[1]}
        after_pass_torch_data=${pt_file_list[0]}
        start_time3=$(date +%s)
        diff_res=""
        if [[ $dir =~ "SubgraphToFunction" ]];then
            diff_res=$(python3 data_diff.py $after_pass_torch_data $dump_tensor_pt_path $genreport)
        fi
        diff_res=$(python3 data_diff.py $before_pass_torch_data $after_pass_torch_data $genreport)
        end_time3=$(date +%s)
        cost_time3=$[$end_time3 - $start_time3]
        echo "Data diff cost time: $cost_time3 s."

        if [[ $diff_res =~ "ok" ]];then
            echo "$dir ok"
        else
            echo "$dir error"
            echo "$diff_res"
        fi
    else
        if [ -f "$dir" ] && [[ "$dir" =~ "program.json" ]] && [[ "$dir" == *.json ]] ; then
            start_time4=$(date +%s)
            py_path=$(python3 ast_json_to_torch.py $dir $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17})
            end_time4=$(date +%s)
            cost_time4=$[$end_time4 - $start_time4]
            echo "py_path: $py_path, json-to-torch cost time: $cost_time4 s"

            file_name="${dir%.*}"
            pt_path="${file_name}_torch_data"
            mkdir -p "$pt_path"
            golden_path="${file_name}_golden_data"
            mkdir -p "$golden_path"

            python3 $py_path
            end_time5=$(date +%s)
            cost_time5=$[$end_time5 - $end_time4]
            echo "pt_path: $pt_path, excute torch.py cost time: $cost_time5 s"

            start_time6=$(date +%s)
            diff_res=$(python3 data_diff.py $pt_path $dump_tensor_pt_path $genreport)
            end_time6=$(date +%s)
            cost_time6=$[$end_time6 - $start_time6]
            echo "Data diff cost time: $cost_time6 s."
            if [[ $diff_res =~ "ok" ]];then
                echo "$dir ok"
            else
                echo "$dir error"
                echo "$diff_res"
            fi
        fi
    fi
done
end_time0=$(date +%s)
cost_time0=$[$end_time0 - $start_time0]
echo "Total cost time: $cost_time0 s"
