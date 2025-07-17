#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
import sys
import torch
import csv
from datetime import datetime, timezone
error_file_dict = dict()
error_file_dict_topoorder = dict()


def print_red(text):
    print(f"\033[91m{text}\033[0m")


def load_data(file_path):
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"load failed from {file_path} , Error: {e}")


def parse_data_dir(dir, is_golden=False, is_program=False):
    meta_data = {}
    for file in os.listdir(dir):
        file_path = f'{dir}/{file}'
        meta_data[file] = load_data(file_path)
    return meta_data


def parse_data_file(dir, is_golden=False):
    file = os.path.basename(dir)
    meta_data = {}
    if is_golden:
        meta_data[file[file.find('-') + 1:]] = load_data(dir)
    else:
        meta_data[file] = load_data(dir)
    return meta_data


def compare_results(aa, bb, thresh_abs=1e-8, thresh_rel=1e-3, thresh_err_percent=None):
    error_str = ""
    a = aa.flatten().to(torch.float64)
    b = bb.flatten().to(torch.float64)
    d_sub = (a - b)
    count = len(b)
    non_finite_count = d_sub.isfinite().logical_not().sum().item()
    if non_finite_count > 0:
        assert False, f'Error: Unsupported non finite float diff, count={non_finite_count}'
    if thresh_err_percent is None:
        thresh_err_percent = (thresh_abs + thresh_rel) / 2
    d_sub_abs = d_sub.abs()
    d_abs_add = a.abs() + b.abs()
    d_rel = d_sub_abs/d_abs_add
    d_zero_mask = d_abs_add == 0.0
    d_inval_mask = d_abs_add.isfinite()
    d_filtered_mask = d_zero_mask.logical_or(d_inval_mask)
    d_rel = torch.where(d_filtered_mask, 0.0, d_rel)

    err_abs_mask = torch.where(d_sub_abs > thresh_abs, 1, 0)
    err_rel_mask = torch.where(d_rel > thresh_rel, 1, 0)
    err_all_mask = err_abs_mask.add(err_rel_mask)
    err_all_count = err_all_mask.sum().item()
    # 当前使用的是整体的值域的count，如果更进一步可以考虑使用把0和非法值去掉之后的count
    if err_all_count > thresh_err_percent * count:
        error_str = f"err_count={err_all_count} thresh_count={thresh_err_percent * count} "
        error_str += f"inf={d_sub_abs.isinf().sum().item()} nan={d_sub_abs.isnan().sum().item()} "
        error_str += f"max={d_sub_abs.max().item()} min={d_sub_abs.min().item()} avg={d_sub_abs.mean().item()}"
    return error_str


def check_dir(data_golden, data_real, is_program=False):
    ret = True
    for key, data in data_real.items():
        tmp_key = key[key.find('-')+1:]
        timeid = eval(key[6:key.find('-')])
        match_key = next((golden_key for golden_key in data_golden.keys() if tmp_key in golden_key), None)
        if match_key is None:
            continue
        if (data_golden[match_key].shape != data.shape):
            error_file_dict[timeid] = f'Diff res of file {key} shape error, '
            error_file_dict[timeid] += f'{data_golden[match_key].shape} vs {data.shape}'
            ret = False
            continue
        error_info = compare_results(data_golden[match_key], data)
        if len(error_info) > 0:
            error_file_dict[timeid] = f'Diff res of file {key} error, error_msg: {error_info}'
            if is_program:
                topo_timeid = eval(match_key[6:match_key.find('-')])
                error_file_dict_topoorder[topo_timeid] = f'Diff res of file {match_key} error, error_msg: {error_info}'
            ret = False
    return ret


def delete_both_zero(a, b):
    a_zero = torch.eq(a, 0.0)
    b_zero = torch.eq(b, 0.0)
    both_zero = torch.logical_and(a_zero, b_zero)
    a_filtered = torch.masked_select(a, ~both_zero)
    b_filtered = torch.masked_select(b, ~both_zero)
    return a_filtered, b_filtered


def diff_abs_avg(a, b):
    assert (a.size() == b.size())
    aa, bb = delete_both_zero(a, b)
    avg = (aa.abs() - bb.abs()).mean()
    return avg.item()


def diff_rel_avg(a, b):
    assert (a.size() == b.size())
    aa, bb = delete_both_zero(a, b)
    sub_abs = aa.abs() - bb.abs()
    sum_abs = aa.abs() + bb.abs()
    avg = (sub_abs / sum_abs).mean()
    return avg.item()


def gen_report(data_golden, data_real, path_golden_dir):
    title = ['id', 'time', 'OP', 'comp factor', 'A info', 'A shape', 'A dtype', 'B info', 'B shape', 'B dtype',
             'large val > diff abs avg', 'large val > diff rel avg', 'small val > diff abs avg',
             'small val > diff rel avg', 'A num inf', 'A num nan', 'A num zero', 'A max', 'A min', 'A pos avg',
             'A neg avg', 'B num inf', 'B num nan', 'B num zero', 'B max', 'B min', 'B pos avg', 'B neg avg', 'RESULT',
             'DESCRIPTION']
    report = []
    report.append(title)
    id = 0
    assert (len(data_golden) == len(data_real))
    threshold = {}
    threshold['float16'] = 0.0001
    for key, data in data_real.items():
        key_golden = key[key.find('-') + 1:]
        if data_golden.get(key_golden) is None:
            continue
        a = data_golden[key_golden]
        b = data
        report_line = []
        id = id + 1
        time = key[6:key.find('-')]
        op = ''
        com_factor = '1'
        a_info = key_golden
        a_shape = f'{a.shape}'[11:-1]
        a_dtype = f'{a.dtype}'[6:]
        b_info = key
        b_shape = f'{b.shape}'[11:-1]
        b_dtype = f'{b.dtype}'[6:]
        cur_threshold = threshold[a_dtype] if a_dtype in threshold else 0.0001
        a_large_indices = torch.where(a > cur_threshold)
        a_small_indices = torch.where(a < cur_threshold)
        a_large = a[a_large_indices].flatten()
        b_large = b[a_large_indices].flatten()
        a_small = a[a_small_indices].flatten()
        b_small = b[a_small_indices].flatten()
        large_val_gt_diff_abs_avg = diff_abs_avg(a_large, b_large)
        large_val_gt_diff_ref_avg = diff_rel_avg(a_large, b_large)
        small_val_gt_diff_abs_avg = diff_abs_avg(a_small, b_small)
        small_val_gt_diff_ref_avg = diff_rel_avg(a_small, b_small)
        a_num_inf = f'{a.isinf().sum().item()}'
        a_num_nan = f'{a.isnan().sum().item()}'
        a_zero_mask = a == 0
        a_num_zero = f'{torch.sum(a_zero_mask).item()}'
        a_max = f'{a.max().item()}'
        a_min = f'{a.min().item()}'
        a_pos_numbers = a[a > 0]
        a_pos_average = a_pos_numbers.to(torch.float64).mean().item()
        a_neg_numbers = a[a < 0]
        a_neg_average = a_neg_numbers.to(torch.float64).mean().item()
        b_num_inf = f'{b.isinf().sum().item()}'
        b_num_nan = f'{b.isnan().sum().item()}'
        b_zero_mask = b == 0
        b_num_zero = f'{torch.sum(b_zero_mask).item()}'
        b_max = f'{b.max().item()}'
        b_min = f'{b.min().item()}'
        b_pos_numbers = b[b > 0]
        b_pos_average = b_pos_numbers.to(torch.float64).mean().item()
        b_neg_numbers = b[b < 0]
        b_neg_average = b_neg_numbers.to(torch.float64).mean().item()
        err_info = compare_results(a, b)
        res = 'ERROR' if len(err_info) > 0 else 'OK'
        description = ''
        report_line.extend([id, time, op, com_factor, a_info, a_shape, a_dtype, b_info, b_shape, b_dtype,
                            large_val_gt_diff_abs_avg, large_val_gt_diff_ref_avg, small_val_gt_diff_abs_avg,
                            small_val_gt_diff_ref_avg, a_num_inf, a_num_nan, a_num_zero, a_max, a_min, a_pos_average,
                            a_neg_average, b_num_inf, b_num_nan, b_num_zero, b_max, b_min, b_pos_average, b_neg_average,
                            res, description])
        report.append(report_line)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(parent_dir, 'report')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    current_datetime = datetime.now(tz=timezone.utc)
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    if os.path.isdir(path_golden_dir):
        pass_name = os.path.basename(path_golden_dir)
    else:
        pass_name = os.path.basename(os.path.dirname(path_golden_dir))
    report_name = 'report_' + pass_name + '_' + formatted_datetime + '.csv'
    report_path = os.path.join(report_dir, report_name)
    with open(report_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(report)
        print(f'REPORT {report_path} generated.')


if __name__ == '__main__':
    golden_dir = sys.argv[1]
    real_dir = sys.argv[2]
    is_genreport = False
    is_program = (golden_dir.find('program') != -1)
    if sys.argv[3] in 'genreport':
        is_genreport = True
    if os.path.isdir(golden_dir) and os.path.isdir(real_dir):
        if (len(os.listdir(golden_dir)) == 0) or (len(os.listdir(real_dir)) == 0):
            print(f'Diff res of file {golden_dir} or {real_dir} is empty')
            sys.exit(0)
        meta_data_golden = parse_data_dir(golden_dir, True, is_program)
        meta_data_real = parse_data_dir(real_dir, False, is_program)
        if is_genreport:
            gen_report(meta_data_golden, meta_data_real, golden_dir)
        if check_dir(meta_data_golden, meta_data_real, is_program):
            print(f'Diff res of file {golden_dir} vs {real_dir} ok')
        else:
            sorted_error_file_dict = {key: error_file_dict[key] for key in sorted(error_file_dict, reverse=True)}
            sorted_keys = sorted(error_file_dict_topoorder, reverse=True)
            sorted_error_file_dict_topoorder = {key: error_file_dict_topoorder[key] for key in sorted_keys}
            if is_program:
                print(f" ")
                print(f" ")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
                print(f"+++++++++++++++++++++++++++ Result in json Topology Order +++++++++++++++++++++++++++++++++ ")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
                print(f" ")
                for value in sorted_error_file_dict_topoorder.values():
                    print(f"Topology Order: {value}")
                print(f" ")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
                print(f"++++++++++++++++++++++++++ Result in Device Execution Time Sequence +++++++++++++++++++++++ ")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
                print(f" ")

            for value in sorted_error_file_dict.values():
                if is_program:
                    print(f"Device Sequence: {value}")
                else:
                    print(value)

    if os.path.isfile(golden_dir) and os.path.isfile(real_dir):
        meta_data_golden = parse_data_file(golden_dir, True)
        meta_data_real = parse_data_file(real_dir, False)
        if is_genreport:
            gen_report(meta_data_golden, meta_data_real, golden_dir)
        error_msg = compare_results(load_data(golden_dir), load_data(real_dir))
        if len(error_msg) > 0:
            print_red(f'Diff res of file {golden_dir} vs {real_dir} error, error_msg: {error_msg}')
        else:
            print_red(f'Diff res of file {golden_dir} vs {real_dir} ok')
