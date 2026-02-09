#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor compare utilility."""
import os
import torch
import logging
from tabulate import tabulate


class TensorComparator:
    def __init__(self):
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", encoding="utf-8")
            ]
        )
    
    def print_tensor_stats_info(self, stats, prefix='tensor'):
        (dtype_str, shape, cnt_all, cnt_z, cnt_infnan, st_max, st_min, st_avg, st_aavg) = stats
        sep = ', '
        dtype_str = dtype_str
        shape = '_'.join([str(i) for i in shape]) if shape else 's'
        logging.info(f'{"data "+prefix:>8s} : {dtype_str:>10s}{sep}{shape}{sep}{cnt_all}{sep}{cnt_z}{sep}{cnt_infnan}\
        {sep}{st_max:g}{sep}{st_min:g}{sep}{st_avg:g}{sep}{st_aavg:g}\t(dtype/shape/all/zero/infnan/max/min/avg/aavg)')
        pass
    
    def calc_tensor_stats(self, a, calc_dtype=torch.float64):
        if calc_dtype not in [torch.float64, torch.float32]:
            raise ValueError(f'不支持的计算数据类型: {calc_dtype}')
        t = a.flatten().to(calc_dtype)

        dtype_str = str(a.dtype).lstrip('torch.')
        shape = list(a.shape)
        cnt_all = t.numel()
        cnt_z = (t == 0).sum().item()
        t_fin_mask = t.isfinite()
        t_fin = t.masked_select(t_fin_mask)
        st_max, st_min, st_avg, st_aavg = 0.0, 0.0, 0.0, 0.0
        if t_fin.numel() != 0:
            st_max = t_fin.max().item()
            st_min = t_fin.min().item()
            st_avg = (t_fin.sum()/t.numel()).item()
            st_aavg = (t_fin.abs().sum()/t.numel()).item()
        cnt_infnan = cnt_all - t_fin.numel()

        stats_t = (dtype_str, shape, cnt_all, cnt_z, cnt_infnan, st_max, st_min, st_avg, st_aavg)
        return stats_t
    
    def print_info(self, d_detail, d_str, topk):
        num = len(d_detail[0])
        if num > 0:
            num_picked = min(topk, num) if topk and topk >= 0 else num
            
            # 准备数据
            table_data = []
            for i in range(num_picked):
                info_off = d_detail[0][i].item()
                off_raw = d_detail[1][i]
                off_raw = off_raw.tolist() if off_raw.dim() > 0 else [off_raw.item()]
                info_off_raw = '_'.join([str(s) for s in off_raw])
                
                row = [
                    i + 1,
                    info_off,
                    f"{d_detail[2][i].item():.6g}",
                    f"{d_detail[3][i].item():.6g}",
                    f"{d_detail[4][i].item():.6g}",
                    f"{d_detail[5][i].item():.6g}",
                    f"{d_detail[6][i].item():.6g}",
                    info_off_raw
                ]
                table_data.append(row)
            
            # 表头
            headers = ["#", "off", "a", "b", "ad", "rd", "ad_tol", "off_raw"]
            
            # 打印
            logging.info(f"-{d_str}({num})")
            logging.info(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            if num_picked < num and num_picked > 0:
                logging.info(f' ...... {topk+1}~{num} ({num-num_picked}) ......')
    
    def print_isclose_info(self, result_is_close, result_reason_str, result_info, topk=16):
        (d_cnt, d_conf, _d_extra, d_detail_warn, d_detail_fail, d_detail_infnan) = result_info
        (cnt_all, cnt_picked, cnt_out_bothzero, cnt_out_pass, cnt_out_warn, cnt_fail, cnt_infnan) = d_cnt
        (rtol, atol, _fail_factor, tol_cnt) = d_conf
        (cnt_warn_ww, cnt_warn_w, cnt_out_warn, cnt_warn_s, cnt_warn_ss) = _d_extra
        sep = ', '
        logging.info(f'cnt : {cnt_all}{sep}{cnt_picked}{sep}{cnt_out_bothzero}{sep}{cnt_out_pass}{sep}{cnt_out_warn}{sep}{cnt_fail}{sep}{cnt_infnan}\t(all/picked/zero/pass/warn/fail/infnan)')
        logging.info(f'conf : {rtol:g}{sep}{atol:g}{sep}{_fail_factor}{sep}{tol_cnt}\t(rtol/atol/_fail_factor/tol_cnt)')

        if sum(_d_extra) != _d_extra[2]:
            logging.info(f'_extra : {cnt_warn_ww}{sep}{cnt_warn_w}{sep}{cnt_out_warn}{sep}{cnt_warn_s}{sep}{cnt_warn_ss}\t(ww/w/warn/s/ss)')
        logging.info(f'is_close : {result_is_close}\t({result_reason_str})')

        self.print_info(d_detail_infnan, 'bad', topk)
        self.print_info(d_detail_fail, 'fail', topk)
        self.print_info(d_detail_warn, 'warn', topk)
    
    # fp32: rtol=1.0e-5, atol=1.0e-5
    # fp16: rtol=1.0e-3, atol=1.0e-3
    # bf16: rtol=1.0e-2, atol=1.0e-2
    # fp8 : rtol=5.0e-2, atol=5.0e-2
    def check_isclose(self, A, B, rtol=1.0e-2, atol=1.0e-2, calc_dtype=torch.float64, shape=None,
            is_ignore_bothzero=True, is_detail=False, _fail_factor=128, _is_extra=False):
        assert calc_dtype in [torch.float64, torch.float32], f'not support calculating dtype: {calc_dtype}'
        aa = A.flatten()
        bb = B.flatten()
        a = aa.to(calc_dtype)
        b = bb.to(calc_dtype)

        a_abs = a.abs()
        b_abs = b.abs()
        ab_sub = (a - b)
        ab_sub_abs = ab_sub.abs()
        ab_abs_add = (a_abs + b_abs)
        tol_warn = ab_abs_add * rtol / 2 + atol
        tol_fail = tol_warn * _fail_factor
        mask_bothzero = (ab_abs_add == 0)
        mask_warn = torch.gt(ab_sub_abs, tol_warn)
        mask_infnan = ab_sub_abs.isfinite().logical_not()
        mask_fail = torch.gt(ab_sub_abs, tol_fail)
        cnt_all = mask_warn.numel()
        cnt_out_warn = mask_warn.sum().item()
        cnt_out_bothzero = mask_bothzero.sum().item()
        cnt_out_pass = cnt_all - cnt_out_warn - cnt_out_bothzero
        assert cnt_out_pass >= 0
        if is_ignore_bothzero:
            cnt_picked = cnt_all - cnt_out_bothzero    #cnt_picked = cnt_out_warn + cnt_out_pass
            assert (cnt_all - cnt_out_bothzero) == (cnt_out_warn + cnt_out_pass)
        else:
            cnt_picked = cnt_all
        cnt_fail = mask_fail.sum().item()
        cnt_infnan = mask_infnan.sum().item()

        tol_cnt = tol_cnt_raw = int(cnt_picked * min(rtol, atol))
        if tol_cnt_raw == 0:
            tol_cnt = min(16, int(cnt_picked**0.5) // 2)    # tol_cnt is zero, adjust to small value                        

        if not is_detail:
            const_empty_arg = torch.tensor([], dtype=torch.int64)
            const_empty_value = torch.tensor([], dtype=calc_dtype)
            data_warn_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
            data_fail_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
            data_infnan_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
        else:
            ab_ad = ab_sub
            ab_rd = ab_sub_abs * 2 / ab_abs_add
            arg_warn_raw = arg_warn = torch.argwhere(mask_warn).flatten()
            arg_fail_raw = arg_fail = torch.argwhere(mask_fail).flatten()
            arg_infnan_raw = arg_infnan = torch.argwhere(mask_infnan).flatten()
            if (A.dim() > 1) or (B.dim() > 1):
                if shape:
                    to_shape = shape
                else:
                    to_shape = A.shape if A.dim() >= B.dim() else B.shape
                arg_warn_raw = torch.argwhere(mask_warn.reshape(to_shape))
                arg_fail_raw = torch.argwhere(mask_fail.reshape(to_shape))
                arg_infnan_raw = torch.argwhere(mask_infnan.reshape(to_shape))
            data_warn_info_list = (arg_warn, arg_warn_raw, aa.take(arg_warn), bb.take(arg_warn), ab_ad.take(arg_warn), ab_rd.take(arg_warn), tol_warn.take(arg_warn))
            data_fail_info_list = (arg_fail, arg_fail_raw, aa.take(arg_fail), bb.take(arg_fail), ab_ad.take(arg_fail), ab_rd.take(arg_fail), tol_fail.take(arg_warn))
            data_infnan_info_list = (arg_infnan, arg_infnan_raw, aa.take(arg_infnan), bb.take(arg_infnan), ab_ad.take(arg_infnan), ab_rd.take(arg_infnan), arg_infnan)

        # weak/strong warning
        if not _is_extra:
            cnt_warn_ww = cnt_warn_w = cnt_warn_s = cnt_warn_ss = 0
        else:
            tol_warn_ww = tol_warn / 4
            tol_warn_w = tol_warn / 2
            tol_warn_s = tol_warn * 2
            tol_warn_ss = tol_warn * 4
            mask_warn_ww = torch.gt(ab_sub_abs, tol_warn_ww)
            mask_warn_w = torch.gt(ab_sub_abs, tol_warn_w)
            mask_warn_s = torch.gt(ab_sub_abs, tol_warn_s)
            mask_warn_ss = torch.gt(ab_sub_abs, tol_warn_ss)
            cnt_warn_ww = mask_warn_ww.sum().item()
            cnt_warn_w = mask_warn_w.sum().item()
            cnt_warn_s = mask_warn_s.sum().item()
            cnt_warn_ss = mask_warn_ss.sum().item()

        diff_cnt = (cnt_all, cnt_picked, cnt_out_bothzero, cnt_out_pass, cnt_out_warn, cnt_fail, cnt_infnan)
        diff_conf = (rtol, atol, _fail_factor, tol_cnt)
        _diff_extra = (cnt_warn_ww, cnt_warn_w, cnt_out_warn, cnt_warn_s, cnt_warn_ss)
        diff_detail_warn = data_warn_info_list
        diff_detail_fail = data_fail_info_list
        diff_detail_infnan = data_infnan_info_list
        result_is_close = (cnt_out_warn <= tol_cnt) and (cnt_fail <= 0) and (cnt_infnan <= 0)
        result_reason_str = []
        if (cnt_out_warn > tol_cnt):
            result_reason_str.append(f'cnt_warn(={cnt_out_warn}) > tol_cnt(={tol_cnt})')
        if (cnt_fail > 0):
            result_reason_str.append(f'cnt_fail(={cnt_fail}) > 0)')
        if (cnt_infnan > 0):
            result_reason_str.append(f'cnt_infnan(={cnt_infnan}) > 0)')
        result_reason_str = ','.join(result_reason_str)

        result_info = (diff_cnt, diff_conf, _diff_extra, diff_detail_warn, diff_detail_fail, diff_detail_infnan)

        return result_is_close, result_reason_str, result_info


if __name__ == '__main__':
    import sys
    from tensor_load import from_file, get_valid_file_type_list

    comparator = TensorComparator()

    def usage_exit():
        logging.info(f'Usage: {sys.argv[0]} <file_a> <file_b> <DTYPE_a>-<DTYPE_b>', file=sys.stderr)
        logging.info(f'  <DTYPE>:   {"|".join(get_valid_file_type_list())}', file=sys.stderr)
        logging.info(f'', file=sys.stderr)
        exit()

    if len(sys.argv) < 4:
        usage_exit()
    elif len(sys.argv) > 4:
        usage_exit()

    fa = sys.argv[1]
    fb = sys.argv[2]

    calc_dtype = torch.float32
    a_dtype_str = sys.argv[3].split('-')[0]
    b_dtype_str = sys.argv[3].split('-')[1]

    A = from_file(fa, a_dtype_str)
    B = from_file(fb, b_dtype_str)

    stats_a = comparator.calc_tensor_stats(A, calc_dtype=calc_dtype)
    stats_b = comparator.calc_tensor_stats(B, calc_dtype=calc_dtype)
    comparator.print_tensor_stats_info(stats_a, prefix='a')
    comparator.print_tensor_stats_info(stats_b, prefix='b')

    result_is_close, result_reason_str, result_info = comparator.check_isclose(
        A, B, rtol=1.0e-3, atol=1.0e-3, 
        calc_dtype=torch.float32, is_ignore_bothzero=True, is_detail=True
    )
    comparator.print_isclose_info(result_is_close, result_reason_str, result_info, topk=None)