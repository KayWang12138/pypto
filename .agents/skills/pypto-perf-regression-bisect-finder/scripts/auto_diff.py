#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json
import argparse
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_json(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(125)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def mode_diff(good_file, bad_file, output_config):
    """阶段1：在成百上千的间隙中，揪出劣化幅度最大的那一个作为二分标尺"""
    good_data = load_json(good_file)
    bad_data = load_json(bad_file)

    good_trans = good_data.get("transitions", [])
    bad_trans = bad_data.get("transitions", [])

    if not good_trans or not bad_trans:
        sys.exit(125)

    max_degradation = -1.0
    target_bottleneck = None

    # 遍历比对每一个对应的批次间隙
    for i in range(min(len(good_trans), len(bad_trans))):
        g_gap = good_trans[i]
        b_gap = bad_trans[i]
        
        # 确保类型匹配 (比如都是 AIC -> AIV)
        if g_gap['type'] == b_gap['type']:
            degradation = b_gap['gap_us'] - g_gap['gap_us']
            if degradation > max_degradation:
                max_degradation = degradation
                target_bottleneck = {
                    "transition_index": i,
                    "type": g_gap['type'],
                    "from_task_id": g_gap['from_task_id'],
                    "good_us": g_gap['gap_us'],
                    "bad_us": b_gap['gap_us'],
                    # 设定判定阈值为两者的中点
                    "threshold": (g_gap['gap_us'] + b_gap['gap_us']) / 2.0
                }

    if not target_bottleneck or max_degradation <= 0:
        logging.error("No degradation found between Good and Bad versions.")
        sys.exit(125)

    with open(output_config, 'w') as f:
        json.dump({"target_bottleneck": target_bottleneck}, f, indent=4)
    
    logging.info(f"Target Bottleneck Locked: Index {target_bottleneck['transition_index']} ({target_bottleneck['type']})")
    logging.info(f"Good: {target_bottleneck['good_us']:.2f}us, Bad: {target_bottleneck['bad_us']:.2f}us, Threshold: {target_bottleneck['threshold']:.2f}us")
    sys.exit(0)

def mode_evaluate(current_file, config_file):
    """阶段2：Git Bisect 运行时，检查那个被锁定的嫌疑间隙"""
    current_data = load_json(current_file)
    config = load_json(config_file)

    target = config.get("target_bottleneck")
    if not target or current_data.get("status") != "success":
        sys.exit(125)

    idx = target["transition_index"]
    current_trans = current_data.get("transitions", [])
    
    if len(current_trans) <= idx or current_trans[idx]['type'] != target['type']:
        logging.warning(f"Pipeline signature shifted at Index {idx}. Skipping commit.")
        sys.exit(125)

    curr_gap_us = current_trans[idx]['gap_us']
    threshold = target["threshold"]

    logging.info(f"Evaluating Index {idx} ({target['type']}): Current={curr_gap_us:.2f}us, Threshold={threshold:.2f}us")

    if curr_gap_us <= threshold:
        logging.info("=> Result: GOOD (0)")
        sys.exit(0)
    else:
        logging.info("=> Result: BAD (1)")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    
    parser.add_argument("--good", type=str)
    parser.add_argument("--bad", type=str)
    parser.add_argument("--current", type=str)
    parser.add_argument("--config", type=str, default="bisect_condition.json")

    args = parser.parse_args()

    if args.diff:
        mode_diff(args.good, args.bad, args.config)
    elif args.evaluate:
        mode_evaluate(args.current, args.config)