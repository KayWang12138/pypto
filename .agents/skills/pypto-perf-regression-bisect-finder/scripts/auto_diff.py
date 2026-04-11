#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import argparse
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_json(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def mode_diff(good_file, bad_file, output_config):
    """阶段1：在各个间隙中，找出劣化幅度最大的那一个作为二分标尺"""
    good_data = load_json(good_file)
    bad_data = load_json(bad_file)

    if good_data is None or bad_data is None:
        return 125

    good_trans = good_data.get("transitions", [])
    bad_trans = bad_data.get("transitions", [])

    if not good_trans or not bad_trans:
        return 125

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
                    # 设定判定阈值为偏向good的75%位置（更接近bad，bad更难产生）
                    "threshold": g_gap['gap_us'] + (b_gap['gap_us'] - g_gap['gap_us']) * 0.75
                }

    if not target_bottleneck or max_degradation <= 0:
        logging.error("No degradation found between Good and Bad versions.")
        return 125

    with open(output_config, 'w') as f:
        json.dump({"target_bottleneck": target_bottleneck}, f, indent=4)

    idx = target_bottleneck['transition_index']
    b_type = target_bottleneck['type']
    logging.info(f"Target Bottleneck Locked: Index {idx} ({b_type})")
    logging.info(f"Good: {target_bottleneck['good_us']:.2f}us, "
                f"Bad: {target_bottleneck['bad_us']:.2f}us, "
                f"Threshold: {target_bottleneck['threshold']:.2f}us")
    return 0


def mode_evaluate(current_file, config_file):
    """阶段2：Git Bisect 运行时，检查那个有问题的间隙"""
    current_data = load_json(current_file)
    config = load_json(config_file)

    if current_data is None or config is None:
        return 125

    target = config.get("target_bottleneck")
    if not target or current_data.get("status") != "success":
        return 125

    idx = target["transition_index"]
    current_trans = current_data.get("transitions", [])

    if len(current_trans) <= idx or current_trans[idx]['type'] != target['type']:
        logging.warning(f"Pipeline signature shifted at Index {idx}. Skipping commit.")
        return 125

    curr_gap_us = current_trans[idx]['gap_us']
    threshold = target["threshold"]

    logging.info(f"Evaluating Index {idx} ({target['type']}): "
                f"Current={curr_gap_us:.2f}us, Threshold={threshold:.2f}us")

    if curr_gap_us <= threshold:
        logging.info("=> Result: GOOD (0)")
        return 0
    else:
        logging.info("=> Result: BAD (1)")
        return 1


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

    exit_code = 0
    if args.diff:
        exit_code = mode_diff(args.good, args.bad, args.config)
    elif args.evaluate:
        exit_code = mode_evaluate(args.current, args.config)

    sys.exit(exit_code)