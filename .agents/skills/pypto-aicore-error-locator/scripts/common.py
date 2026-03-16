#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

import subprocess
import re


def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()


def write_file(file_path, lines):
    with open(file_path, 'w') as f:
        f.writelines(lines)


def get_commentable_lines(lines):
    commentable_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        skip_keywords = ['set_flag', 'wait_flag', 'pipe_barrier']
        should_skip = (
            not stripped or
            stripped.startswith('//') or
            '{' in stripped or
            '}' in stripped or
            '#' in stripped or
            any(keyword in stripped for keyword in skip_keywords)
        )
        if should_skip:
            continue
        else:
            commentable_lines.append(i)

    return commentable_lines


def comment_lines(lines, line_indices):
    lines_to_comment = set(line_indices)
    
    for line_num in sorted(lines_to_comment, reverse=True):
        line_idx = line_num - 1
        lines[line_idx] = '// ' + lines[line_idx]
    
    return lines


def uncomment_lines(lines, line_indices):
    lines_to_uncomment = set(line_indices)
    
    for line_num in sorted(lines_to_uncomment):
        line_idx = line_num - 1
        if lines[line_idx].strip().startswith('//'):
            lines[line_idx] = lines[line_idx][3:]
    
    return lines


def has_error(returncode, output):
    if returncode != 0:
        return True
    
    output_lower = output.lower()
    
    true_error_keywords = [
        ' error',
        'error ',
        'exception',
        'segmentation fault',
        'core dump',
    ]
    
    for keyword in true_error_keywords:
        if keyword in output_lower:
            return True
    
    return False


def run_test(test_cmd, run_dir):
    result = subprocess.run(
        test_cmd,
        shell=True,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=1800
    )
    return result.returncode, result.stdout + result.stderr


def comment_special_lines(lines):
    for i, line in enumerate(lines):
        if 'set_flag' in line or 'wait_flag' in line or 'pipe_barrier' in line:
            if not line.strip().startswith('//'):
                lines[i] = '// ' + line
    return lines


def parse_luid(luid_str):
    match = re.search(r'LUid\{(\d+),(\d+),(\d+),(\d+),(\d+)\}', luid_str)
    if match:
        return {
            'deviceTaskId': int(match.group(1)),
            'funcId': int(match.group(2)),
            'rootIndex': int(match.group(3)),
            'opIdx': int(match.group(4)),
            'leafIndex': int(match.group(5))
        }
    return None


def parse_core_idx(event_str, pattern):
    match = re.search(pattern, event_str)
    if match:
        return int(match.group(1))
    return None
