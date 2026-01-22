#!/usr/bin/env python3

import os
import sys
import re
import argparse
from pathlib import Path

# --- Regex Patterns ---

# Matches a single C-style block comment (non-greedy)
c_style_block = r'/\*.*?\*/'
# Matches C++-style line comments
cpp_style_line = r'//[^\n]*\n'

# Combined regex to match a sequence of one or more comment blocks (License + Doxygen)
# at the very start of the file (\A).
full_header_sequence_regex = re.compile(
    r'\A(?:\s*(?:' + c_style_block + r'|' + cpp_style_line + r'))+',
    re.DOTALL
)

# Individual patterns for initial replacement logic
c_style_header_regex = re.compile(r'\A\s*' + c_style_block, re.DOTALL)
cpp_style_header_regex = re.compile(r'\A(?:\s*' + cpp_style_line + r')+', re.DOTALL)

# Matches "namespace osp {" allowing for whitespace
namespace_osp_regex = re.compile(r'namespace\s+osp\s*\{')

def get_new_header_text(filename):
    """
    Returns the new 2025-2026 header text (License + Doxygen).
    """
    return f"""/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \\file {filename}
 * \\brief
 */"""

def replace_header(file_path: Path, dry_run: bool):
    """
    Replaces the EXISTING header comment with the new License + Doxygen block.
    """
    new_header_text = get_new_header_text(file_path.name)
    
    try:
        content = file_path.read_text(encoding='utf-8')
        content_after_header = None
        
        # 1. Try to find C-style /* ... */ block
        match = c_style_header_regex.search(content)
        if match:
            content_after_header = content[match.end():]
        else:
            # 2. Try to find C++-style // ... block
            match = cpp_style_header_regex.search(content)
            if match:
                content_after_header = content[match.end():]

        if content_after_header is not None:
            new_content = new_header_text + '\n' + content_after_header.lstrip()

            if new_content == content:
                return 

            if not dry_run:
                file_path.write_text(new_content, encoding='utf-8')
                print(f"SUCCESS [Header]: Updated {file_path.name}")
            else:
                print(f"DRY-RUN [Header]: Would update {file_path.name}")
        else:
            print(f"WARN: No header comment found to replace in: {file_path.name}")

    except Exception as e:
        print(f"ERROR [Header]: {file_path.name}: {e}")

def wrap_namespace(file_path: Path, dry_run: bool):
    """
    Wraps 'namespace osp { ... }' inside 'namespace npu::tile_fwk { ... }'.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Avoid double wrapping
        if "namespace npu::tile_fwk" in content:
            return

        # Search for 'namespace osp {'
        match = namespace_osp_regex.search(content)
        
        if match:
            start_index = match.start()
            
            pre_content = content[:start_index]
            post_content = content[start_index:]
            
            # We insert the closing brace at the very end of the current content.
            # The .rstrip() ensures we don't pile up newlines before adding our brace.
            new_content = (
                f"{pre_content.rstrip()}\n\n"
                f"namespace npu::tile_fwk {{\n"
                f"{post_content.rstrip()}\n"
                f"}} // namespace npu::tile_fwk\n"
            )

            if not dry_run:
                file_path.write_text(new_content, encoding='utf-8')
                print(f"SUCCESS [Namespace]: Wrapped around 'namespace osp' in {file_path.name}")
            else:
                print(f"DRY-RUN [Namespace]: Would wrap around 'namespace osp' in {file_path.name}")

    except Exception as e:
        print(f"ERROR [Namespace]: {file_path.name}: {e}")

def process_pragma_once(file_path: Path, dry_run: bool):
    """
    Replaces #pragma once with #ifndef include guards.
    Ensures guards are placed AFTER both the License and Doxygen headers.
    Ensures #endif is placed at the VERY END of the file (wrapping namespaces).
    """
    if file_path.suffix not in {'.hpp', '.h', '.hh'}:
        return

    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check if #pragma once exists
        has_pragma = bool(re.search(r'^\s*#\s*pragma\s+once', content, re.MULTILINE))
        
        if not has_pragma:
            return

        print(f"INFO: Replacing #pragma once in '{file_path.name}'...")

        # 1. Remove #pragma once
        content_no_pragma = re.sub(r'^\s*#\s*pragma\s+once\s*\n?', '', content, flags=re.MULTILINE)

        # 2. Determine Insertion Point (After License + Doxygen)
        header_match = full_header_sequence_regex.match(content_no_pragma)
        
        header_end_idx = 0
        if header_match:
            header_end_idx = header_match.end()
        
        header_part = content_no_pragma[:header_end_idx]
        body_part = content_no_pragma[header_end_idx:].lstrip()

        # 3. Generate Guard Name
        safe_name = file_path.name.upper().replace('.', '_')
        guard_name = f"OSP_{safe_name}"

        # 4. Construct new content
        # We .rstrip() the body_part to remove any trailing newlines/spaces
        # This ensures #endif comes immediately after the code.
        new_content = (
            f"{header_part}\n\n"
            f"#ifndef {guard_name}\n"
            f"#define {guard_name}\n\n"
            f"{body_part.rstrip()}\n"
            f"#endif // {guard_name}\n"
        )

        if not dry_run:
            file_path.write_text(new_content, encoding='utf-8')
            print(f"SUCCESS [Pragma]: Replaced #pragma once in {file_path.name}")
        else:
            print(f"DRY-RUN [Pragma]: Would replace #pragma once in {file_path.name}")

    except Exception as e:
        print(f"ERROR [Pragma]: {file_path.name}: {e}")

def change_osp_location(file_path: Path, dry_run: bool):
    """
    Updates #include paths from "osp/" to "passes/algorithms/osp/"
    """
    prefix_to_be_replaced = '#include "osp/'
    prefix_replacement = '#include "passes/algorithms/osp/'
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        if prefix_to_be_replaced not in content:
            return

        lines = content.splitlines(keepends=True)
        new_lines = []
        modified = False

        for line in lines:
            if line.strip().startswith(prefix_to_be_replaced):
                start_index = line.find(prefix_to_be_replaced)
                if start_index != -1:
                    rest_of_line = line[start_index + len(prefix_to_be_replaced):]
                    new_line = line[:start_index] + prefix_replacement + rest_of_line
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if modified:
            if not dry_run:
                file_path.write_text("".join(new_lines), encoding='utf-8')
                print(f"SUCCESS [Include]: Updated OSP path in {file_path.name}")
            else:
                print(f"DRY-RUN [Include]: Would update OSP path in {file_path.name}")

    except Exception as e:
        print(f"ERROR [Include]: {file_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Refactor C++ headers, namespaces, and includes."
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory to search recursively for .cpp and .hpp files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without modifying any files."
    )
    args = parser.parse_args()

    if not args.target_dir.is_dir():
        print(f"ERROR: Target directory not found: {args.target_dir}")
        return

    print("--- New Header Text Preview ---")
    print(get_new_header_text("filename.cpp"))
    print("-------------------------------")

    if args.dry_run:
        print("\n*** RUNNING IN DRY-RUN MODE - NO FILES WILL BE CHANGED ***\n")

    print(f"Scanning in {args.target_dir}...\n")
    
    extensions = {'.cpp', '.hpp', '.c', '.h', '.cc', '.hh'}
    file_count = 0
    
    for file_path in args.target_dir.rglob('*'):
        if file_path.suffix in extensions:
            file_count += 1
            
            # --- EXECUTION ORDER IS CRITICAL ---
            
            # 1. Update License Header (Puts License + Doxygen at top)
            replace_header(file_path, args.dry_run)
            
            # 2. Update OSP Include paths
            change_osp_location(file_path, args.dry_run)
            
            # 3. Wrap in namespace (if 'namespace osp' exists)
            # Must happen BEFORE pragma replacement so closing brace is inside body
            wrap_namespace(file_path, args.dry_run)
            
            # 4. Replace #pragma once with include guards
            # Must happen LAST. Wraps everything (including new namespace) in #ifndef
            process_pragma_once(file_path, args.dry_run)

    print(f"\nScan complete. Checked {file_count} files.")

if __name__ == "__main__":
    main()