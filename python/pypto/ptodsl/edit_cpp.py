#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""

import re
import sys
from pathlib import Path


# Include swap: generated -> edited
INCLUDE_OLD = '#include "common/pto_instr.hpp"'
INCLUDE_NEW = '#include "pto/pto-inst.hpp"'

# Macro to insert after kernel opening brace
MACRO_IF = """  #if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
"""
# TODO: let `ptoas` distinguish Vec vs Cube macros

# Macro block to insert before return
MACRO_ELSE_ENDIF = """  #else
  #endif

"""

# Pattern for __global__ AICORE void kernel_name(params)
KERNEL_PATTERN = re.compile(
    r"__global__\s+AICORE\s+void\s+(\w+)\s*\((.*)\)\s*\{",
    re.DOTALL,
)

# Pattern for a single param: __gm__ type* name or similar
PARAM_PATTERN = re.compile(r"__gm__\s*(\w+)\s*\*\s*(\w+)")


def parse_kernel_signature(line: str, rest: str = "") -> tuple[str, list[tuple[str, str]]] | None:
    """Match __global__ AICORE void name(...) { and return (name, [(type, name), ...])."""
    combined = (line + rest).strip()
    m = KERNEL_PATTERN.search(combined)  # search in case of leading whitespace
    if not m:
        return None
    kernel_name = m.group(1)
    params_str = m.group(2).strip()
    if not params_str:
        return (kernel_name, [])
    params = []
    for part in params_str.split(","):
        part = part.strip()
        pm = PARAM_PATTERN.search(part)
        if pm:
            params.append((pm.group(1), pm.group(2)))  # type, name
        else:
            # Fallback: treat as "type* name"
            simple = re.match(r"(\w+)\s*\*\s*(\w+)", part)
            if simple:
                params.append((simple.group(1), simple.group(2)))
    return (kernel_name, params)


def convert(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines:
        return content

    out: list[str] = []
    i = 0
    kernel_name: str | None = None
    kernel_params: list[tuple[str, str]] = []
    kernel_signature_line_index = -1
    return_line_index = -1

    # First pass: swap include
    while i < len(lines):
        line = lines[i]
        if INCLUDE_OLD in line:
            out.append(line.replace(INCLUDE_OLD, INCLUDE_NEW))
            i += 1
            break
        out.append(line)
        i += 1

    # Collect rest of file and detect kernel + return
    rest_lines = lines[i:]

    # Find kernel signature (single line or multi-line)
    for idx, rline in enumerate(rest_lines):
        if "__global__" in rline and "AICORE" in rline:
            # Try single line first, then with following lines
            combined = "".join(rest_lines[idx : idx + 3]).strip()
            parsed = parse_kernel_signature(combined, "")
            if parsed:
                kernel_name, kernel_params = parsed
                kernel_signature_line_index = len(out) + idx
                break

    # Re-output remaining lines with insertions
    i = 0
    while i < len(rest_lines):
        line = rest_lines[i]
        line_no_newline = line.rstrip("\n")
        out.append(line)

        # Insert macro immediately after kernel signature line (line ending with "{")
        if (
            kernel_signature_line_index >= 0
            and len(out) - 1 == kernel_signature_line_index
            and "{" in line
            and "__global__" in line
        ):
            out.append("\n")
            out.append(MACRO_IF)
            out.append("  set_mask_norm();\n")

        # Find "  return;" or "  return ;" to insert #else/#endif before it
        if return_line_index < 0 and re.match(r"^\s*return\s*;\s*$", line_no_newline):
            return_line_index = len(out) - 1

        i += 1
    # If we found return, we need to insert before it (retroactively we already output it)
    # So we need to insert the #else/#endif block *before* the return line.
    if return_line_index >= 0:
        # Find the return line in out and insert before it
        insert_at = return_line_index
        block = MACRO_ELSE_ENDIF
        out[insert_at:insert_at] = [block]

    result = "".join(out)

    # Append call_kernel wrapper
    if kernel_name and kernel_params is not None:
        wrapper = build_call_wrapper(kernel_name, kernel_params)
        result = result.rstrip()
        if not result.endswith("\n"):
            result += "\n"
        result += "\n" + wrapper
        if not result.endswith("\n"):
            result += "\n"

    return result


def build_call_wrapper(kernel_name: str, params: list[tuple[str, str]]) -> str:
    """Build extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* v1, ...)."""
    param_decls = ["uint8_t* " + name for _, name in params]
    args = ["uint32_t blockDim", "void* stream"] + param_decls
    cast_args = ["(" + typ + " *)" + name for typ, name in params]
    return '''extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    {param_list})
{{
    {kernel_name}<<<blockDim, nullptr, stream>>>({cast_list});
}}'''.format(
        param_list=", ".join(param_decls),
        kernel_name=kernel_name,
        cast_list=", ".join(cast_args),
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    generated = script_dir / "temp_generated.cpp"
    edited = script_dir / "temp_edited.cpp"

    if not generated.exists():
        print(f"Error: {generated} not found", file=sys.stderr)
        sys.exit(1)

    content = generated.read_text(encoding="utf-8")
    converted = convert(content)
    edited.write_text(converted, encoding="utf-8")
    print(f"Wrote {edited}")


if __name__ == "__main__":
    main()