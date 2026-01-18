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
"""
IR to NumPy Converter

Converts IR functions to NumPy implementations.
This converter reads IR text format and generates equivalent NumPy code.
"""
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class IRToNumPyConverter:
    """Converter from IR text format to NumPy code."""

    def __init__(self):
        self.dtype_map = {
            "fp32": "np.float32",
            "fp64": "np.float64",
            "int32": "np.int32",
            "int64": "np.int64",
            "bool": "bool",
        }
        self.op_map = {
            "tensor.mul": "*",
            "tensor.add": "+",
            "tensor.sub": "-",
            "tensor.div": "/",
            "tile.OP_MUL": "*",
            "tile.OP_ADD": "+",
            "tile.OP_SUB": "-",
            "tile.OP_DIV": "/",
            "tensor.OP_SCALAR_ADD": "+",
            "tensor.OP_SCALAR_MUL": "*",
            "tensor.OP_SCALAR_SUB": "-",
            "tensor.OP_SCALAR_DIV": "/",
            "tensor.neg": "np.negative",
            "tensor.abs": "np.abs",
            "tensor.exp": "np.exp",
            "tensor.sqrt": "np.sqrt",
            "tensor.ln": "np.log",
        }
        # PyTorch operation map for golden computation
        self.torch_op_map = {
            "tensor.mul": "*",
            "tensor.add": "+",
            "tensor.sub": "-",
            "tensor.div": "/",
            "tile.OP_MUL": "*",
            "tile.OP_ADD": "+",
            "tile.OP_SUB": "-",
            "tile.OP_DIV": "/",
            "tensor.OP_SCALAR_ADD": "+",
            "tensor.OP_SCALAR_MUL": "*",
            "tensor.OP_SCALAR_SUB": "-",
            "tensor.OP_SCALAR_DIV": "/",
            "tensor.neg": "torch.neg",
            "tensor.abs": "torch.abs",
            "tensor.exp": "torch.exp",
            "tensor.sqrt": "torch.sqrt",
            "tensor.ln": "torch.log",
        }
        self.symbolic_vars = {}  # Track symbolic variables

    def convert_ir_file(self, ir_file: Path) -> str:
        """Convert an IR file to NumPy code."""
        with open(ir_file, "r") as f:
            ir_text = f.read()
        return self.convert_ir_text(ir_text)

    def convert_ir_text(self, ir_text: str) -> str:
        """Convert IR text to NumPy code."""
        lines = []
        lines.append("import numpy as np")
        lines.append("")
        
        # Extract function signature
        func_match = re.search(
            r"func\.func @(\w+)\((.*?)\)(?: -> \((.*?)\))?",
            ir_text,
            re.DOTALL
        )
        if not func_match:
            raise ValueError("Could not parse function signature")
        
        func_name = func_match.group(1)
        params = func_match.group(2)
        return_type = func_match.group(3) if func_match.group(3) else None
        
        # Parse parameters
        param_list = self._parse_parameters(params)
        
        # Generate function signature
        numpy_params = []
        for param_name, param_type, _ in param_list:
            numpy_params.append(f"{param_name}: np.ndarray")
        
        return_annotation = ""
        if return_type:
            return_annotation = f" -> {self._convert_type(return_type)}"
        
        lines.append(f"def {func_name}({', '.join(numpy_params)}){return_annotation}:")
        lines.append('    """NumPy implementation converted from IR."""')
        lines.append("")
        
        # Check for control flow structures
        has_for_loop = "statement.for" in ir_text
        has_if_stmt = "statement.if" in ir_text
        
        if has_for_loop or has_if_stmt:
            # For control flow, we need to extract and convert them
            # For now, extract operations from all statement.op blocks (including nested ones)
            lines.append("    # Note: Control flow (for/if) detected - simplified conversion")
            lines.append("    # TODO: Implement full control flow conversion")
            lines.append("")
        
        # Extract operations from statement.op blocks (including nested ones)
        # Use a more robust pattern that handles nested braces
        op_block_matches = re.finditer(r"statement\.op \{", ir_text)
        op_blocks = []
        for match in op_block_matches:
            start = match.end()
            brace_count = 1
            pos = start
            while pos < len(ir_text) and brace_count > 0:
                if ir_text[pos] == '{':
                    brace_count += 1
                elif ir_text[pos] == '}':
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                op_blocks.append(ir_text[start:pos-1])
        
        # Track constants and variables defined in op blocks
        constants = {}
        defined_vars = set()  # Track defined variables to handle undefined references
        
        # Store operations for golden data generation
        all_operations = []
        
        # Extract for loop variables and iter_args
        for_loop_vars = {}  # Map loop variable names to their iter_args
        loop_result_vars = {}  # Map loop variable names to their result variables (e.g., %acc_output = statement.for ...)
        if has_for_loop:
            # Extract for loop information - handle both with and without result variable
            # Pattern 1: %result = statement.for %loop_var = ...
            # Pattern 2: statement.for %loop_var = ... (no result variable)
            for_pattern_with_result = r"%(\w+)(?:,\s*%(\w+))?\s*=\s*statement\.for\s+%(\w+)\s*=\s*%(\w+)\s+to\s*%(\w+)\s+step\s*%(\w+).*?iter_args\((.*?)\)"
            for_match = re.search(for_pattern_with_result, ir_text, re.DOTALL)
            if for_match:
                result_vars = [for_match.group(1)]
                if for_match.group(2):
                    result_vars.append(for_match.group(2))
                loop_var = for_match.group(3)
                start_var = for_match.group(4)
                end_var = for_match.group(5)
                step_var = for_match.group(6)
                iter_args_str = for_match.group(7)
                # Parse iter_args: %acc0 = %output_8 : type
                iter_args = re.findall(r"%(\w+)\s*=\s*%(\w+)", iter_args_str)
                for_loop_vars[loop_var] = {
                    "start": start_var,
                    "end": end_var,
                    "step": step_var,
                    "iter_args": {acc: init for acc, init in iter_args}
                }
                # Store loop result variables
                loop_result_vars[loop_var] = result_vars
            else:
                # Try pattern without result variable
                for_match = re.search(r"statement\.for\s+%(\w+)\s*=\s*%(\w+)\s+to\s*%(\w+)\s+step\s*%(\w+).*?iter_args\((.*?)\)", ir_text, re.DOTALL)
                if for_match:
                    loop_var = for_match.group(1)
                    start_var = for_match.group(2)
                    end_var = for_match.group(3)
                    step_var = for_match.group(4)
                    iter_args_str = for_match.group(5)
                    # Parse iter_args: %acc0 = %output_8 : type
                    iter_args = re.findall(r"%(\w+)\s*=\s*%(\w+)", iter_args_str)
                    for_loop_vars[loop_var] = {
                        "start": start_var,
                        "end": end_var,
                        "step": step_var,
                        "iter_args": {acc: init for acc, init in iter_args}
                    }
                    loop_result_vars[loop_var] = []  # No result variable
                # Initialize accumulator variables
                for acc, init in iter_args:
                    if init in [p[0] for p in param_list]:
                        # Always modify the parameter in place for accumulators
                        # This allows the result to be visible after the function call
                        lines.append(f"    # Initialize accumulator {acc} from {init} (modify in place)")
                        lines.append(f"    {acc} = {init}")
                        defined_vars.add(acc)
                    # Also initialize in test function
                    # This will be handled in the test function generation
        
        # Extract if statement results and yield mapping
        if_results = {}  # Map if result variables
        if_branch_vars = {}  # Track variables from then/else branches
        yield_mapping = {}  # Map if result variable to (then_vars_list, else_vars_list)
        if has_if_stmt:
            # Extract if statements and their yield values
            # Pattern: %r0 = statement.if %cond { ... statement.yield %var1 } else { ... statement.yield %var2 }
            # More flexible pattern to handle nested braces
            if_pattern = r"%(\w+)(?:,\s*%(\w+))?\s*=\s*statement\.if\s+%(\w+)\s*\{"
            if_matches = list(re.finditer(if_pattern, ir_text))
            
            for if_match in if_matches:
                result_vars = [if_match.group(1)]
                if if_match.group(2):
                    result_vars.append(if_match.group(2))
                cond_var = if_match.group(3)
                start_pos = if_match.end()
                
                # Find then block (from { to matching })
                brace_count = 1
                pos = start_pos
                while pos < len(ir_text) and brace_count > 0:
                    if ir_text[pos] == '{':
                        brace_count += 1
                    elif ir_text[pos] == '}':
                        brace_count -= 1
                    pos += 1
                then_block = ir_text[start_pos:pos-1]
                
                # Find else block (after "} else {")
                else_start = ir_text.find("} else {", pos-1)
                if else_start != -1:
                    else_start += len("} else {")
                    brace_count = 1
                    pos = else_start
                    while pos < len(ir_text) and brace_count > 0:
                        if ir_text[pos] == '{':
                            brace_count += 1
                        elif ir_text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    else_block = ir_text[else_start:pos-1]
                    
                    # Extract yield from then and else blocks
                    then_yield = re.search(r"statement\.yield\s+(.*?)(?:\n|$)", then_block)
                    else_yield = re.search(r"statement\.yield\s+(.*?)(?:\n|$)", else_block)
                    
                    if then_yield and else_yield:
                        then_vars = re.findall(r"%(\w+)", then_yield.group(1))
                        else_vars = re.findall(r"%(\w+)", else_yield.group(1))
                        # Map result variables to yield variables
                        # Check if this is a tuple result (multiple yield values)
                        is_tuple_result = len(then_vars) > 1 or len(else_vars) > 1
                        for i, rv in enumerate(result_vars):
                            if_results[rv] = cond_var
                            if is_tuple_result:
                                # For tuple access, map all yield vars
                                yield_mapping[rv] = (then_vars, else_vars)
                            elif i < len(then_vars) and i < len(else_vars):
                                # Single yield value
                                yield_mapping[rv] = (then_vars[i], else_vars[i])
                            elif len(then_vars) > 0 and len(else_vars) > 0:
                                # Fallback: map all yield vars
                                yield_mapping[rv] = (then_vars, else_vars)
        
        # Process operations and handle for loops and if statement branches
        # First, identify which op blocks are inside for loops
        for_loop_blocks = {}  # Map loop_var to list of op blocks inside that loop
        if has_for_loop:
            for loop_var, loop_info in for_loop_vars.items():
                # Find the for loop block in IR
                for_pattern = rf"statement\.for\s+%{loop_var}\s*=\s*%(\w+)\s+to\s*%(\w+)\s+step\s*%(\w+).*?iter_args\(.*?\)\s*\{{"
                for_match = re.search(for_pattern, ir_text, re.DOTALL)
                if for_match:
                    loop_start = for_match.end()
                    # Find the matching closing brace
                    brace_count = 1
                    pos = loop_start
                    while pos < len(ir_text) and brace_count > 0:
                        if ir_text[pos] == '{':
                            brace_count += 1
                        elif ir_text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    loop_body = ir_text[loop_start:pos-1]
                    # Extract yield statement - find the LAST statement.yield in the loop body
                    # (the one at the end of the loop, not inside if statements)
                    yield_matches = list(re.finditer(r"statement\.yield\s+(.*?)(?:\n|$)", loop_body))
                    if yield_matches:
                        # Use the last match (the one at the end of the loop)
                        yield_match = yield_matches[-1]
                        yield_vars = re.findall(r"%(\w+)", yield_match.group(1))
                        loop_info["yield_vars"] = yield_vars
                    # Find op blocks inside this loop
                    for_loop_blocks[loop_var] = []
                    for op_block in op_blocks:
                        # Check if this op block is inside the loop body
                        op_start = ir_text.find(op_block)
                        if op_start >= loop_start and op_start < pos-1:
                            for_loop_blocks[loop_var].append(op_block)
        
        # Process operations - handle for loops first
        for loop_var, loop_info in for_loop_vars.items():
            if loop_var in for_loop_blocks:
                # Initialize accumulator variables BEFORE the loop
                iter_args = loop_info.get("iter_args", {})
                for acc, init in iter_args.items():
                    if init in [p[0] for p in param_list]:
                        # Always modify the parameter in place for accumulators
                        # This allows the result to be visible after the function call
                        lines.append(f"    # Initialize accumulator {acc} from {init} (modify in place)")
                        lines.append(f"    {acc} = {init}")
                        defined_vars.add(acc)
                
                # Get loop bounds
                start_var = loop_info["start"]
                end_var = loop_info["end"]
                step_var = loop_info["step"]
                # Convert symbolic bounds to concrete values for testing
                # For now, use a simple range
                lines.append(f"    # For loop: {loop_var} from {start_var} to {end_var} step {step_var}")
                lines.append(f"    # Simplified: iterate over range")
                lines.append(f"    for {loop_var} in range(4):  # TODO: Use actual bounds")
                # Process operations inside the loop
                # First, collect all operations and check if any use tuple access (r0#0, r0#1, etc.)
                loop_operations = []
                needs_tuple_access = {}  # Map rv -> list of indices needed
                for op_block in for_loop_blocks[loop_var]:
                    operations = self._parse_operations(op_block)
                    all_operations.extend(operations)
                    loop_operations.extend(operations)
                    # Check if any operation uses tuple access
                    for op in operations:
                        operands = op.get("operands", [])
                        for operand in operands:
                            if "#" in operand:
                                # Extract base variable and index
                                var_name = operand.replace("%", "").split("#")[0]
                                idx = operand.split("#")[1]
                                if var_name in if_results:
                                    if var_name not in needs_tuple_access:
                                        needs_tuple_access[var_name] = []
                                    if idx not in needs_tuple_access[var_name]:
                                        needs_tuple_access[var_name].append(idx)
                
                # Process operations and track defined variables
                # We'll create tuple access variables after all yield variables are assigned
                for op in loop_operations:
                    numpy_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping, loop_result_vars)
                    if numpy_code:
                        lines.append(f"        {numpy_code}")
                        # Track defined variables
                        if '=' in numpy_code:
                            var_name = numpy_code.split('=')[0].strip()
                            defined_vars.add(var_name)
                            # Check if this variable should be assigned to an if result (for if statements inside loops)
                            for rv, yv_data in yield_mapping.items():
                                if isinstance(yv_data, tuple):
                                    if isinstance(yv_data[0], list):
                                        # Multiple yield values (tuple result)
                                        then_vars, else_vars = yv_data
                                        if var_name in then_vars:
                                            idx = then_vars.index(var_name)
                                            if f"{rv}_then_val_{idx}" not in defined_vars:
                                                lines.append(f"        {rv}_then_val_{idx} = {var_name}  # Assign then yield to if result")
                                                defined_vars.add(f"{rv}_then_val_{idx}")
                                        if var_name in else_vars:
                                            idx = else_vars.index(var_name)
                                            if f"{rv}_else_val_{idx}" not in defined_vars:
                                                lines.append(f"        {rv}_else_val_{idx} = {var_name}  # Assign else yield to if result")
                                                defined_vars.add(f"{rv}_else_val_{idx}")
                                    else:
                                        # Single yield value - only create _then/_else for single values
                                        # For tuple results, we should have already handled it above
                                        then_var, else_var = yv_data
                                        # Only create _then/_else if this is truly a single yield value
                                        # Check if yield_data is a tuple of two strings (not lists)
                                        if not isinstance(then_var, list) and not isinstance(else_var, list):
                                            if var_name == then_var:
                                                if f"{rv}_then" not in defined_vars:
                                                    lines.append(f"        {rv}_then = {var_name}  # Assign yield to if result")
                                                    defined_vars.add(f"{rv}_then")
                                            if var_name == else_var:
                                                if f"{rv}_else" not in defined_vars:
                                                    lines.append(f"        {rv}_else = {var_name}  # Assign yield to if result")
                                                    defined_vars.add(f"{rv}_else")
                                            # Define the if result if both branches are ready
                                            if f"{rv}_then" in defined_vars and f"{rv}_else" in defined_vars and rv not in defined_vars:
                                                cond_var = if_results[rv]
                                                lines.append(f"        {rv} = {rv}_then if {cond_var} else {rv}_else")
                                                defined_vars.add(rv)
                
                # After processing all operations in the loop, create tuple access variables for if results with multiple yields
                # This must be done AFTER all yield variables are assigned, but we need to insert them before operations that use them
                # So we'll find where the first operation using tuple access is and insert the tuple access variable creation before it
                for rv in if_results.keys():
                    if rv in yield_mapping and rv in needs_tuple_access:
                        yield_data = yield_mapping[rv]
                        if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                            # Multiple yield values (tuple result) - create access variables inside the loop
                            then_vars, else_vars = yield_data
                            cond_var = if_results[rv]
                            # Check if all yield variables are assigned
                            all_then_assigned = all(f"{rv}_then_val_{i}" in defined_vars for i in range(len(then_vars)))
                            all_else_assigned = all(f"{rv}_else_val_{i}" in defined_vars for i in range(len(else_vars)))
                            if all_then_assigned and all_else_assigned:
                                # Find the position where the first operation using tuple access appears
                                # We need to insert the tuple access variable creation before that
                                insert_pos = None
                                for i, line in enumerate(lines):
                                    if f"{rv}_val_" in line or f"{rv}#" in line:
                                        # Found a line that uses tuple access - insert before it
                                        insert_pos = i
                                        break
                                if insert_pos is None:
                                    # No operation uses tuple access yet, append at the end of loop operations
                                    # Find the last line with proper indentation (inside the loop)
                                    for i in range(len(lines) - 1, -1, -1):
                                        if lines[i].startswith("        ") and not lines[i].startswith("        #"):
                                            insert_pos = i + 1
                                            break
                                if insert_pos is not None:
                                    for idx in needs_tuple_access[rv]:
                                        if f"{rv}_val_{idx}" not in defined_vars:
                                            then_val = f"{rv}_then_val_{idx}" if f"{rv}_then_val_{idx}" in defined_vars else (then_vars[int(idx)] if int(idx) < len(then_vars) and then_vars[int(idx)] in defined_vars else f"{rv}_then_val_{idx}")
                                            else_val = f"{rv}_else_val_{idx}" if f"{rv}_else_val_{idx}" in defined_vars else (else_vars[int(idx)] if int(idx) < len(else_vars) and else_vars[int(idx)] in defined_vars else f"{rv}_else_val_{idx}")
                                            lines.insert(insert_pos, f"        {rv}_val_{idx} = {then_val} if {cond_var} else {else_val}")
                                            defined_vars.add(f"{rv}_val_{idx}")
                                            insert_pos += 1  # Adjust for next insertion
                # Handle yield - update accumulator
                yield_vars = loop_info.get("yield_vars", [])
                if yield_vars:
                    # The yield statement yields the new accumulator value
                    # Map yield variables to accumulator variables
                    iter_args = loop_info.get("iter_args", {})
                    # Track which operations result in accumulator updates
                    # For assemble operations, the result is the accumulator itself (modified in place)
                    acc_result_vars = {}  # Map result_var -> acc_var for assemble operations
                    for op_block in for_loop_blocks[loop_var]:
                        operations = self._parse_operations(op_block)
                        for op in operations:
                            if op["op"] in ["tensor.assemble", "tile.assemble"]:
                                if len(op.get("operands", [])) >= 2:
                                    dst_var = op["operands"][1].replace("%", "")
                                    result_var = op["result"]
                                    # If destination is an accumulator, the result is the accumulator
                                    if dst_var in iter_args:
                                        acc_result_vars[result_var] = dst_var
                    
                    for i, yield_var in enumerate(yield_vars):
                        # Find corresponding accumulator (by position or name match)
                        acc_list = list(iter_args.keys())
                        if i < len(acc_list):
                            acc = acc_list[i]
                            # Check if yield_var is the result of an assemble operation on this accumulator
                            if yield_var in acc_result_vars and acc_result_vars[yield_var] == acc:
                                # The assemble operation already modified acc in place, no need to assign
                                # The accumulator is already updated by the assemble operation
                                pass
                            elif yield_var == acc:
                                # Yield variable is the accumulator itself - already updated in place
                                pass
                            else:
                                # Only assign if yield_var is different from acc and is defined
                                # But first check if yield_var is actually an assemble result that we missed
                                if yield_var in acc_result_vars:
                                    # This is an assemble result, but for a different accumulator
                                    # Don't assign - the accumulator was already updated
                                    pass
                                elif yield_var in defined_vars:
                                    # This is a different variable - assign it to the accumulator
                                    lines.append(f"        {acc} = {yield_var}  # Update accumulator from yield")
                            defined_vars.add(acc)
                        else:
                            # Fallback: try to match by name or check if it's an assemble result
                            assigned = False
                            for acc in acc_list:
                                if yield_var in acc_result_vars and acc_result_vars[yield_var] == acc:
                                    # The assemble operation already modified acc in place
                                    defined_vars.add(acc)
                                    assigned = True
                                    break
                                elif yield_var == acc or yield_var.endswith(acc):
                                    if yield_var != acc and yield_var in defined_vars:
                                        lines.append(f"        {acc} = {yield_var}  # Update accumulator from yield")
                                    defined_vars.add(acc)
                                    assigned = True
                                    break
                            if not assigned:
                                # If we can't match, try to find the accumulator by checking assemble results
                                for result_var, acc_var in acc_result_vars.items():
                                    if yield_var == result_var:
                                        defined_vars.add(acc_var)
                                        break
                
                # Assign loop result variable(s) after the loop
                if loop_var in loop_result_vars and loop_result_vars[loop_var]:
                    result_vars = loop_result_vars[loop_var]
                    iter_args = loop_info.get("iter_args", {})
                    acc_list = list(iter_args.keys())
                    for i, result_var in enumerate(result_vars):
                        if i < len(acc_list):
                            acc = acc_list[i]
                            # The accumulator was modified in place, so assign it to the result variable
                            lines.append(f"    # Assign loop result {result_var} from accumulator {acc}")
                            lines.append(f"    {result_var} = {acc}")
                            defined_vars.add(result_var)
        
        # Finalize if result variables BEFORE processing operations that use them
        # This ensures r0 is defined before it's used in assemble operations
        # But skip tuple results that are inside loops - they're handled in the loop
        for rv in if_results.keys():
            # Skip if this is inside a loop - it's already handled
            is_in_loop = False
            for loop_var, loop_blocks in for_loop_blocks.items():
                for op_block in loop_blocks:
                    operations = self._parse_operations(op_block)
                    for op in operations:
                        operands = op.get("operands", [])
                        for operand in operands:
                            if "#" in operand:
                                var_name = operand.replace("%", "").split("#")[0]
                                if var_name == rv:
                                    is_in_loop = True
                                    break
                        if is_in_loop:
                            break
                    if is_in_loop:
                        break
                if is_in_loop:
                    break
            
            if is_in_loop:
                continue  # Skip - already handled in the loop
            
            cond_var = if_results[rv]
            
            # Check if we have yield mapping for this if result
            if rv in yield_mapping:
                yield_data = yield_mapping[rv]
                if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                    # Multiple yield values (tuple result)
                    then_vars, else_vars = yield_data
                    # Create tuple result from indexed values
                    lines.append(f"    # Create tuple result for {rv}")
                    then_tuple_parts = []
                    else_tuple_parts = []
                    for i in range(len(then_vars)):
                        then_val = then_vars[i] if then_vars[i] in defined_vars else f"{rv}_then_val_{i}"
                        else_val = else_vars[i] if else_vars[i] in defined_vars else f"{rv}_else_val_{i}"
                        then_tuple_parts.append(then_val)
                        else_tuple_parts.append(else_val)
                        # Create indexed access variables - will be computed when then/else vars are defined
                        # Store for later use
                        if f"{rv}_val_{i}" not in defined_vars:
                            defined_vars.add(f"{rv}_val_{i}")  # Mark as will be defined
                    # Store tuple parts for later assignment
                    if f"{rv}_then" not in defined_vars:
                        defined_vars.add(f"{rv}_then")
                    if f"{rv}_else" not in defined_vars:
                        defined_vars.add(f"{rv}_else")
                    # Will be assigned after then/else vars are computed
                    if rv not in defined_vars:
                        defined_vars.add(rv)  # Mark as will be defined
                else:
                    # Single yield value
                    then_var, else_var = yield_data
                    # Ensure both branches are initialized before creating the if result
                    if f"{rv}_then" not in defined_vars:
                        if then_var in defined_vars:
                            lines.append(f"    {rv}_then = {then_var}")
                        else:
                            # Will be assigned when then_var is computed
                            pass
                        defined_vars.add(f"{rv}_then")
                    if f"{rv}_else" not in defined_vars:
                        if else_var in defined_vars:
                            lines.append(f"    {rv}_else = {else_var}")
                        else:
                            # Will be assigned when else_var is computed
                            pass
                        defined_vars.add(f"{rv}_else")
                    
                    # Create the final if result - but only if both branches are ready
                    if then_var in defined_vars and else_var in defined_vars:
                        lines.append(f"    {rv} = {rv}_then if {cond_var} else {rv}_else")
                        defined_vars.add(rv)
        
        # Process operations outside for loops
        for op_block in op_blocks:
            # Skip if this block is inside a for loop
            is_in_loop = False
            for loop_var, loop_blocks in for_loop_blocks.items():
                if op_block in loop_blocks:
                    is_in_loop = True
                    break
            if is_in_loop:
                continue
            
            operations = self._parse_operations(op_block)
            all_operations.extend(operations)
            
            for op in operations:
                # Check if this operation uses an if result or loop result that hasn't been defined yet
                # If so, define it first
                operands = op.get("operands", [])
                for operand in operands:
                    if operand.startswith("%"):
                        var_name = operand[1:]
                        # Check if it's a loop result variable that needs to be defined
                        is_loop_result = False
                        for loop_var, result_vars in loop_result_vars.items():
                            if var_name in result_vars:
                                is_loop_result = True
                                if var_name not in defined_vars:
                                    # The loop result variable should have been assigned after the loop
                                    # But if it's not in defined_vars, it means the assignment wasn't added
                                    # This shouldn't happen, but we'll add a check
                                    loop_info = for_loop_vars[loop_var]
                                    iter_args = loop_info.get("iter_args", {})
                                    acc_list = list(iter_args.keys())
                                    result_idx = result_vars.index(var_name)
                                    if result_idx < len(acc_list):
                                        acc = acc_list[result_idx]
                                        lines.append(f"    # Assign loop result {var_name} from accumulator {acc} (late assignment)")
                                        lines.append(f"    {var_name} = {acc}")
                                        defined_vars.add(var_name)
                                break
                        if is_loop_result:
                            continue
                        # Check if it's an if result that needs to be defined
                        if var_name in if_results and var_name not in defined_vars:
                            # Define the if result before processing this operation
                            rv = var_name
                            cond_var = if_results[rv]
                            if rv in yield_mapping:
                                yield_data = yield_mapping[rv]
                                if isinstance(yield_data, tuple) and not isinstance(yield_data[0], list):
                                    then_var, else_var = yield_data
                                    # Ensure both branches are defined
                                    if f"{rv}_then" not in defined_vars and then_var in defined_vars:
                                        lines.append(f"    {rv}_then = {then_var}  # Assign yield to if result")
                                        defined_vars.add(f"{rv}_then")
                                    if f"{rv}_else" not in defined_vars and else_var in defined_vars:
                                        lines.append(f"    {rv}_else = {else_var}  # Assign yield to if result")
                                        defined_vars.add(f"{rv}_else")
                                    # Define the if result if both branches are ready
                                    if f"{rv}_then" in defined_vars and f"{rv}_else" in defined_vars and rv not in defined_vars:
                                        lines.append(f"    {rv} = {rv}_then if {cond_var} else {rv}_else")
                                        defined_vars.add(rv)
                
                numpy_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping)
                if numpy_code:
                    lines.append(f"    {numpy_code}")
                    # Track defined variables
                    if '=' in numpy_code:
                        var_name = numpy_code.split('=')[0].strip()
                        defined_vars.add(var_name)
                        # Check if this variable should be assigned to an if result
                        for rv, yv_data in yield_mapping.items():
                            if isinstance(yv_data, tuple):
                                if isinstance(yv_data[0], list):
                                    # Multiple yield values
                                    then_vars, else_vars = yv_data
                                    if var_name in then_vars:
                                        # This is a then branch yield variable
                                        idx = then_vars.index(var_name)
                                        if f"{rv}_then_val_{idx}" not in defined_vars:
                                            lines.append(f"    {rv}_then_val_{idx} = {var_name}  # Assign then yield to if result")
                                            defined_vars.add(f"{rv}_then_val_{idx}")
                                    if var_name in else_vars:
                                        # This is an else branch yield variable
                                        idx = else_vars.index(var_name)
                                        if f"{rv}_else_val_{idx}" not in defined_vars:
                                            lines.append(f"    {rv}_else_val_{idx} = {var_name}  # Assign else yield to if result")
                                            defined_vars.add(f"{rv}_else_val_{idx}")
                                else:
                                    # Single yield value
                                    then_var, else_var = yv_data
                                    if var_name == then_var:
                                        if f"{rv}_then" not in defined_vars or f"{rv}_then" not in [line.split('=')[0].strip() for line in lines if '=' in line]:
                                            lines.append(f"    {rv}_then = {var_name}  # Assign yield to if result")
                                            defined_vars.add(f"{rv}_then")
                                    if var_name == else_var:
                                        if f"{rv}_else" not in defined_vars or f"{rv}_else" not in [line.split('=')[0].strip() for line in lines if '=' in line]:
                                            lines.append(f"    {rv}_else = {var_name}  # Assign yield to if result")
                                            defined_vars.add(f"{rv}_else")
        
        # Finalize if result variables - assign yield variables to if results (after operations)
        # But skip tuple results that are inside loops - they're handled in the loop
        for rv in if_results.keys():
            # Skip if this is inside a loop - it's already handled
            is_in_loop = False
            for loop_var, loop_blocks in for_loop_blocks.items():
                for op_block in loop_blocks:
                    operations = self._parse_operations(op_block)
                    for op in operations:
                        operands = op.get("operands", [])
                        for operand in operands:
                            if "#" in operand:
                                var_name = operand.replace("%", "").split("#")[0]
                                if var_name == rv:
                                    is_in_loop = True
                                    break
                        if is_in_loop:
                            break
                    if is_in_loop:
                        break
                if is_in_loop:
                    break
            
            if is_in_loop:
                continue  # Skip - already handled in the loop
            
            cond_var = if_results[rv]
            
            # Check if we have yield mapping for this if result
            if rv in yield_mapping:
                yield_data = yield_mapping[rv]
                if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                    # Multiple yield values (tuple result)
                    then_vars, else_vars = yield_data
                    # Create tuple result from indexed values
                    lines.append(f"    # Create tuple result for {rv}")
                    then_tuple_parts = []
                    else_tuple_parts = []
                    for i in range(len(then_vars)):
                        then_val = f"{rv}_then_val_{i}" if f"{rv}_then_val_{i}" in defined_vars else then_vars[i]
                        else_val = f"{rv}_else_val_{i}" if f"{rv}_else_val_{i}" in defined_vars else else_vars[i]
                        then_tuple_parts.append(then_val)
                        else_tuple_parts.append(else_val)
                        # Create indexed access variables
                        lines.append(f"    {rv}_val_{i} = {then_val} if {cond_var} else {else_val}")
                        defined_vars.add(f"{rv}_val_{i}")
                    lines.append(f"    {rv}_then = ({', '.join(then_tuple_parts)})")
                    lines.append(f"    {rv}_else = ({', '.join(else_tuple_parts)})")
                    if rv not in defined_vars:
                        lines.append(f"    {rv} = {rv}_then if {cond_var} else {rv}_else")
                        defined_vars.add(rv)
                else:
                    # Single yield value
                    then_var, else_var = yield_data
                    # Ensure both branches are initialized before creating the if result
                    if f"{rv}_then" not in defined_vars:
                        # Try to find then_var in operations
                        if then_var in defined_vars:
                            lines.append(f"    {rv}_then = {then_var}")
                        else:
                            lines.append(f"    {rv}_then = None  # Fallback initialization")
                        defined_vars.add(f"{rv}_then")
                    if f"{rv}_else" not in defined_vars:
                        # Try to find else_var in operations
                        if else_var in defined_vars:
                            lines.append(f"    {rv}_else = {else_var}")
                        else:
                            lines.append(f"    {rv}_else = None  # Fallback initialization")
                        defined_vars.add(f"{rv}_else")
                    
                    # Create the final if result - only if not already defined
                    if rv not in defined_vars:
                        lines.append(f"    {rv} = {rv}_then if {cond_var} else {rv}_else")
                        defined_vars.add(rv)
            else:
                # Fallback: look for variables with "then" or "else" in name
                then_var = None
                else_var = None
                for op in all_operations:
                    result_name = op.get("result", "")
                    if "then" in result_name.lower() and then_var is None:
                        then_var = result_name
                    if "else" in result_name.lower() and else_var is None:
                        else_var = result_name
                
                if then_var and then_var in defined_vars:
                    if f"{rv}_then" not in defined_vars:
                        lines.append(f"    {rv}_then = {then_var}")
                        defined_vars.add(f"{rv}_then")
                if else_var and else_var in defined_vars:
                    if f"{rv}_else" not in defined_vars:
                        lines.append(f"    {rv}_else = {else_var}")
                        defined_vars.add(f"{rv}_else")
                
                # Ensure both branches are initialized
                if f"{rv}_then" not in defined_vars:
                    lines.append(f"    {rv}_then = None  # Fallback initialization")
                    defined_vars.add(f"{rv}_then")
                if f"{rv}_else" not in defined_vars:
                    lines.append(f"    {rv}_else = None  # Fallback initialization")
                    defined_vars.add(f"{rv}_else")
                
                # Create the final if result
                lines.append(f"    {rv} = {rv}_then if {cond_var} else {rv}_else")
                defined_vars.add(rv)
        
        # Handle return statement
        return_match = re.search(r"statement\.return(.*?)(?:\n|$)", ir_text)
        return_vars = []
        if return_match:
            return_values = return_match.group(1).strip()
            if return_values:
                # Extract variable names from return
                return_vars = re.findall(r"%(\w+)", return_values)
                if len(return_vars) == 1:
                    lines.append(f"    return {return_vars[0]}")
                elif len(return_vars) > 1:
                    lines.append(f"    return ({', '.join(return_vars)})")
            else:
                lines.append("    return")
        else:
            lines.append("    return")
        
        # Add test function
        lines.append("")
        lines.append("")
        lines.append("def test_" + func_name + "():")
        lines.append('    """Test function for ' + func_name + ' with PyTorch golden data comparison."""')
        lines.append("    import torch")
        lines.append(f"    print(f\"Testing {func_name}...\")")
        lines.append("    ")
        lines.append("    # Set random seed for reproducibility")
        lines.append("    np.random.seed(42)")
        lines.append("    torch.manual_seed(42)")
        lines.append("    ")
        
        # Generate test inputs based on parameters
        test_inputs = []
        output_params = []
        torch_inputs = []
        
        for param_name, param_type, direction in param_list:
            shape = self._extract_shape_for_test(param_type)
            dtype = self._extract_dtype(param_type)
            numpy_dtype = self.dtype_map.get(dtype, "np.float32")
            torch_dtype = "torch.float32" if dtype == "fp32" else "torch.float64" if dtype == "fp64" else "torch.int32" if dtype == "int32" else "torch.int64"
            
            if direction == "out":
                # Output parameter - create empty array
                lines.append(f"    {param_name} = np.zeros({shape}, dtype={numpy_dtype})")
                lines.append(f"    {param_name}_torch = torch.zeros({shape}, dtype={torch_dtype})")
                test_inputs.append(param_name)
                output_params.append(param_name)
            else:
                # Input parameter - create test data
                # Check if it's a tensor/tile type (contains 'tensor' or 'tile' keyword)
                is_tensor_or_tile = "tensor" in param_type or "tile" in param_type
                if is_tensor_or_tile:
                    lines.append(f"    # Create NumPy input")
                    lines.append(f"    {param_name} = np.random.randn(*{shape}).astype({numpy_dtype})")
                    lines.append(f"    # Create PyTorch golden input (same data)")
                    lines.append(f"    {param_name}_torch = torch.from_numpy({param_name}.copy())")
                    torch_inputs.append(f"{param_name}_torch")
                else:
                    # Scalar
                    lines.append(f"    {param_name} = {numpy_dtype}(2.5)")
                    lines.append(f"    {param_name}_torch = torch.tensor(2.5, dtype={torch_dtype})")
                    torch_inputs.append(f"{param_name}_torch")
                test_inputs.append(param_name)
        
        # Initialize accumulator variables in test function
        # Note: Accumulators are initialized from parameters in the function itself
        # The function modifies the parameter in place, so we don't need separate accumulator variables
        # But we do need to initialize the PyTorch golden accumulator from the parameter
        for loop_info in for_loop_vars.values():
            for acc, init in loop_info.get("iter_args", {}).items():
                if init in [p[0] for p in param_list]:
                    # The accumulator will be initialized from the parameter in the PyTorch golden computation
                    # No need to create a separate variable here
                    pass
        
        lines.append("    ")
        lines.append("    # Generate PyTorch golden output by replicating IR operations")
        lines.append("    with torch.no_grad():")
        lines.append("        # Replicate operations in PyTorch")
        
        # Find modified parameters (outputs) - check for assemble operations
        modified_params = []
        for op in all_operations:
            if op["op"] in ["tensor.assemble", "tile.assemble"]:
                if len(op["operands"]) >= 2:
                    dst_var = op["operands"][1].replace("%", "")
                    if dst_var not in modified_params:
                        modified_params.append(dst_var)
        
        # Generate PyTorch golden computation from stored operations
        # Convert operations to PyTorch equivalents
        torch_vars = {}  # Track PyTorch variable names
        constants_dict = {}  # Track constants for PyTorch computation
        
        # Handle for loops in PyTorch golden computation
        # First, identify which operations are inside for loops
        for loop_var, loop_info in for_loop_vars.items():
            if loop_var in for_loop_blocks:
                # Initialize accumulator for PyTorch
                # Use the init parameter directly since the function modifies it in place
                for acc, init in loop_info.get("iter_args", {}).items():
                    if init in [p[0] for p in param_list]:
                        init_torch = f"{init}_torch"
                        # Use the parameter name directly for the accumulator in PyTorch golden computation
                        # This matches what the NumPy function does (modifies the parameter in place)
                        lines.append(f"        # Initialize accumulator {acc} for loop {loop_var} (using {init})")
                        lines.append(f"        {init_torch}_acc = {init_torch}.clone()")
                        torch_vars[acc] = f"{init_torch}_acc"
                
                # Generate for loop in PyTorch
                lines.append(f"        # For loop: {loop_var} from {loop_info['start']} to {loop_info['end']} step {loop_info['step']}")
                lines.append(f"        for {loop_var} in range(4):  # TODO: Use actual bounds")
                
                # Process operations inside the loop
                # First, collect all operations to determine when to create tuple access variables
                loop_ops_for_torch = []
                for op_block in for_loop_blocks[loop_var]:
                    operations = self._parse_operations(op_block)
                    loop_ops_for_torch.extend(operations)
                
                # Track which yield variables have been assigned
                yield_vars_assigned = set()
                # Track which tuple access variables have been created
                tuple_access_created = set()
                
                # Process operations inside the loop
                for op in loop_ops_for_torch:
                    result_var = op["result"]
                    op_name = op["op"]
                    operands = op.get("operands", [])
                    
                    if op_name == "const":
                        value = op["value"]
                        constants_dict[result_var] = value
                        torch_vars[result_var] = value
                        lines.append(f"            {result_var}_torch = {value}")
                    elif op_name == "tensor.view":
                        if operands and len(operands) > 0:
                            input_var = operands[0].replace("%", "")
                            input_torch = torch_vars.get(input_var, f"{input_var}_torch")
                            op_attrs = op.get("attrs", {})
                            offset = op_attrs.get("offset", None)
                            if offset and len(offset) >= 2:
                                offset_0 = offset[0].strip()
                                if offset_0.startswith("%"):
                                    offset_0 = offset_0[1:]
                                lines.append(f"            {result_var}_torch = {input_torch}[{offset_0}:{offset_0}+1, :]  # view with offset")
                            else:
                                lines.append(f"            {result_var}_torch = {input_torch}[0:1, :]")
                            torch_vars[result_var] = f"{result_var}_torch"
                    elif op_name in ["tensor.mul", "tile.OP_MUL"]:
                        # Track yield variables
                        if result_var in [v for rv, yv_data in yield_mapping.items() if isinstance(yv_data, tuple) and isinstance(yv_data[0], list) for v in (yv_data[0] + yv_data[1])]:
                            yield_vars_assigned.add(result_var)
                        # Map to PyTorch
                        if len(operands) >= 2:
                            op1 = operands[0].replace("%", "")
                            op2 = operands[1].replace("%", "")
                            op1_torch = torch_vars.get(op1, f"{op1}_torch")
                            op2_torch = torch_vars.get(op2, f"{op2}_torch")
                            lines.append(f"            {result_var}_torch = {op1_torch} * {op2_torch}")
                            torch_vars[result_var] = f"{result_var}_torch"
                    elif op_name == "tensor.assemble" or op_name == "tile.assemble":
                        if len(operands) >= 2:
                                src_var_raw = operands[0].replace("%", "")
                                # Handle tuple access like r0#0 -> r0_val_0
                                if "#" in src_var_raw:
                                    base_var, index = src_var_raw.split("#")
                                    src_var = f"{base_var}_val_{index}"
                                    # Check if we need to create tuple access variables
                                    if base_var not in tuple_access_created and base_var in if_results and base_var in yield_mapping:
                                        yield_data = yield_mapping[base_var]
                                        if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                                            then_vars, else_vars = yield_data
                                            # Check if all yield variables are assigned
                                            all_then_assigned = all(v in yield_vars_assigned for v in then_vars)
                                            all_else_assigned = all(v in yield_vars_assigned for v in else_vars)
                                            if all_then_assigned and all_else_assigned:
                                                # Create tuple access variables now, before this operation uses them
                                                cond_var = if_results[base_var]
                                                for idx in range(len(then_vars)):
                                                    then_var = then_vars[idx]
                                                    else_var = else_vars[idx]
                                                    then_torch = torch_vars.get(then_var, f"{then_var}_torch")
                                                    else_torch = torch_vars.get(else_var, f"{else_var}_torch")
                                                    # Insert before the current operation
                                                    lines.append(f"            # If statement result {base_var} tuple access {idx}")
                                                    lines.append(f"            {base_var}_val_{idx}_torch = {then_torch} if {cond_var} else {else_torch}")
                                                    torch_vars[f"{base_var}_val_{idx}"] = f"{base_var}_val_{idx}_torch"
                                                tuple_access_created.add(base_var)
                                else:
                                    src_var = src_var_raw
                                dst_var = operands[1].replace("%", "")
                                src_torch = torch_vars.get(src_var, f"{src_var}_torch")
                                # Check if dst_var is an accumulator - if so, use the init parameter's torch variable
                                if dst_var in iter_args:
                                    init_param = iter_args[dst_var]
                                    if init_param in [p[0] for p in param_list]:
                                        dst_torch = f"{init_param}_torch_acc"
                                    else:
                                        dst_torch = torch_vars.get(dst_var, f"{dst_var}_torch")
                                else:
                                    dst_torch = torch_vars.get(dst_var, f"{dst_var}_torch")
                                op_attrs = op.get("attrs", {})
                                offset = op_attrs.get("offset", None)
                                
                                if offset and len(offset) >= 2:
                                    offset_0 = offset[0].strip()
                                    offset_1 = offset[1].strip()
                                    if offset_0.startswith("%"):
                                        offset_0 = offset_0[1:]
                                    if offset_1.startswith("%"):
                                        offset_1 = offset_1[1:]
                                    if (offset_0 == "0" or offset_0 == "const_0") and (offset_1 == "0" or offset_1 == "const_0"):
                                        lines.append(f"            {dst_torch}[:] = {src_torch}[:]")
                                    else:
                                        lines.append(f"            {dst_torch}[{offset_0}:{offset_0}+1, :] = {src_torch}")
                                else:
                                    lines.append(f"            {dst_torch}[:] = {src_torch}[:]")
                                # Update torch_vars with the correct variable name
                                if dst_var in iter_args:
                                    # This is an accumulator - use the init parameter's torch variable
                                    init_param = iter_args[dst_var]
                                    if init_param in [p[0] for p in param_list]:
                                        torch_vars[dst_var] = f"{init_param}_torch_acc"
                                    else:
                                        torch_vars[dst_var] = dst_torch
                                else:
                                    torch_vars[dst_var] = dst_torch
                    else:
                        # Map operation to PyTorch
                        torch_op = self.torch_op_map.get(op_name, None)
                        if torch_op and operands:
                            clean_operands = []
                            for operand in operands:
                                operand = operand.strip().replace("%", "")
                                if operand.replace(".", "").replace("-", "").isdigit():
                                    clean_operands.append(operand)
                                elif operand in torch_vars:
                                    clean_operands.append(torch_vars[operand])
                                elif operand in [p[0] for p in param_list]:
                                    clean_operands.append(f"{operand}_torch")
                                else:
                                    if operand in constants_dict:
                                        clean_operands.append(constants_dict[operand])
                                    elif operand in constants:
                                        clean_operands.append(constants[operand])
                                    else:
                                        clean_operands.append(f"{operand}_torch" if f"{operand}_torch" in torch_vars else operand)
                            
                            if len(clean_operands) == 2:
                                lines.append(f"            {result_var}_torch = {clean_operands[0]} {torch_op} {clean_operands[1]}")
                                torch_vars[result_var] = f"{result_var}_torch"
                            elif len(clean_operands) == 1:
                                if torch_op.startswith("torch."):
                                    lines.append(f"            {result_var}_torch = {torch_op}({clean_operands[0]})")
                                else:
                                    lines.append(f"            {result_var}_torch = {torch_op}{clean_operands[0]}")
                                torch_vars[result_var] = f"{result_var}_torch"
                
                # Handle yield - update accumulator
                yield_vars = loop_info.get("yield_vars", [])
                if yield_vars:
                    iter_args = loop_info.get("iter_args", {})
                    # Track which operations result in accumulator updates
                    acc_result_vars = {}  # Map result_var -> acc_var for assemble operations
                    for op_block in for_loop_blocks[loop_var]:
                        operations = self._parse_operations(op_block)
                        for op in operations:
                            if op["op"] in ["tensor.assemble", "tile.assemble"]:
                                if len(op.get("operands", [])) >= 2:
                                    dst_var = op["operands"][1].replace("%", "")
                                    result_var = op["result"]
                                    if dst_var in iter_args:
                                        acc_result_vars[result_var] = dst_var
                    
                    for i, yield_var in enumerate(yield_vars):
                        acc_list = list(iter_args.keys())
                        if i < len(acc_list):
                            acc = acc_list[i]
                            init_param = iter_args[acc]
                            # Check if yield_var is the result of an assemble operation on this accumulator
                            if yield_var in acc_result_vars and acc_result_vars[yield_var] == acc:
                                # The assemble operation already modified the accumulator in place, no need to assign
                                pass
                            elif yield_var != acc:
                                # Only assign if yield_var is different and exists in torch_vars
                                yield_torch = torch_vars.get(yield_var, None)
                                if yield_torch is None:
                                    # Try to get it from the accumulator if it's an assemble result
                                    if yield_var in acc_result_vars:
                                        acc_var = acc_result_vars[yield_var]
                                        init_for_acc = iter_args.get(acc_var, acc_var)
                                        if init_for_acc in [p[0] for p in param_list]:
                                            yield_torch = f"{init_for_acc}_torch_acc"
                                        else:
                                            yield_torch = torch_vars.get(acc_var, f"{acc_var}_torch")
                                    else:
                                        yield_torch = f"{yield_var}_torch"
                                # Use the init parameter's torch variable for the accumulator
                                if init_param in [p[0] for p in param_list]:
                                    lines.append(f"            {init_param}_torch_acc = {yield_torch}  # Update accumulator from yield")
                                    torch_vars[acc] = f"{init_param}_torch_acc"
                                else:
                                    lines.append(f"            {acc}_torch = {yield_torch}  # Update accumulator from yield")
                                    torch_vars[acc] = f"{acc}_torch"
                
                # Assign loop result variable(s) after the loop in PyTorch golden computation
                if loop_var in loop_result_vars and loop_result_vars[loop_var]:
                    result_vars = loop_result_vars[loop_var]
                    iter_args = loop_info.get("iter_args", {})
                    acc_list = list(iter_args.keys())
                    for i, result_var in enumerate(result_vars):
                        if i < len(acc_list):
                            acc = acc_list[i]
                            init_param = iter_args[acc]
                            # The accumulator was modified in place, so assign it to the result variable
                            if init_param in [p[0] for p in param_list]:
                                lines.append(f"        # Assign loop result {result_var} from accumulator {acc} (using {init_param})")
                                lines.append(f"        {result_var}_torch = {init_param}_torch_acc")
                            else:
                                lines.append(f"        # Assign loop result {result_var} from accumulator {acc}")
                                lines.append(f"        {result_var}_torch = {acc}_torch")
                            torch_vars[result_var] = f"{result_var}_torch"
        
        # Process operations outside for loops
        for op in all_operations:
            # Skip if this operation is inside a for loop
            is_in_loop = False
            for loop_var, loop_blocks in for_loop_blocks.items():
                # Check if this operation is in any of the loop blocks
                for op_block in loop_blocks:
                    operations_in_block = self._parse_operations(op_block)
                    if op in operations_in_block:
                        is_in_loop = True
                        break
                if is_in_loop:
                    break
            if is_in_loop:
                continue
            result_var = op["result"]
            op_name = op["op"]
            operands = op.get("operands", [])
            
            if op_name == "const":
                value = op["value"]
                constants_dict[result_var] = value
                torch_vars[result_var] = value
                lines.append(f"        {result_var}_torch = {value}")
            elif op_name == "tensor.view":
                # View operation - slice in PyTorch
                if operands and len(operands) > 0:
                    input_var = operands[0].replace("%", "")
                    input_torch = f"{input_var}_torch" if input_var in [p[0] for p in param_list] else input_var
                    # Check for offset attribute
                    op_attrs = op.get("attrs", {})
                    offset = op_attrs.get("offset", None)
                    if offset and len(offset) >= 2:
                        offset_0 = offset[0].strip()
                        if offset_0.startswith("%"):
                            offset_0 = offset_0[1:]  # Remove % prefix
                        lines.append(f"        {result_var}_torch = {input_torch}[{offset_0}:{offset_0}+1, :]  # view with offset")
                    else:
                        lines.append(f"        {result_var}_torch = {input_torch}[0:1, :]")
                    torch_vars[result_var] = f"{result_var}_torch"
            elif op_name == "tensor.assemble" or op_name == "tile.assemble":
                # Assemble - write to output
                if len(operands) >= 2:
                    src_var_raw = operands[0].replace("%", "")
                    # Handle tuple access like r0#0 -> r0_val_0
                    if "#" in src_var_raw:
                        base_var, index = src_var_raw.split("#")
                        src_var = f"{base_var}_val_{index}"
                    else:
                        src_var = src_var_raw
                    dst_var = operands[1].replace("%", "")
                    # Check if src_var is a loop result variable - if so, use the accumulator's torch variable
                    is_loop_result = False
                    src_torch = None
                    for loop_var, result_vars in loop_result_vars.items():
                        if src_var in result_vars:
                            is_loop_result = True
                            loop_info = for_loop_vars[loop_var]
                            iter_args = loop_info.get("iter_args", {})
                            acc_list = list(iter_args.keys())
                            result_idx = result_vars.index(src_var)
                            if result_idx < len(acc_list):
                                acc = acc_list[result_idx]
                                init_param = iter_args[acc]
                                if init_param in [p[0] for p in param_list]:
                                    # Use the accumulator's torch variable directly
                                    src_torch = f"{init_param}_torch_acc"
                                else:
                                    src_torch = f"{acc}_torch"
                            break
                    # Check if src_var is an if result that needs to be defined first
                    if not is_loop_result and src_var in if_results and src_var not in torch_vars:
                        # Define the if result before using it in assemble
                        rv = src_var
                        cond_var = if_results[rv]
                        if rv in yield_mapping:
                            yield_data = yield_mapping[rv]
                            if isinstance(yield_data, tuple) and not isinstance(yield_data[0], list):
                                then_var, else_var = yield_data
                                then_torch = torch_vars.get(then_var, f"{then_var}_torch")
                                else_torch = torch_vars.get(else_var, f"{else_var}_torch")
                                lines.append(f"        # If statement result {rv} (needed for assemble)")
                                lines.append(f"        {rv}_torch = {then_torch} if {cond_var} else {else_torch}")
                                torch_vars[rv] = f"{rv}_torch"
                    if src_torch is None:
                        src_torch = torch_vars.get(src_var, f"{src_var}_torch")
                    dst_torch = f"{dst_var}_torch"
                    # Extract offset from operation attributes if available
                    op_attrs = op.get("attrs", {})
                    offset = op_attrs.get("offset", None)
                    
                    # If offset is [0, 0] or not specified, copy entire tensor
                    if offset and len(offset) >= 2:
                        offset_0 = offset[0].strip()
                        offset_1 = offset[1].strip()
                        # Remove % prefix if present
                        if offset_0.startswith("%"):
                            offset_0 = offset_0[1:]
                        if offset_1.startswith("%"):
                            offset_1 = offset_1[1:]
                        # If offset is [0, 0] or symbolic 0, copy entire tensor
                        if (offset_0 == "0" or offset_0 == "const_0") and (offset_1 == "0" or offset_1 == "const_0"):
                            lines.append(f"        # Assemble {src_var} into {dst_var} at offset {offset} (full copy)")
                            lines.append(f"        {dst_torch}[:] = {src_torch}[:]")
                        else:
                            # Use slice assignment with offset (e.g., offset=[%i, 0] means [i:i+1, :])
                            lines.append(f"        # Assemble {src_var} into {dst_var} at offset {offset}")
                            lines.append(f"        {dst_torch}[{offset_0}:{offset_0}+1, :] = {src_torch}")
                    else:
                        # No offset specified, copy entire tensor
                        lines.append(f"        # Assemble {src_var} into {dst_var} (full copy)")
                        lines.append(f"        {dst_torch}[:] = {src_torch}[:]")
                    torch_vars[dst_var] = dst_torch
            else:
                # Map operation to PyTorch
                torch_op = self.torch_op_map.get(op_name, None)
                if torch_op and operands:
                    clean_operands = []
                    for operand in operands:
                        operand = operand.strip().replace("%", "")
                        if operand.replace(".", "").replace("-", "").isdigit():
                            clean_operands.append(operand)
                        elif operand in torch_vars:
                            clean_operands.append(torch_vars[operand])
                        elif operand in [p[0] for p in param_list]:
                            clean_operands.append(f"{operand}_torch")
                        else:
                            # Check if it's a constant
                            if operand in constants_dict:
                                clean_operands.append(constants_dict[operand])
                            elif operand in constants:
                                clean_operands.append(constants[operand])
                            else:
                                clean_operands.append(f"{operand}_torch" if f"{operand}_torch" in torch_vars else operand)
                    
                    if len(clean_operands) == 2:
                        lines.append(f"        {result_var}_torch = {clean_operands[0]} {torch_op} {clean_operands[1]}")
                        torch_vars[result_var] = f"{result_var}_torch"
                    elif len(clean_operands) == 1:
                        # Unary operation
                        if torch_op.startswith("torch."):
                            lines.append(f"        {result_var}_torch = {torch_op}({clean_operands[0]})")
                        elif torch_op == "-":
                            # Negation operator
                            lines.append(f"        {result_var}_torch = -{clean_operands[0]}")
                        else:
                            lines.append(f"        {result_var}_torch = {torch_op}{clean_operands[0]}")
                        torch_vars[result_var] = f"{result_var}_torch"
        
        # Handle if statement results for PyTorch golden computation (outside loops)
        # Tuple results inside loops are handled within the loop processing above
        
        # Handle if statement results for PyTorch golden computation (outside loops)
        for rv in if_results.keys():
            cond_var = if_results[rv]
            if rv in yield_mapping:
                yield_data = yield_mapping[rv]
                if isinstance(yield_data, tuple) and not isinstance(yield_data[0], list):
                    # Single yield value
                    then_var, else_var = yield_data
                    then_torch = torch_vars.get(then_var, f"{then_var}_torch")
                    else_torch = torch_vars.get(else_var, f"{else_var}_torch")
                    lines.append(f"        # If statement result {rv}")
                    lines.append(f"        {rv}_torch = {then_torch} if {cond_var} else {else_torch}")
                    torch_vars[rv] = f"{rv}_torch"
        
        lines.append("    ")
        
        lines.append("    # Call the NumPy function")
        if return_vars:
            if len(return_vars) == 1:
                lines.append(f"    result = {func_name}({', '.join(test_inputs)})")
                lines.append("    ")
                lines.append("    # Get PyTorch golden result")
                return_var_name = return_vars[0]
                # Find the torch variable for the return value
                golden_var = torch_vars.get(return_var_name, f"{return_var_name}_torch")
                lines.append(f"    result_torch_golden = {golden_var}")
                lines.append("    ")
                lines.append("    # Print test value and golden value")
                if "tensor" in str(return_type).lower() or "tile" in str(return_type).lower():
                    lines.append("    result_torch = torch.from_numpy(result.copy())")
                    lines.append("    print(f\"\\nTest Result (NumPy):\")")
                    lines.append("    print(f\"  Shape: {result.shape}\")")
                    lines.append("    print(f\"  Sum: {result.sum()}\")")
                    lines.append("    print(f\"  Mean: {result.mean()}\")")
                    lines.append("    print(f\"  Min: {result.min()}, Max: {result.max()}\")")
                    lines.append("    print(f\"  Sample values: {result.flatten()[:5].tolist()}\")")
                    lines.append("    print(f\"\\nGolden Result (PyTorch):\")")
                    lines.append("    print(f\"  Shape: {result_torch_golden.shape}\")")
                    lines.append("    print(f\"  Sum: {result_torch_golden.sum().item()}\")")
                    lines.append("    print(f\"  Mean: {result_torch_golden.mean().item()}\")")
                    lines.append("    print(f\"  Min: {result_torch_golden.min().item()}, Max: {result_torch_golden.max().item()}\")")
                    lines.append("    print(f\"  Sample values: {result_torch_golden.flatten()[:5].tolist()}\")")
                    lines.append("    ")
                    lines.append("    # Compare with PyTorch golden")
                    lines.append("    if torch.allclose(result_torch, result_torch_golden, rtol=1e-5, atol=1e-6):")
                    lines.append("        print(f\"✓ Result matches PyTorch golden (within tolerance)\")")
                    lines.append("    else:")
                    lines.append("        max_diff = torch.abs(result_torch - result_torch_golden).max().item()")
                    lines.append("        print(f\"✗ Result differs from PyTorch golden (max diff: {max_diff})\")")
                    lines.append("        assert False, f\"Result does not match PyTorch golden (max diff: {max_diff})\"")
                else:
                    # Scalar return - check if golden is a tensor or scalar
                    lines.append("    print(f\"\\nTest Result (NumPy): {result}\")")
                    lines.append("    # Handle scalar return - check if golden is tensor or scalar")
                    lines.append("    if isinstance(result_torch_golden, torch.Tensor):")
                    lines.append("        golden_scalar = result_torch_golden.item()")
                    lines.append("    else:")
                    lines.append("        golden_scalar = result_torch_golden")
                    lines.append("    print(f\"Golden Result (PyTorch): {golden_scalar}\")")
                    lines.append("    ")
                    lines.append("    # Compare with PyTorch golden")
                    lines.append("    if abs(result - golden_scalar) < 1e-5:")
                    lines.append("        print(f\"✓ Result matches PyTorch golden (within tolerance)\")")
                    lines.append("    else:")
                    lines.append("        diff = abs(result - golden_scalar)")
                    lines.append("        print(f\"✗ Result differs from PyTorch golden (diff: {diff})\")")
                    lines.append("        assert False, f\"Result does not match PyTorch golden (diff: {diff})\"")
            else:
                lines.append(f"    results = {func_name}({', '.join(test_inputs)})")
                lines.append("    print(f\"Results: {results}\")")
                for i, var in enumerate(return_vars):
                    golden_var = torch_vars.get(var, f"{var}_torch")
                    lines.append(f"    result_{i}_torch_golden = {golden_var}")
                    lines.append(f"    print(f\"Result {i} ({var}): {{results[{i}]}}\")")
        else:
            lines.append(f"    {func_name}({', '.join(test_inputs)})")
            lines.append("    ")
            
            # Check output parameters and compare with PyTorch
            # Also include accumulator variables from for loops
            params_to_check = output_params if output_params else modified_params.copy()
            # Add accumulator variables from for loops - use the init parameter if it exists
            # Only add the init parameter, not the accumulator name itself, since the function modifies the parameter in place
            for loop_info in for_loop_vars.values():
                for acc, init in loop_info.get("iter_args", {}).items():
                    # Use the init parameter (which is the accumulator's initial value) for checking
                    # since the function modifies it in place
                    # Only add if it's a parameter (not a local variable)
                    if init in [p[0] for p in param_list]:
                        if init not in params_to_check:
                            params_to_check.append(init)
                    # Don't add acc itself - it's not a parameter, it's a local variable in the function
                    # Also remove acc from params_to_check if it was accidentally added
                    if acc in params_to_check:
                        params_to_check.remove(acc)
                    # Also remove acc from params_to_check if it was accidentally added
                    if acc in params_to_check:
                        params_to_check.remove(acc)
            
            if params_to_check:
                for out_param in params_to_check:
                    lines.append(f"    # Check {out_param} against PyTorch golden")
                    lines.append(f"    print(f\"\\nChecking {out_param}:\")")
                    lines.append(f"    ")
                    lines.append(f"    # Print test value (NumPy result)")
                    lines.append(f"    print(f\"Test Result (NumPy):\")")
                    lines.append(f"    print(f\"  Shape: {out_param}.shape\")")
                    lines.append(f"    sum_val = float({out_param}.sum())")
                    lines.append(f"    mean_val = float({out_param}.mean())")
                    lines.append(f"    min_val = float({out_param}.min())")
                    lines.append(f"    max_val = float({out_param}.max())")
                    lines.append(f"    print(f\"  Sum: {{sum_val}}\")")
                    lines.append(f"    print(f\"  Mean: {{mean_val}}\")")
                    lines.append(f"    print(f\"  Min: {{min_val}}, Max: {{max_val}}\")")
                    lines.append(f"    print(f\"  Sample values: {{{out_param}.flatten()[:5].tolist()}}\")")
                    lines.append(f"    ")
                    lines.append(f"    # Convert NumPy result to PyTorch for comparison")
                    lines.append(f"    {out_param}_pt_result = torch.from_numpy({out_param}.copy())")
                    lines.append(f"    ")
                    lines.append(f"    # Get PyTorch golden output (already computed above)")
                    # For accumulator variables, check if we used _acc suffix in PyTorch golden computation
                    # Check if this parameter is used as an accumulator initial value
                    is_acc_init = False
                    acc_var_name = None
                    for loop_info in for_loop_vars.values():
                        for acc, init in loop_info.get("iter_args", {}).items():
                            if init == out_param:
                                is_acc_init = True
                                acc_var_name = acc
                                break
                        if is_acc_init:
                            break
                    if is_acc_init and acc_var_name:
                        # Use the accumulator's torch variable
                        lines.append(f"    {out_param}_pt_golden = {out_param}_torch_acc")
                    else:
                        lines.append(f"    {out_param}_pt_golden = {out_param}_torch")
                    lines.append(f"    ")
                    lines.append(f"    # Print golden value (PyTorch result)")
                    lines.append(f"    print(f\"Golden Result (PyTorch):\")")
                    lines.append(f"    print(f\"  Shape: {out_param}_pt_golden.shape\")")
                    lines.append(f"    print(f\"  Sum: {{float({out_param}_pt_golden.sum().item())}}\")")
                    lines.append(f"    print(f\"  Mean: {{float({out_param}_pt_golden.mean().item())}}\")")
                    lines.append(f"    print(f\"  Min: {{float({out_param}_pt_golden.min().item())}}, Max: {{float({out_param}_pt_golden.max().item())}}\")")
                    lines.append(f"    print(f\"  Sample values: {{torch.flatten({out_param}_pt_golden)[:5].tolist()}}\")")
                    lines.append(f"    ")
                    lines.append(f"    # Compare NumPy result with PyTorch golden")
                    lines.append(f"    if torch.allclose({out_param}_pt_result, {out_param}_pt_golden, rtol=1e-5, atol=1e-6):")
                    lines.append(f"        print(f\"  ✓ {out_param} matches PyTorch golden (within tolerance)\")")
                    lines.append(f"    else:")
                    lines.append(f"        max_diff = torch.abs({out_param}_pt_result - {out_param}_pt_golden).max().item()")
                    lines.append(f"        mean_diff = torch.abs({out_param}_pt_result - {out_param}_pt_golden).mean().item()")
                    lines.append(f"        print(f\"  ✗ {out_param} differs from PyTorch golden\")")
                    lines.append(f"        print(f\"    Max difference: {{max_diff}}\")")
                    lines.append(f"        print(f\"    Mean difference: {{mean_diff}}\")")
                    lines.append(f"        print(f\"    NumPy result sample: {{torch.flatten({out_param}_pt_result)[:5].tolist()}}\")")
                    lines.append(f"        print(f\"    PyTorch golden sample: {{torch.flatten({out_param}_pt_golden)[:5].tolist()}}\")")
                    lines.append(f"        # Assertion will fail if values don't match")
                    lines.append(f"        assert False, f\"{out_param} does not match PyTorch golden (max diff: {{max_diff}})\"")
                    lines.append("    ")
            else:
                lines.append("    print(\"No output parameters to check\")")
        
        lines.append("    ")
        lines.append("    print(\"✓ Test completed successfully!\")")
        lines.append("    return True")
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append(f"    test_{func_name}()")
        
        return "\n".join(lines)

    def _parse_parameters(self, params_str: str) -> List[Tuple[str, str, Optional[str]]]:
        """Parse function parameters."""
        params = []
        # Match: %name: type or %name: type #in/#out
        # Handle complex types like tensor<[%b_1, 128], fp32>
        # Split parameters by comma, but respect angle brackets
        param_parts = []
        current_part = ""
        depth = 0
        for char in params_str:
            if char == '<':
                depth += 1
                current_part += char
            elif char == '>':
                depth -= 1
                current_part += char
            elif char == ',' and depth == 0:
                param_parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        if current_part:
            param_parts.append(current_part.strip())
        
        # Parse each parameter
        for param_str in param_parts:
            param_str = param_str.strip()
            # Match: %name: type or %name: type #in/#out
            # The type can be complex like tensor<[%b_1, 128], fp32>
            # So we need to match everything after : until # or end
            param_pattern = r"%(\w+):\s*(.+?)(?:\s*#(\w+))?$"
            match = re.search(param_pattern, param_str)
            if match:
                name = match.group(1)
                param_type = match.group(2).strip()
                direction = match.group(3) if match.group(3) else None
                params.append((name, param_type, direction))
        return params

    def _parse_operations(self, op_block: str) -> List[Dict]:
        """Parse operations from an op block."""
        operations = []
        
        # First, match constant assignments: %name = value : type
        const_pattern = r"%(\w+)\s*=\s*([\d.]+)\s*:\s*(\w+)"
        for match in re.finditer(const_pattern, op_block):
            var_name = match.group(1)
            value = match.group(2)
            dtype = match.group(3)
            operations.append({
                "result": var_name,
                "op": "const",
                "value": value,
                "dtype": dtype,
            })
        
        # Then match operations like: %result = op %arg1, %arg2 : (type1, type2) -> result_type
        # Pattern: %result = op %arg1, %arg2 : (types) -> result_type {attrs}
        # Match each line separately to avoid issues with nested braces
        lines = op_block.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Match: %result = op %arg1, %arg2 : (types) -> result_type {attrs}
            # Pattern to handle: %var = op %arg1, %arg2 : (types) -> result_type {attrs}
            # First try to match with type annotation
            op_pattern = r"%(\w+)\s*=\s*(\S+)\s+([^:]+?)\s*:\s*\(([^)]+)\)\s*->\s*([^\s\{]+(?:\s+[^\s\{]+)*?)(?:\s*\{[^}]+\})?"
            match = re.search(op_pattern, line)
            if match:
                result_var = match.group(1)
                op_name = match.group(2)
                operands_str = match.group(3).strip()
                input_types = match.group(4)
                output_type = match.group(5).strip()
                
                # Parse operands - simple split by comma (operands shouldn't have nested commas)
                operand_list = [op.strip() for op in operands_str.split(',')]
                
                # Check if any operand is a literal value in the type annotation
                # For example: (tensor<[1, 128], fp32>, 3.14) means the second operand is 3.14
                # Parse type parts more carefully - split by comma but respect angle brackets
                type_parts = []
                current_part = ""
                depth = 0
                for char in input_types:
                    if char == '<':
                        depth += 1
                        current_part += char
                    elif char == '>':
                        depth -= 1
                        current_part += char
                    elif char == ',' and depth == 0:
                        type_parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                if current_part:
                    type_parts.append(current_part.strip())
                
                # Replace operands with literal values from type annotation
                for i, type_part in enumerate(type_parts):
                    # If type part is a number (literal value), use it instead of the operand
                    if i < len(operand_list):
                        cleaned_type = type_part.strip()
                        # Check if it's a number (including decimals and negatives)
                        if cleaned_type.replace(".", "").replace("-", "").isdigit():
                            operand_list[i] = cleaned_type
                
                # Extract attributes from the operation line (e.g., {offset=[0, 0]})
                attrs = {}
                attrs_match = re.search(r'\{\s*([^}]+)\s*\}', line)
                if attrs_match:
                    attrs_str = attrs_match.group(1)
                    # Parse offset attribute: offset=[0, 0]
                    offset_match = re.search(r'offset\s*=\s*\[([^\]]+)\]', attrs_str)
                    if offset_match:
                        offset_str = offset_match.group(1)
                        offset_values = [v.strip() for v in offset_str.split(',')]
                        attrs["offset"] = offset_values
                
                operations.append({
                    "result": result_var,
                    "op": op_name,
                    "operands": operand_list,
                    "input_types": input_types,
                    "output_type": output_type,
                    "attrs": attrs,
                })
        
        return operations

    def _convert_operation(self, op: Dict, constants: Dict, defined_vars: set = None, for_loop_vars: Dict = None, if_results: Dict = None, param_list: List = None, yield_mapping: Dict = None, loop_result_vars: Dict = None) -> Optional[str]:
        """Convert an IR operation to NumPy code."""
        if defined_vars is None:
            defined_vars = set()
        if for_loop_vars is None:
            for_loop_vars = {}
        if if_results is None:
            if_results = {}
        if param_list is None:
            param_list = []
        if yield_mapping is None:
            yield_mapping = {}
        if loop_result_vars is None:
            loop_result_vars = {}
        
        result_var = op["result"]
        
        if op["op"] == "const":
            value = op["value"]
            dtype = op["dtype"]
            constants[result_var] = value
            if "." in value:
                return f"{result_var} = {value}  # {dtype}"
            else:
                return f"{result_var} = {value}  # {dtype}"
        
        op_name = op["op"]
        operands = op.get("operands", [])
        
        # Clean operand names (remove % prefix) and handle constants
        clean_operands = []
        for operand in operands:
            operand = operand.strip()
            # Check if it's a literal number (like 3.14) - keep as is
            if operand.replace(".", "").replace("-", "").isdigit():
                clean_operands.append(operand)
            elif operand.startswith("%"):
                # Remove % prefix
                var_name = operand[1:]
                # Handle tuple access like r0#0, r0#1
                if "#" in var_name:
                    base_var, index = var_name.split("#")
                    # For if statement results with tuple access
                    if base_var in if_results:
                        # Use the indexed access variable that will be created
                        clean_operands.append(f"{base_var}_val_{index}")
                    else:
                        # This might be a tuple from a multi-return - use the base variable
                        clean_operands.append(base_var)
                # Check if it's a constant we've seen
                elif var_name in constants:
                    clean_operands.append(constants[var_name])
                # Check if it's a loop result variable - if so, it should reference the accumulator
                elif var_name in [rv for loop_var, result_vars in loop_result_vars.items() for rv in result_vars]:
                    # Find which accumulator this loop result variable corresponds to
                    for loop_var, result_vars in loop_result_vars.items():
                        if var_name in result_vars:
                            loop_info = for_loop_vars[loop_var]
                            iter_args = loop_info.get("iter_args", {})
                            acc_list = list(iter_args.keys())
                            result_idx = result_vars.index(var_name)
                            if result_idx < len(acc_list):
                                acc = acc_list[result_idx]
                                # Use the accumulator variable name
                                clean_operands.append(acc)
                            else:
                                clean_operands.append(var_name)
                            break
                    else:
                        clean_operands.append(var_name)
                # Check if it's a loop accumulator variable
                elif var_name in [acc for loop_info in for_loop_vars.values() for acc in loop_info.get("iter_args", {}).keys()]:
                    clean_operands.append(var_name)
                else:
                    clean_operands.append(var_name)
            else:
                clean_operands.append(operand)
        
        # Handle view operation
        if op_name == "tensor.view":
            # Extract view - check for offset attribute
            attrs = op.get("attrs", {})
            offset = attrs.get("offset", None)
            if len(clean_operands) >= 1:
                if offset and len(offset) >= 2:
                    # Use the offset for slicing (e.g., offset=[%i, 0] means [i:i+1, :])
                    offset_0 = offset[0].strip()
                    # Remove % prefix if present
                    if offset_0.startswith("%"):
                        offset_0 = offset_0[1:]  # Remove % prefix
                    return f"{result_var} = {clean_operands[0]}[{offset_0}:{offset_0}+1, :]  # view operation with offset"
                else:
                    return f"{result_var} = {clean_operands[0]}[0:1, :]  # view operation"
            return f"{result_var} = {clean_operands[0]}  # view operation"
        
        # Handle assemble operation
        if op_name == "tensor.assemble" or op_name == "tile.assemble":
            # Assemble - write result to output tensor
            if len(clean_operands) >= 2:
                src_var = clean_operands[0]
                dst_var = clean_operands[1]
                # Check if src_var is an if result that hasn't been defined yet
                # If so, we need to define it first
                if src_var in if_results and src_var not in defined_vars:
                    # Define the if result before using it
                    cond_var = if_results[src_var]
                    if src_var in yield_mapping:
                        yield_data = yield_mapping[src_var]
                        if isinstance(yield_data, tuple) and not isinstance(yield_data[0], list):
                            then_var, else_var = yield_data
                            # Ensure both branches are defined
                            if f"{src_var}_then" not in defined_vars and then_var in defined_vars:
                                return None  # Will be handled by if result finalization
                            if f"{src_var}_else" not in defined_vars and else_var in defined_vars:
                                return None  # Will be handled by if result finalization
                            # If both branches are ready, define the if result
                            if f"{src_var}_then" in defined_vars and f"{src_var}_else" in defined_vars:
                                # Already defined, continue
                                pass
                            else:
                                return None  # Will be handled by if result finalization
                # Check offset attribute to determine how to copy
                attrs = op.get("attrs", {})
                offset = attrs.get("offset", None)
                
                # If offset is [0, 0] or not specified, and shapes match, copy entire tensor
                # Otherwise, use slice assignment
                if offset and len(offset) >= 2:
                    offset_0 = offset[0].strip()
                    offset_1 = offset[1].strip()
                    # If offset is [0, 0] or symbolic 0, copy entire tensor
                    if (offset_0 == "0" or offset_0.startswith("%const_0")) and (offset_1 == "0" or offset_1.startswith("%const_0")):
                        # Copy entire tensor
                        if dst_var in [acc for loop_info in for_loop_vars.values() for acc in loop_info.get("iter_args", {}).keys()]:
                            return f"{dst_var}[:] = {src_var}[:]  # assemble operation (loop accumulator, full copy)"
                        else:
                            return f"{dst_var}[:] = {src_var}[:]  # assemble operation (full copy)"
                
                # Use offset for slice assignment (e.g., offset=[%i, 0] means [i:i+1, :])
                # Remove % prefix if present
                if offset_0.startswith("%"):
                    offset_0 = offset_0[1:]
                if offset_1.startswith("%"):
                    offset_1 = offset_1[1:]
                if dst_var in [acc for loop_info in for_loop_vars.values() for acc in loop_info.get("iter_args", {}).keys()]:
                    # For loop accumulator, update in place with offset
                    return f"{dst_var}[{offset_0}:{offset_0}+1, :] = {src_var}  # assemble operation (loop accumulator with offset)"
                else:
                    # For output parameter, use offset
                    return f"{dst_var}[{offset_0}:{offset_0}+1, :] = {src_var}  # assemble operation (with offset)"
            else:
                # No offset specified - default behavior
                if dst_var in [acc for loop_info in for_loop_vars.values() for acc in loop_info.get("iter_args", {}).keys()]:
                    # For loop accumulator, update in place with proper slicing
                    return f"{dst_var}[0:1, :] = {src_var}[0:1, :] if {src_var}.shape[0] > 1 else {src_var}  # assemble operation (loop accumulator)"
                else:
                    # For output parameter, check if shapes match - if so, copy entire tensor
                    return f"{dst_var}[:] = {src_var}[:]  # assemble operation"
            return f"{result_var} = {clean_operands[0]}  # assemble operation"
        
        # Map operation to NumPy
        numpy_op = self.op_map.get(op_name)
        if numpy_op:
            if len(clean_operands) == 1:
                # Unary operation
                if numpy_op.startswith("np."):
                    return f"{result_var} = {numpy_op}({clean_operands[0]})"
                else:
                    return f"{result_var} = {numpy_op}{clean_operands[0]}"
            elif len(clean_operands) == 2:
                # Binary operation - check for undefined operands
                op1, op2 = clean_operands[0], clean_operands[1]
                # If operand references undefined variable from control flow, use a fallback
                if op1 not in defined_vars and not op1.replace(".", "").replace("-", "").isdigit() and op1 not in [p[0] for p in param_list]:
                    # This is likely from control flow - use a placeholder
                    if "_result_" in op1:
                        # Handle tuple access from if statement
                        base = op1.split("_result_")[0]
                        op1 = f"{base}_then"  # Simplified: use then branch
                if op2 not in defined_vars and not op2.replace(".", "").replace("-", "").isdigit() and op2 not in [p[0] for p in param_list]:
                    if "_result_" in op2:
                        base = op2.split("_result_")[0]
                        op2 = f"{base}_then"
                return f"{result_var} = {op1} {numpy_op} {op2}"
        
        # Fallback: unknown operation
        return f"{result_var} = {op_name}({', '.join(clean_operands)})  # TODO: implement {op_name}"

    def _convert_type(self, type_str: str) -> str:
        """Convert IR type to NumPy/Python type."""
        type_str = type_str.strip()
        if type_str in self.dtype_map:
            return self.dtype_map[type_str]
        return "Any"
    
    def _extract_dtype(self, type_str: str) -> str:
        """Extract data type from type string."""
        # Match dtype at the end: tensor<...> dtype or just dtype
        match = re.search(r">\s*(\w+)$", type_str)
        if match:
            return match.group(1)
        # Or if it's just a scalar type
        if type_str in ["fp32", "fp64", "int32", "int64", "bool"]:
            return type_str
        return "fp32"
    
    def _extract_shape_for_test(self, type_str: str) -> str:
        """Extract shape for test data generation."""
        # Match tensor<[shape], dtype> or tile<[shape], ...>
        match = re.search(r"<\[(.*?)\]", type_str)
        if match:
            shape_str = match.group(1)
            # Replace symbolic dimensions with concrete values for testing
            shape_str = re.sub(r"%\w+", "4", shape_str)  # Replace %b_1, %M, etc. with 4
            # Parse and convert to list
            shape_elements = [s.strip() for s in shape_str.split(",")]
            # Convert to Python list string
            shape_list = "[" + ", ".join(shape_elements) + "]"
            return shape_list
        return "[4, 128]"  # Default shape


def convert_ir_to_numpy(ir_file: Path, output_file: Optional[Path] = None) -> str:
    """Convert an IR file to NumPy code."""
    converter = IRToNumPyConverter()
    numpy_code = converter.convert_ir_file(ir_file)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(numpy_code)
    
    return numpy_code


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert IR text to NumPy code")
    parser.add_argument("ir_file", type=str, help="Path to IR file")
    parser.add_argument("-o", "--output", type=str, help="Output file path (optional)")
    
    args = parser.parse_args()
    
    ir_file = Path(args.ir_file)
    output_file = Path(args.output) if args.output else None
    
    numpy_code = convert_ir_to_numpy(ir_file, output_file)
    print(numpy_code)
