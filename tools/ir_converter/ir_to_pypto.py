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
IR to PyPTO Converter

Converts IR functions to PyPTO implementations.
This converter reads IR text format and generates equivalent PyPTO code.
"""
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class IRToPyPTOConverter:
    """Converter from IR text format to PyPTO code."""

    def __init__(self):
        self.dtype_map = {
            "fp32": "pypto.DT_FP32",
            "fp64": "pypto.DT_FP32",  # PyPTO doesn't support DT_FP64, use DT_FP32 as fallback
            "int32": "pypto.DT_INT32",
            "int64": "pypto.DT_INT64",
            "bool": "pypto.DT_BOOL",
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
            "tensor.neg": "pypto.neg",
            "tensor.abs": "pypto.abs",
            "tensor.exp": "pypto.exp",
            "tensor.sqrt": "pypto.sqrt",
            "tensor.ln": "pypto.log",
        }

    def convert_ir_file(self, ir_file: Path) -> str:
        """Convert an IR file to PyPTO code."""
        with open(ir_file, "r") as f:
            ir_text = f.read()
        return self.convert_ir_text(ir_text)

    def convert_ir_text(self, ir_text: str) -> str:
        """Convert IR text to PyPTO code."""
        lines = []
        lines.append("import pypto")
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
        
        # Detect parameters used as destinations in assemble operations (even if not marked #out)
        # Pattern: tensor.assemble %src, %dst or tile.assemble %src, %dst
        assemble_pattern = r"(?:tensor|tile)\.assemble\s+%[^,]+,\s*%(\w+)"
        assemble_matches = re.findall(assemble_pattern, ir_text)
        output_param_names = set(assemble_matches)
        
        # Update param_list to mark parameters used in assemble as output
        updated_param_list = []
        for param_name, param_type, direction in param_list:
            if param_name in output_param_names and direction != "out":
                # This parameter is used as a destination in assemble, treat it as output
                updated_param_list.append((param_name, param_type, "out"))
            else:
                updated_param_list.append((param_name, param_type, direction))
        param_list = updated_param_list
        
        # Separate parameters into tensors and scalars
        tensor_params = []  # Parameters with shape (tensors/tiles)
        scalar_params = []  # Scalar parameters
        
        for param_name, param_type, direction in param_list:
            shape = self._extract_shape(param_type)
            if shape is not None:
                tensor_params.append((param_name, param_type, direction))
            else:
                scalar_params.append((param_name, param_type, direction))
        
        # Generate factory function: create_<func_name>_kernel
        # Following hello_world pattern: factory takes scalars and run_mode
        factory_name = f"create_{func_name}_kernel"
        factory_signature_parts = []
        
        # Add scalar parameters to factory signature
        for param_name, param_type, direction in scalar_params:
            dtype = self._extract_dtype(param_type)
            if dtype == "fp32" or dtype == "fp64":
                factory_signature_parts.append(f"{param_name}: float")
            elif dtype == "int32" or dtype == "int64":
                factory_signature_parts.append(f"{param_name}: int")
            elif dtype == "bool":
                factory_signature_parts.append(f"{param_name}: bool")
            else:
                factory_signature_parts.append(f"{param_name}: float")
        
        # Add run_mode parameter
        factory_signature_parts.append('run_mode: str = "npu"')
        
        lines.append(f"def {factory_name}({', '.join(factory_signature_parts)}):")
        lines.append('    """Factory function to create kernel with runtime options."""')
        lines.append("    if run_mode == \"npu\":")
        lines.append("        mode = pypto.RunMode.NPU")
        lines.append("    elif run_mode == \"sim\":")
        lines.append("        mode = pypto.RunMode.SIM")
        lines.append("    else:")
        lines.append("        raise ValueError(f\"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'\")")
        lines.append("    ")
        lines.append("    @pypto.frontend.jit(runtime_options={\"run_mode\": mode})")
        lines.append(f"    def {func_name}_kernel(")
        
        # Kernel signature only has tensor parameters (no scalars, no output parameters)
        # Output parameters are returned, not passed in
        input_tensor_params = [p for p in tensor_params if p[2] != "out"]
        kernel_params = []
        for i, (param_name, param_type, direction) in enumerate(input_tensor_params):
            shape = self._extract_shape(param_type)
            dtype = self._extract_dtype(param_type)
            pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
            
            shape_for_type = self._extract_shape_for_test(param_type)
            shape_str = shape_for_type.replace("[", "").replace("]", "")
            shape_elements = [s.strip() for s in shape_str.split(",")]
            shape_tuple_str = "(" + ", ".join(shape_elements) + ")"
            param_str = f"        {param_name}: pypto.Tensor({shape_tuple_str}, {pypto_dtype})"
            if i < len(input_tensor_params) - 1:
                param_str += ","
            kernel_params.append(param_str)
        
        lines.extend(kernel_params)
        
        # Determine return type from IR function signature
        return_tensors = [p for p in tensor_params if p[2] == "out"]
        return_match = re.search(r"statement\.return(.*?)(?:\n|$)", ir_text)
        return_vars = []
        if return_match:
            return_values = return_match.group(1).strip()
            if return_values:
                return_vars = re.findall(r"%(\w+)", return_values)
        
        # Check if return_type from IR function signature is a scalar
        # return_type is parsed from: func.func @name(...) -> (fp64)
        is_scalar_return = False
        scalar_return_dtype = None
        if return_type:
            # Parse return type: could be "(fp64)" or "(tensor<...>)" or "(fp64, fp32)"
            # For now, check if it's a simple scalar type
            return_type_clean = return_type.strip()
            if return_type_clean in ["fp32", "fp64", "int32", "int64", "bool"]:
                is_scalar_return = True
                scalar_return_dtype = return_type_clean
        
        # Generate return type annotation
        # PyPTO requires return type to be pypto.Tensor(shape, dtype), not just pypto.Tensor
        # Note: PyPTO frontend.jit only supports tensor returns, not scalars
        # If IR has scalar return, we'll return a 1-element tensor
        if is_scalar_return:
            # For scalar returns, PyPTO doesn't support scalar types directly
            # Return a 1-element tensor with the scalar dtype
            pypto_dtype = self.dtype_map.get(scalar_return_dtype, "pypto.DT_FP32")
            # 1-element tensor: shape is (1,)
            lines.append(f"    ) -> pypto.Tensor((1,), {pypto_dtype}):")
        elif return_tensors:
            out_param = return_tensors[0]
            shape = self._extract_shape(out_param[1])
            dtype = self._extract_dtype(out_param[1])
            pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
            shape_for_type = self._extract_shape_for_test(out_param[1])
            shape_str = shape_for_type.replace("[", "").replace("]", "")
            shape_elements = [s.strip() for s in shape_str.split(",")]
            shape_tuple_str = "(" + ", ".join(shape_elements) + ")"
            lines.append(f"    ) -> pypto.Tensor({shape_tuple_str}, {pypto_dtype}):")
        elif input_tensor_params:
            # If no explicit output parameters, use the first input tensor's shape/dtype for return type
            # (This is a fallback - actual return should match the computation)
            first_input = input_tensor_params[0]
            param_type = first_input[1]
            shape = self._extract_shape(param_type)
            dtype = self._extract_dtype(param_type)
            pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
            shape_for_type = self._extract_shape_for_test(param_type)
            shape_str = shape_for_type.replace("[", "").replace("]", "")
            shape_elements = [s.strip() for s in shape_str.split(",")]
            shape_tuple_str = "(" + ", ".join(shape_elements) + ")"
            lines.append(f"    ) -> pypto.Tensor({shape_tuple_str}, {pypto_dtype}):")
        else:
            # No tensor parameters at all - unlikely case, but need to handle
            lines.append("):")
        
        lines.append('        """PyPTO kernel converted from IR."""')
        lines.append("")
        
        # Set vector tile shapes based on tensor dimensions (kernel body - 8 spaces indentation)
        # Need to determine a reasonable default tile shape from the function parameters
        # Get the first tensor parameter to determine dimensions
        tile_shape_set = False
        for param_name, param_type, direction in param_list:
            shape = self._extract_shape(param_type)
            if shape is not None and len(shape) > 0:
                # Generate tile shapes - use a reasonable default that matches dimensions
                # Last dimension should be 32-byte aligned (for fp32, that's 8 elements)
                # For now, use a simple approach: match dimensions with aligned last dim
                dim_count = len(shape)
                # Default tile shapes: use 1 for first dims, 16 for last (32-byte aligned for fp32)
                if dim_count == 1:
                    tile_shapes = "1"
                elif dim_count == 2:
                    tile_shapes = "1, 16"
                elif dim_count == 3:
                    tile_shapes = "1, 1, 16"
                elif dim_count == 4:
                    tile_shapes = "1, 1, 1, 16"
                else:
                    # Default for higher dimensions
                    tile_shapes = ", ".join(["1"] * (dim_count - 1) + ["16"])
                lines.append(f"        pypto.set_vec_tile_shapes({tile_shapes})")
                tile_shape_set = True
                break
        
        # If no tensor parameters found, set a default
        if not tile_shape_set:
            lines.append("        pypto.set_vec_tile_shapes(1, 16)")
        
        # Check if there are matmul operations (tensor.mul that will be converted to matmul)
        # We need to set cube tile shapes for matmul operations
        # Simple check: if tensor.mul appears, we might need matmul (will be determined in _convert_operation)
        # For safety, set cube tile shapes if we find tensor.mul operations
        if "tensor.mul" in ir_text:
            # Set cube tile shapes for matmul operations
            # Format: set_cube_tile_shapes([m_L0, m_L1], [k_L0, k_L1], [n_L0, n_L1])
            # Default values: [16, 16] for each dimension (L0, L1)
            # These are reasonable defaults for most matmul operations
            lines.append("        # Set cube tile shapes for matmul operations")
            lines.append("        pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])")
        
        lines.append("")
        
        # Initialize output parameters as local variables (they're not in the function signature)
        output_tensor_params = [p for p in tensor_params if p[2] == "out"]
        for param_name, param_type, direction in output_tensor_params:
            # Use _extract_shape_for_test to get concrete values (replaces symbolic dims with defaults)
            shape = self._extract_shape_for_test(param_type)
            dtype = self._extract_dtype(param_type)
            pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
            if shape:
                # shape is already a Python list string like "[4, 128]"
                # pypto.zeros() signature: zeros(*size, dtype=None)
                # Can be called as: zeros([4, 128], dtype=DT_FP32) or zeros(4, 128, dtype=DT_FP32)
                lines.append(f"        # Initialize output parameter {param_name}")
                lines.append(f"        {param_name} = pypto.zeros({shape}, dtype={pypto_dtype})")
        
        if output_tensor_params:
            lines.append("")
        
        # Check for control flow structures
        has_for_loop = "statement.for" in ir_text
        has_if_stmt = "statement.if" in ir_text
        
        # Extract if statement information (needed before processing operations)
        if_results = {}  # Map result_var -> condition_var
        yield_mapping = {}  # Map result_var -> (then_var, else_var) or (then_vars, else_vars)
        if_blocks = []  # Store if statement blocks for later processing
        
        if has_if_stmt:
            # Parse if statements: %r0 = statement.if %cond { ... } else { ... }
            # Pattern: %result = statement.if %cond { ... } else { ... }
            if_pattern = r"%(\w+)(?:,\s*%(\w+))?\s*=\s*statement\.if\s+%(\w+)"
            if_matches = list(re.finditer(if_pattern, ir_text))
            
            for if_match in if_matches:
                result_vars = [if_match.group(1)]
                if if_match.group(2):
                    result_vars.append(if_match.group(2))
                cond_var = if_match.group(3)
                start_pos = if_match.end()
                
                # Find then block - skip whitespace first
                pos = start_pos
                while pos < len(ir_text) and ir_text[pos] in ' \n\t':
                    pos += 1
                # Now find the opening brace
                if pos < len(ir_text) and ir_text[pos] == '{':
                    brace_count = 1
                    pos += 1
                    while pos < len(ir_text) and brace_count > 0:
                        if ir_text[pos] == '{':
                            brace_count += 1
                        elif ir_text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    then_block_end = pos - 1
                    then_block = ir_text[start_pos:then_block_end]
                else:
                    then_block = ""
                    then_block_end = start_pos
                
                # Find else block - search from the end of then block
                else_start = ir_text.find("} else {", start_pos)
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
                    if_end_pos = pos - 1
                else:
                    else_block = ""
                    if_end_pos = then_block_end
                
                # Extract yield from then and else blocks
                then_yield = re.search(r"statement\.yield\s+(.*?)(?:\n|$)", then_block)
                else_yield = re.search(r"statement\.yield\s+(.*?)(?:\n|$)", else_block)
                
                # Store if block information for later processing
                # Use if_match.start() for the actual start position of the if statement
                if_blocks.append({
                    "result_vars": result_vars,
                    "cond_var": cond_var,
                    "then_block": then_block,
                    "else_block": else_block,
                    "start_pos": if_match.start(),  # Actual start of if statement declaration
                    "end_pos": if_end_pos
                })
                
                if then_yield and else_yield:
                    then_vars = re.findall(r"%(\w+)", then_yield.group(1))
                    else_vars = re.findall(r"%(\w+)", else_yield.group(1))
                    is_tuple_result = len(then_vars) > 1 or len(else_vars) > 1
                    for i, rv in enumerate(result_vars):
                        if_results[rv] = cond_var
                        if is_tuple_result:
                            yield_mapping[rv] = (then_vars, else_vars)
                        elif i < len(then_vars) and i < len(else_vars):
                            yield_mapping[rv] = (then_vars[i], else_vars[i])
        
        # Extract for loop information
        for_loop_vars = {}  # Map loop_var -> {start, end, step, iter_args: {acc: init}}
        for_loop_blocks = {}  # Map loop_var -> list of op blocks inside that loop
        loop_result_vars = {}  # Map loop_var -> list of result variables (e.g., %acc_output = statement.for ...)
        
        if has_for_loop:
            # Parse for loops - handle both with and without result variable
            # Pattern 1: %result = statement.for %i = %start to %end step %step iter_args(%acc = %init)
            for_pattern_with_result = r"%(\w+)(?:,\s*%(\w+))?\s*=\s*statement\.for\s+%(\w+)\s*=\s*%(\w+)\s+to\s*%(\w+)\s+step\s*%(\w+)\s+iter_args\((.*?)\)"
            for_matches = list(re.finditer(for_pattern_with_result, ir_text, re.DOTALL))
            
            if not for_matches:
                # Pattern 2: statement.for %i = %start to %end step %step iter_args(%acc = %init) (no result variable)
                for_pattern = r"statement\.for\s+%(\w+)\s*=\s*%(\w+)\s+to\s*%(\w+)\s+step\s*%(\w+)\s+iter_args\((.*?)\)"
                for_matches = list(re.finditer(for_pattern, ir_text, re.DOTALL))
            
            for for_match in for_matches:
                # Check if this is the pattern with result variable
                # Pattern with result variable has 7 groups: result1, result2(opt), loop_var, start, end, step, iter_args
                if len(for_match.groups()) >= 7:
                    # Pattern with result variable: %result = statement.for ...
                    result_vars = [for_match.group(1)]
                    if for_match.group(2):
                        result_vars.append(for_match.group(2))
                    loop_var = for_match.group(3)
                    start_var = for_match.group(4)
                    end_var = for_match.group(5)
                    step_var = for_match.group(6)
                    iter_args_str = for_match.group(7)
                    loop_result_vars[loop_var] = result_vars
                else:
                    # Pattern without result variable: statement.for ...
                    loop_var = for_match.group(1)
                    start_var = for_match.group(2)
                    end_var = for_match.group(3)
                    step_var = for_match.group(4)
                    iter_args_str = for_match.group(5)
                    loop_result_vars[loop_var] = []
                
                # Parse iter_args: %acc0 = %output_x : type, %acc1 = %output_y : type
                iter_args = {}
                iter_parts = re.findall(r"%(\w+)\s*=\s*%(\w+)", iter_args_str)
                for acc, init in iter_parts:
                    iter_args[acc] = init
                
                # Calculate loop bounds for this loop
                loop_start = for_match.end()  # Position after iter_args(...) {
                # Find the matching closing brace for the loop body
                brace_count = 1
                pos = loop_start
                while pos < len(ir_text) and brace_count > 0:
                    if ir_text[pos] == '{':
                        brace_count += 1
                    elif ir_text[pos] == '}':
                        brace_count -= 1
                    pos += 1
                loop_end = pos - 1  # Position of closing brace
                
                for_loop_vars[loop_var] = {
                    "start": start_var,
                    "end": end_var,
                    "step": step_var,
                    "iter_args": iter_args,
                    "loop_start": loop_start,  # Store loop bounds for later use
                    "loop_end": loop_end
                }
        
        # Extract operations from statement.op blocks (but skip those inside if statements)
        # Note: if_blocks is populated above, so we can check against it
        op_block_matches = re.finditer(r"statement\.op \{", ir_text)
        op_blocks = []
        for match in op_block_matches:
            op_start = match.start()
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
                op_content = ir_text[start:pos-1]
                # Check if this op_block is inside an if statement
                is_in_if = False
                if has_if_stmt:
                    for if_block_info in if_blocks:
                        if_start = if_block_info["start_pos"]
                        if_end = if_block_info["end_pos"]
                        # Check if op_start is within if statement bounds
                        if op_start >= if_start and op_start < if_end:
                            is_in_if = True
                            break
                if not is_in_if:
                    op_blocks.append(op_content)
        
        # Identify which op blocks are inside for loops
        if has_for_loop:
            for loop_var, loop_info in for_loop_vars.items():
                # Pattern to match: (optional result vars) statement.for %loop_var = ... iter_args(...) {
                for_pattern_inner = rf"(?:%\w+(?:,\s*%\w+)?\s*=\s*)?statement\.for\s+%{loop_var}\s*=\s*%\w+\s+to\s*%\w+\s+step\s*%\w+\s+iter_args\(.*?\)\s*\{{"
                for_match = re.search(for_pattern_inner, ir_text, re.DOTALL)
                if for_match:
                    loop_start = for_match.end()
                    brace_count = 1
                    pos = loop_start
                    while pos < len(ir_text) and brace_count > 0:
                        if ir_text[pos] == '{':
                            brace_count += 1
                        elif ir_text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    loop_end = pos - 1
                    
                    for_loop_blocks[loop_var] = []
                    for op_block in op_blocks:
                        op_start = ir_text.find(op_block)
                        if op_start >= loop_start and op_start < loop_end:
                            for_loop_blocks[loop_var].append(op_block)
        
        # Track constants defined in op blocks
        constants = {}
        
        # Extract return variables for later use
        return_match = re.search(r"statement\.return(.*?)(?:\n|$)", ir_text)
        return_vars = []
        if return_match:
            return_values = return_match.group(1).strip()
            if return_values:
                return_vars = re.findall(r"%(\w+)", return_values)
        
        # Note: Scalar parameters are captured from closure (factory function parameters)
        # They don't need to be passed to the kernel
        
        # Process for loops first (operations inside loops)
        defined_vars = set([p[0] for p in param_list])
        
        # Identify which if statements are inside loops
        if_blocks_in_loops = {}  # Map loop_var -> list of if_block_info
        if_blocks_outside_loops = []  # List of if_block_info outside loops
        
        if has_if_stmt:
            if has_for_loop:
                for if_block_info in if_blocks:
                    if_start = if_block_info["start_pos"]
                    if_end = if_block_info["end_pos"]
                    # Check which loop (if any) contains this if statement
                    in_loop = None
                    for loop_var, loop_info in for_loop_vars.items():
                        # Use stored loop bounds if available
                        if "loop_start" in loop_info and "loop_end" in loop_info:
                            loop_start = loop_info["loop_start"]
                            loop_end = loop_info["loop_end"]
                        else:
                            # Fallback: recalculate loop bounds
                            for_pattern_inner = rf"(?:%\w+(?:,\s*%\w+)?\s*=\s*)?statement\.for\s+%{loop_var}\s*=\s*%\w+\s+to\s*%\w+\s+step\s*%\w+\s+iter_args\(.*?\)\s*\{{"
                            for_match = re.search(for_pattern_inner, ir_text, re.DOTALL)
                            if for_match:
                                loop_start = for_match.end()
                                brace_count = 1
                                pos = loop_start
                                while pos < len(ir_text) and brace_count > 0:
                                    if ir_text[pos] == '{':
                                        brace_count += 1
                                    elif ir_text[pos] == '}':
                                        brace_count -= 1
                                    pos += 1
                                loop_end = pos - 1
                            else:
                                continue
                        # Check if if statement is inside loop: if_start should be >= loop_start and if_end should be <= loop_end
                        if if_start >= loop_start and if_end <= loop_end:
                            in_loop = loop_var
                            break
                    if in_loop:
                        if in_loop not in if_blocks_in_loops:
                            if_blocks_in_loops[in_loop] = []
                        if_blocks_in_loops[in_loop].append(if_block_info)
                    else:
                        if_blocks_outside_loops.append(if_block_info)
            else:
                # No for loops, so all if statements are outside loops
                if_blocks_outside_loops = if_blocks
        
        for loop_var, loop_info in for_loop_vars.items():
            # Check if this loop has any blocks (op_blocks or if_blocks) to process
            has_op_blocks = loop_var in for_loop_blocks and len(for_loop_blocks[loop_var]) > 0
            loop_if_blocks = if_blocks_in_loops.get(loop_var, [])
            has_if_blocks = len(loop_if_blocks) > 0
            
            if has_op_blocks or has_if_blocks:
                # Initialize accumulator variables BEFORE the loop
                iter_args = loop_info.get("iter_args", {})
                for acc, init in iter_args.items():
                    if init in [p[0] for p in param_list]:
                        # Always modify the parameter in place for accumulators
                        # This allows the result to be visible after the function call
                        lines.append(f"        # Initialize accumulator {acc} from {init} (modify in place)")
                        lines.append(f"        {acc} = {init}")
                        defined_vars.add(acc)
                
                # Generate pypto.loop() call
                # For now, use a simple range - in real implementation, would use symbolic bounds
                lines.append(f"        # For loop: {loop_var}")
                lines.append(f"        for {loop_var} in pypto.loop(0, 4, 1):  # TODO: Use actual bounds")
                
                # Collect all blocks (op_blocks and if_blocks) within this loop and sort by position
                
                # Find loop bounds in IR - use stored bounds if available
                if "loop_start" in loop_info and "loop_end" in loop_info:
                    loop_start = loop_info["loop_start"]
                    loop_end = loop_info["loop_end"]
                else:
                    # Fallback: recalculate loop bounds
                    # Pattern to match: (optional result vars) statement.for %loop_var = ... iter_args(...) {
                    for_pattern_inner = rf"(?:%\w+(?:,\s*%\w+)?\s*=\s*)?statement\.for\s+%{loop_var}\s*=\s*%\w+\s+to\s*%\w+\s+step\s*%\w+\s+iter_args\(.*?\)\s*\{{"
                    for_match = re.search(for_pattern_inner, ir_text, re.DOTALL)
                    if for_match:
                        loop_start = for_match.end()
                        brace_count = 1
                        pos = loop_start
                        while pos < len(ir_text) and brace_count > 0:
                            if ir_text[pos] == '{':
                                brace_count += 1
                            elif ir_text[pos] == '}':
                                brace_count -= 1
                            pos += 1
                        loop_end = pos - 1
                    else:
                        continue
                
                # Collect all blocks with their positions
                blocks_with_pos = []
                
                # Add op_blocks
                op_blocks_in_loop = for_loop_blocks.get(loop_var, [])
                for op_block in op_blocks_in_loop:
                    # Check if this op_block is inside an if statement
                    is_in_if = False
                    for if_block_info in loop_if_blocks:
                        if_start = if_block_info["start_pos"]
                        if_end = if_block_info["end_pos"]
                        # Find the position of this op_block in the IR
                        # Search for "statement.op {" followed by the op_block content
                        op_search_start = loop_start
                        while True:
                            op_start = ir_text.find("statement.op {", op_search_start, loop_end)
                            if op_start == -1:
                                break
                            # Find the matching closing brace
                            brace_count = 1
                            pos = op_start + len("statement.op {")
                            while pos < len(ir_text) and brace_count > 0:
                                if ir_text[pos] == '{':
                                    brace_count += 1
                                elif ir_text[pos] == '}':
                                    brace_count -= 1
                                pos += 1
                            if brace_count == 0:
                                op_content = ir_text[op_start + len("statement.op {"):pos-1]
                                if op_content.strip() == op_block.strip():
                                    if op_start >= if_start and op_start < if_end:
                                        is_in_if = True
                                    break
                            op_search_start = pos
                    if not is_in_if:
                        # Find the position of this op_block in the IR
                        op_search_start = loop_start
                        op_pos = None
                        while op_search_start < loop_end:
                            op_start = ir_text.find("statement.op {", op_search_start, loop_end)
                            if op_start == -1:
                                break
                            # Find the matching closing brace
                            brace_count = 1
                            pos = op_start + len("statement.op {")
                            while pos < len(ir_text) and brace_count > 0:
                                if ir_text[pos] == '{':
                                    brace_count += 1
                                elif ir_text[pos] == '}':
                                    brace_count -= 1
                                pos += 1
                            if brace_count == 0:
                                op_content = ir_text[op_start + len("statement.op {"):pos-1]
                                if op_content.strip() == op_block.strip():
                                    op_pos = op_start
                                    break
                            op_search_start = pos
                        if op_pos is not None and op_pos >= loop_start and op_pos < loop_end:
                            blocks_with_pos.append(("op", op_block, op_pos))
                
                # Add if_blocks
                for if_block_info in loop_if_blocks:
                    if_start = if_block_info["start_pos"]
                    if if_start >= loop_start and if_start < loop_end:
                        blocks_with_pos.append(("if", if_block_info, if_start))
                
                # Sort by position
                blocks_with_pos.sort(key=lambda x: x[2])
                
                # Process blocks in order
                for block_type, block_data, block_pos in blocks_with_pos:
                        if block_type == "op":
                            op_block = block_data
                            operations = self._parse_operations(op_block)
                            for op in operations:
                                pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping, loop_result_vars)
                                if pypto_code:
                                    # Handle multi-line strings - split and add each line with correct indentation
                                    for line in pypto_code.split('\n'):
                                        if line.strip():  # Skip empty lines
                                            lines.append(f"            {line}")
                                    # Track defined variables
                                    if '=' in pypto_code:
                                        var_name = pypto_code.split('=')[0].strip()
                                        defined_vars.add(var_name)
                        elif block_type == "if":
                            if_block_info = block_data
                            result_vars = if_block_info["result_vars"]
                            cond_var = if_block_info["cond_var"]
                            then_block = if_block_info["then_block"]
                            else_block = if_block_info["else_block"]
                            
                            # Generate if/else structure inside loop
                            lines.append(f"            # If statement: {result_vars[0]}")
                            lines.append(f"            if {cond_var}:")
                            
                            # Process operations inside then block
                            then_op_blocks = re.finditer(r"statement\.op \{", then_block)
                            for then_match in then_op_blocks:
                                then_start = then_match.end()
                                then_brace_count = 1
                                then_pos = then_start
                                while then_pos < len(then_block) and then_brace_count > 0:
                                    if then_block[then_pos] == '{':
                                        then_brace_count += 1
                                    elif then_block[then_pos] == '}':
                                        then_brace_count -= 1
                                    then_pos += 1
                                if then_brace_count == 0:
                                    then_op_content = then_block[then_start:then_pos-1]
                                    operations = self._parse_operations(then_op_content)
                                    for op in operations:
                                        pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping, loop_result_vars)
                                        if pypto_code:
                                            # Handle multi-line strings - split and add each line with correct indentation
                                            for line in pypto_code.split('\n'):
                                                if line.strip():  # Skip empty lines
                                                    lines.append(f"                {line}")
                                            if '=' in pypto_code:
                                                var_name = pypto_code.split('=')[0].strip()
                                                defined_vars.add(var_name)
                            
                            # Define if result variables inside then block
                            for rv in result_vars:
                                if rv in yield_mapping:
                                    yield_data = yield_mapping[rv]
                                    if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                                        # Tuple result - define rv_val_0, rv_val_1, etc. in then block
                                        then_vars = yield_data[0]
                                        for i in range(len(then_vars)):
                                            then_var = then_vars[i]
                                            if then_var in defined_vars:
                                                lines.append(f"                {rv}_val_{i} = {then_var}")
                                                defined_vars.add(f"{rv}_val_{i}")
                                    else:
                                        # Single result - define in then block
                                        then_var = yield_data[0]
                                        if then_var in defined_vars:
                                            lines.append(f"                {rv} = {then_var}")
                                            defined_vars.add(rv)
                            
                            lines.append(f"            else:")
                            
                            # Process operations inside else block
                            else_op_blocks = re.finditer(r"statement\.op \{", else_block)
                            for else_match in else_op_blocks:
                                else_start_inner = else_match.end()
                                else_brace_count = 1
                                else_pos = else_start_inner
                                while else_pos < len(else_block) and else_brace_count > 0:
                                    if else_block[else_pos] == '{':
                                        else_brace_count += 1
                                    elif else_block[else_pos] == '}':
                                        else_brace_count -= 1
                                    else_pos += 1
                                if else_brace_count == 0:
                                    else_op_content = else_block[else_start_inner:else_pos-1]
                                    operations = self._parse_operations(else_op_content)
                                    for op in operations:
                                        pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping, loop_result_vars)
                                        if pypto_code:
                                            # Handle multi-line strings - split and add each line with correct indentation
                                            for line in pypto_code.split('\n'):
                                                if line.strip():  # Skip empty lines
                                                    lines.append(f"                {line}")
                                            if '=' in pypto_code:
                                                var_name = pypto_code.split('=')[0].strip()
                                                defined_vars.add(var_name)
                            
                            # Define if result variables inside else block (for tuple results)
                            for rv in result_vars:
                                if rv in yield_mapping:
                                    yield_data = yield_mapping[rv]
                                    if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                                        # Tuple result - define rv_val_0, rv_val_1, etc. in else block
                                        else_vars = yield_data[1]
                                        for i in range(len(else_vars)):
                                            else_var = else_vars[i]
                                            if else_var in defined_vars:
                                                lines.append(f"                {rv}_val_{i} = {else_var}")
                                                defined_vars.add(f"{rv}_val_{i}")
                                    else:
                                        # Single result - define in else block
                                        else_var = yield_data[1]
                                        if else_var in defined_vars:
                                            lines.append(f"                {rv} = {else_var}")
                                            defined_vars.add(rv)
                
                # Assign loop result variable(s) after the loop
                if loop_var in loop_result_vars and loop_result_vars[loop_var]:
                    result_vars = loop_result_vars[loop_var]
                    iter_args = loop_info.get("iter_args", {})
                    acc_list = list(iter_args.keys())
                    for i, result_var in enumerate(result_vars):
                        if i < len(acc_list):
                            acc = acc_list[i]
                            # The accumulator was modified in place, so assign it to the result variable
                            lines.append(f"        # Assign loop result {result_var} from accumulator {acc}")
                            lines.append(f"        {result_var} = {acc}")
                            defined_vars.add(result_var)
        
        # Process if statements outside loops - generate code for operations inside then/else blocks
        if has_if_stmt:
            for if_block_info in if_blocks_outside_loops:
                result_vars = if_block_info["result_vars"]
                cond_var = if_block_info["cond_var"]
                then_block = if_block_info["then_block"]
                else_block = if_block_info["else_block"]
                start_pos = if_block_info["start_pos"]
                if_end_pos = if_block_info["end_pos"]
                
                # Generate if/else structure
                lines.append(f"        # If statement: {result_vars[0]}")
                lines.append(f"        if {cond_var}:")
                
                # Process operations inside then block
                then_op_blocks = re.finditer(r"statement\.op \{", then_block)
                for then_match in then_op_blocks:
                    then_start = then_match.end()
                    then_brace_count = 1
                    then_pos = then_start
                    while then_pos < len(then_block) and then_brace_count > 0:
                        if then_block[then_pos] == '{':
                            then_brace_count += 1
                        elif then_block[then_pos] == '}':
                            then_brace_count -= 1
                        then_pos += 1
                    if then_brace_count == 0:
                        then_op_content = then_block[then_start:then_pos-1]
                        operations = self._parse_operations(then_op_content)
                        for op in operations:
                            pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping)
                            if pypto_code:
                                # Handle multi-line strings - split and add each line with correct indentation
                                for line in pypto_code.split('\n'):
                                    if line.strip():  # Skip empty lines
                                        lines.append(f"            {line}")
                                if '=' in pypto_code:
                                    var_name = pypto_code.split('=')[0].strip()
                                    defined_vars.add(var_name)
                
                # Define if result variables inside then block
                for rv in result_vars:
                    if rv in yield_mapping:
                        yield_data = yield_mapping[rv]
                        if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                            # Tuple result - define rv_val_0, rv_val_1, etc. in then block
                            then_vars = yield_data[0]
                            for i in range(len(then_vars)):
                                then_var = then_vars[i]
                                if then_var in defined_vars:
                                    lines.append(f"            {rv}_val_{i} = {then_var}")
                                    defined_vars.add(f"{rv}_val_{i}")
                        else:
                            # Single result - define in then block
                            then_var = yield_data[0]
                            if then_var in defined_vars:
                                lines.append(f"            {rv} = {then_var}")
                                defined_vars.add(rv)
                
                lines.append(f"        else:")
                
                # Process operations inside else block
                else_op_blocks = re.finditer(r"statement\.op \{", else_block)
                for else_match in else_op_blocks:
                    else_start_inner = else_match.end()
                    else_brace_count = 1
                    else_pos = else_start_inner
                    while else_pos < len(else_block) and else_brace_count > 0:
                        if else_block[else_pos] == '{':
                            else_brace_count += 1
                        elif else_block[else_pos] == '}':
                            else_brace_count -= 1
                        else_pos += 1
                    if else_brace_count == 0:
                        else_op_content = else_block[else_start_inner:else_pos-1]
                        operations = self._parse_operations(else_op_content)
                        for op in operations:
                            pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping)
                            if pypto_code:
                                # Handle multi-line strings - split and add each line with correct indentation
                                for line in pypto_code.split('\n'):
                                    if line.strip():  # Skip empty lines
                                        lines.append(f"            {line}")
                                if '=' in pypto_code:
                                    var_name = pypto_code.split('=')[0].strip()
                                    defined_vars.add(var_name)
                
                # Define if result variables inside else block
                for rv in result_vars:
                    if rv in yield_mapping:
                        yield_data = yield_mapping[rv]
                        if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                            # Tuple result - define rv_val_0, rv_val_1, etc. in else block
                            else_vars = yield_data[1]
                            for i in range(len(else_vars)):
                                else_var = else_vars[i]
                                if else_var in defined_vars:
                                    lines.append(f"            {rv}_val_{i} = {else_var}")
                                    defined_vars.add(f"{rv}_val_{i}")
                        else:
                            # Single result - define in else block
                            else_var = yield_data[1]
                            if else_var in defined_vars:
                                lines.append(f"            {rv} = {else_var}")
                                defined_vars.add(rv)
        
        # Process operations outside for loops and if statements
        for op_block in op_blocks:
            # Skip if this block is inside a for loop
            is_in_loop = False
            for loop_var, loop_blocks in for_loop_blocks.items():
                if op_block in loop_blocks:
                    is_in_loop = True
                    break
            if is_in_loop:
                continue
            
            # Skip if this block is inside an if statement (already processed above)
            is_in_if = False
            if has_if_stmt:
                for if_block_info in if_blocks:
                    start_pos = if_block_info["start_pos"]
                    if_end = if_block_info["end_pos"]
                    # Check if op_block is within if statement bounds
                    op_start = ir_text.find(op_block)
                    if op_start >= start_pos and op_start < if_end:
                        is_in_if = True
                        break
            if is_in_if:
                continue
            
            operations = self._parse_operations(op_block)
            for op in operations:
                # Handle if statement results that need to be defined before this operation uses them
                operands = op.get("operands", [])
                for operand in operands:
                    if operand.startswith("%"):
                        var_name = operand[1:]
                        if var_name in if_results and var_name not in defined_vars:
                            # Define if result before using it
                            cond_var = if_results[var_name]
                            if var_name in yield_mapping:
                                yield_data = yield_mapping[var_name]
                                if isinstance(yield_data, tuple) and not isinstance(yield_data[0], list):
                                    # Note: If result variables should have been defined in the if/else blocks above
                                    # If we reach here, it means the if statement wasn't properly processed
                                    # This is a fallback that won't work with PyPTO (ternary expressions not supported)
                                    # The if/else blocks should have already defined the result variables
                                    pass
                
                pypto_code = self._convert_operation(op, constants, defined_vars, for_loop_vars, if_results, param_list, yield_mapping)
                if pypto_code:
                    # Handle multi-line strings - split and add each line with correct indentation
                    for line in pypto_code.split('\n'):
                        if line.strip():  # Skip empty lines
                            lines.append(f"        {line}")
                    # Track defined variables
                    if '=' in pypto_code:
                        var_name = pypto_code.split('=')[0].strip()
                        defined_vars.add(var_name)
                        
                        # Check if this variable is a yield variable from an if statement
                        # If so, define the if result immediately after this variable is computed
                        for rv, yv_data in yield_mapping.items():
                            if isinstance(yv_data, tuple) and not isinstance(yv_data[0], list):
                                then_var, else_var = yv_data
                                # Note: If result variables should have been defined in the if/else blocks above
                                # If we reach here, it means the if statement wasn't properly processed
                                # The if/else blocks should have already defined the result variables
                                pass
        
        # Finalize if statement results (outside loops) - ensure r0 is defined before it's used
        # Insert r0 definition right after all operations but before return
        # Find where to insert - right before the return statement
        return_insert_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("return") and "        return" in lines[i]:
                return_insert_idx = i
                break
        
        # If no explicit return found, insert at the end before closing the function
        if return_insert_idx is None:
            return_insert_idx = len(lines) - 1
        
        # Note: All if result variables should have been defined in the if/else blocks above
        # For tuple results, we define rv_val_0, rv_val_1, etc. instead of rv itself
        # PyPTO doesn't support ternary expressions, so we can't define them here
        for rv in if_results.keys():
            if rv not in defined_vars:
                # Check if this is a tuple result - if so, rv_val_0, rv_val_1, etc. should be defined
                if rv in yield_mapping:
                    yield_data = yield_mapping[rv]
                    if isinstance(yield_data, tuple) and isinstance(yield_data[0], list):
                        # Tuple result - check if at least one indexed variable is defined
                        then_vars = yield_data[0]
                        has_indexed_var = any(f"{rv}_val_{i}" in defined_vars for i in range(len(then_vars)))
                        if not has_indexed_var:
                            # This should not happen if the if/else blocks were properly processed
                            lines.append(f"        # WARNING: {rv} (tuple result) and its indexed variables were not defined in if/else blocks - this may cause an error")
                    else:
                        # Single result - this should have been defined
                        lines.append(f"        # WARNING: {rv} was not defined in if/else blocks - this may cause an error")
        
        # Handle return statement
        # If IR has scalar return but PyPTO requires tensor, wrap scalar in tensor
        if is_scalar_return and return_vars:
            # For scalar returns, wrap in pypto.full to create a 1-element tensor with the scalar value
            # Note: pypto.frontend.jit requires return values to be variable names, not expressions
            scalar_var = return_vars[0]
            pypto_dtype = self.dtype_map.get(scalar_return_dtype, "pypto.DT_FP32")
            lines.append(f"        # IR returns scalar {scalar_return_dtype}, wrapping in 1-element tensor")
            # Check if scalar_var is a constant (literal value) - if so, use it directly
            # Otherwise, it's a computed scalar which should be a Python int/float
            # pypto.full() expects fill_value to be int/float/Element/SymbolicScalar
            # For computed scalars (Python arithmetic), they should be Python types
            # But if they're PyPTO types, we need to handle them differently
            # Try creating Element explicitly to ensure proper type conversion
            if scalar_var in constants:
                # It's a constant - use the literal value directly
                scalar_value = constants[scalar_var]
                lines.append(f"        scalar_tensor_result = pypto.full([1], {scalar_value}, {pypto_dtype})")
            else:
                # It's a computed scalar - should be Python int/float
                # But to be safe, wrap it in Element to ensure proper type
                lines.append(f"        scalar_element = pypto.Element({pypto_dtype}, {scalar_var})")
                lines.append(f"        scalar_tensor_result = pypto.full([1], scalar_element, {pypto_dtype})")
            lines.append(f"        return scalar_tensor_result")
        elif return_match:
            return_values = return_match.group(1).strip()
            if return_values:
                if len(return_vars) == 1:
                    lines.append(f"        return {return_vars[0]}")
                elif len(return_vars) > 1:
                    lines.append(f"        return ({', '.join(return_vars)})")
            else:
                # Find output parameters
                output_params = [name for name, _, dir in param_list if dir == "out"]
                if output_params:
                    if len(output_params) == 1:
                        lines.append(f"        return {output_params[0]}")
                    else:
                        lines.append(f"        return ({', '.join(output_params)})")
                else:
                    lines.append("        return")
        else:
            # Find output parameters
            output_params = [name for name, _, dir in param_list if dir == "out"]
            if output_params:
                if len(output_params) == 1:
                    lines.append(f"        return {output_params[0]}")
                else:
                    lines.append(f"        return ({', '.join(output_params)})")
            else:
                lines.append("        return")
        
        # Close kernel function and return it from factory
        lines.append("    ")
        lines.append(f"    return {func_name}_kernel")
        
        # Add test function with argparse for run_mode
        lines.append("")
        lines.append("")
        lines.append("def test_" + func_name + "(run_mode: str = \"npu\"):")
        lines.append('    """Test function for ' + func_name + '."""')
        lines.append("    import torch")
        lines.append(f"    print(f\"Testing {func_name} with run_mode={{run_mode}}...\")")
        lines.append("    ")
        
        # Generate test inputs based on parameters
        test_inputs = []
        output_params = []
        for param_name, param_type, direction in param_list:
            if direction == "out":
                # Output parameter - for frontend.jit, create torch tensor (it will be converted internally)
                # Following hello_world example pattern where outputs are handled by return values
                # But if there's an output parameter, create a torch tensor for it
                shape = self._extract_shape_for_test(param_type)
                lines.append(f"    {param_name} = torch.randn({shape}, dtype=torch.float32)")
                test_inputs.append(param_name)
                output_params.append(param_name)
            else:
                # Input parameter - create test data
                shape = self._extract_shape_for_test(param_type)
                dtype = self._extract_dtype(param_type)
                pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
                if "tensor" in param_type.lower() or "tile" in param_type.lower():
                    # frontend.jit accepts torch tensors directly, no need for pypto.from_torch
                    # Following hello_world example pattern
                    lines.append(f"    # Create test tensor with shape {shape}")
                    lines.append(f"    {param_name} = torch.randn({shape}, dtype=torch.float32)")
                else:
                    # Scalar - just use the numeric value directly
                    # PyPTO functions accept Python scalars directly
                    if dtype == "fp32" or dtype == "fp64":
                        lines.append(f"    {param_name} = 2.5")
                    elif dtype == "int32" or dtype == "int64":
                        lines.append(f"    {param_name} = 2")
                    elif dtype == "bool":
                        lines.append(f"    {param_name} = True")
                    else:
                        lines.append(f"    {param_name} = 2.5")  # Default to float
                test_inputs.append(param_name)
        
        lines.append("    ")
        lines.append("    # Create kernel using factory function")
        # Build factory function call arguments
        factory_args = []
        for param_name, param_type, direction in scalar_params:
            factory_args.append(param_name)
        factory_args.append("run_mode")  # Use run_mode parameter from test function
        lines.append(f"    kernel_func = {factory_name}({', '.join(factory_args)})")
        lines.append("    ")
        
        # Build kernel call arguments (only tensor inputs, no scalars, no outputs)
        kernel_inputs = [p[0] for p in tensor_params if p[2] != "out"]
        if return_vars or return_tensors:
            if (return_vars and len(return_vars) == 1) or (not return_vars and len(return_tensors) == 1):
                lines.append(f"    result = kernel_func({', '.join(kernel_inputs)})")
                lines.append("    print(f\"Result: {result}\")")
                lines.append("    if hasattr(result, 'to_torch'):")
                lines.append("        result_torch = result.to_torch()")
                lines.append("        print(f\"Result (torch): {result_torch}\")")
                lines.append("        print(f\"Result shape: {result_torch.shape}\")")
            else:
                lines.append(f"    results = kernel_func({', '.join(kernel_inputs)})")
                lines.append("    print(f\"Results: {results}\")")
                if return_vars:
                    for i, var in enumerate(return_vars):
                        lines.append(f"    if hasattr(results[{i}], 'to_torch'):")
                        lines.append(f"        print(f\"Result {i} ({var}): {results[{i}].to_torch()}\")")
        else:
            lines.append(f"    kernel_func({', '.join(kernel_inputs)})")
        
        lines.append("    ")
        lines.append("    print(\"Test completed successfully!\")")
        lines.append("    return True")
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    import argparse")
        lines.append(f"    parser = argparse.ArgumentParser(description=\"Test {func_name} function\")")
        lines.append("    parser.add_argument(")
        lines.append("        \"--run_mode\",")
        lines.append("        type=str,")
        lines.append("        default=\"npu\",")
        lines.append("        choices=[\"npu\", \"sim\"],")
        lines.append("        help=\"Run mode: 'npu' or 'sim' (default: npu)\"")
        lines.append("    )")
        lines.append("    args = parser.parse_args()")
        lines.append(f"    test_{func_name}(run_mode=args.run_mode)")
        
        return "\n".join(lines)

    def _parse_parameters(self, params_str: str) -> List[Tuple[str, str, Optional[str]]]:
        """Parse function parameters."""
        params = []
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
            param_pattern = r"%(\w+):\s*(.+?)(?:\s*#(\w+))?$"
            match = re.search(param_pattern, param_str)
            if match:
                name = match.group(1)
                param_type = match.group(2).strip()
                direction = match.group(3) if match.group(3) else None
                params.append((name, param_type, direction))
        return params

    def _extract_shape(self, type_str: str) -> Optional[List]:
        """Extract shape from type string."""
        # Match tensor<[shape], dtype> or tile<[shape], ...>
        match = re.search(r"<\[(.*?)\]", type_str)
        if match:
            shape_str = match.group(1)
            # Parse shape elements (can be symbolic like %b_1 or numeric like 128)
            shape_elements = [s.strip() for s in shape_str.split(",")]
            return shape_elements
        return None

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
            # Updated pattern to capture full output type including spaces and attributes
            # First match up to the attributes
            op_pattern_base = r"%(\w+)\s*=\s*(\S+)\s+([^:]+?)\s*:\s*\(([^)]+)\)\s*->\s*"
            base_match = re.search(op_pattern_base, line)
            if base_match:
                result_var = base_match.group(1)
                op_name = base_match.group(2)
                operands_str = base_match.group(3).strip()
                input_types = base_match.group(4)
                # Find the rest after "-> "
                rest = line[base_match.end():].strip()
                # Match output type and attributes
                # Output type is everything up to { or end of line
                # Use a more robust pattern that handles angle brackets
                if '{' in rest:
                    # Split at the first { to separate output type from attributes
                    parts = rest.split('{', 1)
                    output_type = parts[0].strip()
                    if len(parts) > 1 and parts[1].endswith('}'):
                        attrs_str = parts[1][:-1]  # Remove trailing }
                    else:
                        attrs_str = parts[1] if len(parts) > 1 else ""
                else:
                    output_type = rest.strip()
                    attrs_str = ""
                
                # Parse operands - simple split by comma (operands shouldn't have nested commas)
                operand_list = [op.strip() for op in operands_str.split(',')]
                
                # Parse attributes
                attrs = {}
                if attrs_str:
                    # Match key=value pairs in attributes
                    attr_pattern = r"(\w+)\s*=\s*\[([^\]]+)\]"
                    for attr_match in re.finditer(attr_pattern, attrs_str):
                        key = attr_match.group(1)
                        value_str = attr_match.group(2)
                        # Parse value list (e.g., "%i, 0" -> ["%i", "0"])
                        values = [v.strip() for v in value_str.split(',')]
                        attrs[key] = values
                
                # Check if any operand is a literal value in the type annotation
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
                
                operations.append({
                    "result": result_var,
                    "op": op_name,
                    "operands": operand_list,
                    "input_types": input_types,
                    "output_type": output_type,
                    "attrs": attrs,
                })
                continue
            
            # Fallback to original pattern if base match fails
            op_pattern = r"%(\w+)\s*=\s*(\S+)\s+([^:]+?)\s*:\s*\(([^)]+)\)\s*->\s*([^\s\{]+(?:\s+[^\s\{]+)*?)(?:\s*\{([^}]+)\})?"
            match = re.search(op_pattern, line)
            if match:
                result_var = match.group(1)
                op_name = match.group(2)
                operands_str = match.group(3).strip()
                input_types = match.group(4)
                output_type = match.group(5).strip()
                attrs_str = match.group(6) if match.group(6) else ""
                
                # Parse operands - simple split by comma (operands shouldn't have nested commas)
                operand_list = [op.strip() for op in operands_str.split(',')]
                
                # Parse attributes
                attrs = {}
                if attrs_str:
                    # Match key=value pairs in attributes
                    attr_pattern = r"(\w+)\s*=\s*\[([^\]]+)\]"
                    for attr_match in re.finditer(attr_pattern, attrs_str):
                        key = attr_match.group(1)
                        value_str = attr_match.group(2)
                        # Parse value list (e.g., "%i, 0" -> ["%i", "0"])
                        values = [v.strip() for v in value_str.split(',')]
                        attrs[key] = values
                
                # Check if any operand is a literal value in the type annotation
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
        """Convert an IR operation to PyPTO code."""
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
        operands = op["operands"]
        
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
                    base_var, index = var_name.split("#", 1)
                    clean_operands.append(f"{base_var}_val_{index}")
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
            return f"{result_var} = {clean_operands[0]}  # view operation"
        
        # Handle assemble operation
        if op_name == "tensor.assemble" or op_name == "tile.assemble":
            # Assemble - write result to output tensor
            if len(clean_operands) >= 2:
                src_var = clean_operands[0]
                dst_var = clean_operands[1]
                # Check offset attribute to determine how to copy
                attrs = op.get("attrs", {})
                offset = attrs.get("offset", None)
                
                # Check if dst_var is a loop accumulator
                is_accumulator = dst_var in [acc for loop_info in for_loop_vars.values() for acc in loop_info.get("iter_args", {}).keys()]
                
                if offset and len(offset) >= 2:
                    # Handle both 2D and 3D offsets
                    offset_0 = offset[0].strip()
                    offset_1 = offset[1].strip()
                    offset_2 = offset[2].strip() if len(offset) >= 3 else None
                    
                    # Remove % prefix if present
                    if offset_0.startswith("%"):
                        offset_0 = offset_0[1:]
                    if offset_1.startswith("%"):
                        offset_1 = offset_1[1:]
                    if offset_2 and offset_2.startswith("%"):
                        offset_2 = offset_2[1:]
                    
                    # If offset is [0, 0] or [0, 0, 0] (all zeros), copy entire tensor
                    all_zeros = ((offset_0 == "0" or offset_0.startswith("const_0")) and 
                                (offset_1 == "0" or offset_1.startswith("const_0")))
                    if offset_2:
                        all_zeros = all_zeros and (offset_2 == "0" or offset_2.startswith("const_0"))
                    
                    if all_zeros:
                        # Copy entire tensor
                        if is_accumulator:
                            return f"{dst_var}[:] = {src_var}[:]  # assemble operation (loop accumulator, full copy)\n{result_var} = {dst_var}"
                        else:
                            return f"{dst_var}[:] = {src_var}[:]  # assemble operation (full copy)\n{result_var} = {dst_var}"
                    else:
                        # Use offset for slice assignment
                        # For 2D: offset=[%i, 0] means [i:i+1, :]
                        # For 3D: offset=[0, %i, 0] means [:, i:i+1, :] (keep batch, slice sequence, keep features)
                        loop_var_for_offset = None
                        offset_dim = None  # Which dimension has the loop variable (0, 1, or 2)
                        
                        # Check which offset dimension is a loop variable
                        for loop_var, loop_info in for_loop_vars.items():
                            if offset_0 == loop_var:
                                loop_var_for_offset = loop_var
                                offset_dim = 0
                                break
                            elif offset_1 == loop_var:
                                loop_var_for_offset = loop_var
                                offset_dim = 1
                                break
                            elif offset_2 and offset_2 == loop_var:
                                loop_var_for_offset = loop_var
                                offset_dim = 2
                                break
                        
                        if loop_var_for_offset and offset_dim is not None:
                            # Build slice expression - use loop variable for the dimension that has it
                            if offset_2 is not None:
                                # 3D tensor: [batch, seq, features]
                                # Build slice for each dimension
                                slices = []
                                for dim_idx, off in enumerate([offset_0, offset_1, offset_2]):
                                    if dim_idx == offset_dim:
                                        # This dimension uses the loop variable
                                        slices.append(f"{loop_var_for_offset}:{loop_var_for_offset}+1")
                                    elif off == "0" or off.startswith("const_0"):
                                        # Offset is 0 - use full slice or single element
                                        slices.append(":")
                                    else:
                                        # Use offset value
                                        slices.append(f"{off}:{off}+1")
                                offset_expr = ", ".join(slices)
                            else:
                                # 2D tensor: [rows, cols]
                                if offset_dim == 0:
                                    offset_expr = f"{loop_var_for_offset}:{loop_var_for_offset}+1, :"
                                else:
                                    offset_expr = f":, {loop_var_for_offset}:{loop_var_for_offset}+1"
                        else:
                            # No loop variable found, use offset values as-is
                            if offset_2 is not None:
                                # 3D tensor - use offsets for each dimension
                                slices = []
                                for off in [offset_0, offset_1, offset_2]:
                                    if off == "0" or off.startswith("const_0"):
                                        slices.append(":")
                                    else:
                                        slices.append(f"{off}:{off}+1")
                                offset_expr = ", ".join(slices)
                            else:
                                # 2D tensor
                                offset_expr = f"{offset_0}:{offset_0}+1, :"
                        
                        if is_accumulator:
                            return f"{dst_var}[{offset_expr}] = {src_var}  # assemble operation (loop accumulator with offset)\n{result_var} = {dst_var}"
                        else:
                            return f"{dst_var}[{offset_expr}] = {src_var}  # assemble operation (with offset)\n{result_var} = {dst_var}"
                else:
                    # No offset specified - default behavior
                    if is_accumulator:
                        return f"{dst_var}[0:1, :] = {src_var}  # assemble operation (loop accumulator)\n{result_var} = {dst_var}"
                    else:
                        return f"{dst_var}[:] = {src_var}[:]  # assemble operation\n{result_var} = {dst_var}"
            return f"{result_var} = {clean_operands[0]}  # assemble operation"
        
        # Special handling for tensor.mul - check if it should be matrix multiplication
        if op_name == "tensor.mul" and len(clean_operands) == 2:
            # Check input and output types to determine if this is matrix multiplication
            input_types = op.get("input_types", "")
            output_type = op.get("output_type", "")
            
            # Parse input_types - it contains both operands: (type1, type2)
            # Split by comma but respect angle brackets
            import re
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
            
            if len(type_parts) >= 2:
                first_type = type_parts[0]
                second_type = type_parts[1]
                
                # Extract shapes from both types
                first_shape_match = re.search(r'tensor<\[([^\]]+)\],\s*\w+>', first_type)
                second_shape_match = re.search(r'tensor<\[([^\]]+)\],\s*\w+>', second_type)
                output_shape_match = re.search(r'tensor<\[([^\]]+)\],\s*\w+>', output_type)
                
                if first_shape_match and second_shape_match and output_shape_match:
                    first_dims = [d.strip() for d in first_shape_match.group(1).split(',')]
                    second_dims = [d.strip() for d in second_shape_match.group(1).split(',')]
                    output_dims = [d.strip() for d in output_shape_match.group(1).split(',')]
                    
                    # Heuristic: If output shape is different from element-wise multiplication, use matmul
                    # Element-wise: same shape as inputs
                    # Matmul: different shape (e.g., [b,m,k] @ [k,n] -> [b,m,n])
                    is_likely_matmul = False
                    
                    # Check for batched matmul: [b, m, k] @ [k, n] -> [b, m, n] or [b, m, k] @ [b, k, n] -> [b, m, n]
                    if len(first_dims) == 3 and len(second_dims) == 2 and len(output_dims) == 3:
                        # [b, m, k] @ [k, n] -> [b, m, n]
                        is_likely_matmul = True
                    elif len(first_dims) == 3 and len(second_dims) == 3 and len(output_dims) == 3:
                        # [b, m, k] @ [b, k, n] -> [b, m, n] or element-wise [b, m, k] @ [b, m, k] -> [b, m, k]
                        # If output shape differs from first input, it's likely matmul
                        if output_dims != first_dims:
                            is_likely_matmul = True
                    elif len(first_dims) == 2 and len(second_dims) == 2 and len(output_dims) == 2:
                        # [m, k] @ [k, n] -> [m, n] or element-wise [m, k] @ [m, k] -> [m, k]
                        if output_dims != first_dims:
                            is_likely_matmul = True
                    
                    if is_likely_matmul:
                        dtype = self._extract_dtype(output_type)
                        pypto_dtype = self.dtype_map.get(dtype, "pypto.DT_FP32")
                        return f"{result_var} = pypto.matmul({clean_operands[0]}, {clean_operands[1]}, {pypto_dtype})"
        
        # Map operation to PyPTO
        pypto_op = self.op_map.get(op_name)
        if pypto_op:
            if len(clean_operands) == 1:
                if pypto_op.startswith("pypto."):
                    return f"{result_var} = {pypto_op}({clean_operands[0]})"
                else:
                    return f"{result_var} = {pypto_op}{clean_operands[0]}"
            elif len(clean_operands) == 2:
                return f"{result_var} = {clean_operands[0]} {pypto_op} {clean_operands[1]}"
        
        return f"{result_var} = {op_name}({', '.join(clean_operands)})  # TODO: implement {op_name}"


def convert_ir_to_pypto(ir_file: Path, output_file: Optional[Path] = None) -> str:
    """Convert an IR file to PyPTO code."""
    converter = IRToPyPTOConverter()
    pypto_code = converter.convert_ir_file(ir_file)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(pypto_code)
    
    return pypto_code


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert IR text to PyPTO code")
    parser.add_argument("ir_file", type=str, help="Path to IR file")
    parser.add_argument("-o", "--output", type=str, help="Output file path (optional)")
    
    args = parser.parse_args()
    
    ir_file = Path(args.ir_file)
    output_file = Path(args.output) if args.output else None
    
    pypto_code = convert_ir_to_pypto(ir_file, output_file)
    print(pypto_code)
