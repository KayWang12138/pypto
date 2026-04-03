#!/usr/bin/env python3
# coding: utf-8
import os
import datetime
import re
from typing import List, Dict, Any, Optional, Tuple
from ..log import get_logger

logger = get_logger("triton_pypto.pypto_emitter", "TRITON_PYPTO")

class PyPTOCodeEmitter:
    _instance: Optional['PyPTOCodeEmitter'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()
        self.output_dir = "./triton_pypto_dumps"
        self.enabled = os.environ.get("TRITON_PYPTO_DUMP_CODE", "0") == "1"
        
    def reset(self):
        self.operations: List[Dict[str, Any]] = []
        self.tensor_map: Dict[str, str] = {}
        self.tensor_counter: int = 0
        self.function_name: str = "unknown_kernel"
        self.input_tensors: List[str] = []
        self.output_tensors: List[str] = []
        self.input_addrs: List[str] = []
        self.num_programs: Tuple[int, int, int] = (1, 1, 1)
        self.tensor_shapes: Dict[str, List[int]] = {}
        self.tensor_dtypes: Dict[str, str] = {}
        
    def _extract_tensor_addr(self, obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, str):
            match = re.search(r'0x[0-9a-fA-F]+', obj)
            if match:
                return match.group(0)
            return None
        try:
            addr = hex(id(obj))
        except:
            addr = None
        repr_str = repr(obj)
        match = re.search(r'0x[0-9a-fA-F]+', repr_str)
        if match:
            addr = match.group(0)
        return addr
        
    def _extract_tensor_info(self, obj: Any) -> Tuple[Optional[List[int]], Optional[str]]:
        shape = None
        dtype = None
        repr_str = repr(obj)
        shape_match = re.search(r'\[([^\]]+)\]', repr_str)
        if shape_match:
            shape_str = shape_match.group(1)
            try:
                shape = [int(x.strip()) for x in shape_str.split(',')]
            except:
                pass
        dtype_match = re.search(r'(DT_\w+)', repr_str)
        if dtype_match:
            dtype = dtype_match.group(1)
        return shape, dtype
        
    def _get_or_create_var_name(self, obj: Any, prefix: str = "v") -> str:
        addr = self._extract_tensor_addr(obj)
        if addr is None:
            return "None"
        if addr not in self.tensor_map:
            self.tensor_counter += 1
            self.tensor_map[addr] = f"{prefix}{self.tensor_counter}"
        return self.tensor_map[addr]
        
    def _resolve_tensor_name(self, obj: Any) -> str:
        if obj is None:
            return "None"
        if isinstance(obj, str):
            addr = self._extract_tensor_addr(obj)
            if addr and addr in self.tensor_map:
                return self.tensor_map[addr]
            return obj
        obj_repr = repr(obj)
        if "SymbolicScalar" in obj_repr:
            match = re.search(r'SymbolicScalar\((.+)\)', obj_repr)
            if match:
                return match.group(1)
            return obj_repr
        addr = self._extract_tensor_addr(obj)
        if addr is None:
            return str(obj)
        if addr in self.tensor_map:
            return self.tensor_map[addr]
        self.tensor_counter += 1
        name = f"v{self.tensor_counter}"
        self.tensor_map[addr] = name
        return name

    def start_function(self, name: str, tensors: List[Any], num_programs: Optional[Tuple[int, int, int]] = None):
        if not self.enabled:
            return
        self.reset()
        self.function_name = name
        if num_programs is not None:
            self.num_programs = num_programs
        else:
            self.num_programs = (1, 1, 1)
        for i, t in enumerate(tensors):
            addr = self._extract_tensor_addr(t)
            if addr:
                var_name = f"t{i}"
                self.tensor_map[addr] = var_name
                self.input_addrs.append(addr)
                shape, dtype = self._extract_tensor_info(t)
                if shape:
                    self.tensor_shapes[var_name] = shape
                if dtype:
                    self.tensor_dtypes[var_name] = dtype
                if i < len(tensors) - 1:
                    self.input_tensors.append(var_name)
                else:
                    self.output_tensors.append(var_name)
        logger.info(f"[Emitter] Start function: {name}, tensors: {len(tensors)}, num_programs: {self.num_programs}")
        
    def log_operation(self, op_name: str, args: tuple, kwargs: dict, result: Any):
        if not self.enabled:
            return
        if result is not None:
            result_repr = repr(result)
            if "SymbolicScalar" not in result_repr:
                self._get_or_create_var_name(result)
        self.operations.append({
            'op': op_name,
            'args': args,
            'kwargs': kwargs,
            'result': result
        })
        
    def _format_arg(self, arg: Any) -> str:
        if arg is None:
            return "None"
        arg_repr = repr(arg)
        if "SymbolicScalar" in arg_repr:
            match = re.search(r'SymbolicScalar\((.+)\)', arg_repr)
            if match:
                return match.group(1)
        if isinstance(arg, (list, tuple)):
            formatted_items = [self._format_arg(item) for item in arg]
            return f"[{', '.join(formatted_items)}]"
        if hasattr(arg, 'name') and hasattr(arg, '__class__'):
            if 'DataType' in arg.__class__.__name__:
                return f"pypto.DataType.{arg.name}"
        if isinstance(arg, (int, float)):
            return str(arg)
        addr = self._extract_tensor_addr(arg)
        if addr and addr in self.tensor_map:
            return self.tensor_map[addr]
        if addr:
            self.tensor_counter += 1
            name = f"v{self.tensor_counter}"
            self.tensor_map[addr] = name
            return name
        return str(arg)

    def generate_code(self) -> str:
        lines = []
        lines.append("#!/usr/bin/env python3")
        lines.append("# coding: utf-8")
        lines.append(f"# Auto-generated PyPTO code")
        lines.append(f"# Function: {self.function_name}")
        lines.append(f"# Date: {datetime.datetime.now().isoformat()}")
        lines.append("")
        lines.append("import os")
        lines.append("import pypto")
        lines.append("import torch")
        lines.append("import torch_npu")
        lines.append("")
        input_params = ", ".join(self.input_tensors)
        output_params = ", ".join(self.output_tensors)
        all_params = ", ".join(self.input_tensors + self.output_tensors)
        lines.append(f"def run_kernel({all_params}):")
        lines.append(f"    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))")
        lines.append(f"    torch.npu.set_device(device_id)")
        lines.append(f"    pypto.runtime._device_init()")
        lines.append("")
        lines.append(f"    num_programs = {self.num_programs[0]}")
        lines.append("")
        lines.append(f"    with pypto.function(\"{self.function_name}\", {all_params}):")
        lines.append(f"        for dynamic_pid in pypto.loop(0, num_programs, 1, name=\"dynamic_grid\", idx_name=\"dynamic_pid\"):")
        for op in self.operations:
            op_name = op['op']
            indent = "            "
            if op_name == 'set_vec_tile_shapes':
                val = op['args'][0] if op['args'] else 32
                lines.append(f"{indent}pypto.set_vec_tile_shapes({val})")
            elif op_name == 'view':
                if len(op['args']) >= 3:
                    tensor_arg = self._format_arg(op['args'][0])
                    shape_arg = self._format_arg(op['args'][1])
                    offset_arg = self._format_arg(op['args'][2])
                    res_name = self._resolve_tensor_name(op['result'])
                    lines.append(f"{indent}{res_name} = pypto.view({tensor_arg}, {shape_arg}, [{offset_arg}])")
            elif op_name == 'add':
                lhs = self._format_arg(op['args'][0])
                rhs = self._format_arg(op['args'][1])
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.add({lhs}, {rhs})")
            elif op_name == 'abs':
                arg = self._format_arg(op['args'][0])
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.abs({arg})")
            elif op_name == 'amax':
                arg = self._format_arg(op['args'][0])
                axis = op['args'][1] if len(op['args']) > 1 else -1
                keep_dims = op['args'][2] if len(op['args']) > 2 else False
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.amax({arg}, {axis}, {keep_dims})")
            elif op_name == 'maximum':
                lhs = self._format_arg(op['args'][0])
                rhs = self._format_arg(op['args'][1])
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.maximum({lhs}, {rhs})")
            elif op_name == 'div':
                lhs = self._format_arg(op['args'][0])
                rhs = self._format_arg(op['args'][1])
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.div({lhs}, {rhs})")
            elif op_name == 'expand_clone':
                arg = self._format_arg(op['args'][0])
                shape = self._format_arg(op['args'][1]) if len(op['args']) > 1 else "[]"
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.expand_clone({arg}, {shape})")
            elif op_name == 'cast':
                arg = self._format_arg(op['args'][0])
                dtype = self._format_arg(op['args'][1]) if len(op['args']) > 1 else "pypto.DT_FP32"
                res_name = self._resolve_tensor_name(op['result'])
                lines.append(f"{indent}{res_name} = pypto.cast({arg}, {dtype})")
            elif op_name == 'assemble':
                if len(op['args']) >= 3:
                    src = self._format_arg(op['args'][0])
                    offset = self._format_arg(op['args'][1])
                    dst = self._format_arg(op['args'][2])
                    lines.append(f"{indent}pypto.assemble({src}, [{offset}], {dst})")
            else:
                args_str = ", ".join(self._format_arg(a) for a in op['args'])
                if op['result'] and "SymbolicScalar" not in repr(op['result']):
                    res_name = self._resolve_tensor_name(op['result'])
                    lines.append(f"{indent}{res_name} = pypto.{op_name}({args_str})")
                else:
                    lines.append(f"{indent}pypto.{op_name}({args_str})")
        lines.append("")
        lines.append(f"    pypto.runtime._device_fini()")
        lines.append("")
        lines.append(f"if __name__ == \"__main__\":")
        lines.append(f"    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))")
        lines.append(f"    torch.npu.set_device(device_id)")
        lines.append("")
        for var in self.input_tensors + self.output_tensors:
            shape = self.tensor_shapes.get(var, [1024])
            dtype_str = self.tensor_dtypes.get(var, "DT_FP32")
            torch_dtype = "torch.float32"
            if "FP16" in dtype_str:
                torch_dtype = "torch.float16"
            elif "BF16" in dtype_str:
                torch_dtype = "torch.bfloat16"
            elif "INT" in dtype_str:
                torch_dtype = "torch.int32"
            lines.append(f"    {var} = pypto.from_torch(torch.randn({shape}, dtype={torch_dtype}), \"{var}\")")
        lines.append("")
        lines.append(f"    run_kernel({all_params})")
        lines.append("")
        lines.append(f"    print(f\"Kernel completed\")")
        return "\n".join(lines)
        
    def save_to_file(self):
        if not self.enabled or not self.operations:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.function_name}_{timestamp}.py"
        code = self.generate_code()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"[Emitter] Code saved to: {filename}")
        return filename

emitter = PyPTOCodeEmitter()
