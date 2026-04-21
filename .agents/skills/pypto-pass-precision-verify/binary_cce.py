#!/usr/bin/env python3
"""
Binary CCE Precision Debug Script
通过二分打印法定位CCE中哪个Op出现精度问题
"""

import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple


class BinaryCCEDebugger:
    def __init__(self, work_path: str, pypto_root: str = "."):
        self.work_path = Path(work_path)
        self.pypto_root = Path(pypto_root)
        self.output_dir = self.work_path / "output"
        self.log_dir = self.work_path / "log" / "debug"
        
        self.config_file = self.pypto_root / "framework/src/interface/configs/tile_fwk_config.json"
        self.print_header = self.pypto_root / "framework/src/interface/machine/device/tilefwk/aicore_print.h"
        
        self.cce_files: List[Path] = []
        
    def check_env(self) -> bool:
        """检查必要的文件和目录"""
        if not self.work_path.exists():
            print(f"错误: 工作目录不存在: {self.work_path}")
            return False
        return True
    
    def enable_print_switch(self):
        """开启打印开关"""
        print("=== 开启打印开关 ===")
        
        # 1. 修改 tile_fwk_config.json
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # 修改 global.codegen 配置
                if 'global' not in config:
                    config['global'] = {}
                if 'codegen' not in config['global']:
                    config['global']['codegen'] = {}
                codegen = config['global']['codegen']
                
                # 修改现有配置
                codegen['fixed_output_path'] = True
                codegen['force_overwrite'] = False
                codegen['parallel_compile'] = 1
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  [√] 已修改: {self.config_file}")
            except Exception as e:
                print(f"  [×] 修改配置失败: {e}")
        
        # 2. 修改 aicore_print.h
        if self.print_header.exists():
            try:
                content = self.print_header.read_text()
                if '#define ENABLE_AICORE_PRINT 0' in content:
                    content = content.replace('#define ENABLE_AICORE_PRINT 0', '#define ENABLE_AICORE_PRINT 1')
                    self.print_header.write_text(content)
                    print(f"  [√] 已修改: {self.print_header}")
                elif 'ENABLE_AICORE_PRINT 1' in content:
                    print(f"  [√] 已开启: {self.print_header}")
            except Exception as e:
                print(f"  [×] 修改头文件失败: {e}")
        print("")
        
    def rebuild_pypto(self):
        """重新编译pypto"""
        print("=== 重新编译 PyPTO ===")
        os.chdir(self.pypto_root)
        print(f"  执行: python3 -m pip install . -v")
        result = subprocess.run(
            ["python3", "-m", "pip", "install", ".", "-v"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  [√] 编译成功\n")
        else:
            print(f"  [×] 编译失败")
            print(f"  stderr: {result.stderr[:500]}")
        return result.returncode == 0
    
    def find_cce_files(self) -> List[Path]:
        """查找CCE文件（支持 .cce 和 .cpp 格式）"""
        kernel_dir = self.work_path / "kernel_aicore"
        if kernel_dir.exists():
            self.cce_files = sorted(kernel_dir.glob("*.cpp"), key=lambda x: x.stat().st_mtime)
        else:
            # 查找 .cce 文件
            cce_files = list(self.output_dir.rglob("*.cce"))
            # 查找 .cpp 文件（AICore kernel）
            cpp_files = list(self.output_dir.rglob("*_aiv.cpp"))
            # 合并并排序
            self.cce_files = sorted(cce_files + cpp_files, key=lambda x: x.stat().st_mtime)
        
        print(f"  找到 {len(self.cce_files)} 个CCE文件:")
        for i, f in enumerate(self.cce_files[:20]):
            print(f"    [{i}] {f.name}")
        if len(self.cce_files) > 20:
            print(f"    ... 共 {len(self.cce_files)} 个")
        return self.cce_files
    
    def find_cce_files_in_range(self, start_idx: int, end_idx: int) -> List[Path]:
        """获取指定范围内的CCE文件"""
        cce_files = self.find_cce_files()
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(cce_files):
            end_idx = len(cce_files) - 1
        return cce_files[start_idx:end_idx + 1]
    
    def find_cce_by_name_pattern(self, pattern: str) -> List[Path]:
        """根据名称模式查找CCE文件"""
        cce_files = self.find_cce_files()
        matched = []
        for cce in cce_files:
            if pattern.lower() in cce.name.lower():
                matched.append(cce)
        return matched
    
    def binary_search_by_range(self, start_idx: int, end_idx: int, step: int = 0) -> int:
        """
        在指定范围内二分查找问题CCE
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            step: 可选的步长，默认取中间值
            
        Returns:
            中间位置的CCE索引
        """
        if start_idx > end_idx:
            return -1
        if step == 0:
            mid = (start_idx + end_idx) // 2
        else:
            mid = start_idx + step
        return mid
    
    def find_kernel_functions(self, cce_content: str) -> List[Dict]:
        """解析CCE中的kernel函数结构"""
        kernels = []
        
        # 匹配 kernel 函数定义
        # 格式: [aicore] void NAME(CoreFuncParam *param, ...)
        pattern = r'\[aicore\]\s+void\s+(\w+)\s*\((.*?)\)\s*\{'
        
        for match in re.finditer(pattern, cce_content):
            func_name = match.group(1)
            params = match.group(2)
            start_pos = match.start()
            
            # 找到函数结束位置（简单的括号匹配）
            brace_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(cce_content)):
                if cce_content[i] == '{':
                    brace_count += 1
                elif cce_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            kernels.append({
                'name': func_name,
                'start': start_pos,
                'end': end_pos,
                'params': params
            })
        
        return kernels
    
    def parse_cce_structure(self, cce_file: Path) -> Dict:
        """解析CCE文件结构"""
        content = cce_file.read_text()
        
        # 找到所有GM/UB tensor声明
        gm_tensors = re.findall(r'__gm__\s+\w+\s+(\w+)\s*=', content)
        ub_tensors = re.findall(r'__ub__\s+\w+\s+(\w+)\s*=', content)
        
        # 找到shape变量
        shape_vars = self.parse_shape_variables(content)
        
        # 找到kernel函数
        kernels = self.find_kernel_functions(content)
        
        return {
            'gm_tensors': gm_tensors,
            'ub_tensors': ub_tensors,
            'shape_vars': shape_vars,
            'kernels': kernels,
            'content': content
        }
    
    def parse_shape_variables(self, content: str) -> List[str]:
        """解析 CCE 中的 shape 变量
        
        识别格式如:
        - int64_t sym_15_dim_0 = ...;
        - int64_t sym_15_dim_1 = ...;
        """
        # 匹配 shape 维度变量
        pattern = r'int64_t\s+(\w+_dim_\d+)\s*='
        shape_vars = re.findall(pattern, content)
        return shape_vars
    
    def add_shape_print(self, cce_file: Path, shape_vars: List[str]):
        """添加 shape 打印语句到 CCE 文件"""
        content = cce_file.read_text()
        cce_info = self.parse_cce_structure(cce_file)
        
        # 添加头文件
        if '#include "tilefwk/aicore_print.h"' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('#include') and 'aicore_print' not in line:
                    lines.insert(i + 1, '#include "tilefwk/aicore_print.h"')
                    break
            content = '\n'.join(lines)
        
        # 如果没有指定 shape 变量，使用解析到的
        if not shape_vars:
            shape_vars = cce_info['shape_vars']
        
        if not shape_vars:
            print("  警告: 未找到 shape 变量")
            return
        
        # 准备打印语句
        print_stmts = []
        
        # 按 dim 分组
        dim_groups = {}
        for var in shape_vars:
            # 提取基础名称和维度
            match = re.match(r'(\w+)_dim_(\d+)', var)
            if match:
                base_name = match.group(1)
                dim_num = match.group(2)
                if base_name not in dim_groups:
                    dim_groups[base_name] = []
                dim_groups[base_name].append((int(dim_num), var))
        
        # 为每个 shape 组生成打印语句
        for base_name, dims in dim_groups.items():
            dims.sort()  # 按维度排序
            if len(dims) == 1:
                # 单维度 - 使用 Coord1Dim
                var_name = dims[0][1]
                print_stmts.append(f'AiCorePrintShape(param->ctx, Coord1Dim({var_name}));')
            elif len(dims) == 2:
                # 二维度
                var1 = dims[0][1]
                var2 = dims[1][1]
                print_stmts.append(f'AiCorePrintShape(param->ctx, Shape2Dim({var1}, {var2}));')
            elif len(dims) == 3:
                # 三维度
                var1 = dims[0][1]
                var2 = dims[1][1]
                var3 = dims[2][1]
                print_stmts.append(f'AiCorePrintShape(param->ctx, Shape3Dim({var1}, {var2}, {var3}));')
            elif len(dims) == 4:
                # 四维度
                var1 = dims[0][1]
                var2 = dims[1][1]
                var3 = dims[2][1]
                var4 = dims[3][1]
                print_stmts.append(f'AiCorePrintShape(param->ctx, Shape4Dim({var1}, {var2}, {var3}, {var4}));')
        
        if not print_stmts:
            print("  警告: 无法生成 shape 打印语句")
            return
        
        print_code = "\n".join([f"    {s} // DEBUG SHAPE" for s in print_stmts])
        
        # 插入到第一个 kernel 函数开头
        if cce_info['kernels']:
            kernel = cce_info['kernels'][0]
            first_brace = content.find('{', kernel['start'])
            if first_brace != -1:
                # 在第一个有效语句位置插入
                pos = first_brace + 1
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
                content = content[:pos] + "\n" + print_code + "\n" + content[pos:]
                
                cce_file.write_text(content)
                print(f"  已添加 {len(print_stmts)} 条 shape 打印语句")
        else:
            print("  警告: 未找到 kernel 函数")
    
    def add_print_to_cce(self, cce_file: Path, tensor_names: List[str], 
                         print_type: str = "GM", dtype: str = "float", 
                         end_offset: int = 63, start_offset: int = 0, 
                         insert_pos: str = "kernel_start"):
        """
        添加打印语句到CCE
        
        Args:
            dtype: 数据类型（float/bfloat16_t/half/int32_t）
            end_offset: 打印末尾偏移量（含）
            start_offset: 打印起始偏移量
            元素数量 = end_offset - start_offset + 1
        
        insert_pos 选项:
        - kernel_start: kernel函数开头
        - kernel_end: kernel函数结尾  
        - tensor_after: 在指定tensor声明之后
        """
        # 检查元素数量限制（元素数量 = end_offset - start_offset + 1）
        element_count = end_offset - start_offset + 1
        if element_count > 80:
            print(f"  警告: 元素数量 {element_count} > 80，调整偏移量范围")
            end_offset = start_offset + 79  # 最多80个元素
            
        content = cce_file.read_text()
        cce_info = self.parse_cce_structure(cce_file)
        
        # 添加头文件
        if '#include "tilefwk/aicore_print.h"' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('#include') and 'aicore_print' not in line:
                    lines.insert(i + 1, '#include "tilefwk/aicore_print.h"')
                    break
            content = '\n'.join(lines)
        
        # 准备打印语句
        print_func = "AiCorePrintGmTensor" if print_type == "GM" else "AiCorePrintUbTensor"
        tensor_type = "__gm__" if print_type == "GM" else "__ub__"
        
        tensor_list = tensor_names if tensor_names else (cce_info['gm_tensors'] if print_type == "GM" else cce_info['ub_tensors'])
        
        print_stmts = []
        for tensor_name in tensor_list:
            if tensor_name in content:
                print_stmts.append(f'{print_func}(param->ctx, ({tensor_type}{dtype}*){tensor_name}.GetAddr(), {end_offset}, {start_offset});')
        
        if not print_stmts:
            print(f"  警告: 未找到tensor {tensor_list}")
            return
        
        print_code = "\n".join([f"    {s} // DEBUG" for s in print_stmts])
        
        # 根据insert_pos选择插入位置
        if insert_pos == "kernel_start" and cce_info['kernels']:
            # 插入到第一个kernel函数的开头（在 { 之后）
            kernel = cce_info['kernels'][0]
            # 找到第一个 {
            first_brace = content.find('{', kernel['start'])
            if first_brace != -1:
                # 在第一个有效语句位置插入（跳过空行和注释）
                pos = first_brace + 1
                # 跳过空行
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
                content = content[:pos] + "\n" + print_code + "\n" + content[pos:]
                
        elif insert_pos == "kernel_end" and cce_info['kernels']:
            # 插入到第一个kernel函数的结尾（在 } 之前）
            kernel = cce_info['kernels'][0]
            content = content[:kernel['end']] + "\n" + print_code + "\n" + content[kernel['end']:]
            
        elif insert_pos == "tensor_after" and tensor_list:
            # 在第一个tensor声明之后插入
            first_tensor = tensor_list[0]
            tensor_pos = content.find(f"= {first_tensor}")
            if tensor_pos == -1:
                tensor_pos = content.find(first_tensor)
            if tensor_pos != -1:
                # 找到这行的结束
                line_end = content.find('\n', tensor_pos)
                if line_end != -1:
                    content = content[:line_end+1] + "    " + print_code + "\n" + content[line_end+1:]
        
        cce_file.write_text(content)
        print(f"  已添加 {len(print_stmts)} 条打印语句")
    
    def run_test(self, test_cmd: List[str]) -> Tuple[int, Optional[Path]]:
        """运行测试"""
        env = os.environ.copy()
        env['ASCEND_WORK_PATH'] = str(self.work_path)
        env['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
        
        print(f"  运行: {' '.join(test_cmd)}")
        result = subprocess.run(test_cmd, capture_output=True, text=True, env=env)
        
        # 查找日志
        log_files = list(self.log_dir.rglob("DumpAicoreLog*"))
        log_file: Optional[Path] = log_files[0] if log_files else None
        
        return result.returncode, log_file
    
    def parse_log(self, log_file: Path) -> Dict:
        """解析日志获取打印数据"""
        if not log_file or not log_file.exists():
            return {}
        
        content = log_file.read_text(errors='ignore')
        data = {
            'tensor_data': [],
            'shape_data': [],
            'other_logs': []
        }
        
        # 解析打印数据
        lines = content.split('\n')
        for line in lines:
            if 'tensor data, range=' in line:
                data['tensor_data'].append(line)
            elif 'shape' in line and ('dim' in line or '[' in line):
                data['shape_data'].append(line)
            elif 'AiCorePrintShape' in line or 'AiCoreLogF' in line or 'DumpAicoreLog' in line:
                data['other_logs'].append(line)
        
        return data
    
    def detect_validshape_issues(self, ir_file: Path) -> Dict:
        """检测 validshape 问题
        
        Args:
            ir_file: IR 文件路径
            
        Returns:
            检测结果字典
        """
        if not ir_file or not ir_file.exists():
            return {'error': 'IR 文件不存在'}
        
        # 使用 get_op_info.py 脚本解析 IR 文件
        script_path = self.pypto_root / '.agents/skills/pypto-pass-error-locator/scripts/get_op_info.py'
        
        if not script_path.exists():
            return {'error': 'get_op_info.py 脚本不存在'}
        
        try:
            # 获取所有操作列表
            result = subprocess.run(
                ['python3', str(script_path), '--ir-file', str(ir_file), '--list-ops'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'error': f'解析 IR 文件失败: {result.stderr}'}
            
            # 解析操作列表
            ops_info = []
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('OP Magic:'):
                    # 提取操作信息
                    match = re.search(r'OP Magic: (\d+), Opcode: (\w+), Line: (\d+)', line)
                    if match:
                        op_magic = int(match.group(1))
                        opcode = match.group(2)
                        line_num = int(match.group(3))
                        ops_info.append({
                            'op_magic': op_magic,
                            'opcode': opcode,
                            'line': line_num
                        })
            
            # 检查每个操作的 shape 和 validshape
            issues = []
            for op_info in ops_info:
                result = subprocess.run(
                    ['python3', str(script_path), '--ir-file', str(ir_file), '--op-magic', str(op_info['op_magic']), '--format', 'json'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    try:
                        op_data = json.loads(result.stdout)
                        
                        # 检查 output shape 和 validshape
                        output_shape = op_data.get('output', {}).get('shape', [])
                        output_valid_shape = op_data.get('output', {}).get('valid_shape', [])
                        
                        # 检查输入 tensors 的 shape 和 validshape
                        input_issues = []
                        for input_tensor in op_data.get('inputs', []):
                            input_shape = input_tensor.get('shape', [])
                            input_valid_shape = input_tensor.get('valid_shape', [])
                            
                            if input_shape != input_valid_shape and input_shape and input_valid_shape:
                                input_issues.append({
                                    'tensor_id': input_tensor.get('logic_id'),
                                    'shape': input_shape,
                                    'valid_shape': input_valid_shape,
                                    'issue': 'shape != valid_shape'
                                })
                        
                        # 检查 offset 信息
                        offset_info = op_data.get('offset_info', {})
                        offset = offset_info.get('offset', [])
                        dynoffset = offset_info.get('dynoffset', [])
                        dynvalidshape = offset_info.get('dynvalidshape', [])
                        
                        # 如果有 offset/dynoffset 但没有 dynvalidshape，可能有问题
                        offset_issues = []
                        if (offset or dynoffset) and not dynvalidshape:
                            offset_issues.append({
                                'issue': 'offset/dynoffset 存在但 dynvalidshape 缺失',
                                'offset': offset,
                                'dynoffset': dynoffset
                            })
                        
                        if input_issues or offset_issues:
                            issues.append({
                                'op_magic': op_info['op_magic'],
                                'opcode': op_info['opcode'],
                                'line': op_info['line'],
                                'input_issues': input_issues,
                                'offset_issues': offset_issues
                            })
                    
                    except json.JSONDecodeError:
                        pass
            
            return {
                'total_ops': len(ops_info),
                'issues_count': len(issues),
                'issues': issues
            }
        
        except Exception as e:
            return {'error': f'检测过程出错: {str(e)}'}
    
    def generate_validshape_report(self, ir_file: Path) -> str:
        """生成 validshape 问题检测报告"""
       
        
        result = self.detect_validshape_issues(ir_file)
        
        if 'error' in result:
            return f"错误: {result['error']}"
        
        report_lines = []
        report_lines.append("=== ValidShape 问题检测报告 ===")
        report_lines.append(f"文件: {ir_file.name}")
        report_lines.append(f"总操作数: {result['total_ops']}")
        report_lines.append(f"发现问题数: {result['issues_count']}")
        report_lines.append("")
        
        if result['issues_count'] == 0:
            report_lines.append("✅ 未发现 validshape 问题")
        else:
            for issue in result['issues']:
                report_lines.append(f"### 操作 {issue['op_magic']} ({issue['opcode']}) - 第 {issue['line']} 行")
                
                if issue['input_issues']:
                    report_lines.append("  输入 Tensor 问题:")
                    for input_issue in issue['input_issues']:
                        report_lines.append(f"    Tensor ID: {input_issue['tensor_id']}")
                        report_lines.append(f"    Shape: {input_issue['shape']}")
                        report_lines.append(f"    ValidShape: {input_issue['valid_shape']}")
                        report_lines.append(f"    问题: {input_issue['issue']}")
                        report_lines.append(f"    建议: 检查 offset/dynoffset 配置是否正确")
                
                if issue['offset_issues']:
                    report_lines.append("  Offset 问题:")
                    for offset_issue in issue['offset_issues']:
                        report_lines.append(f"    问题: {offset_issue['issue']}")
                        report_lines.append(f"    Offset: {offset_issue['offset']}")
                        report_lines.append(f"    Dynoffset: {offset_issue['dynoffset']}")
                        report_lines.append(f"    建议: 检查是否需要添加 dynvalidshape")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Binary CCE Precision Debugger - 二分打印定位CCE中错误Op",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 初始化配置（只需执行一次）
  python3 binary_cce.py --init --work-path /path/to/work
 
  # 列出CCE文件信息
  python3 binary_cce.py --work-path /path/to/work --list-cce
 
  # GM数据打印（默认偏移量0~63，共64个元素）
  python3 binary_cce.py --work-path /path/to/work --print-idx 0 --pos kernel_start
  
  # GM数据打印（指定偏移量范围）
  python3 binary_cce.py --work-path /path/to/work --print-idx 0 --end-offset 79 --start-offset 0
  
  # GM数据打印（指定dtype）
  python3 binary_cce.py --work-path /path/to/work --print-idx 0 --dtype bfloat16_t
  
  # UB数据打印
  python3 binary_cce.py --work-path /path/to/work --print-idx 0 --print-type UB
 
  # Shape打印（用于诊断 validshape 问题）
  python3 binary_cce.py --work-path /path/to/work --print-idx 0 --print-shape sym_15_dim_0,sym_15_dim_1

  # 检测 validshape 问题
  python3 binary_cce.py --work-path /path/to/work --ir-file path/to/ir_file.tifwkgr

四种打印方法：
  GM数据打印：--print-type GM（打印DDR/GM上的tensor数据，最常用）
  UB数据打印：--print-type UB（打印UB上的tensor数据）
  Shape打印：--print-shape（打印shape调试信息，使用 AiCorePrintShape）
  Offset打印：手动添加（打印offset调试信息，使用 AiCorePrintShape + Coord2Dim）

偏移量参数说明：
  --end-offset: 打印末尾偏移量（默认63）
  --start-offset: 打印起始偏移量（默认0）
  元素数量 = end-offset - start-offset + 1（必须≤80）
        """
    )
    
    parser.add_argument("--work-path", required=True, help="ASCEND_WORK_PATH 工作目录")
    parser.add_argument("--pypto-root", default=".", help="PyPTO源码根目录")
    parser.add_argument("--init", action="store_true", help="仅初始化配置开关")
    parser.add_argument("--list-cce", action="store_true", help="仅列出CCE文件信息")
    parser.add_argument("--print-idx", type=int, help="指定打印哪个CCE(0-based index)")
    parser.add_argument("--tensor", help="指定要打印的tensor名称（多个用逗号分隔）")
    parser.add_argument("--print-type", choices=["GM", "UB"], default="GM", help="打印类型(GM/UB)")
    parser.add_argument("--dtype", choices=["float", "bfloat16_t", "half", "int32_t"], 
                        default="float", help="打印的数据类型（默认float）")
    parser.add_argument("--end-offset", type=int, default=63, 
                        help="打印末尾偏移量（默认63）")
    parser.add_argument("--start-offset", type=int, default=0, 
                        help="打印起始偏移量（默认0，元素数量=末尾-起始+1）")
    parser.add_argument("--pos", choices=["kernel_start", "kernel_end", "tensor_after"], 
                       default="kernel_start", help="打印语句插入位置")
    parser.add_argument("--rebuild", action="store_true", help="是否重新编译")
    parser.add_argument("--print-shape", help="打印 shape 变量（多个用逗号分隔）")
    parser.add_argument("--check-validshape", help="检测 validshape 问题（指定 IR 文件路径）")
    parser.add_argument("--ir-file", help="指定 IR 文件路径（用于 validshape 检测）")
    
    args = parser.parse_args()
    
    debugger = BinaryCCEDebugger(args.work_path, args.pypto_root)
    
    if not debugger.check_env():
        return 1
    
    # 初始化配置
    debugger.enable_print_switch()
    
    if args.init:
        if args.rebuild:
            debugger.rebuild_pypto()
        print("配置初始化完成")
        return 0
    
    # 列出CCE信息
    if args.list_cce:
        debugger.find_cce_files()
        for i, cce in enumerate(debugger.cce_files):
            info = debugger.parse_cce_structure(cce)
            print(f"\n[{i}] {cce.name}")
            print(f"  GM tensors: {info['gm_tensors']}")
            print(f"  UB tensors: {info['ub_tensors']}")
            print(f"  Shape variables: {info['shape_vars']}")
            print(f"  Kernels: {[k['name'] for k in info['kernels']]}")
        return 0
    
    # 打印 shape 变量
    if args.print_shape is not None:
        cce_files = debugger.find_cce_files()
        if args.print_idx is not None:
            if args.print_idx >= len(cce_files):
                print(f"错误: CCE索引 {args.print_idx} 超出范围(0-{len(cce_files)-1})")
                return 1
            cce_files = [cce_files[args.print_idx]]
        
        for cce_file in cce_files:
            print(f"\n=== 添加 Shape 打印: {cce_file.name} ===")
            
            # 备份
            backup = cce_file.with_suffix('.cpp.bak')
            shutil.copy(cce_file, backup)
            
            # 解析 shape 变量
            shape_vars = [s.strip() for s in args.print_shape.split(',')] if args.print_shape else []
            
            # 添加 shape 打印
            debugger.add_shape_print(cce_file, shape_vars)
            
            print(f"  请运行测试后查看日志:")
            print(f"    {args.work_path}/log/debug/device-*/DumpAicoreLog*")
        
        return 0
    
    # 指定CCE打印
    if args.print_idx is not None:
        cce_files = debugger.find_cce_files()
        if args.print_idx >= len(cce_files):
            print(f"错误: CCE索引 {args.print_idx} 超出范围(0-{len(cce_files)-1})")
            return 1
        
        cce_file = cce_files[args.print_idx]
        print(f"\n=== 打印 CCE[{args.print_idx}]: {cce_file.name} ===")
        
        # 解析结构
        info = debugger.parse_cce_structure(cce_file)
        print(f"  GM tensors: {info['gm_tensors']}")
        print(f"  UB tensors: {info['ub_tensors']}")
        
        # 备份
        backup = cce_file.with_suffix('.cce.bak')
        shutil.copy(cce_file, backup)
        
        # 解析tensor参数
        tensors = []
        if args.tensor:
            tensors = [t.strip() for t in args.tensor.split(',')]
        
        # 添加打印
        debugger.add_print_to_cce(cce_file, tensors, args.print_type, 
                                  args.dtype, args.end_offset, args.start_offset, args.pos)
        
        element_count = args.end_offset - args.start_offset + 1
        print(f"\n  打印类型: {args.print_type}")
        print(f"  数据类型: {args.dtype}")
        print(f"  偏移量范围: {args.start_offset} ~ {args.end_offset}（共{element_count}个元素）")
        print(f"  插入位置: {args.pos}")
        print(f"  请运行测试后查看日志:")
        print(f"    {args.work_path}/log/debug/device-*/DumpAicoreLog*")
        
        # 恢复原文件（可选）
        # shutil.copy(backup, cce_file)
        return 0
    
    # 检测 validshape 问题
    if args.check_validshape or args.ir_file:
        ir_file = Path(args.check_validshape or args.ir_file)
        if not ir_file.exists():
            print(f"错误: IR 文件不存在: {ir_file}")
            return 1
        
        report = debugger.generate_validshape_report(ir_file)
        print(report)
        return 0
    
    parser.print_help()


if __name__ == "__main__":
    exit(main() or 0)