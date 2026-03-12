#!/usr/bin/env python
import re
import os
import glob

def extract_pass_logs(log_file_paths, output_dir):
    """
    提取日志文件中的每个pass部分并保存到单独的文件中
    支持多个输入文件/处理跨文件的pass日志
    
    Args:
        log_file_paths: 输入日志文件路径列表
        output_dir: 输出目录文件路径
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 正则表达式匹配pass开始行
    pass_pattern = re.compile(r'Apply pass <([^>]+)> on function: ([^.]+)\.')
    
    current_pass_lines = []
    current_pass_name = None
    current_function_name = None
    pass_count = 0
    
    # 用于跟踪每个function_name的序号
    function_index_map = {}
    
    # 处理每个日志文件
    for log_file_path in log_file_paths:
        print(f"正在处理文件: {log_file_path}")
        
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            # 检查是否是pass开始行
            match = pass_pattern.search(line)
            
            if match:
                # 如果有当前pass的日志，先保存
                if current_pass_name is not None:
                    save_pass_log(current_pass_lines, current_pass_name, current_function_name, output_dir, function_index_map)
                    pass_count += 1
                
                # 开始新的pass
                current_pass_name = match.group(1)
                current_function_name = match.group(2)
                current_pass_lines = [line]
            else:
                # 添加到当前pass的日志
                if current_pass_name is not None:
                    current_pass_lines.append(line)
    
    # 保存最后一个pass
    if current_pass_name is not None:
        save_pass_log(current_pass_lines, current_pass_name, current_function_name, output_dir, function_index_map)
        pass_count += 1
    
    print(f"提取完成！共提取 {pass_count} 个pass日志到 {output_dir} 目录")

def save_pass_log(pass_lines, pass_name, function_name, output_dir, function_index_map):
    """
    保存单个pass的日志到文件
    
    Args:
        pass_lines: pass日志行列表
        pass_name: pass名称
        function_name: function名称
        output_dir: 输出目录
        function_index_map: 用于跟踪每个function_name的序号字典
    """
    # 获取或初始化当前function_name的序号
    if function_name not in function_index_map:
        function_index_map[function_name] = 0
    
    # 生成文件名，格式为Pass_{xx}_{pass_name}_{function_name}.log
    index = function_index_map[function_name]
    filename = f"Pass_{index:02d}_{pass_name}_{function_name}.log"

    # 创建子目录，格式为Pass_{xx}_{pass_name}
    subdir_name = f"Pass_{index:02d}_{pass_name}"
    subdir_path = os.path.join(output_dir, subdir_name)
    os.makedirs(subdir_path, exist_ok=True)
    
    # 文件完整路径
    filepath = os.path.join(subdir_path, filename)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(pass_lines)
    
    print(f"保存: {subdir_name}/{filename} ({len(pass_lines)} 行)")
    
    # 序号自增
    function_index_map[function_name] += 1

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python3 extract_pass_logs.py <日志文件路径1> [日志文件路径2 ...] [输出目录]")
        print("示例: python3 extract_pass_logs.py pypto-log-xxx.log")
        print("示例: python3 extract_pass_logs.py pypto-log-xxx.log ./output")
        print("示例: python3 extract_pass_logs.py log1.log log2.log log3.log ./output")
        print("示例: python3 extract_pass_logs.py 'log*.log' ./output")
        sys.exit(1)
    
    # 检查最后一个参数是否是输出目录
    if len(sys.argv) >= 3 and (sys.argv[-1].endswith('/') or not sys.argv[-1].endswith('.log')):
        # 最后一个参数是输出目录
        log_patterns = sys.argv[1:-1]
        output_directory = sys.argv[-1]
    else:
        # 所有参数都是日志文件，使用默认输出目录
        log_patterns = sys.argv[1:]
        output_directory = "pass_logs"
    
    # 展开通配符
    log_files = []
    for pattern in log_patterns:
        matched_files = glob.glob(pattern)
        if matched_files:
            log_files.extend(matched_files)
        else:
            print(f"警告: 没有找到匹配 '{pattern}' 的文件")
    
    if not log_files:
        print("错误: 没有找到任何日志文件")
        sys.exit(1)
    
    # 按文件名排序，确保按顺序处理
    log_files.sort()
    
    print(f"找到 {len(log_files)} 个日志文件:")
    for f in log_files:
        print(f"  - {f}")
    
    # 提取pass日志
    extract_pass_logs(log_files, output_directory)
