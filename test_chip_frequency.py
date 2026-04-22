#!/usr/bin/env python3
import subprocess
import re
import os
import glob
import tempfile
import time


def find_ascend_home():
    """自动检测 CANN 安装路径"""
    for env_var in ['ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_PATH']:
        path = os.environ.get(env_var, '')
        if path and os.path.exists(path):
            return path
    
    for path in glob.glob('/usr/local/Ascend/cann-*'):
        if os.path.exists(path):
            return path
    
    for path in ['/usr/local/Ascend/ascend-toolkit', '/usr/local/Ascend']:
        if os.path.exists(path):
            return path
    
    return None


def find_driver_lib_path():
    """自动检测 Driver 库路径"""
    for path in glob.glob('/usr/local/Ascend/driver/lib*/driver'):
        if os.path.exists(path):
            return path
    
    for path in glob.glob('/usr/local/Ascend/*/driver/lib*/driver'):
        if os.path.exists(path):
            return path
    
    return '/usr/local/Ascend/driver/lib64/driver'


def find_log_dirs():
    """自动检测日志目录，支持 ASCEND_WORK_PATH"""
    log_dirs = []
    
    work_path = os.environ.get('ASCEND_WORK_PATH', '')
    if work_path:
        expanded = os.path.expanduser(work_path)
        candidates = [
            os.path.join(expanded, 'log/run'),
            os.path.join(expanded, 'log'),
            expanded,
        ]
        for path in candidates:
            if os.path.exists(path):
                log_dirs.append(path)
    
    for default_path in ['/root/ascend/log/run', '~/ascend/log/run', '/var/log/ascend/run']:
        expanded = os.path.expanduser(default_path)
        if os.path.exists(expanded) and expanded not in log_dirs:
            log_dirs.append(expanded)
    
    return log_dirs


def get_chip_info():
    """获取芯片信息和预期频率"""
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
    output = result.stdout
    
    chip_freq_map = {
        '910A5': ('910A5', 'A5', 1000),
        '910B3': ('910B3', 'A2/A3', 50),
        '910B2': ('910B2', 'A2/A3', 50),
        '910B1': ('910B1', 'A2/A3', 50),
        '910B4': ('910B4', 'A2/A3', 50),
        '310P1': ('310P1', '310P', 50),
        '310P3': ('310P3', '310P', 50),
        '310B': ('310B', '310B', 50),
    }
    
    for chip_id, info in chip_freq_map.items():
        if chip_id in output:
            return info
    
    return 'Unknown', 'Unknown', None


def get_device_count():
    """获取设备数量"""
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
    count = 0
    for line in result.stdout.split('\n'):
        if re.search(r'\|\s+\d+\s+', line) and ('910' in line or '310' in line):
            count += 1
    return max(count, 1)


def get_aicpu_hardware_freq(device_id, ascend_home, driver_lib_path):
    """通过 CANN DSMI API 测量 AICPU 硬件频率"""
    if not ascend_home:
        return None
    
    include_path = f'{ascend_home}/aarch64-linux/include'
    header_file = f'{include_path}/driver/dsmi_common_interface.h'
    
    if not os.path.exists(header_file):
        return None
    
    c_code = f'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "{header_file}"

int main(int argc, char *argv[]) {{
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    
    DSMI_AICPU_INFO aicpu_info;
    memset(&aicpu_info, 0, sizeof(DSMI_AICPU_INFO));
    int ret = dsmi_get_aicpu_info(device_id, &aicpu_info);
    if (ret == 0) {{
        printf("AICPU_NUM=%u\\n", aicpu_info.aicpuNum);
        printf("AICPU_MAX_FREQ_MHZ=%u\\n", aicpu_info.maxFreq);
        printf("AICPU_CUR_FREQ_MHZ=%u\\n", aicpu_info.curFreq);
        return 0;
    }}
    printf("ERROR=%d\\n", ret);
    return 1;
}}
'''
    
    exe_file = tempfile.mktemp(suffix='_aicpu_freq')
    c_file = tempfile.mktemp(suffix='.c')
    
    try:
        with open(c_file, 'w') as f:
            f.write(c_code)
        
        compile_cmd = [
            'gcc', '-o', exe_file, c_file,
            f'-I{include_path}',
            f'-L{driver_lib_path}',
            '-ldrvdsmi_host',
            f'-Wl,-rpath={driver_lib_path}'
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        result = subprocess.run([exe_file, str(device_id)], capture_output=True, text=True)
        
        freq_info = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                parts = line.split('=')
                key = parts[0]
                value = parts[1]
                freq_info[key] = int(value) if value.isdigit() else value
        
        return freq_info
        
    finally:
        try:
            os.unlink(c_file)
            os.unlink(exe_file)
        except:
            pass


def trigger_npu_and_get_tick_freq(device_id):
    """触发 NPU 操作并从日志读取 tickFreq"""
    
    try:
        import torch
        import torch_npu
        
        torch.npu.set_device(device_id)
        
        print(f"  触发 NPU 操作生成日志...")
        
        a = torch.randn(100, 100, device=f'npu:{device_id}', dtype=torch.float16)
        b = torch.randn(100, 100, device=f'npu:{device_id}', dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.npu.synchronize()
        
        time.sleep(2)
        
        log_dirs = find_log_dirs()
        
        for log_base_dir in log_dirs:
            log_dir = os.path.join(log_base_dir, f'device-{device_id}')
            if not os.path.exists(log_dir):
                continue
            
            log_files = glob.glob(os.path.join(log_dir, '*.log'))
            
            for log_file in sorted(log_files, reverse=True):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        match = re.search(r'tickFreq=(\d+)', content)
                        if match:
                            return int(match.group(1)), log_file
                except:
                    continue
        
        return None, None
        
    except ImportError:
        print(f"  torch_npu 未安装")
        return None, None
    except Exception as e:
        print(f"  NPU 操作失败: {e}")
        return None, None


def read_tick_freq_from_logs(log_dirs, device_id):
    """从日志目录读取 tickFreq"""
    for log_base_dir in log_dirs:
        log_dir = os.path.join(log_base_dir, f'device-{device_id}')
        if not os.path.exists(log_dir):
            continue
        
        log_files = glob.glob(os.path.join(log_dir, '*.log'))
        
        for log_file in sorted(log_files, reverse=True):
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    match = re.search(r'tickFreq=(\d+)', content)
                    if match:
                        return int(match.group(1)), log_file
            except:
                continue
    
    return None, None


def main():
    print("=" * 60)
    print("芯片频率实际测量报告")
    print("=" * 60)
    
    ascend_home = find_ascend_home()
    driver_lib_path = find_driver_lib_path()
    log_dirs = find_log_dirs()
    work_path = os.environ.get('ASCEND_WORK_PATH', '未设置')
    
    print(f"\n环境信息:")
    print(f"  ASCEND_HOME: {ascend_home}")
    print(f"  ASCEND_WORK_PATH: {work_path}")
    print(f"  Driver 库路径: {driver_lib_path}")
    if log_dirs:
        print(f"  日志搜索路径: {log_dirs}")
    else:
        print(f"  日志搜索路径: 无")
    
    chip, platform, expected_freq = get_chip_info()
    device_count = get_device_count()
    
    print(f"\n芯片信息:")
    print(f"  芯片型号: {chip}")
    print(f"  平台类型: {platform}")
    print(f"  设备数量: {device_count}")
    if expected_freq:
        print(f"  预期定时器频率: {expected_freq} MHz")
    
    print("\n" + "-" * 60)
    print("方法1: CANN DSMI API 测量 AICPU 硬件频率")
    print("-" * 60)
    
    for device_id in range(min(device_count, 2)):
        freq_info = get_aicpu_hardware_freq(device_id, ascend_home, driver_lib_path)
        
        if freq_info and 'ERROR' not in freq_info:
            print(f"\n设备 {device_id}:")
            print(f"  AICPU 数量: {freq_info.get('AICPU_NUM', 'N/A')}")
            print(f"  AICPU 硬件频率: {freq_info.get('AICPU_MAX_FREQ_MHZ', 'N/A')} MHz")
            print(f"  AICPU 当前频率: {freq_info.get('AICPU_CUR_FREQ_MHZ', 'N/A')} MHz")
        else:
            print(f"\n设备 {device_id}: API 测量失败")
    
    print("\n" + "-" * 60)
    print("方法2: 设备端日志读取定时器频率")
    print("-" * 60)
    
    for device_id in range(min(device_count, 2)):
        tick_freq, log_file = read_tick_freq_from_logs(log_dirs, device_id)
        
        if tick_freq:
            tick_freq_mhz = tick_freq / 1_000_000
            print(f"\n设备 {device_id}:")
            print(f"  tickFreq: {tick_freq} ({tick_freq_mhz} MHz)")
            print(f"  日志文件: {log_file}")
        else:
            print(f"\n设备 {device_id}: 日志中无 tickFreq")
            tick_freq, log_file = trigger_npu_and_get_tick_freq(device_id)
            if tick_freq:
                tick_freq_mhz = tick_freq / 1_000_000
                print(f"  触发后 tickFreq: {tick_freq} ({tick_freq_mhz} MHz)")
            else:
                print(f"  无法获取 tickFreq")
    
    print("\n" + "=" * 60)
    print("测量结论")
    print("=" * 60)
    
    tick_freq, _ = read_tick_freq_from_logs(log_dirs, 0)
    if not tick_freq:
        tick_freq, _ = trigger_npu_and_get_tick_freq(0)
    
    if tick_freq:
        measured_mhz = tick_freq / 1_000_000
        print(f"\n实际测量的定时器频率: {measured_mhz} MHz")
        
        if expected_freq:
            print(f"文档预期频率 ({platform}): {expected_freq} MHz")
            if measured_mhz == expected_freq:
                print(f"\n✓ 测量结果与文档预期一致")
            else:
                print(f"\n⚠ 测量结果与文档预期不同")
    else:
        print("\n无法获取定时器频率，请检查:")
        print("  1. NPU 设备是否正常")
        print("  2. ASCEND_WORK_PATH 是否正确")
        print("  3. torch_npu 是否已安装")
    
    print("\n" + "=" * 60)
    print("频率对照表")
    print("=" * 60)
    
    print("""
芯片平台    定时器频率    芯片型号
----------------------------------------------
A2/A3       50 MHz       910B1/B2/B3/B4
A5          1000 MHz     910A5
310P        50 MHz       Ascend 310P 系列
310B        50 MHz       Ascend 310B 系列

说明:
  - tickFreq: GetCycles()/GetFreq() 返回的定时器频率
  - AICPU 硬件频率: AICPU 核心的物理运行频率
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
