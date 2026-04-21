#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能流水数据采集脚本
自动化采集 PyPTO 算子的性能流水数据
"""

import json
import os
import sys
import subprocess
import glob
from pathlib import Path


class PerfDataCollector:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.config_file = self.project_root / "framework/src/interface/configs/tile_fwk_config.json"
        self.default_exec_cmd = "python models/glm_v4_5/glm_moe_fusion.py"
        self.current_exec_cmd = self.default_exec_cmd
        self.kernel_aicore_dir = self.project_root / "kernel_aicore"
        
    def modify_config(self):
        """步骤1: 修改配置文件"""
        print("=" * 60)
        print("步骤1: 修改配置文件")
        print("=" * 60)
        
        if not self.config_file.exists():
            print(f"错误: 配置文件不存在: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 修改配置项
            if 'global' in config and 'codegen' in config['global']:
                config['global']['codegen']['fixed_output_path'] = True
                config['global']['codegen']['force_overwrite'] = False
                config['global']['codegen']['parallel_compile'] = 1
                
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                
                print(f"✓ 配置文件已更新:")
                print(f"  - fixed_output_path: True")
                print(f"  - force_overwrite: False")
                print(f"  - parallel_compile: 1")
                return True
            else:
                print("错误: 配置文件格式不正确，缺少 global.codegen 配置项")
                return False
                
        except Exception as e:
            print(f"错误: 修改配置文件失败: {e}")
            return False
    
    def build_wheel(self):
        """步骤2: 编译安装 whl 包"""
        print("\n" + "=" * 60)
        print("步骤2: 编译安装 whl 包")
        print("=" * 60)
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", ".", "--verbose"]
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✓ whl 包编译安装成功")
                return True
            else:
                print(f"✗ whl 包编译安装失败，返回码: {result.returncode}")
                return False
        except Exception as e:
            print(f"错误: 编译安装失败: {e}")
            return False
    
    def run_program(self):
        """步骤3: 运行可执行程序"""
        print("\n" + "=" * 60)
        print("步骤3: 运行可执行程序")
        print("=" * 60)
        
        print(f"默认执行命令: {self.default_exec_cmd}")
        user_input = input("请输入执行命令（直接回车使用默认命令）: ").strip()
        
        if user_input:
            self.current_exec_cmd = user_input
            self.default_exec_cmd = user_input
            print(f"已更新默认执行命令为: {self.current_exec_cmd}")
        else:
            print(f"使用默认命令: {self.current_exec_cmd}")
        
        try:
            print(f"\n开始执行: {self.current_exec_cmd}")
            result = subprocess.run(self.current_exec_cmd, shell=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✓ 程序执行成功")
                return True
            else:
                print(f"⚠ 程序执行返回非零状态码: {result.returncode}")
                print("继续执行后续步骤...")
                return True
        except Exception as e:
            print(f"错误: 程序执行失败: {e}")
            return False
    
    def modify_cpp_files(self):
        """步骤4: 修改 kernel_aicore 目录下的所有 cpp 文件"""
        print("\n" + "=" * 60)
        print("步骤4: 修改 kernel_aicore 下的 cpp 文件")
        print("=" * 60)
        
        if not self.kernel_aicore_dir.exists():
            print(f"错误: kernel_aicore 目录不存在: {self.kernel_aicore_dir}")
            return False
        
        # 要插入的代码
        insert_code = """
    // 通过插nop， 增大算子执行时间
    asm volatile("bar.all");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
    asm volatile("nop");
"""
        
        cpp_files = list(self.kernel_aicore_dir.glob("**/*.cpp"))
        
        if not cpp_files:
            print("警告: 未找到任何 cpp 文件")
            return False
        
        print(f"找到 {len(cpp_files)} 个 cpp 文件")
        
        success_count = 0
        for cpp_file in cpp_files:
            try:
                with open(cpp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否已经包含插入的代码
                if '通过插nop， 增大算子执行时间' in content:
                    print(f"  跳过 {cpp_file.name} (已包含插入代码)")
                    continue
                
                # 找到最后一个 } 的位置
                last_brace_pos = content.rfind('}')
                
                if last_brace_pos == -1:
                    print(f"  警告: {cpp_file.name} 中未找到 '}}'")
                    continue
                
                # 在最后一个 } 之前插入代码
                new_content = content[:last_brace_pos] + insert_code + content[last_brace_pos:]
                
                with open(cpp_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  ✓ 已修改: {cpp_file.name}")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ 修改失败 {cpp_file.name}: {e}")
        
        print(f"\n共修改 {success_count}/{len(cpp_files)} 个 cpp 文件")
        return success_count > 0
    
    def run_msprof(self):
        """步骤5: 使用 msprof 执行程序"""
        print("\n" + "=" * 60)
        print("步骤5: 使用 msprof 执行性能采集")
        print("=" * 60)
        
        msprof_cmd = f"msprof --instr-profiling=on {self.current_exec_cmd}"
        print(f"执行命令: {msprof_cmd}")
        
        try:
            result = subprocess.run(msprof_cmd, shell=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✓ msprof 性能采集成功")
                return True
            else:
                print(f"✗ msprof 执行失败，返回码: {result.returncode}")
                return False
        except Exception as e:
            print(f"错误: msprof 执行失败: {e}")
            return False
    
    def run(self):
        """执行完整的采集流程"""
        print("=" * 60)
        print("性能流水数据采集脚本")
        print("=" * 60)
        print(f"项目根目录: {self.project_root}")
        print(f"配置文件: {self.config_file}")
        
        steps = [
            ("修改配置文件", self.modify_config),
            ("编译安装 whl 包", self.build_wheel),
            ("运行可执行程序", self.run_program),
            ("修改 cpp 文件", self.modify_cpp_files),
            ("msprof 性能采集", self.run_msprof),
        ]
        
        results = []
        for step_name, step_func in steps:
            try:
                success = step_func()
                results.append((step_name, success))
                if not success and step_name not in ["运行可执行程序"]:
                    print(f"\n⚠ 步骤 '{step_name}' 失败，是否继续？(y/n)")
                    choice = input().strip().lower()
                    if choice != 'y':
                        break
            except KeyboardInterrupt:
                print("\n\n用户中断执行")
                break
            except Exception as e:
                print(f"\n错误: 步骤 '{step_name}' 发生异常: {e}")
                results.append((step_name, False))
                break
        
        print("\n" + "=" * 60)
        print("执行结果汇总")
        print("=" * 60)
        for step_name, success in results:
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{step_name}: {status}")
        
        return all(success for _, success in results)


def main():
    project_root = Path(__file__).parent
    collector = PerfDataCollector(project_root)
    success = collector.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()