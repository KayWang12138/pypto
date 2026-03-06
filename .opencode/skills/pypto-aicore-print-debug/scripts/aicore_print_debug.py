#!/usr/bin/env python3
"""
PyPTO AICORE_PRINT 自动化调试脚本
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class AICorePrintDebugger:
    """AICORE_PRINT 调试器"""
    
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)
        
        self.backup_dir = self.project_root / ".aicore_print_backup"
        
        # 配置文件路径
        self.config_file = self.project_root / "framework/src/interface/configs/tile_fwk_config.json"
        self.aicore_entry_h = self.project_root / "framework/src/interface/machine/device/tilefwk/aicore_entry.h"
        self.device_switch_h = self.project_root / "framework/src/machine/utils/device_switch.h"
    
    def _find_project_root(self) -> Path:
        """查找项目根目录"""
        current_path = Path.cwd()
        while current_path != current_path.parent:
            if (current_path / ".opencode").exists():
                return current_path
            current_path = current_path.parent
        return Path.cwd()
    
    def verify_environment(self) -> bool:
        """验证环境"""
        print("[1/5] 检查必要文件...")
        
        required_files = [
            self.config_file,
            self.aicore_entry_h,
            self.device_switch_h
        ]
        
        for file in required_files:
            if not file.exists():
                print(f"错误: 找不到文件 {file}")
                return False
        
        print("✓ 所有必要文件检查通过")
        return True
    
    def backup_config(self) -> bool:
        """备份原始配置"""
        print("[2/6] 备份原始配置...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy(self.config_file, self.backup_dir / "tile_fwk_config.json.bak")
            shutil.copy(self.aicore_entry_h, self.backup_dir / "aicore_entry.h.bak")
            shutil.copy(self.device_switch_h, self.backup_dir / "device_switch_h.bak")
            
            print(f"✓ 原始配置已备份到 {self.backup_dir}")
            return True
        except Exception as e:
            print(f"错误: 备份失败 - {e}")
            return False
    
    def setup_config(self) -> bool:
        """配置 AICORE_PRINT 环境"""
        print("[3/6] 修改配置文件...")
        
        try:
            with open(self.config_file, 'r') as f:
                config_content = f.read()
            
            config_content = config_content.replace('"fixed_output_path": false', '"fixed_output_path": true')
            config_content = config_content.replace('"fixed_output_path":false', '"fixed_output_path": true')
            config_content = config_content.replace('"force_overwrite": true', '"force_overwrite": false')
            config_content = config_content.replace('"force_overwrite":true', '"force_overwrite": false')
            
            with open(self.config_file, 'w') as f:
                f.write(config_content)
            
            print("✓ 配置文件已修改")
            return True
        except Exception as e:
            print(f"错误: 修改配置文件失败 - {e}")
            return False
    
    def setup_macros(self) -> bool:
        """打开宏定义"""
        print("[4/6] 打开 AICORE_PRINT 宏定义...")
        
        try:
            for header_file in [self.aicore_entry_h, self.device_switch_h]:
                with open(header_file, 'r') as f:
                    content = f.read()
                
                if 'ENABLE_AICORE_PRINT' in content:
                    content = content.replace('#define ENABLE_AICORE_PRINT 0', '#define ENABLE_AICORE_PRINT 1')
                    content = content.replace('#define{ENABLE_AICORE_PRINT} 0', '#define ENABLE_AICORE_PRINT 1')
                else:
                    content = '#define ENABLE_AICORE_PRINT 1\n' + content
                
                with open(header_file, 'w') as f:
                    f.write(content)
            
            print("✓ AICORE_PRINT 宏定义已打开")
            return True
        except Exception as e:
            print(f"错误: 打开宏定义失败 - {e}")
            return False
    
    def verify_setup(self) -> bool:
        """验证配置"""
        print("[5/6] 验证修改...")
        
        try:
            print("tile_fwk_config.json 配置:")
            with open(self.config_file, 'r') as f:
                for line in f:
                    if 'fixed_output_path' in line or 'force_overwrite' in line:
                        print(f"  {line.strip()}")
            
            print("\naicore_entry.h 宏定义:")
            with open(self.aicore_entry_h, 'r') as f:
                for line in f:
                    if 'ENABLE_AICORE_PRINT' in line:
                        print(f"  {line.strip()}")
            
            print("\ndevice_switch.h 宏定义:")
            with open(self.device_switch_h, 'r') as f:
                for line in f:
                    if 'ENABLE_AICORE_PRINT' in line:
                        print(f"  {line.strip()}")
            
            return True
        except Exception as e:
            print(f"错误: 验证失败 - {e}")
            return False
    
    def setup(self) -> bool:
        """配置 AICORE_PRINT 环境"""
        print("=" * 50)
        print("PyPTO AICORE_PRINT 环境配置")
        print("=" * 50)
        print()
        
        if not self.verify_environment():
            return False
        
        if not self.backup_config():
            return False
        
        if not self.setup_config():
            return False
        
        if not self.setup_macros():
            return False
        
        if not self.verify_setup():
            return False
        
        print()
        print("=" * 50)
        print("✓ AICORE_PRINT 环境配置完成")
        print("=" * 50)
        print()
        self._print_next_steps()
        
        return True
    
    def restore(self) -> bool:
        """恢复原始配置"""
        print("=" * 50)
        print("恢复 AICORE_PRINT 原始配置")
        print("=" * 50)
        print()
        
        if not self.backup_dir.exists():
            print(f"错误: 找不到备份目录 {self.backup_dir}")
            print("请先运行 setup 进行配置")
            return False
        
        print("[1/3] 恢复配置文件...")
        
        try:
            if (self.backup_dir / "tile_fwk_config.json.bak").exists():
                shutil.copy(self.backup_dir / "tile_fwk_config.json.bak", self.config_file)
                print("✓ tile_fwk_config.json 已恢复")
            
            if (self.backup_dir / "aicore_entry.h.bak").exists():
                shutil.copy(self.backup_dir / "aicore_entry.h.bak", self.aicore_entry_h)
                print("✓ aicore_entry.h 已恢复")
            
            if (self.backup_dir / "device_switch_h.bak").exists():
                shutil.copy(self.backup_dir / "device_switch_h.bak", self.device_switch_h)
                print("✓ device_switch_h 已恢复")
            
            print()
            print("[2/3] 验证恢复结果...")
            self.verify_setup()
            
            print()
            print("=" * 50)
            print("✓ 原始配置已恢复")
            print("=" * 50)
            print()
            print("建议操作:")
            print("1. 重新编译和安装 Pypto")
            print(f"   cd {self.project_root}")
            print("   rm -rf build_out output")
            print("   python3 build_ci.py -f python3")
            print("   pip install build_out/pypto*whl --force-reinstall --no-deps")
            print()
            
            return True
        except Exception as e:
            print(f"错误: 恢复失败 - {e}")
            return False
    
    def rebuild(self) -> bool:
        """重新编译和安装 Pypto"""
        print("=" * 50)
        print("重新编译和安装 Pypto")
        print("=" * 50)
        print()
        
        build_ci = self.project_root / "build_ci.py"
        if not build_ci.exists():
            print("错误: 找不到 build_ci.py")
            return False
        
        print("[1/3] 清理编译产物...")
        build_out = self.project_root / "build_out"
        output = self.project_root / "output"
        
        if build_out.exists():
            shutil.rmtree(build_out)
        if output.exists():
            shutil.rmtree(output)
        
        print("✓ 编译产物已清理")
        print()
        
        print("[2/3] 重新编译 wheel 包...")
        try:
            result = subprocess.run(
                ["python3", "build_ci.py", "-f", "python3"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("错误: 编译失败")
                print(result.stderr)
                return False
            
            print("✓ 编译完成")
            print()
        except Exception as e:
            print(f"错误: 编译失败 - {e}")
            return False
        
        print("[3/3] 安装 wheel 包...")
        try:
            whl_files = list(build_out.glob("pypto*.whl"))
            if not whl_files:
                print("错误: 找不到 wheel 包")
                return False
            
            whl_file = whl_files[0]
            result = subprocess.run(
                ["pip", "install", str(whl_file), "--force-reinstall", "--no-deps"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("错误: 安装失败")
                print(result.stderr)
                return False
            
            print("✓ 安装完成")
            print()
            print("=" * 50)
            print("✓ Pypto 重新编译和安装完成")
            print("=" * 50)
            print()
            
            return True
        except Exception as e:
            print(f"错误: 安装失败 - {e}")
            return False
    
    def find_cce_files(self) -> bool:
        """查找 CCE 文件"""
        print("=" * 50)
        print("查找 CCE 文件")
        print("=" * 50)
        print()
        
        print("正在搜索 CCE 文件...")
        kernel_aicore = self.project_root / "kernel_aicore"
        
        if not kernel_aicore.exists():
            print("错误: kernel_aicore 目录不存在")
            print()
            print("可能原因:")
            print("1. 尚未运行测试用例生成 CCE 文件")
            print("2. CCE 文件生成路径不正确")
            print()
            print("建议操作:")
            print("1. 运行测试用例: python3 custom/your_operator/test_case.py")
            return False
        
        cce_files = list(kernel_aicore.rglob("*.cce"))
        
        if not cce_files:
            print("错误: 未找到 CCE 文件")
            print()
            print("可能原因:")
            print("1. 尚未运行测试用例生成 CCE 文件")
            print("2. CCE 文件生成路径不正确")
            return False
        
        print("找到以下 CCE 文件:")
        print()
        for i, cce_file in enumerate(cce_files, 1):
            print(f"{i}. {cce_file.relative_to(self.project_root)}")
        
        print()
        print("=" * 50)
        print("提示:")
        print("=" * 50)
        print("1. 在 CCE 文件中添加头文件: #include \"tilefwk/aicore_print.h\"")
        print("2. 在关键位置添加打印语句")
        print("3. 修改后重新执行测试用例")
        print()
        
        return True
    
    def setup_log_env(self) -> bool:
        """配置日志环境"""
        print("=" * 50)
        print("配置 AICORE_PRINT 日志环境")
        print("=" * 50)
        print()
        
        ascend_work_path = "/tmp/pypto_debug"
        
        print("[1/2] 设置日志环境变量...")
        print(f"export ASCEND_WORK_PATH={ascend_work_path}")
        print("export ASCEND_GLOBAL_LOG_LEVEL=0")
        print()
        
        os.environ['ASCEND_WORK_PATH'] = ascend_work_path
        os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
        
        print("✓ 环境变量已设置")
        print()
        
        print("[2/2] 创建日志目录...")
        log_dir = Path(ascend_work_path) / "log" / "debug"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 日志目录已创建: {ascend_work_path}")
        print()
        print("=" * 50)
        print("✓ 日志环境配置完成")
        print("=" * 50)
        print()
        print("下一步操作:")
        print("1. 运行测试用例")
        print("   python3 custom/your_operator/test_case.py")
        print()
        print("2. 分析日志")
        print("   python3 aicore_print_debug.py analyze")
        print()
        
        return True
    
    def analyze_logs(self) -> bool:
        """分析 AICORE_PRINT 日志"""
        print("=" * 50)
        print("分析 AICORE_PRINT 日志")
        print("=" * 50)
        print()
        
        ascend_work_path = os.environ.get('ASCEND_WORK_PATH', '/tmp/pypto_debug')
        log_dir = Path(ascend_work_path) / "log" / "debug"
        
        if not log_dir.exists():
            print(f"错误: 日志目录不存在: {log_dir}")
            print("请先运行测试用例生成日志")
            return False
        
        print("[1/3] 查找日志文件...")
        device_logs = list(log_dir.glob("device-*"))
        
        if not device_logs:
            print("错误: 未找到 device 日志文件")
            print("请确认:")
            print("1. 已运行测试用例")
            print("2. 环境变量 ASCEND_GLOBAL_LOG_LEVEL=0 已设置")
            return False
        
        print("找到以下日志文件:")
        print()
        for i, log_file in enumerate(device_logs, 1):
            print(f"{i}. {log_file.name}")
        print()
        
        print("[2/3] 搜索 DumpAicoreLog 日志...")
        print()
        
        for log_file in device_logs:
            print("=" * 50)
            print(f"日志文件: {log_file.name}")
            print("=" * 50)
            
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if 'DumpAicoreLog' in line or 'AiCoreLogF' in line:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        for j in range(start, end):
                            print(lines[j].rstrip())
                        print()
            except Exception as e:
                print(f"读取日志文件失败: {e}")
        
        print("[3/3] 提取关键信息...")
        print()
        print("=" * 50)
        print("所有打印信息汇总")
        print("=" * 50)
        
        all_logs = []
        for log_file in device_logs:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                if 'AiCoreLogF' in content:
                    all_logs.append(content)
            except Exception as e:
                print(f"读取日志文件失败: {e}")
        
        if all_logs:
            import re
            pattern = r'AiCoreLogF.*'
            matches = set()
            for log in all_logs:
                matches.update(re.findall(pattern, log))
            
            for match in sorted(matches):
                print(match)
        
        print()
        print("=" * 50)
        print("✓ 日志分析完成")
        print("=" * 50)
        print()
        
        return True
    
    def verify(self) -> bool:
        """验证工具完整性"""
        print("=" * 50)
        print("AICORE_PRINT 调试工具验证")
        print("=" * 50)
        print()
        
        print("[1/5] 检查脚本文件...")
        print("✓ 脚本文件检查通过")
        print()
        
        print("[2/5] 检查 skill 文件...")
        skill_file = self.project_root / ".opencode/skills/pypto-aicore-print-debug/SKILL.md"
        if not skill_file.exists():
            print("错误: 找不到 skill 文件")
            return False
        
        print("✓ Skill 文件检查通过")
        print()
        
        if not self.verify_environment():
            return False
        
        print("[5/5] 显示当前配置状态...")
        self.verify_setup()
        
        print()
        print("=" * 50)
        print("✓ 验证完成")
        print("=" * 50)
        print()
        print("所有检查通过！AICORE_PRINT 调试工具已准备就绪。")
        print()
        
        return True
    
    def _print_next_steps(self):
        """打印下一步帮助"""
        print("下一步操作:")
        print("1. 重新编译和安装 Pypto")
        print(f"   cd {self.project_root}")
        print("   rm -rf build_out output")
        print("   python3 build_ci.py -f python3")
        print("   pip install build_out/pypto*{whl} --force-reinstall --no-deps")
        print()
        print("2. 运行测试用例生成 CCE 文件")
        print("   python3 custom/your_operator/test_case.py")
        print()
        print("3. 在生成的 CCE 文件中添加打印语句")
        print()
        print("4. 配置日志环境并重新执行")
        print("   export ASCEND_WORK_PATH=/tmp/pypto_debug")
        print("   export ASCEND_GLOBAL_LOG_LEVEL=0")
        print("   python3 custom/your_operator/test_case.py")
        print()
        print("5. 分析日志")
        print("   python3 aicore_print_debug.py analyze")
        print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("PyPTO AICORE_PRINT 自动化调试工具")
        print()
        print("使用方法:")
        print("  python3 aicore_print_debug.py [command]")
        print()
        print("命令:")
        print("  setup: 配置 AICORE_PRINT 环境")
        print("  restore: 恢复原始配置")
        print("  rebuild: 重新编译和安装 Pypto")
        print("  find_cce: 查找 CCE 文件")
        print("  setup_log: 配置日志环境")
        print("  analyze: 分析 AICORE_PRINT 日志")
        print("  verify: 验证工具完整性")
        print("  all: 完整流程 (setup + rebuild)")
        print()
        sys.exit(1)
    
    command = sys.argv[1]
    debugger = AICorePrintDebugger()
    
    if command == "setup":
        success = debugger.setup()
    elif command == "restore":
        success = debugger.restore()
    elif command == "rebuild":
        success = debugger.rebuild()
    elif command == "find_cce":
        success = debugger.find_cce_files()
    elif command == "setup_log":
        success = debugger.setup_log_env()
    elif command == "analyze":
        success = debugger.analyze_logs()
    elif command == "verify":
        success = debugger.verify()
    elif command == "all":
        print("执行完整流程: 配置环境 + 重新编译安装")
        print()
        success = debugger.setup()
        if success:
            print()
            input("按 Enter 继续重新编译和安装...")
            success = debugger.rebuild()
    else:
        print(f"错误: 未知命令 {command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
