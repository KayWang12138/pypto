# -*- coding: utf-8 -*-
"""
需求转 SRS 文档生成脚本

将 JSON 格式的需求信息转换为 Word 形式的软件设计 SRS 文档。
扫描项目文件提取现有 C++ 接口，结合需求生成符合规范的软件接口设计。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

# 执行次数状态文件路径
EXECUTION_STATE_FILE = None


def get_execution_state_file(temp_dir: Path) -> Path:
    """获取执行次数状态文件路径"""
    global EXECUTION_STATE_FILE
    if EXECUTION_STATE_FILE is None:
        EXECUTION_STATE_FILE = temp_dir / ".srs_execution_state.json"
    return EXECUTION_STATE_FILE


def get_execution_count(temp_dir: Path) -> int:
    """获取当前执行次数"""
    state_file = get_execution_state_file(temp_dir)
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                return state.get('execution_count', 0)
        except Exception:
            pass
    return 0


def increment_execution_count(temp_dir: Path) -> int:
    """增加执行次数并返回新的次数"""
    state_file = get_execution_state_file(temp_dir)
    current_count = get_execution_count(temp_dir)
    new_count = current_count + 1
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({'execution_count': new_count}, f)
    except Exception:
        pass
    return new_count


def reset_execution_count(temp_dir: Path):
    """重置执行次数（成功生成文档后调用）"""
    state_file = get_execution_state_file(temp_dir)
    if state_file.exists():
        try:
            state_file.unlink()
        except Exception:
            pass

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    print("错误：未安装 python-docx 库")
    print("请执行：pip install python-docx")
    sys.exit(1)


@dataclass
class FunctionInterface:
    """C++ 函数接口"""
    return_type: str
    name: str
    params: List[str]
    description: str = ""
    input_desc: str = ""  # 输入参数说明（常量或值传递的入参）
    output_desc: str = ""  # 输出参数说明（引用传递的入参）
    return_desc: str = ""  # 返回值说明
    usage_desc: str = ""  # 使用说明
    caution_desc: str = ""  # 注意事项
    
    def prototype(self) -> str:
        """生成函数原型"""
        params_str = ", ".join(self.params) if self.params else "void"
        return f"{self.return_type} {self.name}({params_str})"


@dataclass
class SubRequirement:
    """子需求"""
    title: str
    logic: str
    description: str = ""
    mermaid_flow: str = ""
    mermaid_arch: str = ""
    flow_steps: List[str] = field(default_factory=list)


@dataclass
class Requirement:
    """需求"""
    title: str
    background: str = ""
    description: str = ""
    expanded_background: str = ""
    expanded_description: str = ""
    sub_requirements: List[SubRequirement] = field(default_factory=list)
    interfaces: List[FunctionInterface] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    expanded_dependencies: str = ""
    expanded_external_interfaces: str = ""  # 对外接口依赖说明
    mermaid_arch: str = ""
    _has_no_dependencies: bool = False  # 标记是否明确声明无依赖
    # 新增字段
    architecture_analysis: str = ""  # 共架构分析
    feature_crossover: str = ""  # 特性交叉
    performance_requirements: str = ""  # 性能需求
    quality_requirements: str = ""  # 质量需求
    data_description: str = ""  # 数据描述


@dataclass
class SRSConfig:
    """SRS 文档配置"""
    project_path: str = "."
    git_repo_url: str = "https://gitcode.com/cann/pypto"
    include_dirs: List[str] = field(default_factory=lambda: ["framework/include/tilefwk"])


class PlaceholderValidator:
    """占位符验证器 - 检测扩写内容中是否包含占位符"""
    
    PLACEHOLDER_PATTERNS = [
        r'【待补充】',
        r'\[待补充\]',
        r'待补充',
        r'TODO',
        r'FIXME',
        r'请参考项目文档',
        r'请参考相关技术资料',
        r'请Agent',
        r'需要Agent',
    ]
    
    # 无依赖声明关键词
    NO_DEPENDENCY_KEYWORDS = [
        '无依赖',
        '不涉及其他组件',
        '独立模块',
        '无跨模块',
    ]
    
    @classmethod
    def contains_placeholder(cls, text: str) -> bool:
        """检查文本是否包含占位符"""
        if not text:
            return True
        for pattern in cls.PLACEHOLDER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def is_no_dependency_statement(cls, text: str) -> bool:
        """检查是否为无依赖声明"""
        if not text:
            return False
        for keyword in cls.NO_DEPENDENCY_KEYWORDS:
            if keyword in text:
                return True
        return False
    
    @classmethod
    def validate_expanded_content(cls, expanded_data: dict) -> List[str]:
        """验证扩写内容，返回错误列表"""
        errors = []
        
        for req_idx, req in enumerate(expanded_data.get('requirements', [])):
            req_title = req.get('title', f'需求{req_idx + 1}')
            
            # 检查 expanded_background
            expanded_bg = req.get('expanded_background', '')
            if cls.contains_placeholder(expanded_bg):
                errors.append(f"需求 '{req_title}' 的 expanded_background 包含占位符，请扩写")
            elif len(expanded_bg) < 300:
                errors.append(f"需求 '{req_title}' 的 expanded_background 长度不足300字符（当前{len(expanded_bg)}字符）")
            
            # 检查 expanded_description
            expanded_desc = req.get('expanded_description', '')
            if cls.contains_placeholder(expanded_desc):
                errors.append(f"需求 '{req_title}' 的 expanded_description 包含占位符，请扩写")
            elif len(expanded_desc) < 300:
                errors.append(f"需求 '{req_title}' 的 expanded_description 长度不足300字符（当前{len(expanded_desc)}字符）")
            
            # 检查 expanded_dependencies
            expanded_deps = req.get('expanded_dependencies', '')
            if cls.contains_placeholder(expanded_deps):
                errors.append(f"需求 '{req_title}' 的 expanded_dependencies 包含占位符，请扩写")
            else:
                # 检查是否为无依赖声明
                is_no_dep = cls.is_no_dependency_statement(expanded_deps)
                min_length = 50 if is_no_dep else 200
                if len(expanded_deps) < min_length:
                    errors.append(f"需求 '{req_title}' 的 expanded_dependencies 长度不足{min_length}字符（当前{len(expanded_deps)}字符）{'，若无依赖请明确声明' if not is_no_dep else ''}")
            
            # 检查子需求
            for sub_idx, sub_req in enumerate(req.get('sub_requirements', [])):
                sub_title = sub_req.get('title', f'子需求{sub_idx + 1}')
                full_title = f"{req_title} -> {sub_title}"
                
                # 检查 description
                sub_desc = sub_req.get('description', '')
                if cls.contains_placeholder(sub_desc):
                    errors.append(f"子模块 '{full_title}' 的 description 包含占位符，请扩写")
                elif len(sub_desc) < 100:
                    errors.append(f"子模块 '{full_title}' 的 description 长度不足100字符（当前{len(sub_desc)}字符）")
        
        return errors
    
    @classmethod
    def extract_file_paths(cls, text: str) -> List[str]:
        """从文本中提取文件路径"""
        # 使用非捕获组避免 findall 只返回扩展名
        patterns = [
            r'(framework/[\w/\\]+\.(?:py|h|cpp))',  # framework 下的文件（优先匹配）
            r'([\w/\\]+\.(?:py|h|cpp|hpp|c|cc|cxx|md|txt|json))',  # 标准文件路径
            r'\(([\w/\\]+\.(?:py|h|cpp))\)',  # 括号内的文件路径
        ]

        file_paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match
                # 跳过长度过短的匹配（纯扩展名误匹配）
                if not match or len(match) <= 4:
                    continue
                # 跳过含中文字符的路径
                if re.search(r'[\u4e00-\u9fff]', match):
                    continue
                if match not in file_paths:
                    file_paths.append(match)

        return file_paths
    
    @classmethod
    def extract_function_names(cls, text: str) -> List[str]:
        """从文本中提取函数/方法名称"""
        # 匹配驼峰命名的函数名
        patterns = [
            r'\b([A-Z][a-zA-Z]+[A-Z][a-zA-Z]*)\b',  # 驼峰命名如 GetHardwareParam
            r'\b([a-z][a-zA-Z]+[A-Z][a-zA-Z]*)\b',  # 小写开头的驼峰如 getHardwareParam
            r'调用\s+(\w+)\s+接口',  # "调用 xxx 接口"
            r'函数\s+(\w+)',  # "函数 xxx"
            r'方法\s+(\w+)',  # "方法 xxx"
        ]
        
        # 排除常见的关键词
        exclude_words = ['Platform', 'Module', 'Manager', 'Handler', 'Context', 'Config', 
                        'Device', 'Memory', 'Compiler', 'Executor', 'Monitor', 'Result',
                        'Type', 'Info', 'Data', 'Value', 'Error', 'Status', 'True', 'False',
                        'NA', 'TODO', 'FIXME', 'PyPTO', 'Python', 'Bool', 'Float', 'Int',
                        'String', 'Vector', 'Map', 'List', 'Set', 'Dict', 'Tuple']
        
        function_names = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and match not in exclude_words and len(match) > 3:
                    if match not in function_names:
                        function_names.append(match)
        
        return function_names


class LanguageOrganizer:
    """语言组织校验器 - 检测并修复因删除模块导致的语句不通问题"""
    
    # 不合理的标点连续模式
    PUNCTUATION_PATTERNS = [
        (r'[、，。；：！？（）\(\)\[\]【】]{2,}', '标点符号连续出现'),  # 多个标点连续
        (r'[\(（]\s*[\)）]', '空括号'),  # 空括号
        (r'[\[【]\s*[\]】]', '空方括号'),  # 空方括号
        (r'、\s*、', '连续顿号'),  # 连续顿号
        (r'，\s*、', '逗号后顿号'),  # 逗号后顿号
        (r'、\s*，', '顿号后逗号'),  # 顿号后逗号
        (r'[0-9]+\s*[、，]\s*[\)）\]】]', '编号后无内容'),  # 编号后无内容
        (r'[（\(][^）\)]{0,3}[）\)]', '括号内容过短'),  # 括号内容过短（少于3字符）
    ]
    
    # 不合理的短语模式（删除模块后残留）
    UNREASONABLE_PHRASES = [
        r'模块\s*[（\(]\s*[）\)]',  # "模块（）"
        r'接口\s*[（\(]\s*[）\)]',  # "接口（）"
        r'函数\s*[（\(]\s*[）\)]',  # "函数（）"
        r'类\s*[（\(]\s*[）\)]',  # "类（）"
        r'文件\s*[（\(]\s*[）\)]',  # "文件（）"
        r'\d+\)\s*[,，、]\s*\d+\)',  # "1) , 2)" 编号后无内容
        r'[、，]\s*[、，]',  # 连续标点
        r'\s{3,}',  # 多个连续空格
    ]
    
    @classmethod
    def validate_text_organization(cls, text: str) -> dict:
        """验证文本语言组织是否合理，返回问题列表和建议修复"""
        issues = []
        
        for pattern, description in cls.PUNCTUATION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                issues.append({
                    'type': 'punctuation',
                    'description': description,
                    'matches': matches[:3]  # 只显示前3个
                })
        
        for pattern in cls.UNREASONABLE_PHRASES:
            matches = re.findall(pattern, text)
            if matches:
                issues.append({
                    'type': 'phrase',
                    'description': '删除模块后残留的不合理短语',
                    'matches': matches[:3]
                })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    @classmethod
    def reorganize_text(cls, text: str, max_attempts: int = 3) -> str:
        """重新组织文本，修复语句不通问题"""
        organized = text
        
        for attempt in range(max_attempts):
            prev_organized = organized
            
            # 1. 删除空括号及其前面的关联词
            organized = re.sub(r'[（\(]\s*[）\)]', '', organized)
            organized = re.sub(r'[【\[]\s*[】\]]', '', organized)
            
            # 2. 修复"模块（）"等模式
            organized = re.sub(r'(模块|接口|函数|类|文件|组件)\s*[（\(][^）\)]{0,5}[）\)]', r'\1', organized)
            
            # 3. 修复连续标点
            organized = re.sub(r'[、，。；：！？]{2,}', '，', organized)
            organized = re.sub(r'[、，]\s*[、，]', '，', organized)
            
            # 4. 修复编号后无内容
            organized = re.sub(r'\d+[）\)]\s*[,，、]\s*', '', organized)
            organized = re.sub(r'\d+[）\)]\s*(?=[、，。；：！？]|$)', '', organized)
            
            # 5. 清理多余空格
            organized = re.sub(r'\s{2,}', ' ', organized)
            
            # 6. 修复"数字)"开头但无后续内容的情况
            organized = re.sub(r'\s*\d+\)\s*(?=[、，。]|$)', '', organized)
            
            # 7. 修复残留的单个顿号或逗号
            organized = re.sub(r'^\s*[、，]\s*', '', organized)
            organized = re.sub(r'\s*[、，]\s*$', '', organized)
            organized = re.sub(r'\s*[、，]\s*\.', '。', organized)
            
            # 8. 如果修改后与修改前相同，退出循环
            if organized == prev_organized:
                break
        
        # 最终验证
        validation = cls.validate_text_organization(organized)
        if not validation['is_valid']:
            # 如果仍有问题，进行更激进的清理
            organized = cls._aggressive_cleanup(organized)
        
        return organized.strip()
    
    @classmethod
    def _aggressive_cleanup(cls, text: str) -> str:
        """激进清理：处理顽固的语言问题"""
        cleaned = text
        
        # 1. 提取所有句子，逐句检查
        sentences = re.split(r'([。！？])', cleaned)
        result_sentences = []
        
        for i, part in enumerate(sentences):
            # 跳过标点符号
            if part in '。！？':
                if result_sentences:  # 只有前面有句子时才添加标点
                    result_sentences.append(part)
                continue
            
            # 检查句子是否合理
            if len(part.strip()) < 3:
                continue
            
            # 检查是否包含不合理的模式
            has_issue = False
            for pattern in cls.UNREASONABLE_PHRASES:
                if re.search(pattern, part):
                    has_issue = True
                    break
            
            if not has_issue:
                result_sentences.append(part)
        
        cleaned = ''.join(result_sentences)
        
        # 2. 最终清理
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'^[、，\s]+', '', cleaned)
        cleaned = re.sub(r'[、，\s]+$', '', cleaned)
        
        return cleaned


class ContentValidator:
    """内容验证器 - 验证扩写内容中的文件和方法是否存在"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.project_path_posix = str(self.project_path).replace('\\', '/')  # bash 兼容路径
        self.verified_files = {}  # 缓存已验证的文件
        self.verified_functions = {}  # 缓存已验证的函数

    def verify_file_exists(self, file_path: str) -> bool:
        """验证文件是否存在，使用 find 命令"""
        if file_path in self.verified_files:
            return self.verified_files[file_path]

        # 跳过含中文字符的路径
        if re.search(r'[\u4e00-\u9fff]', file_path):
            self.verified_files[file_path] = False
            return False

        # 先尝试直接路径
        direct_path = self.project_path / file_path
        if direct_path.exists() and direct_path.is_file():
            self.verified_files[file_path] = True
            return True

        # 使用 find 命令搜索文件
        file_name = Path(file_path).name
        try:
            result = subprocess.run(
                ['bash', '-c', f'find "{self.project_path_posix}" -type f -name "{file_name}" | head -1'],
                capture_output=True, timeout=10, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0 and result.stdout.strip():
                self.verified_files[file_path] = True
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        self.verified_files[file_path] = False
        return False

    def verify_function_exists(self, function_name: str, file_hint: str = None) -> bool:
        """验证函数或类是否存在，使用 grep 命令"""
        cache_key = f"{function_name}@{file_hint}" if file_hint else function_name
        if cache_key in self.verified_functions:
            return self.verified_functions[cache_key]

        # 确定搜索范围
        search_paths = []
        if file_hint:
            hint_path = self.project_path / file_hint
            if hint_path.exists():
                search_paths.append(str(hint_path).replace('\\', '/'))

        # 添加常见目录
        for d in ['framework', 'src', 'python']:
            dp = self.project_path / d
            if dp.exists():
                search_paths.append(str(dp).replace('\\', '/'))

        if not search_paths:
            search_paths.append(self.project_path_posix)

        # 使用 grep 命令搜索函数/类定义
        include_flags = "--include='*.py' --include='*.h' --include='*.cpp' --include='*.hpp'"
        for search_path in search_paths:
            try:
                result = subprocess.run(
                    ['bash', '-c',
                     f'grep -r -l {include_flags} -E "\\b{function_name}\\b" "{search_path}" 2>/dev/null | head -1'],
                    capture_output=True, timeout=15, encoding='utf-8', errors='ignore'
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.verified_functions[cache_key] = True
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        self.verified_functions[cache_key] = False
        return False
    
    def validate_content(self, text: str) -> dict:
        """验证文本中的文件和方法，返回验证报告"""
        report = {
            'files_found': [],
            'files_not_found': [],
            'functions_found': [],
            'functions_not_found': [],
            'warnings': []
        }
        
        # 提取并验证文件路径
        file_paths = PlaceholderValidator.extract_file_paths(text)
        for file_path in file_paths:
            if self.verify_file_exists(file_path):
                report['files_found'].append(file_path)
            else:
                report['files_not_found'].append(file_path)
                report['warnings'].append(f"文件 '{file_path}' 未在项目中找到，建议使用模块描述替代")
        
        # 提取并验证函数名
        function_names = PlaceholderValidator.extract_function_names(text)
        for func_name in function_names:
            # 尝试从文本中找到对应的文件提示
            file_hint = None
            for file_path in file_paths:
                if file_path in text and func_name in text:
                    file_hint = file_path
                    break
            
            if self.verify_function_exists(func_name, file_hint):
                report['functions_found'].append(func_name)
            else:
                report['functions_not_found'].append(func_name)
                report['warnings'].append(f"函数 '{func_name}' 未在项目中找到，建议使用功能描述替代")
        
        return report
    
    def sanitize_content(self, text: str) -> str:
        """清理内容，删除未验证的文件路径和函数/类名，并重新组织语言"""
        sanitized = text

        # 删除未找到的文件路径
        file_paths = PlaceholderValidator.extract_file_paths(text)
        for file_path in file_paths:
            if not self.verify_file_exists(file_path):
                # 删除括号包裹的路径，如 (framework/xxx.py)
                sanitized = sanitized.replace(f"（{file_path}）", "")
                sanitized = sanitized.replace(f"({file_path})", "")
                sanitized = sanitized.replace(file_path, "")

        # 删除未找到的函数/类名
        function_names = PlaceholderValidator.extract_function_names(text)
        for func_name in function_names:
            if not self.verify_function_exists(func_name):
                sanitized = sanitized.replace(func_name, "")

        # 最终校验：删除所有含中文字符的文件路径
        sanitized = re.sub(
            r'[\w/\\]*[\u4e00-\u9fff]+[\w/\\]*\.(?:py|h|cpp|hpp|c|cc|cxx|md|txt|json)',
            '', sanitized
        )
        # 清理多余的空括号和连续空格
        sanitized = re.sub(r'（\s*）', '', sanitized)
        sanitized = re.sub(r'\(\s*\)', '', sanitized)
        sanitized = re.sub(r'  +', ' ', sanitized)

        # 新增：使用 LanguageOrganizer 重新组织语言
        sanitized = LanguageOrganizer.reorganize_text(sanitized)

        return sanitized
    
    def validate_and_sanitize(self, text: str) -> dict:
        """验证并清理内容，返回处理结果和验证报告"""
        # 先验证语言组织
        language_validation = LanguageOrganizer.validate_text_organization(text)
        
        # 清理内容
        sanitized = self.sanitize_content(text)
        
        # 再次验证清理后的语言组织
        final_validation = LanguageOrganizer.validate_text_organization(sanitized)
        
        return {
            'original_text': text,
            'sanitized_text': sanitized,
            'language_issues': language_validation['issues'],
            'is_valid': final_validation['is_valid'],
            'final_issues': final_validation['issues']
        }


class ProjectEnvironmentDetector:
    """项目环境检测器 - 检测当前目录是否为 pypto 项目"""
    
    @staticmethod
    def is_pypto_project(project_path: str = ".") -> bool:
        """检测当前目录是否为 pypto 项目
        
        判断条件：
        1. 目录名称为 "pypto"
        2. README.md 文件描述的是 pypto 项目
        """
        path = Path(project_path).resolve()  # 获取绝对路径
        
        # 检查目录名称
        dir_name = path.name.lower()
        if dir_name != "pypto":
            return False
        
        # 检查 README.md 文件
        readme_path = path / "README.md"
        if not readme_path.exists():
            return False
        
        try:
            # 尝试多种编码读取
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(readme_path, 'r', encoding=encoding) as f:
                        content = f.read(1000)  # 读取前1000字符
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                return False
            
            # 检查是否包含 pypto 相关关键词（不区分大小写）
            content_lower = content.lower()
            keywords = ['pypto', 'pto编程', 'tensor graph', 'tile graph']
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    return True
        except Exception:
            pass
        
        return False
    
    @staticmethod
    def get_project_info_source(project_path: str = ".") -> str:
        """获取项目信息来源方式
        
        Returns:
            "local" - 使用本地目录
            "remote" - 使用远程仓库
        """
        if ProjectEnvironmentDetector.is_pypto_project(project_path):
            print("检测到当前目录为 pypto 项目，将使用本地文件获取项目信息")
            return "local"
        else:
            print("当前目录不是 pypto 项目，将通过网络访问远程仓库获取项目信息")
            print("远程仓库地址：https://gitcode.com/cann/pypto")
            return "remote"


class ProjectScanner:
    """项目文件扫描器"""
    
    def __init__(self, config: SRSConfig):
        self.config = config
        self.existing_interfaces: Dict[str, FunctionInterface] = {}
        self.class_names: List[str] = []
    
    def scan(self) -> Dict[str, FunctionInterface]:
        """扫描项目文件，提取现有接口"""
        project_path = Path(self.config.project_path)
        
        if not project_path.exists():
            raise FileNotFoundError(f"项目路径不存在：{project_path}")
        
        # 优先扫描指定的 include 目录
        for include_dir in self.config.include_dirs:
            include_path = project_path / include_dir
            if include_path.exists():
                self._scan_directory(include_path)
        
        # 扫描所有 .h 文件
        self._scan_directory(project_path, pattern="**/*.h")
        
        return self.existing_interfaces
    
    def _scan_directory(self, directory: Path, pattern: str = "**/*.h"):
        """扫描目录下的头文件"""
        for file_path in directory.glob(pattern):
            try:
                self._parse_header_file(file_path)
            except Exception as e:
                print(f"警告：解析文件 {file_path} 时出错：{e}")
    
    def _parse_header_file(self, file_path: Path):
        """解析头文件，提取函数声明"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return
        
        # 提取类名
        class_pattern = r'\bclass\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            self.class_names.append(match.group(1))
        
        # 提取函数声明（简化模式，匹配返回值 函数名(参数)）
        func_pattern = r'(?://\s*([^\n]+)\n\s*)?(?:virtual\s+)?(?:static\s+)?(\w+(?:\s*[*&])?)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?;'
        
        for match in re.finditer(func_pattern, content):
            return_type = match.group(2).strip()
            func_name = match.group(3).strip()
            params_str = match.group(4).strip()
            description = match.group(1).strip() if match.group(1) else ""
            
            # 跳过构造函数、析构函数和运算符重载
            if func_name.startswith('~') or func_name.startswith('operator'):
                continue
            
            params = self._parse_params(params_str)
            
            interface = FunctionInterface(
                return_type=return_type,
                name=func_name,
                params=params,
                description=description
            )
            
            self.existing_interfaces[func_name] = interface
    
    def _parse_params(self, params_str: str) -> List[str]:
        """解析参数列表"""
        if not params_str or params_str.strip() == "void":
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                params.append(param)
        
        return params


class InterfaceInferrer:
    """接口推断器"""
    
    VERBS = {
        '获取': 'Get',
        '设置': 'Set',
        '创建': 'Create',
        '销毁': 'Destroy',
        '初始化': 'Initialize',
        '执行': 'Execute',
        '验证': 'Validate',
        '检查': 'Check',
        '计算': 'Calculate',
        '处理': 'Process',
        '加载': 'Load',
        '保存': 'Save',
        '读取': 'Read',
        '写入': 'Write',
        '迁移': 'Migrate',
        '添加': 'Add',
        '删除': 'Remove',
        '更新': 'Update',
        '查询': 'Query',
        '注册': 'Register',
        '注销': 'Unregister',
    }
    
    def __init__(self, existing_interfaces: Dict[str, FunctionInterface], class_names: List[str], templates_path: str = None):
        self.existing_interfaces = existing_interfaces
        self.class_names = class_names
        self.templates_path = templates_path or os.path.join(os.path.dirname(__file__), '..', 'templates.json')
        self._load_templates_from_json()
    
    def _load_templates_from_json(self):
        """从 JSON 文件加载模板"""
        try:
            with open(self.templates_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            self.TYPE_MAPPING = templates_data.get('type_mapping', {})
            self.translation_mapping = templates_data.get('translation_mapping', {})
            
            print(f"已从 {self.templates_path} 加载接口推断模板")
        except FileNotFoundError:
            raise FileNotFoundError(f"模板文件不存在：{self.templates_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"模板文件 JSON 格式错误：{e}")
    
    def infer_from_requirement(self, requirement: Requirement) -> List[FunctionInterface]:
        """从需求推断接口"""
        interfaces = []
        
        # 从描述中推断
        desc = requirement.description.lower()
        for cn_keyword, en_verb in self.VERBS.items():
            if cn_keyword in requirement.description:
                # 尝试提取函数名
                func_name = self._extract_function_name(requirement.description, cn_keyword)
                if func_name and func_name not in self.existing_interfaces:
                    interface = self._create_interface(func_name, requirement.description)
                    interfaces.append(interface)
        
        # 从子需求推断
        for sub_req in requirement.sub_requirements:
            interfaces.extend(self._infer_from_sub_requirement(sub_req))
        
        return interfaces
    
    def _infer_from_sub_requirement(self, sub_req: SubRequirement) -> List[FunctionInterface]:
        """从子需求推断接口"""
        interfaces = []
        
        for cn_keyword, en_verb in self.VERBS.items():
            if cn_keyword in sub_req.title or cn_keyword in sub_req.logic:
                func_name = self._extract_function_name(sub_req.title + " " + sub_req.logic, cn_keyword)
                if func_name and func_name not in self.existing_interfaces:
                    interface = self._create_interface(func_name, sub_req.logic)
                    interfaces.append(interface)
        
        return interfaces
    
    def _extract_function_name(self, text: str, verb: str) -> Optional[str]:
        """从文本中提取函数名"""
        # 使用大驼峰命名法
        en_verb = self.VERBS.get(verb, verb.capitalize())
        
        # 尝试提取名词部分
        patterns = [
            r'(\w+)(?:参数|接口|文件|配置|信息|数据)',
            r'(\w+)的(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                noun = match.group(1)
                func_name = f"{en_verb}{noun.capitalize()}"
                # 校验函数名不包含中文字符
                if self._contains_chinese(func_name):
                    continue
                return func_name
        
        func_name = f"{en_verb}Data"
        # 校验函数名不包含中文字符
        if self._contains_chinese(func_name):
            raise ValueError(f"推断的函数名包含中文字符：{func_name}")
        return func_name
    
    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def _create_interface(self, name: str, description: str) -> FunctionInterface:
        """创建接口"""
        # 根据名称推断参数和返回值
        params = []
        return_type = "void"
        input_params = []  # 输入参数（常量或值传递）
        output_params = []  # 输出参数（引用传递）
        input_desc = "NA"
        output_desc = "NA"
        return_desc = "NA"
        usage_desc = "NA"
        caution_desc = "NA"
        
        # 提取参数并生成参数说明
        for keyword, type_name in self.TYPE_MAPPING.items():
            if keyword in description:
                param_name = keyword.lower()
                # 校验参数名不包含中文字符
                if self._contains_chinese(param_name):
                    # 使用英文参数名映射
                    param_name = self._translate_to_english(keyword)
                # 再次校验参数名不包含中文字符
                if self._contains_chinese(param_name):
                    raise ValueError(f"参数名包含中文字符：{param_name}，请检查翻译映射")
                
                # 根据类型决定是输入还是输出参数
                if '&' in type_name and 'const' not in type_name:
                    # 引用传递且非const -> 输出参数
                    params.append(f"{type_name} {param_name}")
                    output_params.append(f"{type_name} {param_name}：{keyword}")
                else:
                    # 常量或值传递 -> 输入参数
                    params.append(f"{type_name} {param_name}")
                    input_params.append(f"{type_name} {param_name}：{keyword}")
        
        # 生成输入参数说明
        if input_params:
            input_desc = "；".join(input_params)
        
        # 生成输出参数说明
        if output_params:
            output_desc = "；".join(output_params)
        
        # 推断返回值和返回值说明
        if name.startswith("Get") or name.startswith("Query"):
            return_type = "ResultType"
            return_desc = "ResultType：包含操作结果的结构体，data字段存储查询数据，code字段表示状态码"
            usage_desc = "在需要查询数据时调用，调用前确保参数有效性"
        elif name.startswith('Validate') or name.startswith('Check'):
            return_type = "bool"
            return_desc = "bool：- true：验证通过；- false：验证失败"
            usage_desc = "在需要验证数据或条件时调用，根据返回值判断结果"
        elif name.startswith('Create') or name.startswith('Initialize'):
            return_type = "bool"
            return_desc = "bool：- true：创建/初始化成功；- false：创建/初始化失败"
            usage_desc = "在创建对象或初始化资源时调用，失败时检查错误日志"
            caution_desc = "避免重复调用初始化函数"
        elif name.startswith('Set') or name.startswith('Update'):
            return_type = "bool"
            return_desc = "bool：- true：设置/更新成功；- false：设置/更新失败"
            usage_desc = "在需要设置或更新数据时调用，确保参数有效性"
        elif name.startswith('Execute') or name.startswith('Process'):
            return_type = "bool"
            return_desc = "bool：- true：执行成功；- false：执行失败"
            usage_desc = "在需要执行指定操作时调用，注意检查返回值"
        elif name.startswith('Load') or name.startswith('Read'):
            return_type = "bool"
            return_desc = "bool：- true：加载/读取成功；- false：加载/读取失败"
            usage_desc = "在需要加载数据时调用，确保文件路径或数据源有效"
        elif name.startswith('Save') or name.startswith('Write'):
            return_type = "bool"
            return_desc = "bool：- true：保存/写入成功；- false：保存/写入失败"
            usage_desc = "在需要保存数据时调用，确保目标路径可写"
        elif name.startswith('Destroy') or name.startswith('Remove') or name.startswith('Delete'):
            return_type = "bool"
            return_desc = "bool：- true：销毁/删除成功；- false：销毁/删除失败"
            usage_desc = "在需要释放资源或删除对象时调用"
            caution_desc = "避免重复调用销毁函数"
        elif name.startswith('Register'):
            return_type = "bool"
            return_desc = "bool：- true：注册成功；- false：注册失败"
            usage_desc = "在需要注册回调或处理器时调用，确保唯一性"
        elif name.startswith('Unregister'):
            return_type = "bool"
            return_desc = "bool：- true：注销成功；- false：注销失败"
            usage_desc = "在需要注销回调或处理器时调用，确保已注册"
        elif name.startswith('Calculate') or name.startswith('Compute'):
            return_type = "float"
            return_desc = "float：返回计算结果"
            usage_desc = "在需要执行计算时调用，确保输入参数有效"
        else:
            return_type = "void"
            return_desc = "NA"
            usage_desc = "在需要执行指定操作时调用"
        
        # 校验返回值类型不包含中文字符
        if self._contains_chinese(return_type):
            raise ValueError(f"返回值类型包含中文字符：{return_type}")
        
        return FunctionInterface(
            return_type=return_type,
            name=name,
            params=params,
            description=description,
            input_desc=input_desc,
            output_desc=output_desc,
            return_desc=return_desc,
            usage_desc=usage_desc,
            caution_desc=caution_desc
        )
    
    def _translate_to_english(self, chinese: str) -> str:
        """将中文关键词翻译为英文参数名"""
        return self.translation_mapping.get(chinese, chinese.lower())
    
    def infer_dependencies(self, requirement: Requirement) -> List[str]:
        """推断依赖关系"""
        dependencies = []
        
        # 从描述中提取类名和模块名
        for class_name in self.class_names:
            if class_name in requirement.description or class_name in requirement.background:
                dependencies.append(class_name)
        
        # 从子需求中提取
        for sub_req in requirement.sub_requirements:
            for class_name in self.class_names:
                if class_name in sub_req.logic:
                    dependencies.append(class_name)
        
        # 去重
        return list(set(dependencies))


class LogicExpander:
    """逻辑扩展器 - 自动补全子模块执行逻辑"""
    
    def __init__(self, project_path: str, class_names: List[str], templates_path: str = None, expanded_json_path: str = None):
        self.project_path = project_path
        self.class_names = class_names
        self.templates_path = templates_path or os.path.join(os.path.dirname(__file__), '..', 'templates.json')
        self.expanded_json_path = expanded_json_path
        self.expanded_sub_descriptions = {}
        self.expanded_flow_steps = {}
        self.expanded_dependencies = {}
        self._load_templates_from_json()
        self._load_expanded_content()
    
    def _load_templates_from_json(self):
        """从 JSON 文件加载所有模板"""
        try:
            with open(self.templates_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            self.logic_templates = templates_data.get('logic_templates', {})
            self.generic_flow_template = templates_data.get('generic_flow_template', '')
            self.type_mapping = templates_data.get('type_mapping', {})
            self.translation_mapping = templates_data.get('translation_mapping', {})
            
            print(f"已从 {self.templates_path} 加载模板")
        except FileNotFoundError:
            raise FileNotFoundError(f"模板文件不存在：{self.templates_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"模板文件 JSON 格式错误：{e}")
    
    def _load_expanded_content(self):
        """从扩写JSON文件加载子模块描述、流程步骤和依赖信息"""
        if self.expanded_json_path and os.path.exists(self.expanded_json_path):
            try:
                with open(self.expanded_json_path, 'r', encoding='utf-8') as f:
                    expanded_data = json.load(f)
                
                for req in expanded_data.get('requirements', []):
                    req_title = req.get('title', '')
                    
                    if req.get('expanded_dependencies'):
                        self.expanded_dependencies[req_title] = req.get('expanded_dependencies')
                    
                    for sub_req in req.get('sub_requirements', []):
                        sub_title = sub_req.get('title', '')
                        key = f"{req_title}::{sub_title}"
                        
                        if sub_req.get('description'):
                            self.expanded_sub_descriptions[key] = sub_req.get('description')
                        
                        if sub_req.get('flow_steps'):
                            self.expanded_flow_steps[key] = sub_req.get('flow_steps')
                
                total_items = len(self.expanded_sub_descriptions) + len(self.expanded_flow_steps) + len(self.expanded_dependencies)
                if total_items > 0:
                    print(f"已从扩写JSON加载 {len(self.expanded_sub_descriptions)} 个子模块描述, "
                          f"{len(self.expanded_flow_steps)} 个流程步骤, "
                          f"{len(self.expanded_dependencies)} 个依赖信息")
            except Exception as e:
                print(f"警告：加载扩写内容失败：{e}")
    
    def get_flow_steps(self, sub_title: str, parent_title: str) -> Optional[List[str]]:
        """从扩写JSON获取流程步骤"""
        key = f"{parent_title}::{sub_title}"
        return self.expanded_flow_steps.get(key)
    
    def get_dependencies(self, req_title: str) -> Optional[str]:
        """从扩写JSON获取依赖信息"""
        return self.expanded_dependencies.get(req_title)
    
    def expand_logic(self, sub_req: SubRequirement, parent_title: str = "") -> tuple:
        """扩展子需求逻辑和描述，如果过短则自动补全"""
        original_logic = sub_req.logic
        
        if len(original_logic) >= 100:
            description = self._get_description_from_expanded(sub_req.title, parent_title, original_logic)
            return original_logic, description
        
        expanded_logic = self._match_and_expand(original_logic)
        
        if expanded_logic:
            description = self._get_description_from_expanded(sub_req.title, parent_title, original_logic)
            return expanded_logic, description
        
        generic_logic = self._generate_generic_flow(sub_req.title, original_logic)
        description = self._get_description_from_expanded(sub_req.title, parent_title, original_logic)
        return generic_logic, description
    
    def _get_description_from_expanded(self, sub_title: str, parent_title: str, logic: str) -> str:
        """从扩写JSON获取子模块描述，如不存在则使用逻辑内容"""
        key = f"{parent_title}::{sub_title}"
        
        if key in self.expanded_sub_descriptions:
            desc = self.expanded_sub_descriptions[key]
            if len(desc) >= 50 and not PlaceholderValidator.contains_placeholder(desc):
                return desc
        
        # 不使用占位符，直接使用逻辑内容
        if logic and len(logic) >= 50:
            return logic[:150] + "..." if len(logic) > 150 else logic
        
        # 基于标题生成描述
        return f"负责实现{sub_title}相关功能，包括核心处理逻辑、数据验证和结果输出等关键步骤。"
    
    def _match_and_expand(self, logic: str) -> Optional[str]:
        """匹配并扩展逻辑"""
        for keyword, template in self.logic_templates.items():
            if keyword in logic:
                return template
        return None
    
    def _generate_generic_flow(self, title: str, original_logic: str) -> str:
        """生成通用流程描述"""
        return self.generic_flow_template.format(original_logic=original_logic)
    
    def save_expanded_content(self, requirements: List[Requirement], output_path: str):
        """保存扩展后的内容到 JSON 文件"""
        expanded_data = {
            'requirements': []
        }
        
        for req in requirements:
            req_data = {
                'title': req.title,
                'background': req.background,
                'expanded_background': req.expanded_background,
                'description': req.description,
                'expanded_description': req.expanded_description,
                'mermaid_architecture': req.mermaid_arch,
                'expanded_dependencies': req.expanded_dependencies,
                'sub_requirements': []
            }
            
            for sub_req in req.sub_requirements:
                sub_data = {
                    'title': sub_req.title,
                    'original_logic': sub_req.logic[:100] + '...' if len(sub_req.logic) > 100 else sub_req.logic,
                    'expanded_logic': sub_req.logic,
                    'description': sub_req.description,
                    'flow_steps': sub_req.flow_steps,
                    'mermaid_flow': sub_req.mermaid_flow
                }
                req_data['sub_requirements'].append(sub_data)
            
            expanded_data['requirements'].append(req_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expanded_data, f, ensure_ascii=False, indent=2)
        
        print(f"扩展内容已保存到：{output_path}")


class ContentExpander:
    """内容扩写器 - 从JSON读取扩写内容，不提供通用模板"""
    
    MIN_LENGTH = 300
    
    def __init__(self, project_path: str, expanded_json_path: str = None):
        self.project_path = project_path
        self.expanded_json_path = expanded_json_path
        self.expanded_data = {}
        self._load_expanded_content()
    
    def _load_expanded_content(self):
        """从JSON文件加载扩写内容"""
        if self.expanded_json_path and os.path.exists(self.expanded_json_path):
            try:
                with open(self.expanded_json_path, 'r', encoding='utf-8') as f:
                    self.expanded_data = json.load(f)
                print(f"已从 {self.expanded_json_path} 加载扩写内容")
            except Exception as e:
                print(f"警告：加载扩写内容失败：{e}")
                self.expanded_data = {'requirements': []}
        else:
            self.expanded_data = {'requirements': []}
    
    def get_expanded_background(self, req_title: str) -> Optional[str]:
        """获取指定需求的扩写背景"""
        for req in self.expanded_data.get('requirements', []):
            if req.get('title') == req_title:
                return req.get('expanded_background')
        return None
    
    def get_expanded_description(self, req_title: str) -> Optional[str]:
        """获取指定需求的扩写描述"""
        for req in self.expanded_data.get('requirements', []):
            if req.get('title') == req_title:
                return req.get('expanded_description')
        return None
    
    def expand_background(self, requirement: Requirement, errors_collector: List[str] = None) -> str:
        """扩写background - 从JSON读取，如不存在或包含占位符则记录错误
        
        Args:
            requirement: 需求对象
            errors_collector: 错误收集器列表，用于收集所有错误而不立即退出
        """
        expanded = self.get_expanded_background(requirement.title)
        if expanded and len(expanded) >= self.MIN_LENGTH and not PlaceholderValidator.contains_placeholder(expanded):
            return expanded
        
        original = requirement.background
        if len(original) >= self.MIN_LENGTH and not PlaceholderValidator.contains_placeholder(original):
            return original
        
        # 记录错误而非立即退出
        error_msg = f"需求 '{requirement.title}' 的 expanded_background 未提供或长度不足{self.MIN_LENGTH}字符。"
        if errors_collector is not None:
            errors_collector.append(error_msg)
        else:
            # 兼容旧调用方式
            print(f"\n{'='*60}")
            print(f"错误：{error_msg}")
            print("请基于项目信息进行扩写：")
            print("1. 扫描项目代码、文档获取相关实现细节")
            print("2. 结合项目架构和技术栈生成完整背景描述")
            print("3. 确保内容不包含占位符，长度>=300字符")
            print(f"{'='*60}\n")
            raise ValueError(error_msg)
        
        # 返回可用的内容（即使不符合要求）
        return expanded if expanded else original
    
    def expand_description(self, requirement: Requirement, errors_collector: List[str] = None) -> str:
        """扩写description - 从JSON读取，如不存在或包含占位符则记录错误
        
        Args:
            requirement: 需求对象
            errors_collector: 错误收集器列表，用于收集所有错误而不立即退出
        """
        expanded = self.get_expanded_description(requirement.title)
        if expanded and len(expanded) >= self.MIN_LENGTH and not PlaceholderValidator.contains_placeholder(expanded):
            return expanded
        
        original = requirement.description
        if len(original) >= self.MIN_LENGTH and not PlaceholderValidator.contains_placeholder(original):
            return original
        
        # 记录错误而非立即退出
        error_msg = f"需求 '{requirement.title}' 的 expanded_description 未提供或长度不足{self.MIN_LENGTH}字符。"
        if errors_collector is not None:
            errors_collector.append(error_msg)
        else:
            # 兼容旧调用方式
            print(f"\n{'='*60}")
            print(f"错误：{error_msg}")
            print("请基于项目信息进行扩写：")
            print("1. 扫描项目代码、文档获取相关实现细节")
            print("2. 结合项目架构和技术栈生成完整需求描述")
            print("3. 确保内容不包含占位符，长度>=300字符")
            print(f"{'='*60}\n")
            raise ValueError(error_msg)
        
        # 返回可用的内容（即使不符合要求）
        return expanded if expanded else original


class FlowchartValidator:
    """流程图合理性校验器 - 检测不合理的流程图结构"""
    
    # 不合理的流程图模式
    UNREASONABLE_PATTERNS = {
        'isolated_error_handler': {
            'pattern': r'E\["错误处理"\]\s*\n\s*(?!.*E\s*--)',  # 错误处理块没有连接到任何决策节点
            'description': '独立的错误处理块，未与任何决策节点连接'
        },
        'error_handler_no_branch': {
            'pattern': r'E\["[^"]*"\]\s*\n\s*E\s*-->\s*Z',  # 错误处理直接连接到结束，没有前置分支
            'description': '错误处理块直接连接到结束节点，缺少前置条件判断'
        },
        'empty_flow': {
            'pattern': r'A\(\[开始\]\)\s*\n\s*Z\(\[结束\]\)',  # 只有开始和结束，没有中间步骤
            'description': '流程图缺少中间处理步骤'
        },
        'single_step_flow': {
            'pattern': r'A\(\[开始\]\)\s*\n\s*N1\[.*\]\s*\n\s*Z\(\[结束\]\)',  # 只有一个步骤
            'description': '流程图只有一个处理步骤，过于简单'
        },
        'orphan_node': {
            'pattern': r'N\d+\[[^\]]+\]\s*\n(?![\s\S]*N\d+\s*--)',  # 孤立节点（没有连接）
            'description': '存在孤立的处理节点，未与其他节点连接'
        }
    }
    
    # 合理的流程图结构要求
    REASONABLE_REQUIREMENTS = {
        'min_steps': 3,  # 最少步骤数
        'max_steps': 8,  # 最多步骤数
        'should_have_branch': True,  # 应该有分支或循环结构
        'branch_keywords': ['验证', '检查', '初始化', '执行', '加载', '保存', '配置', '处理',
                            '判断', '匹配', '解析', '确认', '校验', '是否'],
        'loop_keywords': ['遍历', '循环', '迭代', '逐一', '逐个', '依次',
                          '重复', '批量', '轮询', '扫描', '每个', '所有',
                          '对每', '针对每', '分别'],
    }
    
    @classmethod
    def validate_flowchart(cls, flowchart: str) -> dict:
        """验证流程图是否合理，返回验证结果和问题列表"""
        issues = []
        is_valid = True
        
        # 1. 检查不合理的模式
        for pattern_name, pattern_info in cls.UNREASONABLE_PATTERNS.items():
            if re.search(pattern_info['pattern'], flowchart, re.MULTILINE):
                issues.append({
                    'type': pattern_name,
                    'description': pattern_info['description'],
                    'severity': 'error'
                })
                is_valid = False
        
        # 2. 检查步骤数量
        steps = re.findall(r'N\d+\[([^\]]+)\]', flowchart)
        if len(steps) < cls.REASONABLE_REQUIREMENTS['min_steps']:
            issues.append({
                'type': 'too_few_steps',
                'description': f'流程步骤过少（{len(steps)}个），建议至少{cls.REASONABLE_REQUIREMENTS["min_steps"]}个步骤',
                'severity': 'warning'
            })
        
        # 3. 检查是否有分支判断或循环结构
        decision_nodes = re.findall(r'N\d+\{([^}]+)\}', flowchart)
        loop_edges = re.findall(r'-.->.*继续循环', flowchart)
        has_structure = len(decision_nodes) > 0 or len(loop_edges) > 0

        if cls.REASONABLE_REQUIREMENTS['should_have_branch'] and not has_structure:
            # 检查步骤中是否包含分支或循环关键词
            has_branch_keyword = False
            for step in steps:
                for keyword in cls.REASONABLE_REQUIREMENTS['branch_keywords']:
                    if keyword in step:
                        has_branch_keyword = True
                        break
                if not has_branch_keyword:
                    for keyword in cls.REASONABLE_REQUIREMENTS.get('loop_keywords', []):
                        if keyword in step:
                            has_branch_keyword = True
                            break
                if has_branch_keyword:
                    break

            if has_branch_keyword:
                issues.append({
                    'type': 'missing_decision_node',
                    'description': '步骤包含分支/循环关键词但未生成对应的决策节点或循环结构',
                    'severity': 'warning'
                })
        
        # 4. 检查错误处理块的位置
        error_block_pattern = r'E\["[^"]*"\]'
        error_to_z_pattern = r'E\s*-->\s*Z'
        branch_to_error_pattern = r'N\d+\{[^}]+\}\s*-->\|[^|]+\|\s*E'
        
        if re.search(error_block_pattern, flowchart):
            has_branch_to_error = re.search(branch_to_error_pattern, flowchart)
            has_error_to_z = re.search(error_to_z_pattern, flowchart)
            
            if has_error_to_z and not has_branch_to_error:
                issues.append({
                    'type': 'isolated_error_handler',
                    'description': '错误处理块缺少前置分支判断，不应独立存在',
                    'severity': 'error'
                })
                is_valid = False
        
        # 5. 检查开始节点是否连接所有子块
        # 提取所有处理节点ID（N1, N2, ...）
        all_node_ids = re.findall(r'\b(N\d+)\b', flowchart)
        all_node_ids = list(set(all_node_ids))  # 去重
        
        # 检查从A开始的连接
        a_connections = re.findall(r'A\s*-->\s*(\w+)', flowchart)
        
        # 如果有节点未被A直接或间接连接，报告问题
        connected_from_a = set(a_connections)
        
        # 检查每个节点是否有到A的路径（直接或间接）
        for node_id in all_node_ids:
            node_connected_to_start = cls._is_node_connected_to_start(flowchart, node_id)
            if not node_connected_to_start:
                issues.append({
                    'type': 'node_not_connected_to_start',
                    'description': f'节点 {node_id} 未与开始节点A连接',
                    'severity': 'error'
                })
                is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'steps_count': len(steps),
            'decision_count': len(decision_nodes),
            'loop_count': len(loop_edges)
        }
    
    @classmethod
    def _is_node_connected_to_start(cls, flowchart: str, node_id: str) -> bool:
        """检查节点是否与开始节点A有连接（直接或间接）"""
        # 直接连接检查
        if re.search(rf'A\s*-->\s*{node_id}', flowchart):
            return True
        
        # 间接连接检查：通过BFS查找从A到node_id的路径
        lines = flowchart.split('\n')
        connections = {}  # node -> [connected_nodes]
        
        # 构建连接图（匹配 --> 和 -->|label| 两种边）
        for line in lines:
            match = re.search(r'(\w+)\s*-->(?:\|[^|]*\|)?\s*(\w+)', line)
            if match:
                src, dst = match.group(1), match.group(2)
                if src not in connections:
                    connections[src] = []
                connections[src].append(dst)
        
        # BFS检查是否可达
        visited = set()
        queue = ['A']
        
        while queue:
            current = queue.pop(0)
            if current == node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            if current in connections:
                queue.extend(connections[current])
        
        return False
    
    @classmethod
    def regenerate_if_invalid(cls, flowchart: str, title: str, steps: List[str], 
                               max_attempts: int = 3) -> str:
        """如果流程图不合理，尝试重新生成"""
        validation = cls.validate_flowchart(flowchart)
        
        if validation['is_valid']:
            return flowchart
        
        # 尝试重新生成
        for attempt in range(max_attempts):
            # 根据问题类型调整生成策略
            regenerated = cls._adjust_and_regenerate(title, steps, validation['issues'], attempt)
            new_validation = cls.validate_flowchart(regenerated)
            
            if new_validation['is_valid']:
                print(f"    流程图校验通过（第{attempt + 1}次尝试）")
                return regenerated
        
        # 如果多次尝试仍失败，生成简化版本（移除独立的错误处理块）
        print(f"    流程图校验未通过，生成简化版本")
        return cls._generate_simplified_flowchart(title, steps)
    
    @classmethod
    def _adjust_and_regenerate(cls, title: str, steps: List[str], issues: List[dict], 
                                attempt: int) -> str:
        """根据问题调整并重新生成流程图"""
        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart TD\n"
        flowchart += "    A([开始])\n"
        
        # 根据尝试次数调整策略
        if attempt == 0:
            # 第一次尝试：正常生成，确保错误处理块有连接
            flowchart = cls._generate_with_connected_error(title, steps)
        elif attempt == 1:
            # 第二次尝试：减少决策节点，简化流程
            flowchart = cls._generate_simple_linear(title, steps)
        else:
            # 第三次尝试：最简化版本
            flowchart = cls._generate_minimal(title, steps)
        
        return flowchart
    
    @classmethod
    def _generate_with_connected_error(cls, title: str, steps: List[str]) -> str:
        """生成错误处理块正确连接的流程图，支持条件分支和循环"""
        from typing import List as _List
        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart TD\n"
        flowchart += "    A([开始])\n"

        branch_keywords = cls.REASONABLE_REQUIREMENTS['branch_keywords']
        loop_keywords = ['遍历', '循环', '迭代', '逐一', '逐个', '依次',
                         '重复', '批量', '轮询', '扫描', '每个', '所有',
                         '对每', '针对每', '分别']

        node_ids = []
        node_types = []  # 'process' | 'decision' | 'loop'

        for i, step in enumerate(steps):
            node_id = f"N{i + 1}"
            node_ids.append(node_id)
            step_text = step.replace('"', "'")

            is_decision = any(kw in step for kw in branch_keywords)
            is_loop = any(kw in step for kw in loop_keywords)

            if is_decision:
                flowchart += f'    {node_id}{{"{step_text}"}}\n'
                node_types.append('decision')
            else:
                flowchart += f'    {node_id}["{step_text}"]\n'
                node_types.append('loop' if is_loop else 'process')

        flowchart += "    Z([结束])\n"

        has_decision = 'decision' in node_types
        if has_decision:
            flowchart += '    E["错误处理"]\n'

        # 连接节点
        if node_ids:
            flowchart += f"    A --> {node_ids[0]}\n"

        for i in range(len(node_ids)):
            nid = node_ids[i]
            ntype = node_types[i]
            next_nid = node_ids[i + 1] if i + 1 < len(node_ids) else 'Z'

            if ntype == 'decision':
                flowchart += f"    {nid} -->|成功| {next_nid}\n"
                if has_decision:
                    flowchart += f"    {nid} -->|失败| E\n"
            elif ntype == 'loop':
                flowchart += f"    {nid} --> {next_nid}\n"
                flowchart += f"    {nid} -.->|继续循环| {nid}\n"
            else:
                flowchart += f"    {nid} --> {next_nid}\n"

        if has_decision:
            flowchart += "    E --> Z\n"

        return flowchart
    
    @classmethod
    def _generate_simple_linear(cls, title: str, steps: List[str]) -> str:
        """生成简单的线性流程图（无独立错误处理块），保留循环回边"""
        loop_keywords = ['遍历', '循环', '迭代', '逐一', '逐个', '依次',
                         '重复', '批量', '轮询', '扫描', '每个', '所有',
                         '对每', '针对每', '分别']

        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart TD\n"
        flowchart += "    A([开始])\n"

        node_ids = []
        is_loop = []
        for i, step in enumerate(steps):
            node_id = f"N{i + 1}"
            node_ids.append(node_id)
            step_text = step.replace('"', "'")
            flowchart += f'    {node_id}["{step_text}"]\n'
            is_loop.append(any(kw in step for kw in loop_keywords))

        flowchart += "    Z([结束])\n"

        if node_ids:
            flowchart += f"    A --> {node_ids[0]}\n"
            for i in range(len(node_ids) - 1):
                flowchart += f"    {node_ids[i]} --> {node_ids[i + 1]}\n"
                if is_loop[i]:
                    flowchart += f"    {node_ids[i]} -.->|继续循环| {node_ids[i]}\n"
            if is_loop[-1]:
                flowchart += f"    {node_ids[-1]} -.->|继续循环| {node_ids[-1]}\n"
            flowchart += f"    {node_ids[-1]} --> Z\n"

        return flowchart
    
    @classmethod
    def _generate_minimal(cls, title: str, steps: List[str]) -> str:
        """生成最简化的流程图"""
        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart LR\n"
        flowchart += "    A([开始])\n"
        
        # 只保留3-5个步骤
        limited_steps = steps[:5] if len(steps) > 5 else steps
        node_ids = []
        
        for i, step in enumerate(limited_steps):
            node_id = f"N{i + 1}"
            node_ids.append(node_id)
            step_text = step.replace('"', "'")
            flowchart += f'    {node_id}["{step_text}"]\n'
        
        flowchart += "    Z([结束])\n"
        
        # 确保所有子块都连接到开始节点
        if node_ids:
            connections = " --> ".join(node_ids)
            flowchart += f"    A --> {connections} --> Z\n"
        
        return flowchart
    
    @classmethod
    def _generate_simplified_flowchart(cls, title: str, steps: List[str]) -> str:
        """生成简化版流程图，确保结构合理"""
        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart TD\n"
        flowchart += "    A([开始])\n"
        
        limited_steps = steps[:6]  # 最多6个步骤
        node_ids = []
        
        for i, step in enumerate(limited_steps):
            node_id = f"N{i + 1}"
            node_ids.append(node_id)
            step_text = step.replace('"', "'")
            flowchart += f'    {node_id}["{step_text}"]\n'
        
        flowchart += "    Z([结束])\n"
        
        # 确保所有子块都连接到开始节点
        if node_ids:
            flowchart += f"    A --> {node_ids[0]}\n"
            for i in range(len(node_ids) - 1):
                flowchart += f"    {node_ids[i]} --> {node_ids[i + 1]}\n"
            flowchart += f"    {node_ids[-1]} --> Z\n"
        
        return flowchart


class MermaidGenerator:
    """Mermaid图表生成器 - 生成带有分支的流程图"""
    
    def generate_flow_diagram_from_steps(self, title: str, steps: List[str]) -> str:
        """根据提供的步骤列表生成流程图，并进行合理性校验"""
        # 首先生成流程图
        flowchart = self._generate_flowchart_internal(title, steps)
        
        # 校验流程图合理性
        validation = FlowchartValidator.validate_flowchart(flowchart)
        
        if not validation['is_valid']:
            print(f"    流程图校验发现问题：")
            for issue in validation['issues']:
                print(f"      - {issue['description']}")
            # 重新生成合理的流程图
            flowchart = FlowchartValidator.regenerate_if_invalid(flowchart, title, steps)
        else:
            loop_info = f"，{validation['loop_count']}个循环" if validation.get('loop_count', 0) > 0 else ""
            print(f"    流程图校验通过（{validation['steps_count']}个步骤，{validation['decision_count']}个决策节点{loop_info}）")
        
        return flowchart
    
    def _generate_flowchart_internal(self, title: str, steps: List[str]) -> str:
        """内部方法：生成流程图（不包含校验）

        根据步骤内容自动插入条件分支和循环结构：
        - 包含 BRANCH_KEYWORDS 的步骤生成菱形决策节点（成功/失败分支）
        - 包含 LOOP_KEYWORDS 的步骤生成循环回边结构
        """
        branches = self._detect_branch_points(steps)
        loops = self._detect_loop_points(steps)

        branch_indices = {b['step_index'] for b in branches}
        loop_indices = {lp['step_index'] for lp in loops}

        flowchart = f"%% {title} 流程图\n"
        flowchart += "flowchart TD\n"
        flowchart += "    A([开始])\n"

        # --- 1. 声明所有节点 ---
        node_ids = []
        node_types = []  # 'process' | 'decision' | 'loop'

        for i, step in enumerate(steps):
            node_id = f"N{i + 1}"
            node_ids.append(node_id)
            escaped = step.replace('"', "'").replace('\n', ' ')

            if i in branch_indices:
                flowchart += f'    {node_id}{{"{escaped}"}}\n'
                node_types.append('decision')
            else:
                flowchart += f'    {node_id}["{escaped}"]\n'
                if i in loop_indices:
                    node_types.append('loop')
                else:
                    node_types.append('process')

        flowchart += "    Z([结束])\n"

        # 错误处理节点：仅在存在决策节点时添加
        has_decision = 'decision' in node_types
        if has_decision:
            flowchart += '    E["错误处理"]\n'

        # --- 2. 连接节点 ---
        # A -> 第一个节点
        if node_ids:
            flowchart += f"    A --> {node_ids[0]}\n"

        for i in range(len(node_ids)):
            nid = node_ids[i]
            ntype = node_types[i]
            next_nid = node_ids[i + 1] if i + 1 < len(node_ids) else 'Z'

            if ntype == 'decision':
                # 菱形节点：成功走下一步，失败走错误处理
                kw = next((b['keyword'] for b in branches if b['step_index'] == i), '')
                labels = self.BRANCH_KEYWORDS.get(kw, {'success': '是', 'fail': '否'})
                flowchart += f"    {nid} -->|{labels['success']}| {next_nid}\n"
                if has_decision:
                    flowchart += f"    {nid} -->|{labels['fail']}| E\n"
            elif ntype == 'loop':
                # 循环节点：正常走下一步，并有回边到自身
                flowchart += f"    {nid} --> {next_nid}\n"
                flowchart += f"    {nid} -.->|继续循环| {nid}\n"
            else:
                # 普通处理节点
                flowchart += f"    {nid} --> {next_nid}\n"

        # 错误处理 -> 结束
        if has_decision:
            flowchart += "    E --> Z\n"

        return flowchart
    
    BRANCH_KEYWORDS = {
        '验证': {'success': '验证通过', 'fail': '验证失败'},
        '检查': {'success': '检查通过', 'fail': '检查失败'},
        '初始化': {'success': '初始化成功', 'fail': '初始化失败'},
        '执行': {'success': '执行成功', 'fail': '执行失败'},
        '加载': {'success': '加载成功', 'fail': '加载失败'},
        '保存': {'success': '保存成功', 'fail': '保存失败'},
        '配置': {'success': '配置成功', 'fail': '配置失败'},
        '处理': {'success': '处理完成', 'fail': '处理异常'},
        '判断': {'success': '条件满足', 'fail': '条件不满足'},
        '匹配': {'success': '匹配成功', 'fail': '匹配失败'},
        '解析': {'success': '解析成功', 'fail': '解析失败'},
        '确认': {'success': '确认通过', 'fail': '确认失败'},
        '校验': {'success': '校验通过', 'fail': '校验失败'},
        '是否': {'success': '是', 'fail': '否'},
    }

    LOOP_KEYWORDS = [
        '遍历', '循环', '迭代', '逐一', '逐个', '依次',
        '重复', '批量', '轮询', '扫描', '每个', '所有',
        '对每', '针对每', '分别',
    ]
    
    def generate_flow_diagram(self, title: str, logic: str) -> str:
        """生成带有分支的流程图，并进行合理性校验"""
        steps = self._extract_flow_steps(logic)
        # 直接调用带校验的方法
        return self.generate_flow_diagram_from_steps(title, steps)
    
    def _detect_branch_points(self, steps: List[str]) -> List[Dict]:
        """检测需要分支的步骤"""
        branches = []
        for i, step in enumerate(steps):
            for keyword in self.BRANCH_KEYWORDS.keys():
                if keyword in step:
                    branches.append({
                        'step_index': i,
                        'keyword': keyword
                    })
                    break
        return branches

    def _detect_loop_points(self, steps: List[str]) -> List[Dict]:
        """检测需要循环结构的步骤"""
        loops = []
        for i, step in enumerate(steps):
            for keyword in self.LOOP_KEYWORDS:
                if keyword in step:
                    loops.append({
                        'step_index': i,
                        'keyword': keyword
                    })
                    break
        return loops
    
    def _summarize_feature_title(self, title: str) -> str:
        """概括特性标题
        
        将完整的需求标题概括为简短的特性名称，移除常见前缀词。
        不限制字符数，保留完整的语义。
        
        例如：
            "增加一个将reshape op替换为view op的pass" -> "将reshape op替换为view op的pass"
            "将platform组件拆分为最底层组件" -> "将platform组件拆分为最底层组件"
        
        Args:
            title: 原始需求标题
            
        Returns:
            概括后的特性名称（保留完整语义，不截断）
        """
        # 移除常见的前缀词
        prefixes = ['增加一个', '新增一个', '开发一个', '实现一个', '添加一个', 
                    '增加', '新增', '开发', '实现', '添加', '创建一个', '创建']
        cleaned_title = title
        for prefix in prefixes:
            if cleaned_title.startswith(prefix):
                cleaned_title = cleaned_title[len(prefix):]
                break
        
        # 直接返回清理后的标题，不限制字符数
        return cleaned_title

    def _is_generic_component_name(self, comp_name: str) -> bool:
        """检测是否为通用/无意义的功能模块名称
        
        检测"功能模块x"、"功能x"等形式的命名
        
        Args:
            comp_name: 功能模块名称
            
        Returns:
            True如果是通用命名，False如果有实际含义
        """
        generic_patterns = [
            r'^功能模块\d+$',  # 功能模块1, 功能模块2
            r'^功能\d+$',       # 功能1, 功能2
            r'^模块\d+$',       # 模块1, 模块2
            r'^子模块\d+$',     # 子模块1, 子模块2
            r'^组件\d+$',       # 组件1, 组件2
        ]
        for pattern in generic_patterns:
            if re.match(pattern, comp_name):
                return True
        return False

    def generate_architecture_diagram(self, requirement: Requirement, project_components: List[str] = None, max_retries: int = 3) -> str:
        """基于expanded_description生成详细架构关系图
        
        三层结构：
        - 第一层：核心功能（主节点）- 使用概括的特性名称
        - 第二层：核心功能的子模块（2~5个）
        - 第三层：子模块所涵盖的功能（每个子模块2~5个功能）- 不能使用"功能模块x"形式
        
        Args:
            requirement: 需求对象
            project_components: 项目中已有的组件名称列表（可选，此参数保留但不再使用）
            max_retries: 最大重试次数，用于处理通用命名的情况
            
        Raises:
            ValueError: 当重试次数耗尽仍无法生成有效的第三层功能名称时
        """
        title = requirement.title
        # 使用概括的特性名称作为第一层主节点
        summarized_title = self._summarize_feature_title(title)
        
        diagram = f"%% {title} 架构图\n"
        diagram += "graph TD\n"
        diagram += f'    MAIN["{summarized_title}"]\n\n'

        # 获取子模块列表，限制在2~5个
        sub_requirements = requirement.sub_requirements
        if len(sub_requirements) > 5:
            sub_requirements = sub_requirements[:5]
        
        # 用于收集所有第三层功能名称，便于后续验证
        all_third_layer_components = []
        
        # 添加子模块（第二层）
        for i, sub_req in enumerate(sub_requirements):
            mod_id = f"M{i + 1}"
            # 放宽字符限制：从12增加到18
            mod_label = sub_req.title[:18] if len(sub_req.title) > 18 else sub_req.title
            diagram += f'    {mod_id}["{mod_label}"]\n'
            diagram += f'    MAIN --> {mod_id}\n'

            # 生成子模块的功能（第三层），限制在2~5个
            comps = self._generate_sub_components(sub_req.title, sub_req.logic, 2)
            # 确保功能数量在2~5之间
            if len(comps) < 2:
                # 不使用通用命名，尝试从description中提取
                if hasattr(sub_req, 'description') and sub_req.description:
                    extracted = self._extract_components(sub_req.description, 2)
                    comps = extracted if extracted else []
                if len(comps) < 2:
                    # 使用子模块标题的关键词生成
                    comps = self._generate_meaningful_components(sub_req.title, 2 - len(comps), existing=comps)
            elif len(comps) > 5:
                comps = comps[:5]
            
            # 收集第三层功能名称
            all_third_layer_components.extend(comps)
            
            for j, comp in enumerate(comps):
                comp_id = f"M{i + 1}S{j + 1}"
                # 放宽字符限制：从12增加到15
                comp_label = comp[:15] if len(comp) > 15 else comp
                diagram += f'    {comp_id}["{comp_label}"]\n'
                diagram += f'    {mod_id} --> {comp_id}\n'

        # 验证第三层是否存在通用命名
        invalid_components = [c for c in all_third_layer_components if self._is_generic_component_name(c)]
        if invalid_components:
            raise ValueError(f"架构图第三层存在无意义的功能命名: {invalid_components}。"
                           f"请提供有实际含义的功能名称，如'参数加载'、'配置验证'等，而非'功能模块x'形式。"
                           f"建议检查子需求的description或logic字段，确保包含足够的信息用于生成有意义的功能名称。")

        return diagram

    def _generate_meaningful_components(self, title: str, count: int, existing: List[str] = None) -> List[str]:
        """根据子模块标题生成有意义的功能组件名称
        
        Args:
            title: 子模块标题
            count: 需要生成的数量
            existing: 已存在的组件名称列表
            
        Returns:
            有意义的功能组件名称列表
        """
        if existing is None:
            existing = []
        
        components = list(existing)
        
        # 根据标题关键词生成有意义的组件
        meaningful_map = {
            '识别': ['特征提取', '条件匹配'],
            '检测': ['规则校验', '结果判定'],
            '构造': ['对象创建', '属性设置'],
            '创建': ['实例化', '初始化配置'],
            '生成': ['数据处理', '结果输出'],
            '更新': ['状态同步', '数据刷新'],
            '替换': ['源目标映射', '转换执行'],
            '转换': ['格式解析', '类型适配'],
            '处理': ['输入解析', '结果封装'],
            '分析': ['数据采集', '结果统计'],
            '管理': ['状态维护', '资源调度'],
            '配置': ['参数读取', '设置应用'],
            '验证': ['规则检查', '结果确认'],
            '计算': ['输入准备', '运算执行'],
            '执行': ['任务分发', '状态跟踪'],
            '加载': ['资源定位', '数据读取'],
            '保存': ['数据序列化', '存储写入'],
            '查询': ['条件解析', '结果返回'],
            '监控': ['指标采集', '状态报告'],
        }
        
        for keyword, comps in meaningful_map.items():
            if keyword in title:
                for comp in comps:
                    if comp not in components:
                        components.append(comp)
                    if len(components) >= count + len(existing):
                        break
                break
        
        # 如果仍然不足，使用通用但有意义的名称
        fallback_names = ['数据预处理', '核心处理', '结果输出', '状态管理', '错误处理']
        for name in fallback_names:
            if len(components) >= count + len(existing):
                break
            if name not in components:
                components.append(name)
        
        return components[:count + len(existing)]

    def _extract_project_components_from_deps(self, deps_text: str) -> List[str]:
        """从依赖描述中提取项目组件名称"""
        components = []
        
        # 匹配"xxx模块"、"xxx组件"等模式
        patterns = [
            r'([\u4e00-\u9fa5]+(?:模块|组件|服务|处理器|管理器))',
            r'([\u4e00-\u9fa5]+(?:Manager|Compiler|Executor|Monitor|Handler))',
            r'([A-Z][a-zA-Z]+(?:Manager|Compiler|Executor|Monitor|Handler))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, deps_text)
            for match in matches:
                if match and len(match) >= 3 and len(match) <= 20:
                    if match not in components:
                        components.append(match)
        
        return components[:4]  # 最多返回4个

    def _generate_sub_components(self, title: str, logic: str, count: int) -> List[str]:
        """根据子模块标题和逻辑生成有意义的子组件名称（名词或名词+动词形式）
        
        若无法生成足够的有意义功能名称，将抛出ValueError异常，
        要求用户提供更详细的子需求描述信息。
        
        Args:
            title: 子模块标题
            logic: 子模块逻辑描述
            count: 需要生成的功能数量
            
        Returns:
            有意义的功能组件名称列表
            
        Raises:
            ValueError: 当无法从标题和逻辑中提取足够的有意义功能名称时
        """
        components = []
        
        # 策略1：根据标题关键词匹配预设的功能名称
        title_keywords = {
            '硬件抽象层': ['接口定义', '适配器管理'],
            '设备管理': ['资源分配', '状态监控'],
            '配置管理': ['参数加载', '配置验证'],
            '性能监控': ['数据采集', '指标分析'],
            '初始化': ['环境检查', '资源准备'],
            '执行': ['任务调度', '结果处理'],
            '验证': ['规则检查', '结果输出'],
            '处理': ['数据预处理', '核心计算'],
            '加载': ['文件解析', '数据校验'],
            '保存': ['数据序列化', '文件写入'],
            '计算': ['算法选择', '结果缓存'],
            '查询': ['条件解析', '结果聚合'],
            '注册': ['唯一性校验', '表维护'],
            '注销': ['资源清理', '状态更新'],
            '更新': ['数据定位', '事务处理'],
            '删除': ['条件验证', '资源释放'],
            '监控': ['指标采集', '告警检测'],
            '优化': ['瓶颈分析', '策略选择'],
            '识别': ['特征提取', '条件匹配'],
            '检测': ['规则校验', '结果判定'],
            '构造': ['对象创建', '属性设置'],
            '创建': ['实例化', '初始化配置'],
            '生成': ['数据处理', '结果输出'],
            '替换': ['源目标映射', '转换执行'],
            '转换': ['格式解析', '类型适配'],
            '分析': ['数据采集', '结果统计'],
            '管理': ['状态维护', '资源调度'],
            '合并': ['操作融合', '属性合并'],
            '场景': ['条件判断', '结果筛选'],
            '图结构': ['节点更新', '依赖维护'],
            '集成': ['接口对接', '功能组合'],
        }
        
        for keyword, comps in title_keywords.items():
            if keyword in title:
                components = comps[:count]
                break
        
        # 策略2：从logic中提取名词+模块/组件等模式
        if not components:
            noun_patterns = [
                r'([\u4e00-\u9fa5]{2,4})(?:模块|组件|服务|接口|管理器|处理器)',
                r'([\u4e00-\u9fa5]{2,4})(?:层|器|池|表|缓存)',
            ]
            for pattern in noun_patterns:
                for match in re.finditer(pattern, logic):
                    term = match.group(1)
                    suffix = match.group(0)[-2:] if len(match.group(0)) >= 2 else '模块'
                    full_term = term + suffix
                    if full_term not in components and 4 <= len(full_term) <= 8:
                        components.append(full_term)
                if len(components) >= count:
                    break
        
        # 策略3：匹配标题中的设计/重构等关键词
        default_components = {
            '设计': ['方案设计', '接口定义'],
            '重构': ['模块拆分', '接口优化'],
            '优化': ['性能提升', '代码改进'],
            '集成': ['接口对接', '数据同步'],
        }
        
        if len(components) < count:
            for keyword, comps in default_components.items():
                if keyword in title:
                    for comp in comps:
                        if comp not in components:
                            components.append(comp)
                        if len(components) >= count:
                            break
                    break
        
        # 策略4：从logic文本中提取动词+名词形式的功能名称
        if len(components) < count:
            action_patterns = [
                r'(\d+[\.\)、]\s*)?([\u4e00-\u9fa5]{2,4})([\u4e00-\u9fa5]{2,4})',
            ]
            for pattern in action_patterns:
                for match in re.finditer(pattern, logic):
                    action = match.group(2) if match.lastindex >= 2 else ''
                    obj = match.group(3) if match.lastindex >= 3 else ''
                    if action and obj:
                        comp = action + obj
                        if comp not in components and 4 <= len(comp) <= 8:
                            components.append(comp)
                        if len(components) >= count:
                            break
                if len(components) >= count:
                    break
        
        # 检查是否生成了足够的有意义功能名称
        if len(components) < count:
            raise ValueError(
                f"无法为子模块 '{title}' 生成足够的({count}个)有意义的功能名称。\n"
                f"当前已生成: {components}\n"
                f"请在扩写JSON的sub_requirements中为该子模块提供更详细的description字段，\n"
                f"确保description包含足够的功能描述信息，以便生成有意义的功能名称。\n"
                f"建议格式：描述该模块包含的具体功能，如'负责参数加载、配置验证、错误处理等功能'。"
            )
        
        return components[:count]

    def _extract_components(self, text: str, count: int) -> List[str]:
        """从描述文本中提取关键子组件名称"""
        found = []
        patterns = [
            r'([\u4e00-\u9fa5]{2,6}(?:模块|格式|层|接口|机制|策略|工具|文件|数据))',
            r'([\u4e00-\u9fa5]{2,6}(?:验证|分析|配置|管理|采集|处理|存储|加载|序列化))',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1)
                if term not in found and 3 <= len(term) <= 10:
                    found.append(term)
            if len(found) >= count:
                break
        return found[:count]

    def _extract_tech_deps(self, text: str) -> List[str]:
        """从描述文本中提取技术依赖名称"""
        found = []
        patterns = [
            r'([\u4e00-\u9fa5]{2,8}(?:框架|平台|工具链|引擎|库|体系))',
            r'([A-Z][a-zA-Z]{2,14}(?:Tensors?|Format|Type|API|SDK))',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1)
                if term not in found and len(term) <= 15:
                    found.append(term)
        return found[:3]
    
    def _extract_flow_steps(self, logic: str) -> List[str]:
        """从逻辑描述中提取关键步骤 - 仅作为fallback使用
        
        注意：推荐通过扩写JSON的flow_steps字段提供流程步骤，由Agent通过搜索项目文档、
        接口信息和网络资料生成。此方法仅在未提供flow_steps时作为fallback。
        """
        print("提示：未在扩写JSON中提供流程步骤，请通过搜索项目文档、接口信息和网络资料生成flow_steps")
        return ['初始化环境', '检查参数', '执行任务', '验证结果', '返回状态']


class MermaidRenderer:
    """Mermaid 图表渲染器"""
    
    def __init__(self):
        self.mermaid_cmd = self._find_mermaid_cmd()
        self.mermaid_cli_available = self.mermaid_cmd is not None
    
    def _find_mermaid_cmd(self) -> Optional[str]:
        """查找 mermaid-cli 命令"""
        # 尝试多种可能的命令名称
        possible_commands = ['mmdc', 'mmdc.cmd', 'mmdc.exe']
        
        for cmd in possible_commands:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=True
                )
                if result.returncode == 0:
                    return cmd
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        # 尝试从 npm 全局路径查找
        try:
            import shutil
            cmd_path = shutil.which('mmdc')
            if cmd_path:
                return cmd_path
        except Exception:
            pass
        
        return None
    
    def render_to_png(self, mermaid_code: str) -> bytes:
        """将 mermaid 代码渲染为 PNG，失败时抛出异常"""
        if not self.mermaid_cli_available:
            raise RuntimeError("mermaid-cli 未安装或不可用，请执行：npm install -g @mermaid-js/mermaid-cli")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as f:
                f.write(mermaid_code)
                input_file = f.name
            
            output_file = input_file.replace('.mmd', '.png')
            
            result = subprocess.run(
                [self.mermaid_cmd, '-i', input_file, '-o', output_file, '-b', 'white', '-s', '3'],
                capture_output=True,
                timeout=60,
                shell=True
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "未知错误"
                raise RuntimeError(f"Mermaid 渲染失败：{error_msg}")
            
            if not os.path.exists(output_file):
                raise RuntimeError("Mermaid 渲染失败：输出文件未生成")
            
            with open(output_file, 'rb') as f:
                png_data = f.read()
            
            os.unlink(input_file)
            os.unlink(output_file)
            return png_data
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Mermaid 渲染超时（30秒）")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Mermaid 渲染失败：{str(e)}")


class SRSDocumentGenerator:
    """SRS 文档生成器"""
    
    def __init__(self, config: SRSConfig):
        self.config = config
        self.document = Document()
        self.mermaid_renderer = MermaidRenderer()
        self.figure_counter = 0  # 图号计数器
        self.table_counter = 0   # 表号计数器
        self._setup_styles()
    
    def _setup_styles(self):
        """设置文档样式"""
        # 设置默认字体
        style = self.document.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(10.5)
        style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    
    def _set_chinese_font(self, run, font_name: str, font_size: int):
        """设置中英文字体"""
        run.font.name = 'Times New Roman'
        run.font.size = Pt(font_size)
        run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    
    def _add_heading(self, text: str, level: int):
        """添加标题
        
        Args:
            text: 标题文本
            level: 标题级别 (1-5)
                   1级=14磅黑体, 2级=12磅黑体, 3级=12磅黑体
                   4级=11磅黑体, 5级=10.5磅黑体
        """
        heading = self.document.add_heading(text, level=level)
        for run in heading.runs:
            # 根据级别设置字体大小
            if level == 1:
                self._set_chinese_font(run, '黑体', 14)
            elif level in [2, 3]:
                self._set_chinese_font(run, '黑体', 12)
            elif level == 4:
                self._set_chinese_font(run, '黑体', 11)
            elif level == 5:
                self._set_chinese_font(run, '黑体', 10.5)
            else:
                self._set_chinese_font(run, '黑体', 12)
            run.bold = True
    
    def _add_paragraph(self, text: str, bold: bool = False):
        """添加段落（段首空两格）"""
        para = self.document.add_paragraph()
        # 添加两个全角空格作为段首缩进
        run = para.add_run("　　")
        self._set_chinese_font(run, '宋体', 10.5)
        # 添加正文内容
        run = para.add_run(text)
        run.bold = bold
        self._set_chinese_font(run, '宋体', 10.5)
        return para
    
    def _add_code_block(self, code: str, language: str = ""):
        """添加代码块"""
        para = self.document.add_paragraph()
        run = para.add_run(f"```{language}\n{code}\n```")
        run.font.name = 'Consolas'
        run.font.size = Pt(9)
    
    def _add_mermaid_diagram(self, mermaid_code: str, caption: str = "", diagram_type: str = "architecture"):
        """添加 Mermaid 图表
        
        Args:
            mermaid_code: Mermaid代码
            caption: 图表标题
            diagram_type: 图表类型，"architecture"为架构图，"flow"为流程图
        """
        try:
            png_data = self.mermaid_renderer.render_to_png(mermaid_code)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(png_data)
                temp_path = f.name

            from docx.shared import Cm
            
            arch_size_cm = 15.0
            flow_size_cm = 12.0
            fixed_size_cm = arch_size_cm if diagram_type == "architecture" else flow_size_cm

            try:
                from PIL import Image
                with Image.open(temp_path) as img:
                    img_width, img_height = img.size
                    dpi = img.info.get('dpi', (96, 96))[0] or 96

                if img_width > 0 and img_height > 0:
                    if diagram_type == "architecture":
                        picture = self.document.add_picture(temp_path, width=Cm(fixed_size_cm))
                    else:
                        picture = self.document.add_picture(temp_path, height=Cm(fixed_size_cm))
                    
                    actual_width_cm = picture.width.cm
                    actual_height_cm = picture.height.cm
                    if diagram_type == "architecture":
                        if abs(actual_width_cm - fixed_size_cm) > 0.1:
                            raise RuntimeError(
                                f"架构图宽度校验失败：实际宽度 {actual_width_cm:.2f}cm，"
                                f"要求宽度 {fixed_size_cm}cm"
                            )
                    else:
                        if abs(actual_height_cm - fixed_size_cm) > 0.1:
                            raise RuntimeError(
                                f"流程图高度校验失败：实际高度 {actual_height_cm:.2f}cm，"
                                f"要求高度 {fixed_size_cm}cm"
                            )
                else:
                    if diagram_type == "architecture":
                        picture = self.document.add_picture(temp_path, width=Cm(fixed_size_cm))
                    else:
                        picture = self.document.add_picture(temp_path, height=Cm(fixed_size_cm))
            except ImportError:
                if diagram_type == "architecture":
                    picture = self.document.add_picture(temp_path, width=Cm(fixed_size_cm))
                else:
                    picture = self.document.add_picture(temp_path, height=Cm(fixed_size_cm))
            
            last_paragraph = self.document.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            os.unlink(temp_path)
            
            self.figure_counter += 1
            caption_text = f"图{self.figure_counter} {caption}" if caption else f"图{self.figure_counter}"
            caption_para = self.document.add_paragraph(caption_text)
            caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in caption_para.runs:
                self._set_chinese_font(run, '宋体', 10.5)
                run.bold = True
                
        except RuntimeError as e:
            raise RuntimeError(f"Mermaid 图表转换失败，无法生成 SRS 文档：{str(e)}")
    
    def _add_interface_table(self, interfaces: List[FunctionInterface]):
        """添加接口表格（每个接口一个独立表格，转置格式）"""
        if not interfaces:
            self._add_paragraph("无新增接口。")
            return
        
        # 为每个接口创建一个独立的表格
        for idx, interface in enumerate(interfaces, 1):
            # 添加居中的表格说明（如"表1 ExecuteData接口"）
            self.table_counter += 1
            caption_para = self.document.add_paragraph(f"表{self.table_counter} {interface.name}接口")
            caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in caption_para.runs:
                self._set_chinese_font(run, '宋体', 10.5)
                run.bold = True
            
            # 创建表格（7行2列，转置格式）
            table = self.document.add_table(rows=7, cols=2)
            table.style = 'Table Grid'
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            
            # 定义表头和对应的数据
            table_data = [
                ('函数原型', interface.prototype()),
                ('函数功能', interface.description if interface.description else "功能描述待补充"),
                ('增加输入', interface.input_desc if interface.input_desc else "NA"),
                ('函数输出', interface.output_desc if interface.output_desc else "NA"),
                ('返回', interface.return_desc if interface.return_desc else f"{interface.return_type}"),
                ('使用说明', interface.usage_desc if interface.usage_desc else "NA"),
                ('注意事项', interface.caution_desc if interface.caution_desc else "NA")
            ]
            
            # 填充表格数据
            for row_idx, (header, value) in enumerate(table_data):
                # 第一列：表头
                cell = table.rows[row_idx].cells[0]
                cell.text = header
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
                        self._set_chinese_font(run, '黑体', 10)
                
                # 第二列：数据值
                cell = table.rows[row_idx].cells[1]
                cell.text = value
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        self._set_chinese_font(run, '宋体', 10)
            
            # 表格之间添加空行
            if idx < len(interfaces):
                self._add_paragraph("")
    
    def generate(self, requirements: List[Requirement], output_path: str):
        """生成 SRS 文档
        
        文档结构：
        - 2. 特性描述 (一级标题)
          - 2.x {需求标题} (三级标题)
            - 2.x.1 功能需求 (四级标题)
            ...
        """
        if not requirements:
            raise ValueError("需求列表为空，无法生成 SRS 文档")
        
        # 文档一级标题：2. 特性描述
        self._add_heading("2. 特性描述", level=1)
        
        # 生成每个需求的章节
        for idx, req in enumerate(requirements, 1):
            self._generate_requirement_section(req, idx)
        
        # 保存文档
        self.document.save(output_path)
        print(f"SRS 文档已生成：{output_path}")
    
    def _generate_requirement_section(self, req: Requirement, index: int):
        """生成需求章节
        
        排版结构（所有标题加深一层）：
        - 2.x {需求标题} (二级，对应原三级)
          - 2.x.1 功能需求 (三级，对应原四级)
            - 2.x.1.1 模块划分 (四级，对应原五级)
            - 2.x.1.2 特性依赖 (四级，对应原五级)
              - 2.x.1.2.1 组件依赖及关系 (五级，对应原六级)
              - 2.x.1.2.2 共架构分析 (五级，对应原六级)
              - 2.x.1.2.3 特性交叉 (五级，对应原六级)
            - 2.x.1.3 介绍 (四级，对应原五级)
            - 2.x.1.4 流程图 (四级，对应原五级)
          - 2.x.2 性能需求 (三级，对应原四级)
          - 2.x.3 质量需求 (三级，对应原四级)
            - 2.x.3.1 可维护性 (四级，对应原五级)
            - 2.x.3.2 可测试性 (四级，对应原五级)
            - 2.x.3.3 历史债务 (四级，对应原五级)
          - 2.x.4 软件接口 (三级，对应原四级)
            - 2.x.4.1 对外接口依赖说明 (四级，对应原五级)
          - 2.x.5 详细设计 (三级，对应原四级)
            - 2.x.5.1 数据描述 (四级，对应原五级)
            - 2.x.5.2 函数设计 (四级，对应原五级)
        """
        # 2.x {需求标题} (二级标题，原三级)
        self._add_heading(f"2.{index} {req.title}", level=2)
        
        # 2.x.1 功能需求 (三级标题，原四级)
        self._add_heading(f"2.{index}.1 功能需求", level=3)
        
        # 2.x.1.1 模块划分 (四级标题，原五级)
        self._add_heading(f"2.{index}.1.1 模块划分", level=4)
        
        for sub_idx, sub_req in enumerate(req.sub_requirements, 1):
            # 模块名单独成行，开头不空格，结尾不加冒号
            para = self.document.add_paragraph()
            run = para.add_run(f"({sub_idx}) {sub_req.title}")
            run.bold = True
            self._set_chinese_font(run, '黑体', 10.5)
            
            # 后续内容换行并空两格
            if sub_req.description:
                self._add_paragraph(sub_req.description)
            if sub_req.logic:
                self._add_paragraph(sub_req.logic)
        
        # 2.x.1.2 特性依赖 (四级标题，原五级)
        self._add_heading(f"2.{index}.1.2 特性依赖", level=4)
        
        # 2.x.1.2.1 组件依赖及关系 (五级标题，原六级)
        self._add_heading(f"2.{index}.1.2.1 组件依赖及关系", level=5)
        if req.expanded_dependencies:
            self._add_paragraph(req.expanded_dependencies)
        elif req.dependencies:
            for dep in req.dependencies:
                self._add_paragraph(f"- {dep}")
        else:
            self._add_paragraph("无特殊依赖。")
        
        # 2.x.1.2.2 共架构分析 (五级标题，原六级)
        self._add_heading(f"2.{index}.1.2.2 共架构分析", level=5)
        if req.architecture_analysis:
            self._add_paragraph(req.architecture_analysis)
        else:
            self._add_paragraph("本次更新为独立模块，不影响现有架构。")
        
        # 2.x.1.2.3 特性交叉 (五级标题，原六级)
        self._add_heading(f"2.{index}.1.2.3 特性交叉", level=5)
        if req.feature_crossover:
            self._add_paragraph(req.feature_crossover)
        else:
            self._add_paragraph("本次更新与其他特性无交叉影响。")
        
        # 2.x.1.3 介绍 (四级标题，原五级)
        self._add_heading(f"2.{index}.1.3 介绍", level=4)
        
        # 背景信息介绍
        if req.expanded_background:
            self._add_paragraph(req.expanded_background)
        elif req.background:
            self._add_paragraph(req.background)
        
        # 需求描述
        if req.expanded_description:
            self._add_paragraph(req.expanded_description)
        elif req.description:
            self._add_paragraph(req.description)
        
        # 架构图（迁移到介绍章节中）
        if req.mermaid_arch:
            self._add_mermaid_diagram(req.mermaid_arch, f"{req.title}架构图", diagram_type="architecture")
        
        # 2.x.1.4 流程图 (四级标题，原五级)
        self._add_heading(f"2.{index}.1.4 流程图", level=4)
        
        # 各子模块流程图
        for sub_idx, sub_req in enumerate(req.sub_requirements, 1):
            if sub_req.mermaid_flow:
                self._add_mermaid_diagram(sub_req.mermaid_flow, f"{sub_req.title}流程图", diagram_type="flow")
        
        # 2.x.2 性能需求 (三级标题，原四级)
        self._add_heading(f"2.{index}.2 性能需求", level=3)
        if req.performance_requirements:
            self._add_paragraph(req.performance_requirements)
        else:
            self._add_paragraph("本需求对性能无特殊要求。")
        
        # 2.x.3 质量需求 (三级标题，原四级)
        self._add_heading(f"2.{index}.3 质量需求", level=3)
        
        # 2.x.3.1 可维护性 (四级标题，原五级)
        self._add_heading(f"2.{index}.3.1 可维护性", level=4)
        self._add_paragraph("代码需遵循PyPTO项目的编码规范，使用统一的命名风格和注释规范。采用模块化设计，各模块职责单一，接口清晰。")
        
        # 2.x.3.2 可测试性 (四级标题，原五级)
        self._add_heading(f"2.{index}.3.2 可测试性", level=4)
        self._add_paragraph("需编写完整的单元测试用例，覆盖核心功能的正常场景、边界场景和异常场景。测试覆盖率应不低于80%。")
        
        # 2.x.3.3 历史债务 (四级标题，原五级)
        self._add_heading(f"2.{index}.3.3 历史债务", level=4)
        self._add_paragraph("本次更新不涉及历史债务处理。")
        
        # 2.x.4 软件接口 (三级标题，原四级)
        self._add_heading(f"2.{index}.4 软件接口", level=3)
        
        # 2.x.4.1 对外接口依赖说明 (四级标题，原五级)
        self._add_heading(f"2.{index}.4.1 对外接口依赖说明", level=4)
        if req.expanded_external_interfaces:
            self._add_paragraph(req.expanded_external_interfaces)
        else:
            self._add_paragraph("本需求无特殊对外接口依赖。")
        
        # 2.x.5 详细设计 (三级标题，原四级)
        self._add_heading(f"2.{index}.5 详细设计", level=3)
        
        # 2.x.5.1 数据描述 (四级标题，原五级)
        self._add_heading(f"2.{index}.5.1 数据描述", level=4)
        if req.data_description:
            self._add_paragraph(req.data_description)
        else:
            self._add_paragraph("本需求不涉及新增数据结构。")
        
        # 2.x.5.2 函数设计 (四级标题，原五级)
        self._add_heading(f"2.{index}.5.2 函数设计", level=4)
        self._add_interface_table(req.interfaces)
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        filename = re.sub(r'[\\/:*?"<>|\s]', '_', filename)
        if len(filename) > 50:
            filename = filename[:50]
        return filename or "SRS_文档"


def parse_json_file(json_path: str) -> List[Requirement]:
    """解析 JSON 文件"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 文件不存在：{json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误：{e}")
    
    if 'requirements' not in data:
        raise ValueError("缺少必需字段：requirements")
    
    requirements_data = data['requirements']
    if not requirements_data:
        raise ValueError("需求列表为空")
    
    requirements = []
    for req_data in requirements_data:
        if 'title' not in req_data:
            raise ValueError("缺少必需字段：title")
        
        sub_requirements = []
        for sub_data in req_data.get('sub_requirements', []):
            sub_req = SubRequirement(
                title=sub_data.get('title', ''),
                logic=sub_data.get('logic', '')
            )
            sub_requirements.append(sub_req)
        
        # 处理 dependencies 字段
        dependencies = req_data.get('dependencies', None)
        # 检查是否为空数组（明确无依赖）
        has_no_dependencies = isinstance(dependencies, list) and len(dependencies) == 0
        
        requirement = Requirement(
            title=req_data['title'],
            background=req_data.get('background', ''),
            description=req_data.get('description', ''),
            sub_requirements=sub_requirements,
            dependencies=dependencies if dependencies else [],
            # 新增字段
            architecture_analysis=req_data.get('architecture_analysis', ''),
            feature_crossover=req_data.get('feature_crossover', ''),
            performance_requirements=req_data.get('performance_requirements', ''),
            quality_requirements=req_data.get('quality_requirements', ''),
            data_description=req_data.get('data_description', ''),
            expanded_external_interfaces=req_data.get('expanded_external_interfaces', ''),
        )
        # 标记是否明确无依赖
        requirement._has_no_dependencies = has_no_dependencies
        
        requirements.append(requirement)
    
    return requirements


def main():
    parser = argparse.ArgumentParser(
        description='将 JSON 格式的需求信息转换为 Word SRS 文档',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('json_file', help='需求信息 JSON 文件路径')
    parser.add_argument('output_path', nargs='?', help='输出 Word 文件路径（可选）')
    parser.add_argument('--project-path', default='.', help='项目路径（默认当前目录）')
    parser.add_argument('--expanded-json', help='扩写内容 JSON 文件路径（包含background和description的扩写）')
    parser.add_argument('--output-dir', default=None, help='SRS文档输出目录（默认当前工作目录）')
    parser.add_argument('--temp-dir', default=None, help='临时文件输出目录（默认为skill目录下的temp）')
    parser.add_argument('--skip-validation', action='store_true', help='跳过占位符验证（不推荐）')
    
    args = parser.parse_args()
    
    skill_base_dir = Path(__file__).parent.parent.resolve()
    temp_dir = Path(args.temp_dir) if args.temp_dir else skill_base_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 检测项目环境
        project_source = ProjectEnvironmentDetector.get_project_info_source(args.project_path)
        
        requirements = parse_json_file(args.json_file)
        
        config = SRSConfig(project_path=args.project_path)
        
        scanner = ProjectScanner(config)
        existing_interfaces = scanner.scan()
        print(f"已扫描到 {len(existing_interfaces)} 个现有接口")
        
        inferrer = InterfaceInferrer(existing_interfaces, scanner.class_names)
        for req in requirements:
            req.interfaces = inferrer.infer_from_requirement(req)
        
        if args.output_path:
            output_path = Path(args.output_path)
            if not output_path.is_absolute():
                output_path = output_dir / output_path
        else:
            first_title = requirements[0].title
            safe_title = re.sub(r'[\\/:*?"<>|\s]', '_', first_title)
            if len(safe_title) > 50:
                safe_title = safe_title[:50]
            output_path = output_dir / f"{safe_title}.docx"
        
        if args.expanded_json:
            expanded_json_path = Path(args.expanded_json)
            if not expanded_json_path.is_absolute():
                expanded_json_path = temp_dir / expanded_json_path.name
        else:
            expanded_json_path = temp_dir / f"{output_path.stem}_expanded.json"
        
        # 获取当前执行次数并增加
        execution_count = increment_execution_count(temp_dir)
        print(f"\n当前为第 {execution_count} 次执行（最多允许3次）")
        
        # 验证扩写 JSON 文件中的占位符
        validation_errors = []
        if expanded_json_path.exists() and not args.skip_validation:
            try:
                with open(expanded_json_path, 'r', encoding='utf-8') as f:
                    expanded_data = json.load(f)
                
                errors = PlaceholderValidator.validate_expanded_content(expanded_data)
                if errors:
                    validation_errors = errors
            except json.JSONDecodeError as e:
                print(f"警告：无法解析扩写JSON文件：{e}")
        
        # 收集所有校验错误（不立即退出，跑完所有校验逻辑后统一打印）
        all_errors = list(validation_errors)

        logic_expander = LogicExpander(args.project_path, scanner.class_names, expanded_json_path=str(expanded_json_path))
        mermaid_generator = MermaidGenerator()
        
        for req in requirements:
            for sub_req in req.sub_requirements:
                expanded_logic, description = logic_expander.expand_logic(sub_req, parent_title=req.title)
                sub_req.logic = expanded_logic
                sub_req.description = description
                
                flow_steps = logic_expander.get_flow_steps(sub_req.title, req.title)
                if flow_steps:
                    sub_req.flow_steps = flow_steps
                else:
                    print(f"提示：子模块 '{sub_req.title}' 的流程步骤未在扩写JSON中提供")
                    print(f"      请优先扫描项目文件获取信息，若不足再使用网络搜索")
                    print(f"      格式要求：动词+名词，如['初始化环境', '检查参数', '执行任务']")
                    sub_req.flow_steps = mermaid_generator._extract_flow_steps(sub_req.logic)
            
            expanded_deps = logic_expander.get_dependencies(req.title)
            if expanded_deps and not PlaceholderValidator.contains_placeholder(expanded_deps):
                req.expanded_dependencies = expanded_deps
            elif getattr(req, '_has_no_dependencies', False):
                # 用户明确声明无依赖
                req.expanded_dependencies = "本次更新为独立模块，不涉及对其他 pypto 内部组件的影响，无需进行跨模块适配。"
                print(f"提示：需求 '{req.title}' 已声明无依赖")
            else:
                # 基于需求标题和子模块生成默认特性依赖
                affected_modules = []
                for sub_req in req.sub_requirements:
                    affected_modules.append(sub_req.title)
                
                if affected_modules:
                    modules_str = "、".join(affected_modules[:3])
                    req.expanded_dependencies = f"本次{req.title}可能影响以下 pypto 内部模块：{modules_str}。这些模块需要适配新的接口规范，建议进行回归测试以验证功能兼容性。具体受影响的文件和接口请通过扫描项目代码进一步确认。"
                else:
                    req.expanded_dependencies = f"本次{req.title}可能影响 pypto 框架中的相关组件。请通过扫描项目文件（使用 Glob 和 Grep 工具）分析具体的模块依赖关系，确定受影响的组件列表并进行适配测试。"
                print(f"提示：需求 '{req.title}' 的特性依赖信息已生成默认内容，建议通过扫描项目文件完善")
        
        content_expander = ContentExpander(args.project_path, str(expanded_json_path))
        
        # 创建内容验证器
        content_validator = ContentValidator(args.project_path)
        
        print("正在从JSON读取扩写内容...")
        # 用于收集所有扩写相关的错误
        expansion_errors = []
        
        for req in requirements:
            req.expanded_background = content_expander.expand_background(req, expansion_errors)
            req.expanded_description = content_expander.expand_description(req, expansion_errors)
            
            # 验证并清理扩写内容中的文件和方法
            if project_source == "local":
                # 只有本地项目才能验证文件
                validation_report = content_validator.validate_content(req.expanded_description)
                if validation_report['warnings']:
                    print(f"  警告 [{req.title}] 内容验证：")
                    for warning in validation_report['warnings'][:3]:  # 只显示前3个警告
                        print(f"    - {warning}")
                    # 清理未验证的内容
                    req.expanded_description = content_validator.sanitize_content(req.expanded_description)
                
                # 验证依赖信息
                deps_report = content_validator.validate_content(req.expanded_dependencies)
                if deps_report['warnings']:
                    req.expanded_dependencies = content_validator.sanitize_content(req.expanded_dependencies)
            
            bg_len = len(req.expanded_background) if req.expanded_background else 0
            desc_len = len(req.expanded_description) if req.expanded_description else 0
            deps_status = "已扩写" if req.expanded_dependencies and not PlaceholderValidator.contains_placeholder(req.expanded_dependencies) else "已生成默认内容"
            print(f"  - {req.title}: 背景{bg_len}字符, 描述{desc_len}字符, 依赖{deps_status}")
        
        # 合并所有错误
        all_errors.extend(expansion_errors)
        
        # 所有校验逻辑已跑完，统一打印所有校验错误并决定是否退出
        if all_errors:
            print(f"\n{'='*60}")
            print(f"校验失败（第 {execution_count} 次执行）")
            print(f"{'='*60}")
            for error in all_errors:
                print(f"  - {error}")
            print(f"{'='*60}")

            # 前两次执行失败时提示整改并退出，第三次强制输出
            if execution_count < 3:
                print(f"请 Agent 执行以下步骤进行整改：")
                print("1. 优先扫描项目代码、文档获取相关实现细节")
                print("2. 若项目文件信息不足，再通过网络搜索获取技术背景")
                print("3. 基于需求上下文推理生成完整内容")
                print("4. 确保所有字段长度符合要求且不包含占位符")
                print("注意：expanded_dependencies 只能从项目文件获取，禁止网络搜索")
                print(f"{'='*60}\n")
                sys.exit(1)
            else:
                print(f"已达到第3次执行，将强制输出 SRS 文档（忽略校验错误）")
                print(f"{'='*60}\n")

        force_output = (execution_count >= 3)

        print("正在生成Mermaid图表...")
        for req in requirements:
            try:
                req.mermaid_arch = mermaid_generator.generate_architecture_diagram(req)
                print(f"  - {req.title} 架构图已生成")
            except Exception as e:
                if force_output:
                    print(f"  - {req.title} 架构图生成失败（第3次执行，跳过）：{e}")
                    req.mermaid_arch = ""
                else:
                    raise
            for sub_req in req.sub_requirements:
                try:
                    steps = sub_req.flow_steps if sub_req.flow_steps else mermaid_generator._extract_flow_steps(sub_req.logic)
                    sub_req.mermaid_flow = mermaid_generator.generate_flow_diagram_from_steps(sub_req.title, steps)
                    print(f"    - {sub_req.title} 流程图已生成")
                except Exception as e:
                    if force_output:
                        print(f"    - {sub_req.title} 流程图生成失败（第3次执行，跳过）：{e}")
                        sub_req.mermaid_flow = ""
                    else:
                        raise
        logic_expander.save_expanded_content(requirements, str(expanded_json_path))

        try:
            generator = SRSDocumentGenerator(config)
            generator.generate(requirements, str(output_path))
        except Exception as e:
            if force_output:
                print(f"警告：文档生成过程中出错（第3次执行，尝试简化输出）：{e}")
                # 第3次执行时，清空可能导致问题的Mermaid内容后重试
                for req in requirements:
                    req.mermaid_arch = ""
                    for sub_req in req.sub_requirements:
                        sub_req.mermaid_flow = ""
                generator = SRSDocumentGenerator(config)
                generator.generate(requirements, str(output_path))
            else:
                raise

    except FileNotFoundError as e:
        print(f"错误：{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"错误：{e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"权限错误：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
