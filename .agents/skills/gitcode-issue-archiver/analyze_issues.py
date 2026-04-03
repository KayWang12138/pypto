#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
from datetime import datetime
from pathlib import Path

# 配置
ARCHIVE_DIR = Path('/data/s00454010/issues/archive/pypto_issues')
OUTPUT_DIR = Path('/data/s00454010/code/pypto/.agents/skills/gitcode-issue-archiver/docs')

# 问题领域筛选关键词
ENVIRONMENT_KEYWORDS = [
    '安装', 'install', '配置', 'config', '环境', 'environment',
    '编译失败', '编译报错', '编译错误', 'build fail', 'compile error',
    '驱动', 'driver', '固件', 'firmware', 'npu-smi',
    'cann安装', 'cann install', 'torch_npu',
    'g++', 'gcc', 'cmake', 'pip install',
    'docker', 'container', '镜像', 'image',
    '硬件', 'hardware', '设备', 'device',
    'python版本', 'python version', '依赖', 'dependency',
    'license', '版权', 'pre-commit',
    '文档错误', '文档缺失', 'documentation', '文档描述',
]

PYPTO_KEYWORDS = [
    '精度', 'precision', 'accuracy', '数值', 'value',
    '性能', 'performance', '耗时', 'time', '速度', 'speed',
    '算子', 'operator', 'op', 'kernel',
    'pass', '编译优化', '图优化', 'optimization',
    'tensor', '张量', 'reshape', 'view', 'assemble',
    'loop', '循环', 'stitch', 'tile', '切分',
    'matmul', '矩阵乘', 'attention', 'softmax', 'relu',
    'cast', 'transpose', 'concat', 'scatter', 'gather',
    'aicore', 'aicpu', 'device', 'npu',
    '合轴', 'combine_axis', 'unroll', '展开',
    '动态轴', 'dynamic', '静态轴', 'static',
    'valid_shape', 'validshape', 'raw_shape',
    '内存', 'memory', 'buffer', '溢出', 'overflow',
    '依赖', 'dependency', 'dep', '调度', 'schedule',
    '泳道图', 'swimlane', 'profiling',
    'ooschedule', 'pregraph', 'codegen',
    'spill', 'inplace', 'copyin', 'copyout',
    'mte', 'ub', 'l1', 'l0c', 'gm',
]


def should_filter_env_issue(title, description=''):
    """判断是否为环境类问题（需要过滤）"""
    text = (title + ' ' + description).lower()
    
    # 明确的环境问题
    env_patterns = [
        r'安装.*失败|失败.*安装',
        r'编译.*失败|失败.*编译',
        r'环境.*问题|问题.*环境',
        r'配置.*错误|错误.*配置',
        r'pip install|pip.*安装',
        r'docker|容器|镜像',
        r'驱动|driver|npu-smi',
        r'g\+\+|gcc|cmake.*错误',
        r'python.*版本|版本.*python',
        r'文档.*错误|文档.*缺失|链接.*失效',
        r'license|版权',
        r'pre-commit|commit.*检查',
    ]
    
    for pattern in env_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def should_keep_pypto_issue(title, description=''):
    """判断是否为 PyPTO 相关问题（需要保留）"""
    text = (title + ' ' + description).lower()
    
    # PyPTO 核心关键词
    for keyword in PYPTO_KEYWORDS:
        if keyword in text:
            return True
    
    return False


def classify_issue(title, description=''):
    """分类问题类型"""
    text = (title + ' ' + description).lower()
    
    if any(kw in text for kw in ['编译', 'compile', 'codegen', 'pass', 'ooschedule', 'pregraph']):
        return 'A. 编译错误类'
    elif any(kw in text for kw in ['运行', 'runtime', '执行', 'aicore', 'aicpu', 'device', '卡死', '崩溃', 'coredump', 'segmentation']):
        return 'B. 运行时错误类'
    elif any(kw in text for kw in ['精度', 'precision', 'accuracy', '数值', '结果', '输出', '误差', '对比']):
        return 'C. 精度问题类'
    elif any(kw in text for kw in ['性能', 'performance', '耗时', '时间', '速度', '慢', '快', '劣化']):
        return 'D. 性能问题类'
    elif any(kw in text for kw in ['不支持', '缺失', '缺少', '未实现', 'not support', 'missing']):
        return 'E. 功能缺失类'
    elif any(kw in text for kw in ['文档', 'documentation', 'readme', '描述', '说明']):
        return 'F. 文档问题类'
    else:
        return 'G. 其他问题类'


def extract_trigger_condition(issue_content):
    """提取触发条件"""
    # 查找关键短语
    patterns = [
        r'触发条件[：:]\s*(.+?)(?:\n\n|\n规避|\n状态|$)',
        r'复现步骤[：:]\s*(.+?)(?:\n\n|\n规避|\n状态|$)',
        r'如何触发[：:]\s*(.+?)(?:\n\n|\n规避|\n状态|$)',
        r'问题描述[：:]\s*(.+?)(?:\n\n|\n规避|\n状态|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, issue_content, re.DOTALL)
        if match:
            condition = match.group(1).strip()
            # 清理格式
            condition = re.sub(r'\s+', ' ', condition)
            return condition
    
    return None


def extract_workaround(issue_content):
    """提取规避方案"""
    patterns = [
        r'规避方案[：:]\s*(.+?)(?:\n\n|\n状态|$)',
        r'解决方案[：:]\s*(.+?)(?:\n\n|\n状态|$)',
        r'解决方法[：:]\s*(.+?)(?:\n\n|\n状态|$)',
        r'workaround[：:]\s*(.+?)(?:\n\n|\n状态|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, issue_content, re.DOTALL)
        if match:
            workaround = match.group(1).strip()
            workaround = re.sub(r'\s+', ' ', workaround)
            return workaround
    
    return None


def has_explicit_trigger(title, description):
    """判断是否有明确的触发条件"""
    text = title + ' ' + description
    
    # 明确的触发关键词
    trigger_keywords = [
        '开启', '关闭', '设置', '调用', '使用', '配置',
        'combine_axis', 'unroll', 'stitch', 'tile_shape',
        'valid_shape', '动态轴', '静态轴',
        'reshape', 'view', 'assemble', 'transpose',
        'matmul', 'attention', 'scatter', 'gather',
        '大于', '小于', '等于', '超过',
        '特定', '某些', '部分',
    ]
    
    for keyword in trigger_keywords:
        if keyword in text:
            return True
    
    return False


def analyze_single_issue(issue_num):
    """分析单个 issue"""
    issue_file = ARCHIVE_DIR / f'issue-{issue_num}.md'
    
    if not issue_file.exists():
        return None
    
    with open(issue_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键信息
    lines = content.split('\n')
    title = ''
    description = ''
    state = 'closed'
    
    # 解析标题和状态
    for line in lines[:20]:  # 前20行包含基本信息
        if line.startswith('# '):
            title = line[2:].strip()
        elif 'State:' in line or '状态:' in line:
            if 'open' in line.lower():
                state = 'open'
            elif 'resolved' in line.lower():
                state = 'closed, resolved'
            else:
                state = 'closed'
    
    # 提取描述部分
    desc_start = False
    desc_lines = []
    for line in lines:
        if '## 问题描述' in line or '## Description' in line:
            desc_start = True
            continue
        if desc_start:
            if line.startswith('## '):
                break
            desc_lines.append(line)
    
    description = ' '.join(desc_lines).strip()
    
    return {
        'number': issue_num,
        'title': title,
        'state': state,
        'description': description,
        'content': content,
    }


def main():
    """主函数"""
    print("开始分析 Bug-Report issue...")
    
    # 读取归档记录
    with open(ARCHIVE_DIR / 'archive_record.json', 'r') as f:
        record = json.load(f)
    
    # 筛选 Bug-Report issue
    bug_issues = []
    for issue_num, issue_info in record['issues'].items():
        if issue_info.get('exists') and '[Bug-Report' in issue_info.get('title', ''):
            bug_issues.append({
                'number': int(issue_num),
                'state': issue_info.get('state', 'unknown'),
                'title': issue_info.get('title', ''),
                'comment_count': issue_info.get('comment_count', 0)
            })
    
    bug_issues.sort(key=lambda x: x['number'])
    print(f"找到 {len(bug_issues)} 个 Bug-Report issue")
    
    # 分析每个 issue
    analyzed_issues = []
    for idx, issue in enumerate(bug_issues):
        if (idx + 1) % 50 == 0:
            print(f"处理进度: {idx + 1}/{len(bug_issues)}")
        
        detail = analyze_single_issue(issue['number'])
        if detail:
            # 应用筛选规则
            if should_filter_env_issue(detail['title'], detail['description']):
                continue  # 过滤环境类问题
            
            if not should_keep_pypto_issue(detail['title'], detail['description']):
                continue  # 不是 PyPTO 核心问题
            
            # 分类
            category = classify_issue(detail['title'], detail['description'])
            
            # 提取触发条件和规避方案
            trigger = extract_trigger_condition(detail['content'])
            workaround = extract_workaround(detail['content'])
            
            analyzed_issues.append({
                'number': issue['number'],
                'title': detail['title'],
                'state': detail['state'],
                'category': category,
                'trigger': trigger,
                'workaround': workaround,
                'description': detail['description'][:200],  # 截断描述
            })
    
    print(f"筛选后保留 {len(analyzed_issues)} 个 PyPTO 相关问题")
    
    # 生成报告
    generate_reports(analyzed_issues)


def generate_reports(issues):
    """生成两个报告文件"""
    
    # ========== 文件一：不支持场景清单 ==========
    unsupported_scenarios = []
    
    for issue in issues:
        # 只包含有明确触发条件和规避方案的高质量条目
        if issue['trigger'] and issue['workaround']:
            # 判断触发条件是否足够明确
            if has_explicit_trigger(issue['title'], issue['description']):
                unsupported_scenarios.append(issue)
    
    # 生成文件一
    file1_content = f"""# PyPTO 不支持场景清单

> **用途**: 开发前必读，了解已知限制和规避方案
> **更新时间**: {datetime.now().strftime('%Y-%m-%d')}
> **数据来源**: /data/s00454010/issues/archive/pypto_issues/

---

## 明确的不支持场景

本部分列出 PyPTO 明确不支持的场景及规避方案。只包含有明确触发条件和规避方案的高质量条目。

"""
    
    # 按分类组织
    categories = {}
    for issue in unsupported_scenarios:
        cat = issue['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(issue)
    
    scenario_id = 1
    for category in sorted(categories.keys()):
        file1_content += f"### {scenario_id}. {category.split('.')[-1].strip()}\n\n"
        
        for issue in categories[category]:
            state_str = '已修复' if 'resolved' in issue['state'] else '仍需规避'
            if issue['state'] == 'open':
                state_str = '功能缺失'
            
            file1_content += f"""**不支持场景**：{issue['title']}
- **触发条件**：
  - {issue['trigger']}
- **规避方案**：{issue['workaround']}
- **状态**：{state_str}
- **来源**：Issue #{issue['number']} ({issue['state']})

"""
            scenario_id += 1
    
    # 写入文件一
    with open(OUTPUT_DIR / 'pypto_unsupported_scenarios.md', 'w', encoding='utf-8') as f:
        f.write(file1_content)
    
    print(f"文件一生成完成: {len(unsupported_scenarios)} 个高质量场景")
    
    # ========== 文件二：问题现象索引 ==========
    file2_content = f"""# PyPTO 问题现象索引

> **用途**: 遇到精度问题时检索相似案例
> **更新时间**: {datetime.now().strftime('%Y-%m-%d')}
> **数据来源**: /data/s00454010/issues/archive/pypto_issues/

---

## 问题现象索引

本部分包含所有 Bug-Report Issue，按类型分类。

"""
    
    # 按类型分类
    type_issues = {}
    for issue in issues:
        cat = issue['category']
        if cat not in type_issues:
            type_issues[cat] = []
        type_issues[cat].append(issue)
    
    for category in sorted(type_issues.keys()):
        issues_in_cat = type_issues[category]
        file2_content += f"### {category}\n\n"
        file2_content += f"**Issue 数量**: {len(issues_in_cat)}\n\n"
        
        for issue in issues_in_cat:
            state_str = issue['state']
            file2_content += f"""#### Issue #{issue['number']} - {issue['title']} [{state_str}]

- **现象**: {issue['description'][:100]}
- **触发条件**: {issue['trigger'] if issue['trigger'] else '见 Issue 描述'}
- **状态**: {state_str}
- **来源**: Issue #{issue['number']}

"""
        
        file2_content += "\n"
    
    # 写入文件二
    with open(OUTPUT_DIR / 'pypto_issue_index.md', 'w', encoding='utf-8') as f:
        f.write(file2_content)
    
    print(f"文件二生成完成: {len(issues)} 个问题")


if __name__ == '__main__':
    main()
EOF