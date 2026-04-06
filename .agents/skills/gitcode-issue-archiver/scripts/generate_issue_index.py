#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""
Generate improved issue index document from archived issues.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def extract_actual_description(content: str) -> str:
    """Extract actual problem description, skipping template text."""
    lines = content.split('\n')
    
    # Find description section
    desc_start = None
    for i, line in enumerate(lines):
        if '## Description' in line:
            desc_start = i + 1
            break
    
    if desc_start is None:
        return ""
    
    # Skip template lines like "Thanks for sending..."
    actual_desc = []
    for i in range(desc_start, min(desc_start + 50, len(lines))):
        line = lines[i].strip()
        
        # Skip empty lines and template text
        if not line:
            continue
        
        # Skip template markers
        if line.startswith('Thanks for sending an issue'):
            continue
        if '### Describe the current behavior' in line:
            continue
        if '(Mandatory / 必填)' in line:
            continue
        
        # Extract actual description
        if line and not line.startswith('###') and not line.startswith('!['):
            # Clean up the line
            actual_desc.append(line)
        
        # Stop at next section
        if line.startswith('### Environment'):
            break
        if line.startswith('### Steps'):
            break
    
    # Join and clean
    desc_text = ' '.join(actual_desc[:3])  # Take first 3 meaningful lines
    desc_text = re.sub(r'<[^>]+>', '', desc_text)  # Remove HTML tags
    desc_text = desc_text.strip()
    
    # Limit length
    if len(desc_text) > 150:
        desc_text = desc_text[:150] + '...'
    
    return desc_text


def check_if_resolved(content: str, state: str) -> Tuple[bool, str]:
    """Check if issue is resolved or expected behavior."""
    if state != 'closed':
        return False, ""
    
    lines = content.split('\n')
    
    # Check if this is expected behavior (not a bug)
    expected_keywords = [
        '预期范围',
        '预期行为',
        '正常行为',
        '正常现象',
        '符合预期',
        '不是bug',
        '不是问题',
        '设计如此',
        'as expected',
        'by design'
    ]
    
    # Check comments section
    comments_section = False
    for line in lines:
        if '## Comments' in line:
            comments_section = True
            continue
        
        if comments_section:
            for keyword in expected_keywords:
                if keyword in line:
                    return True, f"预期行为: {keyword}"
    
    # First check if there's a "resolved" label
    in_labels = False
    for line in lines:
        if '## Labels' in line:
            in_labels = True
            continue
        if in_labels and line.startswith('## '):
            break
        if in_labels and 'resolved' in line:
            return True, "resolved label"
    
    # Then check resolution keywords in comments
    resolution_keywords = [
        '问题已解决',
        '已修复',
        '已合入',
        'PR已合入',
        '问题已修复',
        '本issue关闭',
        '本ISSUE关闭',
        'resolved',
        'fixed',
        'merged',
        '不再复现',
        '无法复现',
        '问题不再出现',
        '如有任何疑问',
        '问题已收到',
        '进入关闭流程',
        '即将进入关闭'
    ]
    
    # Check comments section (after "## Comments")
    comments_section = False
    for line in lines:
        if '## Comments' in line:
            comments_section = True
            continue
        
        if comments_section:
            for keyword in resolution_keywords:
                if keyword in line:
                    # Found resolution keyword in comments
                    return True, keyword
    
    return False, ""


def extract_root_cause_and_solution(content: str) -> Tuple[str, str]:
    """Extract root cause and solution from comments."""
    root_cause = ""
    solution = ""
    
    lines = content.split('\n')
    
    # Look for key patterns in comments
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Root cause patterns
        if '根因' in line or '原因' in line or '定位结论' in line:
            # Extract next few lines
            end_j = min(i + 5, len(lines))
            for j in range(i + 1, end_j):
                if lines[j].strip() and not lines[j].strip().startswith('###'):
                    root_cause += lines[j].strip() + ' '
        
        # Solution patterns
        has_solution_kw = '规避' in line or '解决方案' in line
        has_solution_kw = has_solution_kw or '修改' in line or '写法' in line
        if has_solution_kw:
            end_j = min(i + 5, len(lines))
            for j in range(i + 1, end_j):
                if lines[j].strip() and not lines[j].strip().startswith('###'):
                    solution += lines[j].strip() + ' '
    
    # Clean up
    root_cause = root_cause.strip()
    solution = solution.strip()
    
    if len(root_cause) > 200:
        root_cause = root_cause[:200] + '...'
    if len(solution) > 200:
        solution = solution[:200] + '...'
    
    return root_cause, solution


def categorize_issue(title: str) -> str:
    """Categorize issue based on title."""
    title_lower = title.lower()
    
    if any(kw in title_lower for kw in ['编译', 'pass', 'codegen', 'schedule', '构图']):
        return 'A. 编译错误类'
    elif any(kw in title_lower for kw in ['运行', 'runtime', 'device', 'aicore', '执行', '报错', '崩溃', '卡死']):
        return 'B. 运行时错误类'
    elif any(kw in title_lower for kw in ['精度', 'precision', '数值', '结果', '偏差', '误差']):
        return 'C. 精度问题类'
    elif any(kw in title_lower for kw in ['性能', 'perf', '耗时', '劣化', '优化']):
        return 'D. 性能问题类'
    elif any(kw in title_lower for kw in ['不支持', '缺失', '缺少', '未实现']):
        return 'E. 功能缺失类'
    else:
        return 'G. 其他问题类'


def main():
    archive_dir = Path('/data/s00454010/issues/archive/pypto_issues')
    record_file = archive_dir / 'archive_record.json'
    
    # Load record
    with open(record_file) as f:
        record = json.load(f)
    
    # Filter Bug-Report issues
    bug_issues = []
    for num, issue in record['issues'].items():
        if issue['exists'] and '[Bug-Report' in issue['title']:
            bug_issues.append((int(num), issue))
    
    # Sort by number
    bug_issues.sort(key=lambda x: x[0])
    
    # Process each issue
    categorized = {}
    stats = {'total': 0, 'resolved': 0, 'closed': 0, 'open': 0, 'expected': 0}
    
    for num, issue in bug_issues:
        # Read issue file
        issue_file = archive_dir / f'issue-{num}.md'
        if not issue_file.exists():
            continue
        
        with open(issue_file) as f:
            content = f.read()
        
        # Extract information
        actual_desc = extract_actual_description(content)
        is_resolved, resolution_info = check_if_resolved(content, issue['state'])
        root_cause, solution = extract_root_cause_and_solution(content)
        category = categorize_issue(issue['title'])
        
        # Check if it's expected behavior
        is_expected = resolution_info.startswith('预期行为')
        
        # Update stats
        stats['total'] += 1
        if is_resolved:
            stats['resolved'] += 1
            if is_expected:
                stats['expected'] += 1
        elif issue['state'] == 'closed':
            stats['closed'] += 1
        else:
            stats['open'] += 1
        
        # Prepare issue entry
        issue_entry = {
            'num': num,
            'title': issue['title'],
            'state': issue['state'],
            'is_resolved': is_resolved,
            'is_expected': is_expected,
            'desc': actual_desc,
            'root_cause': root_cause,
            'solution': solution,
            'resolution_info': resolution_info
        }
        
        # Add to category
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(issue_entry)
    
    # Generate markdown
    output = []
    output.append("# PyPTO 问题现象索引")
    output.append("")
    output.append("> **用途**: 遇到问题时检索相似案例，了解规避方案")
    output.append("> **更新时间**: 2026-04-03")
    output.append("> **数据来源**: /data/s00454010/issues/archive/pypto_issues/")
    output.append("")
    output.append("---")
    output.append("")
    output.append("## 统计信息")
    output.append("")
    output.append(f"- **Bug-Report总数**: {stats['total']}")
    output.append(f"- **已修复**: {stats['resolved']} (问题已解决，无需关注)")
    if stats['expected'] > 0:
        output.append(f"  - 其中预期行为: {stats['expected']} (不是bug，是正常行为)")
    output.append(f"- **仍需关注**: {stats['closed'] + stats['open']} (closed={stats['closed']}, open={stats['open']})")
    output.append("")
    output.append("---")
    output.append("")
    
    # Section 1: Resolved issues (not important, just list)
    if stats['resolved'] > 0:
        output.append("## 第一部分：已修复问题（无需关注）")
        output.append("")
        output.append(f"以下 {stats['resolved']} 个问题已在评论中确认修复或不再复现，无需重点关注。")
        output.append("")
        
        # Separate expected behavior issues
        expected_issues = []
        fixed_issues = []
        
        for category in categorized:
            for issue in categorized[category]:
                if issue['is_resolved']:
                    if issue.get('is_expected'):
                        expected_issues.append(issue)
                    else:
                        fixed_issues.append(issue)
        
        # Show expected behavior issues first
        if expected_issues:
            output.append("### 预期行为（不是bug）")
            output.append("")
            for issue in expected_issues:
                title_clean = issue['title'].replace('[Bug-Report|缺陷反馈]: ', '')
                output.append(f"- Issue #{issue['num']} - {title_clean}")
            output.append("")
        
        # Show fixed issues by category
        if fixed_issues:
            resolved_by_category = {}
            for category in categorized:
                for issue in categorized[category]:
                    if issue['is_resolved'] and not issue.get('is_expected'):
                        if category not in resolved_by_category:
                            resolved_by_category[category] = []
                        resolved_by_category[category].append(issue)
        
        for category in ['A. 编译错误类', 'B. 运行时错误类', 'C. 精度问题类', 
                         'D. 性能问题类', 'E. 功能缺失类', 'G. 其他问题类']:
            if category in resolved_by_category:
                issues = resolved_by_category[category]
                output.append(f"### {category}")
                output.append("")
                
                for issue in issues:
                    title_clean = issue['title'].replace('[Bug-Report|缺陷反馈]: ', '')
                    output.append(f"- Issue #{issue['num']} - {title_clean}")
                
                output.append("")
        
        output.append("---")
        output.append("")
    
    # Section 2: Issues that need attention
    output.append("## 第二部分：仍需关注的问题")
    output.append("")
    output.append("以下问题尚未明确修复，可能需要规避或仍在处理中。")
    output.append("")
    output.append("**说明**: ")
    output.append("- `[closed]` - 问题已关闭但未明确修复，可能需要规避方案")
    output.append("- `[open]` - 问题仍在处理中")
    output.append("")
    
    # Output each category (only closed and open issues)
    for category in ['A. 编译错误类', 'B. 运行时错误类', 'C. 精度问题类', 
                     'D. 性能问题类', 'E. 功能缺失类', 'G. 其他问题类']:
        if category not in categorized:
            continue
        
        # Filter only issues that need attention (not resolved)
        issues_need_attention = [i for i in categorized[category] if not i['is_resolved']]
        
        if not issues_need_attention:
            continue
        
        output.append(f"### {category}")
        output.append("")
        output.append(f"**Issue 数量**: {len(issues_need_attention)}")
        output.append("")
        
        for issue in issues_need_attention:
            # Determine status label
            if issue['state'] == 'closed':
                status_label = '[closed]'
                status_note = '已关闭'
            else:
                status_label = '[open]'
                status_note = '仍在处理'
            
            # Issue header
            title_clean = issue['title'].replace('[Bug-Report|缺陷反馈]: ', '')
            output.append(f"#### Issue #{issue['num']} - {title_clean} {status_label}")
            output.append("")
            
            # Description
            if issue['desc']:
                output.append(f"- **现象**: {issue['desc']}")
            else:
                output.append(f"- **现象**: 见Issue #{issue['num']} 详细描述")
            
            # Root cause (if available)
            if issue['root_cause']:
                output.append(f"- **根因**: {issue['root_cause']}")
            
            # Solution (if available)
            if issue['solution']:
                output.append(f"- **规避方案**: {issue['solution']}")
            
            # Status
            output.append(f"- **状态**: {status_note}")
            output.append(f"- **来源**: Issue #{issue['num']}")
            output.append("")
        
        output.append("")
    
    # Write output
    output_file = Path('/data/s00454010/code/pypto/.agents/skills/gitcode-issue-archiver/docs/pypto_issue_index.md')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))
    
    logger.info("✅ 已生成新的索引文档")
    logger.info("   文件: %s", output_file)
    logger.info("   Bug-Report总数: %s", stats['total'])
    logger.info("   已修复: %s", stats['resolved'])
    logger.info("   已关闭: %s", stats['closed'])
    logger.info("   仍开启: %s", stats['open'])


if __name__ == '__main__':
    main()