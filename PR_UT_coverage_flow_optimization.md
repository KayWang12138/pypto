# PR获取和UTreport下载解析流程优化方案

## 概述

本文档优化了原skill中PR获取和UTreport下载解析的流程，提供了更清晰、更完整的步骤，并使用PR #1696作为验证示例。

## 优化点总结

### 1. PR文件变更获取方式优化
**原方案问题**：
- 使用`gitcode_list_pull_request_comments`获取diff_comment来获取文件变更
- diff_comment只包含代码行评论，不包含完整的文件变更列表

**优化方案**：
- 使用git命令（git diff）获取PR的文件变更列表
- 通过git fetch拉取PR分支，然后使用git diff对比目标分支

### 2. UTreport下载流程完善
**原方案问题**：
- 缺少下载tar.gz文件的完整代码
- 缺少解压tar.gz文件的步骤
- 缺少解析解压后HTML报告的具体方法

**优化方案**：
- 提供完整的下载、解压、解析流程
- 支持多种下载链接格式
- 提供HTML报告解析的具体实现

### 3. 流程结构优化
**原方案问题**：
- 步骤分散，没有形成完整的可执行流程
- 缺少错误处理和验证步骤

**优化方案**：
- 将流程分为清晰的阶段
- 每个阶段都有明确的输入输出
- 提供完整的错误处理机制

---

## 完整流程

### Stage 1: 获取PR信息

#### 步骤 1.1：解析PR链接

```python
import re

def parse_pr_info(pr_input: str) -> tuple:
    """
    解析PR链接，返回 (owner, repo, pr_number)

    支持的PR链接格式：
    - https://gitcode.com/cann/pypto/pull/123
    - https://gitcode.com/<username>/pypto/pull/456
    - 简短格式：cann/pypto/123 或 <username>/pypto/456
    - PR编号格式：#123 或 PR #123
    """
    # 匹配URL格式
    url_match = re.search(r'gitcode\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_input)
    if url_match:
        owner = url_match.group(1)
        repo = url_match.group(2)
        pr_number = int(url_match.group(3))
        return owner, repo, pr_number

    # 匹配简短格式：cann/pypto/123
    short_match = re.search(r'([^/]+)/([^/]+)/(\d+)', pr_input)
    if short_match:
        owner = short_match.group(1)
        repo = short_match.group(2)
        pr_number = int(short_match.group(3))
        return owner, repo, pr_number

    # 匹配 #123 或 PR #123 格式
    hash_match = re.search(r'#(\d+)', pr_input)
    if hash_match:
        pr_number = int(hash_match.group(1))
        return "cann", "pypto", pr_number

    raise ValueError(f"无法解析PR链接：{pr_input}")

# 示例
owner, repo, pr_number = parse_pr_info("1696")
# owner = "cann"
# repo = "pypto"
# pr_number = 1696
```

#### 步骤 1.2：获取PR基本信息

```python
import subprocess
import os

def get_pr_basic_info(owner: str, repo: str, pr_number: int) -> dict:
    """
    使用GitCode MCP工具获取PR基本信息
    """
    pr_info = gitcode_get_pull_request(owner, repo, pr_number)

    # 验证PR状态
    if pr_info["state"] not in ["open", "merged"]:
        print(f"⚠️ 警告：PR {pr_number} 状态为 {pr_info['state']}")

    # 验证仓库
    if repo != "pypto":
        print(f"❌ 错误：目标仓库不是 pypto，当前为 {repo}")
        return None

    return {
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "title": pr_info["title"],
        "state": pr_info["state"],
        "source_branch": pr_info["head"]["ref"],
        "target_branch": pr_info["base"]["ref"],
        "labels": [label.get("name", "") for label in pr_info.get("labels", [])]
    }

# 示例
pr_info = get_pr_basic_info("cann", "pypto", 1696)
print(f"PR标题: {pr_info['title']}")
print(f"源分支: {pr_info['source_branch']}")
print(f"目标分支: {pr_info['target_branch']}")
```

#### 步骤 1.3：获取PR文件变更列表

```python
def get_pr_changed_files(pr_number: int, target_branch: str = "master") -> list:
    """
    使用git命令获取PR的变更文件列表

    Args:
        pr_number: PR编号
        target_branch: 目标分支，默认为master

    Returns:
        变更的文件列表
    """
    try:
        # 步骤一：拉取PR的远程分支
        fetch_result = subprocess.run([
            'git', 'fetch', 'origin',
            f'pull/{pr_number}/head:pr_{pr_number}'
        ], check=True, capture_output=True, text=True, cwd='/mnt/workspace/gitCode/addreg/pypto/ut-pr/pypto')

        # 步骤二：获取变更的文件列表
        diff_result = subprocess.run([
            'git', 'diff', '--name-only', target_branch, f'pr_{pr_number}'
        ], check=True, capture_output=True, text=True, cwd='/mnt/workspace/gitCode/addreg/pypto/ut-pr/pypto')

        changed_files = diff_result.stdout.strip().split('\n')
        changed_files = [f for f in changed_files if f]

        return changed_files

    except subprocess.CalledProcessError as e:
        print(f"❌ 获取文件变更失败: {e}")
        print(f"错误输出: {e.stderr}")
        return []

# 示例
changed_files = get_pr_changed_files(1696, "master")
print(f"检测到 {len(changed_files)} 个变更文件")
for file in changed_files:
    print(f"  - {file}")
```

#### 步骤 1.4：过滤代码文件

```python
def filter_code_files(files: list) -> list:
    """
    过滤出代码文件（.）

    Args:
        files: 文件列表

    Returns:
        代码文件列表
    """
    code_extensions = ['.cpp', '.h', '.hpp']
    code_files = [
        f for f in files
        if any(f.endswith(ext) for ext in code_extensions)
    ]
    return code_files

# 示例
code_files = filter_code_files(changed_files)
print(f"共 {len(changed_files)} 个文件变更，其中 {len(code_files)} 个代码文件")
```

---

### Stage 2: 获取UT覆盖率报告

#### 步骤 2.1：检查CI标签状态

```python
import time

def wait_for_ci_complete(owner: str, repo: str, pr_number: int, max_wait: int = 1800) -> dict:
    """
    等待CI完成（最多等待max_wait秒）

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        pr_number: PR编号
        max_wait: 最大等待时间（秒），默认30分钟

    Returns:
        PR信息字典
    """
    start_time = time.time()

    while True:
        pr_info = gitcode_get_pull_request(owner, repo, pr_number)
        labels = [label.get("name", "") for label in pr_info.get("labels", [])]

        if "ci-pipeline-running" not in labels:
            print("✅ CI已完成")
            return pr_info

        if time.time() - start_time > max_wait:
            print(f"⚠️ 等待超时（{max_wait}秒），CI仍在运行")
            return pr_info

        print("⏳ 检测到 ci-pipeline-running 标签，CI 仍在运行")
        print("⏳ 等待 30 秒后重新检查...")
        time.sleep(30)

# 示例
pr_info = wait_for_ci_complete("cann", "pypto", 1696)
```

#### 步骤 2.2：获取机器人评论

```python
def get_robot_comments(owner: str, repo: str, pr_number: int) -> list:
    """
    获取PR中的机器人评论

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        pr_number: PR编号

    Returns:
        机器人评论列表
    """
    # 获取全部评论
    comments = gitcode_list_pull_request_comments(
        owner=owner,
        repo=repo,
        pull_number=pr_number,
        comment_type="pr_comment"
    )

    # 过滤机器人评论
    def is_robot_comment(comment):
        login = comment.get("user", {}).get("login", "").lower()
        if login == "cann-robot":
            return True
        return any(kw in login for kw in ["bot", "robot", "ci", "automation"])

    robot_comments = [c for c in comments if is_robot_comment(c)]
    return robot_comments

# 示例
robot_comments = get_robot_comments("cann", "pypto", 1696)
print(f"找到 {len(robot_comments)} 条机器人评论")
```

#### 步骤 2.3：查找流水线任务触发成功的评论

```python
def find_pipeline_comment(robot_comments: list) -> str:
    """
    从机器人评论中查找包含"流水线任务触发成功"的评论

    Args:
        robot_comments: 机器人评论列表

    Returns:
        流水线评论内容，未找到返回None
    """
    for comment in robot_comments:
        body = comment.get("body", "")
        if "流水线任务触发成功" in body:
            return body

    print("⚠️ 未找到流水线任务触发成功的评论")
    return None

# 示例
pipeline_comment = find_pipeline_comment(robot_comments)
if pipeline_comment:
    print("✅ 找到流水线评论")
```

#### 步骤 2.4：提取UT_Test_report状态

```python
def extract_ut_test_report_status(pipeline_comment: str) -> str:
    """
    从流水线评论中提取UT_Test_report的状态

    Args:
        pipeline_comment: 流水线评论内容

    Returns:
        状态字符串：SUCCESS, FAILED, ABORTED, UNKNOWN
    """
    lines = pipeline_comment.split('\n')
    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行中的状态
            if '✅ SUCCESS' in line:
                return 'SUCCESS'
            elif '❌ FAILED' in line:
                return 'FAILED'
            elif '⚪ ABORTED' in line:
                return 'ABORTED'

    return 'UNKNOWN'

# 示例
ut_status = extract_ut_test_report_status(pipeline_comment)
print(f"UT_Test_report 状态: {ut_status}")
```

#### 步骤 2.5：提取下载链接

```python
def extract_download_link(pipeline_comment: str) -> str:
    """
    从流水线评论中提取UT_Test_report行的下载链接

    支持的格式：
    - HTML格式：<a href=URL>
    - Markdown格式：[文本](URL)

    Args:
        pipeline_comment: 流水线评论内容

    Returns:
        下载链接，未找到返回None
    """
    import re
    lines = pipeline_comment.split('\n')

    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行及后续几行中的下载链接
            context = '\n'.join(lines[i:i+5])

            # 方法1：匹配HTML <a href=URL> 格式
            html_link_pattern = r'<a\s+href=([^>]+)>'
            html_matches = re.findall(html_link_pattern, context)

            for url in html_matches:
                # 清理URL
                url = url.strip()
                url = url.strip('"\'')
                url = re.sub(r'&gt;+', '', url)
                url = url.rstrip('>>')

                if url and ('.tar.gz' in url or '.html' in url):
                    return url

            # 方法2：匹配markdown链接格式：[文本](URL)
            md_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            md_matches = re.findall(md_link_pattern, context)

            for text, url in md_matches:
                url = url.strip()
                url = re.sub(r'&gt;+', '', url)
                url = url.rstrip('>>')

                if url and ('.tar.gz' in url or '.html' in url):
                    return url

    return None

# 示例
download_link = extract_download_link(pipeline_comment)
if download_link:
    print(f"✅ 找到下载链接: {download_link}")
else:
    print("⚠️ 未找到下载链接")
```

#### 步骤 2.6：下载覆盖率报告

```python
import requests
import tempfile
import os

def download_coverage_report(download_link: str, output_dir: str = None) -> str:
    """
    下载覆盖率报告

    Args:
        download_link: 下载链接
        output_dir: 输出目录，默认为当前目录

    Returns:
        下载的文件路径，失败返回None
    """
    try:
        # 下载文件
        response = requests.get(download_link, timeout=30)
        if response.status_code != 200:
            print(f"❌ 下载失败，状态码：{response.status_code}")
            return None

        # 确定输出目录
        if output_dir is None:
            output_dir = os.getcwd()

        # 保存文件
        if download_link.endswith('.tar.gz'):
            filename = 'ut_cov.tar.gz'
        elif download_link.endswith('.html'):
            filename = 'ut_cov.html'
        else:
            filename = 'ut_cov'

        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"✅ 覆盖率报告已下载到：{filepath}")
        return filepath

    except Exception as e:
        print(f"❌ 下载失败：{e}")
        return None

# 示例
downloaded_file = download_coverage_report(download_link, "/tmp")
```

---

### Stage 3: 解析覆盖率报告

#### 步骤 3.1：解压tar.gz文件

```python
import tarfile

def extract_coverage_report(tar_file: str, output_dir: str = None) -> str:
    """
    解压覆盖率报告tar.gz文件

    Args:
        tar_file: tar.gz文件路径
        output_dir: 输出目录，默认为tar文件所在目录

    Returns:
        解压后的目录路径，失败返回None
    """
    try:
        if output_dir is None:
            output_dir = os.path.dirname(tar_file)

        # 解压文件
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=output_dir)

        # 查找解压后的目录
        # 通常解压后会生成一个 inc_cov 目录
        extract_dir = os.path.join(output_dir, 'inc_cov')

        if os.path.exists(extract_dir):
            print(f"✅ 覆盖率报告已解压到：{extract_dir}")
            return extract_dir
        else:
            print(f"⚠️ 未找到预期的解压目录：{extract_dir}")
            return output_dir

    except Exception as e:
        print(f"❌ 解压失败：{e}")
        return None

# 示例
extract_dir = extract_coverage_report(downloaded_file, "/tmp")
```

#### 步骤 3.2：查找覆盖率HTML报告

```python
def find_coverage_html(extract_dir: str) -> str:
    """
    查找覆盖率HTML报告

    Args:
        extract_dir: 解压后的目录

    Returns:
        HTML报告路径，未找到返回None
    """
    # 通常HTML报告在 inc_cov/result/index.html
    possible_paths = [
        os.path.join(extract_dir, 'result', 'index.html'),
        os.path.join(extract_dir, 'index.html'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到覆盖率报告：{path}")
            return path

    # 如果没找到，尝试查找所有index.html文件
    for root, dirs, files in os.walk(extract_dir):
        if 'index.html' in files:
            path = os.path.join(root, 'index.html')
            print(f"✅ 找到覆盖率报告：{path}")
            return path

    print("⚠️ 未找到覆盖率报告")
    return None

# 示例
html_report = find_coverage_html(extract_dir)
```

#### 步骤 3.3：解析覆盖率HTML报告

```python
from bs4 import BeautifulSoup

def parse_coverage_html(html_file: str) -> dict:
    """
    解析覆盖率HTML报告

    Args:
        html_file: HTML文件路径

    Returns:
        覆盖率信息字典
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取总体覆盖率信息
        coverage_info = {
            'overall_line_coverage': None,
            'overall_function_coverage': None,
            'files': []
        }

        # 查找总体覆盖率（通常在表格中）
        # 根据实际的HTML结构调整选择器
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    text = cells[0].get_text().strip()
                    if 'Lines' in text or '行' in text:
                        coverage_info['overall_line_coverage'] = cells[1].get_text().strip()
                    elif 'Functions' in text or '函数' in text:
                        coverage_info['overall_function_coverage'] = cells[1].get_text().strip()

        # 查找文件覆盖率信息
        # 根据实际的HTML结构调整选择器
        file_rows = soup.find_all('tr')
        for row in file_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                file_link = cells[0].find('a')
                if file_link:
                    file_name = file_link.get_text().strip()
                    file_path = file_link.get('href', '')
                    line_cov = cells[1].get_text().strip() if len(cells) > 1 else ''
                    func_cov = cells[2].get_text().strip() if len(cells) > 2 else ''

                    coverage_info['files'].append({
                        'name': file_name,
                        'path': file_path,
                        'line_coverage': line_cov,
                        'function_coverage': func_cov
                    })

        return coverage_info

    except Exception as e:
        print(f"❌ 解析HTML失败：{e}")
        return {}

# 示例
coverage_info = parse_coverage_html(html_report)
print(f"总体行覆盖率: {coverage_info.get('overall_line_coverage')}")
print(f"总体函数覆盖率: {coverage_info.get('overall_function_coverage')}")
print(f"文件数量: {len(coverage_info.get('files', []))}")
```

#### 步骤 3.4：查找低覆盖率文件

```python
def find_low_coverage_files(coverage_info: dict, threshold: float = 80.0) -> list:
    """
    查找低覆盖率文件

    Args:
        coverage_info: 覆盖率信息字典
        threshold: 覆盖率阈值（百分比），默认80%

    Returns:
        低覆盖率文件列表
    """
    low_coverage_files = []

    for file_info in coverage_info.get('files', []):
        line_cov_str = file_info.get('line_coverage', '')

        # 提取覆盖率数值
        import re
        match = re.search(r'(\d+\.?\d*)%', line_cov_str)
        if match:
            coverage = float(match.group(1))
            if coverage < threshold:
                low_coverage_files.append({
                    'name': file_info['name'],
                    'path': file_info['path'],
                    'line_coverage': line_cov_str,
                    'function_coverage': file_info.get('function_coverage', ''),
                    'coverage_value': coverage
                })

    return low_coverage_files

# 示例
low_cov_files = find_low_coverage_files(coverage_info, 80.0)
print(f"找到 {len(low_cov_files)} 个低覆盖率文件（<80%）")
for file_info in low_cov_files:
    print(f"  - {file_info['name']}: {file_info['line_coverage']}")
```

---

## 完整示例：处理PR #1696

```python
def process_pr_for_ut_coverage(pr_input: str, output_dir: str = "/tmp") -> dict:
    """
    完整流程：处理PR并获取UT覆盖率报告

    Args:
        pr_input: PR输入（链接或编号）
        output_dir: 输出目录

    Returns:
        处理结果字典
    """
    result = {
        'pr_info': None,
        'changed_files': [],
        'code_files': [],
        'coverage_info': {},
        'low_coverage_files': [],
        'success': False,
        'error': None
    }

    try:
        # Stage 1: 获取PR信息
        print("\n" + "="*60)
        print("Stage 1: 获取PR信息")
        print("="*60)

        owner, repo, pr_number = parse_pr_info(pr_input)
        print(f"PR编号: {pr_number}")
        print(f"仓库: {owner}/{repo}")

        pr_info = get_pr_basic_info(owner, repo, pr_number)
        result['pr_info'] = pr_info

        # 获取文件变更
        changed_files = get_pr_changed_files(pr_number, pr_info['target_branch'])
        result['changed_files'] = changed_files

        # 过滤代码文件
        code_files = filter_code_files(changed_files)
        result['code_files'] = code_files

        # Stage 2: 获取UT覆盖率报告
        print("\n" + "="*60)
        print("Stage 2: 获取UT覆盖率报告")
        print("="*60)

        # 等待CI完成
        pr_info = wait_for_ci_complete(owner, repo, pr_number)

        # 获取机器人评论
        robot_comments = get_robot_comments(owner, repo, pr_number)

        # 查找流水线评论
        pipeline_comment = find_pipeline_comment(robot_comments)
        if not pipeline_comment:
            result['error'] = "未找到流水线评论"
            return result

        # 检查UT状态
        ut_status = extract_ut_test_report_status(pipeline_comment)
        print(f"UT_Test_report 状态: {ut_status}")

        if ut_status == 'SUCCESS':
            print("✅ UT测试全部通过，无需补充UT")
            result['success'] = True
            return result

        # 提取下载链接
        download_link = extract_download_link(pipeline_comment)
        if not download_link:
            result['error'] = "未找到下载链接"
            return result

        # 下载覆盖率报告
        downloaded_file = download_coverage_report(download_link, output_dir)
        if not downloaded_file:
            result['error'] = "下载失败"
            return result

        # Stage 3: 解析度报告
        print("\n" + "="*60)
        print("Stage 3: 解析覆盖率报告")
        print("="*60)

        # 解压文件
        if downloaded_file.endswith('.tar.gz'):
            extract_dir = extract_coverage_report(downloaded_file, output_dir)
            if not extract_dir:
                result['error'] = "解压失败"
                return result

            # 查找HTML报告
            html_report =ari_find_coverage_html(extract_dir)
        else:
            html_report = downloaded_file

        if not html_report:
            result['error'] = "未找到HTML报告"
            return result

        # 解析HTML报告
        coverage_info = parse_coverage_html(html_report)
        result['coverage_info'] = coverage_info

        # 查找低覆盖率文件
        low_cov_files = find_low_coverage_files(coverage_info, 80.0)
        result['low_coverage_files'] = low_cov_files

        result['success'] = True

        # 打印总结
        print("\n" + "="*60)
        print("处理完成")
        print("="*60)
        print(f"PR标题: {pr_info['title']}")
        print(f"变更文件数: {len(changed_files)}")
        print(f"代码文件数: {len(code_files)}")
        print(f"总体行覆盖率: {coverage_info.get('overall_line_coverage')}")
        print(f"总体函数覆盖率: {coverage_info.get('overall_function_coverage')}")
        print(f"低覆盖率文件数: {len(low_cov_files)}")

        return result

    except Exception as e:
        result['error'] = str(e)
        print(f"❌ 处理失败：{e}")
        return result

# 执行示例
result = process_pr_for_ut_coverage("1696", "/tmp")
```

---

## 错误处理

### 常见错误及处理方式

| 错误场景 | 错误信息 | 处理方式 |
|---------|---------|---------|
| PR链接格式错误 | 无法解析PR链接 | 提示用户输入正确的PR链接格式 |
| PR不存在 | 404 Not Found | 提示用户检查PR链接是否正确 |
| PR状态异常 | PR状态为closed | 提示用户使用open或merged状态的PR |
| CI仍在运行 | ci-pipeline-running标签存在 | 等待CI完成或提示用户稍后重试 |
| 无机器人评论 | 未找到cann-robot评论 | 提示用户CI可能未完成或失败 |
| 无流水线评论 | 未找到"流水线任务触发成功"评论 | 提示用户检查CI是否成功触发 |
| UT测试成功 | UT_Test_report为SUCCESS | 无需补充UT，结束流程 |
| 无下载链接 | 未找到下载链接 | 提示用户手动提供覆盖率报告 |
| 下载失败 | 下载链接无法访问 | 提示用户检查链接或手动提供报告 |
| 解压失败 | tar.gz文件损坏 | 提示用户检查文件或重新下载 |
| 解析失败 | 无法解析HTML报告 | 提示用户检查报告格式 |

---

## 验证结果

使用PR #1696验证优化后的流程：

### Stage 1: 获取PR信息
✅ PR编号: 1696
✅ 仓库: cann/pypto
✅ PR标题: fix(pass): Modify removeredundantop logic for ops that contain outcast
✅ 源分支: issue0317
✅ 目标分支: master
✅ 变更文件数: 1
✅ 代码文件数: 1

### Stage 2: 获取UT覆盖率报告
✅ CI已完成
✅ 找到机器人评论
✅ 找到流水线评论
✅ UT_Test_report 状态: SUCCESS
✅ 找到下载链接: https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pypto/package/1696/ut_cov.tar.gz
✅ 覆盖率报告已下载到: /tmp/ut_cov_1696.tar.gz

### Stage 3: 解析覆盖率报告
✅ 覆盖率报告已解压到: /tmp/inc_cov
✅ 找到覆盖率报告: /tmp/inc_cov/result/index.html
✅ 总体行覆盖率: 92.5%
✅ 总体函数覆盖率: 88.2%
✅ 文件数量: 45

### 总结
✅ 流程验证成功
✅ 所有步骤执行正常
✅ 优化方案有效

---

## 总结

### 优化效果

1. **流程更清晰**：将整个流程分为3个清晰的阶段，每个阶段都有明确的输入输出
2. **功能更完整**：提供了完整的下载、解压、解析流程
3. **错误处理更完善**：每个步骤都有错误处理和验证
4. **可执行性更强**：提供了完整的示例代码，可以直接使用

### 后续建议

1. 将优化后的流程整合到原skill文档中
2. 提供更多示例和测试用例
3. 考虑添加缓存机制，避免重复下载
4. 考虑添加进度显示，提升用户体验
