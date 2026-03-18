---
name: pass-ut-generate
description: 根据Pass业务描述生成PyPTO Pass单元测试用例（UT）。支持指定Pass模块、指定功能、功能相关Pass、UT用例示意图、PR修改内容等多种场景。提供手动构建计算图和ComputationalGraphBuilder两种生成流程，包含完整的错误诊断和故障排查机制。
license: 完整条款见 LICENSE.txt
---

# Pass 业务单元测试生成

## 概述

本技能用于根据用户描述生成 PyPTO Pass 侧单元测试用例（UT）。结合 `pypto-pass-module-analyzer` 技能分析 Pass 业务，帮助设计相关单元测试用例，确保测试覆盖充分且符合规范。

## 触发机制

当用户输入包含以下关键字或相关内容时，自动触发此技能：

| 触发场景 | 触发关键字 | 说明 |
|---------|-----------|------|
| 指定Pass的UT用例 | `设计Pass模块XXX的UT用例` | 设计指定 Pass 模块功能的 UT 用例 |
| 指定Pass的指定功能 | `设计Pass模块XXX的XXX功能` | 设计指定 Pass 模块的指定功能的 UT 用例 |
| 功能相关Pass的UT | `设计XXX功能的相关Pass模块的UT用例` | 设计与 XXX 功能相关 Pass 的 UT 用例 |
| 指定Pass UT用例示意图 | `设计Pass模块的XXX功能，UT示意图为XXX` | 设计指定 Pass 模块的指定示意图的 UT 用例 |
| 指定PR设计相关UT用例 | `设计该PR修改内容的UT` 或 `针对PR #xxx生成UT` | 设计给定 PR 的 UT 用例 |

**触发示例：**
- "设计Pass模块AutoCast的UT用例"
- "设计Pass模块AutoCast的对于不支持BF16 OP插入Cast的功能"
- "设计删除冗余Op功能的相关Pass的UT用例"
- "针对PR #1604生成UT"

## 完成标准

使用本技能生成 UT 用例时，必须满足以下验收标准：

- ✅ **UT 用例执行通过**：所有生成的测试用例必须能够编译成功并执行通过，无断言失败
- ✅ **代码覆盖率 ≥ 80%**：行覆盖率和方法覆盖率均需达到 80% 以上
- ✅ **覆盖率验证**：若覆盖率不足，需针对未覆盖的业务逻辑补充测试用例，直至达到要求
- ✅ **代码规范符合**：生成的 UT 代码需符合 CPP 编程规范，避免未使用变量、魔鬼数字等问题

**验证命令：**
```bash
# 执行测试并生成覆盖率报告
python3 build_ci.py -c -u=TestXXX.* --gcov -j=24 -f=cpp

# 查看覆盖率报告
# 打开 build/cov_result/index.html 查看详细覆盖率
```

## 使用场景

当需要设计 PyPTO Pass 中模块的功能或业务单元测试时使用此技能。

### 默认流程选择

本技能提供两种 UT 生成流程，请根据以下原则选择：

| 流程 | 适用场景 | 特点 |
|------|---------|------|
| **流程一：手动构建计算图** | 需要精细控制、复杂测试场景 | 灵活性高，可精确控制每个 Tensor 和 Operation 的创建 |
| **流程二：使用 ComputationalGraphBuilder** | 快速构建简单测试用例 | 代码简洁，适合标准化的测试场景 |

**推荐默认使用流程一（手动构建计算图）**，适合需要精细控制的场景；流程二（使用 ComputationalGraphBuilder）适合快速构建简单测试用例。

### 流程选择策略

在选择生成流程时，应遵循以下策略：

1. **检查现有测试文件**：在 `pypto/framework/tests/ut/passes/src/test_xxx.cpp` 查找对应的测试文件
2. **分析已有测试用例风格**：
   - 如果现有测试用例使用 `std::make_shared<Function>` 手动创建 → 选择流程一
   - 如果现有测试用例使用 `ComputationalGraphBuilder` → 选择流程二
3. **保持代码风格一致性**：确保新生成的 UT 用例与现有测试用例风格一致，便于维护

## Trouble Shooting 机制

当在 UT 开发过程中遇到问题时，按以下流程进行故障排查：

### 快速诊断流程

```
遇到问题
    ↓
检查错误类型
    ↓
┌─────────────┬─────────────┬─────────────┐
│  编译错误    │  运行错误    │  断言失败    │
└──────┬──────┴──────┬──────┴──────┬──────┘
       ↓             ↓             ↓
   [语法错误类]   [逻辑错误类]   [验证相关类]
       ↓             ↓             ↓
   [编译相关类]   [环境配置类]   [Pass执行类]
                      ↓
                [内存管理类]
                      ↓
                [数据结构类]
                      ↓
                [Opcode相关类]
                      ↓
                [Attribute相关类]
                      ↓
                [Tensor连接类]
                      ↓
                [遍历相关类]
```

### 错误类型快速索引

| 错误现象 | 可能原因 | 参考章节 |
|---------|---------|---------|
| 编译失败 | 语法错误、头文件缺失、命名空间错误 | 语法错误类、编译相关类 |
| 运行时时崩溃 | 空指针访问、内存泄漏、容器遍历错误 | 内存管理类、遍历相关类 |
| 断言失败 | 期望值错误、验证不完整、连接关系错误 | 验证相关类、逻辑错误类 |
| 测试通过但覆盖率低 | 测试用例覆盖不全、未测试边界情况 | 测试覆盖类 |
| Pass 执行失败 | 编译阶段配置错误、环境未重置 | 环境配置类、Pass执行类 |
| 图结构错误 | Tensor 连接关系错误、操作数错误 | Tensor连接类、逻辑错误类 |

### Top 10 最常见错误

在开始开发前，请优先查看以下最常见错误，避免重复犯错：

1. **Operations() 调用语法错误** - `Operations().()` 多余的括号
2. **变量重复定义** - 在同一测试用例中多次定义相同变量名
3. **编译阶段配置错误** - 使用了错误的 COMPILE_STAGE 值
4. **数据类型验证不完整** - 只检查操作数，不检查数据类型
5. **中文字符混入** - 输入法未切换导致代码中混入中文字符
6. **未重置 Program 和 Config** - 测试用例间未重置环境
7. **定义变量未使用** - 声明一个变量，但在后续过程中未使用
8. **智能指针使用不当** - 未使用智能指针导致内存泄漏
9. **Operation 引用获取错误** - AddOperation 返回的是引用，不是指针
10. **遍历时修改容器** - 遍历时修改容器导致未定义行为

详细错误说明和解决方案请参考 [common_errors.md](./common_errors.md)

## UT生成流程一：手动构建计算图

### 步骤 1：分析业务

根据用户描述的业务情况，分析相关业务：

| 业务类型 | 处理方式 | 示例 |
|---------|---------|------|
| 具体Pass的具体业务 | 根据业务场景设计相应UT用例 | 设计Splitk Pass消除RedunceAcc功能的UT |
| 模糊Pass业务 | 总结相关Pass业务，挑选符合业务的Pass，设计对应UT用例 | 针对Pass中对于视图类Op（view、assemble）处理的Pass设计UT |
| 设计Pass的UT | 总结相关Pass业务，分析业务场景，设计该Pass业务的UT用例 | 针对Splitk Pass设计相关UT用例 |
| 设计Pass UT用例示意图的UT | 总结相关Pass业务，分析示意图业务场景，设计该Pass业务的UT用例 | 针对Splitk Pass 消除Redunce acc功能，设计相关UT用例，UT示意图为XXX |
| 指定PR设计相关UT用例 | 读取PR链接修改的代码逻辑，分析修改内容的相关业务逻辑，设计相关的UT用例 | 设计 PR（xxx网站地址） 的 UT 用例 |

### 错误处理

| 错误场景 | 错误误信息 | 处理方式 |
---------|---------|---------|
| | CI 仍在运行 | `ci-pipeline-running` 标签存在 | 等待 3 分钟后重新检查 |
| | 无机器人评论 | 未找到 cann-robot 评论 | 提示用户用户 CI 可能未完成或失败 |
| | 无流水线评论 | 未找到"流水线任务触发成功"评论 | 提示用户检查 CI 是否成功触发 |
| | UT_Test_report 成功 | UT_Test_report 为 SUCCESS | 无需补充 UT，结束流程 |
| 无法确定状态 | UT_Test_report 状态未知 | 提示用户手动检查 |
| 无下载链接 | 未找到下载链接 | 提示用户手动提供覆盖率报告 |
| 下载失败 | 下载链接无法访问 | 提示用户检查链接或手动提供报告 |
| 解析失败 | 无法解析覆盖率报告 | 提示用户检查报告格式 |

---

## Stage 3: 针对未涉及到的代码行设计对应的 UT

### 步骤 3.1：分析未覆盖代码行

**目标**：分析低覆盖率文件中未覆盖的代码行，识别需要测试的业务逻辑。

**执行逻辑**：

1. **读取覆盖率详细报告**
    ```python
    # 使用 gcov 工具生成详细的覆盖率报告
    import subprocess

    # 执行测试并生成覆盖率报告
    subprocess result = subprocess.run([
        'python3', 'build_ci.py',
        '-c',
        '-u=TestXXX.*',
        '--gcov',
        '-j=24',
        '-f=cpp'
    ], capture_output=True, text=True)

    # 检查执行结果
    if result.returncode != 0:
        print(f"❌ 测试执行失败：{result.stderr}")
        return
    ```

2. **解析覆盖率 HTML 报告**
    ```python
    # 解析 build/cov_result/index.html
    # 提取每个文件的详细覆盖率信息
    # 包括：行覆盖率、函数覆盖率、未覆盖的行号等

    # 示例：使用 BeautifulSoup 解析 HTML
    from bs4 import BeautifulSoup

    with open('build/cov_result/index.html', 'r') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取未覆盖的代码行
    uncovered_lines = {}
    for file_path in low_coverage_files:
        file_name = file_path['file']
        # 从 HTML 中提取该文件的未覆盖行号
        # 实际实现需要根据 HTML 结构调整
        uncovered_lines[file_name] = extract_uncovered_lines(soup, file_name)
    ```

### 步骤 3.2：根据未覆盖代码行设计 UT 用例

**目标**：根据未覆盖的代码行，设计对应的 UT 用例。

**执行逻辑**：

1. **分析未覆盖代码的业务逻辑**
    ```python
    # 对每个未覆盖的代码行，分析其业务逻辑
    for file_path, lines in uncovered_lines.items():
        print(f"\n📄 文件：{file_path}")
        print(f"   未覆盖行号：{lines}")

        # 读取文件内容
        with open(file_path, 'r') as f:
            file_content = f.read()

        # 分析未覆盖行的业务逻辑
        for line_num in lines:
            line_content = file_content.split('\n')[line_num - 1]
            print(f"   行 {line_num}: {line_content.strip()}")

            # 识别业务逻辑类型
            if 'if' in line_content:
                print(f"      → 条件分支逻辑")
            elif 'for' in line_content or 'while' in line_content:
                print(f"      → 循环逻辑")
            elif 'return' in line_content:
                print(f"      → 返回逻辑")
            elif 'throw' in line_content or 'assert' in line_content:
                print(f"      → 异常处理逻辑")
    ```

2. **生成 UT 用例**
    ```python
    # 根据业务逻辑类型，生成对应的 UT 用例
    for file_path, lines in uncovered_lines.items():
        pass_module = extract_pass_module(file_path)
        
        # 选择生成流程
        flow = select_generation_flow(pass_module)

        if flow == "manual":
            # 使用流程一：手动构建计算图
            generate_ut_manual(pass_module, uncovered_lines[file_path])
        else:
            # 使用流程二：使用 ComputationalGraphBuilder
            generate_ut_builder(pass_module, uncovered_lines[file_path])
    ```

### 步骤 3.3：执行 UT 并验证

**执行逻辑**：

```bash
# 编译并执行所有测试用例
python3 build_ci.py -c -u=TestXXX.* -j=24 -f=cpp

# 执行单个测试用例
python3 build_ci.py -c -u=TestXXX.TestName -j=24 -f=cpp
```

**执行结果处理：**

| 执行结果 | 可能原因 | 处理方式 |
|---------|---------|---------|
| | 编译失败 | 语法错误、头文件缺失、命名空间错误 | 检查编译错误信息，参考 [语法错误类](./common_errors.md#语法错误类)、[编译相关类](./common_errors.md#编译相关类) |
| 运行时崩溃 | 空指针、内存错误 | 检查崩溃堆栈，参考 [内存管理类](./common_errors.md#内存管理类) |
| 断言失败 | 期望值错误 | 检查断言信息，参考 [验证相关类](./common_errors.md#验证相关类) |
| 测试超时 | 测试用例设计不当 | 简化测试用例，检查是否有死循环 |

### 步骤 3.4：重新统计 UT 覆盖率

**执行逻辑**：

```bash
# 重新执行测试并生成覆盖率报告
python3 build_ci.py -c -u=TestXXX.* --gcov -j=24 -f=cpp
```

执行结束后，在 `build/cov_result/` 目录下生成覆盖率报告，打开 `index.html` 查看覆盖率。

**覆盖率验证：**
- 行覆盖率和方法覆盖率应 ≥ 80%
- 若覆盖率仍不足，返回步骤 3.2 继续补充测试用例
- 循环直至达到覆盖率要求或无法继续补充

### 错误处理

| 错误场景 | 错误误信息 | 处理方式 |
|---------|---------|---------|
| 测试执行失败 | 测试用例执行错误 | 检查测试用例代码，修复错误 |
| 覆盖率报告生成失败 | gcov 工具执行失败 | 检查 gcov 配置，重新生成报告 |
| 覆盖率率仍不足 | 覆盖率 < 80% | 继续补充测试用例或标记为无法覆盖 |
| 无法设计 UT | 业务逻辑过于复杂 | 标记为需要人工设计 |

### 工具依赖

| 工具 | 用途 |
|------|------|
| | `gitcode_get_pull_request` | 获取 PR 基本信息 |
| `gitcode_list_pull_request_comments` | 获取 PR 评论（包含文件变更信息） |
| `build_ci.py` | 编译和执行测试用例 |
| `gcov` | 生成代码覆盖率报告 |
| `BeautifulSoup` | 解析 HTML 覆盖率报告 |


### 步骤 1.5：使用 git fetch 获取 PR 的源分支代码

**目标**：如果 PR 的源分支不在本地，使用 git fetch 获取源分支代码。

**执行逻辑**：

```python
# 切换到 PR 对应的源分支
git fetch origin alog

# 切换到 PR 对应的源分支
git checkout alog

# 验证分支切换成功
git branch -a | grep -q | head -n
```

**验证命令**：
```bash
# 查看当前分支
git branch -a

# 查看 PR 的源分支和目标分支
git show pr_info['source_branch'] pr_info['target_branch']
```

**错误处理**：
| 错误场景 | 错误误信息 | 处理方式 |
|---------|---------|---------|
| | git fetch 失败 | 无法从远程仓库获取代码 | 提示用户检查网络连接和权限 |
| | 分支切换失败 | 分支切换到失败 | 提示用户手动切换到正确的分支 |
| | 分支不存在 | 分支不存在 | 提示用户手动创建或切换分支 |
```

2. **使用 GitCode MCP 工具获取 PR 信息**
    ```python
    # 获取 PR 元数据（包含标签信息）
    pr_info = gitcode_get_pull_request(owner, repo, pull_number)

    # 验证 PR 状态（open/merged 均可继续）
    if pr_info["state"] not in ["open", "merged"]:
        print(f"⚠️ 警告：PR {pr_number} 状态为 {pr_info['state']}，建议使用 open 或 merged 状态的 PR")

    # 验证仓库是否为 pypto
    if repo != "pypto":
        print(f"❌ 错误：目标仓库不是 pypto，当前为 {repo}")
        return
    ```

3. **用户确认**
    ```
    检测到 PR 信息：
    - PR 链接：{pr_url}
    - PR 标题：{pr_info['title']}
    - PR 状态：{pr_info['state']}
    - 源分支：{pr_info['source_branch']}
    - 目标分支：{pr_info['target_branch']}

    是否继续读取该 PR 的修改内容？[y/N]
    ```

### 步骤 1.2：获取 PR 文件变更列表

**目标**：获取 PR 中所有变更文件的详细信息。

**执行逻辑**：

1. **使用 GitCode MCP 工具获取 PR 评论（包含 diff 信息）**
    ```python
    # 获取全部评论（包含 pr_comment 和 diff_comment）
    comments = gitcode_list_pull_request_comments(
        owner=owner,
        repo=repo,
        pull_number=pr_number,
        comment_type="diff_comment"  # 只获取代码行评论，包含文件变更信息
    )
    ```

2. **从评论中提取文件变更信息**
    ```python
    # 从 diff_comment 中提取变更的文件路径
    changed_files = set()
    for comment in comments:
        if 'path' in comment:
            changed_files.add(comment['path'])

    print(f"检测到 {len(changed_files)} 个变更文件")
    ```

3. **过滤代码文件**
    ```python
    # 只关注代码文件（.cpp, .h）
    code_extensions = ['.cpp', '.h', '.hpp']
    code_files = [
       f for f in files
       if any(f['filename'].endswith(ext) for ext in code_extensions)
   ]

   print(f"共 {len(files)} 个文件变更，其中 {len(code_files)} 个代码文件")
   ```

### 步骤 1.3：分析变更文件的业务逻辑

**目标**：识别每个变更文件涉及的 Pass 模块和业务逻辑。

**执行逻辑**：

1. **识别 Pass 模块**
    ```python
    # 从文件路径中提取 Pass 模块名
    # 示例：pypto/framework/src/passes/tile_graph_pass/graph_optimization/split_k.cpp
    # → Pass 模块：split_k

    pass_modules = set()
    for file_path in code_files:
        if 'passes/' in file_path:
            # 提取 passes/ 后的目录结构
            pass_path = file_path.split('passes/')[1]
            # 提取主要 Pass 模块（通常在 passes/xxx_pass/ 或 passes/xxx/ 下）
            if '_pass' in pass_path:
                pass_module = pass_path.split('_pass')[0].split('/')[-1]
            else:
                pass_module = pass_path.split('/')[0]
            pass_modules.add(pass_module)

    print(f"涉及 Pass 模块：{', '.join(pass_modules)}")
    ```

2. **分析代码变更内容**
    ```python
    # 对每个变更文件，读取文件内容并分析
    for file_path in code_files:
        print(f"\n📄 文件：{file_path}")

        # 读取文件内容
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except FileNotFoundError:
            print(f"   ⚠️ 文件不存在，跳过")
            continue

        # 识别关键修改模式
        if '&&' in file_content and '||' in file_content:
            print(f"   ⚠️ 检测到逻辑运算符修改（&& → ||）")

        if 'GetConsumers()' in file_content:
            print(f"   ⚠️ 检测到消费者数量检查修改")

        if 'PreCheck' in file_content:
            print(f"   ⚠️ 检测到 PreCheck 函数修改")
    ```

3. **业务逻辑识别**
    ```python
    # 根据变更内容，识别业务逻辑类型
    business_logic = []

    for file_path in code_files:
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except FileNotFoundError:
            continue

        # 检查是否涉及 Opcode 处理
        if 'Opcode::' in file_content:
            business_logic.append("Opcode 处理逻辑")

        # 检查是否涉及 Tensor 操作
        if 'Tensor' in file_content or 'LogicalTensor' in file_content:
            business_logic.append("Tensor 操作逻辑")

        # 检查是否涉及图遍历
        if 'for.*Operations()' in file_content or 'for.*Tensors()' in file_content:
            business_logic.append("图遍历逻辑")

        # 检查是否涉及条件判断
        if 'if.*GetOpcode()' in file_content:
            business_logic.append("条件判断逻辑")

        # 检查是否涉及消费者数量检查
        if 'GetConsumers()' in file_content or 'consumer' in file_content.lower():
            business_logic.append("消费者数量检查逻辑")

    print(f"\n识别的业务逻辑：{', '.join(set(business_logic))}")
    ```

### 步骤 1.4：生成 PR 修改分析报告

**目标**：生成 PR 修改内容的分析报告，为后续 UT 设计提供依据。

**执行逻辑**：

```python
print("\n" + "="*60)
print("PR 修改分析报告")
print("="*60)

print(f"\n📋 涉及的 Pass 模块：{', '.join(pass_modules)}")
print(f"📋 涉及的业务逻辑：{', '.join(set(business_logic))}")
print(f"📋 变更的代码文件：{len(code_files)} 个")

for file_path in code_files:
    print(f"\n📄 {file_path}")
```

### 错误处理

| 错误场景 | 错误信息 | 处理方式 |
|---------|---------|---------|
| PR 链接格式错误 | 无法解析 PR 链接 | 提示用户输入正确的 PR 链接格式 |
| PR 不存在 | `404 Not Found` | 提示用户检查 PR 链接是否正确 |
| PR 状态异常 | PR 状态为 closed/merged | 提示用户使用 open 状态的 PR |
| 无文件变更 | `files` 列表为空 | 提示用户 PR 没有代码变更 |
| 非代码文件 | 所有变更都是文档文件 | 提示用户 PR 不涉及代码修改 |
| Pass 模块识别失败 | 无法从文件路径提取 Pass 模块 | 提示用户手动指定 Pass 模块 |

---

## Stage 2: 获取 CI 冒烟的 UT report 覆盖率报告

### 步骤 2.1：检查 CI 标签状态

**目标**：检查 PR 的 CI 标签状态，确定是否需要等待 CI 完成。

**执行逻辑**：

```python
import time
import logging

def wait_for_ci_complete(owner: str, repo: str, pull_number: int) -> dict:
    """循环等待直到 ci-pipeline-running 标签消失。"""
    while True:
        pr_info = gitcode_get_pull_request(owner, repo, pull_number)
        labels = [label.get("name", "") for label in pr_info.get("labels", [])]

        if "ci-pipeline-running" not in labels:
            return pr_info

        logging.info("检测到 ci-pipeline-running 标签，CI 仍在运行")
        logging.info("等待 3 分钟后重新检查...")
        time.sleep(180)
```

### 步骤 2.2：获取机器人评论

**目标**：从 PR 评论中查找 cann-robot 机器人的评论，特别是包含流水线任务触发成功的评论。

**执行逻辑**：

1. **获取 PR 评论**
    ```python
    # 获取全部评论（包含 pr_comment 和 diff_comment）
    comments = gitcode_list_pull_request_comments(
        owner=owner,
        repo=repo,
        pull_number=pr_number,
        comment_type="pr_comment"  # 只获取普通评论
    )
    ```

2. **过滤机器人评论**
    ```python
    def is_robot_comment(comment):
        login = comment.get("user", {}).get("login", "").lower()
        if login == "cann-robot":
            return True
        return any(kw in login for kw in ["bot", "robot", "ci", "automation"])

    # 过滤机器人评论
    robot_comments = [c for c in comments if is_robot_comment(c)]
    ```

3. **查找流水线任务触发成功的评论**
    ```python
    # 从机器人评论中查找包含"流水线任务触发成功"的评论
    pipeline_comment = None
    for comment in robot_comments:
        body = comment.get("body", "")
        if "流水线任务触发成功" in body:
            pipeline_comment = body
            break

    if pipeline_comment:
        print("\n" + "="*60)
        print("流水线任务触发成功评论")
        print("="*60)
        print(pipeline_comment)
    else:
        print("⚠️ 未找到流水线任务触发成功的评论")
        return None
    ```

### 步骤 2.3：检查 UT_Test_report 状态

**目标**：从流水线评论中查找 UT_Test_report 的状态，判断是否为 FAILED。

**执行逻辑**：

```python
# 从流水线评论中提取 UT_Test_report 的状态
def extract_ut_test_report_status(pipeline_comment: str) -> str:
    """从流水线评论中提取 UT_Test_report 的状态。"""
    lines = pipeline_comment.split('\n')
    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行中的状态（SUCCESS 或 FAILED）
            if 'SUCCESS' in line:
                return 'SUCCESS'
            elif 'FAILED' in line:
                return 'FAILED'
    return 'UNKNOWN'

ut_test_report_status = extract_ut_test_report_status(pipeline_comment)

print(f"\nUT_Test_report 状态：{ut_test_report_status}")

### 步骤 2.4：提取下载链接

**目标**：从 UT_Test_report 行中提取下载链接。

**执行逻辑**：

```python
# 从流水线评论中提取 UT_Test_report 行的下载链接
def extract_download_link(pipeline_comment: str) -> str:
    """
    从流水线评论中提取 UT_Test_report 行的下载链接
    
    支持两种格式：
    1. HTML 格式：<a href=URL>
    2. Markdown 格式：[文本](URL)
    """
    import re
    lines = pipeline_comment.split('\n')
    
    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行及后续几行中的下载链接
            context = '\n'.join(lines[i:i+5])
            
            # 方法1：匹配 HTML <a href=URL> 格式
            html_link_pattern = r'<a\s+href=([^>]+)>'
            html_matches = re.findall(html_link_pattern, context)
            
            for url in html_matches:
                # 清理 URL
                url = url.strip()
                url = url.strip('"\'')
                url = re.sub(r'&gt;+', '', url)  # 移除 &gt; 字符
                url = url.rstrip('>>')
                
                if url and ('.tar.gz' in url or '.html' in url):
                    return url
            
            # 方法2：匹配 markdown 链接格式：[文本](URL)
            md_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            md_matches = re.findall(md_link_pattern, context)
            
            for text, url in md_matches:
                url = url.strip()
                url = re.sub(r'&gt;+', '', url)
                url = url.rstrip('>>')
                
                if url and ('.tar.gz' in url or '.html' in url):
                    return url
    
    return None

download_link = extract_download_link(pipeline_comment)

if download_link:
    print(f"\n找到下载链接：{download_link}")
else:
    print("⚠️ 未找到下载链接")
    return None
```
    ```python
    import requests
    import tempfile
    import os

    # 下载覆盖率报告
    response = requests.get(download_link)
    if response.status_code != 200:
        print(f"❌ 下载失败，状态码：{response.status_code}")
        return None

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as f:
        f.write(response.content)
        temp_file = f.name

    print(f"\n覆盖率报告已下载到：{temp_file}")
    ```

2. **解析覆盖率报告**
    ```python
    # 解析覆盖率报告，提取未覆盖的代码行
    def parse_coverage_report(html_file: str) -> dict:
        """解析覆盖率报告，返回未覆盖的代码行。"""
        from bs4 import BeautifulSoup

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取未覆盖的代码行
        uncovered_lines = {}

        # 查找所有文件覆盖率信息
        # 实际实现需要根据 HTML 结构调整
        file_sections = soup.find_all('div', class_='file-section')  # 示例类名

        for section in file_sections:
            file_path = section.find('span', class_='file-path').text  # 示例类名

            # 查找未覆盖的代码行
            line_elements = section.find_all('tr', class_='uncovered')  # 示例类名
            lines = []
            for elem in line_elements:
                line_num = elem.find('td', class_='line-number').text  # 示例类名
                lines.append(int(line_num))

            if lines:
                uncovered_lines[file_path] = lines

        return uncovered_lines

    uncovered_lines = parse_coverage_report(temp_file)

    # 清理临时文件
    os.unlink(temp_file)

    print(f"\n解析到 {len(uncovered_lines)} 个文件有未覆盖的代码行")
    ```

3. **显示未覆盖的代码行**
    ```python
    print("\n" + "="*60)
    print("未覆盖的代码行")
    print("="*60)

    for file_path, lines in uncovered_lines.items():
        print(f"\n📄 {file_path}")
        print(f"   未覆盖行号：{lines}")
    ```

### 错误处理

| 错误场景 | 错误信息 | 处理方式 |
|---------|---------|---------|
| CI 仍在运行 | `ci-pipeline-running` 标签存在 | 等待 3 分钟后重新检查 |
| 无机器人评论 | 未找到 cann-robot 评论 | 提示用户用户 CI 可能未完成或失败 |
| 无流水线评论 | 未找到"流水线任务触发成功"评论 | 提示用户检查 CI 是否成功触发 |
| UT_Test_report 成功 | UT_Test_report 为 SUCCESS | 无需补充 UT，结束流程 |
| 无法确定状态 | UT_Test_report 状态未知 | 提示用户手动检查 |
| 无下载链接 | 未找到下载链接 | 提示用户手动提供覆盖率报告 |
| 下载失败 | 下载链接无法访问 | 提示用户检查链接或手动提供报告 |
| 解析失败 | 无法解析覆盖率报告 | 提示用户检查报告格式 |

---

## Stage 3: 针对未涉及到的代码行设计对应的 UT

### 步骤 3.1：分析未覆盖代码行

**目标**：分析低覆盖率文件中未覆盖的代码行，识别需要测试的业务逻辑。

**执行逻辑**：

1. **读取覆盖率详细报告**
    ```python
    # 使用 gcov 工具生成详细的覆盖率报告
    import subprocess

    # 执行测试并生成覆盖率报告
    subprocess result = subprocess.run([
        'python3', 'build_ci.py',
        '-c',
        '-u=TestXXX.*',
        '--gcov',
        '-j=24',
        '-f=cpp'
    ], capture_output=True, text=True)

    # 检查执行结果
    if result.returncode != 0:
        print(f"❌ 测试执行失败：{result.stderr}")
        return
    ```

2. **解析覆盖率 HTML 报告**
    ```python
    # 解析 build/cov_result/index.html
    # 提取每个文件的详细覆盖率信息
    # 包括：行覆盖率、函数覆盖率、未覆盖的行号等

    # 示例：使用 BeautifulSoup 解析 HTML
    from bs4 import BeautifulSoup

    with open('build/cov_result/index.html', 'r') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取未覆盖的代码行
    uncovered_lines = {}
    for file_path in low_coverage_files:
        file_name = file_path['file']
        # 从 HTML 中提取该文件的未覆盖行号
        # 实际实现需要根据 HTML 结构调整
        uncovered_lines[file_name] = extract_uncovered_lines(soup, file_name)
    ```

### 步骤 3.2：根据未覆盖代码行设计 UT 用例

**目标**：根据未覆盖的代码行，设计对应的 UT 用例。

**执行逻辑**：

1. **分析未覆盖代码的业务逻辑**
    ```python
    # 对每个未覆盖的代码行，分析其业务逻辑
    for file_path, lines in uncovered_lines.items():
        print(f"\n📄 文件：{file_path}")
        print(f"   未覆盖行号：{lines}")

        # 读取文件内容
        with open(file_path, 'r') as f:
            file_content = f.read()

        # 分析未覆盖行的业务逻辑
        for line_num in lines:
            line_content = file_content.split('\n')[line_num - 1]
            print(f"   行 {line_num}: {line_content.strip()}")

            # 识别业务逻辑类型
            if 'if' in line_content:
                print(f"      → 条件分支逻辑")
            elif 'for' in line_content or 'while' in line_content:
                print(f"      → 循环逻辑")
            elif 'return' in line_content:
                print(f"      → 返回逻辑")
            elif 'throw' in line_content or 'assert' in line_content:
                print(f"      → 异常处理逻辑")
    ```

2. **生成 UT 用例**
    ```python
    # 根据业务逻辑类型，生成对应的 UT 用例
    for file_path, lines in uncovered_lines.items():
        pass_module = extract_pass_module(file_path)
        
        # 选择生成流程
        flow = select_generation_flow(pass_module)

        if flow == "manual":
            # 使用流程一：手动构建计算图
            generate_ut_manual(pass_module, uncovered_lines[file_path])
        else:
            # 使用流程二：使用 ComputationalGraphBuilder
            generate_ut_builder(pass_module, uncovered_lines[file_path])
    ```

### 步骤 3.3：执行 UT 并验证

**执行逻辑**：

```bash
# 编译并执行所有测试用例
python3 build_ci.py -c -u=TestXXX.* -j=24 -f=cpp

# 执行单个测试用例
python3 build_ci.py -c -u=TestXXX.TestName -j=24 -f=cpp
```

**执行结果处理：**

| 执行结果 | 可能原因 | 处理方式 |
|---------|---------|---------|
| 编译失败 | 语法错误、头文件缺失 | 检查编译错误信息，参考 [语法错误类](./common_errors.md#语法错误类)、[编译相关类](./common_errors.md#编译相关类) |
| 运行时崩溃 | 空指针、内存错误 | 检查崩溃堆栈，参考 [内存管理类](./common_errors.md#内存管理类) |
| 断言失败 | 期望值错误 | 检查断言信息，参考 [验证相关类](./common_errors.md#验证相关类) |
| 测试超时 | 测试用例设计不当 | 简化测试用例，检查是否有死循环 |

### 步骤 3.4：重新统计 UT 覆盖率

**执行逻辑**：

```bash
# 重新执行测试并生成覆盖率报告
python3 build_ci.py -c -u=TestXXX.* --gcov -j=24 -f=cpp
```

执行结束后，在 `build/cov_result/` 目录下生成覆盖率报告，打开打开 `index.html` 查看覆盖率。

**覆盖率验证：**
- 行覆盖率和方法覆盖率应 ≥ 80%
- 若覆盖率仍不足，返回步骤 3.2 继续补充测试用例
- 循环直至达到覆盖率要求或无法继续补充

### 错误处理

| 错误场景 | 错误信息 | 处理方式 |
|---------|---------|---------|
| 测试执行失败 | 测试用例执行错误 | 检查测试用例代码，修复错误 |
| 覆盖率报告生成失败 | gcov 工具执行失败 | 检查 gcov 配置，重新生成报告 |
| 覆盖率仍不足 | 覆盖率 < 80% | 继续补充测试用例或标记为无法覆盖 |
| 无法设计 UT | 业务逻辑过于复杂 | 标记为需要人工设计 |

### 工具依赖

| 工具 | 用途 |
|------|------|
| `gitcode_get_pull_request` | 获取 PR 基本信息 |
| `gitcode_list_pull_request_comments` | 获取 PR 评论（包含文件变更信息） |
| `build_ci.py` | 编译和执行测试用例 |
| `gcov` | 生成代码覆盖率报告 |
| `BeautifulSoup` | 解析 HTML 覆盖率报告 |

### 参考资料

| 资源类型 | 路径 |
|---------|------|
| GitCode API 文档 | https://gitcode.com/help/api/pull_requests |
| GitCode MCP 工具 | `.agents/skills/gitcode-mcp-install/SK`ILL.md |
| 业务分析技能 | `.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md` |
| PR 创建技能 | `.agents/skills/pypto-pr-creator/SKILL.md` |

---

## 业务分析技能

`.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md`

**⚠️ 常见错误：** 未全面分析 Pass 的所有功能点，导致测试用例覆盖不全
**🔧 解决方案：** 参考 [测试覆盖类 - 错误9](./common_errors.md#错误-9测试用例覆盖不全)

### 步骤 2：环境配置

为确保测试环境正确初始化，需检查测试文件状态。在 `pypto/framework/tests/ut/passes/src/test_xxx.cpp` 寻找对应的测试文件。

**文件命名规则：**
- 测试文件：`test_xxx.cpp`（xxx 为 Pass 名称）
- 测试类名：`TestXXX`（首字母大写）

**初始化环境步骤：**

1. **创建测试类**
   ```cpp
   class TestXXX : public testing::Test {
   public:
       static void SetUpTestCase() {}
       static void TearDownTestCase() {}
       void SetUp() override {}
       void TearDown() override {}
   };
   ```

2. **实现 SetUp() 方法**
   ```cpp
   void SetUp() override {
       Program::GetInstance().Reset();
       config::Reset();
       config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
       config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy");
       config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
       TileShape::Current().SetVecTile({64, 64});
       TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
   }
   ```

3. **实现其他相关方法**
    ```cpp
         (1) 所有测试用例全局初始化函数--static void SetUpTestCase() {}，若未明确指定内容，则为空实现；
         (2) 所有测试用例全局清理函数--static void TearDownTestCase() {}， 若未明确指定内容，则为空实现；
     ```
**编译阶段配置参考：**

| Pass所在目录 | COMPILE_STAGE 值 |
|-------------|-----------------|
| tensor_graph_pass | CS_TENSOR_GRAPH |
| tile_graph_pass | CS_TILE_GRAPH |
| block_graph_pass | CS_EXECUTE_GRAPH |

**⚠️ 常见错误：**
- 错误 5：错误的编译阶段配置
- 错误 12：未重置 Program 和 Config
- 错误 13：NPU 架构未恢复

**🔧 解决方案：** 参考 [环境配置类](./common_errors.md#环境配置类)

### 步骤 3：搭建测试用例框架

使用 `TEST_F` 宏创建测试用例：

```cpp
/*
TESTRemoveDummyExpand
输入图：
    inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                        ->sqrt->outCast2{8,16}
                                        ->reciprocal->outCast3{8,16}
期望输出：
    inCast{8,16}->exp->outCast1
                ->sqrt->outCast2
                ->reciprocal->outCast3
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
    // 测试代码
}
```

**⚠️ 常见错误：** 中文字符混入代码
**🔧 解决方案：** 参考 [语法错误类 - 错误2](./common_errors.md#错误-2中文字符混入代码)

### 步骤 4：构建 Function

使用智能指针创建 Function：

```cpp
auto function = std::make_shared<Function>(
    Program::GetInstance(),
    "TestXXX",        // funcMagicName
    "TestXXX",        // funcRawName
    nullptr           // parentFunc
);
ASSERT_NE(function, nullptr);
```

**⚠️ 常见错误：**
- 错误 14：智能指针使用不当
- 错误 16：未使用智能指针管理资源

**🔧 解决方案：** 参考 [内存管理类](./common_errors.md#内存管理类)

### 步骤 5：创建 Tensor

使用 `LogicalTensor` 类创建 Tensor：

```cpp
auto tensor = std::make_shared<LogicalTensor>(
    *function,                    // Function 引用
    DataType::DT_FP32,            // 数据类型
    Shape({8, 16}),               // Shape
    TileOpFormat::TILEOP_ND,     // 格式（可选）
    "tensor_name",                // 名称（可选）
    NodeType::LOCAL               // 节点类型（可选）
);
```

**⚠️ 常见错误：**
- 错误 17：Shape 比较错误
- 错误 18：DataType 比较错误
- 错误 19：直接访问成员变量

**🔧 解决方案：** 参考 [数据结构类](./common_errors.md#数据结构类)

### 步骤 6：创建 Operation 及绑定 Function 输入输出

使用 `Program::AddOperation` 创建 Operation：

```cpp
Operation &op = Program::GetInstance().AddOperation(
    Opcode::OP_ADD,                              // Opcode
    {tensor1, tensor2},                          // 输入 Tensor
    {outputTensor}                               // 输出 Tensor
);
```

**⚠️ 常见错误：**
- 错误 1：Operations() 调用语法错误
- 错误 15：Operation 引用获取错误
- 错误 20：Opcode 枚举使用错误
- 错误 22：使用错误的 Opcode 值

**🔧 解决方案：** 参考 [Opcode 相关类](./common_errors.md#opcode-相关类)

### 步骤 7：对业务功能进行校验

遍历 Function 验证 Pass 执行结果：

```cpp
// 统计 assemble 操作数量
uint32_t assemble_num = 0;
for (auto &op : function->Operations()) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        ++assemble_num;
    }
}
EXPECT_EQ(assemble_num, 0);
```

**⚠️ 常见错误：**
- 错误 6：错误的期望操作数
- 错误 7：未验证数据类型转换
- 错误 8：未验证 Tensor 连接关系
- 错误 32：遍历时修改容器

**🔧 解决方案：** 参考 [逻辑错误类](./common_errors.md#逻辑错误类)、[遍历相关类](./common_errors.md#遍历相关类)

### 步骤 8：执行 UT 并验证

**编译并执行：**
```bash
# 编译并执行所有测试用例
python3 build_ci.py -c -u=TestXXX.* -j=24 -f=cpp

# 仅执行测试用例（不清理缓存）
python3 build_ci.py -u=TestXXX.* -j=24 -f=cpp

# 执行单个测试用例
python3 build_ci.py -c -u=TestXXX.TestName -j=24 -f=cpp
```

**执行结果处理：**

| 执行结果 | 可能原因 | 处理方式 |
|---------|---------|---------|
| 编译失败 | 语法错误、头文件缺失 | 检查编译错误信息，参考 [语法错误类](./common_errors.md#语法错误类)、[编译相关类](./common_errors.md#编译相关类) |
| 运行时崩溃 | 空指针、内存错误 | 检查崩溃堆栈，参考 [内存管理类](./common_errors.md#内存管理类) |
| 断言失败 | 期望值错误 | 检查断言信息，参考 [验证相关类](./common_errors.md#验证相关类) |
| 测试超时 | 测试用例设计不当 | 简化测试用例，检查是否有死循环 |

### 步骤 9：统计 UT 覆盖率

```bash
python3 build_ci.py -c -u=TestXXX.* --gcov -j=24 -f=cpp
```

执行结束后，在 `build/cov_result/` 目录下生成覆盖率报告，打开 `index.html` 查看覆盖率。

**覆盖率要求：**
- 行覆盖率和方法覆盖率应 ≥ 80%
- 若覆盖率不足，针对未覆盖的业务补充测试用例

**⚠️ 常见错误：**
- 错误 9：测试用例覆盖不全
- 错误 10：未测试架构差异
- 错误 11：未测试边界情况

**🔧 解决方案：** 参考 [测试覆盖类](./common_errors.md#测试覆盖类)

## UT生成流程二：使用 ComputationalGraphBuilder

### 步骤 1-3：同流程一

分析业务、环境配置、搭建测试用例框架的步骤与流程一相同。

### 步骤 4：使用 ComputationalGraphBuilder 构建 Function

**创建 ComputationalGraphBuilder：**
```cpp
ComputationalGraphBuilder builder(*function);
```

**添加 Tensor：**
```cpp
builder.AddTensor(DataType::DT_FP32, Shape({8, 16}), "input");
builder.AddTensors({DataType::DT_FP32, DataType::DT_FP32}, {Shape({8, 16}), Shape({8, 16})}, {"tensor1", "tensor2"});
```

**添加 Operation：**
```cpp
builder.AddOp(Opcode::OP_ADD, {"input", "bias"}, {"output"});
builder.AddOps({Opcode::OP_ADD, Opcode::OP_MUL}, {{"input1", "input2"}, {"input3", "input4"}}, {"output1", "output2"});
```

**设置输入输出 Cast：**
```cpp
builder.SetInCast(DataType::DT_FP16);
builder.SetOutCast(DataType::DT_FP32);
```

**⚠️ 常见错误：**
- 错误 23：Attribute 获取错误
- 错误 24：Attribute 类型转换错误
- 错误 25：未检查 Attribute 是否为空

**🔧 解决方案：** 参考 [Attribute 相关类](./common_errors.md#attribute-相关类)

### 步骤 5-7：同流程一

对业务功能进行校验、执行 UT 并验证、统计 UT 覆盖率的步骤与流程一相同。

## PR 读取与 UT 设计流程

> ⚠️ **当用户指定 PR 链接时，必须使用本流程读取 PR 内容并设计 UT 用例**

## Stage 1: 拉取 PR 信息，获得 PR 修改内容

### 步骤 1.1：PR 链接解析与验证

**目标**：从用户提供的 PR 链接中提取必要信息，验证 PR 可访问性。

**执行逻辑**：

1. **解析 PR 链接**
    ```python
    import re

    def parse_pr_info(pr_input: str) -> tuple:
        """
        解析 PR 链接，返回 (owner, repo, pr_number)
        
        支持的 PR 链接格式：
        - https://gitcode.com/cann/pypto/pull/123
        - https://gitcode.com/<username>/pypto/pull/456
        - 简短格式：cann/pypto/123 或 <username>/pypto/456
        - PR 编号格式：#123 或 PR #123
        """
        # 匹配 URL 格式
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
        
        raise ValueError(f"无法解析 PR 链接：{pr_input}")

    # 解析示例
    pr_url = "https://gitcode.com/cann/pypto/pull/123"
    owner, repo, pr_number = parse_pr_info(pr_url)
    # owner = "cann"          # 仓库所有者
    # repo = "pypto"          # 仓库名称
    # pr_number = 123          # PR 编号
    ```

2. **使用 GitCode MCP 工具获取 PR 信息**
    ```python
    # 获取 PR 元数据（包含标签信息）
    pr_info = gitcode_get_pull_request(owner, repo, pull_number)

    # 验证 PR 状态（open/merged 均可继续）
    if pr_info["state"] not in ["open", "merged"]:
        print(f"⚠️ 警告：PR {pr_number} 状态为 {pr_info['state']}，建议使用 open 或 merged 状态的 PR")

    # 验证仓库是否为 pypto
    if repo != "pypto":
        print(f"❌ 错误：目标仓库不是 pypto，当前为 {repo}")
        return
    ```

3. **用户确认**
    ```
    检测到 PR 信息：
    - PR 链接：{pr_url}
    - PR 标题：{pr_info['title']}
    - PR 状态：{pr_info['state']}
    - 源分支：{pr_info['source_branch']}
    - 目标分支：{pr_info['target_branch']}

    是否继续读取该 PR 的修改内容？[y/N]
    ```

### 步骤 1.2：获取 PR 文件变更列表 与 拉取 PR 代码

**目标**：获取 PR 中所有变更文件的详细信息。

**执行逻辑**：

1. **使用 git 命令获取 PR 文件变更列表**
    ```python
    import subprocess

    def get_pr_changed_files(pr_number: int, target_branch: str = "master") -> list:
        """
        使用 git 命令获取 PR 的变更文件列表
        
        Args:
            pr_number: PR 编号
            target_branch: 目标分支，默认为 master
        
        Returns:
            变更的文件列表
        """
        try:
            # 步骤一：拉取 PR 的远程分支
            subprocess.run([
                'git', 'fetch', 'origin',
                f'pull/{pr_number}/head:pr_{pr_number}'
            ], check=True, capture_output=True, text=True)
            
            # 步骤二：获取变更的文件列表
            result = subprocess.run([
                'git', 'diff', '--name-only', target_branch, f'pr_{pr_number}'
            ], check=True, capture_output=True, text=True)
            
            changed_files = result.stdout.strip().split('\n')
            changed_files = [f for f in changed_files if f]
            
            return changed_files
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 获取文件变更失败: {e}")
            print(f"错误输出: {e.stderr}")
            return []

    # 获取变更文件
    changed_files = get_pr_changed_files(pr_number, "master")
    print(f"检测到 {len(changed_files)} 个变更文件")
    ```

2. **过滤代码文件**
    ```python
    # 只关注代码文件（.cpp, .h, .hpp）
    code_extensions = ['.cpp', '.h', '.hpp']
    code_files = [
       f for f in changed_files
       if any(f.endswith(ext) for ext in code_extensions)
   ]
    print(f"共 {len(changed_files)} 个文件变更，其中 {len(code_files)} 个代码文件")
    ```

3. **切换到 PR 分支（可选）**
    ```bash
    # 如果需要查看 PR 的完整代码，可以切换到 PR 分支
    git checkout pr_XXXX  # XXXX 为 PR 编号
    
    # 查看当前分支
    git branch
    ```
        
### 步骤 1.3：分析变更文件的业务逻辑

**目标**：识别每个变更文件涉及的 Pass 模块和业务逻辑。

**执行逻辑**：

1. **识别 Pass 模块**
    ```python
    # 从文件路径中提取 Pass 模块名
    # 示例：pypto/framework/src/passes/tile_graph_pass/graph_optimization/split_k.cpp
    # → Pass 模块：split_k

    pass_modules = set()
    for file_path in code_files:
        if 'passes/' in file_path:
            # 提取 passes/ 后的目录结构
            pass_path = file_path.split('passes/')[1]
            # 提取主要 Pass 模块（通常在 passes/xxx_pass/ 或 passes/xxx/ 下）
            if '_pass' in pass_path:
                pass_module = pass_path.split('_pass')[0].split('/')[-1]
            else:
                pass_module = pass_path.split('/')[0]
            pass_modules.add(pass_module)

    print(f"涉及 Pass 模块：{', '.join(pass_modules)}")
    ```

2. **分析代码变更内容**
    ```python
    # 对每个变更文件，读取文件内容并分析
    for file_path in code_files:
        print(f"\n📄 文件：{file_path}")

        # 读取文件内容
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except FileNotFoundError:
            print(f"   ⚠️ 文件不存在，跳过")
            continue

        # 识别关键修改模式
        if '&&' in file_content and '||' in file_content:
            print(f"   ⚠️ 检测到逻辑运算符修改（&& → ||）")

        if 'GetConsumers()' in file_content:
            print(f"   ⚠️ 检 检     检测到消费者数量检查修改")

        if 'PreCheck' in file_content:
            print(f"   ⚠️ �测到 PreCheck 函数修改")
    ```

3. **业务逻辑识别**
    ```python
    # 根据变更内容，识别业务逻辑类型
    business_logic = []

    for file_path in code_files:
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except FileNotFoundError:
            continue

        # 检查是否涉及 Opcode 处理
        if 'Opcode::' in file_content:
            business_logic.append("Opcode 处理逻辑")

        # 检查是否涉及 Tensor 操作
        if 'Tensor' in file_content or 'LogicalTensor' in file_content:
            business_logic.append("Tensor 操作逻辑")

        # 检查是否涉及图遍历
        if 'for.*Operations()' in file_content or 'for.*Tensors()' in file_content:
            business_logic.append("图遍历逻辑")

        # 检查是否涉及条件判断
        if 'if.*GetOpcode()' in file_content:
            business_logic.append("条件判断逻辑")

        # 检查是否涉及消费者数量检查
        if 'GetConsumers()' in file_content or 'consumer' in file_content.lower():
            business_logic.append("消费者数量检查逻辑")

    print(f"\n识别的业务逻辑：{', '.join(set(business_logic))}")
    ```

### 步骤 1.4：生成 PR 修改分析报告

**目标**：生成 PR 修改内容的分析报告，为后续 UT 设计提供依据。

**执行逻辑**：

```python
print("\n" + "="*60)
print("PR 修改分析报告")
print("="*60)

print(f"\n📋 涉及的 Pass 模块：{', '.join(pass_modules)}")
print(f"📋 涉及的业务逻辑：{', '.join(set(business_logic))}")
print(f"📋 变更的代码文件：{len(code_files)} 个")

for file_path in code_files:
    print(f"\n📄 {file_path}")
```

### 错误处理

| 错误场景 | 错误误信息 | 处理方式 |
|---------|---------|---------|
| | PR 链接格式错误 | 无法解析 PR 链接 | 提示用户输入正确的 PR 链接格式 |
| | PR 不存在 | `404 Not Found` | 提示用户检查 PR 链接是否正确 |
| | PR 状态异常 | PR 状态为 closed/merged | 提示用户使用 open 状态的 PR |
| 无文件变更 | `files` 列表为空 | 提示用户 PR 没有代码变更 |
| 非代码文件 | 所有变更都是文档文件 | 提示用户 PR 不涉及代码修改 |
| Pass 模块识别失败 | 无法从文件路径提取 Pass 模块 | 提示用户手动指定 Pass 模块 |

---

## Stage 2: 获取 CI 冒烟的 UT report 覆盖率报告

### 步骤 2.1：检查 CI 标签状态

**目标**：检查 PR 的 CI 标签状态，确定是否需要等待 CI 完成。

**执行逻辑**：

```python
import time
import logging

def wait_for_ci_complete(owner: str, repo: str, pull_number: int) -> dict:
    """循环等待直到 ci-pipeline-running 标签消失。"""
    while True:
        pr_info = gitcode_get_pull_request(owner, repo, pull_number)
        labels = [label.get("name", "") for label in pr_info.get("labels", [])]

        if "ci-pipeline-running" not in labels:
            return pr_info

        logging.info("检测到 ci-pipeline-running 标签，CI 仍在运行")
        logging.info("等待 3 分钟后重新检查...")
        time.sleep(180)
```

### 步骤 2.2：获取机器人评论

**目标**：从 PR 评论中查找 cann-robot 机器人的评论，特别是包含流水线任务触发成功的评论。

**执行逻辑**：

1. **获取 PR 评论**
    ```python
    # 获取全部评论（包含 pr_comment 和 diff_comment）
    comments = gitcode_list_pull_request_comments(
        owner=owner,
        repo=repo,
        pull_number=pr_number,
        comment_type="pr_comment"  # 只获取普通评论
    )
    ```

2. **过滤机器人评论**
    ```python
    def is_robot_comment(comment):
        login = comment.get("user", {}).get("login", "").lower()
        if login == "cann-robot":
            return True
        return any(kw in login for kw in ["bot", "robot", "ci", "automation"])

    # 过滤机器人评论
    robot_comments = [c for c in comments if is_robot_comment(c)]
    ```

3. **查找流水线任务触发成功的评论**
    ```python
    # 从机器人评论中查找包含"流水线任务触发成功"的评论
    pipeline_comment = None
    for comment in robot_comments:
        body = comment.get("body", "")
        if "流水线任务触发成功" in body:
            pipeline_comment = body
            break

    if pipeline_comment:
        print("\n" + "="*60)
        print("流水线任务触发成功评论")
        print("="*60)
        print(pipeline_comment)
    else:
        print("⚠️️ 未找到流水线任务触发成功的评论")
        return None
    ```

### 步骤 2.3：检查 UT_Test_report 状态

**目标**：从流水线评论中查找 UT_Test_report 的状态，判断是否为 FAILED。

**执行逻辑**：

```python
# 从流水线评论中提取 UT_Test_report 的状态
def extract_ut_test_report_status(pipeline_comment: str) -> str:
    """从流水线评论中提取 UT_Test_report 的状态。"""
    lines = pipeline_comment.split('\n')
    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行中的状态（SUCCESS 或 FAILED）
            if 'SUCCESS' in line:
                return 'SUCCESS'
            elif 'FAILED' in line:
                return 'FAILED'
    return 'UNKNOWN'

ut_test_report_status = extract_ut_test_report_status(pipeline_comment)

print(f"\nUT_Test_report 状态：{ut_test_report_status}")

if ut_test_report_status == 'SUCCESS':
    print("✅ UT_Test_report 为 SUCCESS，无需补充 UT")
    return None
elif ut_test_report_status == 'FAILED':
    print("❌ UT_Test_report 为 FAILED，需要补充 UT")
else:
    print("⚠️ 无法确定 UT_Test_report 状态")
    return None
```

### 步骤 2.4：提取下载链接

**目标**：从 UT_Test_report 行中提取下载链接。

**执行逻辑**：

```python
# 从流水线评论中提取 UT_Test_report 行的下载链接
def extract_download_link(pipeline_comment: str) -> str:
    """从流水线评论中提取 UT_Test_report 行的下载链接。"""
    lines = pipeline_comment.split('\n')
    for i, line in enumerate(lines):
        if 'UT_Test_report' in line:
            # 查找该行中的下载链接
            import re
            # 匹配 markdown 链接格式：[文本](URL)
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            matches = re.findall(link_pattern, line)
            for text, url in matches:
                if '下载' in text or 'download' in text.lower():
                    return url
    return None

download_link = extract_download_link(pipeline_comment)

if download_link:
    print(f"\n找到下载链接：{download_link}")
else:
    print("⚠️ 未找到下载链接")
    return None
```

### 步骤 2.5：下载并解析覆盖率报告

**目标**：从下载链接下载覆盖率报告，并解析未覆盖的代码行。

**执行逻辑**：

1. **下载覆盖率报告**
    ```python
    import requests
    import tempfile
    import os

    # 下载覆盖率报告
    response = requests.get(download_link)
    if response.status_code != 200:
        print(f"❌ 下载失败，状态码：{response.status_code}")
        return None

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as f:
        f.write(response.content)
        temp_file = f.name

    print(f"\n覆盖率报告已下载到：{temp_file}")
    ```

2. **解析覆盖率报告**
    ```python
    # 解析覆盖率报告，提取未覆盖的代码行
    def parse_coverage_report(html_file: str) -> dict:
        """解析覆盖率报告，返回未覆盖的代码行。"""
        from bs4 import BeautifulSoup

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取未覆盖的代码行
        uncovered_lines = {}

        # 查找所有文件覆盖率信息
        # 实际实现需要根据 HTML 结构调整
        file_sections = soup.find_all('div', class_='file-section')  # 示例类名

        for section in file_sections:
            file_path = section.find('span', class_='file-path').text  # 示例类名

            # 查找未覆盖的代码行
            line_elements = section.find_all('tr', class_='uncovered')  # 示例类名
            lines = []
            for elem in line_elements:
                line_num = elem.find('td', class_='line-number').text  # 示例类名
                lines.append(int(line_num))

            if lines:
                uncovered_lines[file_path] = lines

        return uncovered_lines

    uncovered_lines = parse_coverage_report(temp_file)

    # 清理临时文件
    os.unlink(temp_file)

    print(f"\n解析到 {len(uncovered_lines)} 个文件有未覆盖的代码行")
    ```

3. **显示未覆盖的代码行**
    ```python
    print("\n" + "="*60)
    print("未覆盖的代码行")
    print("="*60)

    for file_path, lines in uncovered_lines.items():
        print(f"\n📄 {file_path}")
        print(f"   未覆盖行号：{lines}")
    ```

## 注意事项

### 代码规范

- 遵循 CPP 代码编程规范
- 避免未使用的变量定义
- 未修改的引用加入 const
- 避免魔鬼数字，使用命名常量

### 测试验证

- 生成的用例必须真实执行并验证
- 执行前参考 [common_errors.md](./common_errors.md) 查看最容易犯发生的错误（Top 10）
- 打印当前所设计 UT 的覆盖率

### 错误处理

遇到问题时，按以下步骤进行故障排查：

1. **识别错误类型**：根据错误信息判断是编译错误、运行时错误还是断言失败
2. **查找对应章节**：使用错误类型快速索引表找到对应的错误类别
3. **查看详细说明**：在 [common_errors.md](./common_errors.md) 中查看详细的错误说明和解决方案
4. **应用解决方案**：按照文档中的解决方案修复问题
5. **验证修复结果**：重新执行测试用例，确认问题已解决

## 参考资料

| 类别 | 文件路径 |
| |------|----------|
| | 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
| | 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` |
| | ComputationalGraphBuilder | `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` |
| | Function | `pypto/framework/src/interface/function/function.h` |
| | Program | `pypto/framework/src/interface/program/program.cpp` |
| | Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` |
| | Opcode 实现 | `pypto/framework/src/interface/operation/opcode.cpp` |
| | Operation 属性 | `pypto/framework/src/interface/operation/attribute.h` |
| | Operation 实现 | `pypto/framework/src/interface/operation/operation.cpp` |
| | DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
| | LogicalTensor | `pypto/framework/src/interface/tensor/logical_tensor.h` |
| | COMPILE_STAGE 策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` |
