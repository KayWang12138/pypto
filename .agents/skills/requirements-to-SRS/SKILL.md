---
name: requirements-to-SRS
description: "将 JSON 格式的需求信息转换为 Word 形式的软件设计 SRS 文档。扫描项目文件提取现有 C++ 接口，结合需求生成符合规范的软件接口设计。触发词：生成 SRS 文档、将需求转换为 SRS、创建软件设计文档。"
---

# 需求转 SRS 文档生成器

将 JSON 格式的需求列表结合项目文件进行扩写，转换为符合规范的 Word SRS 文档。

## 输入

用户会提供一个需求信息 JSON 文件，结构如下：

```json
{
  "requirements": [
    {
      "title": "需求标题",
      "background": "需求背景",
      "description": "需求描述",
      "dependencies": ["可选：若明确无依赖可设为空数组 []"],
      "sub_requirements": [
        {
          "title": "子需求标题",
          "logic": "实现逻辑"
        }
      ]
    }
  ]
}
```

**字段说明**：

| 字段 | 必需 | 说明 |
|------|------|------|
| title | 是 | 需求标题 |
| background | 是 | 需求背景描述 |
| description | 是 | 需求功能描述 |
| dependencies | 否 | 若明确无依赖可设为 `[]`，否则可省略 |
| sub_requirements | 是 | 子需求列表 |

## 参考文件

| File | Purpose | Load Timing |
|------|---------|-------------|
| [scripts/generate_srs.py](scripts/generate_srs.py) | 主脚本：解析 JSON、扫描项目、生成 Word 文档 | 执行时加载 |
| [templates.json](templates.json) | 模板文件：包含逻辑扩展模板、类型映射等 | 执行时加载 |
| [references/srs_format.md](references/srs_format.md) | SRS 文档格式规范说明 | 生成文档时按需加载 |

## 工作流程

### 步骤 1：验证输入

1. 检查 JSON 文件是否存在
2. 验证 JSON 格式是否正确
3. 检查必需字段是否存在：
   - `requirements` 数组（必需，不能为空）
   - 每个需求的 `title` 字段（必需）
4. 若验证失败，报告具体错误并终止

### 步骤 2：检测项目环境

**Agent 必须执行此步骤**，确定项目信息获取方式：

#### 2.1 检测当前目录

1. **检查当前工作目录名称**：获取当前工作目录的名称
2. **检查 README.md 文件**：
   - 读取当前目录下的 `README.md` 文件
   - 验证文件内容是否描述 pypto 项目
3. **判断条件**：
   - 若目录名称为 `pypto` **且** README.md 描述的是 pypto 项目 → 使用当前目录
   - 否则 → 通过网络访问项目仓库

#### 2.2 获取项目信息

**情况一：使用当前目录**

若检测到当前目录为 pypto 项目，则：
1. 使用 Read 工具读取本地 `README.md` 文件
2. 使用 Glob 工具扫描项目结构（`**/*.h`, `**/CMakeLists.txt`）
3. 使用 Read 工具读取 `CMakeLists.txt` 获取依赖信息
4. 使用 Grep 工具搜索接口定义

**情况二：访问远程仓库**

若当前目录不是 pypto 项目，则：
1. **访问项目仓库**：使用 WebFetch 工具访问 https://gitcode.com/cann/pypto
2. **获取项目结构**：分析项目的目录结构和模块组成
3. **阅读 README.md**：了解项目背景、目标和技术栈
4. **分析 CMakeLists.txt**：提取项目的编译依赖和组件关系
5. **扫描接口文件**：查看 `framework/include/tilefwk` 目录下的头文件

**获取的关键信息**：
- 项目整体架构和模块划分
- 现有组件之间的依赖关系
- 核心接口和数据结构
- 第三方库依赖

### 步骤 3：扩写需求内容

**关键规则：文件和方法验证**

在扩写内容时，若涉及具体的文件路径或函数/方法名称，**必须**进行验证：

#### 3.0 文件和方法验证规则

##### 文件存在性验证

**规则**：扩写内容中提及的具体文件路径，必须通过 Glob 工具确认文件存在于项目中。

**验证流程**：
```
步骤 1：提取文件路径
- 从扩写内容中识别文件路径（如 framework/platform/device.py）

步骤 2：使用 Glob 验证
- 执行：Glob framework/platform/device.py
- 或使用通配符：Glob **/device.py

步骤 3：处理验证结果
- 若文件存在 → 可写入 SRS
- 若文件不存在 → 不能写入具体路径，改为使用模块描述
```

**示例**：
```
❌ 错误：framework/compiler/tile_compiler.py（未验证）
✅ 正确：先 Glob framework/compiler/tile_compiler.py 确认存在
         若存在，则可写入；若不存在，不得写入；
```

##### 方法存在性验证

**规则**：扩写内容中提及的具体函数/方法名称，必须通过 Grep 工具确认函数存在于对应文件中。

**验证流程**：
```
步骤 1：提取函数名和所属文件
- 函数名：GetHardwareParam
- 所属文件：framework/platform/interface.py

步骤 2：使用 Grep 验证
- 执行：Grep "def GetHardwareParam" framework/platform/interface.py
- 或：Grep "GetHardwareParam" framework/platform/

步骤 3：处理验证结果
- 若函数存在 → 可写入 SRS，说明具体接口
- 若函数不存在 → 不能写入
```

**示例**：
```
❌ 错误：调用 Platform 的 GetHardwareParam 接口（未验证）
✅ 正确：先 Grep "GetHardwareParam" framework/platform/ 确认存在
         若存在，则可写入具体接口名；若不存在，不可写入
```

##### 验证失败的处理方式

| 验证类型 | 验证失败时的处理 |
|---------|-----------------|
| 文件路径 | 使用模块名称替代具体路径，如"Platform模块"替代"framework/platform/device.py" |
| 函数名称 | 使用功能描述替代具体函数名，如"获取硬件参数接口"替代"GetHardwareParam" |
| 类名称 | 使用组件描述替代具体类名，如"设备管理组件"替代"DeviceManager类" |

##### Agent 执行验证的时机

```
生成扩写 JSON 前必须执行：
1. 遍历 expanded_dependencies 中的文件路径 → Glob 验证
2. 遍历 expanded_dependencies 中的函数名 → Grep 验证
3. 遍历 expanded_description 中的文件路径 → Glob 验证
4. 遍历 expanded_description 中的函数名 → Grep 验证

验证失败的项 → 替换为通用描述后再写入 JSON
```

#### 3.1 扩写背景和描述

**扩写内容获取优先级**：

1. **优先：扫描项目文件**
   - 使用 Read 工具读取项目 README.md、文档文件
   - 使用 Glob 工具扫描相关源文件
   - 使用 Grep 工具搜索关键词获取上下文
   - 从项目代码和注释中提取技术细节

2. **备选：网络搜索**
   - 仅当项目文件中不存在相关资料时使用
   - 检索相关技术文档、最佳实践、行业标准
   - 了解类似功能的实现方案

**扩写流程**：
```
步骤 1：尝试从项目文件获取信息
1. Read README.md - 获取项目背景和目标
2. Read docs/*.md - 获取技术文档
3. Grep 关键词 - 搜索相关代码和注释

步骤 2：评估信息完整性
- 若项目文件信息充足 → 直接使用
- 若项目文件信息不足 → 使用网络搜索补充

步骤 3：整合扩写内容
- 结合项目文件和网络搜索结果
- 生成完整的扩写内容
```

#### 3.2 扩写子模块描述

为每个子需求生成：
- `description`：子模块功能描述（约150字符）
- `flow_steps`：流程步骤列表（动词+名词格式）

#### 3.3 扩写特性依赖

**重要说明**：`expanded_dependencies` 字段用于描述**本次更新可能影响到的 pypto 项目内部组件**，**只能通过扫描项目文件获取，禁止通过网络搜索获取**。

##### 特性依赖的定义

**特性依赖**是指本次需求更新可能影响到的 pypto 项目内部组件或模块，包括：
- 直接依赖的内部模块（如被调用的函数、类）
- 间接影响的组件（如数据结构变更影响的模块）
- 需要同步修改的关联模块
- 接口变更影响的上游/下游组件

##### 无依赖的情况

**以下情况可以声明无依赖**：

1. **用户明确声明无依赖**：
   - 需求 JSON 中 `dependencies` 字段为空数组 `[]`
   - 需求描述中明确写明"无依赖"、"不涉及其他组件"

2. **扫描结果确认无依赖**：
   - 扫描项目文件后确认该功能为独立模块
   - 不存在引用或被引用关系
   - 所有描述中均不涉及其他组件信息

**无依赖时的输出格式**：
```json
{
  "expanded_dependencies": "本次更新为独立模块，不涉及对其他 pypto 内部组件的影响，无需进行跨模块适配。"
}
```

##### 获取方式

**Agent 必须执行以下步骤**（仅限项目文件扫描）：

1. **扫描项目目录结构**：
   - 使用 Glob 工具扫描 `**/*.py`、`**/*.h`、`**/*.cpp` 文件
   - 识别项目模块划分和组件边界

2. **分析需求涉及的模块**：
   - 根据需求标题和描述，定位相关的源文件
   - 使用 Grep 工具搜索相关类名、函数名、模块名

3. **分析依赖关系**：
   - 使用 Grep 工具搜索 `import`、`#include` 语句
   - 分析模块间的调用关系
   - 识别可能被影响的相关组件

4. **生成扩写内容**：
   - 若存在依赖：编写完整的依赖描述（>=200字符），说明受影响组件及其影响方式
   - 若无依赖：明确声明本次更新为独立模块，不涉及其他组件

##### 扫描示例

**需求**：Platform组件重构

**Agent 扫描过程**：
```
1. Glob **/platform/**/*.py  → 定位 Platform 相关文件
2. Grep "from platform"      → 查找引用 Platform 的模块
3. Grep "import.*Platform"   → 查找依赖 Platform 的组件
4. Read framework/platform/*.py → 分析接口定义
```

**扩写结果**：
```json
{
  "expanded_dependencies": "本次 Platform 组件重构可能影响以下 pypto 内部模块：1) TileCompiler 模块：依赖 Platform 提供的硬件参数查询接口，需适配新的抽象层接口；2) MemoryManager 模块：通过 Platform 获取内存分配策略，需更新内存池配置逻辑；3) PerformanceMonitor 模块：依赖 Platform 的性能计数器接口，需适配新的监控 API；4) OpExecutor 模块：通过 Platform 执行算子调度，需更新执行流程。所有受影响模块需进行接口适配测试，确保功能兼容性。"
}
```

##### 禁止事项

- ❌ 禁止通过网络搜索获取依赖信息
- ❌ 禁止添加外部第三方库依赖（如 torch_npu、ascendcl 等）
- ❌ 禁止添加与 pypto 项目无关的组件
- ❌ 禁止使用通用描述，必须具体到项目内的模块名

### 步骤 4：生成扩写 JSON 文件

Agent 生成完整的扩写 JSON 文件，结构如下：

```json
{
  "requirements": [
    {
      "title": "需求标题",
      "background": "原始背景",
      "expanded_background": "扩写后的背景（>=300字符）",
      "description": "原始描述",
      "expanded_description": "扩写后的描述（>=300字符）",
      "expanded_dependencies": "依赖信息扩写（>=200字符）",
      "sub_requirements": [
        {
          "title": "子需求标题",
          "logic": "原始逻辑",
          "expanded_logic": "扩展后的逻辑（>=100字符）",
          "description": "子模块描述（约150字符）",
          "flow_steps": ["初始化环境", "检查参数", "执行任务", "验证结果", "返回状态"]
        }
      ]
    }
  ]
}
```

**扩写 JSON 字段说明**：

| 字段 | 必需 | 最小长度 | 说明 |
|------|------|----------|------|
| expanded_background | 是 | 300字符 | 需求背景扩写 |
| expanded_description | 是 | 300字符 | 需求描述扩写 |
| expanded_dependencies | 是 | 50字符/200字符 | 特性依赖：本次更新可能影响的 pypto 内部组件。若存在依赖需>=200字符；若无依赖可简短声明（>=50字符） |
| sub_requirements[].description | 是 | 100字符 | 子模块功能描述 |
| sub_requirements[].flow_steps | 是 | 3-6个步骤 | 流程步骤（动词+名词格式） |

#### 4.1 内容完整性验证（强制）

**所有扩写字段禁止包含以下占位信息**：
- `【待补充】`
- `[待补充]`
- `待补充`
- `TODO`
- `FIXME`
- `请参考项目文档`
- `请参考相关技术资料`

**验证流程**：

1. **Agent 自检**：在生成扩写 JSON 文件前，必须检查所有字段是否包含占位信息
2. **脚本校验**：`generate_srs.py` 会自动检测占位信息，若发现则：
   - 打印错误信息，指明哪个字段包含占位信息
   - 终止文档生成
   - 提示 Agent 需要扩写的字段

**如果字段内容不足，Agent 必须**：
1. 通过搜索项目代码、文档、网络资料获取信息
2. 基于需求上下文推理生成合理内容
3. 参考类似模块的实现方式

**禁止**：
- 使用占位符文本
- 留空字段
- 使用不完整或无意义的描述

### 步骤 5：运行脚本生成 SRS 文档

```bash
python scripts/generate_srs.py <json_file> --expanded-json <expanded_json> --output-dir <output_dir>
```

参数说明：
- `json_file`：需求信息 JSON 文件路径（必需）
- `--expanded-json`：扩写内容 JSON 文件路径（必需）
- `--output-dir`：SRS 文档输出目录（可选，默认为当前工作目录）
- `--project-path`：项目路径（可选，默认当前目录）
- `--temp-dir`：临时文件输出目录（可选，默认为 skill 目录下的 temp）

**重要：迭代执行机制**

当 `generate_srs.py` 执行失败时，Agent 必须按照以下流程进行迭代处理：

```
迭代执行流程：
┌─────────────────────────────────────────────────────────────┐
│  1. 运行 generate_srs.py                                     │
│     ↓                                                        │
│  2. 检查执行结果                                              │
│     ├─ 成功 → 输出 .docx 文件 → 结束                         │
│     └─ 失败 → 进入步骤 3                                      │
│     ↓                                                        │
│  3. 分析错误信息                                              │
│     - 解析错误类型（扩写内容不足、字段缺失、验证失败等）       │
│     - 定位需要修改的具体字段                                  │
│     ↓                                                        │
│  4. 整改扩写 JSON 文件                                        │
│     - 根据错误提示修改对应的 expanded_*.json 文件             │
│     - 确保修改内容完整且不包含占位符                          │
│     ↓                                                        │
│  5. 重新运行 generate_srs.py                                 │
│     - 返回步骤 2，检查执行结果                                │
│     ↓                                                        │
│  6. 达到最大迭代次数（3 次）                                  │
│     - 第3次执行时无论是否存在校验错误，都强制输出 .docx 文件  │
│     - 同时打印所有校验失败的内容供 Agent 参考                 │
└─────────────────────────────────────────────────────────────┘
```

**错误类型及处理方式**：

| 错误类型 | 错误示例 | 处理方式 |
|---------|---------|---------|
| 扩写字段长度不足 | `description 长度不足100字符` | 扩充对应字段内容，增加具体细节 |
| 字段缺失 | `expanded_background 未提供` | 添加缺失字段，基于项目文档扩写 |
| 占位符检测 | `包含 '待补充' 占位符` | 移除占位符，填充实际内容 |
| 架构图生成失败 | `无法生成足够的功能名称` | 丰富 sub_requirements 的 description |
| 验证失败 | `字段包含中文函数名` | 修正为英文函数名或使用通用描述 |

**迭代计数规则**：
- 每次执行 `generate_srs.py` 计为一次迭代
- 成功输出 `.docx` 文件时重置计数
- 最大迭代次数：3 次
- 第3次执行时，无论是否存在校验失败都强制输出一版 .docx 文件
- 单次执行中若有多项校验失败，不提前退出，而是跑完所有校验逻辑后一次性打印全部错误

### 步骤 6：验证输出

1. 确认 Word 文档生成成功（检查 `.docx` 文件是否存在）
2. 检查文档内容完整性
3. 验证扩写 JSON 文件是否正确保存
4. 若步骤 5 执行失败，按照迭代执行机制重新处理

## Agent 执行规范

### 检测项目环境的具体步骤

```
步骤 1：检测当前目录
1. 获取当前工作目录名称（os.getcwd() 或 pwd）
2. 判断目录名称是否为 "pypto"
3. 若是，读取 README.md 文件前 500 字符
4. 判断 README.md 内容是否包含 "PyPTO" 或 "pypto" 关键词

步骤 2：选择信息获取方式
情况 A：目录名为 "pypto" 且 README.md 描述 pypto
  - 使用本地文件读取方式
  - Read README.md
  - Read CMakeLists.txt
  - Glob **/*.h 扫描头文件
  - Grep 搜索接口定义

情况 B：不满足上述条件
  - 使用 WebFetch 访问远程仓库
  - WebFetch https://gitcode.com/cann/pypto
  - WebFetch https://gitcode.com/cann/pypto/tree/main/framework
  - WebFetch https://gitcode.com/cann/pypto/blob/main/CMakeLists.txt
  - WebFetch https://gitcode.com/cann/pypto/blob/main/README.md
```

### 内容完整性验证步骤

```
步骤 1：生成扩写 JSON 文件后，Agent 必须自检
1. 检查 expanded_background 是否包含 "待补充"、"TODO" 等占位符
2. 检查 expanded_description 是否包含占位符
3. 检查 expanded_dependencies 是否包含占位符
4. 检查 expanded_dependencies 是否仅包含 pypto 内部组件（禁止外部依赖）
5. 检查每个 sub_requirements[].description 是否包含占位符

步骤 2：若发现占位符，必须进行扩写
1. 优先通过扫描项目代码获取相关实现细节
2. 若项目文件信息不足，再通过网络搜索获取技术背景和最佳实践
3. 基于需求上下文推理生成完整内容

步骤 3：确认无占位符后，保存 JSON 文件
```

### 迭代执行规范（重要）

**Agent 在执行 generate_srs.py 时必须遵循以下迭代规范**：

#### 迭代执行检查清单

每次运行 `generate_srs.py` 后，Agent 必须：

```
□ 检查命令返回值
  - 返回值 = 0：成功，检查 .docx 文件是否生成
  - 返回值 ≠ 0：失败，解析错误信息

□ 解析错误信息
  - 提取错误类型（字段长度不足、验证失败、生成失败等）
  - 识别需要修改的具体字段
  - 记录当前迭代次数

□ 执行整改
  - 使用 Edit 工具修改 expanded_*.json 文件
  - 确保修改内容满足最低长度要求
  - 移除所有占位符

□ 重新执行
  - 再次运行 generate_srs.py
  - 重复检查流程

□ 达到最大迭代次数处理
  - 迭代次数 = 3 次：强制输出 .docx 文件（即使存在校验错误）
  - 同时打印所有校验失败的内容供参考
```

#### 迭代执行伪代码

```python
max_iterations = 3
iteration_count = 0

while iteration_count < max_iterations:
    result = run_generate_srs()

    if result.success and docx_file_exists():
        print("SRS 文档生成成功")
        break

    iteration_count += 1
    error_info = parse_error(result.stderr)

    if iteration_count >= max_iterations:
        # 第3次执行：脚本已强制输出 .docx 文件
        print(f"第 {max_iterations} 次执行，已强制输出 .docx（可能包含校验问题）")
        break

    # 整改扩写 JSON
    fix_expanded_json(error_info)

# 成功退出
verify_docx_content()
```

#### 常见错误整改示例

**示例 1：description 字段长度不足**

```
错误信息：
  - 子模块 'xxx' 的 description 长度不足100字符，当前84字符

整改方式：
  1. 定位 expanded_requirement_xxx.json 中对应的 sub_requirements[].description
  2. 扩充内容，添加更多功能细节描述
  3. 确保扩充后长度 >= 100 字符
  4. 重新运行 generate_srs.py
```

**示例 2：架构图生成失败**

```
错误信息：
  - 无法为子模块 'xxx' 生成足够的(2个)有意义的功能名称

整改方式：
  1. 定位 expanded_requirement_xxx.json 中对应的 sub_requirements[].description
  2. 添加更详细的功能描述，如"该模块包含参数加载、配置验证、错误处理等功能"
  3. 确保描述中包含可提取的功能关键词
  4. 重新运行 generate_srs.py
```

**示例 3：占位符检测**

```
错误信息：
  - 字段包含 '待补充' 占位符

整改方式：
  1. 定位包含占位符的字段
  2. 移除占位符，填充基于项目文档的实际内容
  3. 重新运行 generate_srs.py
```

### 语言组织校验（新增）

脚本会自动检测并修复因删除模块导致的语句不通问题：

#### 检测的问题类型

| 问题类型 | 示例 | 处理方式 |
|---------|------|---------|
| 连续标点符号 | "模块A、，模块B" | 替换为正确的标点 |
| 空括号 | "模块（）" | 删除空括号 |
| 编号后无内容 | "1) , 2)" | 删除无效编号 |
| 连续顿号/逗号 | "、，、" | 合并为单个标点 |
| 残留短语 | "调用 接口" | 清理无效描述 |

#### 自动修复流程

```
1. 检测不合理的标点模式（连续标点、空括号等）
2. 删除因模块删除后残留的不合理短语
3. 修复编号后无内容的情况
4. 清理多余空格
5. 验证修复后的文本是否合理
6. 如仍有问题，进行句子级别的清理
```

### 架构图生成优化（新增）

#### 三层结构规范

架构图采用严格的三层结构：

| 层级 | 内容 | 数量限制 | 命名要求 |
|------|------|----------|----------|
| 第一层 | 特性概括名称 | 1个 | 从需求标题概括，如"reshape替换pass" |
| 第二层 | 核心功能子模块 | 2~5个 | 使用子需求的title字段 |
| 第三层 | 子模块涵盖的功能 | 每个子模块2~5个 | 必须有实际含义，禁止"功能模块x" |

#### 特性标题概括规则

系统会自动将需求标题概括为简短的特性名称：

| 原始标题 | 概括结果 |
|---------|---------|
| "增加一个将reshape op替换为view op的pass" | "reshape替换pass" |
| "增加一个根据platform型号切换cv比例的pass" | "cv比例切换pass" |
| "开发一个数据处理模块" | "数据处理模块" |

#### 第三层功能名称生成规则

**重要**：第三层功能名称必须具有实际含义，禁止使用以下形式：
- ❌ "功能模块1"、"功能模块2"
- ❌ "功能1"、"功能2"
- ❌ "模块1"、"组件1"

**生成策略**（按优先级）：
1. 根据子模块标题关键词匹配预设功能名称（如"识别"→"特征提取"、"条件匹配"）
2. 从子模块logic字段提取名词+模块模式
3. 从子模块description字段提取功能描述

**若生成失败**，脚本将抛出错误并提示：
```
错误：无法为子模块 'xxx' 生成足够的(2个)有意义的功能名称。
请在扩写JSON的sub_requirements中为该子模块提供更详细的description字段。
建议格式：描述该模块包含的具体功能，如'负责参数加载、配置验证、错误处理等功能'。
```

#### 放宽字符限制

- 主标题：从 15 字符增加到 20 字符
- 子模块标题：从 12 字符增加到 18 字符
- 子组件标签：从 12 字符增加到 15 字符

### 流程图合理性校验（新增）

#### 检测的不合理结构

| 问题类型 | 说明 | 严重程度 |
|---------|------|---------|
| 独立错误处理块 | 错误处理节点未与任何决策节点连接 | error |
| 空流程图 | 只有开始和结束，无中间步骤 | warning |
| 单步骤流程 | 只有一个处理步骤 | warning |
| 孤立节点 | 存在未连接的处理节点 | error |

#### 合理性要求

- 最少步骤数：3 个
- 最多步骤数：8 个
- 包含分支关键词时应有决策节点
- 错误处理块必须连接到决策节点

#### 自动修复流程

```
1. 生成初步流程图
2. 执行合理性校验
3. 若校验失败：
   a. 第一次尝试：确保错误处理块正确连接
   b. 第二次尝试：生成简化线性流程图
   c. 第三次尝试：生成最简化版本
4. 输出校验通过的流程图
```

### 扩写内容获取优先级

```
优先级 1：项目文件扫描（优先）
- Read README.md, docs/*.md
- Grep 搜索关键词
- 从代码和注释提取信息

优先级 2：网络搜索（备选）
- 仅当项目文件信息不足时使用
- 搜索技术文档、最佳实践
- 了解行业标准实现方案

注意：expanded_dependencies 只能通过项目文件扫描获取
```

### 特性依赖扫描步骤

```
步骤 1：定位需求相关模块
1. 根据需求标题，使用 Glob 扫描相关目录
   - 例：Glob framework/platform/**/*.py
2. 识别需求涉及的主要源文件

步骤 2：分析模块依赖关系
1. 使用 Grep 搜索引用关系
   - Grep "from platform" framework/
   - Grep "import.*Platform" framework/
2. 识别依赖该模块的上游组件

步骤 3：生成特性依赖描述
1. 列出所有受影响的模块名称和文件路径
2. 说明每个模块的作用
3. 描述本次更新对模块的影响方式
4. 建议适配或测试方案
```

### 获取项目信息的具体步骤（远程仓库）

```
1. WebFetch https://gitcode.com/cann/pypto
   - 获取项目概览信息
   - 提取主要模块列表

2. WebFetch https://gitcode.com/cann/pypto/tree/main/framework
   - 分析框架目录结构
   - 识别核心组件

3. WebFetch https://gitcode.com/cann/pypto/blob/main/CMakeLists.txt
   - 提取编译依赖
   - 分析组件关系

4. WebFetch https://gitcode.com/cann/pypto/blob/main/README.md
   - 了解项目背景
   - 提取技术特性
```

### 特性依赖扩写示例

**原始输入**：
```json
{
  "title": "Platform组件重构",
  "description": "重构Platform组件，实现硬件抽象层设计..."
}
```

**Agent 扫描过程（仅项目文件）**：
```
1. Glob framework/platform/**/*.py
   → 发现 platform/device_manager.py, platform/hardware_abstraction.py

2. Grep "from platform" framework/
   → 发现 TileCompiler、MemoryManager、OpExecutor 等模块引用 platform

3. Grep "import.*DeviceManager" framework/
   → 发现 PerformanceMonitor 模块依赖 DeviceManager

4. Read framework/platform/interface.py
   → 分析现有接口定义，确定影响范围
```

**扩写结果**：
```json
{
  "expanded_dependencies": "本次 Platform 组件重构可能影响以下 pypto 内部模块：1) TileCompiler 模块（framework/compiler/tile_compiler.py）：依赖 Platform 的 GetHardwareParam 接口获取硬件参数，需适配新的抽象层接口；2) MemoryManager 模块（framework/memory/manager.py）：通过 Platform 的 AllocateMemory 接口分配设备内存，需更新内存池配置逻辑；3) OpExecutor 模块（framework/executor/op_executor.py）：调用 Platform 的 ExecuteKernel 接口执行算子，需适配新的执行流程；4) PerformanceMonitor 模块（framework/monitor/perf_monitor.py）：依赖 Platform 的 GetPerformanceCounter 接口采集性能数据，需适配新的监控 API。建议对所有受影响模块进行回归测试。"
}
```

## 输出文档结构

生成的 SRS 文档包含以下章节：

### 1. 文档命名

- 文件名：需求列表第一个需求的标题（特殊字符替换为下划线）
- 若标题为空，使用 `SRS_文档.docx`
- 若需求列表为空，报错终止

### 2. 需求章节结构

每个独立需求作为二级标题，内部包含以下三级标题：

#### 2.1 功能需求
- 背景信息介绍
- 需求描述
- 需求与总体架构的关系图（PNG 格式，宽度固定15cm）

#### 2.2 模块划分
- 子需求的简单描述
- 子需求的实现逻辑描述
- 抽象流程图（PNG 格式，高度固定15cm）

#### 2.3 特性依赖

**特性依赖**描述本次更新可能影响到的 pypto 项目内部组件，包括：
- 说明受影响的模块名称及其在项目中的位置
- 描述每个受影响模块的作用
- 说明本次更新对模块的具体影响方式
- 建议的适配或测试方案

内容从 `expanded_dependencies` 字段读取。

#### 2.4 软件接口

以表格形式展示 C++ 新增接口，每个接口对应一个独立的转置表格：

| 第一列 | 第二列 |
|--------|--------|
| 函数原型 | 以C++函数声明的形式提供的函数声明 |
| 函数功能 | 功能描述 |
| 增加输入 | 函数中以常量或值传递的入参，说明变量类型与含义；无时填 NA |
| 函数输出 | 函数中以引用传递的入参，说明变量类型与含义；无时填 NA |
| 返回 | 函数返回值的数据类型；bool 等需逐项说明含义；void 函数填 NA |
| 使用说明 | 使用场景说明；不存在时填 NA |
| 注意事项 | 特例说明；不存在时填 NA |

**表格格式**：全边、左对齐、第一列加粗

### 3. 字体格式

| 元素 | 中文字体 | 英文字体 | 字号 | 字体颜色 |
|------|----------|----------|------|------|
| 二级/三级标题 | 黑体 | Times New Roman | 12磅 | 黑色 |
| 正文 | 宋体 | Times New Roman | 10.5磅 | 黑色 |

## Mermaid 图表规范

### 架构图
- 固定宽度：15cm
- 组件命名：名词或名词+动词形式（如"接口定义"、"资源分配"）

### 流程图
- 固定高度：15cm
- 步骤命名：动词+名词形式（如"初始化环境"、"检查参数"）
- 分支逻辑：每个分支节点最多一条成功边和一条失败边

### 组件命名映射表

| 子模块关键词 | 推荐组件名称 |
|-------------|-------------|
| 硬件抽象层 | 接口定义、适配器管理 |
| 设备管理 | 资源分配、状态监控 |
| 配置管理 | 参数加载、配置验证 |
| 性能监控 | 数据采集、指标分析 |
| 初始化 | 环境检查、资源准备 |
| 执行 | 任务调度、结果处理 |
| 验证 | 规则检查、结果输出 |
| 处理 | 数据预处理、核心计算 |

## 依赖安装

```bash
# 安装 python-docx
pip install python-docx

# 安装 mermaid-cli（用于图表转换）
npm install -g @mermaid-js/mermaid-cli
```

## 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| JSON 文件不存在 | 提示用户提供正确的文件路径 |
| JSON 格式错误 | 提示用户检查 JSON 格式，显示具体错误位置 |
| 必需字段缺失 | 提示缺少的字段名称 |
| 需求列表为空 | 报错并终止 |
| 项目扫描失败 | 提示用户检查项目路径权限 |
| 输出目录不可写 | 提示检查目录权限 |
| mermaid 转换失败 | 报错并终止，提示安装 mermaid-cli |
| python-docx 未安装 | 提示安装：`pip install python-docx` |
| mermaid-cli 未安装 | 提示安装：`npm install -g @mermaid-js/mermaid-cli` |
| 网络连接失败 | 提示检查网络或使用本地项目路径 |
| **扩写内容包含占位符** | 报错并终止，提示 Agent 必须扩写该字段 |
| **扩写内容长度不足** | 报错并终止，提示 Agent 补充扩写内容 |

## 使用示例

```bash
# 基本用法
python scripts/generate_srs.py requirements.json --expanded-json expanded_content.json

# 指定输出路径
python scripts/generate_srs.py requirements.json ./output/SRS.docx --expanded-json expanded_content.json

# 使用本地项目路径
python scripts/generate_srs.py requirements.json --project-path /path/to/project --expanded-json expanded_content.json
```

## 注意事项

1. **扩写 JSON 必须由 Agent 生成**：不使用通用模板，必须根据具体需求定制
2. **扩写内容优先从项目文件获取**：优先扫描项目文件，信息不足时再使用网络搜索
3. **特性依赖只能从项目文件获取**：`expanded_dependencies` 只能通过扫描 pypto 项目文件获取，禁止通过网络搜索
4. **特性依赖仅限项目内部组件**：只描述本次更新可能影响到的 pypto 内部模块，不涉及外部第三方库
5. **无依赖可明确声明**：若确认无依赖，可简短声明为独立模块
6. **流程步骤格式要求**：必须是"动词+名词"形式
7. **图片尺寸固定**：架构图宽度15cm，流程图高度15cm
8. **内容分离原则**：扩写内容保存在独立 JSON 文件，便于维护和修改
