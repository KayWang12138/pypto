# Pass UT 生成工具使用指南

## 概述

本工具用于为 PyPTO Pass 模块生成单元测试用例（UT），支持：
- 在线处理 PR，获取 diff 和覆盖率报告
- 离线分析 diff 文件和覆盖率报告
- 生成 UT 设计建议

## 工具目录结构

```
pypto-pass-ut-generate/
├── SKILL.md                  # 主技能文档
├── scripts/
│   ├── pr_utils.py           # PR 处理工具
│   └── ut_coverage.py        # 覆盖率分析工具
└── references/
    ├── checklist.md          # 检查清单
    └── usage.md             # 本文档
```

## 使用方式

### 方式一：在线处理 PR

```bash
# 处理 PR
python scripts/pr_utils.py 1894

# 指定输出目录
python scripts/pr_utils.py 1894 -o /tmp/output

# 输出 JSON 格式
python scripts/pr_utils.py 1894 --json
```

### 方式二：离线分析 Diff

```bash
# 分析本地 diff 文件
python scripts/pr_utils.py --diff /path/to/diff.file

# 分析并输出 JSON
python scripts/pr_utils.py --diff /path/to/diff.file --json
```

### 方式三：覆盖率分析

```bash
# 解析本地覆盖率报告
python scripts/ut_coverage.py --report /path/to/coverage.html

# 从 URL 下载并解析覆盖率报告
python scripts/ut_coverage.py --report https://example.com/ut_cov.tar.gz

# 同时分析 diff 和覆盖率
python scripts/ut_coverage.py --diff diff.file --report report.html
```

## 工具输出

### pr_utils.py 输出

| 字段 | 说明 |
|------|------|
| `pr_info` | PR 基本信息（编号、仓库、作者等） |
| `diff_content` | PR 的代码变更内容 |
| `analysis` | 变更文件分析（Pass 文件、测试文件等） |
| `ut_report` | UT-REPORT 状态（通过/失败/中止） |
| `diff_applied` | Diff 是否成功应用到本地仓库 |
| `build_status` | 编译状态 |
| `need_design_ut` | 是否需要设计 UT |

### ut_coverage.py 输出

| 字段 | 说明 |
|------|------|
| `diff_info` | Diff 变更分析结果 |
| `coverage_info` | 覆盖率信息（总体覆盖率、文件列表等） |
| `low_coverage_files` | 低覆盖率文件列表 |
| `ut_design_suggestions` | UT 设计建议列表 |

## 典型工作流

### 场景 1：为新 Pass 生成 UT

1. 分析 Pass 代码，识别关键功能
2. 在 `pypto/framework/tests/ut/passes/src/` 创建测试文件
3. 参考 `test_removeredundantop.cpp` 编写测试用例
4. 运行 `python3 build_ci.py -c -u=TestPassName.* -j=24 -f=cpp`
5. 检查覆盖率

### 场景 2：分析 PR 的覆盖率

```bash
# 获取 PR diff 并分析
python scripts/pr_utils.py 1894

# 检查 UT-REPORT 状态和覆盖率
# 根据输出判断是否需要补充 UT
```

### 场景 3：离线分析

```bash
# 下载 PR 的 diff 和覆盖率报告
# 假设已有 diff.file 和 coverage.html

# 分析覆盖率
python scripts/ut_coverage.py --diff diff.file --report coverage.html

# 输出 JSON 便于程序处理
python scripts/ut_coverage.py --diff diff.file --report coverage.html --json
```

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `GITCODE_TOKEN` | GitCode API Token | 是 |
| `TILE_FWK_DEVICE_ID` | NPU 设备 ID | 运行 UT 时 |
| `PTO_TILE_LIB_CODE_PATH` | PTO 库路径 | 运行 UT 时 |

## 常见错误

### 1. API 请求失败

```
⚠️ API 请求失败: [Errno -2] Name or service not known
```

解决方案：检查网络连接，确保可以访问 gitcode.com

### 2. Diff 应用冲突

```
⚠️ Diff 应用发生冲突
```

解决方案：选择保留本地版本或接受 incoming 版本

### 3. 覆盖率报告下载失败

```
下载失败: HTTP 404 - Not Found
```

解决方案：检查 URL 是否正确，报告是否已过期

---

## 相关文档

- [SKILL.md](../SKILL.md) - 主技能文档
- [checklist.md](./checklist.md) - 检查清单