# PyPTO 算子开发总指南

> 本文档将 Skills（专家技能）、Agents（协作代理）、知识库文档和代码仓结构整合为一份统一的算子开发流程指南。

---

## 一、整体架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│                     用户 / 开发者                                 │
│     （自然语言描述需求 / 数学公式 / 方案文档 / 论文链接）          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  AGENTS.md — 项目规范（自动加载）                  │
│     定义通用原则、环境配置、开发规范、默认值                       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│  Skills 专家技能  │    │  Agents 协作代理                      │
│  （按需加载）      │    │  .opencode/agents/                   │
│  .agents/skills/  │    │  ┌─ Orchestrator (编排)               │
│                   │    │  ├─ Analyst (Golden/Design)          │
│                   │    │  ├─ Developer (实现/精度修复)         │
│                   │    │  ├─ PerfTuner (性能调优)             │
│                   │    │  └─ CodeMergeAgent (PR)              │
└──────────────────┘    └──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 知识库（基础知识文档）                              │
│     .agents/skills/Skills/ 下的准则文档                           │
│     开发流程.md / 交付件.md / operation开发准则.md /              │
│     TileOP开发准则.md / codegen_vector开发准则.md 等              │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 PyPTO 代码仓（pypto/）                            │
│     framework/    — C++ Operation/TileOp/Codegen 核心            │
│     python/pypto/ — Python 前端 API                              │
│     docs/         — API 文档与教程                                │
│     examples/     — 分级示例代码                                  │
│     tools/        — 验证/性能分析/tiling 工具                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、算子开发全流程（7 阶段状态机）

### 流程总图

```
Stage 1          Stage 2          Stage 3          Stage 4
需求理解    →    API 探索    →    Golden 生成  →   Design 设计
(Skill)          (Skill)          (Subagent)        (Subagent)
                                                       │
                                                       ▼
                               Stage 6          Stage 5
                               精度修复    ←    代码实现
                               (Subagent)        (Subagent)  ──→ [PRECISION_PASS]
                                                       │
                                                       ▼
                                                  Stage 7
                                                  性能调优
                                                  (Subagent)
```

### 阶段详解

| Stage | 名称 | 执行方式 | 负责方 | 核心输入 | 核心输出 |
|-------|------|----------|--------|----------|----------|
| 1 | 需求理解 | Skill 调用 | `pypto-intent-understanding` | 用户自然语言描述 | `spec.md` |
| 2 | API 探索 | Skill 调用 | `pypto-api-explorer` | `spec.md` | `api_report.md` |
| 3 | Golden 生成 | Subagent | `pypto-op-analyst` | `spec.md` | `{op}_golden.py` |
| 4 | Design 设计 | Subagent | `pypto-op-analyst` | spec + api_report + golden | `design.md` |
| 5 | 代码实现 | Subagent | `pypto-op-developer` | design + golden | `{op}_impl.py` + `test_{op}.py` |
| 6 | 精度修复 | Subagent | `pypto-op-developer` | impl + golden + 失败信息 | 修复后的 `{op}_impl.py` |
| 7 | 性能调优 | Subagent | `pypto-op-perftuner` | 精度通过的 impl | 优化后的 `{op}_impl.py` |

### 标准工件目录

```
custom/{op}/
├── spec.md                    # Stage 1 产出：结构化需求
├── api_report.md              # Stage 2 产出：API 可行性报告
├── {op}_golden.py             # Stage 3 产出：PyTorch 参考实现
├── design.md                  # Stage 4 产出：实现设计方案
├── {op}_impl.py               # Stage 5/6/7 产出：PyPTO kernel 实现
├── test_{op}.py               # Stage 5 产出：测试入口（三态标记）
├── README.md                  # Stage 5 产出：算子文档
├── .orchestrator_state.json   # Orchestrator 状态文件
└── history_version/           # Stage 6/7 备份目录
```

---

## 三、两类使用模式

### 模式 A：编排模式（Orchestrator Agent）

**适用场景**：完整算子开发，从需求到性能调优全流程

**使用方式**：
1. 按 `Tab` 键切换到 `pypto-op-orchestrator` 代理
2. 或使用 `/pypto-op-workflow` 斜杠命令

**特点**：
- Orchestrator 自动管理 7 阶段状态机
- 自动调度 Subagent（Analyst/Developer/PerfTuner）
- 支持中断恢复、失败重试、状态持久化
- 工件门禁验证，不跳阶段

### 模式 B：单步 Skill 模式

**适用场景**：只需某个阶段的独立任务

| 需求 | 调用的 Skill |
|------|-------------|
| 只需生成 Golden | `pypto-golden-generator` |
| 只需查 API 可行性 | `pypto-api-explorer` |
| 只需设计方案 | `pypto-op-design` |
| 只需写实现代码 | `pypto-op-develop` |
| 精度问题排查 | `pypto-precision-debugger` |
| AICore 错误定位 | `pypto-aicore-error-locator` |
| 性能分析 | `pypto-op-perf-analyzer` |
| 性能调优 | `pypto-op-perf-autotuner` |

---

## 四、Skill 体系详解

### 4.1 算子开发与编排

| Skill | 触发场景 | 输入 | 输出 | 关键行为 |
|-------|---------|------|------|---------|
| `pypto-op-workflow` | 接到算子开发任务 | 自然语言需求 | 全套工件 | 无状态串联各阶段 Skill |
| `pypto-intent-understanding` | Stage 1 | 自然语言需求 | `spec.md` | 结构化需求提取，含数据流图 |
| `pypto-api-explorer` | Stage 2 | `spec.md` | `api_report.md` | 搜索 `docs/` API 文档，验证可行性 |
| `pypto-golden-generator` | Stage 3 | `spec.md` | `{op}_golden.py` | 生成纯 PyTorch 参考实现 |
| `pypto-op-design` | Stage 4 | spec + api_report + golden | `design.md` | API 选型、Tiling 策略、Loop 结构 |
| `pypto-op-develop` | Stage 5 | design + golden | impl + test + README | 生成 PyPTO 实现，执行首跑判定 |

### 4.2 精度验证与调试

| Skill | 触发场景 | 核心方法 |
|-------|---------|---------|
| `pypto-precision-debugger` | Stage 6 精度失败 | 基础检查 → 内存排查 → 特性排除 → 二分定位 |
| `pypto-binary-search-verify` | Verify 模式精度定位 | `pass_verify_save` 循环条件保存 |
| `pypto-binary-search-without-verify` | Checkpoint 模式精度定位 | 检查点 tensor + `pypto.assemble` |
| `pypto-aicore-error-locator` | AICore error | 启用追踪日志 → 重编译 → trace 分析 → 二分定位 |

### 4.3 性能分析

| Skill | 触发场景 | 核心指标/手段 |
|-------|---------|-------------|
| `pypto-op-perf-analyzer` | Stage 7 分析阶段 | 核心利用率、气泡率、AicoreTime |
| `pypto-op-perf-autotuner` | Stage 7 调优阶段 | Stitch/loop_unroll/Tilesize/L2 亲和 |

### 4.4 环境与工具

| Skill | 用途 |
|-------|------|
| `pypto-environment-setup` | PyPTO 环境安装、import 修复、NPU 检测 |
| `gitcode-mcp-install` | GitCode MCP Server 配置 |

### 4.5 PR 与代码质量

| Skill | 用途 |
|-------|------|
| `pypto-pr-creator` | 创建规范 PR |
| `pypto-pr-fixer` | 修复 CI 失败和 review 意见 |
| `pypto-issue-creator` | 创建 GitCode Issue |
| `pypto-fracture-point-detector` | 识别框架/文档断裂点 |
| `pypto-skill-reviewer` | Skill 质量评审（48 条规则） |

---

## 五、基础知识库（Skills 知识文档）

位于 `.agents/skills/Skills/` 下，为 Skill 和 Agent 提供领域知识支撑：

| 知识文档 | 核心内容 | 使用方式 |
|---------|---------|---------|
| `开发流程.md` | PyPTO 全链路调用流程（Python → Operation → Tile → Codegen → TileOp → PTO） | Stage 2/4/5 参考，理解数据流 |
| `交付件.md` | 各层交付件清单（Operation/TileOp/Codegen）与文件位置 | Stage 4/5 参考，确定需交付的文件 |
| `交付件调用关联关系.md` | 各层调用链路图与代码级关联 | 问题定位时参考调用链 |
| `operation开发准则.md` | Operation 层开发规范、命名规范、模板代码 | Stage 5 编写 Operation 时参考 |
| `TileOP开发准则.md` | TileOp 层三层实现模式（ComputeImpl → Compute → 用户接口） | Stage 5 编写 TileOp 时参考 |
| `codegen_vector开发准则.md` | Codegen 层代码生成规范 | Stage 5 编写 Codegen 时参考 |
| `permute_operation_summary.md` | Permute 算子实现总结 | 特定算子参考 |
| `permute_operation_design.md` | Permute 算子设计文档 | 特定算子参考 |
| `ST_Test_Guide.md` | ST 测试指南 | 测试阶段参考 |

---

## 六、代码仓资源索引

### 6.1 核心代码目录

| 目录 | 内容 | 开发阶段 |
|------|------|---------|
| `framework/src/interface/operation/` | Operation 接口层（Opcode/Shape/Tiling） | Stage 5 |
| `framework/src/interface/tileop/` | TileOp 执行层（PTO 指令调用） | Stage 5 |
| `framework/src/codegen/cloudnpu/` | Codegen 代码生成层 | Stage 5 |
| `framework/src/interface/function/` | Function 管理层 | 理解框架 |
| `framework/src/interface/tensor/` | Tensor 定义层 | 理解框架 |
| `python/pypto/` | Python 前端 API | Stage 5（Python 封装） |
| `python/pypto/op/` | Python 算子接口定义 | Stage 5 |

### 6.2 文档与示例

| 目录 | 内容 | 用途 |
|------|------|------|
| `docs/api/` | PyPTO API 文档（element/tensor/operation/datatype 等） | Stage 2 API 查询 |
| `docs/tutorials/` | 教程（开发/编译/tiling/tensor 操作等） | 学习参考 |
| `docs/install/` | 环境安装指南 | 环境配置 |
| `examples/00_hello_world/` | 入门示例 | 快速入门 |
| `examples/01_beginner/` | 初级示例（basic/compute/tiling/transform） | 基础学习 |
| `examples/02_intermediate/` | 中级示例（basic_nn/controlflow/operators） | 进阶学习 |
| `examples/03_advanced/` | 高级示例（aclgraph/advanced_nn/patterns） | 高级场景 |

### 6.3 工具

| 目录/文件 | 用途 |
|----------|------|
| `tools/verifier/` | 精度验证（tensor_diff/pass_compare/float_diff） |
| `tools/profiling/` | 性能分析（swim_lane/pm_to_csv/pipe_time_trace） |
| `tools/scripts/tiling_tool.py` | Tiling 配置工具 |
| `tools/scripts/run_operation_test_with_config.py` | ST 测试执行 |

---

## 七、核心原则速查

### 7.1 通用原则（来自 AGENTS.md）

1. **如实报告，禁止伪完成** — 未验证不标记完成
2. **先验证，再下结论** — 基于证据判断
3. **区分事实、推断与建议** — 禁止编造
4. **最小必要改动** — 复用现有实现

### 7.2 算子开发原则

1. **理解原理后实现** — 先确认需求/API 映射再编码
2. **三文件分离** — golden（参考）、impl（实现）、test（测试）各司其职
3. **遇问题先定位** — 搜索 `docs/` → 查阅 `examples/` → 定位修复
4. **禁止简化代码** — 不因"能跑"就降级优化
5. **优先真实 NPU** — 有 NPU 时禁止 sim 模式

---

## 八、快速开始

### 场景 1：从零开发一个新算子

```
1. 在 pypto/ 目录启动 OpenCode
2. 切换到 pypto-op-orchestrator Agent（按 Tab）
3. 描述需求，例如：
   "开发一个 sinh 算子，数学公式是 (e^x - e^(-x)) / 2，
    输入是 shape 为 [b, s, n, d] 的 float32 tensor"
4. Orchestrator 自动编排 7 阶段流程
```

### 场景 2：只做精度调试

```
1. 直接在对话中调用：
   "请使用 pypto-precision-debugger 排查 custom/sinh/sinh_impl.py 的精度问题"
2. 或使用斜杠命令：/pypto-precision-debugger
```

### 场景 3：提交 PR

```
1. 切换到 pypto-code-merge-agent
2. 输入 "go" 或 "创建PR"
3. Agent 自动完成：变更检测 → 方案生成 → 确认 → Issue + PR 创建
```

---

## 九、知识查找优先级

当遇到问题时，按以下顺序查找资料：

```
1. docs/api/           — PyPTO API 文档（本地）
       ↓ 找不到
2. examples/           — 分级示例代码
       ↓ 找不到
3. framework/src/      — 现有实现代码
       ↓ 找不到
4. .agents/skills/Skills/ — 知识库准则文档
       ↓ 仍无法解决
5. ascendc-docs-search Skill — 在线搜索（华为昇腾社区）
```

---

## 十、总结：Skill 与流程对应关系

```
                    ┌─────────────────────────────────┐
                    │  ascendc-kernel-develop-workflow │  ← 外部 Skill（Ascend C）
                    │  ascendc-docs-search             │  ← 外部 Skill（文档搜索）
                    └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    pypto-op-orchestrator (编排 Agent)            │
│                                                                  │
│  Stage 1: pypto-intent-understanding ─── 知识: 开发流程.md       │
│  Stage 2: pypto-api-explorer        ─── 知识: 交付件.md          │
│  Stage 3: pypto-golden-generator    ─── (Subagent: analyst)      │
│  Stage 4: pypto-op-design           ─── 知识: 交付件调用关联关系  │
│  Stage 5: pypto-op-develop          ─── 知识: operation开发准则   │
│                                      ─── 知识: TileOP开发准则     │
│                                      ─── 知识: codegen_vector准则  │
│  Stage 6: pypto-precision-debugger  ─── (Subagent: developer)    │
│           pypto-binary-search-*                                       │
│           pypto-aicore-error-locator                                  │
│  Stage 7: pypto-op-perf-analyzer    ─── (Subagent: perftuner)     │
│           pypto-op-perf-autotuner                                     │
│                                                                  │
│  辅助:   pypto-environment-setup                                   │
│          pypto-pr-creator / pypto-pr-fixer / pypto-issue-creator    │
│          pypto-fracture-point-detector / pypto-skill-reviewer       │
│          pypto-code-merge-agent                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键文件速查表

| 用途 | 路径 |
|------|------|
| 项目规范 | `AGENTS.md` |
| Agent 定义 | `.opencode/agents/*.md` |
| Skill 定义 | `.agents/skills/*/SKILL.md` |
| 知识库 | `.agents/skills/Skills/*.md` |
| Orchestrator 状态 | `custom/{op}/.orchestrator_state.json` |
| API 文档 | `docs/api/` |
| 开发教程 | `docs/tutorials/development/` |
| Opcode 定义 | `framework/src/interface/operation/opcode.h` |
| Operation 模板 | `framework/src/interface/operation/vector/` |
| TileOp 模板 | `framework/src/interface/tileop/vector/` |
| Codegen 模板 | `framework/src/codegen/cloudnpu/` |
| Python API | `python/pypto/op/` |
| 入门示例 | `examples/00_hello_world/` |
| ST 测试用例 | `framework/tests/st/operation/test_case/*.csv` |
| ST 执行脚本 | `tools/scripts/run_operation_test_with_config.py` |
