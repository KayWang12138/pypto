# AGENTS.md

## Codex 协作说明

- 保留现有仓内 agent 配置与约定，不删除、不重载其它系统的专用配置文件
- 将 Codex 视为当前仓的并行开发代理，遵循本文件作为 Codex 在本仓的补充事实源
- 如不同说明存在重叠，优先采用与当前仓真实构建、测试和目录结构一致的约束

## 项目概述

本项目用于开发华为昇腾 AI 处理器（CANN PyPTO）自定义算子，支持完整的开发、测试及性能调优流程。

### 核心功能

- 使用 PyPTO 编程语言开发昇腾 AI 处理器自定义算子
- 提供完整的开发、构建、测试及性能调优工作流支持
- 遵循官方开发规范和性能优化最佳实践

---

## Skills 索引

#### 算子开发与编排
- `pypto-op-workflow`：无状态的全流程 Skill 入口，用于手动串联算子开发阶段
- `pypto-intent-understanding`：将自然语言算子需求转化为结构化规格
- `pypto-api-explorer`：探索 API 映射、约束条件与实现可行性
- `pypto-golden-generator`：生成用于精度对比的 golden 参考实现
- `pypto-op-design`：生成算子设计方案，明确数据切分、tiling 与 loop 结构
- `pypto-op-develop`：Stage 5 实现阶段 Skill，生成实现、测试入口与 README

#### 精度验证与调试
- `pypto-precision-debugger`：定位并修复精度问题
- `pypto-precision-compare`：精度对比与定位，支持文件保存和二分对比两种方法
- `pypto-aicore-error-locator`：定位 aicore error 的问题文件和代码行

#### 性能分析
- `pypto-operator-auto-tuner`：分析性能数据、定位瓶颈并给出优化依据，基于实测性能数据迭代调优，并验证精度与性能收益

#### 环境与工具
- `pypto-environment-setup`：PyPTO 环境安装与环境问题修复
- `gitcode-mcp-install`：安装和配置 GitCode MCP Server

#### Pass 分析与优化
- `pypto-pass-error-fixer`：Pass 模块错误诊断与修复，提供从问题定位到修复验证的完整工作流程
- `pypto-pass-module-analyzer`：Pass 模块代码分析，生成模块分析文档，帮助理解接口、功能与特殊场景
- `pypto-pass-perf-optimizer`：Pass 编译性能优化，分析和优化 Pass 模块的编译性能
- `pypto-pass-ut-generate`：根据 Pass 业务描述，生成单元测试用例（UT）
- `pypto-pass-workflow-analyzer`：Pass 业务流分析，帮助理解业务执行流程、模块职责与数据流转

#### PR 与代码质量
- `pypto-pr-creator`：准备并创建符合规范的 PR
- `pypto-pr-fixer`：修复 PR 的 CI 失败与 review 意见
- `pypto-issue-creator`：基于上下文创建 GitCode Issue
- `pypto-fracture-point-detector`：识别 PyPTO 框架或文档断裂点
- `pypto-skill-reviewer`：评审 skill 目录的质量与规范符合性

---

## 开发环境

- 首选 conda 环境：`pto312_w00613560`
- 若当前 shell 未激活该环境，执行 Python、pytest、ruff、pyright、pip 等命令时优先使用：
  - `conda run -n pto312_w00613560 ...`
- 涉及 NPU 运行时验证时，先执行：
  - `source /data/w00613560/setup.sh`
- 当设备选择或设备状态影响结论时，先执行：
  - `npu-smi info`

## 构建与安装

当前仓使用 `setuptools.build_meta + CMake`，不是源仓的 `scikit-build-core`。

常用安装方式：

```bash
conda run -n pto312_w00613560 pip install -e .
conda run -n pto312_w00613560 pip install -e . --no-build-isolation
```

补充事实：

- Python 包根在 `python/`
- 顶级包同时包含 `pypto` 和 `pypto_block`
- 当前核心扩展产物包括：
  - `python/pypto/pypto_impl*.so`
  - `python/pypto_block/pypto_core*.so`
- 第三方依赖通过 `PYPTO_THIRD_PARTY_PATH` 和 `cmake/third_party/*` 规则解析

## 测试与验证

常用命令：

```bash
conda run -n pto312_w00613560 pytest python/tests/ -v
conda run -n pto312_w00613560 pytest python/tests/ut/block -v
conda run -n pto312_w00613560 pytest python/tests/ut/block/ir/printing/test_python_printer.py -v
```

测试分层：

- 单元测试位于 `python/tests/ut/`
- 系统测试和设备相关测试位于 `python/tests/st/`
- 即便位于 `python/tests/ut/`，只要测试中出现 `torch_npu`、`torch.npu.set_device(...)`、`npu:*` 张量、`fe.launch(...)` 或编译后 NPU kernel 执行，也按 NPU 板侧测试处理

验证要求：

- 代码修改后，至少运行与改动直接相关的测试，再汇报完成
- 涉及 codegen、PTOAS IR、NPU 执行链路时，硬件可用时优先做端到端验证
- 对 NPU/frontend 套件，不要把单次全量失败直接视为 definitive baseline；优先做失败用例复跑与基线对比
- NPU/frontend 失败分诊流程：
  1. 跑与改动相关的测试选择
  2. 对每个失败测试，单独重跑一次
  3. 若单独重跑仍失败，尽量在改前基线重跑同一测试
  4. 若基线也失败，按既有失败报告
  5. 若基线通过，按新引入问题处理
- 对新增特性和缺陷修复，能做 TDD 时优先 TDD：
  1. 先补或更新测试
  2. 确认测试能暴露缺失行为
  3. 实现代码
  4. 重跑相关测试并报告结果

## 文档与参考资料

- 修改某个子系统前，优先阅读 `docs/` 中对应文档
- 官方资料、仓内样例、当前实现三者冲突时，先指出冲突，再回到可验证依据
- 若修改 `README.md` 或 `docs/` 中英文对应内容，需同步更新或明确指出未同步部分

推荐参考：

- `docs/`
- `examples/`
- `python/tests/ut/block/`
- `python/tests/st/`

## Review 与提交标准

- 优先做最小必要改动，避免无依据重写
- 未运行的命令、测试、构建、验证不得声称已完成
- 明确区分已确认事实、推断、建议
- 临时脚本仅在难以定位问题时使用，优先放仓外；稳定后以正式测试替代
- 新增测试优先放入 `python/tests/`
- Python 测试优先使用 pytest 约定，不回退到 `unittest`

## pypto_block 独立命名空间约束

`pypto_block` 现在必须作为真正独立的顶级包维护，不能再与 `pypto` 命名空间互相污染。

严格要求：

- 禁止重新引入 `sys.modules["pypto"] = ...`、`setdefault("pypto", ...)` 或类似 alias 注入
- 禁止在 `pypto_block` 内通过 `_alias_namespace("pypto_block", "pypto")`、`_alias_namespace("pypto", "pypto_block")` 之类机制做双向镜像
- `pypto_block` 内部源码、类型存根、示例和打印输出，统一使用 `pypto_block.*`
- 若需要引用主包 `pypto` 的真实功能，必须显式区分“这是主包依赖”而不是“block 包兼容别名”
- 修改 `pypto_block` 时，必须同时检查：
  - `python/pypto_block/` 源码导入
  - `python/pypto_block/pypto_core/*.pyi`
  - block Python printer / 生成字符串
  - `python/tests/ut/block/` 中的断言和示例文本

特别注意：

- `pypto` 和 `pypto_block` 可以共存，但不能依赖导入顺序来决定谁占用顶级命名空间
- 不得把“兼容旧写法”作为理由重新抢占 `pypto` 命名空间

## Worktree 与 editable install 注意事项

- editable install 可能仍指向旧 worktree 或别的源码目录
- 若 Python 改动看起来未生效，重新从当前 worktree 执行安装，并检查实际加载路径：

```bash
python -c "import inspect, pypto_block; print(inspect.getfile(pypto_block))"
python -c "import inspect, pypto; print(inspect.getfile(pypto))"
```

---

## 通用原则

> **严格遵循以下原则**

1. **如实报告，禁止伪完成**
   - 未验证的结果，不得表述为“已完成”或“已通过”
   - 未实际执行的命令、测试、构建、提交或发布，不得声称已执行
   - 遇到失败、阻塞、权限不足或信息缺失时，必须明确说明，不得伪造过程或结果
2. **先验证，再下结论**
   - 能通过代码、文件、日志、测试或工具直接确认的事项，优先基于证据判断，不以猜测代替验证
   - 若当前环境无法完成验证，必须明确说明验证缺口、已知范围与剩余风险
3. **区分事实、推断与建议**
   - 结论应明确区分“已确认事实”“基于上下文的推断”“建议采取的动作”
   - 禁止编造不存在的文件、输出、报错、性能收益、验证状态或用户意图
4. **遵循最小必要改动原则**
   - 优先复用现有实现、既有模式和项目约定，避免无依据的重写、扩面或过度设计
   - 只解决当前任务要求的问题，不擅自引入额外功能、依赖或流程复杂度

---

## 核心算子开发原则

> **严格遵循以下原则**

1. **理解官方示例原理后实现**
   - 先确认需求、工件和 API 映射，再进入实现
   - 保持 golden、实现、测试三文件分离
2. **遇问题先定位，不简化代码**
   - 第一步：直接搜索 `docs/` API 文档
   - 第二步：查阅官方示例 `examples/`
   - 第三步：定位问题点后修复，**禁止简化代码或推翻重写**
3. **保持工件与职责清晰**
   - 需求、设计、golden、实现、测试、README 各司其职
   - 状态机、检查点、重试、恢复只由编排层定义
4. **PyPTO 场景以官方资料和仓内样例为准**
   - 在 API 映射、约束、算子行为、编译、精度和性能判断等场景中，优先依据 `docs/`、`examples/`、现有实现和官方文档
   - 当文档、样例与经验推断冲突时，应先指出冲突并回到可核实依据，不凭经验强行定论
5. **验证模式优先使用真实 NPU 环境**
   - 若 `npu-smi info` 检测到可用 NPU 环境，且用户未明确要求使用 sim 模式，则禁止使用 sim 模式进行验证
