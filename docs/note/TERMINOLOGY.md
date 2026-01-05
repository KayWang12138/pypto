# PyPTO 术语对照表

本文件用于统一 `docs/note/` 内的术语含义，减少"同名不同义/同义不同名"导致的沟通成本。

**注意：** 本文件只定义概念，详细字段说明与使用示例见对应的专题文档（如产物文件详见[输出文件说明](03-mechanisms/output-files/)）。

---

## 产物与可视化

- **输出目录（output directory）**
  - PyPTO 运行后在工作目录下生成的产物目录，常见形态为 `output/output_<timestamp>_<pid>/`
  - 用于收集编译与运行的日志、图、性能数据等

- **`run.log`**
  - 单次运行最重要的文本日志之一
  - 典型用途：定位失败点、确认 run_mode、确认产物路径、追踪编译/执行阶段的关键事件

- **计算图（compute graph）**
  - 表示算子/子图的计算结构（节点与依赖关系）
  - 典型用途：确认算子融合边界、确认 IR 结构是否符合预期、定位“多了/少了某个 op”

- **泳道图（swimlane diagram）**
  - 表示执行时间线上各类任务（host/device、不同 stream/队列等）的排布
  - 典型用途：分析性能瓶颈（同步点、空洞、串行化、调度问题）
  - 经验：在 **SIM 场景**下，很多“结果查看/行为理解”更依赖泳道图而不是直接看设备输出

- **ToolKit / VSCode 插件**
  - 用于把计算图/泳道图等产物可视化，并与源码位置关联

---

## 编译与执行链路

- **JIT（Just-In-Time）**
  - Python 函数被装饰后，在第一次调用时触发解析/编译，并生成可执行产物

- **前端（frontend）**
  - 负责解析 Python 代码（AST/doc AST）、生成 IR，并做部分语义检查

- **IR（Intermediate Representation）**
  - 编译中间表示；PyPTO 通常采用多层级 IR（从更抽象到更贴近硬件）

- **Pass**
  - 在某个 IR 层级上运行的优化/变换阶段（如消除冗余、内存重用、调度重排等）

- **Codegen**
  - 负责把更低层的 IR 转换为可执行代码（例如生成/调用 CCE、生成二进制）

- **Machine / Runtime**
  - 执行层：负责工作空间、任务准备、设备端调度与运行时资源管理（内存/stream 等）

---

## 数据与性能

- **Tile**
  - 计算与搬运的硬件友好数据块；“基于 Tile 的编程模型”强调以 tile 为基本粒度组织计算

- **Tiling**
  - 把张量按某种形状切成 tile 的策略
  - 典型目的：提升并行度、提高 cache 命中、减少搬运、贴合向量/矩阵单元

- **Vector Tiling / Cube Tiling**
  - Vector：偏逐元素/向量运算常用
  - Cube：偏矩阵乘法/立方单元常用

- **动态形状（dynamic shape）**
  - 某些维度在编译时不固定；常通过“动态轴标注/符号化标量”表达

- **Cost Model（代价模型）**
  - 用于在仿真/优化阶段估算开销，并辅助选择策略（如 tiling/调度）
  - 详见：[examples 全景速览](01-examples/01-examples-catalog.md)中的 cost_model 说明

---

## 产物文件索引

本部分提供产物文件到用途的快速映射。详细说明见：[输出文件说明](03-mechanisms/output-files/README.md)

| 产物文件/目录 | 文件名/路径 | 用途 |
|-------------|------------|------|
| **run.log** | `output/output_<timestamp>_<pid>/run.log` | 最重要的文本日志，用于定位失败点、确认 run_mode、追踪编译/执行阶段 |
| **topo.json** | `output/output_<timestamp>_<pid>/topo.json` | 拓扑与依赖关系，用于理解计算图结构、定位某个 kernel |
| **program.json** | `output/output_<timestamp>_<pid>/program.json` | 程序/任务描述，用于理解执行计划、与 topo/run.log 关联 |
| **kernel_aicore/** | `output/output_<timestamp>_<pid>/kernel_aicore/` | AICore 侧产物（代码/二进制/编译中间产物） |
| **kernel_aicpu/** | `output/output_<timestamp>_<pid>/kernel_aicpu/` | AICPU 侧产物（代码/二进制/编译中间产物） |
| **built_in/** | `output/output_<timestamp>_<pid>/built_in/` | 内置/公共产物（共享库、公共资源等） |

**详细说明：**
- 产物总览与关系图：见[输出目录与产物总览](03-mechanisms/output-files/README.md)
- 各文件/目录的详细字段说明：见[输出文件说明](03-mechanisms/output-files/)目录下的对应文档：
  - [run.log 详细说明](03-mechanisms/output-files/run-log.md)
  - [topo.json 详细说明](03-mechanisms/output-files/topo-json.md)
  - [program.json 详细说明](03-mechanisms/output-files/program-json.md)
  - [kernel_aicore 目录说明](03-mechanisms/output-files/kernel-aicore.md)
  - [kernel_aicpu 目录说明](03-mechanisms/output-files/kernel-aicpu.md)
  - [built_in 目录说明](03-mechanisms/output-files/built-in.md)


