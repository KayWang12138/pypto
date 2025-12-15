# PyPTO

## PyPTO 简介

PyPTO（发音：pai p-t-o）是一款面向 AI 加速器的高性能编程框架，旨在简化复杂融合算子乃至整个模型网络的开发流程，同时保持高性能计算能力。该框架采用创新的 **PTO（Parallel Tensor/Tile Operation）编程范式**，以 **基于 Tile 的编程模型** 为核心设计理念，通过多层次的中间表示（IR）系统，将用户通过 API 构建的 AI 模型应用从高层次的 Tensor 图逐步编译成硬件指令，最终生成可在目标平台上高效执行的可执行代码。

### 核心特性

- **基于 Tile 的编程模型**：所有计算都基于 Tile（硬件感知的数据块）进行，充分利用硬件并行计算能力和内存层次结构
- **多层级计算图转换**：通过编译 Pass 将 Tensor Graph 转换为 Tile Graph、Block Graph 和 Execution Graph，每一步包括一系列 Pass 优化流程
- **自动化代码生成**：编译结果通过 CodeGen 生成底层 PTO 虚拟指令代码，再通过编译器将虚拟指令代码编译成目标平台的可执行代码
- **MPMD 执行调度**：可执行代码被加载到设备侧，通过 MPMD（Multiple Program Multiple Data）的方式调度到设备上的处理器核
- **完整的工具链支持**：全流程的编译中间产物和运行时性能数据可通过 IDE 集成的工具链可视化识别性能瓶颈，开发者也可以通过工具链控制编译和调度行为
- **Python 友好 API**：提供直观的 Tensor 级别抽象，贴近算法开发者的思维模式，支持动态 Shape 和符号化编程
- **分层抽象设计**：对不同开发者暴露不同抽象层次，算法开发者使用 Tensor 层次，性能专家使用 Tile 层次，系统开发者使用 Block 层次

### 目标用户

- **算法开发者**：主要使用 Tensor 层次编程，快速实现和验证算法，专注于算法逻辑
- **性能优化专家**：可使用 Tile 或 Block 层次，进行深度性能调优，以实现极致性能
- **系统开发者**：可在 Tensor/Tile/Block 和 PTO 虚拟指令集层次上进行三方框架对接或集成，以及工具链开发

## 最佳实践样例

PyPTO 提供了丰富的示例代码，涵盖从基础操作到复杂模型实现的多个层级。一些最佳实践样例参考：

### 大模型实现样例

- [DeepSeekV3.2 SFA](examples/models/deepseek_v32_exp/sparse_flash_attention_quant.py) - 稀疏 Flash Attention 量化实现
- [DeepSeekV3.2 MLA-PROLOG](examples/models/deepseek_v32_exp/mla_indexer_prolog_quant.py) - MLA Indexer Prolog 量化实现
- [GLM V4.5 Attention](examples/models/glm/glm_attention.py) - GLM 注意力机制实现
- [GLM V4.5 ExpertsSelector](examples/models/glm/glm_experts_selector.py) - GLM 专家选择器实现

### 学习路径

在 `examples/` 目录下，我们规划了多个层级的样例：

- **01_beginner/**：基础操作示例，帮助初学者快速上手 PyPTO 编程
- **02_intermediate/**：中级示例，包括自定义操作、神经网络模块等
- **03_advanced/**：高级示例，包括复杂模式和多函数组合
- **models/**：完整的大模型实现样例，供快速移植和部署

这些示例可以帮助开发者学习如何编写 PyPTO 算子，从简单的 Tensor 操作到复杂的模型网络实现。

## 性能表现

PyPTO 在多个 AI 工作负载上展现出优异的性能表现。框架通过多层级优化和硬件感知的代码生成，能够充分利用 AI 加速器的计算资源。

（性能对比数据待补充）

## 目录结构说明

PyPTO 项目的目录结构如下：

```
pypto-dev/
├── build.py                    # 构建、UTest、STest 执行辅助脚本
├── CMakeLists.txt              # 顶层 CMakeLists.txt，定义所有对外公开编译开关
├── pyproject.toml              # Python 编译工具配置文件
├── setup.py                    # Python 包安装配置
├── LICENSE                     # 许可证文件
│
├── python/                     # Python 源码
│   ├── pypto/                  # Python 包源码根目录
│   ├── src/                    # pybind11 源码根目录
│   └── tests/                  # Python 测试用例源码（UTest, STest）
│
├── framework/                  # C++ 源码根目录
│   ├── include/                # C++ 对外头文件
│   ├── src/                    # C++ 源码
│   │   ├── codegen/           # 代码生成模块
│   │   ├── passes/            # 编译 Pass 模块
│   │   ├── operator/          # 算子实现模块
│   │   └── ...
│   └── tests/                  # C++ 测试用例源码
│
├── examples/                    # 示例代码
│   ├── 01_beginner/           # 初级示例
│   ├── 02_intermediate/        # 中级示例
│   ├── 03_advanced/            # 高级示例
│   └── models/                 # 模型实现示例
│
├── ops/                        # 算子相关代码
├── tools/                      # 工具脚本
└── cmake/                      # 构建所需的 CMake 公共配置及脚本
```

## 运行环境搭建

可查看[环境安装说明](docs/installation/prepare_env.md)，根据使用环境，获取环境安装指南，快速搭建 PyPTO 运行基础环境。

**环境安装说明文档内容**：
- 有卡搭建说明：在真实 NPU 硬件环境下的安装配置
- 无卡搭建说明：在仿真环境下的安装配置
- 提供脚本一键式获取 CANN 包和安装运行依赖

### 系统要求

- **操作系统**：支持 OpenEuler、Ubuntu 等 Linux 发行版
- **Python**：Python 3.9 及以上版本（推荐使用 `venv` 虚拟环境）
- **编译器**：gcc >= 7.3.0
- **构建工具**：CMake >= 3.16.0，make，ninja（可选，可提升编译性能）
- **硬件平台**：Ascend 910B、910C 等 AI 加速器（或仿真环境）

## 安装说明

### 方法1：通过 PyPI 安装

PyPTO 已发布在 PyPI 上，可以通过 pip 直接安装：

```bash
pip3 install pypto
```

### 方法2：通过源码安装

#### 基础构建环境依赖

- CMake 版本 >= 3.16
- make
- ninja（可选，可提升编译性能）
- gcc >= 7.3.0
- Python 3.9+（推荐使用 `venv` 模式）

#### 编译过程

##### 1. Clone 代码 & 安装 Python 编译时依赖

```bash
git clone <pypto-url>
cd pypto
pip install -r python/requirements.txt  # 编译时的依赖
```

##### 2. 编译安装

**如果部署环境可以访问 [CANN 三方开源仓](https://gitcode.com/cann-src-third-party)**，可以使用默认编译安装方法：

```bash
pip3 install -e .
```

**如果部署环境无法访问 CANN 三方开源仓**，参考[编译依赖下载安装指导](docs/installation/third_party_install.md)，编译安装使用如下方法：

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
pip3 install -e .
```

**编译依赖下载安装指导文档内容**：详细说明需要下载哪些依赖包及其安装方法。

### 方法3：使用 Docker 环境

为了方便快速搭建环境，同样提供已完成 PyPTO 运行环境搭建的 Docker 镜像，详细使用请参考 [Docker README](docker/README.md)。

Docker 运行命令：

```bash
<pypto-sourcecode-path>/docker/setup_docker_env.sh <docker-container-id>
```

**Container ID 组成格式**：
```
{cann-version}-{device_type}-{os_version}-{python-version}
```

例如：`8.1.RC1.alpha001-910b-openeuler22.03-py3.10`

## 样例运行

### 仿真环境（无 NPU 真实硬件）

```bash
cd examples/01_beginner/01_basic_operations
python3 basic_operations.py simulator
```

### 真实可运行环境（有 NPU 真实硬件）

```bash
cd examples/01_beginner/01_basic_operations
python3 basic_operations.py npu
```

## 快速开始

以下是一个简单的 PyPTO 使用示例：

```python
import pypto

# 定义计算函数
@pypto.jit
def add_example(a, b):
    return pypto.add(a, b)

# 创建 Tensor
a = pypto.Tensor([1.0, 2.0, 3.0], dtype=pypto.float32)
b = pypto.Tensor([4.0, 5.0, 6.0], dtype=pypto.float32)

# 执行计算
result = add_example(a, b)
print(result)
```

更多示例请参考 `examples/` 目录下的示例代码。

## 文档资源

- **[IDP 文档](../IDP%20docs/)**：当前发布版本的详细文档，包括编程指南、API 参考等
- **[白皮书](../white%20paper/)**：PyPTO 框架的深度技术文档，包括设计理念、架构详解等
- **[示例代码](../examples/)**：丰富的示例代码，从基础到高级应用

## 贡献指南

我们欢迎社区贡献！如果您想为 PyPTO 项目做出贡献，请：

1. Fork 本仓库
2. 创建您的特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交您的更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启一个 Pull Request

在提交代码前，请确保：
- 代码符合项目的代码规范
- 添加了必要的测试用例
- 更新了相关文档

## 安全声明

PyPTO 项目致力于保障用户数据和应用的安全。在使用本框架时，请注意：

- 请从官方渠道获取 PyPTO 软件包
- 定期更新到最新版本以获取安全补丁
- 在生产环境中使用前，请进行充分的安全测试
- 如发现安全问题，请通过安全渠道报告

## 许可证

PyPTO 项目采用 CANN Open Software License Agreement Version 2.0 许可证。详情请参阅 [LICENSE](../LICENSE) 文件。

## 联系我们

- **问题反馈**：通过 GitHub Issues 提交问题
- **功能建议**：通过 GitHub Discussions 参与讨论
- **技术支持**：参考文档或提交 Issue

---

**注意**：本文档会持续更新，请关注最新版本。
