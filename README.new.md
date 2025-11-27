# Pypto
## Pypto简介
*pypto的介绍，特点，*
*TileLang参考：Tile Language (tile-lang) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of TVM, tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.*
## 核心特性
- 易用性
- 高性能
- ...
## 典型Examples介绍
## QuickStart
### 安装指导
#### 方法1：通过whl包（pip）安装
1. whl包安装
#### 方法2：通过源码编译安装
##### 场景1：仿真环境（无NPU环境，适合仿真调测）

1. 环境准备：

   确保本地 Python3 版本 >= 3.9, 且正确安装 python3-dev 包，对应安装方式如下:
   ```shell
   # Ubuntu
   sudo apt update # 首先更新软件包列表
   sudo apt install python3 python3-dev # 安装 python3 和 python3-dev
   python3 --version # 查看 python3 版本, 应 >= 3.9
   
   # EularOS
   sudo yum install python3 python3-devel
   ```

2. 依赖安装：
   1. 编译工具
      - cmake >= 3.16.0
      - make
      - ninja (可在一定程度上提升编译速度)
      - gcc >= 7.3.0
   2. pip 包
      
      相关依赖请参见本项目根目录的 `pyproject.toml` 文件, 您可以使用如下命令完成编译 whl 包过程所依赖的 pip 包安装:
      ```shell
      python3 -m pip install "[.dev]"
      ```

   3. 开源第三方软件源码准备

      如果您本地开发环境可以正常访问 [CANN 开源第三方源码“中心仓”](https://gitcode.com/cann-src-third-party), 
      则可以选择使用本仓构建脚本的自动下载开源软件功能(详见下文 '编译' 章节描述)来完成所依赖开源软件的下载、安装。

      否则您需要手工下载以下开源软件源码压缩包(**要求 `.tar.gz` 格式**)至您开发环境任意目录(如 `/home/cann_src_third_party` 目录)。

      - [JSON for Modern C++ version 3.11.3](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz)
      - [libboundscheck v1.1.16](https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz)

3. 编译：

   推荐使用一键式构建脚本 `build.py` 完成编译, 编译完成后会在 `build_out` 目录生成 `pypto` 对应的 `whl` 包;

   一键式构建脚本 `build.py` , 对应命令如下
   ```shell
   # 说明:
   # 1. --clean 表示在构建前清理构建缓存;
   # 2. --third_party_path 指定所依赖的开源软件源码压缩包下载、解压、构建及安装路径，
   #      如果您的开发环境可以正常访问 CANN 开源第三方源码“中心仓”, 
   #      则在构建过程中会自动下载、安装所依赖的开源第三方软件至 third_party_path 指定目录;
   python3 build.py --frontend=python3 --clean --third_party_path=/home/cann_src_third_party
   ```

   本项目也支持 python3 构建 whl 包常用的 `python3 -m pip install .` 命令及 `python3 -m pip install -e .` 来构建 whl 包。
   但是通过这种方式构建前，您需要额外配置环境变量 `PYPTO_3RD_SRC_PATH` 的值为 `build.py` 中 `--third_party_path` 参数指定的路径。
   
   对应示例如下:
   ```shell
   export PYPTO_3RD_SRC_PATH=/home/cann_src_third_party
   python3 -m pip install .    # 常规安装
   python3 -m pip install -e . # 可编辑模式安装
   ```

4. 安装

   在使用一键式构建脚本 `build.py` 完成`pypto` 对应的 `whl` 包的编译后, 可以使用 `pip` 标准命令完成对应 `whl` 包的安装, 对应示例如下:
   ```shell
   # 如果在 pypto 源码根目录下执行, 以 python3.9 版本, x86_64 平台为例
   python3 -m pip install build_out/pypto-1.0.0-cp39-cp39-linux_x86_64.whl
   ```

##### 场景2：实际运行环境（有NPU环境，适合实际精度、性能调测和部署验证）
1. 环境准备：
2. 依赖安装：
3. 编译：
4. 安装
#### 方法3：使用docker镜像
*代码中归档docker参考tilelang*
1. docker下载
2. docker运行
### Sample运行指南（有没有docker，运行方式相同，用docker不需要再安装依赖）
#### 仿真sample运行
- 依赖安装
- 执行sample
- 计算图、泳道图？
#### 真实环境sample运行
- 依赖安装
- 执行sample
- 计算图、泳道图？
### 测试用例运行指南
- 测试用例运行依赖安装
- 测试用例运行指南
## 性能基准
## 贡献指南