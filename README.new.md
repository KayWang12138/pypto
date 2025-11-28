# Pypto
## Pypto简介
*pypto的介绍, 特点, *
*TileLang参考: Tile Language (tile-lang) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of TVM, tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.*
## 核心特性
- 易用性
- 高性能
- ...
## 典型Examples介绍
## QuickStart
### 安装指导
#### 方法1: 通过whl包(pip)安装
1. whl包安装
#### 方法2: 通过源码编译安装
##### 场景1: 仿真环境(无NPU环境, 适合仿真调测)

1. 环境准备: 

   确保本地 Python3 版本 >= 3.9, 并已正确安装 python3-dev (或 python3-devel) 包. 安装方式如下: 
   ```shell
   # Ubuntu
   sudo apt update # 更新软件包列表
   sudo apt install python3 python3-dev # 安装 python3 和 python3-dev
   python3 --version # 查看 python3 版本, 应 >= 3.9
   
   # EulerOS
   sudo yum install python3 python3-devel # 在 EulerOS 上 python3-dev 包名称为 python3-devel
   python3 --version # 查看 python3 版本, 应 >= 3.9
   ```

2. 依赖安装: 
   1. 编译工具
      - cmake >= 3.16
      - make
      - ninja (可选, 可提升编译速度)
      - gcc >= 7.3.0

   2. 开源第三方软件源码准备

      如果您的本地开发环境能够访问 [CANN 开源第三方软件仓库](https://gitcode.com/cann-src-third-party),
      构建脚本将自动下载所需软件(详见下文“编译”章节). 

      如果无法访问, 则需要手动下载以下开源软件的源码压缩包(**必须为 .tar.gz格式**)至开发环境的任意目录(例如 /home/third_party_path): 

      - [JSON for Modern C++ version 3.11.3](https://gitcode.com/cann-src-third-party/json/releases/v3.11.3)
      - [libboundscheck v1.1.16](https://gitcode.com/cann-src-third-party/libboundscheck/releases/v1.1.16)

3. 编译: 

   推荐使用一键式构建脚本 `build.py` 完成编译. 编译完成后, 将在 `build_out` 目录生成 `pypto` 对应的 `whl` 包.

   对应命令如下:
   ```shell
   # 说明:
   # 1. --clean 表示在构建前清理构建缓存;
   # 2. --third_party_path 指定依赖的开源软件源码包的下载、解压、构建和安装路径, 
   #      若可访问 CANN 开源软件仓库, 构建时将自动下载并安装依赖至该路径.
   python3 build.py --frontend=python3 --clean --third_party_path=/home/third_party_path
   ```

   本项目也支持使用标准的 Python 包安装命令: 
   ```shell
   # 使用前需设置环境变量 PYPTO_THIRD_PARTY_PATH, 其值为 build.py 中 --third_party_path 指定的路径
   export PYPTO_THIRD_PARTY_PATH=/home/third_party_path

   python3 -m pip install .    # 常规安装
   python3 -m pip install -e . # 可编辑模式安装
   ```

4. 安装

   使用 `build.py` 脚本生成 whl 包后, 可通过 pip命令安装. 例如(在 pypto 源码根目录下, 以 Python 3.9 和 x86_64 平台为例): 
   ```shell
   python3 -m pip install build_out/pypto-1.0.0-cp39-cp39-linux_x86_64.whl
   ```

##### 场景2: 实际运行环境(有NPU环境, 适合实际精度、性能调测和部署验证)
1. 环境准备: 
2. 依赖安装: 
3. 编译: 
4. 安装
#### 方法3: 使用docker镜像
*代码中归档docker参考tilelang*
1. docker下载
2. docker运行
### Sample运行指南(有没有docker, 运行方式相同, 用docker不需要再安装依赖)
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