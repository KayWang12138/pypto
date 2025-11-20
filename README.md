# PyPTO

## 1. 概述

下文介绍 PyPTO 项目编译, UTest, STest 用例编译执行方法。

### 1.1 术语, 约束

| 缩写    | 全拼                            | 解释                                                                                    |
|:------|:------------------------------|:--------------------------------------------------------------------------------------|
| UTest | Unit Test                     | 单元测试用例，本文指用以看护不依赖 NPU 硬件的相关测试(如绘制 TensorOp 图等)                                        |
| STest | System Test                   | 系统测试，本文指依赖 NPU 硬件的相关测试(如 Add/Concat 等单 Op 的精度看护, Llama Layer, MLA, MoE 等网络结构等功能及精度看护) |

## 2. 路径结构说明

下文对关键路径进行说明.

```text
.
├── build.py                                # 构建, UTest, STest执行 辅助脚本
├── cmake                                   # 构建所需的 CMake 公共配置及脚本
├── CMakeLists.txt                          # 顶层 CMakeLists.txt, 定义所有对外公开编译开关
├── pyproject.toml                          # Python 编译工具配置文件
├── LICENSE
│
├── python                                  # Python 源码
│   ├── pypto                               # Python 包源码根目录
│   ├── src                                 # pybind11 源码跟目录
│   └── tests                               # Python 测试用例源码(UTest, STest)
│       ├── st
│       └── ut
│
└── framework                               # C++ 源码根目录
    ├── include                             # C++ 对外头文件
    ├── src                                 # C++ 源码
    └── tests                               # C++ 测试用例源码(UTest, STest)
        ├── cmake
        ├── st
        └── ut
```

## 3. 环境准备

PyPTO 支持由源码编译 whl 包, 并基于 pytest 对 whl 包含的 python 接口能力进行测试, 支持基于 googletest 对 C++ 侧模块进行测试。

上述功能均需进行源码编译, 在进行源码编译前，请根据如下步骤完成环境的基础准备。

### 3.1 安装依赖

以下所列仅为 PyPTO 源码编译用到的依赖。
- python >= 3.9.5
- gcc >= 7.3.0
- cmake >= 3.16.0
- ninja（可选, 编译 whl 包时需要）
- JSON for Modern C++（建议版本 [v3.11.3](https://github.com/nlohmann/json/releases/tag/v3.11.3)）

  如下以[JSON for Modern C++源码](https://github.com/nlohmann/json/releases/tag/v3.11.3)编译安装为例，安装命令如下：

  ```bash
  mkdir temp && cd temp                # 在 JSON for Modern C++ 源码根目录下创建临时目录并进入
  cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
  make
  make install                         # root用户安装
  # sudo make install                  # 非root用户安装
  ```

- googletest（可选，仅执行 C++ STest/UTest 时依赖，建议版本 [v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)）

  如下以[googletest源码](https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)编译安装为例，安装命令如下：

  ```bash
  mkdir temp && cd temp                # 在 googletest 源码根目录下创建临时目录并进入
  cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
  make
  make install                         # root用户安装
  # sudo make install                  # 非root用户安装
  ```

### 3.2 安装 CANN 包

略

## 4.编译执行

下文分场景介绍构建一键式入口脚本 `build.py` 的常用使用方法，执行 `python3 build.py --help` 即可查看当前构建脚本 `build.py` 所支持的所有参数, 常用参数如下:

| 缩写 | 全写                     | 类型  | 场景          | 说明                                         |
|:---|:-----------------------|:----|:------------|:-------------------------------------------|
| -h | --help                 | -   | 公共          | 查看命令参数帮助信息                                 |
| -f | --frontend             | str | 公共          | 指定前端类型, 可选 [python3, cpp], 默认值为 cpp        |
| -c | --clean                | -   | 公共          | 清理构建中间结果及输出结果, 即 `build` 及 `output` 路径     |
| -u | --utest                | str | UTest       | 标识 UTest 场景, 支持通过该参数指定具体用例, 用例名间以 `,` 间隔分割 |
| -s | --stest                | str | STest       | 标识 STest 场景, 支持通过该参数指定具体用例, 用例名间以 `,` 间隔分割 |

### 4.1 whl 包编译

#### 4.1.1 环境准备

在编译 whl 包时, 需要额外安装 ninja 编译器和 一些 pip 包, 对应 pip 包依赖的 `requirements.txt` 内容如下:

```txt
# 编译 whl 包时所需的 pip 包
setuptools
wheel
pybind11>=2.0.1
tomli>=2.0.0 ; python_version >= "3.0" and python_version < "3.11"
```

需要注意 `torch` 及 `torch_npu` 包安装, 对应内容参考 [Ascend Extension for PyTorch 安装说明](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html).

#### 4.1.2 编译执行

可通过如下命令一键式编译 PyPTO 对应 whl 包, 编译完成后会在源码根目录 `build_out` 目录下产生 `pypto-*.whl` 包. 而后可以通过 pip 包管理命令进行安装.

```shell
# source CANN 包环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 执行编译
python3 build.py --clean --frontend=python3
```

***注意:*** 如果把 `pypto-*.whl` 安装在非默认路径, 则需要根据实际安装路径, 结合 pip 包管理机制要求, 额外配置 `PYTHONPATH` 环境变量.

***注意:*** CANN 包安装结束后, 需要按照对应要求安装其运行所依赖的 pip 包, 可参考对应 CANN 版本的 "安装后配置" 章节描述，
如[CANN 8.2.RC1 版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)。

#### 4.1.3 UTest/STest 的编译执行

当前 UTest 及 STest 有两种常用编译, 执行方式:
1. 指定 -u/-s 但不传入任何参数内容(如 `python3 build.py -u`), 此时 UTest/STest 执行默认范围内的用例;
2. 指定 -u/-s 并传入参数(如 `python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2`), 此时仅会触发对应参数传入的用例执行;


##### 4.1.4 C++ 场景常见使用方法

```shell
# 执行看护范围内 UTest
python3 build.py --utest          # 执行所有 UTest 工程内所有看护用例
python3 build.py --utest --clean  # --clean/-c 可选

python3 build.py -u
python3 build.py -u -c

# 通过参数指定执行 UTest 用例, 用例名称间以 ':' 分割
python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2
python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2,FunctionTest.TestAddTensorFunctionDim4

# 执行看护范围内 STest
python3 build.py --stest                                    # 不指定 --stest_golden_path 时, 会默认使用 build/golden 作为 Golden 目录
python3 build.py --stest
python3 build.py --stest --stest_golden_path=/home/golden   # 指定 /home/golden 作为 Golden 目录

python3 build.py -s
python3 build.py -s -c            # 命令缩写, -c 可选

# 通过参数指定执行 STest 用例, 用例名称间以 ':' 分割
python3 build.py -s=AscendOnBoardTest.test_operation_tensor_dim2_add
python3 build.py -s=AscendOnBoardTest.test_operation_tensor_dim2_add,AscendOnBoardTest.test_operation_tensor_dim4_add
```

##### 4.1.5 Python 场景常见使用方法

Python 场景的使用方式与 C++ 场景类似, 一般仅需额外添加 `--frontend=python3` 参数. UTest/STest 场景需要安装一些额外的 pip 包, 对应安装包要求如下:

```txt
pytest
pytest-forked
pytest-xdist
bfloat16   # 后续版本中会去除该依赖
```

常见使用方式如下:

```shell
python3 build.py --clean --frontend=python3 --utest                                 # 执行全量 Python UTest 用例

python3 build.py --clean --frontend=python3 --utest=python/tests/ut/test_dtype.py   # 指定内容与 pytest 使用方式一致

python3 build.py --clean --frontend=python3 --stest                                 # 执行全量 Python STest 用例
```
