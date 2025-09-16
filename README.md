# PYPTO

## 1. 概述

下文介绍 PyPTO 项目编译、UTest、STest 用例编译执行方法。

### 1.1 术语、约束

| 缩写    | 全拼                            | 解释                                                                                    |
|:------|:------------------------------|:--------------------------------------------------------------------------------------|
| UTest | Unit Test                     | 单元测试用例，本文指用以看护不依赖 NPU 硬件的相关测试(如绘制 TensorOp 图等)                                        |
| STest | System Test                   | 系统测试，本文指依赖 NPU 硬件的相关测试(如 Add/Concat 等单 Op 的精度看护, Llama Layer, MLA, MoE 等网络结构等功能及精度看护) |

## 2. 路径结构说明

```text
.
├── build.py                                # 构建、UTest/STest执行、性能/精度工具总入口
│
├── cmake                                   # 构建所需的 CMake 公共配置及脚本
├── CMakeLists.txt                          # 顶层 CMakeLists.txt, 定义所有对外公开编译开关
│
├── include                                 # 对外头文件
├── src                                     # 源码
│
└── tests                                   # 测试相关路径
    ├── cmake                               # 测试所需的 CMake 公共配置及脚本
    │
    ├── st
    │   ├── interface                       # STest用例实现(Interface 模块), 拟开源
    │   ├── machine                         # STest用例实现(Machine 模块), 闭源
    │   └── utils                           # STest 场景公共逻辑
    │
    └── ut
        ├── codegen                         # UTest 用例实现(CodeGen 模块), 闭源
        ├── interface                       # UTest 用例实现(Interface 模块), 拟开源
        ├── machine                         # UTest 用例实现(Machine 模块), 闭源
        ├── simulation                      # UTest 用例实现(Simulation 模块), 拟开源
        └── stubs                           # UTest 场景公共桩
```

## 3. 环境准备

PyPTO 支持由源码编译、并在编译后执行 STest/UTest，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **安装依赖**

   以下所列仅为 PyPTO 源码编译用到的依赖。
   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - JSON for Modern C++（建议版本 [v3.11.3](https://github.com/nlohmann/json/releases/tag/v3.11.3)）

     如下以[JSON for Modern C++源码](https://github.com/nlohmann/json/releases/tag/v3.11.3)编译安装为例，安装命令如下：

     ```bash
     mkdir temp && cd temp                # 在 JSON for Modern C++ 源码根目录下创建临时目录并进入
     cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
     make
     make install                         # root用户安装
     # sudo make install                  # 非root用户安装
     ```

   - googletest（可选，仅执行 STest/UTest 时依赖，建议版本 [v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)）

     如下以[googletest源码](https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)编译安装为例，安装命令如下：

      ```bash
     mkdir temp && cd temp                # 在 googletest 源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装
     # sudo make install                  # 非root用户安装
     ```

## 4.编译执行

下文分场景介绍构建(Build)一键式入口脚本 `build.py` 的使用方法，执行 `python3 build.py --help` 即可查看当前构建脚本 `build.py` 所支持的所有参数, 常用参数如下:

| 缩写 | 全写                        | 类型   | 场景          | 说明                                                                  |
|:---|:--------------------------|:-----|:------------|:--------------------------------------------------------------------|
| -h | --help                    | -    | 公共          | 查看命令参数帮助信息                                                          |
| -t | --targets                 | str  | 构建          | 指定构建目标, 若指定多个(`-t=a -t=b`), 则所有目标(`a` `b`)均会被构建. 一般不需指定.            |
| -j | --job_Num                 | int  | 构建          | 指定构建过程使用的任务数, 一般不需指定.                                               |
| -c | --clean                   | -    | 公共          | 清理构建中间结果及输出结果, 即 `build` 及 `output` 路径                              |
| -u | --utest                   | str  | UTest       | 标识 UTest 场景, 支持通过该参数指定具体用例, 用例名间以 `:` 或 `,` 间隔分割                    |
| -s | --stest                   | str  | STest       | 标识 STest 场景, 支持通过该参数指定具体用例, 用例名间以 `:` 或 `,` 间隔分割                    |
| -  | --stest_golden_path       | Path | STest       | 指定 STest 场景所使用的 Golden 生成路径, 不指定时使用 `build/golden` 路径               |
| -  | --stest_golden_path_clean | -    | STest       | 指定 STest 场景 Golden 清理标记                                             |
| -  | --disable_auto_execute    | -    | UTest/STest | 指定 UTest / STest 场景不自动执行                                            |
| -  | --gcov                    | -    | UTest       | 使能 GNU Coverage Instrumentation Tool, 用于 UTest 场景分析代码覆盖率情况.         |

### 4.1 UTest/STest 的编译执行

当前 UTest 及 STest 有两种常用编译、执行方式:
1. 指定 -u/-s 但不传入任何参数内容(如 `python3 build.py -u`), 此时 UTest/STest 执行默认范围内的用例, 用例范围在各 UTest/STest 模块的的 `CMakeLists.txt` 内指定;
2. 指定 -u/-s 并传入参数(如 `python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2`), 此时仅会触发对应参数传入的用例执行;

编译产物:

| 场景    | 二进制                             |
|:------|:--------------------------------|
| UTest | `build/tests/ut/tile_fwk_utest` |
| STest | `build/tests/ut/tile_fwk_stest` |

#### 4.1.1 基本使用场景:

```shell
# 执行看护范围内 UTest
python3 build.py --utest          # 执行所有 UTest 工程内所有看护用例
python3 build.py --utest --clean  # --clean/-c 可选

python3 build.py -u
python3 build.py -u -c

# 通过参数指定执行 UTest 用例, 用例名称间以 ':' 分割
python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2
python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2:FunctionTest.TestAddTensorFunctionDim4

# 执行看护范围内 STest
python3 build.py --stest                                    # 不指定 --stest_golden_path 时, 会默认使用 build/golden 作为 Golden 目录
python3 build.py --stest
python3 build.py --stest --stest_golden_path=/home/golden   # 指定 /home/golden 作为 Golden 目录

python3 build.py -s
python3 build.py -s -c            # 命令缩写, -c 可选

# 通过参数指定执行 STest 用例, 用例名称间以 ':' 分割
python3 build.py -s=AscendOnBoardTest.test_operation_tensor_dim2_add
python3 build.py -s=AscendOnBoardTest.test_operation_tensor_dim2_add:AscendOnBoardTest.test_operation_tensor_dim4_add
```

#### 4.1.2 扩展使用场景(不自动执行用例)

默认场景下 UTest 或 STest 在编译后会自动触发执行(在 CMake 中通过 `add_custom_command` 命令方式实现).
若有用例失败, CMake 会将其视为是一种编译失败, 进而删除 UTest/STest 的二进制产物, 不便于本地 GDB 等调试.

此时通过在 `build.py` 增加 `--disable_auto_execute` 参数, 即可关闭用例自动执行功能, 进而不触发用例执行, 以便本地调试 UTest/Stest 的可执行二进制.

```shell
# 由于 `--disable_auto_execute` 参数所控制的 CMake 开关不会影响 C/C++ 源码
# 所以由改变该参数配置(不添加该参数->添加该参数 / 添加该参数 -> 不添加该参数)时需要添加 -c/--clean 以清理编译缓存, 以便该参数配置生效.
python3 build.py -u --disable_auto_execute -c
```

#### 4.1.5 UTest 支持 GCov

GCov 全称 GNU Coverage Instrumentation Tool, 用于在 GNU 编译器场景下代码覆盖率统计. 其使用方法如下:

```text
python3 build.py --utest --gcov     # 使能 GCov
# 执行结束后, 会在 build/tests/ut 路径下 生成 cov_result 目录, 传输至本地即可打开 index.html
```
