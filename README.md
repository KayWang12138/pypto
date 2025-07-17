# Tile Framework

## 1. 概述

Tile Framework 项目介绍, 待完善。

下文介绍 Tile Framework 项目编译、UTest、STest 用例执行及性能、精度工具使用方法。

### 1.1 术语、约束

| 缩写    | 全拼                            | 解释                                                                                      |
|:------|:------------------------------|:----------------------------------------------------------------------------------------|
| Aihac | Ai Hardware Acceleration Chip | 标识 NPU 芯片品牌, 常见型号如 Atlas A2/A3 训练系列产品                                                   |
| UTest | Unit Test                     | 单元测试用例，本文指用以看护不依赖 Aihac 硬件的相关测试(如绘制 TileOp 图等)                                          |
| STest | System Test                   | 系统测试，本文指依赖 Aihac 硬件的相关测试(如 Add/Concat 等单 Op 的精度看护, Llama Layer, MLA, MoE 等网络结构等功能及精度看护) |
| Tools | Tools                         | 工具，本文指在 `tools/python` 路径下实现的调用入口归一、具备一键式执行的工具集，当前主要有性能采集工具                             |

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
├── tests                                   # 测试相关路径
│   ├── cmake                               # 测试所需的 CMake 公共配置及脚本
│   │   └── scripts
│   │       ├── golden                      # STest场景 Golden 实现脚本路径
│   │       └── golden_ctrl.py              # STest场景 Golden 生成入口脚本
│   │
│   ├── st
│   │   ├── cases
│   │   │   └── default.csv                 # 性能工具用例
│   │   │
│   │   ├── interface                       # STest用例实现(Interface 模块), 拟开源
│   │   ├── runtime                         # STest用例实现(Runtime 模块), 闭源
│   │   └── utils                           # STest 场景公共逻辑
│   │
│   └── ut
│       ├── codegen                         # UTest 用例实现(CodeGen 模块), 闭源
│       ├── interface                       # UTest 用例实现(Interface 模块), 拟开源
│       ├── runtime                         # UTest 用例实现(Runtime 模块), 闭源
│       ├── simulation                      # UTest 用例实现(Simulation 模块), 拟开源
│       └── stubs                           # UTest 场景公共桩
│
├── tools
│   └──python                               # Tools(归一的一键式工具)路径
│      ├── profiling.py
│      ├── tools.py
│      └── utils                            # Tools 公共实现
│          ├── case.py                      # 定义测试用例基类
│          ├── environ.py                   # 环境相关公共接口实现
│          └── tools_abc.py                 # 定义工具实现基类
│
└── output_tools                            # Tools(归一的一键式工具)的结果路径(执行后生成)
```

## 3. 环境准备

Tile Framework 支持由源码编译、并在编译后执行 STest/UTest/Tools，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **安装依赖**

   以下所列仅为 Tile Framework 源码编译用到的依赖，其中python、gcc的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。
   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - JSON for Modern C++（建议版本 [v3.11.3](https://github.com/nlohmann/json/releases/tag/v3.11.3)）

     如下以[JSON for Modern C++源码](https://github.com/nlohmann/json/releases/tag/v3.11.3)编译安装为例，安装命令如下：

     ```bash
     mkdir temp && cd temp                # 在 JSON for Modern C++ 源码根目录下创建临时目录并进入
     cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

   - googletest（可选，仅执行 STest/UTest 时依赖，建议版本 [v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)）

     如下以[googletest源码](https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)编译安装为例，安装命令如下：

      ```bash
     mkdir temp && cd temp                # 在 googletest 源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

2. **获取并安装CANN开发套件包**

   略

3. **设置环境变量**

   略

## 4.编译执行

下文分场景介绍构建(Build)、工具(Tools)一键式入口脚本 `build.py` 的使用方法，执行 `python3 build.py --help` 即可查看当前构建脚本 `build.py` 所支持的所有参数, 常用参数如下:

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
| -  | --asan                    | -    | 公共          | 使能 AddressSanitizer, 常用于 UTest/STest 场景, GNU/Clang 编译器均可使用.         |
| -  | --ubsan                   | -    | 公共          | 使能 UndefinedBehaviorSanitizer, 常用于 UTest/STest 场景, 推荐在 Clang 编译器使用. |
| -  | --gcov                    | -    | UTest       | 使能 GNU Coverage Instrumentation Tool, 用于 UTest 场景分析代码覆盖率情况.         |

### 4.1 UTest/STest 的编译执行

当前 UTest 及 STest 有两种常用编译、执行方式:
1. 指定 -u/-s 但不传入任何参数内容(如 `python3 build.py -u`), 此时 UTest/STest 执行默认范围内的用例, 用例范围在各 UTest/STest 模块的的 `CMakeLists.txt` 内指定;
2. 指定 -u/-s 并传入参数(如 `python3 build.py -u=FunctionTest.TestAddTensorFunctionDim2`), 此时仅会触发对应参数传入的用例执行;

编译产物:

| 场景    | 二进制                              |
|:------|:---------------------------------|
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

#### 4.1.3 扩展使用场景(STest 指定 DeviceId)

支持通过 `-d/--device` 方式指定 STest 场景所使用的 Device Id, 以充分使用多卡环境资源.

若通过 Clion 等 IDE 工具直接 Load CMakeLists.txt 方式编译, 需通过 CMake 参数指定或直接修改根目录 CMakeLists.txt 中 `ENABLE_TESTS_EXECUTE_DEVICE_ID` 取值.

```text
# 根目录 CMakeLists.txt

option(ENABLE_TESTS_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id"  0)
```

```shell
python3 build.py -s -d=1  # 使用 Device 1 执行 STest 用例
```

注意: 此功能需要 STest 用例改造, 对应接口如下:

```c++
// src/utils/stubs.h

int32_t GetCurrentDeviceId();               // 获取当前 Device ID, 取值由 build.py -d/--device 参数传入
```

#### 4.1.4 扩展使用场景(使能 ASAN/UBSAN)

ASAN, 全称 AddressSanitizer, 主要用于检查内存相关错误(越界读/写, 释放后使用, 泄露等);
UBSAN, 全称 UndefinedBehaviorSanitizer, 主要用于检查未定义行为错误(除零, 溢出等);

ASAN, UBSAN 使用的都是编译态插桩技术, 都会导致编译耗时, 编译使用内存, 运行性能, 运行内存上涨.
<font color=red>建议 ASAN / UBSAN 仅用于功能自验, 不用于性能测试. </font>

> 说明:
> 1. ASAN 在 GNU 及 Clang 编译器均可使用, 检查能力基本健全;
> 2. 实测发现 UBSAN 在 GNU 编译器存在漏报, 在 Clang 编译器检查能力相对完备, 故建议优先在 Clang 编译器使用 UBSAN;

```shell
# 使能 ASAN / UBSAN
python3 build.py --utest                  # 不使能 ASAN / UBSAN
python3 build.py --utest --asan           # 使能 ASAN
python3 build.py --utest --ubsan          # 使能 UBSAN
python3 build.py --utest --asan --ubsan   # 使能 ASAN & UBSAN
```

#### 4.1.5 UTest 支持 GCov

GCov 全称 GNU Coverage Instrumentation Tool, 用于在 GNU 编译器场景下代码覆盖率统计. 其使用方法如下:

```text
python3 build.py --utest --gcov     # 使能 GCov
# 执行结束后, 会在 build/tests/ut 路径下 生成 cov_result 目录, 传输至本地即可打开 index.html
```

### 4.2 UTest/STest 看护用例添加

1. **Step1: 用例改造**

   **确定需要添加的用例名**, 与 GTest 对应 `TEST_F` 定义一致, 如增加 `LlamaTest.Llama_1_1_1024_partitionVC_dfs` 用例, 其对应测试用例源码示例如下:

   **改造对应用例Golden读取路径**, 一般仅需要上板跑精度的 STest 才涉及, 使用 `GetGoldenDir()` 获取对应用例 Golden 路径.

   ```c++
   TEST_F(LlamaTest, Llama_1_1_1024_partitionVC_dfs) {  // 用例名由 TestSuiteName + . + TestCaseName 拼接而来
       void *x_ptr = readToDev(GetGoldenDir() + "/add_dim4_x.bin", capacity_dim4);  // 输入 x 路径, add_dim4_x.bin 名称由用例自己定义
       void *y_ptr = readToDev(GetGoldenDir() + "/add_dim4_y.bin", capacity_dim4);  // 输入 y 路径, add_dim4_x.bin 名称由用例自己定义

       readInput(GetGoldenDir() + "/add_dim4_res.bin", golden);                     // Golden 结果 add_dim4_res.bin 名称由用例自己定义
   }
   ```

2. **Step2: 补充 Golden 生成逻辑 (一般仅 Stest 涉及)**

   在执行 STest 前, 当前 CMake 工程会调用 `tests/cmake/scripts/golden_ctrl.py` 脚本生成待执行 STest 用例的 Golden.
   该脚本内根据**用例名**, 调用对应**Golden生成函数**, 生成对应 Golden. 一般所需完成动作如下:
   1. 实现Golden生成函数: 在`tests/cmake/scripts/golden` 路径下合理位置实现对应用例生成函数. 如 `AscendOnBoardTest.test_operation_tensor_dim2_add` 用例对应生成函数为 `tests/cmake/scripts/golden/op/binary_operator.py` 文件内的 `binary_operator_func1` 函数.
   2. 注册Golden生成函数: 修改 `tests/script/golden_ctrl.py` 脚本 `GoldenCtrl.__init__()` 函数, 添加用例名->Golden生成函数的对应关系.

   **注意:** 建议Golden生成函数内添加避免 Golden 重复生成逻辑, 避免每次执行均重复生成 Golden.
   若需要重新生成 Golden, 可在调用 `build.py` 时指定 `--stest_golden_path_clean` 参数, 框架会在调用生成函数前清除原有 Golden.
   ```shell
   python3 build.py --stest --stest_golden_path_clean  # 清理 Golden
   ````

3. **Step3: 增加看护列表**

   修改对应 UTest/STest 模块的的 `CMakeLists.txt` 内形如 `_CaseFilterList` 变量内容, 增加对应用例名.
   ```text
   # 上板用例(STest)配置
   #[[
   ]]
   set(_STest_CaseFilterList
           # ‘单 Op’精度用例
           "AscendOnBoardTest.test_operation_tensor_dim2_add"
           "AscendOnBoardTest.test_operation_tensor_dim4_add"
           # DeepSeekV3 'MLA 子图/整图' 精度用例
           # DeepSeekV3 'MOE 子图/整图' 精度用例
           "LlamaTest.Llama_1_1_1024_partitionVC_dfs"  # 增加用例名
           # Llama Layer 'FA' 精度用例
           # Llama Layer '整图' 精度用例
   )

       ... ...
   ```

### 4.3 Tools 工具使用

此处所指的 Tools 指在 `tools/python` 路径下实现的调用入口归一、具备一键式执行的工具集，当前主要有性能采集工具。

因 Tools 执行依赖构建(Build) 过程产生的可执行二进制(当前主要是 tile_fwk_stest)。
为便于用户使用，将 Tools 的调用入口设计为由 `build.py` 入口，并通过 `build.py` 的子命令的方式调用，常用参数如下:

执行 `python3 build.py -s -d=0 tools --help` 即可查看 Tools 所支持的所有参数, 常用参数如下:

| 缩写 | 全写                   | 类型   | 场景    | 说明                               |
|:---|:---------------------|:-----|:------|:---------------------------------|
| -  | --cases_csv_file     | Path | Tools | 指定 Tools 执行的所有用例定义 csv 文件路径      |
| -  | --tools_output_clean | -    | Tools | 指定清理 Tools 结果文件路径 `output_tools` |
| -  | --intercept          | -    | Tools | 指定存在用例失败时，进行结果拦截(主要用于 CI门禁场景)    |

执行 `python3 build.py -s -d=0 tools profling --help` 即可查看性能采集工具 Tools.Profiling 所支持的所有参数, 常用参数如下:

注意:
1. <font color=red>避免 -s/-u 后直接跟 tools 子命令</font>, 由于 -s/-u 支持传入/不传入具体用例名,
   通过 `build.py` 调用 tools 子命令时 -s/-u 后紧跟 tools, python3 会将 tools 视为 -u/-s 所指定的用例名, 不符合预期.
   可通过指定 -d/-c 等其他参数在 -u/-s 与 tools 之间, 以便正确识别;

| 缩写 | 全写                 | 类型  | 场景              | 说明                                       |
|:---|:-------------------|:----|:----------------|:-----------------------------------------|
| -  | --prof_level       | str | Tools.Profiling | 指定性能采集级别(l1 采集性能、WorkFlow图, l2 采集 PMU数据) |
| -  | --prof_warn_up_cnt | int | Tools.Profiling | 性能采集前预热次数，一般不需设置                         |
| -  | --prof_try_cnt     | int | Tools.Profiling | 性能采集时尝试次数，一般不需设置                         |
| -  | --prof_max_cnt     | int | Tools.Profiling | 性能采集时最大次数，一般不需设置                         |

#### 4.3.1 Profiling 工具使用

说明： 当前 Profiling 执行用例时，会划分为以下个阶段:
1. 预热阶段: 指在执行 Profiling 数据采集前，执行多次待执行用例. 其目的主要为避免 Device 长时间空闲而降频;
   - 预热次数默认值为 5 ;
   - 可通过 csv 文件对应用例的 `ProfWarnUpCnt` 字段来为某个用例单独配置, 命令行 `--prof_warn_up_cnt` 参数指定的优先级更高;
   - 当通过 csv 文件对应用例的 `ProfWarnUpCnt` 字段或命令行 `--prof_warn_up_cnt` 参数指定预热次数为 0 时, 表示关闭预热;
2. 采集阶段: 指性能数据采集阶段, 期间会在开启 Profiling 采集开关的情况下, 多次执行用例;
   - 当前默认会尝试执行 10 次数据采集(Profiling Try Count), 当采集成功次数达到最大值(Profiling Max Count)时停止采集, 并开始数据统计;
   - 可通过 csv 文件对应用例的 `ProfTryCnt`、`ProfMaxCnt` 字段来为某个用例单独配置; 若同时通过命令行参数(--prof_try_cnt/--prof_max_cnt)指定，则命令行参数指定值优先生效；
3. 统计阶段: 指对采集结果进行统计;
   - 时间维度, 对多次采集的 Cycle 求平均值, 并取实际 Cycle 距平均值最近的一次作为**最终结果**;
   - 抖动率维度, 计算各次采集 Cycle 与**最终结果 Cycle** 间的抖动率, 并取负向抖动率最大的作为抖动率结果;
4. 结果判定: 根据统计结果判定当前用例结果;
   - 时间维度, 若**最终结果 Cycle** 较阈值(CycleThreshold) 劣化超过 5%, 则判定失败;
   - 抖动率维度, 若**最终结果抖动率(JitterRate)** 劣于阈值(JitterRateThreshold, 默认为 -8%), 则判定失败;

以上用例采集前预热次数(Warn-up cnt)、 采集时尝试次数(Try cnt)、采集时最大次数(Max cnt)、抖动率阈值均有默认值，并可通过在 csv 内具体 Case 定义时指定单独值:

| 字段                  | 全拼                    | 作用&解释                                        |
|:--------------------|:----------------------|:---------------------------------------------|
| NetworkType         | -                     | 用例所属网络, 用于标识                                 |
| TestCaseName        | -                     | 用例名, 格式即 GTest 的 `TestSuitName.TestCaseName` |
| Enable              | -                     | 用例使能标记, 便于用例管理, 实际只会执行使能的用例                  |
| CycleThreshold      | Cycle Threshold       | Cycle 阈值                                     |
| ProfWarnUpCnt       | Profiling Warn        | 预热次数                                         |
| ProfTryCnt          | Profiling Try Count   | 尝试采集次数                                       |
| ProfMaxCnt          | Profiling Max Count   | 最大采集次数                                       |
| JitterRateThreshold | Jitter Rate Threshold | 抖动率阈值                                        |

```shell
# 仅执行 csv 文件内定义的性能用例
#   通过 --disable_auto_execute 关闭其他 STest 用例执行
python3 build.py --stest -d=0 --disable_auto_execute tools --cases_csv_file=${TILE_FWK_PATH}/tests/st/cases/default.csv profiling

# 执行 csv 文件内定义的性能用例后, 执行其他 STest 用例执行
python3 build.py --stest -d=0 tools --cases_csv_file=${TILE_FWK_PATH}/tests/st/cases/default.csv profiling
```

执行手工指定的用例

```shell
# 仅执行性能用例
python3 build.py --stest=AscendOnBoardPaCostTest.test_page_attention_low_latency_cost_precision -d=0 --disable_auto_execute tools profiling
```

执行成功后 Profiling 相关结果会在代码根目录下的 `output_tools` 路径产生，相关路径结果说明如下:

```text
output_tools
└── 20250423_093454                                                                 # 脚本时间戳，用以区分不同的采集
    └── profiling
        ├── AscendOnBoardPaCostTest.test_page_attention_low_latency_cost_precision  # 用例名
        │   └── l1                                                                  # Profiling Level
        │       ├── origin                                                          # 采集原始信息
        │       │   ├── PROF_000001_20250423173515794_03351357LFRQQREJ
        │       │   │   ├── device_0
        │       │   │   │   ├── data
        │       │   │   │   │   ├── aicpu.data.0.slice_0
        │       │   │   │   │   └── aicpu.data.0.slice_0.done
        │       │   │   │   └── result                                              # 采集统计结果
        │       │   │   │       ├── prof_statistic.csv
        │       │   │   │       └── work_flow
        │       │   │   │           ├── tilefwk_prof_data.csv
        │       │   │   │           ├── tilefwk_prof_data.json
        │       │   │   │           ├── tilefwk_prof_data.png                            # Work-Flow 图
        │       │   │   │           └── tilefwk_task_info.csv
        │       │   │   └── host
        │       │   ├── PROF_000001_20250423173520188_03352052HJEACCHO              # 其他采集结果
        │       │   ├── PROF_000001_20250423173524564_03352702OEBDENEB              #  当前 Profiling 默认会采集多次
        │       │   ├── PROF_000001_20250423173529137_03353310FJGHROEG              #  并从多次结果中挑选某次结果作为最终结果
        │       │   └── PROF_000001_20250423173533467_03353457NKJDFPHE
        │       └── result                                                          # 采集结果
        │           ├── PROF_000001_20250423173529137_03353310FJGHROEG
        │           │   ├── device_0
        │           │   │   ├── data
        │           │   │   │   ├── aicpu.data.0.slice_0
        │           │   │   │   └── aicpu.data.0.slice_0.done
        │           │   │   └── result
        │           │   │       ├── prof_statistic.csv
        │           │   │       └── work_flow
        │           │   │           ├── tilefwk_prof_data.csv
        │           │   │           ├── tilefwk_prof_data.json
        │           │   │           ├── tilefwk_prof_data.png                            # Work-Flow 图
        │           │   │           └── tilefwk_task_info.csv
        │           │   └── host
        │           ├── prof_statistic_all.csv                                      # 本 Case 采集统计信息（包含多次采集结果）
        │           └── prof_statistic_result.csv                                   # 本 Case 采集统计信息（仅有最终采集结果）
        ├── prof_statistic_all.csv                                                  # 所有 Case 采集统计信息（包含多次采集结果）
        └── prof_statistic_result.csv                                               # 所有 Case 采集统计信息（仅有最终采集结果）
```


#### 4.3.2 Precision 工具使用
**开启配置选项：**
1. 指定输入输出和golden（该部分代码要在function代码之前指定）
   AscendProgramData::GetInstance().AppendInputs({
        AscendTensorData::CreateTensor<ast2::float16>(t0, t0Data),
        .....
    });
    AscendProgramData::GetInstance().AppendOutputs({
        AscendTensorData::CreateConstantTensor<ast2::float16>(out, 0),
        .....
    });
    AscendProgramData::GetInstance().AppendGoldens({
        AscendTensorData::CreateTensor<ast2::float16>(out, r0Data),
        .....
    });

2. 开启前置配置
   config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
   config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);

3. 开启指定功能选项
   | 配置选项                                  | 类型         | 作用&解释                               |
   |:-----------------------------------------|:-------------|:---------------------------------------|
   | KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE    | bool         | 使能先生成TensorFlow graph              |
   | KEY_VERIFY_TENSOR_GRAPH                  | bool         | 验证tensor graph                        |
   | KEY_VERIFY_TENSOR_GRAPH_DUMP_OPERATION   | bool         | 导出执行op                              |
   | KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR      | bool         | 导出tensor的执行结果                     |
   | KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION  | bool         | 执行精度校验，如果为false则不进行精度校验  |
   | KEY_VERIFY_PASS                          | bool         | 验证pass                                |
   | KEY_VERIFY_PASS_SELECT                   | int          | 指定相应编号的pass进行验证                |
   | KEY_VERIFY_PASS_DUMP_OPERATION           | bool         | 导出所有pass执行的op                     |
   | KEY_VERIFY_PASS_DUMP_OPERATION_SELECT    | vector<int>  | 只选择相应编号的pass导出op                |
   | KEY_VERIFY_PASS_DUMP_TENSOR              | bool         | 导出所有pass执行的tensor                 |
   | KEY_VERIFY_PASS_DUMP_TENSOR_SELECT       | vector<int>  | 只选择相应编号的pass导出tensor            |
   | KEY_VERIFY_PASS_CHECK_PRECISION          | bool         | 执行精度校验，如果为false则不进行精度校验  |
   | KEY_VERIFY_EXECUTE_GRAPH                 | bool         | 验证execute graph                       |
   | KEY_VERIFY_EXECUTE_GRAPH_DUMP_OPERATION  | bool         | 导出执行的op                             |
   | KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR     | bool         | 导出tensor的执行结果                     |
   | KEY_VERIFY_EXECUTE_GRAPH_CHECK_PRECISION | bool         | 执行精度校验，如果为false则不进行精度校验  |
   | KEY_VERIFY_THREAD_NUMBER                 | int          | 设置线程数量，默认为64                    |

4. 执行ST或UT用例
   python3 build.py -s=DynamicAttentionPostTest.dynamic_pa_post_cast_first_bmm4

5. 执行完成后命令行界面会打印校验结果，执行过程中dump operation/tensor数据会存储在build/tests/st/output_XXX/verify目录中