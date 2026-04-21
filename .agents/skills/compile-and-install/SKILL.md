---
name: compile-and-install
description: 
  对当前项目进行编译、打包、安装。
---


# pypto 编译、打包、安装

## 编译与打包
编译与打包的目的，是基于当前项目，生成最新的pypto包，然后把该包安装部署到本身的环境当中。

### 步骤一：环境检查
1. pypto的编译打包，不强制依赖cann；
2. 如果是通过NPU运行算子，或者要运行pvModel（pvModel又称为精度仿真），则需要依赖cann；
3. 为了同时支持NPU模式和pvModel，在进行编译之前，需要先检查环境变量 ASCEND_HOME_PATH；
4. 如果这个环境变量不存在，则需要获取cann的安装路径(通常在 ~/Ascend/ascend-toolkit目录下)，然后获取 set_env.sh文件，然后执行
```bash
source set_env.sh
```
5. 执行完成后，检视环境变量 ASCEND_HOME_PATH 是否存在 ，并指向cann的安装目录，比如 /home/{username}/Ascend/cann-9.0.0
6. 设置第三个方依赖
```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/pto-isa
export PYPTO_THIRD_PARTY_PATH=/mnt/workspace/gitCode/cann/thirdparty
```
备注：PTO_TILE_LIB_CODE_PATH环境变量，指向的是pto-isa项目的源码路径，pto是一个独立的项目，项目在： https://gitcode.com/cann/pto-isa.git

### 步骤二：编译并打包pypto
1. 执行以下命令，编译并打包pypto
```bash
python3 build_ci.py -f=python3 -c --build_type=Debug
```
2. 命令执行完成后，如果出现异常，比如编译错误，请尝试分析异常的原因，并给出修复的方案，待用户确认是否需要按方案进行修复。
3. 如果没有出现异常，执行完成后，会在当前项目下的build_out目录下生成一个pypto-*.whl文件（比如pypto-0.2.0-cp311-cp311-linux_aarch64.whl）


## 安装与验证
### 步骤一：安装pypto
1. 安装这个whl文件，为了避免pypto已经安装的情况，我们使用覆盖式安装，可以在项目根目录下，直接执行以下命令
```bash
pip3 install build_out/pypto-*.whl --force-reinstall --no-deps
```
2. 若安装无异常，则安装成功；有异常的话，请分析并反馈出来

### 步骤二：验证安装是否成功
1. 我们执行 examples/00_hello_world/hello_world.py，以验证安装的whl包是OK的。验证分为`NPU验证`和`CPU仿真验证`

2. NPU验证：
1）先查看当前环境是否存在NPU设备，可以执行以下命令检查
```bash
npu-smi info
```
2）如果存在npu设备，则执行以下命令，验证npu模式下算子运行是否正常
```bash
python3 examples/00_hello_world/hello_world.py --run_mode=npu
```
3）hello_world.py会生成 golden数据，与运算执行返回的结果进行对比，如果成功获取到运算结果，并精度与golden数据的偏差不大，则说明验证通过。

3. 仿真模式验证：
1）仿真模式，不需要检查是否存在 NPU设备，直接 执行
```bash
python3 examples/00_hello_world/hello_world.py --run_mode=sim
```
2）hello_world.py会生成 golden数据，与运算执行返回的结果进行对比，如果成功获取到运算结果，并精度与golden数据的偏差不大，则说明验证通过。

4. 如果验证通过，则当前流程结束；如果验证不通过，请分析原因，并返回分析的结果，由用户决策进一步的动作。


## 备注：
如果我们需要定位问题，我们可以打开控制台的输出日志，只需要配置两个环境变量：
```bash
export ASCEND_MODULE_LOG_LEVEL=PYPTO=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

## 外部参考

- PyPTO环境装备： https://pypto.gitcode.com/install/prepare_environment.html