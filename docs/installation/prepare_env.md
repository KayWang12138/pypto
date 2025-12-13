# 环境搭建指导 #
本文描述如何快速搭建Pypto的运行环境
## 仿真环境（无真实NPU环境） ## 
Pypto支持在CPU仿真环境运行，可以支持在CPU环境上查看计算图和泳道图
### 依赖安装 ### 
```

```
## 真实环境（有NPU硬件） ##
在上诉仿真环境的基础上，如果需要在实际NPU设备上运行，需要安装Ascend-toolkit包和torch_npu包
### Ascend-Toolkit安装指南 ###
参考[Ascend安装指南]()

### 安装torch_npu ###
参考[TorchNpu安装指南]()


## 一键式搭建脚本 ## 
我们提供了一键式环境搭建脚本，可以用于快速搭建仿真或者真实NPU环境
如果是仿真环境

```
bash prepare_env.sh simulator
```

如果是真实环境

```
bash prepare_env.sh npu [device-type:910B/910C] [with_install_driver]
```