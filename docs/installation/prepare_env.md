# 环境搭建指导 #

本文描述如何快速搭建PyPTO的运行环境

## 安装依赖

在运行本项目前，请确保如下基础依赖已经安装完成

```txt
- CMake >= 3.16.3
- make
- gcc >= 7.3.1
- python >=3.9
```

上述依赖包可通过项目 `tools` 目录下 `prepare_env.sh` 安装，命令如下，若遇到不支持系统，请参考该文件自行适配

```sheel
bash tools/prepare_env.sh --type=deps
```

**另需注意:**
1. 无论是仿真环境, 还是真实环境均需要安装 `torch`;
2. 真实环境需额外安装 `torch_npu` ;
3. 上述 `torch` 及 `torch_npu` 的安装需根据实际环境的 Python版本自行安装, 参考 [Ascend Extension for PyTorch 安装说明](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html).

## 仿真环境（无真实NPU环境） ##

PyPTO支持在CPU仿真环境运行，除上述依赖外，不需要额外再安装其他内容

## 真实环境（有NPU硬件） ##

在真实环境运行前，请确保NPU驱动和固件已安装

### NPU驱动和固件安装 ###

- 驱动版本：Ascend NDK 25.2.0

驱动与固件安装，请参考[CANN 软件安装指南](xxxxx) 自行安装

### CANN环境安装 ###

#### 安装CANN基础软件包 ####

- CANN版本：8.5.0.RC1

##### 1. 安装CANN-ToolKit #####

根据实际环境，下载对应`Ascend-cann-toolkit_8.5.0_linux-${aarch}.run`， 下载链接[CANN_TOOLKIT-8.5.0_RC1.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-toolkit_8.5.0_linux-x86_64.run) 、[CANN_TOOLKIT-8.5.0.RC1.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)

```
# 确保安装包有可执行权限
chmod +x Ascend-cann-toolkit_8.5.0_linux-${aarch}.run
# 安装命令
./Ascend-cann-toolkit_8.5.0_linux-${aarch}.run --full --force --install-path=${install_path}
```

- aarch： CPU架构，如aarch64、x86_64
- install-path：表示制定安装路径，默认安装在`/usr/local/Ascend`目录

##### 2. 安装CANN-ops #####

根据实际环境和硬件类型(支持910b/910c)，下载对应`Ascend-cann-${device_type}-ops_8.5.0_linux-${aarch}.run`， 下载链接[CANN_910b-OPS-8.5.0_RC1.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run) 、[CANN_910b-OPS-8.5.0.RC1.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run)、[CANN_910c-OPS-8.5.0_RC1.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251213_newest/Ascend-cann-910_93-ops_8.5.0_linux-x86_64.run) 、[CANN_910c-OPS-8.5.0.RC1.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251213_newest/Ascend-cann-910_93-ops_8.5.0_linux-aarch64.run)
```
# 确保安装包有可执行权限
chmod +x Ascend-cann-${device_type}-ops_8.5.0_linux-${aarch}.run
# 安装命令
./Ascend-cann-${device_type}-ops_8.5.0_linux-${aarch}.run --full --force --install-path=${install_path}
```

- device_type:  NPU 型号，如910b、910c
- aarch： CPU架构，如aarch64、x86_64
- install-path：表示制定安装路径，默认安装在`/usr/local/Ascend`目录

##### 3. 安装CANN-PTO-inst #####

根据实际环境，下载对应`Ascend-cann-pto-inst_8.5.0_linux-${aarch}.run`， 下载链接[CANN_PTO_INST-8.5.0_RC1.x86](https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-tile-lib-dev/version_compile/r1.25.0/202512/20251216_115126_901e0fb6/ubuntu_x86/cann-pto-inst_8.5.0_linux-x86_64.run) 、[CANN_PTO_INST-8.5.0.RC1.aarch64](https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-tile-lib-dev/version_compile/r1.25.0/202512/20251216_115126_901e0fb6/ubuntu_aarch64/cann-pto-inst_8.5.0_linux-aarch64.run)

```
# 确保安装包有可执行权限
chmod +x Ascend-cann-pto-inst_8.5.0_linux-${aarch}.run
# 安装命令
.Ascend-cann-pto-inst_8.5.0_linux-${aarch}.run --full --force --install-path=${install_path}
```

- aarch： CPU架构，如aarch64、x86_64
- install-path：表示制定安装路径，默认安装在`/usr/local/Ascend`目录

#### 安装脚本 ####

上述依赖包可通过项目tools目录下prepare_env.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配
` bash tools/prepare_env.sh --type=cann --device_type=910b`
| 全写                   | 类型  | 是否必须          | 说明                                         |
|:-----------------------|:----|:------------|:-------------------------------------------|
| --type                 | str   | 是          | 脚本安装类型，可选[deps, cann, third_party, all] |
| --device-type          | str | 是          | 指定 NPU 型号，可选 [910b, 910c] |
| --install-path         | str   | 否          | 指定 CANN 包安装路径 |
| --download-path        | str | 否       | 指定 CANN 包以及三方依赖包下载路径 |
| --with-install-driver  | bool | 否       | 指定是否下载 NPU 驱动和固件包，默认为 false |
| --help                 | -    | 否       | 查看命令参数帮助信息 |

#### 安装TorchNPU ####
在NPU环境运行，本项目依赖TorchNPU的基础能力，TorchNPU的安装，请参考[TorchNPU 安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)，根据实际环境的python版本，自行安装
