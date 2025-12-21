# 环境准备

PyPTO 除了支持在有 Ascend-NPU 硬件的 ***‘真实环境’** 中运行外，还支持在仅有 CPU 硬件的 ***‘仿真环境’*** 中运行，对应说明如下：

| 名称 | 特点 | 运行模式 |
| :-- | :-- | :-- |
| 真实环境 | 有 CPU 及 Ascend-NPU 硬件 | 既支持在 Ascend-NPU 上执行计算, 也支持通过 CPU 仿真获取预估性能 |
| 仿真环境 | 仅有 CPU 硬件 | 仅支持通过 CPU 仿真获取预估性能 |

- Ascend-NPU：Ascend 910B、910C 等 AI 加速器
- 环境类型支持 OpenEuler、Ubuntu 等 Linux 发行版

## 前提条件

使用本项目前，请确保如下基础依赖已安装。

1. **安装依赖**

    - python >=3.9
    - torch 及 torch_npu

        根据实际环境的 python 版本手工安装, 参考 [Ascend Extension for PyTorch 安装说明](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)，并确保torch, torch_npu 与 pytho 三者版本匹配。**在'仿真环境'场景，可跳过 torch_npu 的安装，但仍需安装 torch。**

2. **安装编译依赖**

    若不涉及编译 PyPTO，则可跳过本步骤。

    本项目源码编译用到的依赖如下，请注意版本要求。

    - cmake >= 3.16.3
    - make
    - gcc >= 7.3.1
    - 依赖的 pip 包及对应版本在 `python/requirements.txt` 中描述，可以使用如下命令完成安装：

        ```bash
        # 进入 pypto 项目源码根目录
        cd pypto

        # 安装相关 pip 包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - 第三方开源软件

        本项目编译过程所依赖的第三方开源软件如下：

        - JSON for Modern C++：版本 [v3.11.3](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz)
        - libboundscheck：版本 [v1.1.16](https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz)
        - googletest：版本 [1.14.0](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)

        如您的开发环境可以正常访问 [cann-src-third-party](https://gitcode.com/cann-src-third-party)，则以上软件源码包会在 PyPTO 编译过程中自动下载及编译，您可可跳过本步骤。
        否则您可以选择以下方式完成对应软件源码包的下载和准备。

        > 方法一：手工下载
        >
        > ```bash
        > # 创建用于存放第三方开源软件源码包的目录 path-to-your-thirdparty
        > mkdir -p <path-to-your-thirdparty>
        >
        > # 将上述三方库源码压缩包，下载到本地并上传到开发环境对应的 `path-to-your-thirdparty` 目录中
        > ```

        > 方法二：通过辅助脚本下载
        >
        > ```bash
        > # 创建用于存放第三方开源软件源码包的目录 path-to-your-thirdparty
        > mkdir -p <path-to-your-thirdparty>
        >
        > # 下载辅助脚本
        > *TODO：1）三方库归档到obs上，2）脚本中直接从obs上下载*
        >
        > # 执行辅助脚本
        > # - 如果未指定 `--download-path` 参数，脚本会将所需三方依赖下载到 pypto 同级目录的 `pypto_download/third_party_packages` 路径下
        > # - 如果指定了 `--download-path` 参数，脚本会将所需三方依赖下载到 `path-to-your-thirdparty/third_party_packages` 路径下
        > bash prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty]
        > ```

## 软件包安装

在真实环境编译运行 PyPTO 时必须安装如下软件包，若仅在 '仿真环境' 中编译运行 PyPTO ，可跳过本操作。

1. **安装驱动与固件**

   详细安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

    - 推荐版本：Ascend NDK 25.3.0
    - 支持版本：Ascend NDK 25.3.0、Ascend NDK 25.2.0

2. **安装CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`， 下载链接如下：
    - [Ascend-cann-toolkit_8.5.0_linux-x86_64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-toolkit_8.5.0_linux-x86_64.run)
    - [Ascend-cann-toolkit_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251216_newest/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)

    ```bash
    # 确保安装包有可执行权限
    chmod +x Ascend-cann-toolkit_8.5.0_linux-${arch}.run

    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

3. **环境变量配置**

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/cann/set_env.sh

    # 指定路径安装
    # source ${install_path}/cann/set_env.sh
    ```
