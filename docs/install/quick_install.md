# 环境部署

在使用PyPTO开发或运行算子之前，请您先参考下面步骤完成基础环境搭建和PyPTO安装。

PyPTO支持在具备NPU硬件的**真实环境**和仅有CPU硬件的**仿真环境**中运行：

| 环境类型 | 硬件要求 | 运行模式 |
|:-----|:------|:------|
| 真实环境 | 配备CPU及NPU硬件 | 支持在NPU上执行计算，也可以通过CPU仿真获取预估性能和执行计算 |
| 仿真环境 | 仅有CPU硬件 | 支持通过CPU仿真，获取预估性能和执行计算 |

**说明:**
- NPU：指昇腾AI处理器，目前仅支持如下产品型号：
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品
- 支持的系统：PyPTO支持在OpenEuler、Ubuntu等主流Linux发行版上编译和运行

## 环境准备

本项目提供多种搭建昇腾环境的方式，请按需选择。

> **说明**：本文提到的编译态和运行态含义如下，请根据实际情况选择。
>
> - 编译态：针对仅编译PyPTO不运行的场景，只需安装CANN toolkit包。
> - 运行态：针对运行PyPTO的场景（编译运行或纯运行），需安装驱动与固件、CANN toolkit包、CANN ops包。

|  安装方式  |  使用说明  |  使用场景  |
| ----- | ------ | ------ |
|  WebIDE  | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新商发版CANN包**。 | 适用于没有昇腾设备的开发者。|
|  手动安装  | - |适用有昇腾设备，想体验手动安装CANN包的开发者。|
|  Docker  | Docker镜像是一种高效部署方式，已预集成运行所需依赖。<br>当前支持Atlas A2/A3系列产品，OS支持Ubuntu和OpenEuler。 |适用有昇腾设备，需要快速搭建环境的开发者。|

### 方式1：WebIDE环境

对于无昇腾设备的开发者，可直接使用WebIDE开发平台，即"**算子一站式开发平台**"，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新商发版CANN包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考[LINK](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击"`云开发`"按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="../figures/cloudIDE.png" alt="云平台"  width="750px" height="90px">

2. 根据页面提示创建并启动云开发环境，单击"`连接 > WebIDE`"进入算子一站式开发平台，开源项目的源码资源默认在`/mnt/workspace`目录下。

   <img src="../figures/webIDE.png" alt="云平台"  width="1000px" height="150px">

环境准备完成后，请参考[PyPTO安装](#pypto安装)章节完成PyPTO的安装。

### 方式2：手动安装

对于有昇腾设备的开发者，若您想手动搭建昇腾环境，请参考下述步骤。

#### 前提条件

1. **安装Python依赖**

    - Python：版本 >= 3.9
        - **重要**：若后续需要通过源码编译安装PyPTO，还需安装Python的Development组件（常称为`python3-dev`）。

    - 安装Python依赖包：

        依赖的pip包及对应版本在`python/requirements.txt`中描述，可以使用如下命令完成安装：

        ```bash
        # 进入pypto项目源码根目录
        cd pypto

        # 安装相关pip包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - PyTorch及Ascend Extension for PyTorch：
        - **顺序说明**：请务必参考下文"软件包安装"章节完成对应工具包安装后，再安装`Ascend Extension for PyTorch`。
        - 请根据实际环境的Python版本单独安装，请参考[Ascend Extension for PyTorch文档中心]（https://hiascend.com/document/redirect/pytorchuserguide）中的《软件安装》手册。
        - **重要**：需确保`PyTorch`、`Ascend Extension for PyTorch`与`PyPTO`三者的Python版本一致。
        - **仿真环境说明**：在仿真环境中可跳过`Ascend Extension for PyTorch`的安装，但仍需安装`PyTorch`。

2. **安装编译依赖**

    若不需要编译PyPTO，可跳过本步骤。

    **安装编译工具：**

    - cmake >= 3.16.3
    - make
    - g++ >= 7.3.1

    **准备第三方开源软件源码包**

    PyPTO编译过程依赖以下第三方开源软件源码包，若您的环境可正常访问[cann-src-third-party](https://gitcode.com/cann-src-third-party)，
    这些软件的源码包会在编译时自动下载和编译，否则请手动准备：

    | 软件包                 | 版本      |
    |:--------------------|:--------|
    | JSON for Modern C++ | v3.11.3 |
    | libboundscheck      | v1.1.16 |

    手工准备第三方开源源码包的方法:

    方法一：手工下载

    ```bash
    # 创建并进入用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    mkdir -p <path-to-your-thirdparty> && cd $_

    # 下载 JSON for Modern C++ 三方库
    wget https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz

    # 下载 libboundscheck 三方库
    wget https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz
    ```

    方法二：通过辅助脚本下载

    ```bash
    # 创建用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    mkdir -p <path-to-your-thirdparty>

    # 执行辅助脚本
    # 如果未指定--download-path参数，脚本会将所需三方依赖下载到pypto同级目录的pypto_download/third_party_packages路径下
    # 如果指定了--download-path参数，脚本会将所需三方依赖下载到path-to-your-thirdparty/third_party_packages路径下
    bash tools/prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty]
    ```

#### 软件包安装

> PyPTO支持两种仿真模式：
>
> - **性能仿真**：仅评估程序运行性能，无需安装CANN、NPU驱动与固件。
> - **精度仿真**：模拟真实NPU的执行逻辑，获取运算结果，必须依赖CANN工具包。
>
> 因此：
> - 若仅编译和运行PyPTO**性能仿真**，可跳过本节。
> - 若需编译和运行**精度仿真**，或计划在**真实NPU环境**中编译运行PyPTO，必须安装如下软件包。

##### 使用安装脚本

toolkit包、ops包、PTO-inst包的下载与安装可通过项目tools目录下prepare_env.sh一键执行，命令如下，若遇到不支持系统，请参考该文件自行适配：

```bash
bash tools/prepare_env.sh --type=cann --device-type=a2
```

| 参数                    | 类型   | 是否必须 | 说明                                       |
|:----------------------|:-----|:-----|:-----------------------------------------|
| --type                | str  | 是    | 脚本安装类型，可选：deps, cann, third_party, all |
| --device-type         | str  | 是    | 指定NPU型号，可选：a2, a3              |
| --install-path        | str  | 否    | 指定CANN包安装路径                            |
| --download-path       | str  | 否    | 指定CANN包以及三方依赖包下载路径                     |
| --with-install-driver | bool | 否    | 指定是否下载NPU驱动和固件包，默认为false             |
| --help                | -    | 否    | 查看命令参数帮助信息                               |

##### 手动安装

1. **安装驱动与固件（运行态依赖）**

    详细安装指导请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"安装NPU驱动和固件"章节。驱动与固件是运行态依赖，若仅编译，可以不安装。

    CANN-8.5.0 社区版:

    - 推荐版本：Ascend HDK 25.5.0
    - 支持版本：Ascend HDK 25.5.0

    CANN-8.5.0 商发版:

    - 推荐版本：Ascend HDK 25.5.0
    - 支持版本：Ascend HDK 25.5.0、Ascend HDK 25.3.0、Ascend HDK 25.2.0

2. **安装CANN包**

    请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

    - 安装CANN toolkit包

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

    - 安装CANN ops包（运行态依赖）

        ops包是运行态依赖，若仅编译算子，可不安装此包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        - \$\{cann\_version\}：表示CANN包版本号。
        - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
        - \$\{soc\_name\}：表示NPU型号名称。
        - \$\{install\_path\}：表示指定安装路径，ops包需与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

4. **获取pto-isa源码**

    方法一：安装CANN pto-isa包

    根据实际环境下载对应的安装包（如果浏览器不支持自动下载，请选择右键，"链接另存为..."）：
    - x86：[cann-pto-isa_8.5.0_linux-x86_64.run](http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/release_version/ubuntu_x86/cann-pto-isa_linux-x86_64.run)
    - aarch64：[cann-pto-isa_8.5.0_linux-aarch64.run](http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/release_version/ubuntu_aarch64/cann-pto-isa_linux-aarch64.run)

    ```bash
    # 安装命令
    bash ./cann-pto-isa_linux-*.run --full
    ```

    方法二：下载源码方式

    ```bash
    # 创建用于存放pto-isa源码的目录
    mkdir -p ${path-to-your-pto-isa}
    git clone https://gitcode.com/cann/pto-isa.git
    # 设置环境变量
    export PTO_TILE_LIB_CODE_PATH="${path-to-your-pto-isa}/pto-isa"
    # 检查目录是否存在
    ls ${PTO_TILE_LIB_CODE_PATH}/include/pto/
    ```

    - \$\{path-to-your-pto-isa\}：存放pto-isa源码的路径。

#### 环境变量配置

安装完成后请配置环境变量。上述环境变量配置只在当前窗口生效，用户可以按需将以下命令写入环境变量配置文件（如`.bashrc`文件）。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定路径安装
# source ${install_path}/ascend-toolkit/set_env.sh
```

#### 环境验证

安装完CANN包后，可验证环境和驱动是否正常。

- **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

环境准备完成后，请参考[PyPTO安装](#pypto安装)章节完成PyPTO的安装。

### 方式3：Docker安装

对于有昇腾设备的开发者，若您想快速搭建昇腾环境，可使用Docker镜像部署。

> **说明**：
>
> - 使用Docker前，请务必**完成宿主机NPU驱动和固件安装**，请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"安装NPU驱动和固件"章节。
> - 建议Docker版本：**v27.2.1及以上**。
> - 镜像文件比较大，下载需要一定时间，请您耐心等待。关于docker命令的选项介绍可通过`docker --help`查询。

当前提供两类Dockerfile，均安装PyPTO运行所需依赖，区别在于是否在镜像内预装CANN包：

| Dockerfile版本 | 说明 |
|:---|:---|
| 版本1：安装CANN包 | 基于Ascend CANN基础镜像，已预集成CANN包。适用于希望开箱即用的场景。 |
| 版本2：不安装CANN包 | 仅安装基础依赖，CANN包需在容器内单独安装。适用于需要自定义CANN版本的场景。 |

#### 版本1：安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler24.03，架构支持x86_64和aarch64，Python 3.11，CANN 8.5.0，支持A2/A3。

在使用前，请根据**操作系统 + 硬件类型**指定`CANN_VERSION`：

- **Ubuntu + A3**：`ARG CANN_VERSION=8.5.0-a3-ubuntu22.04-py3.11`
- **Ubuntu + A2**：`ARG CANN_VERSION=8.5.0-910b-ubuntu22.04-py3.11`
- **openEuler + A3**：`ARG CANN_VERSION=8.5.0-a3-openeuler24.03-py3.11`
- **openEuler + A2**：`ARG CANN_VERSION=8.5.0-910b-openeuler24.03-py3.11`

根据CPU架构指定`TARGETPLATFORM`：

- **x86_64**：`ARG TARGETPLATFORM=linux/amd64`
- **aarch64**：`ARG TARGETPLATFORM=linux/arm64`

> **说明**：若上述信息与实际硬件及驱动不匹配，将导致CANN包安装失败，从而导致镜像构建失败。

```dockerfile
# step1: 指定 CANN 基础镜像版本
ARG CANN_VERSION=8.5.0-a3-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:$CANN_VERSION

# 指定目标平台架构
ARG TARGETPLATFORM=linux/amd64

# [Optional] 设置 HTTP/HTTPS 代理（按需配置）
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 工作目录
WORKDIR /tmp

# step2: 安装 PyPTO 项目构建/运行所需依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gdb gawk wget curl tar lcov openssl ca-certificates \
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache \
    python-is-python3 python3-pip python3-venv ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 安装指定版本 torch / torch-npu（CPU 源 + NPU 插件）
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir torch-npu==2.6.0

# [Optional] step3: 在镜像中安装由 PyPTO 提供的 CANN 包（按需开启）
# 以下内容默认注释，如需要可取消注释并根据网络环境及仓库地址调整。
#
# WORKDIR /mount_home
# RUN git clone https://gitcode.com/cann/pypto.git
# WORKDIR /mount_home/pypto
# ARG CANN_VERSION
# RUN if echo "${CANN_VERSION}" | grep -iq "910b"; then \
#         DEVICE_TYPE="a2"; \
#     elif echo "${CANN_VERSION}" | grep -iq "a3"; then \
#         DEVICE_TYPE="a3"; \
#     else \
#         echo "ERROR: Unsupported CANN_VERSION format: ${CANN_VERSION}" 1>&2 && \
#         echo "Version should contain '910b' or 'a3' (case-insensitive)" 1>&2 && \
#         exit 1; \
#     fi && \
#     echo "DEVICE_TYPE=${DEVICE_TYPE}" && \
#     chmod +x tools/prepare_env.sh && \
#     bash tools/prepare_env.sh --type=cann --device-type=${DEVICE_TYPE} --install-path=/usr/local/Ascend/CANN_pypto --quiet
#
# # Note: 设置环境变量，容器登录自动生效
# RUN \
#     CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/CANN_pypto/ascend-toolkit/set_env.sh" && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc
#
# ENTRYPOINT ["/bin/bash", "-c", "\
#     source /usr/local/Ascend/CANN_pypto/ascend-toolkit/set_env.sh && \
#     exec \"$@\"", "--"]

# step4: 安装 cann-pto-isa
ARG PTO_ISA_INSTALL_PATH=/usr/local/Ascend
ENV PTO_ISA_INSTALL_PATH=$PTO_ISA_INSTALL_PATH
WORKDIR /tmp
RUN set -e; \
    ARCH="unknown"; \
    URL_SUFFIX=""; \
    case "${TARGETPLATFORM}" in \
        "linux/amd64") \
            ARCH="x86_64"; \
            URL_SUFFIX="ubuntu_x86/cann-pto-isa_linux-x86_64.run"; \
            ;; \
        "linux/arm64") \
            ARCH="aarch64"; \
            URL_SUFFIX="ubuntu_aarch64/cann-pto-isa_linux-aarch64.run"; \
            ;; \
        *) \
            echo "ERROR: Unsupported or undefined TARGETPLATFORM: ${TARGETPLATFORM}"; \
            echo "Please set TARGETPLATFORM to 'linux/amd64' or 'linux/arm64' during build."; \
            exit 1; \
            ;; \
    esac; \
    echo "Target platform: ${TARGETPLATFORM}, architecture: ${ARCH}"; \
    PACKAGE_URL="http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/release_version/${URL_SUFFIX}"; \
    PACKAGE_NAME="cann-pto-isa_8.5.0_linux-${ARCH}.run"; \
    echo "Downloading package from: ${PACKAGE_URL}"; \
    wget --quiet --no-check-certificate -O "${PACKAGE_NAME}" "${PACKAGE_URL}"; \
    chmod +x "${PACKAGE_NAME}"; \
    echo "Installing ${PACKAGE_NAME} to ${PTO_ISA_INSTALL_PATH}"; \
    ./"${PACKAGE_NAME}" --quiet --full --install-path="${PTO_ISA_INSTALL_PATH}"; \
    echo "cann-pto-isa installation completed."

# step5: [Optional] 设置默认代理（仅当需要统一代理时启用）
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考Ascend社区提供的基础镜像：[https://quay.io/repository/ascend/cann](https://quay.io/repository/ascend/cann)

#### 版本2：不安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler22.03，架构支持x86_64和aarch64，Python 3.11，支持A2/A3。

根据操作系统指定`PY_VERSION`：

- **Ubuntu 22.04**：`ARG PY_VERSION=3.11-ubuntu22.04`
- **openEuler 22.03**：`ARG PY_VERSION=3.11-openeuler22.03`

```dockerfile
ARG PY_VERSION=3.11-ubuntu22.04
FROM quay.io/ascend/python:$PY_VERSION

# [Optional] 设置 HTTP/HTTPS 代理
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 安装系统依赖并清理 APT 缓存索引
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gdb gawk wget curl tar lcov openssl ca-certificates \
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache \
    python-is-python3 python3-pip python3-venv ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 升级 setuptools，满足 pypto 要求
RUN pip install --no-cache-dir --upgrade setuptools

# 安装 torch / torch-npu
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0

# [Optional] 设置默认代理，便于容器内访问外网
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他Python/OS组合的镜像，可参考：[https://quay.io/repository/ascend/python](https://quay.io/repository/ascend/python)

> **说明**：PyPTO考虑到会在国内外都有部署使用，因此提供的参考Dockerfile基于国内外更常用的quay.io。若在国内存在访问quay.io较慢的情况，可通过配置Docker代理解决：
>
> ```bash
> # 配置信任证书
> mkdir -p /etc/systemd/system/docker.service.d/
> tee -a /etc/docker/daemon.json > /dev/null << 'EOF'
> {
>   "insecure-registries":["quay.io", "cdn01.quay.io"]
> }
> EOF
>
> # 配置Docker代理
> tee -a /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null << 'EOF'
> [Service]
> Environment="HTTP_PROXY=<代理地址>"
> Environment="HTTPS_PROXY=<代理地址>"
> EOF
>
> systemctl daemon-reexec
> systemctl daemon-reload
> systemctl restart docker.service
> ```

#### 构建镜像

在本地准备好对应版本的Dockerfile（例如保存为`Dockerfile`），执行镜像构建命令：

```bash
docker build -t <镜像名:版本> -f ./Dockerfile .
# 示例：
# docker build -t pyptox86/a3:latest -f ./Dockerfile .
```

#### 创建并启动容器

仅有镜像无法直接作为开发环境使用，需要基于该镜像创建容器。为确保容器能够正确访问NPU硬件和相关驱动，需在启动时映射宿主机设备和驱动目录。示例命令如下：

```bash
sudo docker run -u root -itd --name <容器名> --ipc=host --net=host --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -w /mount_home \
    <镜像名:版本> \
    /bin/bash
```

| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `-u root` | 以root用户运行容器。 | - |
| `--name <容器名>` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--ipc=host` | 使用宿主机的IPC命名空间，便于进程间通信。 | - |
| `--net=host` | 使用宿主机网络，便于网络访问。 | - |
| `--privileged` | 赋予容器特权模式，确保设备访问权限。 | - |
| `--device=/dev/davinci0` | 将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行`npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。如需映射多张卡，增加多个`--device`参数即可。|
| `--device=/dev/davinci_manager` | 映射NPU设备管理接口。 | - |
| `--device=/dev/devmm_svm` | 映射设备内存管理接口。 | - |
| `--device=/dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro` | 挂载宿主机的NPU驱动库到容器内（只读）。 | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info:ro` | 挂载CANN软件安装信息文件（只读）。 | - |
| `-w /mount_home` | 指定容器内工作目录。 | - |
| `-itd` | `-i`（交互式）、`-t`（分配伪终端）、`-d`（后台运行）的组合参数。 | - |
| `<镜像名:版本>` | 指定要运行的Docker镜像。 | 请确保与`docker build`时指定的镜像名和标签一致。 |
| `/bin/bash` | 容器启动后立即执行的命令。 | - |

启动并进入容器：

```bash
# 启动容器
docker start <容器名>

# 进入容器
docker exec -it <容器名> /bin/bash
```

环境准备完成后，请参考[PyPTO安装](#pypto安装)章节在容器内完成PyPTO的安装。

> **说明**：出于兼容性考虑，当前Docker环境中编译构建得到的`whl`包建议仅在对应Docker容器内使用。

## PyPTO安装

### 源码下载

请根据CANN软件版本下载对应分支源码，\$\{tag\_version\}表示分支标签名。

```bash
# 下载项目对应分支源码
git clone -b ${tag_version} https://gitcode.com/cann/pypto.git
```

对于WebIDE环境，**已默认提供最新商发版本的项目源码**，如需获取其他版本源码，也需通过上述命令下载源码。

> [!NOTE] 注意
>
> - gitcode平台在使用HTTPS协议的时候要配置并使用个人访问令牌代替登录密码进行克隆，推送等操作。
> - 若您的编译环境无法访问网络，无法通过git指令下载代码，请先在联网环境中下载源码，再手动上传。

### 通过源码编译安装（推荐）

#### 环境自检

如果您的开发环境可以正常访问[cann-src-third-party](https://gitcode.com/cann-src-third-party)，PyPTO编译所需的第三方开源软件将在编译过程中自动下载及编译。
如果无法访问，请参考[手动安装 - 前提条件](#前提条件)中"准备第三方开源软件源码包"的相关章节完成源码包准备，并在编译前设置如下环境变量：

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

#### 常规安装

此方式适用于生产环境或代码稳定后使用。编译安装后，对Python源码的修改不会体现到已安装的`pypto`包中。对应命令如下：

```bash
# (可选)若开发环境无法访问cann-src-third-party，需设置
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装
python3 -m pip install . --verbose
```

**参数说明**：
 `--verbose`：输出安装流程的基础详细信息(如下载的包版本、安装路径、依赖解析结果等)。

**高级配置**：

以下功能依赖`pip`支持`--config-setting`参数，如需使用，请确保`pip`版本不低于**22.1**。

1. 调整编译类型

   默认情况下，常规安装编译出的C++层二进制文件为`Release`类型。若需进行调试，可通过`pip`的`--config-setting`参数指定不同的编译类型：

   ```bash
   # 通过--config-setting参数指定C++层二进制编译为Debug类型
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug'
   ```

2. 开启C++编译器详细输出模式

   ```bash
   # 额外开启C++编译器详细输出模式(便于定位C++编译问题)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug --cmake-verbose'
   ```

3. 指定CMake Generator类型

   ```bash
   # 指定CMake Generator类型(Ninja，Ninja需提前安装)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-generator=Ninja'

   # 指定CMake Generator类型(Unix Makefiles)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-generator="Unix Makefiles"'
   ```

#### 可编辑安装

此方式适用于开发调试阶段。该模式会在`site-packages`目录中创建指向本地源码的软链接，对Python源码的修改会即时生效，无需重新安装。对应命令如下：

```bash
# (可选)设置PYPTO_THIRD_PARTY_PATH，若开发环境无法访问cann-src-third-party
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装(可编辑模式)
python3 -m pip install -e . --verbose
```

**参数说明**：
- `-e`：即`--editable`的简写形式，标识采用可编辑安装模式；
- `--verbose`：会输出安装流程的基础详细信息(如下载的包版本，安装路径，依赖解析结果等)；

**高级配置**：

PyPTO使用`setuptools`作为其编译打包工具。需要注意的是，当前版本的`setuptools`尚不支持直接接收`pip`命令通过`--config-setting`参数传递的配置。
如需指定特定的C++编译选项，您需要将相关配置预先设置在`PYPTO_BUILD_EXT_ARGS`环境变量中。该环境变量的值将在编译过程(`setup.py`)中被自动识别并使用。

以下示例演示了如何配置环境变量以编译Debug版本的C++二进制文件并开启编译器的详细输出模式，然后进行安装。

```bash
# 设置编译参数：指定编译类型为Debug，并开启编译器详细输出(便于诊断问题)
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose'

# 其他配置方式参考：指定编译类型为Debug，并开启编译器详细输出(便于诊断问题)，指定CMake Generator为Unix Makefiles
# export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose --cmake-generator="Unix Makefiles"'

# 执行编译及安装(可编辑模式)
python3 -m pip install -e . --verbose
```

### 通过PyPI安装

PyPTO已发布至[PyPI](https://pypi.org/)，若不涉及对PyPTO源码的修改，可以直接使用`pip`命令安装：

```bash
# 从PyPI源下载并安装
python3 -m pip install pypto
```

### 安装验证

完成以上步骤后，参考[样例运行](docs/invocation/examples_invocation.md)执行相关用例，验证PyPTO是否成功安装。

## 可选安装

### PyPTO Toolkit插件

如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：

1. 单击[Link](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/pypto-toolkit-1.1.0.vsix)，下载`.vsix`插件文件。

2. 打开Visual Studio Code，进入"扩展"选项卡界面，单击右上角的"..."，选择"从VSIX安装..."。
 ![vscode_install](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/images/vscode_install.png)

3. 选择已下载的`.vsix`插件文件，完成安装。

### MPI依赖

PyPTO的分布式用例依赖MPI，推荐版本 >= 3.2.1。

**方式一：通过系统包管理器安装**

```bash
# Ubuntu/Debian系统
apt-get update && apt-get install -y mpich

# CentOS/RHEL系统
yum install -y mpich
```

**方式二：通过源码编译安装**

```bash
# 以3.2.1版本为例
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

安装完成后设置环境变量：

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```
