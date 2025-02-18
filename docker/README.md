# Pypto Docker ENV

说明：本文描述如何快速创建运行PyPTO的docker容器，在使用docker容器前请**完成主机NPU硬件部署、NPU驱动及固件安装**。参考文档*docs/installation/prepare_env.md*

## 版本说明

当前提供两类dockerfile，第一类是完成cann包环境安装的版本，第二类是不涉及cann包环境安装的版本。两类版本都安装了Pypto运行所依赖的环境。

### 版本1：安装cann包

当前dockerfile构建镜像支持的环境信息如下：

```
#**************docker info*******************#
# os: ubuntu22.04, openEuler24.03
# arch: x86, arm
# python: 3.11
# cann env
# cann_verison: 8.5.0alpha001 
# torch: 2.6.0
# torch_npu: 2.6.0
# device_type: 910b，910c
#**************docker info*******************#
```

dockerfile内容如下：
使用前请根据自身环境指定ARG CANN_VERSION：
Ubuntu+910c :ARG CANN_VERSION=8.5.0.alpha001-a3-ubuntu22.04-py3.11
Ubuntu+910b :ARG CANN_VERSION=8.5.0.alpha001-910b-ubuntu22.04-py3.11
openEuler+910b :ARG CANN_VERSION=8.5.0.alpha001-910b-openeuler24.03-py3.11

```


   # check your version
   ARG CANN_VERSION=8.5.0.alpha001-a3-ubuntu22.04-py3.11
   FROM quay.io/ascend/cann:$CANN_VERSION

   # To set proxy,
   ARG PROXY=http://p_atlas:proxy%40123@172.18.100.92:8080
   ENV https_proxy=$PROXY
   ENV http_proxy=$PROXY
   ENV GIT_SSL_NO_VERIFY=1

   WORKDIR /tmp
   # extra utils, for PyPTO project
   RUN pip install --no-cache-dir \
       wheel tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
       tabulate pandas matplotlib build ml_dtypes
   RUN pip install --no-cache-dir --upgrade \
       setuptools
   # pypto wants `setuptools>=70.3.0`
   #set  torch-npu&npu version
   RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
   RUN pip install --no-cache-dir torch-npu==2.6.0

   WORKDIR /installers/3rd_party/
   RUN wget --no-check-certificate https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz
   ENV PYPTO_THIRD_PARTY_PATH=/installers/3rd_party


   RUN tar -zxvf json-3.11.3.tar.gz \
       && rm -f json-3.11.3.tar.gz \
       && cd json-3.11.3 \
       && mkdir temp && cd temp \
       && cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_HOME -DCMAKE_PREFIX_PATH=$CONDA_HOME \
       && make \
       && make install \
       && cd ../../ && rm -rf json-3.11.3

   # set default proxy
   ENV PROXY=http://p_atlas:proxy%40123@172.18.100.92:8080
   ENV https_proxy=$PROXY
   ENV http_proxy=$PROXY

```

### 版本2：不安装cann包

支持的镜像信息如下：

```
#**************docker info*******************#
# os: ubuntu22.04, openEuler24.03
# arch: x86, arm
# python: 3.11
# cann env: none
# torch: 2.6.0
# torch_npu: 2.6.0
# device_type: 910b，910c
#**************docker info*******************#
```

dockerfile内容如下：
使用Ubuntu22.04 : ARG CANN_VERSION=3.11-ubuntu22.04
使用openeuler ：ARG CANN_VERSION=3.11-openeuler22.03

```
ARG PY_VERSION=3.11-ubuntu22.04
FROM quay.io/ascend/python:$PY_VERSION

# To overwrite proxy, pass `--build-arg PROXY=` to `docker build`,
ARG PROXY=http://p_atlas:proxy%40123@172.18.100.92:8080
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1
# install system dependencies and clean index cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim wget curl unzip tar lcov openssl ca-certificates\
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev unzip pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache python-is-python3 python3-pip python3-venv ninja-build python3-dev \
    && rm -rf /var/lib/apt/list/*     # clean apt index cache
#Setup pip proxy
RUN pip config set global.index http://cmc-cd-mirror.rnd.huawei.com/pypi && pip config set global.index-url http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/ && pip config set global.trusted-host cmc-cd-mirror.rnd.huawei.com

# # Python dependencies for CANN
RUN pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py

RUN pip install --no-cache-dir \
    wheel tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes
RUN pip install --no-cache-dir --upgrade \
    setuptools
# pypto wants `setuptools>=70.3.0`
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0
# extra utils, for PyPTO project

WORKDIR /installers/3rd_party/
RUN wget --no-check-certificate https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz
ENV PYPTO_THIRD_PARTY_PATH=/installers/3rd_party


RUN tar -zxvf json-3.11.3.tar.gz \
    && rm -f json-3.11.3.tar.gz \
    && cd json-3.11.3 \
    && mkdir temp && cd temp \
    && cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_HOME -DCMAKE_PREFIX_PATH=$CONDA_HOME \
    && make \
    && make install \
    && cd ../../ && rm -rf json-3.11.3

# set default proxy
ENV PROXY=http://p_atlas:proxy%40123@172.18.100.92:8080
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考[https://quay.io/repository/ascend/](https://)，Ascend社区提供了丰富的基础镜像。

## 使用指导

### 构建镜像

这步是基于本地创建的dockerfile，构建docker镜像。构建镜像的命令如下：

```
docker build -t <镜像名：版本> -f ./dockerfile .
exp:  docker build -t pyptox86/a3:latest -f /home/dockerfiles/Cann83rc2/dockerfile .
```

### 构建容器

上一步骤构建的镜像不能直接作为开发环境使用，需基于该镜像生成使用的容器，构建容器的命令如下：

```
sudo docker run -u root -itd --name <容器名> --ipc=host --net=host --privileged \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-w /mount_home \
<镜像名：版本> \
/bin/bash
```

exp:

```
sudo docker run -u root -itd --name pypto_x86a3 --ipc=host --net=host --privileged \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-w /mount_home \
pyptox86/a3:latest \
/bin/bash
```

### 启动容器

```
docker start <容器名>
docker exec -it <容器名>  /bin/bash
```

exp:

```
docker start pypto_x86a3
docker exec -it pypto_x86a3 /bin/bash
```

进入容器拉取代码：
git clone [https://gitcode.com/cann/pypto-dev.git](https://gitcode.com/cann/pypto-dev.git)