# 编译安装

## 前提条件

- **环境准备**: 编译安装 PyPTO 项目前，请先参考[环境准备](prepare_environment.md)完成基础环境搭建.

## 通过 PyPI 安装

PyPTO 已发布至 [PyPI](https://pypi.org/)，若不涉及对 PyPTO 源码的修改，可以直接使用 `pip` 命令安装:

```bash
# root 用户
python3 -m pip install pypto

# 非root 用户
python3 -m pip install pypto --user
```

## 通过源码编译安装

### 环境自检

如果您的开发环境可以正常访问 [cann-src-third-party](https://gitcode.com/cann-src-third-party)，PyPTO 编译所需的第三方开源软件将在编译过程中自动下载及编译.
如果无法访问，请参考[环境准备](prepare_environment.md)中 “准备第三方开源软件源码包” 的相关章节完成源码包准备，并在编译前设置如下环境变量:

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

### 常规安装

此方式适用于生产环境或代码稳定后使用. 编译安装后，对 Python 源码的修改不会反映到已安装的 'pypto' 包中. 对应命令如下:

```bash
# (可选) 若开发环境无法访问 cann-src-third-party，需设置
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装(root 用户)
python3 -m pip install . --verbose

# 执行编译及安装(非 root 用户)
python3 -m pip install . --verbose --user
```

**参数说明**:
- `--verbose`: 输出安装流程的基础详细信息（如下载的包版本、安装路径、依赖解析结果等）.

**其他说明**
1. 通过常规安装手段

### 可编辑安装

此方式适用于开发调试阶段. 该模式会在 `site-packages` 目录中创建指向本地源码的软链接, 对 Python 源码的修改会即时生效，无需重新安装. 对应命令如下:

```bash
# (可选) 设置 PYPTO_THIRD_PARTY_PATH, 若开发环境无法访问 cann-src-third-party
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装(root 用户)
python3 -m pip install -e . --verbose

# 执行编译及安装(非 root 用户)
python3 -m pip install -e . --verbose --user
```

**参数说明**:
- `-e`: 即 `--editable` 的简写形式，标识采用可编辑安装模式；
- `--verbose`: 会输出安装流程的基础详细信息（如下载的包版本, 安装路径，依赖解析结果等）；

## 通过 Docker 镜像安装

为了方便快速搭建环境, 同样提供已完成 PyPTO 运行环境搭建的 Docker 镜像, 详细使用请参考 [Docker README](../../docker/README.md).

Docker 运行命令:

```bash
<pypto-sourcecode-path>/docker/setup_docker_env.sh <docker-container-id>
```
