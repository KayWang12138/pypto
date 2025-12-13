# Pypto
## Pypto简介

## 最佳实践样例
Pypto提供了xxxx。一些最佳实践样例参考：
-  [DeepSeekV3,2 xxxxx]()
- [DeepSeekV3.2 xxxx]()
- [GlmV4.5 XXXXX]()
- [GLMV4,5 XXXX]()

在Example目录下，我们规划了多个层级的样例，可以供学习如何开始写一个pypto算子，也提供了多个典型的大模型实现样例，供快速移植和部署

## 性能对比
参考tilelang，提供部分大模型推理典型算子的性能对比数据？

## 目录结构说明

## 运行环境搭建
可查看[环境安装说明]()，根据使用环境，获取环境安装指南，快速搭建pypto运行基础环境

*docs/installation/pre_env.md : 内容包括 有卡搭建说明和无卡搭建说明，并提供脚本一键式获取CANN包和安装运行依赖*

## 安装说明
### 方法1：通过PyPI安装
Pypto已发布在PyPI上，可以通过pip直接安装：

`pip3 install pypto`

### 方法2：通过源码安装
#### 基础构建环境依赖：
- CMake版本 >= 3.16
- make
- ninja（可选，可提升编译性能）
- gcc >= 7.3.0
- python 3.9+ 推荐使用`venu`模式
#### 编译过程
##### clone代码 & 安装Python编译时依赖
```
git clone <pypto-url>
cd pypto
pip install -r python/requirments.txt # 编译时的依赖
```
##### 编译过程
如果部署环境可以访问[xxxx]()，可以使用默认编译安装方法

``` pip3 install -e .```

如果部署环境无法访问[xxxxx]()，参考[编译依赖下载安装指导]()，编译安装使用如下方法
*docs/installation/third_party_install.md 内容包括需要下载哪些依赖*
```
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
pip3 install -e .
```
### 方法3：使用docker环境

为了方便快速搭建环境，同样提供已完成pypto运行环境搭建的docker镜像，详细使用请参考[docker-ReadMe]()，docker运行命令：
```
<pypto-sourcecode-path>/docker/docker_bash.sh  <docker-file-name>
```

## 样例运行
仿真环境（无NPU真实硬件）

```
cd examples/Initialization
python3 init_example.py simulator
```
真实可运行环境（有NPU真实硬件）

```
cd examples/Initialization

python3 init_example.py npu
```
## 贡献指南

## 安全声明

## 许可证
