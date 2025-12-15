# Pypto Docker ENV
## 版本说明

*通过表格说明，各个docker container 与 执行环境的关系 如果使用cann包，默认使用社区可获取最新版本，历史版本同步归档*
*x86_simulator, x86_npu, arm_simulator, arm_npu*
*以ascend-docker下的内容为基础，创建docker-file，并后续维护*

```
base env
os: ubuntu, openEuler
arch: x86, arm
python: 3.11
cann env
cann_verison: 8.3.rc1 (need replace by 1215 release version)
torch: 2.8.0
torch_npu: 2.8.0 

```
## 使用指导
```
bash setup_docker_env.sh <container-id>
```