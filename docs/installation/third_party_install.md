# Pypto 三方依赖安装 # 

- JSON for Modern C++：建议版本 v3.11.3
- Securec： 官方推荐版本

如果无法访问[CANN三方开源仓](https://gitcode.com/cann-src-third-party)，可使用如下方法：

## 1. 手动下载 ##

```
mkdir -p ${path-to-your-thirdparty}
```

- [JSON for Modern C++](https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz)
- [Securec](https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz)

将上述两个三方库，下载到`path-to-your-thirdparty`中

## 2. 通过脚本一键下载 ##

*TODO：1）三方库归档到obs上，2）脚本中直接从obs上下载*

```
mkdir -p <path-to-your-thirdparty>
# download third_party_des 
cd pypto-source-code
bash toos/prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty] # 归档依赖到内部obs仓，可以互联网访问
```
- 如果未指定 `--download-path` 参数，脚本会将所需三方依赖下载到 pypto 同级目录的 `pypto_download/third_party_packages` 路径下
- 如果指定了 `--download-path` 参数，脚本会将所需三方依赖下载到 `path-to-your-thirdparty/third_party_packages` 路径下
