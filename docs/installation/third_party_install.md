# Pypto 三方依赖安装 # 
如果无法访问[CANN三方开源仓]()，按照如下步骤，安装三方依赖
```
mkdir -p <path-to-your-thirdparty>
# download third_party_des 
cd <pypto-source-code>/tools
bash download_thirdparty.sh <path-to-your-thirdpary>  # 归档依赖到内部obs仓，可以互联网访问
export PYPTO_THIRDPARTY_PATH= *absolute path for <path-to-your-thirdpary>*
```