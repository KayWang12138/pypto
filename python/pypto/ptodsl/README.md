
```bash
# 76环境
# docker 映射的路径需要添加自己的目录，通过-v /home/g00895580:/mounted_home -w /mounted_home 这样的形式
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /home/g00895580:/mounted_home -w /mounted_home \
    ptoas:py3.12_installed /bin/bash


source /mouted_home/Ascend/cann-8.5.0/setenv.sh
export PTO_LIB_PATH=/mounted_home/pto-isa/pto-isa
# cd到ptodsl的代码路径 例如cd /mounted_home/pypto/code/ptodsl/pypto/python/pypto/ptodsl
cd PATH_TO_ptodsl 
pip3 install -e .
python3 test.py
```

