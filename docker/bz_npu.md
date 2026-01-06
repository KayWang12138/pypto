# Steps to develop on Linux NPU environment

1. Install `Docker Engine` by following the official Linux instructions: https://docs.docker.com/engine/install/. Installer links are:
- For Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- For Cent-OS: https://docs.docker.com/engine/install/centos/
- For Debian: https://docs.docker.com/engine/install/debian/

2. Pull base image:

```bash
docker pull quay.io/ascend/cann:8.5.0.alpha002-910b-ubuntu22.04-py3.11
```

3. Build pypto development image

```bash
docker build . -t pypto_dev:npu -f Dockerfile.npu
```

4. Start image and run NPU tests

```bash
docker run \
  --rm -it \
  --ipc=host \
  --privileged \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit core=-1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOME":/mounted_home \
  -w /installers/pypto \
  pypto_dev:npu \
  /bin/bash

# commands below all inside container

# just run build-in pre-compiled package
python3 examples/00_hello_world/hello_world.py --run_mode npu

# or, re-compile your own repo
cd /mounted_home/your_work_dir_on_host
git clone https://gitcode.com/cann/pypto.git
cd pypto
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib:${LD_LIBRARY_PATH} 
pip install -vv -e .
```
