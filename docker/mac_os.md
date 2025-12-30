# Steps to develop on Macbook with CPU-only environment

1. Download `Docker Desktop for Mac` from https://docs.docker.com/desktop/setup/install/mac-install/. Installer links are:
- For Apple Silicon (M1,M2,M3,M4,M5 chips): https://desktop.docker.com/mac/main/arm64/Docker.dmg
- For old Mac (Intel chips): https://desktop.docker.com/mac/main/amd64/Docker.dmg

2. Pull base image:

```bash
docker pull quay.io/ascend/python:3.11-ubuntu22.04
```

Note: this CPU-only image is relatively small, only ~124 MB.
To execute on NPU, need the ~15 GB full CANN image from https://quay.io/repository/ascend/cann.
such as `quay.io/ascend/cann:8.3.rc2`

3. Build pypto development image

```bash
docker build . -t pypto_dev:cpu -f Dockerfile.cpu
```

4. Start image and run CPU-only tests

```bash
docker run --rm -it \
    -v $HOME:/mounted_home \
    -w /mounted_home \
    pypto_dev:cpu \
    /bin/bash
# commands below all inside container

# just run build-in pre-compiled package
cd /installers/pypto
python examples/hello_world/hello_world.py --run_mode sim

# or, re-compile your own repo
cd /mounted_home/your_work_dir_on_host
git clone https://gitcode.com/cann/pypto.git
cd pypto
pip install -vv -e .
```
