# Docker build (optional)

```bash
cd pypto/python/triton_pypto/docker/
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t triton_pypto:v1 .
```

# Setup environment

```bash
source /usr/local/Ascend/driver/bin/setenv.bash 
source /usr/local/Ascend/cann/set_env.sh
export WORKDIR=$PWD
export PYTHONPATH=$PYTHONPATH:$WORKDIR/pypto/python/triton_pypto/
export PYTHONPATH=$PYTHONPATH:$WORKDIR/pypto/build_out
export PTO_TILE_LIB_CODE_PATH=$WORKDIR/pto-isa/
export TILE_FWK_DEVICE_ID=0
```

# Build PyPTO

```bash
cd pypto/
python3 -c 'import torch'
python3 build_ci.py -j 48 -d 3 -f python3 -s python/tests/st/test_sin.py::test_sin_FP32 --verbose
```

# Run tests

```bash
pytest --sv pytest -sv --forked python/triton_pypto/tests/
```
