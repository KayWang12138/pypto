## Build steps

First build C++ part as shared library

```bash
# in main project dir
rm -rf build
cmake -B build -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0" -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
cmake --build build -- -j $(nproc) 2>&1 | tee build.log
# generates all the `libtile_fwk_*.so` files to be used with pybind later
# search for `Linking CXX shared library` in build log

# Copy config json files into conf/ folder
cd ${ASCENDCPP_DIR}/build/src/
mkdir -p conf && cd conf/
cp ../interface/tile_fwk_config.json .
cp ../passes/tile_fwk_platform_info.json .
```

Then build python binding:

```bash
cd pypto

# Setup a Python environment
python3.10 -m venv venv
source venv/bin/activate
pip install pybind11 pytest

rm -rf build
export ASCENDCPP_DIR=$(dirname $(pwd))  # point to main ascendcpp repo location
export pybind11_DIR=$(python -m site --user-site)/pybind11  # for cmake to look for 
# NOTE: or `pip install "pybind11[global]"` https://pybind11.readthedocs.io/en/stable/installing.html#include-with-pypi
cmake -B build
cmake --build build
# generates `pto.cpython-*.so` to be imported by Python

# or just `make build` for cleaning + build
```

Run tests:

```bash
# NOTE: without modifying `LD_LIBRARY_PATH` like this, will get error 
# `ImportError: libtile_fwk_passes.so: cannot open shared object file: No such file or directory` 
# although these libs are already inside `target_link_directories` and `target_link_libraries` of CMakeLists.txt
# (tested with gcc 11.4.0 and cmake 3.22.1)
export LD_LIBRARY_PATH=${ASCENDCPP_DIR}/build/src/passes:${LD_LIBRARY_PATH}

pytest ./test/test_dtype.py
make test  # run all unit tests
```
