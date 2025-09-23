## Build steps

# Setup a Python environment
```bash
pip3 install pybind11 pytest
```

# Build
```bash
python3 build.py -c
```

# Run tests:

```bash
# NOTE: without modifying `LD_LIBRARY_PATH` like this, will get error
# `ImportError: libtile_fwk_passes.so: cannot open shared object file: No such file or directory`
# although these libs are already inside `target_link_directories` and `target_link_libraries` of CMakeLists.txt
# (tested with gcc 11.4.0 and cmake 3.22.1)

cd python/src/tensor_op

pytest ./test/test_dtype.py
make test  # run all unit tests
```
