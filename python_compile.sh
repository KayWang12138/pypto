export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/GBowen666/pto-isa

cd /mnt/workspace/gitCode/GBowen666/pypto/

pip3 uninstall pypto

PYPTO_BUILD_EXT_ARGS='--cmake-options="-DBUILD_WITH_CANN_MOBILE=ON"' python3 -m pip install -e . --verbose

cd /mnt/workspace/gitCode/GBowen666/pypto/python/tests/ut/litenpu_codegen/

rm -r output
rm -r simulator

python -m unittest -v test_matmul.py
