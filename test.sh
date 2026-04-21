source /home/developer/Ascend/cann-9.0.0/set_env.sh

export CC="ccache gcc"
export CXX="ccache g++"

export ASCEND_MODULE_LOG_LEVEL=PYPTO=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/pto-isa
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TILE_FWK_DEVICE_ID=0
export PYPTO_THIRD_PARTY_PATH=/mnt/workspace/gitCode/cann/thirdparty

python3 build_ci.py -f=python3 -c 
pip3 install build_out/pypto-*.whl --force-reinstall --no-deps

python3 examples/00_hello_world/hello_world.py --run_mode=sim
