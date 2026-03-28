export PYPTO_THIRD_PARTY_PATH=/mnt/workspace/permute/pypto/third_party_path/
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/permute/pto-isa/
source /home/developer/Ascend/cann-9.0.0/set_env.sh
python3 ./tools/scripts/run_operation_test_with_config.py Permute -d=1 -s=0 -e=0