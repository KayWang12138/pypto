export GLOBAL_LOG_LEVEL=0

source ../../ssh.sh

export PTO_TILE_LIB_CODE_PATH="/data/y00949728/workspace/pto-isa"

python build_ci.py -c -f=cpp

python build_ci.py -f=cpp -s=ParallelSortSTest.fp32_64k -d=8

# python build_ci.py -f=cpp -s=LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2 -d=8