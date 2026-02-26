# distributed examples

pypto 分布式集合通信的测试用例，包括 allgather、allreduce、reducescatter 和 MoE 相关操作。

## 环境准备

CANN runtime 和 shmem runtime 装好，pypto 用 editable install。NPU 数量看具体用例，有的要 2 张，有的要 4 张。

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
source /usr/local/Ascend/shmem/latest/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/home/llx/pto-isa
```

## 安装依赖

### pto-isa

```bash
cd /home/llx/pto-isa
pip install ml_dtypes numpy -q
./build.sh --pkg
./build_out/cann-pto-isa_9.0.0_linux-aarch64.run --full --install-path=/usr/local/Ascend/cann --quiet
```

### shmem

```bash
cd /home/llx/shmem
bash scripts/build.sh -package
chmod +x SHMEM_1.0.0_linux-aarch64.run
./SHMEM_1.0.0_linux-aarch64.run --check
./SHMEM_1.0.0_linux-aarch64.run --install
source /usr/local/Ascend/shmem/latest/set_env.sh
```

### pypto

```bash
cd /home/llx/llx_pypto
python3 -m pip install -e . --verbose
```

## 运行

这些脚本内部用 `torch.multiprocessing.spawn` 拉进程，直接 `python3` 跑就行，不用 torchrun:

```bash
python3 examples/distributed/distributed_allgather_test.py
python3 examples/distributed/distributed_allreduce_test.py
python3 examples/distributed/distributed_reducescatter_test.py

# MoE 相关的需要 4 卡
WORLD_SIZE=4 python3 examples/distributed/distributed_moe_dispatch_test.py
WORLD_SIZE=4 python3 examples/distributed/distributed_moe_combine_test.py
WORLD_SIZE=4 python3 examples/distributed/distributed_moe_combine_ffn_fused_multiturn.py
```

## 怎么算跑成功

- allgather / allreduce / reducescatter: 日志出现 `SUCCESS!`
- moe_dispatch: rank0 打印 `Total dispatched tokens matches`
- moe_combine: 各 rank 打印 `Output matches expected`
- moe_combine_ffn_fused_multiturn: 各 rank 打印 `Fused output matches torch baseline`

## 注意

- 脚本里写死了 `MASTER_PORT=29500`，一次只跑一个，不然端口会撞
- 如果刚 Ctrl-C 中断过，等几秒再重跑，端口可能还没释放
