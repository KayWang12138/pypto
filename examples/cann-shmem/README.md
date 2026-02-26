# cann-shmem examples

演示 pypto 通过 shmem 做多卡通信的两种用法: 内部句柄流程和外部初始化流程。

## 环境准备

需要 CANN runtime、shmem runtime 都装好，pypto 用 editable install，至少 2 张 NPU。

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
# 产物类似: build_out/cann-pto-isa_9.0.0_linux-aarch64.run
./build_out/cann-pto-isa_9.0.0_linux-aarch64.run --full --install-path=/usr/local/Ascend/cann --quiet
```

### shmem

```bash
cd /home/llx/shmem
bash scripts/build.sh -package
# 产物类似: SHMEM_1.0.0_linux-aarch64.run
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

两个脚本都是分布式程序，用 `torchrun` 拉起:

```bash
# 内部 shmem handle 流程
MASTER_PORT=29600 torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/cann-shmem/internal_shmem.py

# 外部 shmem init 流程
MASTER_PORT=29601 torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/cann-shmem/shmem_init.py
```

## 怎么算跑成功

每个 rank 的日志里看到这两行就行:

- `SUCCESS! Output matches ground truth.`
- `init_test.py running success!`

## 踩坑记录

**`KeyError: LOCAL_RANK`** -- 你直接 `python3 xxx.py` 跑了分布式脚本，换成 `torchrun` 就好。

**`Failed to bind the IP port` / address already in use** -- 端口被占了，换个 `MASTER_PORT`。实在不行也可以设 `PYPTO_SHMEM_IP_PORT`。
