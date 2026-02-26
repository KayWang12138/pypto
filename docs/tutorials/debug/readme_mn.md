# MoE Combine+FFN 的 M/N 分阶段调度

## 这是干什么的

`shmem_moe_combine_ffn_fused` 这条路径上，combine 等远端 token 的时候 AIV/AIC 经常闲着。这个改动把 combine -> ffn 按阶段拆开跑，让本地数据先算起来，通信和计算尽量重叠，减少尾部空等。

## M 维分解（默认开）

沿 token 维（M）把 `combine -> ffn` 分阶段执行。`MoeDistributedCombineSend` 改成 local-first 顺序 -- 先发本地 token，再处理远端的。

本地 token 先到位，FFN 首段更早启动，流水线填充更快。16 卡的时候通信抖动大，local-first 的收益尤其明显。

## N 维分解（实验中，默认关）

FFN matmul 改用列优先（column-major）推进 N 轴分块。数值还没完全收敛，先别在生产上开。

## 怎么开

```
PYTO_ENABLE_DIST_SCHEDULE=1          # M 路径生效
PYTO_DIST_ENABLE_N_COLUMN_MAJOR=1    # N 轴 column-major（实验）
```

## 怎么确认生效了

看 JIT 产物里的模板参数就行：

- `MoeDistributedCombineSend<..., true>` -- M local-first 开了
- `MoeDistributedCombineSend<..., false>` -- 没开
- `MoeFfnFusedKernel<..., true/false>` -- N 轴 column-major 的开关

## 代码在哪

调度属性定义在 `distributed_common.h`，`DistOpAttr` 持有 `enableLocalFirstSchedule` 和 `enableNDimColumnMajor`。

建图阶段的打标逻辑在 `moe_distributed_combine.cpp` 和 `moe_combine_ffn_fused.cpp`，把 M/N 调度标志写进 `distOpAttr`。CodeGen 那边 `codegen_distributed.cpp` 负责把标志映射成模板参数 -- `MoeDistributedCombineSend/Receive<..., bool>` 和 `MoeFfnFusedKernel<..., bool>`。

实际执行分支在 `moe_combine.h` 和 `moe_combine_ffn_fused.h`，bool 模板参数决定走 local-first 还是 column-major。要改执行逻辑就看这两个文件。

## 跑一下试试

```bash
WORLD_SIZE=16 USE_SHMEM_GROUP=1 USE_TORCH_FFN=0 WARMUP_ROUNDS=3 TEST_ROUNDS=10 \
PYTO_ENABLE_DIST_SCHEDULE=1 python3 examples/distributed/distributed_moe_combine_ffn_fused_multiturn.py
```

## A/B 对比

- A（基线）：`PYTO_ENABLE_DIST_SCHEDULE=0`
- B（只开 M）：`PYTO_ENABLE_DIST_SCHEDULE=1`，不设 `PYTO_DIST_ENABLE_N_COLUMN_MAJOR`
- C（M+N）：B 基础上加 `PYTO_DIST_ENABLE_N_COLUMN_MAJOR=1`

看 `PyPTO avg` 和 `p50/p90/p95/p99`，顺便留意有没有 NaN 冒出来。

## 注意

dispatch/combine 验证链路目前写死了 `BATCH_SIZE=8`，约束检查在 `moe_dispatch.cpp`。
