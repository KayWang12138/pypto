# Step-by-Step Optimization Guide

## Vector-Only Kernels

| Step | Parameter            | Value                                  | Comment                                                                             |
| :--- | :------------------- | :------------------------------------- | :---------------------------------------------------------------------------------- |
| 1    | `BLOCK_SIZE`         | Maximum possible   | Increases input tensor size for better compute unit utilization.                    |
| 2    | `enable_auto_tiling` | `True`                                 | Enables automatic tile size selection (cube and vector).                            |
| 3    | `unroll_factor`      | Tuned (see options.md)             | Configures loop unrolling based on grid size (refer to parameter documentation).    |
| 4    | `partition`          | `False`                                | Disables subgraph partitioning into tasks (recommended only for vector-only kernels). |

## Mixed Kernels

| Step | Parameter                              | Value                      | Comment                                                          |
| :--- | :------------------------------------- | :------------------------- | :--------------------------------------------------------------- |
| 1    | `BLOCK_SIZE`                           | Maximum possible           | Increases input tensor size for better compute unit utilization. |
| 2    | `enable_auto_tiling`                   | `True`                     | Enables automatic tile size selection (cube and vector).         |
| 3    | `unroll_factor`                        | Tuned (see options.md) | Configures loop unrolling based on grid size.                    |
| 4    | `vec_merge_tasks` / `cube_merge_tasks` | Tuned combination          | Merges tasks with identical structures and data dependencies.    |
| 5    | `cube_l1_reuse`                        | `{}` (default)             | Merges subgraphs to enable L1 data reuse.                        |

> **Recommendation for `cube_l1_reuse`:** In most cases, automatic mode (`value={}`) outperforms manual tuning. Only resort to manual configuration after analyzing memory behavior via swimlane diagrams.
