# QuantIndexerProlog Operator Performance Optimization Case

## Task and Goal

In the DeepSeekV3.2-Exp network, an Indexer module is introduced. The first half of this module's computation is referred to as the IndexerProlog computation. The computation flow of the quantized version is as follows:

![](../figures/docs_models_deepseek-v3-2-exp_figures_IndexerPrologQuant.png)

The operator has the following characteristics:

-   The computation includes three parts: Indexer Q, Indexer Cache, and Indexer Weight. These three parts are mutually independent and each executes serially.
-   Among the three independent computation flows, Indexer Q has the longest execution time and can mask the execution time of Indexer Cache and Indexer Weight.
-   In a typical scenario (Batch=4, MTP1, KV Cache length 64k), the computation volume is small, so not all cores are fully utilized for computation. The performance bottleneck is on data transfer.

## Analyzing the Main Bottlenecks

After verifying precision, the initial performance — also referred to as out-of-the-box performance — was obtained. The out-of-the-box performance is as follows:

![](../figures/pre_optimization_state.png)

From the out-of-the-box performance swimlane graph, the following performance optimization points can be observed:

-   Vector tasks are both numerous and sparse, with a large number of bubbles in the execution subgraphs. The issue manifests as the dequantization and RoPE computations not being merged into the same subgraph, leading to an increased number of tasks, greater scheduling overhead, and scheduling gaps. The root cause is an unreasonable TileShape setting.
-   Cube computation takes a long time; adjusting the TileShape can improve Cube performance.
-   L1 Reuse is not yet enabled, and some subgraphs that could be merged are not merged. This increases the number of tasks and causes large amounts of redundant transfers of the right matrix. In typical scenarios, transfer is usually the bottleneck; reducing the transfer volume can improve performance.

## Main Optimization Workflow

-   Tile block adjustment: Upon observation, in the dequantization and RoPE computation segments, there are many tasks, and they are not merged into a single isomorphic subgraph. If the expected result — Vector computations concentrated in one isomorphic subgraph — can be achieved, redundant transfer elimination can be avoided. By adjusting the tile blocks in the relevant computations to keep them consistent, the pass will tile the related computations into the same isomorphic subgraph. After this optimization, performance improved to 76 us. The swimlane graph is as follows:

    ![](../figures/vec_optimization_swimlane.png)

-   Cube tile block adjustment: The initial tile block is \(\[128, 128\], \[128, 128\], \[128, 128\]\), which generally yields decent performance. However, because m=8 in this operator's implementation, a more suitable tile block can be set for better performance — for example, tiling the m axis to 16, increasing the k axis to 512/1024, and tiling the n axis to 64/32. After optimization, performance reached 56 us. The swimlane graph is as follows:

    ![](../figures/cube_optimization.png)

-   The number of Cube tasks for the Q computation is too large, causing redundant transfers of the right matrix. Therefore, L1Reuse is enabled to merge tasks and reduce redundant transfers. After optimization, performance reached 49 us. The swimlane graph is as follows:

    ![](../figures/optimized_swimlane.png)

## Getting the Complete Example

The implemented example code is located at: [lightning_indexer_prolog_quant.py](../../../models/deepseek_v32_exp/deepseekv32_lightning_indexer_prolog_quant.py). This file primarily demonstrates the specific implementation of QuantIndexerProlog.

In a typical scenario (Batch=4, MTP1, KV Cache length 64k), the QuantIndexerProlog operator can be run using the following example script:

```bash
python3 models/deepseek_v32_exp/testdsv32_lightning_indexer_prolog_quant.py
```

This script provides a rich set of test cases. For different scenarios, users can modify the script to run different test cases as needed.
