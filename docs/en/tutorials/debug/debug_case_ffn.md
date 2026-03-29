# ffn_shared_expert_quant Operator NPU On-Board Debug Case

## Task and Goal

The ffn\_shared\_expert\_quant operator corresponds to the MoE shared expert computation logic in the GLM4.5 network. It includes symmetric\_quantization\_per\_token, matmul, dequant\_dynamic, and swiglu, and is used for the quantized forward propagation computation of a single shared expert. By reusing the same set of weight parameters across different tasks or data flows, it learns universal feature representations while reducing the total number of model parameters.

The following uses this operator to describe the general steps for functional debugging. For the complete example, refer to: [glm_ffn_shared_expert_quant](../../../models/glm_v4_5/glm_ffn_shared_expert_quant.py).

## Problem Localization

Suppose that when running the ffn\_shared\_expert\_quant operator test case on the NPU, the execution fails with an error. Debugging is then required to localize the problem.

You can first use the Tensor Graph to localize the problem. The main steps are as follows:

1.  Follow the steps described in [Enabling Debug Mode](debug.md) to enable debug mode, re-run the test case, and obtain the ffn\_shared\_expert\_quant operator computation graph.
2.  As described in [Viewing the Computation Graph](debug.md), use the PyPTO Toolkit visualization tool to open the Tensor Graph stage computation graph file, for example: Before\_004\_ExpandFunction\_TENSOR\_share\_loop\_idx\_Unroll1\_PATH0\_4.json:

    ![](../figures/zh-cn_image_0000002500534720.png)

3.  The actual operator code calls the Matmul interface for operations. However, as shown in the figure above, there are no Matmul operation nodes, and all subsequent Tensor and Operation nodes also fail to load, indicating an obvious anomaly.

    Check whether each Matmul operation in the operator code is used correctly. It is found that because the A/B matrix positions of the first Matmul input are swapped, the reduce axes of the A/B matrices do not satisfy the equality constraint. The error is in:

    ```python
    up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
    ```

In addition to using the computation graph for debugging, you can also use the internal DFX check mechanism to localize the problem.

The following shows the ERROR information when the ffn\_shared\_expert\_quant operator test case fails:

```text
ERROR:root:Record function share_expert_moe_main failed: ASSERTION FAILED: kSizeA == kSizeB
Matrix K dimemsion mismatch, kSizeA: 384, kSizeB: 8
, func ConstructTensorGraph, file cube_operation_impl.cpp, line 1220
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::ConstructTensorGraph<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Matrix::MatmulExtendParam const&)+0x25d) [0x7fe34630ad3d]
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::Matmul<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&)+0x14e) [0x7fe34630b51e]
```

The key information "Matrix K dimension mismatch" indicates that the error is caused by unequal K-axis sizes in the tensor shapes passed to a Matmul operation.

## Solution

Modify the implementation code for this operator:

```python
up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
```

The corrected code is:

```python
up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)
```

Re-run the ffn\_shared\_expert\_quant operator test case. It passes successfully and the problem is resolved.
