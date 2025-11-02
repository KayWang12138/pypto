# pylint: disable=missing-docstring
import os

import numpy as np
import pypto
import torch
from numpy.testing import assert_allclose

BS = pypto.frontend.dynamic("BS")
NE = 128
TOP_K = 8
VIEW_SHAPE = (1024, NE)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@pypto.frontend.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True},
)
def select_experts(
    logits_input: pypto.Tensor((BS, NE), pypto.DT_FP16),
    renormalize: bool,
) -> (
    pypto.Tensor((BS, TOP_K), pypto.DT_INT32),
    pypto.Tensor((BS, TOP_K), pypto.DT_FP16),
):
    ids_k = pypto.tensor((BS, TOP_K), pypto.DT_INT32)
    weight_k = pypto.tensor((BS, TOP_K), pypto.DT_FP16)
    for bs_idx in pypto.loop(ceil_div(BS, VIEW_SHAPE[0]), unroll_List={1}):
        # View logits
        tile_logits = pypto.view(
            logits_input,
            VIEW_SHAPE,
            [bs_idx * VIEW_SHAPE[0], 0],
            valid_shape=[(BS - bs_idx * VIEW_SHAPE[0]).min(VIEW_SHAPE[0]), NE],
        )
        pypto.set_vec_tile_shapes(64, 128)

        # Cast logits to fp32
        tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
        # Softmax
        softmax_out = pypto.softmax(tile_logits_fp32, -1)

        # TopK
        topk_weight_tmp, topk_ids = pypto.topk(softmax_out, TOP_K, -1, True)

        pypto.set_vec_tile_shapes(128, 8)
        # Conditional renormalization
        if renormalize:
            denominator = pypto.sum(topk_weight_tmp, -1, True)
            topk_weight = pypto.div(topk_weight_tmp, denominator)
        else:
            topk_weight = topk_weight_tmp
        # Cast weight to fp16
        topk_weight_fp16 = pypto.cast(topk_weight, pypto.DT_FP16)

        ids_k[bs_idx * VIEW_SHAPE[0] :, :] = topk_ids
        weight_k[bs_idx * VIEW_SHAPE[0] :, :] = topk_weight_fp16

    return ids_k, weight_k


def test_select_experts():
    # 1. 设置参数
    bs = 4959
    batch_sizes = [4959, 1, 129]
    ne = 128
    top_k = 8
    renormalizes = [True, True, False]
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(len(batch_sizes)):
        bs = batch_sizes[i]
        renormalize = renormalizes[i]
        # 3. 准备测试数据
        np.random.seed(0)
        router_logits = torch.rand(
            (bs, ne), dtype=torch.float16, device=f"npu:{device_id}"
        )

        # 4. 执行kernel并获取结果
        topk_ids, topk_weights = select_experts(router_logits, renormalize)
        pypto.runtime._device_synchronize()
        print(f"topk_ids: {topk_ids}")
        print(f"topk_weights: {topk_weights}")

        # 5. 与PyTorch参考实现对比
        result = torch.softmax(router_logits.to(torch.float32), dim=-1)
        topk_weight_tensor, topk_ids_tensor = torch.topk(
            result, top_k, dim=-1, largest=True, sorted=True
        )
        topk_ids_tensor_list = topk_ids_tensor.flatten().tolist()

        if renormalize:
            denominator_g = torch.sum(topk_weight_tensor, dim=-1, keepdim=True)
            topk_weight_2_tensor = torch.div(topk_weight_tensor, denominator_g).to(
                torch.float16
            )
        else:
            topk_weight_2_tensor = topk_weight_tensor.to(torch.float16)
        topk_weight_2_tensor_list = topk_weight_2_tensor.flatten().tolist()

        # idx result
        assert_allclose(
            np.array(topk_ids.cpu().flatten().tolist()),
            np.array(topk_ids_tensor_list),
            rtol=5e-3,
            atol=5e-3,
        )

        # weight result
        assert_allclose(
            np.array(topk_weights.cpu().flatten().tolist()),
            np.array(topk_weight_2_tensor_list),
            rtol=5e-3,
            atol=5e-3,
        )


if __name__ == "__main__":
    test_select_experts()
