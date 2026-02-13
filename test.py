import torch
import torch_npu
import pypto
import argparse
import numpy as np
from dataclasses import dataclass

@dataclass
class MatmulTileConfig:
    tile_shape: list
    mdl_flag: bool
    enable_k_split: bool

@pypto.jit
def call_matmul(mma, mmb, mmc, param:MatmulTileConfig):
    pypto.set_cube_tile_shapes(*param.tile_shape, param.mdl_flag, param.enable_k_split)
    mmc[:] = pypto.matmul(mma, mmb, out_dtype=pypto.DT_FP16)

@pypto.jit
def call_matmul_mx(mma, mmb, scale_a, scale_b, mmc, param:MatmulTileConfig):
    pypto.set_cube_tile_shapes(*param.tile_shape, param.mdl_flag, param.enable_k_split)
    mmc[:] = pypto.scaled_mm(mma, mmb, pypto.DT_FP16, scale_a, scale_b)

def test_matmul_normal(device_id = 0):
    torch.npu.set_device(device_id)
    m, k, n = 128, 256, 512
    tensor_mma = torch.rand((m, k), dtype=torch.float16, device=f"npu:{device_id}")
    tensor_mmb = torch.rand((k, n), dtype=torch.float16, device=f"npu:{device_id}")
    tensor_mmc = torch.zeros((m, n), dtype=torch.float16, device=f"npu:{device_id}")

    tensor_mma_pto = pypto.from_torch(tensor_mma, name="mma")
    tensor_mmb_pto = pypto.from_torch(tensor_mmb, name="mmb")
    tensor_mmc_pto = pypto.from_torch(tensor_mmc, name="mmc")

    param = MatmulTileConfig([[64, 64], [64, 64], [64, 64]], False, False)
    call_matmul(tensor_mma_pto, tensor_mmb_pto, tensor_mmc_pto, param)

    pypto.runtime._device_synchronize()

    res_golden = torch.matmul(tensor_mma, tensor_mmb)
    assert(torch.allclose(tensor_mmc.cpu(), res_golden.cpu(), atol=1e-3, rtol=1e-3))

def test_matmul_mx(device_id = 0):
    torch.npu.set_device(device_id)
    m, k, n = 128, 256, 128
    tensor_mma = torch.empty((m, k), dtype=torch.float32).uniform_(0, 100)
    tensor_mmb = torch.empty((k, n), dtype=torch.float32).uniform_(0, 100)
    tensor_mmc = torch.zeros((m, n), dtype=torch.float32)

    tensor_mma = tensor_mma.to(torch.float8_e4m3fn).npu()
    tensor_mmb = tensor_mmb.to(torch.float8_e4m3fn).npu()
    tensor_mmc = tensor_mmc.to(torch.float8_e4m3fn).npu()

    scale_a = torch.empty((128, 4, 2), dtype=torch.float32).uniform_(127, 130).to(torch.uint8)
    scale_a = 2**(scale_a - 127)
    scale_a = scale_a.to(torch.float8_e8m0fnu).npu()

    scale_b = torch.empty((4, 128, 2), dtype=torch.float32).uniform_(127, 130).to(torch.uint8)
    scale_b = 2**(scale_b - 127)
    scale_b = scale_b.to(torch.float8_e8m0fnu).npu()

    ###################golden###################
    # 将scale_a和scale_b转换为CPU上的float32类型
    scale_a_golden = scale_a.cpu().float()
    scale_b_golden = scale_b.cpu().float()
    
    scale_a_golden = scale_a_golden.reshape(scale_a_golden.size(0), scale_a_golden.size(1) * scale_a_golden.size(2))
    scale_b_golden = scale_b_golden.reshape(scale_b_golden.size(0) * scale_b_golden.size(2), scale_b_golden.size(1))
    # 使用numpy进行维度扩展
    scale_a_golden = np.repeat(scale_a_golden.numpy(), 32, axis=1)
    scale_b_golden = np.repeat(scale_b_golden.numpy(), 32, axis=0)

    tensor_a = tensor_mma.cpu() * scale_a_golden
    tensor_b = tensor_mmb.cpu() * scale_b_golden

    res_golden = torch.matmul(tensor_a, tensor_b)

    tensor_mma_pto = pypto.from_torch(tensor_mma, name="mma")
    tensor_mmb_pto = pypto.from_torch(tensor_mmb, name="mmb")
    tensor_mmc_pto = pypto.from_torch(tensor_mmc, name="mmc")
    tensor_scale_a_pto = pypto.from_torch(scale_a, name="scaleA")
    tensor_scale_b_pto = pypto.from_torch(scale_b, name="scaleB")

    ##########pypto#################
    param = MatmulTileConfig([[64, 64], [64, 64], [64, 64]], False, False)
    print("*********Start pypto**********")
    call_matmul_mx(tensor_mma_pto, tensor_mmb_pto, tensor_scale_a_pto, tensor_scale_b_pto, tensor_mmc_pto, param)
    print("*********Finish pypto**********")
    # print("pto result:".format(tensor_mmc))

    pypto.runtime._device_synchronize()

    assert(torch.allclose(tensor_mmc.cpu(), res_golden.cpu(), atol=1e-3, rtol=1e-3))

if __name__ == "__main__":
    test_matmul_mx()
    