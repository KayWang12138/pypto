import os
import enum
import math
from itertools import product
from typing import List, Optional, Union

import pytest
import torch
import torch_npu
import numpy as np

import pypto
from pypto import (
    tensor, view, function,
    set_conv_tile_shapes,
)
from pypto.symbolic_scalar import SymInt
from pypto import pypto_impl


def run_mm():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    a = pypto.tensor([16, 32], pypto.DT_FP16, "tensor_a")
    b = pypto.tensor([32, 64], pypto.DT_FP16, "tensor_b")
    c = pypto.tensor([16, 64], pypto.DT_FP16, "tensor_c")
    with function("Matmul", a, b, c):
        pypto.set_debug_options(compile_debug_mode=1)
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        res = pypto.matmul(a, b, pypto.DT_FP16)
        print(res.shape)


@pypto.jit
def conv_kernel(a, b, d):
    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(
            tileHin=4,
            tileHout=4,
            tileWin=8,
            tileWout=8,
            tileCinFmap=16,
            tileCinWeight=32,
            tileCout=16,
            tileN=1
        ),
        pypto_impl.TileL0Info(
            tileH=2,
            tileW=8,
            tileK=16,
            tileN=16
        )
    )
    conv_tile = pypto.get_conv_tile_shapes()
    print(conv_tile)
    print(f"TileL1Info: Hin={conv_tile[0].tileHin}, Hout={conv_tile[0].tileHout}," 
          f" Win={conv_tile[0].tileWin}, Wout={conv_tile[0].tileWout},"
          f" CinFmap={conv_tile[0].tileCinFmap}, CinWeight={conv_tile[0].tileCinWeight},"
          f" Cout={conv_tile[0].tileCout}, N={conv_tile[0].tileN}")
    print(f"TileL0Info: H={conv_tile[1].tileH}, W={conv_tile[1].tileW}, K={conv_tile[1].tileK}, N={conv_tile[1].tileN}")
    print(f"SetL0Tile flag: {conv_tile[2]}")
    pypto.set_debug_options(compile_debug_mode=1)
    #d[:] = pypto.conv(a, b, pypto.DT_FP16, [1,1], [0,0,0,0], [1,1], extend_params = {'bias_tensor': c})
    d[:] = pypto.conv(a, b, pypto.DT_FP16, [1,2,3], [1,2,3,4,5,6], [1,4,5], extend_params = {})


@pypto.jit
def mm_kernel(A, B, C):
    pypto.set_debug_options(compile_debug_mode=1)
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    C[:] = pypto.matmul(A, B, pypto.DT_FP16)


def conv2d_a2a3_test():
    torch.npu.set_device(0)
    a = torch.ones([1, 2, 8, 8, 16], dtype=torch.float16)
    b = torch.ones([2, 2, 16, 16], dtype=torch.float16)
    c = torch.ones([32], dtype=torch.float16)
    d = torch.empty([1, 2, 8, 8, 16], dtype=torch.float16)
    conv_kernel(pypto.from_torch(a), pypto.from_torch(b), pypto.from_torch(c), pypto.from_torch(d))


def mm_test():
    A = torch.ones([16, 32], dtype=torch.float16)
    B = torch.ones([32, 64], dtype=torch.float16)
    C = torch.ones([16, 64], dtype=torch.float16)
    # mm_kernel(pypto.from_torch(A), pypto.from_torch(B), pypto.from_torch(C))
    # print(d)

def conv1d_a5_test():
    torch.npu.set_device(0)
    a = torch.ones([1, 32, 8], dtype=torch.float16)
    b = torch.ones([32, 32, 1], dtype=torch.float16)
    c = torch.ones([32], dtype=torch.float16)
    d = torch.empty([1, 32, 8], dtype=torch.float16)
    conv_kernel(pypto.from_torch(a), pypto.from_torch(b), pypto.from_torch(c), pypto.from_torch(d))


def conv2d_a5_test():
    torch.npu.set_device(0)
    a = torch.ones([1, 32, 8, 8], dtype=torch.float16)
    b = torch.ones([32, 32, 1, 1], dtype=torch.float16)
    c = torch.ones([32], dtype=torch.float16)
    d = torch.empty([1, 32, 8, 8], dtype=torch.float16)
    conv_kernel(pypto.from_torch(a), pypto.from_torch(b), pypto.from_torch(d))

def conv3d_a5_test():
    torch.npu.set_device(0)
    a = torch.ones([1, 96, 1, 8, 8], dtype=torch.float16)
    b = torch.ones([32, 96, 1, 1, 1], dtype=torch.float16)
    c = torch.ones([32], dtype=torch.float16)
    d = torch.empty([1, 32, 1, 8, 8], dtype=torch.float16)
    conv_kernel(pypto.from_torch(a), pypto.from_torch(b), pypto.from_torch(d))

conv3d_a5_test()
