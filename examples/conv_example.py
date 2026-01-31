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
    set_vec_tile_shapes,
    set_conv_tile_shapes,
)
from pypto.symbolic_scalar import SymInt
from pypto import pypto_impl


tileL1Info = pypto_impl.TileL1Info(
    tileHin=8,       
    tileHout=4,       
    tileWin=8,       
    tileWout=4,      
    tileCinFmap=32,  
    tileCinWeight=32,
    tileCout=64,     
    tileN=1          
)

tileL0Info = pypto_impl.TileL0Info(
    tileM=64,        
    tileK=32,        
    tileN=4          
)


@pypto.jit
def conv_kernel(a, b, c, d):
    pypto.set_conv_tile_shapes(tileL1Info, tileL0Info)
    conv_tile = pypto.get_conv_tile_shapes()
    print(conv_tile)
    print(f"TileL1Info: Hin={conv_tile[0].tileHin}, Hout={conv_tile[0].tileHout}," 
          f" Win={conv_tile[0].tileWin}, Wout={conv_tile[0].tileWout},"
          f" CinFmap={conv_tile[0].tileCinFmap}, CinWeight={conv_tile[0].tileCinWeight},"
          f" Cout={conv_tile[0].tileCout}, N={conv_tile[0].tileN}")
    print(f"TileL0Info: M={conv_tile[1].tileM}, K={conv_tile[1].tileK}, N={conv_tile[1].tileN}")
    print(f"SetL0Tile flag: {conv_tile[2]}")
    pypto.set_debug_options(compile_debug_mode=1)
    d[:] = pypto.conv(a, b, pypto.DT_FP16, c, [1,1,1,1],[0,0,0,0],[1,1,1,1],1) # 将x+1的结果写入函数参数y的原有内存空间

torch.npu.set_device(0)
a = torch.ones([1, 2, 8, 8, 16], dtype=torch.float16)
b = torch.ones([2, 2, 16, 16], dtype=torch.float16)
c = torch.ones([32], dtype=torch.float16)
d = torch.empty([64, 32], dtype=torch.float16)
conv_kernel(pypto.from_torch(a), pypto.from_torch(b), pypto.from_torch(a), pypto.from_torch(b))
print(y) # 输出x + 1的结果