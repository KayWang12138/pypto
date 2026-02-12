#!/usr/bin/env python3
# coding: utf-8
"""
Pad Operator System Test (Fixed map::at error)
"""
import pypto
import torch
import numpy as np
from st.pypto_test import TestBuilder


# params: (view_shape, tile_shape, padding_config, pad_value)
def op_pad(params, a, b):
    view_shape, tile_shape, padding, pad_val = params
    pad_l, pad_r, pad_t, pad_b = padding

    n_in, m_in = a.shape

    # 外层循环遍历输入空间（而非输出空间），避免 offset 超出输入 tensor 边界
    # padding 扩展由 pypto.pad 内部的 TiledPadImpl 在输出侧自动处理
    for b_idx in pypto.loop(int(np.ceil(n_in / view_shape[0])), name="LOOP_PAD_L0", idx_name="b_idx"):
        for s_idx in pypto.loop(int(np.ceil(m_in / view_shape[1])), name="LOOP_PAD_L1", idx_name="s_idx"):
            
            offset_x = b_idx * view_shape[0]
            offset_y = s_idx * view_shape[1]
            
            valid_x = pypto.min(pypto.symbolic_scalar(n_in) - offset_x, pypto.symbolic_scalar(view_shape[0]))
            valid_y = pypto.min(pypto.symbolic_scalar(m_in) - offset_y, pypto.symbolic_scalar(view_shape[1]))

            tile_a = pypto.view(a, view_shape, [offset_x, offset_y], valid_shape=[valid_x, valid_y])
            
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            

            tile_res = pypto.pad(tile_a, padding, mode="constant", value=pad_val)
            

            pypto.assemble(tile_res, [offset_x, offset_y], b)


def op_pad_golden(params, a, b):
    # a 是输入 tensor, b 是系统预留的 tensor (这里不用管 b)
    padding = params[2]
    val = params[3]
    # 直接返回 PyTorch 计算结果
    # TestBuilder 会用这个返回值来和实际运行结果做对比
    return torch.nn.functional.pad(a, padding, mode='constant', value=val)


class PadTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
        # 1. 构造输入
        n_in, m_in = self.tiling * 1, self.tiling * 1
        a_tensor = torch.rand(n_in, m_in, dtype=torch.float32) * 10
        
        # 2. 注册输入
        # 【关键】只注册 a_tensor。不要自己创建 output tensor 传进去。
        self.setup_inputs(a_tensor)
        
        self.set_tol(rtol=1e-3, atol=1e-3)
        
        # 3. 返回输入元组
        return (a_tensor, )


def test():

    params = ((16, 16), (8, 8), (0, 4, 0, 4), 1.5)
    
    st = PadTest(params, op_pad, op_pad_golden, tiling=16)
    st()
    print("Test Passed!")

if __name__ == "__main__":
    test()