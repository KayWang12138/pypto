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
    padding = params[2]
    val = params[3]

    return torch.nn.functional.pad(a, padding, mode='constant', value=val)


class PadTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
 
        n_in, m_in = self.tiling * 1, self.tiling * 1
        a_tensor = torch.rand(n_in, m_in, dtype=torch.float32) * 10

        self.setup_inputs(a_tensor)
        
        self.set_tol(rtol=1e-3, atol=1e-3)
        return (a_tensor, )


def test():
    params = ((16, 16), (8, 8), (0, 4, 0, 4), 0)
    st = PadTest(params, op_pad, op_pad_golden, tiling=16)
    st()
    print("Test Passed!")

if __name__ == "__main__":
    test()