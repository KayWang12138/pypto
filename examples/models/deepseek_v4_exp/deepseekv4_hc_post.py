'''
'''
import os
from pathlib import Path
import pytest
import logging
import math
import time
import torch
import torch_npu
import pypto

from mla_prolog_quant_impl import mla_prolog_quant_p, mla_prolog_quant_d, MlaTileConfig
from utils.compare import compare
from hc_post_impl import hc_post_torch_graph, HcPostTileConfig

torch.manual_seed(5)


class HP(torch.nn.Module):
    def forward(self, x, residual, post, comb, y, tile_config):
        hc_post_torch_graph(x, residual, post, comb, y, tile_config)

def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(5)
    torch_npu.npu.config.allow_internal_format = True

def convert_torch_tensor(tensor_dict, dynamic_axis_dict, name_prefix):
    dynamic_count = 0
    pypto_tensors = []
    for name, tensor in tensor_dict.items():
        if name in dynamic_axis_dict.keys():
            dynamic_axis = dynamic_axis_dict[name]
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name, dynamic_axis=dynamic_axis))
            dynamic_count += 1
        else:
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name))
    assert dynamic_count == len(dynamic_axis_dict)
    return pypto_tensors

def gen_hc_post_input_data(params, dtypes):
    x_dtype = dtypes
    logging.debug(f"gen_hc_post_input_data  dtype:{x_dtype}")
    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    hc = params.get("hc")  # s2=4k
    d = params.get("d")
    
    x_shape = [b * s, d]
    residual_shape = [b * s, hc, d]
    post_shape = [b * s, hc]
    comb_shape = [b * s, hc, hc]
    
    x = torch.empty(x_shape, dtype=x_dtype).uniform_(-1, 1)
    residual = torch.empty(residual_shape, dtype=torch.float32).uniform_(-1, 1)
    post = torch.empty(post_shape, dtype=torch.float32).uniform_(-1, 1)
    comb = torch.empty(comb_shape, dtype=torch.float32).uniform_(-1, 1)
    return x, residual, post, comb

def hc_post_commpute(input_tensors, params):
    x, residual, post, comb = input_tensors

    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    hc = params.get("hc")  # s2=4k
    d = params.get("d")

    post_reshape = post.reshape(b * s, hc, 1)
    x_reshape = x.reshape(b * s, 1, d).to(torch.float32)
    comb_reshape = comb.reshape(b * s, hc, hc, 1)
    residual_reshape = residual.reshape(b * s, hc, 1, d)
    
    y_shape = [b * s, hc, d]

    y = torch.empty(y_shape, dtype=x.dtype)
    post_res = post_reshape * x_reshape
    residual_res = comb_reshape * residual_reshape
    residual_reduce = torch.sum(residual_res, 1)
    y = torch.add(post_res, residual_reduce).to(x.dtype)
    return y


def test_b4_s64k2_nd_bf16_hc_post():
    '''
    hc post测试函数
    '''
    prep_env()
    params = {
        'b': 4,
        's': 128,
        'hc': 4,
        'd': 4096,
    }
    b = params.get('b')
    s = params.get('s')
    hc = params.get('hc')
    d = params.get('d')
    dtypes = torch.bfloat16
    x, residual, post, comb = gen_hc_post_input_data(params, dtypes)
    input_tensors = [x, residual, post, comb]
    y = hc_post_commpute(input_tensors, params)
    tile_config = HcPostTileConfig()

    y_out_shape = [b * s, hc, d]
    y_out = torch.empty(y_out_shape, dtype=torch.bfloat16).npu()

    hc_post_torch_graph(x.npu(), residual.npu(), post.npu(), comb.npu(), y_out, tile_config)
    compare(y_out.cpu(), y.cpu(), 'y', 0.0001, 0.0078125,
            0.005)


def test_b4_s64k2_nd_bf16_hc_post_graph():
    '''
    hc post测试函数
    '''
    prep_env()
    params = {
        'b': 4,
        's': 128,
        'hc': 4,
        'd': 4096,
    }
    b = params.get('b')
    s = params.get('s')
    hc = params.get('hc')
    d = params.get('d')
    dtypes = torch.bfloat16
    x, residual, post, comb = gen_hc_post_input_data(params, dtypes)
    input_tensors = [x, residual, post, comb]
    y = hc_post_commpute(input_tensors, params)
    tile_config = HcPostTileConfig()

    y_out_shape = [b * s, hc, d]
    y_out = torch.empty(y_out_shape, dtype=torch.bfloat16).npu()

    model = torch.compile(HP(), backend="eager", dynamic=True)
    
    x_npu = x.npu()
    residual_npu = residual.npu()
    post_npu = post.npu()
    comb_npu = comb.npu()
    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        model(x_npu, residual_npu, post_npu, comb_npu, y_out, tile_config)
    
    g.replay()
    compare(y_out.cpu(), y.cpu(), 'y', 0.0001, 0.0078125,
            0.005)


def test_b4_s64k2_nd_bf16_hc_post_npu_graph():
    '''
    hc post测试函数
    '''
    prep_env()
    params = {
        'b': 4,
        's': 128,
        'hc': 4,
        'd': 4096,
    }
    b = params.get('b')
    s = params.get('s')
    hc = params.get('hc')
    d = params.get('d')
    dtypes = torch.bfloat16
    x, residual, post, comb = gen_hc_post_input_data(params, dtypes)
    input_tensors = [x, residual, post, comb]
    y = hc_post_commpute(input_tensors, params)
    tile_config = HcPostTileConfig()

    y_out_shape = [b * s, hc, d]
    y_out = torch.empty(y_out_shape, dtype=torch.bfloat16).npu()
    
    x_npu = x.npu()
    residual_npu = residual.npu()
    post_npu = post.npu()
    comb_npu = comb.npu()
    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        hc_post_torch_graph(x_npu, residual_npu, post_npu, comb_npu, y_out, tile_config)
    
    g.replay()
    compare(y_out.cpu(), y.cpu(), 'y', 0.0001, 0.0078125,
            0.005)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )

    test_b4_s64k2_nd_bf16_hc_post()