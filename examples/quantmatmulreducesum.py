import pypto
import torch
import torch.nn.functional as F
from numpy.testing import assert_allclose

def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data

def quant_matmul_reduce_sum_pypto(batch, m, k, n):
    x1_shape = [batch, m, k]
    x2_shape = [batch, k, n]
    x1_scale_shape = [batch, m]
    x2_scale_shape = [n]
    out_shape = [m, n]
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1})
    def quant_matmul_reduce_sum_impl(
        x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
        x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
        x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
        x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16)
    ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
        pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
        pypto.set_vec_tile_shapes(1, 256, 256)
        out_tensor = pypto.Tensor([batch, m, n], pypto.DT_BF16)
        
        for idx in range(batch):
            x1_view = x1[idx]
            x2_view = x2[idx]

            x1_scale_view = x1_scale[idx]
            x1_scale_2d = pypto.unsqueeze(x1_scale_view, 1)
            x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [m, n])

            x2_scale_2d = pypto.unsqueeze(x2_scale, 0)
            x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])
            x2_scale_broadcast_fp32 = pypto.cast(x2_scale_broadcast, pypto.DT_FP32)

            extend_params = {'scale': 1.0}
            matmul_result = pypto.matmul(x1_view, x2_view, pypto.DT_FP16, extend_params=extend_params)
            matmul_result = pypto.cast(matmul_result, pypto.DT_FP32)
            scale_mul = pypto.mul(x1_scale_broadcast, x2_scale_broadcast_fp32)
            out = pypto.mul(matmul_result, scale_mul)
            out_tensor[idx, :, :] = pypto.cast(out, pypto.DT_BF16)
        
        print("out_tensor shape:",out_tensor.shape)
        out_bf16 = pypto.sum(out_tensor, 0)
        print("out_bf16 shape:",out_bf16.shape)
        return out_bf16
    return quant_matmul_reduce_sum_impl

def golden_quant_matmul_reduce_sum(x1, x2, x1_scale, x2_scale):
    batch, m, k = x1.shape
    _, _, n = x2.shape
    
    result = []    
    for i in range(batch):
        matmul_result = torch.matmul(x1[i].float(), x2[i].float())
        x1_scale_tmp = x1_scale[i].unsqueeze(1)
        x2_scale_tmp = x2_scale.unsqueeze(0)
        scale_broadcast = x1_scale_tmp.float() * x2_scale_tmp.float()
        result.append(matmul_result * scale_broadcast)
    
    result = torch.stack(result)
    return result.to(torch.bfloat16)       

def test_quant_matmul_reduce_sum_pypto():
    b, m, k, n = (5, 16, 32, 32)
    
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
    x2 = trans_nd_to_fractal_nz(x2_nd).npu()
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0, 2).npu()
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0, 2).npu()
    
    pypto_out = quant_matmul_reduce_sum_pypto(b, m, k, n)(x1, x2, x1_scale, x2_scale)
    
    golden_out = golden_quant_matmul_reduce_sum(x1.cpu(), x2_nd.cpu(), x1_scale.cpu(), x2_scale.cpu())
    print("golden_out shape:",golden_out.shape)
    golden_out = torch.sum(golden_out, dim=0)
    print("golden_out shape:",golden_out.shape)
    
    assert_allclose(pypto_out.cpu().float(), golden_out.cpu().float(), rtol=0.001, atol=0.001)
    print("Test passed!")

if __name__ == "__main__":
    test_quant_matmul_reduce_sum_pypto()



# import pypto
# import torch
# import torch.nn.functional as F
# from numpy.testing import assert_allclose

# def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
#     def _gen_axes_for_transpose(offset, base):
#         return [x for x in range(offset)] + [x + offset for x in base]

#     def _ceil_div(a, b):
#         return (a + b - 1) // b

#     ori_shape = data.shape
#     m_ori, n_ori = ori_shape[-2:]
#     batch_ori = ori_shape[:-2]
#     batch_num = len(batch_ori)
#     m0 = 16
#     n0 = 32 // data.dtype.itemsize
#     if data.dtype == torch.int32:
#         n0 = 16
#     m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
#     padding_m = m1 * m0 - m_ori
#     padding_n = n1 * n0 - n_ori
#     if not keep_m_dim:
#         pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
#         data = F.pad(data, pad_list, "constant")
#         array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
#         data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
#     else:
#         pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
#         data = F.pad(data, pad_list, "constant")
#         array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
#         data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
#     return data

# def quant_matmul_reduce_sum_pypto(m, k, n):
#     x1_shape = [m, k]
#     x2_shape = [k, n]
#     x1_scale_shape = [m]
#     x2_scale_shape = [n]
#     out_shape = [m, n]
#     @pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1})
#     def quant_matmul_reduce_sum_impl(
#         x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
#         x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
#         x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
#         x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16)
#     ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
#         pypto.set_cube_tile_shapes([128, 128], [256, 256], [256, 256])
#         pypto.set_vec_tile_shapes(1, 256, 256)
#         out_tensor = pypto.Tensor(out_shape, pypto.DT_FP32)
        
#         x1_view = x1
#         x2_view = x2

#         x1_scale_view = x1_scale
#         x1_scale_2d = pypto.unsqueeze(x1_scale_view, 1)
#         x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [m, n])

#         x2_scale_2d = pypto.unsqueeze(x2_scale, 0)
#         x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])
#         x2_scale_broadcast_fp32 = pypto.cast(x2_scale_broadcast, pypto.DT_FP32)

#         extend_params = {'scale': 1.0}
#         matmul_result = pypto.matmul(x1, x2, pypto.DT_FP16, extend_params=extend_params)
#         matmul_result = pypto.cast(matmul_result, pypto.DT_FP32)
#         scale_mul = pypto.mul(x1_scale_broadcast, x2_scale_broadcast_fp32)
#         out_tensor = pypto.mul(matmul_result, scale_mul)
#         out_bf16 = pypto.cast(out_tensor, pypto.DT_BF16)
#         return out_bf16
#     return quant_matmul_reduce_sum_impl

# def golden_quant_matmul_reduce_sum(x1, x2, x1_scale, x2_scale):
#     batch, m, k = x1.shape
#     _, _, n = x2.shape
    
#     result = []    
#     for i in range(batch):
#         matmul_result = torch.matmul(x1[i].float(), x2[i].float())
#         x1_scale_tmp = x1_scale[i].unsqueeze(1)
#         x2_scale_tmp = x2_scale.unsqueeze(0)
#         scale_broadcast = x1_scale_tmp.float() * x2_scale_tmp.float()
#         result.append(matmul_result * scale_broadcast)
    
#     result = torch.stack(result)
#     return result.to(torch.bfloat16)

# def test_quant_matmul_reduce_sum_pypto():
#     b, m, k, n = (5, 16, 128, 128)
    
#     x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
#     x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
#     x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0, 2).npu()
#     x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0, 2).npu()
    
#     out_list = []
#     for idx in range(b):
#         x1_tmp = x1[idx]
#         x2_tmp = x2_nd[idx]
#         x1_scale_tmp = x1_scale[idx]

#         x2_tmp = trans_nd_to_fractal_nz(x2_tmp, True).npu()
#         pypto_out = quant_matmul_reduce_sum_pypto(m, k, n)(x1_tmp, x2_tmp, x1_scale_tmp, x2_scale)
#         out_list.append(pypto_out)

#     out = torch.stack(out_list)
#     print("out shape:",out.shape)
#     out = torch.sum(out, dim=0)
#     print("out shape:",out.shape)
    
#     golden_out = golden_quant_matmul_reduce_sum(x1.cpu(), x2_nd.cpu(), x1_scale.cpu(), x2_scale.cpu())
#     print("golden_out shape:",golden_out.shape)
#     golden_out = torch.sum(golden_out, dim=0)
#     print("golden_out shape:",golden_out.shape)
    
#     print("diff sample:", (out.cpu().float() - golden_out.cpu().float()))
    
#     assert_allclose(out.cpu().float(), golden_out.cpu().float(), rtol=0.001, atol=0.001)
#     print("Test passed!")

# if __name__ == "__main__":
#     test_quant_matmul_reduce_sum_pypto()

