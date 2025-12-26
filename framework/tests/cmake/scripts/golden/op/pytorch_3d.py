#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
""" pythorch3d 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import os
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

import compare

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

def softmax_rgb_blend_np(
    mask: np.ndarray,
    colors: np.ndarray,
    fragments_pix_to_face: np.ndarray,
    fragments_dists: np.ndarray,
    fragments_zbuf: np.ndarray,
    blend_params_sigma: np.ndarray = np.array([1e-4]),
    blend_params_gamma: np.ndarray = np.array([1e-4]),
    background_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
    znear: np.ndarray = np.array([1.0]),
    zfar: np.ndarray = np.array([100.0]),
) -> np.ndarray:

    print("softmax_rgb_blend_np")

    eps = 1e-10
    # mask = np.greater_equal(fragments_pix_to_face, 0)
    fragments_dists_f = np.multiply(fragments_dists, -1)

    sigmoid_in = np.divide(fragments_dists_f, blend_params_sigma)

    sigmoid_res = 1 / (1 + np.exp(-sigmoid_in))

    prob_map = np.multiply(sigmoid_res, mask)

    alpha = np.prod(np.subtract(1.0, prob_map), axis=-1, keepdims=True)
    # alpha = np.max(np.subtract(1.0, prob_map), axis=-1, keepdims=True)

    z_inv = np.multiply(np.divide(np.subtract(zfar, fragments_zbuf), np.subtract(zfar, znear)), mask)

    z_inv_max_ori = np.max(z_inv, axis=-1, keepdims=True)
    z_inv_max = np.clip(z_inv_max_ori, a_min=eps, a_max=None)
    # z_inv_max = z_inv_max_ori

    weights_num = np.multiply(prob_map, np.exp(np.divide(np.subtract(z_inv, z_inv_max), blend_params_gamma)))

    delta_ori = np.exp(np.divide(np.subtract(eps, z_inv_max), blend_params_gamma))
    delta = np.clip(delta_ori, a_min=eps, a_max=None)
    # delta = delta_ori

    denom = np.add(np.sum(weights_num, axis=-1, keepdims=True), delta)

    weighted_background = np.multiply(delta, background_color)

    weighted_colors = np.sum(np.multiply(np.expand_dims(weights_num, axis=-1), colors), axis=-2)

    color_rgb = np.divide(np.add(weighted_colors, weighted_background), denom)
    color_alpha = np.subtract(1.0, alpha)

    pixel_colors = np.concatenate([color_rgb, color_alpha], axis=-1)

    return pixel_colors

def softmax_rgb_blend_torch(
    mask_in: torch.Tensor,
    colors: torch.Tensor,
    fragments_pix_to_face: torch.Tensor, 
    fragments_dists: torch.Tensor, 
    fragments_zbuf: torch.Tensor, 
    blend_params_sigma: torch.Tensor = (1e-4),
    blend_params_gamma: torch.Tensor = (1e-4),
    background_color: torch.Tensor =  (1.0, 1.0, 1.0),
    znear: torch.Tensor = 1.0,
    zfar: torch.Tensor = 100,
) -> torch.Tensor:

    print("softmax_rgb_blend_torch")

    eps = 1e-10
    # mask = mask_in #torch.greater_equal(fragments_pix_to_face, 0)       # [N,H,W,K]
    mask = torch.greater_equal(fragments_pix_to_face, 0)       # [N,H,W,K]

    fragments_dists_f = torch.mul(fragments_dists, -1)        # [N,H,W,K]

    sigmoid_in = torch.div(fragments_dists_f , blend_params_sigma) # [N,H,W,K]

    sigmoid_res = torch.sigmoid(sigmoid_in)   # [N,H,W,K]

    prob_map = torch.mul(sigmoid_res, mask)   # [N,H,W,K]

    alpha = torch.prod(torch.sub(1.0, prob_map), dim=-1, keepdim=True)       # [N,H,W,1]
    # alpha = torch.max(torch.sub(1.0, prob_map), dim=-1, keepdim=True).values       # [N,H,W,1]

    # z_inv = (zfar - fragments_zbuf) / (zfar - znear) * mask
    z_inv = torch.mul(torch.div(torch.sub(zfar, fragments_zbuf), torch.sub(zfar, znear)), mask)  # [N,H,W,K]

    # z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    z_inv_max_ori = torch.max(z_inv, dim=-1, keepdim=True).values    # [N,H,W,K] -> [N,H,W,1]   
    z_inv_max = torch.clamp(z_inv_max_ori, min=eps)                  # [N,H,W,1]
    # z_inv_max = z_inv_max_ori

    # weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
    weights_num = torch.mul(prob_map, torch.exp(torch.div(torch.sub(z_inv , z_inv_max), blend_params_gamma)))  # [N,H,W,K]

    # delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)
    delta_ori = torch.exp(torch.div(torch.sub(eps, z_inv_max), blend_params_gamma))
    delta = torch.clamp(delta_ori, min=eps)        # [N,H,W,1]
    # delta = delta_ori 

    # denom = weights_num.sum(dim=-1)[..., None] + delta
    denom = torch.add(torch.sum(weights_num, dim=-1, keepdim=True) , delta)    # [N,H,W,1]

    weighted_background = torch.mul(delta, background_color)  #[N,H,W,1]  * [3] = [N,H,W,3] 

    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)    
    weighted_colors = torch.sum(torch.mul(torch.unsqueeze(weights_num, dim = -1), colors), dim=-2)        # [N,H,W,3]
    # pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
    color_rgb = torch.div(torch.add(weighted_colors, weighted_background), denom)   #[N,H,W,3] 
    color_alpha = torch.sub(1.0, alpha) #[N,H,W,1] 
    pixel_colors = torch.concat([color_rgb, color_alpha], dim = -1)   #[N,H,W,4] 

    return pixel_colors

def operation_softmax_rgb_blend2_1_16_128_4(output_dir: Path):
    color_path = Path(output_dir, 'color.bin')
    mask_path = Path(output_dir, 'mask.bin')

    pix_to_face_path = Path(output_dir, 'pix_to_face.bin')
    zbuf_path = Path(output_dir,  'zbuf.bin')
    dists_path = Path(output_dir, 'dists.bin')
    background_color_path = Path(output_dir, 'background_color.bin')
    images_path = Path(output_dir, 'images.bin')
    images2_path = Path(output_dir, 'images2.bin')


    sigma_path = Path(output_dir, 'sigma.bin')
    gamma_path = Path(output_dir, 'gamma.bin')
    znear_path = Path(output_dir, 'znear.bin')
    zfar_path = Path(output_dir, 'zfar.bin')
    sigma = np.array([1e-4], dtype=np.float32)
    sigma.tofile(sigma_path)
    gamma = np.array([1e-4], dtype=np.float32)
    gamma.tofile(gamma_path)
    znear = np.array([1.0], dtype=np.float32)
    znear.tofile(znear_path)
    zfar = np.array([100.0], dtype=np.float32)
    zfar.tofile(zfar_path)

    N = 1
    H = 16
    W = 128
    K = 4

    mask = np.random.uniform(0, 1, [N, H, W, K]).astype(np.bool8).astype(np.float32)
    mask.tofile(mask_path)

    a = np.random.uniform(0, 0.999, [N, H, W, K, 3]).astype(np.float32)
    color_mask = np.reshape(mask, [N, H, W, K, 1])
    color = np.multiply(a, color_mask)
    color.tofile(color_path)

    x = np.random.uniform(1, 10000, [N, H, W, K]).astype(np.int32)
    pix_to_face = np.add(np.multiply(x, mask), -1)
    pix_to_face.tofile(pix_to_face_path)

    y = np.random.uniform(1, 4, [N, H, W, K]).astype(np.float32)
    zbuf = np.multiply(y, mask)
    zbuf.tofile(zbuf_path)

    z = np.random.uniform(0.003, 0.0000005, [N, H, W, K]).astype(np.float32)
    dists = np.multiply(z, mask)
    dists.tofile(dists_path)

    background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    background_color.tofile(background_color_path)

    images = softmax_rgb_blend_np(colors = color, fragments_pix_to_face = pix_to_face,
                                  fragments_zbuf = zbuf, fragments_dists = dists,
                                  background_color = background_color, mask = mask,
                                  blend_params_sigma = sigma, blend_params_gamma = gamma, znear = znear, zfar = zfar)

    images.astype('float32').tofile(images_path)


    images2 = softmax_rgb_blend_torch(colors = torch.tensor(color), fragments_pix_to_face = torch.tensor(pix_to_face),
                                  fragments_zbuf = torch.tensor(zbuf), fragments_dists = torch.tensor(dists),
                                  background_color = torch.tensor(background_color), mask_in = torch.tensor(mask),
                                  blend_params_sigma = torch.tensor(sigma), blend_params_gamma =  torch.tensor(gamma), 
                                  znear =  torch.tensor(znear), zfar =  torch.tensor(zfar))
    images2.numpy().tofile(images2_path)

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np
    result, fulfill_percent, max_error = data_compare_np(images, images2.numpy(), rtol, atol, is_display=True)

    print(result, fulfill_percent, max_error)

def operation_softmax_rgb_blend2_1_16_128_8(output_dir: Path):
    color_path = Path(output_dir, 'color.bin')
    mask_path = Path(output_dir, 'mask.bin')

    pix_to_face_path = Path(output_dir, 'pix_to_face.bin')
    zbuf_path = Path(output_dir,  'zbuf.bin')
    dists_path = Path(output_dir, 'dists.bin')
    background_color_path = Path(output_dir, 'background_color.bin')
    images_path = Path(output_dir, 'images.bin')
    images2_path = Path(output_dir, 'images2.bin')


    sigma_path = Path(output_dir, 'sigma.bin')
    gamma_path = Path(output_dir, 'gamma.bin')
    znear_path = Path(output_dir, 'znear.bin')
    zfar_path = Path(output_dir, 'zfar.bin')
    sigma = np.array([1e-4], dtype=np.float32)
    sigma.tofile(sigma_path)
    gamma = np.array([1e-4], dtype=np.float32)
    gamma.tofile(gamma_path)
    znear = np.array([1.0], dtype=np.float32)
    znear.tofile(znear_path)
    zfar = np.array([100.0], dtype=np.float32)
    zfar.tofile(zfar_path)

    N = 1
    H = 16
    W = 128
    K = 8

    mask = np.random.uniform(0, 1, [N, H, W, K]).astype(np.bool8).astype(np.float32)
    mask.tofile(mask_path)

    a = np.random.uniform(0, 0.999, [N, H, W, K, 3]).astype(np.float32)
    color_mask = np.reshape(mask, [N, H, W, K, 1])
    color = np.multiply(a, color_mask)
    color.tofile(color_path)

    x = np.random.uniform(1, 10000, [N, H, W, K]).astype(np.int32)
    pix_to_face = np.add(np.multiply(x, mask), -1)
    pix_to_face.tofile(pix_to_face_path)

    y = np.random.uniform(1, 4, [N, H, W, K]).astype(np.float32)
    zbuf = np.multiply(y, mask)
    zbuf.tofile(zbuf_path)

    z = np.random.uniform(0.003, 0.0000005, [N, H, W, K]).astype(np.float32)
    dists = np.multiply(z, mask)
    dists.tofile(dists_path)

    background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    background_color.tofile(background_color_path)

    images = softmax_rgb_blend_np(colors = color, fragments_pix_to_face = pix_to_face,
                                  fragments_zbuf = zbuf, fragments_dists = dists,
                                  background_color = background_color, mask = mask,
                                  blend_params_sigma = sigma, blend_params_gamma = gamma, znear = znear, zfar = zfar)

    images.astype('float32').tofile(images_path)


    images2 = softmax_rgb_blend_torch(colors = torch.tensor(color), fragments_pix_to_face = torch.tensor(pix_to_face),
                                  fragments_zbuf = torch.tensor(zbuf), fragments_dists = torch.tensor(dists),
                                  background_color = torch.tensor(background_color), mask_in = torch.tensor(mask),
                                  blend_params_sigma = torch.tensor(sigma), blend_params_gamma =  torch.tensor(gamma), 
                                  znear =  torch.tensor(znear), zfar =  torch.tensor(zfar))
    images2.numpy().tofile(images2_path)

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np
    result, fulfill_percent, max_error = data_compare_np(images, images2.numpy(), rtol, atol, is_display=True)

    print(result, fulfill_percent, max_error)

def operation_softmax_rgb_blend2_1_128_128_2(output_dir: Path):
    color_path = Path(output_dir, 'color.bin')
    mask_path = Path(output_dir, 'mask.bin')

    pix_to_face_path = Path(output_dir, 'pix_to_face.bin')
    zbuf_path = Path(output_dir,  'zbuf.bin')
    dists_path = Path(output_dir, 'dists.bin')
    background_color_path = Path(output_dir, 'background_color.bin')
    images_path = Path(output_dir, 'images.bin')
    images_torch_path = Path(output_dir, 'images_torch.bin')


    sigma_path = Path(output_dir, 'sigma.bin')
    gamma_path = Path(output_dir, 'gamma.bin')
    znear_path = Path(output_dir, 'znear.bin')
    zfar_path = Path(output_dir, 'zfar.bin')
    sigma = np.array([1e-4], dtype=np.float32)
    sigma.tofile(sigma_path)
    gamma = np.array([1e-4], dtype=np.float32)
    gamma.tofile(gamma_path)
    znear = np.array([1.0], dtype=np.float32)
    znear.tofile(znear_path)
    zfar = np.array([100.0], dtype=np.float32)
    zfar.tofile(zfar_path)

    N = 1
    H = 128
    W = 128
    K = 2

    mask = np.random.uniform(0, 1, [N, H, W, K]).astype(np.bool).astype(np.float32)
    mask.tofile(mask_path)

    a = np.random.uniform(0, 0.999, [N, H, W, K, 3]).astype(np.float32)
    color_mask = np.reshape(mask, [N, H, W, K, 1])

    color = np.multiply(a, color_mask)
    color.tofile(color_path)

    x = np.random.uniform(1, 10000, [N, H, W, K]).astype(np.int32)
    pix_to_face = np.add(np.multiply(x, mask), -1)
    pix_to_face.tofile(pix_to_face_path)

    y = np.random.uniform(1, 4, [N, H, W, K]).astype(np.float32)
    zbuf = np.multiply(y, mask)
    zbuf.tofile(zbuf_path)

    z = np.random.uniform(0.003, 0.0000005, [N, H, W, K]).astype(np.float32)
    dists = np.multiply(z, mask)
    dists.tofile(dists_path)

    background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    background_color.tofile(background_color_path)

    images = softmax_rgb_blend_np(colors = color, fragments_pix_to_face = pix_to_face,
                                  fragments_zbuf = zbuf, fragments_dists = dists,
                                  background_color = background_color, mask = mask,
                                  blend_params_sigma = sigma, blend_params_gamma = gamma, znear = znear, zfar = zfar)

    images.astype('float32').tofile(images_path)


    images_torch = softmax_rgb_blend_torch(colors = torch.tensor(color), fragments_pix_to_face = torch.tensor(pix_to_face),
                                  fragments_zbuf = torch.tensor(zbuf), fragments_dists = torch.tensor(dists),
                                  background_color = torch.tensor(background_color), mask_in = torch.tensor(mask),
                                  blend_params_sigma = torch.tensor(sigma), blend_params_gamma =  torch.tensor(gamma), 
                                  znear =  torch.tensor(znear), zfar =  torch.tensor(zfar))
    images_torch.numpy().tofile(images_torch_path)

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np
    result, fulfill_percent, max_error = data_compare_np(images, images_torch.numpy(), rtol, atol, is_display=True)

    print(result, fulfill_percent, max_error)


def operation_softmax_rgb_blend2_large(output_dir: Path):
    color_path = Path(output_dir, 'color.bin')
    mask_path = Path(output_dir, 'mask.bin')

    pix_to_face_path = Path(output_dir, 'pix_to_face.bin')
    zbuf_path = Path(output_dir,  'zbuf.bin')
    dists_path = Path(output_dir, 'dists.bin')
    background_color_path = Path(output_dir, 'background_color.bin')
    images_path = Path(output_dir, 'images.bin')
    images2_path = Path(output_dir, 'images2.bin')


    sigma_path = Path(output_dir, 'sigma.bin')
    gamma_path = Path(output_dir, 'gamma.bin')
    znear_path = Path(output_dir, 'znear.bin')
    zfar_path = Path(output_dir, 'zfar.bin')
    sigma = np.array([1e-4], dtype=np.float32)
    sigma.tofile(sigma_path)
    gamma = np.array([1e-4], dtype=np.float32)
    gamma.tofile(gamma_path)
    znear = np.array([1.0], dtype=np.float32)
    znear.tofile(znear_path)
    zfar = np.array([100.0], dtype=np.float32)
    zfar.tofile(zfar_path)

    N = 1
    H = 1024
    W = 1024
    K = 2

    mask = np.random.uniform(0, 1, [N, H, W, K]).astype(np.bool8).astype(np.float32)
    mask.tofile(mask_path)

    a = np.random.uniform(0, 0.999, [N, H, W, K, 3]).astype(np.float32)
    color_mask = np.reshape(mask, [N, H, W, K, 1])
    color = np.multiply(a, color_mask)
    color.tofile(color_path)

    x = np.random.uniform(1, 10000, [N, H, W, K]).astype(np.int32)
    pix_to_face = np.add(np.multiply(x, mask), -1)
    pix_to_face.tofile(pix_to_face_path)

    y = np.random.uniform(1, 4, [N, H, W, K]).astype(np.float32)
    zbuf = np.multiply(y, mask)
    zbuf.tofile(zbuf_path)

    z = np.random.uniform(0.003, 0.0000005, [N, H, W, K]).astype(np.float32)
    dists = np.multiply(z, mask)
    dists.tofile(dists_path)

    background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    background_color.tofile(background_color_path)

    images = softmax_rgb_blend_np(colors = color, fragments_pix_to_face = pix_to_face,
                                  fragments_zbuf = zbuf, fragments_dists = dists,
                                  background_color = background_color, mask = mask,
                                  blend_params_sigma = sigma, blend_params_gamma = gamma, znear = znear, zfar = zfar)

    images.astype('float32').tofile(images_path)


    images2 = softmax_rgb_blend_torch(colors = torch.tensor(color), fragments_pix_to_face = torch.tensor(pix_to_face),
                                  fragments_zbuf = torch.tensor(zbuf), fragments_dists = torch.tensor(dists),
                                  background_color = torch.tensor(background_color), mask_in = torch.tensor(mask),
                                  blend_params_sigma = torch.tensor(sigma), blend_params_gamma =  torch.tensor(gamma), 
                                  znear =  torch.tensor(znear), zfar =  torch.tensor(zfar))
    images2.numpy().tofile(images2_path)

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np
    result, fulfill_percent, max_error = data_compare_np(images, images2.numpy(), rtol, atol, is_display=True)


    print(result, fulfill_percent, max_error)


def operation_cow_data(output_dir: Path):
    color_path = Path(output_dir, 'g/color.bin')

    pix_to_face_path = Path(output_dir, 'g/pix_to_face.bin')
    zbuf_path = Path(output_dir,  'g/zbuf.bin')
    dists_path = Path(output_dir, 'g/dists.bin')
    png_path = Path(output_dir,   'g/test.png')
    mask_path = Path(output_dir,  'g/mask.bin')

    sigma_path = Path(output_dir, 'g/sigma.bin')
    gamma_path = Path(output_dir, 'g/gamma.bin')
    znear_path = Path(output_dir, 'g/znear.bin')
    zfar_path = Path(output_dir,  'g/zfar.bin')
    background_color_path = Path(output_dir, 'g/background_color.bin')

    image_golden_path = Path(output_dir, 'g/images_origin.bin')


    sigma = np.array([1e-4], dtype=np.float32)
    sigma.tofile(sigma_path)
    gamma = np.array([1e-4], dtype=np.float32)
    gamma.tofile(gamma_path)
    znear = np.array([1.0], dtype=np.float32)
    znear.tofile(znear_path)
    zfar = np.array([100.0], dtype=np.float32)
    zfar.tofile(zfar_path)
    N = 1
    H = 128
    W = 128
    K = 2

    color = np.fromfile(color_path, dtype= np.float32).reshape([N, H, W, K, 3])
    # color = color_ori.transpose(0,1,2,4,3)
    print("color.shape: ", color.shape)

    pix_to_face = np.fromfile(pix_to_face_path, dtype= np.int32).reshape([N, H, W, K])
    zbuf  = np.fromfile(zbuf_path, dtype= np.float32).reshape([N, H, W, K])
    dists = np.fromfile(dists_path, dtype= np.float32).reshape([N, H, W, K])

    mask = np.greater_equal(pix_to_face, 0).astype(np.float32)
    mask.tofile(mask_path)
    # print(mask)

    background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    background_color.tofile(background_color_path)

    images3 = softmax_rgb_blend_torch(colors = torch.tensor(color), fragments_pix_to_face = torch.tensor(pix_to_face),
                                  fragments_zbuf = torch.tensor(zbuf), fragments_dists = torch.tensor(dists),
                                  background_color = torch.tensor(background_color), mask_in = torch.tensor(mask),
                                  blend_params_sigma = torch.tensor(sigma), blend_params_gamma =  torch.tensor(gamma), 
                                  znear =  torch.tensor(znear), zfar =  torch.tensor(zfar))

    print("images3.shape: ", images3.shape)
    images3_path = Path(output_dir, 'g/images2.bin')

    images3.numpy().tofile(images3_path)
    from PIL import Image
    Image.fromarray((images3[0, ..., :3].squeeze().numpy() * 255).astype(np.uint8)).save(png_path)

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np
    result, fulfill_percent, max_error = data_compare_np(np.fromfile(image_golden_path, dtype= np.float32), images3.numpy(), rtol, atol, is_display=True)
    print(result, fulfill_percent, max_error)


def operation_operation_prod(output_dir: Path):
    S0 = 128
    S1 = 23
    prod_x_path = Path(output_dir, 'prod_x.bin')
    prod_res_path = Path(output_dir, 'prod_res.bin')

    prod_x = np.random.uniform(1, 2, [S0, S1]).astype(np.float32)
    prod_x.tofile(prod_x_path)

    prod_res = np.prod(prod_x , -1)
    prod_res.tofile(prod_res_path)

    # print(prod_res)


def operation_operation_expand(output_dir: Path):
    S0 = 1024
    S1 = 8
    S2 = 3

    expand_x_path = Path(output_dir, 'x.bin')
    expand_res_path = Path(output_dir, 'res.bin')

    expand_x = np.random.uniform(1, 2, [S0, S1]).astype(np.float32)
    expand_x.tofile(expand_x_path)

    expand_res = torch.unsqueeze(torch.tensor(expand_x), -1).expand(-1, -1, S2)
    expand_res.numpy().tofile(expand_res_path)

from dataclasses import dataclass
from typing import Tuple
from enum import Enum, auto
import math
import torch.nn.functional as F

INF_Z = math.inf

class FrontFace(Enum):
    CCW = auto()
    CW = auto()

@dataclass
class RenderSettings:
    image_size: Tuple[int, int] = (256, 256)
    max_blend_depth: int = 8
    bin_size: int = 64
    blur_radius_ndc: float = 0
    clip_barycentric_coords: bool = False
    cull_backfaces: bool = False
    front_face: FrontFace = FrontFace.CCW
    persp_correct: bool = False
    output_dir: Path = None
    need_dump: bool = False
    dump_binx: int = 0
    dump_biny: int = 0

    def front_face_direction(self) -> int:
        return 1 if self.front_face == FrontFace.CCW else -1

    def ndc_to_screen_scale(self) -> float:
        return min(self.image_size[0], self.image_size[1]) * 0.5

    def screen_to_ndc_scale(self) -> float:
        return 1 / self.ndc_to_screen_scale()

    def blur_radius_scr(self) -> float:
        return self.blur_radius_ndc * self.ndc_to_screen_scale()


def _cross2d(
    a: torch.Tensor, # (..., 2)
    b: torch.Tensor, # (..., 2)
) -> torch.Tensor:
    assert a.shape[-1] == b.shape[-1] == 2
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def _edge_function(
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    p = p.unsqueeze(dim=-2) # (#points..., 1, 2)
    return _cross2d(p - a, b - a).sign()

def _edge_function_2(
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    p = p.unsqueeze(dim=-2) # (#points..., 1, 2)
    A = b[..., 1] - a[..., 1] # (#faces,)
    B = a[..., 0] - b[..., 0] # (#faces,)
    C = _cross2d(b, a) # (#faces,)
    return ((torch.stack((A, B), dim=-1) * p).sum(dim=-1) + C).sign()

def _triangle_inside_direction(
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#faces,)
    """
    Do not return 0
    """
    return (_cross2d(c - a, b - a) > 0).int() * 2 - 1

def _internal_is_inside_triangle(
    edge_ab: torch.Tensor,
    edge_bc: torch.Tensor,
    edge_ca: torch.Tensor,
    inside_dir,
) -> torch.Tensor:
    return (edge_ab == inside_dir) & (edge_ab == edge_bc) & (edge_bc == edge_ca)

def is_inside_triangle(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    assert p.dim() >= 2
    assert p.shape[-1] == 2

    assert a.dim() == b.dim() == c.dim() == 2
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 2
    assert a.shape[0] == b.shape[0] == c.shape[0]

    edge_ab = _edge_function_2(p, a, b) # (#points..., #faces)
    edge_bc = _edge_function_2(p, b, c) # (#points..., #faces)
    edge_ca = _edge_function_2(p, c, a) # (#points..., #faces)

    if settings.cull_backfaces:
        inside_dir = settings.front_face_direction()
    else:
        inside_dir = _triangle_inside_direction(a, b, c) # (#faces,)

    return _internal_is_inside_triangle(edge_ab, edge_bc, edge_ca, inside_dir)

def _point_segment_distance_square(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    p_unsq = p.unsqueeze(dim=-2) # (#points..., 1, 2)

    ab = b - a # (#faces, 2)
    t = ((p_unsq - a) * ab).sum(dim=-1) / (ab ** 2).sum(dim=-1) # (#points..., #faces)
    t_clamped = torch.clip(t, 0, 1) # (#points..., #faces)
    projection = a + t_clamped.unsqueeze(dim=-1) * ab # (#points..., #faces, 2)
    d = (projection - p_unsq) * torch.tensor(settings.screen_to_ndc_scale()) # (#points..., #faces, 2)
    return (d ** 2).sum(dim=-1) # (#points..., #faces)


def dump_triangle_signed_squared_distance(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
    dists: torch.Tensor, # (#points..., #faces)
    face_sd2: torch.Tensor, # (#points..., #faces)
):
    if settings.need_dump:
        print("dump distance: bin_",settings.dump_binx, '_', settings.dump_biny," start...................")

        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"
        point_path = "/point_" + bin_x_y_path

        point_path        = Path(settings.output_dir.__fspath__() + "/point_" + bin_x_y_path)
        face_verts_0_path = Path(settings.output_dir.__fspath__() + "/face_verts_0_" + bin_x_y_path)
        face_verts_1_path = Path(settings.output_dir.__fspath__() + "/face_verts_1_" + bin_x_y_path)
        face_verts_2_path = Path(settings.output_dir.__fspath__() + "/face_verts_2_" + bin_x_y_path)
        scale_path        = Path(settings.output_dir.__fspath__() + "/scale_" + bin_x_y_path)
        dists_path        = Path(settings.output_dir.__fspath__() + "/distance_" + bin_x_y_path)
        face_sd2_path        = Path(settings.output_dir.__fspath__() + "/face_sd2_" + bin_x_y_path)

        p.numpy().tofile(point_path)
        a.numpy().tofile(face_verts_0_path)
        b.numpy().tofile(face_verts_1_path)
        c.numpy().tofile(face_verts_2_path)
        scale = torch.tensor(settings.screen_to_ndc_scale())
        scale.numpy().tofile(scale_path)
        dists.numpy().tofile(dists_path)
        face_sd2.numpy().tofile(face_sd2_path)

        face_verts_stack = torch.concat([a.unsqueeze(dim=-2),b.unsqueeze(dim=-2),c.unsqueeze(dim=-2)], dim=-2)

        face_verts_path = Path(settings.output_dir.__fspath__() + "/face_verts_" + bin_x_y_path)
        face_verts = np.fromfile(face_verts_path , dtype= np.float32).reshape([face_verts_stack.shape[0], face_verts_stack.shape[1], 3])
        face_verts_xy = face_verts[:,:,:2]

        atol = 0.001
        rtol = 0.001
        print(f"\n=================================1111111=================================================== output")
        data_compare_np = compare.data_compare_np
        result, fulfill_percent, max_error = data_compare_np(face_verts_stack.numpy(), face_verts_xy, rtol, atol, is_display=False)
        print(result, fulfill_percent, max_error)
        print(f"\n=================================2222222=================================================== output")

        print("point.shape: ", p.shape, " dtype: ", p.dtype)
        print("face_verts_0.shape: ", a.shape, " dtype: ", a.dtype)
        print("face_verts_1.shape: ", b.shape, " dtype: ", b.dtype)
        print("face_verts_2.shape: ", c.shape, " dtype: ", c.dtype)
        print("scale.shape: ", scale.shape, " dtype: ", scale.dtype)
        print("dump distance: bin_",settings.dump_binx, '_', settings.dump_biny," end...................")


def dump_barycentric_coords(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
    bary_coords: torch.Tensor, # (#points..., #faces)
):
    if settings.need_dump:

        print("dump bary_coords: bin_",settings.dump_binx, '_', settings.dump_biny," start...................")

        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"
        point_path = "/point_" + bin_x_y_path

        point_path        = Path(settings.output_dir.__fspath__() + "/point_" + bin_x_y_path)
        face_verts_0_path = Path(settings.output_dir.__fspath__() + "/face_verts_0_" + bin_x_y_path)
        face_verts_1_path = Path(settings.output_dir.__fspath__() + "/face_verts_1_" + bin_x_y_path)
        face_verts_2_path = Path(settings.output_dir.__fspath__() + "/face_verts_2_" + bin_x_y_path)
        bary_coords_path  = Path(settings.output_dir.__fspath__() + "/bary_coords_" + bin_x_y_path)

        p.numpy().tofile(point_path)
        a.numpy().tofile(face_verts_0_path)
        b.numpy().tofile(face_verts_1_path)
        c.numpy().tofile(face_verts_2_path)
        bary_coords.numpy().tofile(bary_coords_path)

        face_verts_stack = torch.concat([a.unsqueeze(dim=-2),b.unsqueeze(dim=-2),c.unsqueeze(dim=-2)], dim=-2)

        face_verts_path = Path(settings.output_dir.__fspath__() + "/face_verts_" + bin_x_y_path)
        face_verts = np.fromfile(face_verts_path , dtype= np.float32).reshape([face_verts_stack.shape[0], face_verts_stack.shape[1], 3])
        face_verts_xy = face_verts[:,:,:2]

        atol = 0.001
        rtol = 0.001
        print(f"\n=================================3333333=================================================== output")
        data_compare_np = compare.data_compare_np
        result, fulfill_percent, max_error = data_compare_np(face_verts_stack.numpy(), face_verts_xy, rtol, atol, is_display=False)
        print(result, fulfill_percent, max_error)
        print(f"\n=================================4444444=================================================== output")


        print("point.shape: ", p.shape, " dtype: ", p.dtype)
        print("face_verts_0.shape: ", a.shape, " dtype: ", a.dtype)
        print("face_verts_1.shape: ", b.shape, " dtype: ", b.dtype)
        print("face_verts_2.shape: ", c.shape, " dtype: ", c.dtype)
        print("bary_coords.shape: ", bary_coords.shape, " dtype: ", bary_coords.dtype)
        print("dump bary_coords: bin_",settings.dump_binx, '_', settings.dump_biny," end...................")


def dump_barycentric_coords_correction(
    settings: RenderSettings,
    bary_nopersp: torch.Tensor, # (#p..., #tri, 3)
    a: torch.Tensor, # (#tri, 3)
    b: torch.Tensor, # (#tri, 3)
    c: torch.Tensor, # (#tri, 3)
    bary_coords_corect: torch.Tensor, # (#points..., #faces)
):
    if settings.need_dump:

        print("dump bary_coords_correction: bin_",settings.dump_binx, '_', settings.dump_biny," start...................")

        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"

        bary_nopersp_path = Path(settings.output_dir.__fspath__() + "/bary_nopersp_" + bin_x_y_path)
        face_verts_0_path = Path(settings.output_dir.__fspath__() + "/face_verts_0_xyz_" + bin_x_y_path)
        face_verts_1_path = Path(settings.output_dir.__fspath__() + "/face_verts_1_xyz_" + bin_x_y_path)
        face_verts_2_path = Path(settings.output_dir.__fspath__() + "/face_verts_2_xyz_" + bin_x_y_path)
        bary_coords_corect_path  = Path(settings.output_dir.__fspath__() + "/bary_coords_corect_" + bin_x_y_path)

        bary_nopersp.numpy().tofile(bary_nopersp_path)
        a.numpy().tofile(face_verts_0_path)
        b.numpy().tofile(face_verts_1_path)
        c.numpy().tofile(face_verts_2_path)
        bary_coords_corect.numpy().tofile(bary_coords_corect_path)

        print("bary_nopersp.shape: ", bary_nopersp.shape, " dtype: ", bary_nopersp.dtype)
        print("face_verts_0.shape: ", a.shape, " dtype: ", a.dtype)
        print("face_verts_1.shape: ", b.shape, " dtype: ", b.dtype)
        print("face_verts_2.shape: ", c.shape, " dtype: ", c.dtype)
        print("bary_coords_corect.shape: ", bary_coords_corect.shape, " dtype: ", bary_coords_corect.dtype)
        print("dump bary_coords_correction: bin_",settings.dump_binx, '_', settings.dump_biny," end...................")

def dump_fine_rasterize(
    settings: RenderSettings,
    pix_to_face: torch.Tensor, 
    zbuf : torch.Tensor, 
    barycentrics: torch.Tensor,
    dists: torch.Tensor,
    index: torch.Tensor):

    if settings.need_dump:
        print("dump fine_rasterize out: bin_",settings.dump_binx, '_', settings.dump_biny," start...................")
        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"
        pix_to_face_path = Path(settings.output_dir.__fspath__() + "/pix_to_face_" + bin_x_y_path)
        zbuf_path = Path(settings.output_dir.__fspath__() + "/zbuf_" + bin_x_y_path)
        barycentrics_path = Path(settings.output_dir.__fspath__() + "/barycentrics_" + bin_x_y_path)
        dists_path = Path(settings.output_dir.__fspath__() + "/dists_" + bin_x_y_path)
        index_path = Path(settings.output_dir.__fspath__() + "/index_" + bin_x_y_path)

        barycentrics_trans_path = Path(settings.output_dir.__fspath__() + "/barycentrics_trans_" + bin_x_y_path)

        pix_to_face.int().numpy().tofile(pix_to_face_path)
        zbuf.numpy().tofile(zbuf_path)
        barycentrics.numpy().tofile(barycentrics_path)
        dists.numpy().tofile(dists_path)
        index.int().numpy().tofile(index_path)

        barycentrics.transpose(0,2).numpy().tofile(barycentrics_trans_path)


        print("pix_to_face.shape: ", pix_to_face.shape, " dtype: ", pix_to_face.dtype)
        print("zbuf.shape: ", zbuf.shape, " dtype: ", zbuf.dtype)
        print("barycentrics.shape: ", barycentrics.shape, " dtype: ", barycentrics.dtype)
        print("dists.shape: ", dists.shape, " dtype: ", dists.dtype)
        print("index.shape: ", index.shape, " dtype: ", index.dtype)

        print("dump fine_rasterize: bin_",settings.dump_binx, '_', settings.dump_biny," end...................")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        tensor_100_path = Path(settings.output_dir.__fspath__() + "/tensor_100.bin")
        tensor_010_path = Path(settings.output_dir.__fspath__() + "/tensor_010.bin")
        tensor_001_path = Path(settings.output_dir.__fspath__() + "/tensor_001.bin")

        tensor_100 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        tensor_010 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        tensor_001 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        tensor_100.numpy().tofile(tensor_100_path)
        tensor_010.numpy().tofile(tensor_010_path)
        tensor_001.numpy().tofile(tensor_001_path)


def triangle_squared_distance(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    assert p.dim() >= 2
    assert p.shape[-1] == 2

    assert a.dim() == b.dim() == c.dim() == 2
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 2
    assert a.shape[0] == b.shape[0] == c.shape[0]

    dist_ab = _point_segment_distance_square(settings, p, a, b) # (#points..., #faces)
    dist_bc = _point_segment_distance_square(settings, p, b, c) # (#points..., #faces)
    dist_ca = _point_segment_distance_square(settings, p, c, a) # (#points..., #faces)

    dists = torch.stack([dist_ab, dist_bc, dist_ca]).min(dim=0).values

    return dists

def triangle_signed_squared_distance(
    settings: RenderSettings,
    p: torch.Tensor, # (#points..., 2)
    a: torch.Tensor, # (#faces, 2)
    b: torch.Tensor, # (#faces, 2)
    c: torch.Tensor, # (#faces, 2)
) -> torch.Tensor: # (#points..., #faces)
    assert p.dim() >= 2
    assert p.shape[-1] == 2

    assert a.dim() == b.dim() == c.dim() == 2
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 2
    assert a.shape[0] == b.shape[0] == c.shape[0]

    is_inside = is_inside_triangle(settings, p, a, b, c) # (#points..., #faces)

    dists = triangle_squared_distance(settings, p, a, b, c)

    sign = (is_inside.int() * -2) + 1 # True -> -1, False -> 1

    return dists * sign

def barycentric_coords_noperspective(
    pxy: torch.Tensor, # (#p..., 2)
    axy: torch.Tensor, # (#tri, 2)
    bxy: torch.Tensor, # (#tri, 2)
    cxy: torch.Tensor, # (#tri, 2)
) -> torch.Tensor: # (#p..., #tri, 3)
    assert axy.dim() == bxy.dim() == cxy.dim() == 2
    assert axy.shape[-1] == bxy.shape[-1] == cxy.shape[-1] == pxy.shape[-1] == 2
    assert axy.shape[0] == bxy.shape[0] == cxy.shape[0]

    v0 = bxy - axy # (#tri, 2)
    v1 = cxy - axy # (#tri, 2)
    v2 = pxy.unsqueeze(-2) - axy # (#p..., #tri, 2)
    d00 = (v0 * v0).sum(-1) # (#tri,)
    d01 = (v0 * v1).sum(-1) # (#tri,)
    d11 = (v1 * v1).sum(-1) # (#tri,)
    d20 = (v2 * v0).sum(-1) # (#p..., #tri)
    d21 = (v2 * v1).sum(-1) # (#p..., #tri)
    denom = d00 * d11 - d01 * d01 # (#tri,)
    denom_inv = 1 / denom
    v = (d11 * d20 - d01 * d21) * denom_inv # (#p..., #tri)
    w = (d00 * d21 - d01 * d20) * denom_inv # (#p..., #tri)
    u = 1.0 - v - w # (#p..., #tri)
    return torch.stack((u, v, w), dim=-1)

def barycentric_coords_perspective_correction(
    bary_nopersp: torch.Tensor, # (#p..., #tri, 3)
    a: torch.Tensor, # (#tri, 3)
    b: torch.Tensor, # (#tri, 3)
    c: torch.Tensor, # (#tri, 3)
) -> torch.Tensor: # (#p..., #tri, 3)
    assert bary_nopersp.dim() >= 3
    assert bary_nopersp.shape[-1] == 3

    assert a.dim() == b.dim() == c.dim() == 2
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 3
    assert a.shape[0] == b.shape[0] == c.shape[0] == bary_nopersp.shape[-2]

    eps = 1e-6

    az = a[..., 2] # (#tri,)
    bz = b[..., 2] # (#tri,)
    cz = c[..., 2] # (#tri,)
    w0_top = bary_nopersp[..., 0] * bz * cz # (#p..., #tri)
    w1_top = az * bary_nopersp[..., 1] * cz # (#p..., #tri)
    w2_top = az * bz * bary_nopersp[..., 2] # (#p..., #tri)
    denom = torch.max(w0_top + w1_top + w2_top, torch.tensor(eps)) # (#p..., #tri)
    denom_inv = 1 / denom
    w0 = w0_top * denom_inv # (#p..., #tri)
    w1 = w1_top * denom_inv # (#p..., #tri)
    w2 = w2_top * denom_inv # (#p..., #tri)

    return torch.stack((w0, w1, w2), dim=-1)

def _fine_rasterize_3(
    settings: RenderSettings,
    pix_samples: torch.Tensor, # (#p..., 2)
    face_verts: torch.Tensor, # (#face, 3, 3)
    face_idx: torch.Tensor, # (#face,)
):
    assert pix_samples.shape[-1] == 2
    assert len(face_verts.shape) == 3
    assert face_verts.shape[1] == face_verts.shape[2] == 3

    max_blend_depth = settings.max_blend_depth
    blur_radius     = settings.blur_radius_ndc

    flattened_samples = pix_samples.view(-1, 2)

    if settings.need_dump:
        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"
        face_verts_path = Path(settings.output_dir.__fspath__() + "/face_verts_" + bin_x_y_path)
        face_idx_path = Path(settings.output_dir.__fspath__() + "/face_idx_" + bin_x_y_path)
        face_verts.numpy().tofile(face_verts_path)
        face_idx.int().numpy().tofile(face_idx_path)

    tri_vert_tuple = face_verts[..., :2].unbind(dim=-2)
    # face_sd2 = triangle_signed_squared_distance(settings, flattened_samples, *tri_vert_tuple) # (flat_#p, #face)

    barycentrics = barycentric_coords_noperspective(flattened_samples, *tri_vert_tuple) # (flat_#p, #face, 3)
    dump_barycentric_coords(settings, flattened_samples, *tri_vert_tuple, barycentrics)

    # if settings.persp_correct:
    #     # Using clip space z, needs perspective correction
    #     barycentrics = barycentric_coords_perspective_correction(barycentrics, *face_verts.unbind(dim=-2)) # (flat_#p, #face, 3)
    #     dump_barycentric_coords_correction(settings, barycentrics, *face_verts.unbind(dim=-2), barycentrics)

    barycentrics_greater_zero = barycentrics > 0.0

    is_inside = barycentrics_greater_zero[:, :, 0] & barycentrics_greater_zero[:, :, 1] & barycentrics_greater_zero[:, :, 2]

    distance = triangle_squared_distance(settings, flattened_samples, *tri_vert_tuple)
    # sign = (is_inside * -2.0) + 1.0 # True -> -1, False -> 1
    # face_sd2 =  distance * sign
    distance_neg = distance * -1.0
    face_sd2 = torch.where(is_inside, distance_neg, distance)  # True -> -1, False -> 1
    dump_triangle_signed_squared_distance(settings, flattened_samples, *tri_vert_tuple, distance, face_sd2)

    z_frags = (barycentrics * face_verts[..., 2]).sum(-1) # (flat_#p, #face)
    should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)

    pix_to_face = face_idx.squeeze(dim=0).expand(flattened_samples.shape[0], -1) # (flat_#p, #face)
    zbuf = torch.where(should_write, z_frags, INF_Z) # (flat_#p, #face)
    bary_coords = barycentrics # (flat_#p, #face, 3)
    dists = face_sd2 # (flat_#p, #face)

    if settings.need_dump:
        bin_x_y_path = str(settings.dump_binx) + "_" + str(settings.dump_biny) + ".bin"

        is_inside_path = Path(settings.output_dir.__fspath__() + "/is_inside_" + bin_x_y_path)
        is_inside.numpy().tofile(is_inside_path)

        z_frags_path = Path(settings.output_dir.__fspath__() + "/z_frags_" + bin_x_y_path)
        z_frags.numpy().tofile(z_frags_path)

        should_write_path = Path(settings.output_dir.__fspath__() + "/should_write_" + bin_x_y_path)
        should_write.numpy().tofile(should_write_path)

        zbuf_where_path = Path(settings.output_dir.__fspath__() + "/zbuf_where_" + bin_x_y_path)
        zbuf.numpy().tofile(zbuf_where_path)

    if max_blend_depth == 1:
        zbuf, min_idx = zbuf.min(dim=-1, keepdim = True) # zbuf (flat_#p, #face)
        pix_to_face = torch.gather(pix_to_face, dim=-1, index=min_idx)
        bary_coords = torch.gather(bary_coords, dim=-2, index=min_idx.unsqueeze(dim=-1).expand(*min_idx.shape, 3))
        dists = torch.gather(dists, dim=-1, index=min_idx)
    else:
        zbuf, sorted_idx = zbuf.sort(dim=-1) # zbuf (flat_#p, #face)
        pix_to_face = pix_to_face.gather(dim=-1, index=sorted_idx) # (flat_#p, #face)
        # zbuf = zbuf.gather(dim=-1, index=sorted_idx) # (flat_#p, #face)
        bary_coords = bary_coords.gather(dim=-2, index=sorted_idx.unsqueeze(dim=-1).expand(*sorted_idx.shape, 3)) # (flat_#p, #face, 3)
        dists = dists.gather(dim=-1, index=sorted_idx) # (flat_#p, #face)

        if face_verts.shape[0] >= max_blend_depth:
            pix_to_face = pix_to_face[:, :max_blend_depth]
            zbuf = zbuf[:, :max_blend_depth]
            bary_coords = bary_coords[:, :max_blend_depth, :]
            dists = dists[:, :max_blend_depth]
        else:
            pad_len = max_blend_depth - face_verts.shape[0]
            pix_to_face = F.pad(pix_to_face, (0, pad_len), value=-1)
            zbuf = F.pad(zbuf, (0, pad_len), value=INF_Z)
            bary_coords = F.pad(bary_coords, (0, 0, 0, pad_len), value=0)
            dists = F.pad(dists, (0, pad_len), value=-1)

    dump_fine_rasterize(settings, pix_to_face, zbuf, bary_coords, dists, min_idx)

    # for i in range(1, max_blend_depth):
    #     assert not (zbuf[..., i] < zbuf[..., i - 1]).any()

    if settings.clip_barycentric_coords:
        bary_coords = bary_coords.clamp(0, 1)

    pix_to_face=pix_to_face.view(*pix_samples.shape[:-1], -1)

    zbuf=zbuf.view(*pix_samples.shape[:-1], -1)
    
    bary_coords=bary_coords.view(*pix_samples.shape[:-1], -1, 3)
    
    dists=dists.view(*pix_samples.shape[:-1], -1)

    return pix_to_face,zbuf,bary_coords,dists


def rasterize(
    settings: RenderSettings,
    face_verts: torch.Tensor, # (#face, 3, 3)
):
    # binning / coarse rasterization
    bin_size = settings.bin_size
    bin_size_inv = 1 / bin_size

    # bin_w, bin_h = tuple(map(lambda x: int(math.ceil(x * bin_size_inv)), settings.image_size))

    # bin_w = math.ceil(settings.image_size[0] * bin_size_inv)
    # bin_h = math.ceil(settings.image_size[1] * bin_size_inv)
    bin_w, bin_h = tuple(map(lambda x: int(math.ceil(x * bin_size_inv)), settings.image_size))

    verts_without_z = face_verts[..., :2] # (#face, 3, 2)
    min_wh = verts_without_z.min(dim=-2).values # (#face, 2)
    max_wh = verts_without_z.max(dim=-2).values # (#face, 2)
    bin_min_wh = (min_wh * bin_size_inv).floor().long() # (#face, 2)
    bin_max_wh = (max_wh * bin_size_inv).ceil().long()  # (#face, 2)

    w_grid = torch.arange(bin_w).view(bin_w, 1, 1)
    h_grid = torch.arange(bin_h).view(1, bin_h, 1)

    blur_radius_scr = settings.blur_radius_scr() #* bin_size    #需要扩大bin_size倍，因为bin_min_wh已经缩小了bin_size倍。

    w_min = bin_min_wh[:, 0] - blur_radius_scr # (#face,)
    w_max = bin_max_wh[:, 0] + blur_radius_scr # (#face,)
    h_min = bin_min_wh[:, 1] - blur_radius_scr # (#face,)
    h_max = bin_max_wh[:, 1] + blur_radius_scr # (#face,)

    # 按bin_size缩小后的坐标进行比较。
    binning_mask = (w_grid >= w_min) & (w_grid < w_max) & (h_grid >= h_min) & (h_grid < h_max) # (BIN_W, BIN_H, #face)

    max_blend_depth = settings.max_blend_depth

    default_bin_pix_to_face = torch.full((bin_size, bin_size, max_blend_depth), -1, dtype=torch.long)
    default_bin_zbuf        = torch.full((bin_size, bin_size, max_blend_depth), INF_Z, dtype=torch.float)
    default_bin_bary_coords = torch.full((bin_size, bin_size, max_blend_depth, 3), 0, dtype=torch.float)
    default_bin_dists       = torch.full((bin_size, bin_size, max_blend_depth), -1, dtype=torch.float)

    pix_to_face = []
    zbuf = []
    bary_coords = []
    dists = []
    for bin_x in range(bin_w):
        bin_pix_to_face = []
        bin_zbuf = []
        bin_bary_coords = []
        bin_dists = []
        for bin_y in range(bin_h):
            binning_index = binning_mask[bin_x, bin_y].nonzero().squeeze(dim=-1)
            if binning_index.numel() == 0:
                bin_pix_to_face.append(default_bin_pix_to_face)
                bin_zbuf.append(default_bin_zbuf)
                bin_bary_coords.append(default_bin_bary_coords)
                bin_dists.append(default_bin_dists)
                continue
            # if( binning_index.numel() != 0):
            #     print(bin_x, bin_y)
            #     print(binning_index.numel())
            #     print(binning_index)
            # process each bin respectively
            if( binning_index.numel() > 32) & (binning_index.numel() % 16 == 0):
            # if( bin_x == 13) & (bin_y == 10):
            # if( binning_index.numel() != 0):
                print("\n\n=====================================================\n\n")
                print("------fine_rasterize:bin ", bin_x, "_" , bin_y, ", face_num: ", binning_index.shape)
                settings.need_dump = True
                settings.dump_binx = bin_x
                settings.dump_biny = bin_y
            else:
                settings.need_dump = False

            samp_x_grid = torch.arange(bin_size) + (bin_x * bin_size + 0.5)
            samp_y_grid = torch.arange(bin_size) + (bin_y * bin_size + 0.5)
            samples = torch.cartesian_prod(samp_x_grid, samp_y_grid).view(bin_size, bin_size, 2) # (bin_size, bin_size, 2)

            bin_frags_pix_to_face, bin_frags_zbuf, bin_frags_bary_coords, bin_frags_dists = \
                                            _fine_rasterize_3(settings,
                                            samples,
                                            face_verts[binning_index],
                                            binning_index)
            settings.need_dump = False
            bin_pix_to_face.append(bin_frags_pix_to_face)
            bin_zbuf.append(bin_frags_zbuf)
            bin_bary_coords.append(bin_frags_bary_coords)

            if bin_frags_dists is not None:
                bin_dists.append(bin_frags_dists)

        pix_to_face.append(torch.cat(bin_pix_to_face, dim=1))
        zbuf.append(torch.cat(bin_zbuf, dim=1))
        bary_coords.append(torch.cat(bin_bary_coords, dim=1))
        if len(bin_dists) == bin_h:
            dists.append(torch.cat(bin_dists, dim=1))

    pix_to_face = torch.cat(pix_to_face, dim=0)
    pix_to_face = pix_to_face[:settings.image_size[0], :settings.image_size[1]]

    zbuf = torch.cat(zbuf, dim=0)
    zbuf = zbuf[:settings.image_size[0], :settings.image_size[1]]

    bary_coords = torch.cat(bary_coords, dim=0)
    bary_coords = bary_coords[:settings.image_size[0], :settings.image_size[1]]

    if len(dists) == bin_w:
        dists = torch.cat(dists, dim=0)
        dists = dists[:settings.image_size[0], :settings.image_size[1]]
    else:
        dists = None

    return binning_mask,pix_to_face,zbuf,bary_coords,dists


def operation_operation_rasterize(output_dir: Path):
    settings = RenderSettings(
        image_size=(2048, 2048),
        max_blend_depth=1,
        bin_size=128,
        blur_radius_ndc=0,
        clip_barycentric_coords=False,
        cull_backfaces=False,
        front_face=FrontFace.CCW,
        persp_correct=False,
    )
    settings.output_dir = output_dir

    bin_mask_golden_path    = Path('/home/j00560472/torch3d/binning_mask_golden.npy')
    pix_to_face_golden_path = Path('/home/j00560472/torch3d/pix_to_face_golden.npy')
    zbuf_golden_path        = Path('/home/j00560472/torch3d/zbuf_golden.npy')
    bary_coords_golden_path = Path('/home/j00560472/torch3d/bary_coords_golden.npy')
    dists_golden_path       = Path('/home/j00560472/torch3d/dists_golden.npy')

    face_verts_path = Path('/home/j00560472/torch3d/face_verts.npy')
    face_verts = np.load(face_verts_path)

    mask_golden        = np.load(bin_mask_golden_path)
    pix_to_face_golden = np.load(pix_to_face_golden_path)
    zbuf_golden        = np.load(zbuf_golden_path)
    bary_coords_golden = np.load(bary_coords_golden_path)
    dists_golden       = np.load(dists_golden_path)

    print("face_verts.shape: ", face_verts.shape, ", dypte: ", face_verts.dtype)
    print("mask_golden.shape: ", mask_golden.shape, ", dypte: ", mask_golden.dtype)
    print("pix_to_face_golden.shape: ", pix_to_face_golden.shape, ", dypte: ", pix_to_face_golden.dtype)
    print("zbuf_golden.shape: ", zbuf_golden.shape, ", dypte: ", zbuf_golden.dtype)
    print("bary_coords_golden.shape: ", bary_coords_golden.shape, ", dypte: ", bary_coords_golden.dtype)
    print("dists_golden.shape: ", dists_golden.shape, ", dypte: ", dists_golden.dtype)

    mask,pix_to_face,zbuf,bary_coords,dists = rasterize(settings, torch.tensor(face_verts))

    atol = 0.001
    rtol = 0.001
    print(f"\n==================================================================================== output")
    data_compare_np = compare.data_compare_np

    print(f"\n\n============= mask compare result ============")
    result, fulfill_percent, max_error = data_compare_np(mask_golden, mask.numpy(), rtol, atol, is_display=False)
    print(result, fulfill_percent, max_error)

    print(f"\n\n============= pix_to_face compare result ============")
    result, fulfill_percent, max_error = data_compare_np(pix_to_face_golden.astype(np.int32), pix_to_face.numpy().astype(np.int32), rtol, atol, is_display=False)
    print(result, fulfill_percent, max_error)

    print(f"\n\n============= zbuf compare result ============")
    result, fulfill_percent, max_error = data_compare_np(zbuf_golden, zbuf.numpy(), rtol, atol, is_display=False)
    print(result, fulfill_percent, max_error)

    print(f"\n\n============= bary_coords compare result ============")
    result, fulfill_percent, max_error = data_compare_np(bary_coords_golden, bary_coords.numpy(), rtol, atol, is_display=False)
    print(result, fulfill_percent, max_error)

    print(f"\n\n============= dists compare result ============")
    result, fulfill_percent, max_error = data_compare_np(dists_golden, dists.numpy(), rtol, atol, is_display=False)
    print(result, fulfill_percent, max_error)

    face_verts_path = Path(output_dir, 'face_verts.bin')
    face_verts.tofile(face_verts_path)

    samp_x_grid = torch.arange(1024) + 0.5
    samp_y_grid = torch.arange(1024) + 0.5
    point = torch.cartesian_prod(samp_x_grid, samp_y_grid).view(1024, 1024, 2) # (bin_size, bin_size, 2)
    point_path = Path(output_dir, 'point.bin')
    point.numpy().tofile(point_path)

    bin_mask_gen_path = Path(output_dir, 'bin_mask.bin')
    mask.numpy().tofile(bin_mask_gen_path)

    pix_to_face_gen_path = Path(output_dir, 'pix_to_face.bin')
    mask.numpy().tofile(pix_to_face_gen_path)

    zbuf_gen_path = Path(output_dir, 'zbuf.bin')
    mask.numpy().tofile(zbuf_gen_path)    

    bary_coords_gen_path = Path(output_dir, 'barycentrics.bin')
    mask.numpy().tofile(bary_coords_gen_path)

    dists_gen_path = Path(output_dir, 'dists.bin')
    mask.numpy().tofile(dists_gen_path)



def operation_operation_greater(output_dir: Path):
    x_path = Path(output_dir, 'greater_x.bin')
    y_path = Path(output_dir, 'greater_y.bin')
    res_path = Path(output_dir, 'greater_res.bin')

    S0 = 2
    S1 = 200

    x = np.random.uniform(1, 4, [S0, S1]).astype(np.float32)
    x.tofile(x_path)

    y = np.full([S0, S1], 3).astype(np.float32)
    y.tofile(y_path)

    greater_res = np.greater(x, y)
    greater_res.tofile(res_path)

    where_res = np.where(greater_res, x, np.inf)

    where_index_path = Path(output_dir, 'index.bin')
    where_input_path = Path(output_dir, 'where_y.bin')
    where_res_path = Path(output_dir, 'where_res.bin')

    greater_res.tofile(where_index_path)
    x.tofile(where_input_path)
    where_res.tofile(where_res_path)

    print(x)
    print(greater_res)
    # print(where_res)


def operation_operation_and(output_dir: Path):
    x_path = Path(output_dir, 'index_1.bin')
    y_path = Path(output_dir, 'index_2.bin')
    res_path = Path(output_dir, 'and_res.bin')

    S0 = 2
    S1 = 201

    x = np.random.uniform(1, 4, [S0, S1]).astype(np.float32)
    y = np.full([S0, S1], 3).astype(np.float32)

    index1 = np.greater(x, y)
    index1.tofile(x_path)

    m = np.random.uniform(1, 100, [S0, S1]).astype(np.float32)
    n = np.full([S0, S1], 35).astype(np.float32)
    index2 = np.greater(m, n)
    index2.tofile(y_path)

    and_res = np.bitwise_and(index1, index2)
    and_res.tofile(res_path)

    # print(index1)
    # print(index2)
    # print(and_res)
    # print(and_res.dtype)


def operation_operation_gatherelement(output_dir: Path):
    x_path = Path(output_dir, 'input.bin')
    index_path = Path(output_dir, 'sort_index.bin')
    res_path = Path(output_dir, 'gather_res.bin')

    S0 = 16
    S1 = 10000

    x = np.random.uniform(1, 100, [S0, S1]).astype(np.float32)
    y = np.full([S0, S1], 35).astype(np.float32)

    index1 = np.greater(x, y)
    where_res = np.where(index1, x, np.inf)

    sort, sort_index = torch.sort(torch.tensor(where_res), axis=-1)
    # sort, sort_index = torch.tensor(where_res).min(dim=-1, keepdim = True)

    z = torch.arange(S0 * S1).view(S0, S1).float()

    gather_res = torch.gather(z, -1, sort_index)

    z.numpy().tofile(x_path)
    sort_index.int().numpy().tofile(index_path)
    gather_res.numpy().tofile(res_path)

    # print(sort)
    # print(sort_index)
    print(gather_res)
    print(sort_index.dtype)

def operation_operation_min_max(output_dir: Path):
    x_path = Path(output_dir, 'input.bin')
    index_path = Path(output_dir, 'min_index.bin')
    sort_index_path = Path(output_dir, 'sort_index.bin')
    gather_path = Path(output_dir, 'gather_res.bin')

    S0 = 16
    S1 = 32

    x = np.random.uniform(1, 100, [S0, S1]).astype(np.float32)
    x.tofile(x_path)

    # y = np.full([S0, S1], 35).astype(np.float32)

    # index1 = np.greater(x, y)
    # where_res = np.where(index1, x, np.inf)

    # sort, sort_index = torch.sort(torch.tensor(where_res), axis=-1)
    min, min_index = torch.tensor(x).min(dim=-1, keepdim = True)
    min_index.int().numpy().tofile(index_path)

    argSort = torch.argsort(torch.tensor(x), dim=-1, descending=False)
    argSort.int().numpy().tofile(sort_index_path)

    gather_res = torch.gather(torch.tensor(x), -1, min_index)
    gather_res.numpy().tofile(gather_path)

    # print(x)
    print(min_index)
    print(argSort)

@GoldenRegister.reg_golden_func(
    case_names=[
        "SoftmaxRgbBlendOnBoardTest.test_operation_prod",
        "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic",
        "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic_loop",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4_dynamic",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8_dynamic",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_dynamic",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_cow_data",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_large_shape",
        "SoftmaxRgbBlendOnBoardTest.test_operation_expand_dynamic_loop",
        "SoftmaxRgbBlendOnBoardTest.test_operation_expand",
        "RasterizeMeshOnBoardTest.test_point_triangle_distance",
        "RasterizeMeshOnBoardTest.test_point_triangle_singed_distance",
        "RasterizeMeshOnBoardTest.test_barycentric_coords_noperspective",
        "RasterizeMeshOnBoardTest.test_barycentric_coords_perspective_correct",
        "RasterizeMeshOnBoardTest.test_rasterize",
        "RasterizeMeshOnBoardTest.test_operation_greater",
        "RasterizeMeshOnBoardTest.test_operation_where",
        "RasterizeMeshOnBoardTest.test_operation_and",
        "RasterizeMeshOnBoardTest.test_operation_gatherelement",
        "RasterizeMeshOnBoardTest.test_operation_min_max"
    ]
)
def expand_operator_func1(case_name: str, output: Path) -> bool:
    dtype = np.float32

    # x_path = Path(output, 'x.bin')
    # res_path = Path(output, 'res.bin')
    # complete = x_path.exists() and res_path.exists()
    complete = False

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        if case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4_dynamic" :
            operation_softmax_rgb_blend2_1_16_128_4(output)
            return True

        if case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8_dynamic" :
            operation_softmax_rgb_blend2_1_16_128_8(output)
            return True

        if case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_dynamic":
            operation_softmax_rgb_blend2_1_128_128_2(output)
            return True
        if case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_cow_data":
            operation_softmax_rgb_blend2_1_128_128_2(output)
            # operation_cow_data(output_dir=output)
            return True
        if case_name == "SoftmaxRgbBlendOnBoardTest.test_operation_prod" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic_loop":
            operation_operation_prod(output)
            return True
        if case_name == "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_large_shape":
            operation_softmax_rgb_blend2_large(output)
            return True
        if case_name == "SoftmaxRgbBlendOnBoardTest.test_operation_expand_dynamic_loop" or \
           case_name == "SoftmaxRgbBlendOnBoardTest.test_operation_expand":
            operation_operation_expand(output)
            return True
        if case_name == "RasterizeMeshOnBoardTest.test_point_triangle_distance" or \
           case_name == "RasterizeMeshOnBoardTest.test_point_triangle_singed_distance" or \
           case_name == "RasterizeMeshOnBoardTest.test_barycentric_coords_noperspective" or \
           case_name == "RasterizeMeshOnBoardTest.test_barycentric_coords_perspective_correct" or \
           case_name == "RasterizeMeshOnBoardTest.test_rasterize":
            operation_operation_rasterize(output)
            return True
        if case_name == "RasterizeMeshOnBoardTest.test_operation_greater" or \
           case_name == "RasterizeMeshOnBoardTest.test_operation_where":
            operation_operation_greater(output)
            return True
        if case_name == "RasterizeMeshOnBoardTest.test_operation_and":
            operation_operation_and(output)
            return True
        if case_name == "RasterizeMeshOnBoardTest.test_operation_gatherelement":
            operation_operation_gatherelement(output)
            return True
        if case_name == "RasterizeMeshOnBoardTest.test_operation_min_max":
            operation_operation_min_max(output)
            return True        
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        # "SoftmaxRgbBlendOnBoardTest.test_operation_prod",
        # "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic",
        # "SoftmaxRgbBlendOnBoardTest.test_operation_prod_dynamic_loop",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_4_dynamic",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_16_128_8_dynamic",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2",
        "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_dynamic",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_1_128_128_2_cow_data",
        # "SoftmaxRgbBlendOnBoardTest.test_softmax_rgb_blend2_large_shape",
        # "SoftmaxRgbBlendOnBoardTest.test_operation_expand_dynamic_loop",
        # "SoftmaxRgbBlendOnBoardTest.test_operation_expand",
        # "RasterizeMeshOnBoardTest.test_point_triangle_distance",
        # "RasterizeMeshOnBoardTest.test_point_triangle_singed_distance",
        # "RasterizeMeshOnBoardTest.test_barycentric_coords_noperspective",
        # "RasterizeMeshOnBoardTest.test_barycentric_coords_perspective_correct",
        "RasterizeMeshOnBoardTest.test_rasterize",
        # "RasterizeMeshOnBoardTest.test_operation_greater",
        # "RasterizeMeshOnBoardTest.test_operation_where",
        # "RasterizeMeshOnBoardTest.test_operation_and",
        "RasterizeMeshOnBoardTest.test_operation_gatherelement",
        # "RasterizeMeshOnBoardTest.test_operation_min_max"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        print(g_src_root, output)
        ret = expand_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
