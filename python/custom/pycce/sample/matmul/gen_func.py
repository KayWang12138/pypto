#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import sys
sys.dont_write_bytecode = True
from pycce.stub_functions import *


BASEM = 128
BASEN = 128
BASEK = 128

@cube_func(name='cust_matmul')
def cust_matmul(M: Var, N: Var, K: Var, x: GMTensor, y: GMTensor, z: GMTensor):
    M_PERCORE = CeilDiv(CeilDiv(M, BASEM), GetCubeNum()) * BASEM
    M1 = Var(M_PERCORE * GetCubeIdx())
    M2 = Var(Min(M1 + M_PERCORE, M))

    l1a = DBuff(DT.half, BASEM*BASEK, Position.L1)
    l1b = DBuff(DT.half, BASEN*BASEK, Position.L1)
    l0a = DBuff(DT.half, BASEN*BASEK, Position.L0A)
    l0b = DBuff(DT.half, BASEN*BASEK, Position.L0B)
    l0c = DBuff(DT.float, BASEM*BASEN, Position.L0C)

    l1cnt = Var(0)
    l0cnt = Var(0)
    outcnt = Var(0)

    with Loop('m', M1, M2, BASEM) as m:
        with Loop('n', 0, N, BASEN) as n:
            with auto_sync():
                with Loop('k', 0, K, BASEK) as k:
                    # mte2
                    gm_to_l1_nd2nz(l1a[l1cnt], x[m, k], BASEM, BASEK, K)
                    gm_to_l1_nd2nz(l1b[l1cnt], y[n, k], BASEN, BASEK, K)
                    # mte1
                    l1_to_l0_nz2zz(l0a[l0cnt], l1a[l1cnt], BASEM, BASEK, BASEM, BASEK)
                    l1_to_l0(l0b[l0cnt], l1b[l1cnt], BASEN, BASEK)
                    l1cnt += 1
                    # matmul
                    mad(l0c[outcnt], l0a[l0cnt], l0b[l0cnt], BASEM, BASEK, BASEN, k==0)
                    l0cnt += 1
                l0c_to_gm_nz2nd(z[m,n], l0c[outcnt], BASEM, BASEN, N, BASEM)
                outcnt += 1


@kernel_func()
def cust_kernel(M: Var, N: Var, K: Var, x: GMTensor, y: GMTensor, z: GMTensor):
    cust_matmul(M, N, K, x, y, z)


if __name__=='__main__':
    M = Var('M', DT.int, value=10240)
    N = Var('N', DT.int, value=512)
    K = Var('K', DT.int, value=1024)

    xmtx = GMTensor('xmtx', DT.half, shape=[M, K])
    ymtx = GMTensor('ymtx', DT.half, shape=[N, K])
    zmtx = GMTensor('zmtx', DT.half, shape=[M, N], is_output=True)

    kernel = cust_kernel(M, N, K, xmtx, ymtx, zmtx)

    kernel.gen_code('cust_kernel.cpp', 'cust_matmul_sigmoid')
    kernel.gen_test_entry('main.cpp', 'cust_kernel.cpp', 'cust_matmul_sigmoid', n_cores=20, test_speed=True, force=True)
    kernel.gen_golden('golden.py')
    kernel.gen_checker('check.py')
