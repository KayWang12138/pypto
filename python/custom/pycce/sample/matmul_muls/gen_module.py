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

class CustCube(CubeModule):
    def initialize(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K
        self.M_PERCORE = CeilDiv(CeilDiv(self.M, BASEM), GetCubeNum()) * BASEM
        self.M1 = self.M_PERCORE * GetCubeIdx()
        self.M2 = Min(self.M1 + self.M_PERCORE, self.M)

        self.l1a = DBuff(DT.half, BASEM*BASEK)
        self.l1b = DBuff(DT.half, BASEN*BASEK)
        self.l0a = self.create_l0a(DBuff, DT.half, BASEM*BASEK)
        self.l0b = self.create_l0b(DBuff, DT.half, BASEN*BASEK)
        self.l0c = self.create_l0c(DBuff, DT.float, BASEM*BASEK)

    def forward(self, x, y, z):
        l0cnt = Var('l0cnt', DT.int, 0)
        l1cnt = Var('l1cnt', DT.int, 0)
        outcnt = Var('outcnt', DT.int, 0)
        with Loop('m', self.M1, self.M2, BASEM) as m:
            with Loop('n', 0, self.N, BASEN) as n:
                self.process_base_block(x, y, z, m, n, l1cnt, l0cnt, outcnt)
                cube_ready()

    @auto_sync()
    def process_base_block(self, x, y, z, m, n, l1cnt, l0cnt, outcnt):
        with Loop('k', 0, self.K, BASEK) as k:
            # mte2
            gm_to_l1_nd2nz(self.l1a[l1cnt], x[m, k], BASEM, BASEK, self.K)
            gm_to_l1_nd2nz(self.l1b[l1cnt], y[n, k], BASEN, BASEK, self.K)
            # mte1
            l1_to_l0_nz2zz(self.l0a[l0cnt], self.l1a[l1cnt], BASEM, BASEK, BASEM, BASEK)
            l1_to_l0(self.l0b[l0cnt], self.l1b[l1cnt], BASEN, BASEK)
            l1cnt += 1
            # matmul
            mad(self.l0c[outcnt], self.l0a[l0cnt], self.l0b[l0cnt], BASEM, BASEK, BASEN, k==0)
            l0cnt += 1
        l0c_to_gm_nz2nd(z[m,n], self.l0c[outcnt], BASEM, BASEN, self.N, BASEM)
        outcnt += 1


class CustVec(VecModule):
    def initialize(self, M, N):
        self.M = M
        self.N = N
        self.M_PERCORE = CeilDiv(CeilDiv(self.M, BASEM), GetCubeNum()) * BASEM
        self.M1 = self.M_PERCORE * GetCubeIdx()
        self.M2 = Min(self.M1 + self.M_PERCORE, self.M)

        self.xbuf = DBuff(DT.half, BASEM*BASEN)
        self.outbuf = DBuff(DT.half, BASEM*BASEN)

    def forward(self, z):
        cnt = Var('cnt', DT.int, 0)
        with Loop('m', self.M1, self.M2, BASEM) as m:
            with Loop('n', 0, self.N, BASEN) as n:
                wait_cube()
                self.process_base_block(z, m, n, cnt)

    @auto_sync()
    def process_base_block(self, z, m, n, cnt):
        # copy in
        gm_to_ub(self.xbuf[cnt], z[m,n], BASEM, BASEN//16, (self.N-BASEN)//16, 0)
        # compute
        muls(self.outbuf[cnt], self.xbuf[cnt], 2.0, BASEM*BASEN//128, 1, 1, 8, 8)
        # copy out
        ub_to_gm(z[m,n], self.outbuf[cnt], BASEM, BASEN//16, 0, (self.N-BASEN)//16)


class CustKernel(KernelBase):
    def initialize(self, M, N, K):
        self.cube0 = CustCube(M, N, K)
        self.vec0 = CustVec(M, N)

    def forward(self, x, y, z):
        self.cube0(x, y, z)
        self.vec0(z)


if __name__=='__main__':
    M = Var('M', DT.int, value=10240)
    N = Var('N', DT.int, value=512)
    K = Var('K', DT.int, value=1024)

    xmtx = GMTensor('xmtx', DT.half, shape=[M, K])
    ymtx = GMTensor('ymtx', DT.half, shape=[N, K])
    zmtx = GMTensor('zmtx', DT.half, shape=[M, N], is_output=True)

    kernel = CustKernel(M, N, K)
    kernel(xmtx, ymtx, zmtx)

    kernel.gen_code('cust_kernel.cpp', 'cust_matmul_sigmoid')
    kernel.gen_test_entry('main.cpp', 'cust_kernel.cpp', 'cust_matmul_sigmoid', n_cores=20, test_speed=True, force=True)
    kernel.gen_golden('golden.py')
    kernel.gen_checker('check.py')
