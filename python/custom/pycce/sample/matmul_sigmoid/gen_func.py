import sys
sys.dont_write_bytecode = True
from pycce.stub_functions import * 


BASEM = 128
BASEN = 128
BASEK = 128

def simple_tiling(M: Var):
    M_PERCORE = CeilDiv(CeilDiv(M, BASEM), GetCubeNum()) * BASEM
    M1 = Var(M_PERCORE * GetCubeIdx())
    M2 = Var(Min(M1 + M_PERCORE, M))
    return M1, M2


@cube_func()
def cust_matmul(M: Var, N: Var, K: Var, x: GMTensor, y: GMTensor, z: GMTensor):
    M1, M2 = simple_tiling(M)

    # double buffers 
    l1a = DBuff(DT.half, BASEM*BASEK, Position.L1)
    l1b = DBuff(DT.half, BASEN*BASEK, Position.L1)
    l0a = DBuff(DT.half, BASEN*BASEK, Position.L0A)
    l0b = DBuff(DT.half, BASEN*BASEK, Position.L0B)
    l0c = DBuff(DT.float, BASEM*BASEN, Position.L0C)

    # ping-pong counter 
    l1cnt = Var(0)
    l0cnt = Var(0)
    outcnt = Var(0)

    # main computation loop 
    for m in Range(M1, M2, BASEM):
        for n in Range(0, N, BASEN):
            with auto_sync():
                for k in Range(0, K, BASEK):
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
            cube_ready()


@vec_func()
def cust_vec(M: Var, N: Var, z: GMTensor):
    # simple tiling 
    M1, M2 = simple_tiling(M)

    # declare double buffers 
    xbuf = DBuff(DT.half, BASEM*BASEN)
    outbuf = DBuff(DT.half, BASEM*BASEN)

    # ping-pong counter
    cnt = Var(0)

    rep = BASEM*BASEN//128
    # main computation loop 
    for m in Range(M1, M2, BASEM):
        for n in Range(0, N, BASEN):
            wait_cube()
            with auto_sync():
                # mte2
                gm_to_ub(xbuf[cnt], z[m,n], BASEM, BASEN//16, (N-BASEN)//16, 0)
                # v
                muls(outbuf[cnt], xbuf[cnt], -1.0, rep, 1, 1, 8, 8)
                exp(outbuf[cnt], outbuf[cnt], rep, 1, 1, 8, 8)
                adds(outbuf[cnt], outbuf[cnt], 1.0, rep, 1, 1, 8, 8)
                rec(outbuf[cnt], outbuf[cnt], rep, 1, 1, 8, 8)
                # mte3
                ub_to_gm(z[m,n], outbuf[cnt], BASEM, BASEN//16, 0, (N-BASEN)//16)


@kernel_func()
def cust_kernel(M: Var, N: Var, K: Var, x: GMTensor, y: GMTensor, z: GMTensor):
    cust_matmul(M, N, K, x, y, z)
    cust_vec(M, N, z)


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
