import pypto
from pypto import Tensor

__all__ = ["sin"]


def sin(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise sine values of input.

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements
        are the sine values of the corresponding elements in the input tensor.


    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.sin(x)

    Input x:[-0.5461,  0.1347, -2.7266, -0.2746]
    Output y:[-0.5194,  0.1343, -0.4032, -0.2711]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    number2048 = 2048.0
    one_over_n = 1.0 / 2048.0
    inv_half_pi = 0.63661975
    pi0 = 1.5708008
    pi1 = -0.0000044535846
    pi2 = -8.706138e-10
    f_025 = 0.25
    f_05 = 0.5
    f_4 = 4.0
    f_1 = 1.0
    f_nega_1 = -1.0
    f_nega_2 = -2.0

    x_scaled = pypto.mul(input, one_over_n)
    x_over_pi = pypto.mul(x_scaled, inv_half_pi)
    n = pypto.cast(x_over_pi, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(x_over_pi, one_over_n)
    n0 = pypto.cast(n0, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(n0, number2048)

    n1 = pypto.sub(n, n0)

    fix = pypto.mul(n0, pi0)
    x_fix = pypto.sub(x_scaled, fix)
    fix = pypto.mul(n1, pi0)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi2)
    x_fix = pypto.sub(x_fix, fix)

    pi_02 = 1.5703125
    pi_12 = 0.0004837513

    remain_x = pypto.mul(x_fix, number2048)
    temp = pypto.mul(remain_x, inv_half_pi)
    n2 = pypto.cast(temp, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)

    n0 = pypto.mul(n0, number2048)
    n1 = pypto.mul(n1, number2048)
    fix = pypto.mul(n0, pi_02)
    x_fix = pypto.sub(input, fix)
    fix = pypto.mul(n1, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_12)
    x_fix = pypto.sub(x_fix, fix)

    pi_22 = 0.000000075495336
    fix = pypto.mul(n2, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_22)
    x_fix = pypto.sub(x_fix, fix)

    pi_32 = 2.5579538e-12
    fix = pypto.mul(n2, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_32)
    x_fix = pypto.sub(x_fix, fix)

    pi_42 = 5.389786e-15
    fix = pypto.mul(n2, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_42)
    x_fix = pypto.sub(x_fix, fix)

    pi_52 = 5.166901e-19
    fix = pypto.mul(n2, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_52)
    x_fix = pypto.sub(x_fix, fix)

    pi_62 = 3.281839e-22
    fix = pypto.mul(n2, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    fix = pypto.mul(n2, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_62)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n2, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    half_n2 = pypto.mul(n2, f_05)
    half4_n2 = pypto.mul(n2, f_025)
    n_half2 = pypto.cast(half_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)
    n_half4 = pypto.cast(half4_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)

    k1 = pypto.mul(n_half2, f_nega_2)
    k2 = pypto.mul(n_half4, f_4)
    sign = pypto.add(k1, k2)
    sign = pypto.add(sign, f_1)

    ifcos = pypto.add(n2, k1)
    ifsin = pypto.mul(ifcos, f_nega_1)
    ifsin = pypto.add(ifsin, f_1)

    scoef4 = 0.0000027183114939898219064
    scoef3 = -0.000198393348360966317347
    scoef2 = 0.0083333293858894631756
    scoef1 = -0.166666666416265235595
    x_pow = pypto.mul(x_fix, x_fix)
    sin_poly = pypto.mul(x_pow, scoef4)
    sin_poly = pypto.add(sin_poly, scoef3)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef2)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef1)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, f_1)
    sin_poly = pypto.mul(x_fix, sin_poly)

    ccoef4 = 0.0000243904487962774090654
    ccoef3 = -0.00138867637746099294692
    ccoef2 = 0.0416666233237390631894
    ccoef1 = -0.499999997251031003120
    cos_poly = pypto.mul(x_pow, ccoef4)
    cos_poly = pypto.add(cos_poly, ccoef3)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef2)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef1)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, f_1)

    temp1 = pypto.mul(sin_poly, ifsin)
    cos_poly = pypto.mul(cos_poly, ifcos)
    res = pypto.add(temp1, cos_poly)
    res = pypto.mul(res, sign)

    if dtype != res.dtype:
        res = pypto.cast(res, dtype)
    return res