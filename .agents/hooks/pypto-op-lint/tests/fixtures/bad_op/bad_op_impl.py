"""bad_op_impl.py — 触发 OL01-OL08 全部违规的 fixture"""
# OL07 FAIL: 缺少 import pypto
import torch


# OL01 FAIL: 没有 @pypto.frontend.jit 装饰器
def bad_op_kernel(input_tensor, output_tensor):  # OL05 FAIL: 无类型注解
    # OL04 FAIL: 没有 set_*_tile_shapes
    result = min(input_tensor, output_tensor)  # OL06 FAIL: 原生 min()
    output_tensor = result  # OL02 FAIL: 非 [:]/move()/assemble() 写回
    return result  # OL03 FAIL: 有 return


# OL08 FAIL: 没有 _wrapper 函数
def bad_op_run(x):
    return x
