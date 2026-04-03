"""test_bad_op.py — 触发 OL17-OL22 违规的 test fixture"""
# OL18: 缺少从 impl 和 golden 的导入
# OL20: 缺少设备 ID 环境变量处理
# OL22: 缺少随机种子设置

import pypto


@pypto.frontend.jit
def bad_kernel(x: pypto.Tensor([], pypto.DT_FP32)):  # OL17: 包含 kernel 实现
    pass


# OL19: 使用了手写 assert 而非 allclose
# OL21: 没有 level0/level1 函数
def test_bad_op():
    max_diff = 0.001
    assert max_diff < 0.01, "precision fail"
