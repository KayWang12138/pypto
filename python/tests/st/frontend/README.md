python/tests/frontend 是 PyPTO 前端测试用例目录，用于验证前端 JIT、控制流与张量操作的正确性与稳定性。

### 测试用例介绍

- `test_frontend_arithmetic.py`：基础张量算术（add/mul/sub）、广播加法
- `test_frontend_view_assemble.py`：view 切片、assemble 回填
- `test_frontend_controlflow.py`：循环体、unroll 参数、条件分支控制流


### 测试用例开发规范




### 测试用例开发模版

```
def create_elementwise_kernel(shape: tuple[int, ...]):

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def elementwise_kernel(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 4)
        c = (a + b) * 2.0 - b
        return c

    return elementwise_kernel


def test_elementwise_add_mul_sub():
    # 初始化
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    # 构造数据
    shape = (2, 4)
    torch.manual_seed(2026)
    a = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")

    # 创建算子
    kernel = create_elementwise_kernel(shape)

    # 调用算子
    out = kernel(a, b)
    torch.npu.synchronize()

    # 验证算子
    expected = (a + b) * 2.0 - b
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)
```