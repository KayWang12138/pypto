import torch
import pypto

verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
}

@pypto.jit(
    verify_options=verify_options
)
def kernel(a, b):
    pypto.set_vec_tile_shapes(16, 16, 16, 16)
    b[:] = pypto.sum(a, dim = 2, keepdim=True)


def test_kernel(a):
    B, L, N, D = a.shape
    b = torch.zeros([B, L, 1, D], device=a.device).float()
    inputs = {
        a: [0, 1]
    }
    outputs = {
        b: [0, 1]
    }
    inputs = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in inputs.items()]
    outputs = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in outputs.items()]
    kernel(*inputs, *outputs)
    return b


def test():
    L = 8
    device_id = 1
    torch.npu.set_device(int(device_id))
    a = torch.rand([L, L, L, L], device=f"npu:{device_id}").float()
    b = test_kernel(a)
    print("===== input =====")
    print(a)
    print("===== output 1=====")
    print(b)


if __name__ == '__main__':
    test()