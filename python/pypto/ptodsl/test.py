from ptodsl.jit import *
from ptodsl.compiler import *
import ptodsl.language as pto

@pto_meta_data
def meta_data():
    # common, reusable type declarations
    # TODO: infer required type info from function body, or pre-declare all common meta types
    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)  # TODO: omit shape https://github.com/zhangstevenunity/PTOAS/issues/31
    tile_cfg = pto.TileBufConfig(
        blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null")
    tile_type = pto.TileBufType(
        shape=[32, 32], dtype=dtype, memory_space="UB", config=tile_cfg)
    # NOTE: valid_shape=shape if not specified
    return [ptr_type, tensor_type, subtensor_type, tile_type]


@kernel(None, meta_data)
def relu_kernel(x_ptr: "ptr_type", y_ptr: "ptr_type"):
    c0 = pto.const(0)
    c1 = pto.onst(1)
    c32 = pto.const(32)

    tv0 = pto.as_tensor(tensor_type, ptr=x_ptr, shape=[c32, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=y_ptr, shape=[c32, c32], strides=[c32, c1])

    # TODO: overload __getitem__
    sv0 = pto.slice_view(subtensor_type, source=tv0, offsets=[c0, c0], sizes=[c32, c32])
    sv1 = pto.slice_view(subtensor_type, source=tv1, offsets=[c0, c0], sizes=[c32, c32])

    tb0 = pto.alloc_tile(tile_type)
    tb1 = pto.alloc_tile(tile_type)

    pto.load(sv0, tb0)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    pto.relu(tb0, tb1)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)

    pto.store(tb1, sv1)
    # default to `return None`
    return

@jit()
def TestJITFunction():
    print("This is a test JIT function.")
    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.float32
    shape = [32, 32]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    x = torch.rand(shape, device=device, dtype=dtype) - 0.5
    y = torch.zeros(shape, device=device, dtype=dtype)

    relu_executor = relu_kernel()
    relu_executor(x, y)
    torch.npu.synchronize()
    y_ref = torch.nn.functional.relu(x)
    max_error = (y - y_ref).abs().max().item()
    print(f"Max error: {max_error}")
    return


if __name__ == "__main__":
    TestJITFunction()