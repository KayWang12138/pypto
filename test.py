import pypto
import torch
import sys


# 定义计算函数
def add_kernel(run_mode, shape):
    dtype = pypto.DT_FP32
    mode = pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def kernel(
        x0: pypto.tensor(shape, dtype),
        x1: pypto.tensor(shape, dtype),
    ) -> pypto.tensor(shape, dtype):
        pypto.set_vec_tile_shapes(4, 4)
        y = x0 + x1
        return y
    return kernel


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please specify the running mode as npu or sim via the args parameter.")
        sys.exit(1)
    run_mode = sys.argv[2].lower()

    if run_mode == "npu":
        torch.npu.set_device(0)
        device = torch.device("npu:0")
    elif run_mode == "sim":
        device = torch.device("cpu")
    else:
        print("Invalid parameters")
        sys.exit(1)

    # 创建 Tensor
    shape = (4, 4)
    x0 = torch.ones(shape, dtype=torch.float32, device=device)
    x1 = torch.ones(shape, dtype=torch.float32, device=device)

    # 执行计算
    if run_mode == "npu":
        torch.npu.set_device(0)
        y = add_kernel(run_mode, shape)(x0, x1)
        print(y)
    elif run_mode == "sim":
        y = add_kernel(run_mode, shape)(x0, x1)
        print("Simulation completed, please view the results through the swimlane diagram.")
    else:
        print("Invalid parameters")
