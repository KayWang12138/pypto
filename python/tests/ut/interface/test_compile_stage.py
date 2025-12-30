import os
import pypto
import torch
import torch_npu

def test_compile_stage(run_mode):
    @pypto.jit(runtime_options={"run_mode": run_mode})
    def compile_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b

    a = torch.ones((4, 4), dtype=torch.float32, device="npu") * 2
    b = torch.ones((4, 4), dtype=torch.float32, device="npu") * 3
    c = torch.ones((4, 4), dtype=torch.float32, device="npu")
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    compile_func(input_a, input_b, output_c)

    print(c)
    print(f"run_mode {run_mode} test success")

def test_all_compile_stages():
    print("Testing COMPILE_STAGE1 (run_mode=2) - Generate Tensor Graph")
    test_compile_stage(2)
    print()

    print("Testing COMPILE_STAGE2 (run_mode=3) - Subgraph To Function")
    test_compile_stage(3)
    print()

    print("Testing COMPILE_STAGE3 (run_mode=4) - Generate Execute Graph")
    test_compile_stage(4)
    print()

    print("Testing COMPILE_STAGE4 (run_mode=5) - Generate Instructions")
    test_compile_stage(5)
    print()

    print("Testing COMPILE_STAGE5 (run_mode=6) - Generate Kernel Code")
    test_compile_stage(6)
    print()

    print("All compile stage tests completed!")

if __name__ == "__main__":
    test_all_compile_stages()