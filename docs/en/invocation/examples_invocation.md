# Running Examples

## Simulation Environment (No Physical NPU Hardware)

```bash
cd examples/00_hello_world
python3 hello_world.py --run_mode=sim
```

## Real Execution Environment (With Physical NPU Hardware)

```bash
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

For more examples, refer to the sample code under the `examples/` directory.

## Viewing Results

After this basic example runs successfully, compilation and execution artifacts are generated in the `${work_path}/output/` directory. These artifacts include a [computation graph](../tutorials/appendix/glossary.md) and a [swimlane graph](../tutorials/appendix/glossary.md). Both graphs can be viewed in VS Code and linked to the source code using the PyPTO Toolkit plugin. For details on using the Toolkit, see [Quick Start - Viewing the Computation Graph](../tutorials/introduction/quick_start.md#viewing-the-computation-graph) and [Quick Start - Viewing the Swimlane Graph](../tutorials/introduction/quick_start.md#viewing-the-swimlane-graph).

## Quick Start

The following is a simple PyPTO usage example. You can specify whether to run the simulation or real environment example via the `--run_mode` parameter:

```python
import pypto
import torch
import argparse

shape = (1, 4, 1, 64)

# Create compute kernel based on run mode
def create_add_kernel(run_mode: str):
    mode = pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = x + y

    return add_kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu", "sim"])
    args = parser.parse_args()

    # Prepare input data
    device = "cpu"
    if args.run_mode == "npu":
        import torch_npu
        torch.npu.set_device(0)
        device = "npu:0"

    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)

    # Execute computation and view results
    create_add_kernel(args.run_mode)(x, y, output)
    print(f"Output shape: {output.shape}")
```

- For the real environment or accuracy simulation, you can view the results by directly inspecting the output tensor values.
- For performance simulation, view the simulation results via the swimlane graph under `output/`.

For the complete example, refer to: [hello_world.py](../../examples/00_hello_world/hello_world.py).

