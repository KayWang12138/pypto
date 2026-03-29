# PyTorch Integration

PyPTO currently supports two execution modes: single-operator mode (eager) and graph capture mode (aclgraph):

-   Single-operator mode (eager): Code executes in the same way as a regular Python program — executing immediately, with functions executing as soon as they are called, without building a computation graph. This is developer- and debugging-friendly, but introduces task dispatch overhead on the Host side. As performance optimization deepens, this Host overhead gradually becomes a bottleneck that cannot be ignored.
-   Graph capture mode (aclgraph): Uses Capture & Replay to achieve one-time task capture and multiple executions. During the Capture phase, Stream tasks are captured to the Device side without executing; during the Replay phase, an execution command is sent from the Host side, and the Device side executes the previously captured tasks. This reduces Host scheduling overhead and improves performance.

Adding the `@pypto.frontend.jit` decorator before a kernel function defaults to using single-operator mode in the PyTorch framework. To enable graph capture mode, refer to the following code:

```python
B = pypto.DYNAMIC
N1, N2, DIM = 32, 1, 256

# enable frontend.jit for softmax
@pypto.frontend.jit()
def softmax_kernel(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
    output_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
):
    ...



@allow_in_graph
def softmax(x: torch.Tensor, dynamic: bool = True) -> torch.Tensor:
    if isinstance(x, FakeTensor):
        return torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')
    # launch the kernel
    out = torch.zeros(shape, dtype=x.dtype, device=f'{x.device}')
    softmax_kernel(x, out)
    return out


class MM(torch.nn.Module):
    def forward(self, x, dynamic):
        out = softmax(x, dynamic)
        return out


def test_softmax_capture(device_id=None, dynamic: bool = True) -> None:
    # prepare data
    ...
    model = torch.compile(MM(), backend="eager", dynamic=True)
    #graph capture
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y = model(x, dynamic)

    #execute graph
    g.replay()
    torch.npu.synchronize()
    ...

if __name__ == "__main__":
    test_softmax_capture()
```

You can view the info-level Host compilation log and search for the keyword `capture mode`. A value of 0 (default) indicates that graph capture mode is disabled; a value of 1 indicates that graph capture mode is enabled.

```text
2025-12-08 20:56:27.043	capture mode[1]
```

For the complete example, refer to: [aclgraph.py](../../../examples/03_advanced/aclgraph/aclgraph.py).

