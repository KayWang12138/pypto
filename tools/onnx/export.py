import torch
import torch.nn as nn
import torch.onnx
import onnx
import os

def export_to_onnx(model: nn.Module, inputs: torch.Tensor, path: str, input_names: list[str], output_names: list[str]):
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        do_constant_folding=False,
        dynamo=False,
        report=True
    )

    print(f"Exported ONNX model to {path}")

    m = onnx.load(path)
    onnx.checker.check_model(m)
    print("ONNX graph:")
    print(onnx.helper.printable_graph(m.graph))

# XXX legacy, not related to onnx
def export_to_pt2(model: nn.Module, inputs: torch.Tensor, path: str):
    ep = torch.export.export(model, args=(x0, x1), strict=False)
    print(ep)
    print(ep.graph)
    torch.export.save(ep, path)
