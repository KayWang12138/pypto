import torch
import torch.nn as nn
import torch.onnx
import onnx
import torchair

from pathlib import Path
from typing import Union


def export_to_onnx(model: nn.Module, inputs: torch.Tensor, path: str, input_names: list[str], output_names: list[str]):
    """Export a PyTorch model to ONNX and run checker."""
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


def export_to_torchair(model: nn.Module, inputs: torch.Tensor, path: str):
    """Export a PyTorch model to TorchAir format."""
    path = Path(path)

    torchair.dynamo_export(
        *inputs,
        model=model,
        export_path=str(path.parent),
        export_name=str(path.stem),
    )

    print(f"Exported TorchAir model to {path}")
