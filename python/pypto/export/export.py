import torch
import torch.nn as nn
import torch.onnx
import onnx

from pathlib import Path

__all__ = ("export_to_onnx", "export_to_torchair")


def export_to_onnx(
    model: nn.Module,
    inputs: torch.Tensor,
    path: str,
    input_names: list[str],
    output_names: list[str],
    forward_kwargs: dict | None = None,
):
    """Export a PyTorch model to ONNX format.

    *forward_kwargs* are forwarded to ``torch.onnx.export`` as ``kwargs`` so tracing
    runs ``model.forward(*inputs, **forward_kwargs)`` — required when the forward
    signature has non-tensor args (e.g. ``run_mode``) whose default would diverge
    from what the caller just evaluated (e.g. NPU vs simulation mode).
    """
    torch.onnx.export(
        model,
        inputs,
        path,
        kwargs=forward_kwargs,
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


def export_to_torchair(
    model: nn.Module,
    inputs: torch.Tensor,
    path: str,
    forward_kwargs: dict | None = None,
):
    """Export a PyTorch model to TorchAir format.

    *forward_kwargs* are forwarded to ``torchair.dynamo_export`` as ``**kwargs``
    so tracing runs ``model(*inputs, **forward_kwargs)`` (see torchair's
    ``dynamo_export`` which calls ``model(*args, **kwargs)`` internally).
    Required when the forward signature has non-tensor args (e.g. ``run_mode``)
    whose default would diverge from what the caller just evaluated
    (e.g. NPU vs simulation mode).
    """
    try:
        import torchair
    except ImportError as e:
        raise ImportError(
            "export_to_torchair requires torchair package"
        ) from e

    path = Path(path)

    torchair.dynamo_export(
        *inputs,
        model=model,
        export_path=str(path.parent),
        export_name=str(path.stem),
        **(forward_kwargs or {}),
    )

    print(f"Exported TorchAir model to {path}")
