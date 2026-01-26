import base64, io, zipfile, os
from pathlib import Path
import onnx
from typing import Union

def extract_attr_from_onnx(
    onnx_model: str,
    domain: str,
    op_type: str,
    attr_names: Union[list[str], tuple[str]], #extract any of these
):
    for node in onnx_model.graph.node:
        if node.domain == domain and node.op_type == op_type:
            attrs = {a.name: a for a in node.attribute}

            attr = None
            for name in attr_names:
                if name in attrs:
                    attr = attrs[name]
                    break
            if attr is None:
                raise KeyError(f"Could not find any of {attr_names} on {domain}::{op_type}")

            return attr

    raise ValueError(f"No node found for {domain}::{op_type}")


def extract_zip_from_onnx(
    onnx_model: str,
    domain: str,
    op_type: str,
    b64_attr_names: Union[list[str], tuple[str]],
    out_dir: Union[str, None] = None,
):
    b64_attr = extract_attr_from_onnx(
        onnx_model=onnx_model,
        domain=domain,
        op_type=op_type,
        attr_names=b64_attr_names,
    )

    if b64_attr.type != onnx.AttributeProto.STRING:
        raise TypeError(f"{b64_attr.name} is not a STRING attribute")

    b64 = b64_attr.s.decode("utf-8", errors="strict").strip()
    if not b64:
        raise ValueError(f"{b64_attr.name} is empty")

    zip_bytes = base64.b64decode(b64.encode("ascii"))
    zbuf = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(zbuf, "r") as zf:
        names = zf.namelist()

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            zf.extractall(out_dir)
            return {"extracted_to": out_dir, "files": names}

        contents = {name: zf.read(name) for name in names}
        return {"files": names, "contents": contents}

def extract_pypto_meta_from_onnx(
    onnx_model: str,
    domain: str,
    op_type: str,
):
    return extract_attr_from_onnx(
        onnx_model=onnx_model,
        domain=domain,
        op_type=op_type,
        attr_names=("pypto_meta",),
    )

