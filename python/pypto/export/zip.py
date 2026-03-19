import base64
import inspect
import io
import os
import zipfile

from pathlib import Path
from typing import Optional

__all__: tuple[str, ...] = ()


def _zip_file_to_b64(file_path: str):
    """Zip a single file and return its base64-encoded contents."""
    file_path = Path(file_path)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(file_path.name, file_path.read_bytes())

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64


def _zip_source_file_to_b64(fn):
    """Zip the source file of the given function and return path with base64-encoded contents."""
    src_path = inspect.getsourcefile(fn)
    if not src_path:
        return None, None
    return src_path, _zip_file_to_b64(src_path)


def _zip_pto_file_to_b64(pto_path: str):
    """Zip a .pto file and return base64."""
    return _zip_file_to_b64(pto_path)


def _zip_dir_to_b64(dir: str, prefix: Optional[str] = None):
    """Zip a directory (optionally under prefix) and return base64."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:  # is allowZip64 required ?
        for root, dirs, files in os.walk(dir):
            if prefix is not None and not prefix in root:
                dirs[:] = [d for d in dirs if d.startswith(prefix)]
            for file in files:
                file_path = os.path.join(root, file)
                if prefix is not None and not prefix in file_path:
                    continue
                archive_file_path = os.path.relpath(file_path, start=dir)
                zf.write(file_path, arcname=archive_file_path)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64


def _zip_kernel_dir_to_b64(kernel_dir: str):
    """Zip only the kernel subdir of kernel_dir and return base64."""
    return _zip_dir_to_b64(kernel_dir, prefix="kernel")


def _zip_cpp_sources_dir_to_b64(cpp_sources_dir: str):
    """Zip a C++ sources directory and return base64."""
    return _zip_dir_to_b64(cpp_sources_dir)


def _unzip_b64_to_dir(b64: str, out_dir: str):
    """Decode base64 zip and extract to out_dir; return manifest. If out_dir is None, return in-memory contents."""
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
