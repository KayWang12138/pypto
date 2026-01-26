import inspect
import base64, io, zipfile
from pathlib import Path

META_KEY_ZIP_FILE = "zip_file"

def zip_source_file_to_b64(fn):
    src_file = inspect.getsourcefile(fn)
    if not src_file:
        return None, None

    src_path = Path(src_file)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(src_path.name, src_path.read_bytes())

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return str(src_path), b64


# 通过@pypto_op装饰器输出pypto算子相关信息 
def pypto_op(incl_src=False, **meta):
    def deco(fn):
        meta_local = dict(meta)
        if incl_src:
            src_path, b64 = zip_source_file_to_b64(fn)
            if src_path and b64:
                meta_local[META_KEY_ZIP_FILE] = b64
        fn.__pypto_meta__ = meta_local
        return fn
    return deco

