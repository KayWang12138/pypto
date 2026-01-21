import atexit
import os
from typing import Optional

import pypto
import torch

from ..log import get_logger

logger = get_logger("triton_pypto.device", "TRITON_PYPTO")


class State:
    auto_finalize_set = False
    device_set = False


def initialize(device_id: Optional[int] = None, auto_finalize: bool = True) -> None:
    if not State.device_set:
        if device_id is None:
            device_id = os.environ.get("TILE_FWK_DEVICE_ID", None) or os.environ.get("TILE_FWK_STEST_DEVICE_ID", None)
        if device_id is not None and hasattr(torch, "npu"):
            logger.info("Device %s", device_id)
            torch.npu.set_device(int(device_id))
        State.device_set = True
    else:
        logger.info("Device is already set")
    pypto.runtime._device_init()
    if auto_finalize and not State.auto_finalize_set:
        atexit.register(finalize)
        State.auto_finalize_set = True


def run_once(*tensors) -> None:
    pypto.runtime._device_run_once_data_from_host(*tensors)


def finalize() -> None:
    pypto.runtime._device_fini()
