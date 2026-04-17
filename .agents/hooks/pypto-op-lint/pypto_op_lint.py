#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Backward-compatible entry point for pypto-op-lint."""

import importlib
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_pkg = importlib.import_module("pypto_op_lint")
_obs = importlib.import_module("pypto_op_lint.observability")

LOGS_DIR = getattr(_obs, "LOGS_DIR", None)
LOGS_EVENTS_FILE = getattr(_obs, "LOGS_EVENTS_FILE", None)


def _sync_obs():
    _obs.LOGS_DIR = LOGS_DIR
    _obs.LOGS_EVENTS_FILE = LOGS_EVENTS_FILE


def __getattr__(name: str) -> Any:
    return getattr(_pkg, name)


def _wrap(name: str):
    target = getattr(_pkg, name)

    def wrapped(*args, **kwargs):
        _sync_obs()
        return target(*args, **kwargs)

    return wrapped


_build_context = _wrap("_build_context")
_run_checks = _wrap("_run_checks")
hook_post_edit = _wrap("hook_post_edit")
hook_post_bash = _wrap("hook_post_bash")
hook_pre_edit_backup = _wrap("hook_pre_edit_backup")
hook_stop = _wrap("hook_stop")
_parse_front_matter = _wrap("_parse_front_matter")
_validate_doc_schema = _wrap("_validate_doc_schema")
main = _wrap("main")

if __name__ == "__main__":
    raise SystemExit(main())
