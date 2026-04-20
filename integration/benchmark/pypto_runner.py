#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
"""pypto 7 阶段 agent 工作流子进程调度器.

通过 ``opencode run --agent pypto-op-orchestrator`` 在 pypto 仓内启动 7 阶段
agent 工作流, 等待算子产物落到 ``custom/{op}/`` 后返回结果.

设计要点:
- ``opencode`` 是外部 CLI; 我们不做 IPC, 只通过 ``stdout/stderr`` 抓日志, 通过
  目标产物文件存在性 + ``.orchestrator_state.json`` 状态判断成功/失败.
- 单 case 一次子进程调用; 多 case 并发交给上层调度.
- 设备号通过 ``TILE_FWK_DEVICE_ID`` 环境变量隔离.
- 子进程 stdout 通过 ``Popen`` + 后台线程**逐行落盘**到 ``log_file``,
  ``tail -f log_file`` 能实时看到 agent 进度 (默认不传 ``--print-logs``,
  避免 opencode 内部 server log 把真正的 agent 输出淹没).
- 主线程仅做 ``timeout_sec`` 硬墙轮询; 不做 early-stop, 让 agent 跑完它
  自己的 7 阶段状态机 (含 Stage 5↔6 修正循环和 Stage 7 性能调优).
- prompt 含 ``{op}_pypto_impl.py`` (ModelNew 包装) 的硬约束; runner 不替
  agent 兜底生成. 若 agent 没产出, ``ARTIFACT_MISSING``, 由调用方决策.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TextIO


class PyptoRunStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ARTIFACT_MISSING = "artifact_missing"
    BLOCKED = "blocked"
    OPENCODE_NOT_FOUND = "opencode_not_found"
    SUBPROCESS_ERROR = "subprocess_error"


@dataclass
class PyptoRunResult:
    op_name: str
    status: PyptoRunStatus
    workdir: Path
    artifacts: Dict[str, Path] = field(default_factory=dict)
    log_file: Optional[Path] = None
    duration_sec: float = 0.0
    message: str = ""
    orchestrator_state: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.status in (PyptoRunStatus.SUCCESS, PyptoRunStatus.SKIPPED)

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "status": self.status.value,
            "workdir": str(self.workdir),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "log_file": str(self.log_file) if self.log_file else None,
            "duration_sec": round(self.duration_sec, 2),
            "message": self.message,
            "orchestrator_state": self.orchestrator_state,
        }


# ────────────────────────────────────────────────────────────
# 产物清单
# ────────────────────────────────────────────────────────────

# 必须存在才算成功的最少产物 (按 plan 与 pypto-op-orchestrator 工件契约).
REQUIRED_ARTIFACTS = (
    "{op}_impl.py",
    "{op}_golden.py",
    "test_{op}.py",
)

# KernelBench 桥接所需的额外产物. agent 应根据外层 prompt 里的硬约束自行产出.
KERNELBENCH_ARTIFACTS = (
    "{op}_pypto_impl.py",
)


def expected_artifact_paths(op_name: str, op_dir: Path,
                            need_kernelbench: bool = True) -> Dict[str, Path]:
    """返回 ``{key: path}`` 形式的预期产物表."""
    out: Dict[str, Path] = {}
    for tpl in REQUIRED_ARTIFACTS:
        rel = tpl.format(op=op_name)
        out[rel] = op_dir / rel
    if need_kernelbench:
        for tpl in KERNELBENCH_ARTIFACTS:
            rel = tpl.format(op=op_name)
            out[rel] = op_dir / rel
    out["SPEC.md"] = op_dir / "SPEC.md"
    return out


def all_artifacts_present(artifacts: Dict[str, Path]) -> List[str]:
    """返回缺失的 artifact 文件名列表; 空列表表示齐全."""
    return [k for k, p in artifacts.items() if not p.exists()]


# ────────────────────────────────────────────────────────────
# Prompt 渲染
# ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
请以 pypto-op-orchestrator 角色为算子 `{op_name}` 跑完 7 阶段端到端开发流程.

工作目录: `{op_dir_rel}/`
SPEC.md (已就绪, 请直接读取并按其内容推进): `{op_dir_rel}/SPEC.md`

请严格按 pypto 现有 7 阶段产出以下标准产物 (按 pypto-op-orchestrator 自带规范):
  - SPEC.md (已存在)
  - API_REPORT.md
  - DESIGN.md
  - {op_name}_golden.py
  - {op_name}_impl.py            (必须导出 {op_name}_wrapper)
  - test_{op_name}.py
  - README.md
  - .orchestrator_state.json     (整体状态: SUCCESS 或 BLOCKED_*)

================================================================
【外部桥接附加要求 -- 仅本次任务额外完成, 不要修改 pypto 内置 SKILL/agent】
================================================================

Stage 5 完成且自验证通过后, 请在同一目录 `{op_dir_rel}/` 下额外生成一个文件
`{op_name}_pypto_impl.py`. 该文件是给下游 KernelBench 风格评测器
(KernelVerifier) 的入口, 内容必须与下方代码模板严格一致
(只替换 `{op_name}` 占位符, 类签名 / forward 行为 / import 关系不得改动):

----------------------------------------------------------------------
import torch
import torch.nn as nn

from {op_name}_impl import {op_name}_wrapper


class ModelNew(nn.Module):
    \"\"\"KernelBench-compatible entry, forwards to {op_name}_wrapper.\"\"\"

    def __init__(self, *init_args, **init_kwargs):
        super().__init__()
        self._init_args = init_args
        self._init_kwargs = init_kwargs

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return {op_name}_wrapper(*inputs)
----------------------------------------------------------------------

ModelNew 文件硬约束:
1. 必须是新文件 `{op_name}_pypto_impl.py`, 与 `{op_name}_impl.py` 同目录.
2. 必须 `from {op_name}_impl import {op_name}_wrapper` (不得内联 wrapper 实现).
3. `__init__` 签名固定为 `(*init_args, **init_kwargs)`, 兼容 KernelBench
   `get_init_inputs()` 返回的任意展开 (大多数用例为空, 但保留通用签名).
4. `forward` 内部禁止复制 kernel 逻辑或调用任何其他实现, 只能转发到
   `{op_name}_wrapper`.
5. 不得修改已生成的 `{op_name}_impl.py` 的导出符号或函数签名.

ModelNew 文件自检 (必须通过):
   python -c "import sys; sys.path.insert(0, '{op_dir_rel}'); from {op_name}_pypto_impl import ModelNew; m = ModelNew(); print(type(m).__name__)"
预期输出: `ModelNew`.

================================================================
其它约束:
- 所有产物落在 `{op_dir_rel}/`, 不要写到其它目录.
- 走真实 NPU 验证 (有可用 NPU 时), 不要降级到 sim 模式.
- 完成或阻塞时, 在最后一行打印一条机读标记:
  `[BENCHMARK_DONE] op={op_name} state=<SUCCESS|BLOCKED_*> artifacts=impl,golden,test,pypto_impl`
  缺失任一产物时, 在 artifacts= 后只列出实际存在的项.

请立即开始, 不要再问我问题.
"""


def render_prompt(op_name: str, op_dir_rel: str) -> str:
    return _PROMPT_TEMPLATE.format(op_name=op_name, op_dir_rel=op_dir_rel)


# ────────────────────────────────────────────────────────────
# 子进程调度
# ────────────────────────────────────────────────────────────

def _resolve_opencode(opencode_bin: str = "") -> Optional[str]:
    if opencode_bin:
        return opencode_bin if Path(opencode_bin).exists() else None
    return shutil.which("opencode")


def _read_orchestrator_state(op_dir: Path) -> Optional[dict]:
    state_file = op_dir / ".orchestrator_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ────────────────────────────────────────────────────────────
# 后台 stdout 排水线程 (实时落盘)
# ────────────────────────────────────────────────────────────

def _stream_stdout(proc: subprocess.Popen, log_handle: Optional[TextIO]) -> None:
    """把 ``proc.stdout`` 逐行写到 ``log_handle`` (实时落盘 + flush).

    设计:
    - 与 ``Popen(bufsize=1, text=True)`` 配套, 让 ``tail -f log_file`` 能实时
      看到 agent 输出. ``subprocess.run`` 默认全缓冲, 看不到进度 — 这是 runner
      改用 ``Popen`` 的唯一动机.
    - 子进程被 SIGTERM/SIGKILL 时 ``readline`` 会拿到 ``''`` 然后退出, 不需要
      额外信号.
    """
    if log_handle is None or proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            log_handle.write(line)
            log_handle.flush()
    except (ValueError, OSError):
        pass


def _kill_process_group(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    """SIGTERM 子进程组, 不退就 SIGKILL.

    ``Popen(start_new_session=True)`` 让 opencode 跟它 spawn 的所有 node 子进程
    在同一个 pgid 下, 一次性能干掉. 仅在 ``timeout_sec`` 硬超时时触发.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

# 主循环轮询 (检查进程是否退出 / 是否硬超时) 频率.
_POLL_INTERVAL_SEC = 5


def run_pypto_workflow(
    op_name: str,
    pypto_repo_root: Path,
    workdir_root: str = "custom",
    *,
    opencode_bin: str = "",
    agent: str = "pypto-op-orchestrator",
    timeout_sec: int = 5400,
    device_id: Optional[int] = None,
    log_file: Optional[Path] = None,
    output_format: str = "default",
    extra_env: Optional[Dict[str, str]] = None,
    skip_if_done: bool = True,
    need_kernelbench: bool = True,
) -> PyptoRunResult:
    """跑一次 pypto 7 阶段工作流.

    架构:
    - ``subprocess.Popen`` + 后台线程逐行排水 stdout 到 ``log_file``,
      ``tail -f log_file`` 立刻能看到 agent 进度.
    - 主线程每 ``_POLL_INTERVAL_SEC`` 秒检查 2 件事:
        (1) 子进程是否退出
        (2) 是否到 ``timeout_sec`` (硬墙)
      不做任何 artifact-based 的 early-stop — agent 自己有 Stage 5↔6 修正
      循环和 Stage 7 性能调优中止条件 (10 轮 / 连续 3 次无提升), 切断它就是
      切断它的自我修正能力.
    - 子进程退出后, 用 ``expected_artifact_paths`` 检查产物齐全性. 缺
      ``{op}_pypto_impl.py`` 也算 ``ARTIFACT_MISSING`` — runner 不兜底,
      由 prompt 里的硬约束驱动 agent 自己产出.

    Args:
        op_name: 算子名 (= ``custom/{op_name}/`` 子目录名).
        pypto_repo_root: pypto 仓根, 子进程 cwd.
        workdir_root: 算子产物根目录 (相对 pypto 仓根).
        opencode_bin: opencode 可执行路径; 留空则按 PATH 查找.
        agent: opencode agent 名.
        timeout_sec: 子进程整体超时 (硬墙). 默认 90 min, 覆盖 7 阶段
            含 Stage 7 性能调优 10 轮迭代.
        device_id: 注入 ``TILE_FWK_DEVICE_ID``; ``None`` 时不覆盖外部已设值.
        log_file: 子进程 stdout+stderr 落地; ``None`` 时不落盘.
        output_format: ``opencode run --format`` 参数.
        extra_env: 额外环境变量, 优先级最高.
        skip_if_done: 若产物齐全且 state file 存在则跳过.
        need_kernelbench: 是否要求 ``{op}_pypto_impl.py`` 也存在才算齐全.

    Returns:
        ``PyptoRunResult``.
    """
    pypto_repo_root = pypto_repo_root.resolve()
    op_dir = pypto_repo_root / workdir_root / op_name
    op_dir_rel = f"{workdir_root}/{op_name}"
    artifacts = expected_artifact_paths(op_name, op_dir, need_kernelbench=need_kernelbench)

    # 断点续跑: 已完成则跳过
    if skip_if_done:
        missing = all_artifacts_present(artifacts)
        state = _read_orchestrator_state(op_dir)
        state_success = bool(state and state.get("current_stage") and state.get("stage_status"))
        if not missing and state_success:
            return PyptoRunResult(
                op_name=op_name,
                status=PyptoRunStatus.SKIPPED,
                workdir=op_dir,
                artifacts=artifacts,
                duration_sec=0.0,
                message="所有产物齐全, 且 .orchestrator_state.json 存在; 跳过.",
                orchestrator_state=state,
            )

    opencode = _resolve_opencode(opencode_bin)
    if opencode is None:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.OPENCODE_NOT_FOUND,
            workdir=op_dir,
            message="opencode 可执行未找到; 请安装或在 configs/default.yaml 指定 pypto.opencode_bin.",
        )

    spec_path = op_dir / "SPEC.md"
    if not spec_path.exists():
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.ARTIFACT_MISSING,
            workdir=op_dir,
            message=f"SPEC.md 不存在: {spec_path} (应由 case_loader 预先写入).",
        )

    prompt = render_prompt(op_name, op_dir_rel)

    # 注意: 不加 --print-logs! 它会把 opencode 内部 server/storage/agent 的
    # 每一笔操作都以单行平铺 JSON 倒进 stdout (一行常 5000+ 字符), 单 case
    # 跑下来 log 能轻松 20MB+ 还淹没掉 agent 真正的思考输出. 去掉之后只剩
    # agent message stream, 人能读, tail -f 也清爽.
    cmd = [
        opencode, "run",
        "--agent", agent,
        "--format", output_format,
        prompt,
    ]

    env = os.environ.copy()
    if device_id is not None:
        env["TILE_FWK_DEVICE_ID"] = str(device_id)
    if extra_env:
        env.update(extra_env)

    log_handle: Optional[TextIO] = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8", buffering=1)
        log_handle.write(f"$ cd {pypto_repo_root}\n")
        log_handle.write(f"$ TILE_FWK_DEVICE_ID={env.get('TILE_FWK_DEVICE_ID', '<unset>')} "
                         f"{' '.join(cmd[:5])} <prompt>\n")
        log_handle.write("# (实时 stdout 从下行起追加; tail -f 可观察 agent 进度)\n")
        log_handle.flush()

    start = time.monotonic()
    deadline = start + timeout_sec
    timed_out = False

    proc = subprocess.Popen(
        cmd,
        cwd=str(pypto_repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    reader = threading.Thread(target=_stream_stdout, args=(proc, log_handle), daemon=True)
    reader.start()

    try:
        while True:
            if proc.poll() is not None:
                break

            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                if log_handle is not None:
                    log_handle.write(f"\n[TIMEOUT] {timeout_sec}s 硬超时, SIGTERM 进程组\n")
                _kill_process_group(proc)
                break

            time.sleep(_POLL_INTERVAL_SEC)

        # 等 reader 把最后一段 stdout 排干
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
        reader.join(timeout=5)
    finally:
        if log_handle is not None:
            log_handle.write(f"\n[run finished, returncode={proc.returncode}, "
                             f"timed_out={timed_out}]\n")
            log_handle.close()

    duration = time.monotonic() - start
    state = _read_orchestrator_state(op_dir)
    missing = all_artifacts_present(artifacts)

    if timed_out:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.TIMEOUT,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            duration_sec=duration,
            message=f"opencode run 超时 (>{timeout_sec}s); 缺失产物: {missing or '(齐全)'}",
            orchestrator_state=state,
        )

    if missing:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.ARTIFACT_MISSING,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            duration_sec=duration,
            message=f"opencode 退出 code={proc.returncode}, 缺少产物: {missing}",
            orchestrator_state=state,
        )

    if proc.returncode != 0:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.SUBPROCESS_ERROR,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            duration_sec=duration,
            message=f"opencode 异常退出 code={proc.returncode}, 但产物齐全 (可能仍可用).",
            orchestrator_state=state,
        )

    return PyptoRunResult(
        op_name=op_name,
        status=PyptoRunStatus.SUCCESS,
        workdir=op_dir,
        artifacts=artifacts,
        log_file=log_file,
        duration_sec=duration,
        message="产物齐全, opencode 正常退出.",
        orchestrator_state=state,
    )


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run pypto 7-stage workflow for a single op")
    parser.add_argument("op_name")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--workdir-root", default="custom")
    parser.add_argument("--timeout-sec", type=int, default=5400)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--no-skip", action="store_true",
                        help="即使产物齐全也强制重跑")
    args = parser.parse_args()

    result = run_pypto_workflow(
        op_name=args.op_name,
        pypto_repo_root=args.repo_root,
        workdir_root=args.workdir_root,
        timeout_sec=args.timeout_sec,
        device_id=args.device,
        log_file=args.log_file,
        skip_if_done=not args.no_skip,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_main_cli())
