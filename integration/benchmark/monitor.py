#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""算子运行监控 CLI.

用法:
    python3 -m integration.benchmark.monitor
    python3 -m integration.benchmark.monitor monitor
    python3 -m integration.benchmark.monitor monitor --once

设计要点:
- 默认模式前台启动 ``run_kernelbench.py`` 并持续写 ``state.json``.
- ``monitor`` 子命令仅读取指定 ``BENCHMARK_MONITOR_DIR`` 下的状态文件渲染看板.
- 用户通过 ``Ctrl+C`` 中断前台运行; 不再提供后台 start/stop 模式.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from integration.benchmark.case_loader import derive_op_name
from integration.benchmark.run_kernelbench import (
    DEFAULT_CONFIG_PATH,
    discover_cases,
    load_yaml_config,
    parse_cases_by_level,
    parse_csv_str_list,
)


def _load_wcswidth():
    install_cmd = [sys.executable, "-m", "pip", "install", "wcwidth"]

    def _manual_install_hint() -> str:
        return (
            f"请手动执行: {' '.join(install_cmd)}\n"
            f"如果当前 Python 没有 pip, 请先执行: {sys.executable} -m ensurepip --upgrade"
        )

    try:
        from wcwidth import wcswidth as imported_wcswidth
        return imported_wcswidth
    except ModuleNotFoundError as first_error:
        print(
            "[benchmark.monitor] Python package 'wcwidth' not found; "
            f"trying automatic install: {' '.join(install_cmd)}",
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as install_error:
            raise SystemExit(
                "缺少 Python 依赖 wcwidth, 且自动安装启动失败.\n"
                f"{_manual_install_hint()}\n"
                f"原始 import 错误: {first_error}\n"
                f"自动安装错误: {install_error}"
            ) from install_error

        if proc.returncode != 0:
            install_output = "\n".join(
                part.strip()
                for part in (proc.stdout, proc.stderr)
                if part and part.strip()
            )
            if len(install_output) > 2000:
                install_output = install_output[-2000:]
            raise SystemExit(
                "缺少 Python 依赖 wcwidth, 且自动安装失败.\n"
                f"{_manual_install_hint()}\n"
                f"pip 退出码: {proc.returncode}\n"
                f"pip 输出:\n{install_output}"
            ) from first_error

        try:
            from wcwidth import wcswidth as installed_wcswidth
            return installed_wcswidth
        except ModuleNotFoundError as second_error:
            raise SystemExit(
                "自动安装 wcwidth 已返回成功, 但当前 Python 仍无法 import.\n"
                f"{_manual_install_hint()}\n"
                f"import 错误: {second_error}"
            ) from second_error


wcswidth = _load_wcswidth()


_STATE_DIR_DEFAULT = "/tmp/pypto-benchmark-monitor"
_POLL_SEC = 3
_DEFAULT_TIMEOUT = 7200
_OVER_THRESHOLD = 5400
logger = logging.getLogger("benchmark.monitor")

_OP_RE_TITLE = re.compile(r"pypto-bench:(pypto|verifier):([A-Za-z0-9_.:-]+):")
_OP_RE_WORKFLOW = re.compile(r"算子 `(\w+)`")
_OP_RE_VERIFIER = re.compile(r"op_name\s*=\s*(\w+)")


# ────────────────────────────────────────────────────────────
# 路径约定
# ────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    env_dir = os.environ.get("BENCHMARK_MONITOR_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path(_STATE_DIR_DEFAULT)


def _benchmark_log_root() -> Optional[Path]:
    root = os.environ.get("BENCHMARK_LOG_DIR", "").strip()
    if not root:
        state_dir = _state_dir()
        if state_dir.name == "monitor_state":
            return state_dir.parent
        return None
    return Path(root).expanduser()


def _state_file() -> Path:
    return _state_dir() / "state.json"


# ────────────────────────────────────────────────────────────
# 进程工具 (Linux /proc)
# ────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _read_ppid(pid: int) -> int:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        i = data.rfind(")")
        if i < 0:
            return 0
        return int(data[i + 2:].split()[1])
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return 0


def _descendants(root: int) -> List[int]:
    """BFS 获取 root 的全部后代 PID."""
    ppid_map: Dict[int, int] = {}
    try:
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                pid = int(entry.name)
                ppid = _read_ppid(pid)
                if ppid:
                    ppid_map[pid] = ppid
    except OSError:
        return []

    result: List[int] = []
    queue = [root]
    visited = {root}
    while queue:
        parent = queue.pop(0)
        for pid, ppid in ppid_map.items():
            if ppid == parent and pid not in visited:
                visited.add(pid)
                result.append(pid)
                queue.append(pid)
    return result


def _detect_opencode_phase(cmd: str) -> str:
    if "pypto-bench:verifier:" in cmd or "pypto-kernel-validator" in cmd:
        return "verifier"
    if "pypto-bench:pypto:" in cmd or "pypto-op-orchestrator" in cmd:
        return "pypto"
    return "unknown"


def _find_opencode_procs(root_pid: int) -> Dict[Tuple[str, str], int]:
    """扫描 root_pid 的后代, 返回 {(op_name, phase): pid}."""
    out: Dict[Tuple[str, str], int] = {}
    for pid in _descendants(root_pid):
        cmd = _read_cmdline(pid)
        if "opencode" not in cmd:
            continue
        title_match = _OP_RE_TITLE.search(cmd)
        if title_match:
            phase, op_name = title_match.groups()
            out[(op_name, phase)] = pid
            continue

        phase = _detect_opencode_phase(cmd)
        m = _OP_RE_WORKFLOW.search(cmd) or _OP_RE_VERIFIER.search(cmd)
        if m and phase != "unknown":
            out[(m.group(1), phase)] = pid
    return out


def _kill_tree(root_pid: int, grace_sec: float = 5.0) -> None:
    """SIGTERM root 及其全部后代, 超时则 SIGKILL."""
    desc = _descendants(root_pid)
    for pid in reversed(desc):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        os.kill(root_pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if not _pid_alive(root_pid):
            return
        time.sleep(0.3)
    for pid in reversed(desc) + [root_pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _kill_process_group_by_pid(pid: int, grace_sec: float = 5.0) -> None:
    """按 pid 所属 pgid 清理整组进程, 用于 start_new_session=True 的 opencode."""
    if not _pid_alive(pid):
        return
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.3)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _kill_opencode_groups_from_state(state: Dict[str, Any]) -> None:
    """优先按进程组清理 state 中已记录的 opencode 进程, 避免 node 子进程残留."""
    seen: set[int] = set()
    for op in state.get("operators", []):
        pids: List[Any] = [op.get("opencode_pid")]
        phases = op.get("phases") or {}
        for phase in ("pypto", "verifier"):
            if isinstance(phases.get(phase), dict):
                pids.append(phases[phase].get("pid"))
        for pid in pids:
            if isinstance(pid, int) and pid > 0 and pid not in seen:
                seen.add(pid)
                _kill_process_group_by_pid(pid)


# ────────────────────────────────────────────────────────────
# 状态文件 I/O
# ────────────────────────────────────────────────────────────

def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_state() -> Optional[Dict[str, Any]]:
    p = _state_file()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: Dict[str, Any]) -> None:
    p = _state_file()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")
    tmp.rename(p)


def _stream_to_outputs(proc: subprocess.Popen, file_handle: Optional[Any], tee_stdout: bool) -> None:
    """把子进程 stdout 同时写到日志文件和当前 stdout."""
    if proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            if file_handle is not None:
                file_handle.write(line)
                file_handle.flush()
            if tee_stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
    except (ValueError, OSError):
        pass


# ────────────────────────────────────────────────────────────
# 开发状态判定
# ────────────────────────────────────────────────────────────

def _elapsed_since(ts: str) -> float:
    try:
        return (dt.datetime.now() - dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        try:
            return (dt.datetime.now() - dt.datetime.fromisoformat(ts)).total_seconds()
        except ValueError:
            return 0.0


def _compute_dev_status(op: Dict[str, Any], timeout_sec: int) -> str:
    """基于进程与报告信息计算算子开发状态."""
    rs = op.get("result_status")
    if rs == "success":
        return "已完成"
    if rs == "pypto_failed":
        if op.get("pypto_status") == "timeout":
            return "PyPTO超时"
        return "PyPTO失败"
    if rs == "verify_failed":
        return "Verifier失败"
    if rs == "baseline_failed":
        return "Baseline失败"
    if rs == "verify_error":
        return "Verifier异常"

    phases = op.get("phases") or {}
    for phase, label in (("verifier", "Verifier验证中"), ("pypto", "PyPTO生成中")):
        info = phases.get(phase) if isinstance(phases, dict) else None
        if not isinstance(info, dict):
            continue
        pid = info.get("pid")
        started = info.get("started_at")
        elapsed = _elapsed_since(started) if started else 0.0
        if pid and _pid_alive(pid):
            if phase == "pypto" and elapsed >= timeout_sec:
                return "PyPTO超时"
            if phase == "pypto" and elapsed >= _OVER_THRESHOLD:
                return "PyPTO超过5400s"
            return label

    pid = op.get("opencode_pid")
    started = op.get("started_at")
    phase = op.get("phase")
    phase_status = op.get("phase_status")
    if phase == "prepare" and phase_status == "running":
        return "准备中"
    if phase == "pypto":
        if phase_status == "skipped":
            return "PyPTO跳过"
        if phase_status == "running":
            return "PyPTO生成中"
    if phase == "verifier" and phase_status == "running":
        return "Verifier验证中"

    if not pid and not started:
        return "未开始"

    elapsed = _elapsed_since(started) if started else 0.0
    if pid and _pid_alive(pid):
        if elapsed >= timeout_sec:
            return "超时"
        if elapsed >= _OVER_THRESHOLD:
            return "未超时但已超过5400s"
        return "正在进行"

    return "正在进行"


# ────────────────────────────────────────────────────────────
# benchmark 命令构建
# ────────────────────────────────────────────────────────────

def _build_bench_cmd(cases: str, skip_gen: bool, report_dir: str,
                     log_dir: str, timeout_sec: int) -> List[str]:
    """复用 test-integration.sh 的命令构建逻辑."""
    cmd = [
        sys.executable, "-m", "integration.benchmark.run_kernelbench",
        "--cases", cases,
        "--mode", os.environ.get("MODE", "performance"),
        "--devices", os.environ.get("TILE_FWK_DEVICE_ID", "0"),
        "--concurrency", os.environ.get("CONCURRENCY", "1"),
        "--verifier-mode", os.environ.get("VERIFIER_MODE", "opencode"),
        "--skill-timeout", os.environ.get("SKILL_TIMEOUT", "1500"),
        "--skill-retry", os.environ.get("SKILL_RETRY", "2"),
        "--skill-retry-interval", os.environ.get("SKILL_RETRY_INTERVAL", "600"),
        "--timeout-sec", str(timeout_sec),
        "--report-dir", report_dir,
        "--log-dir", log_dir,
        "--log-level", "INFO",
    ]
    model = os.environ.get("OPENCODE_MODEL", "")
    if model:
        cmd += ["--opencode-model", model]
    if os.environ.get("SKIP_STAGE7_PERF_TUNE", "0") == "1":
        cmd.append("--skip-stage7-perf-tune")
    for env_key, flag in [("VERIFY_RTOL", "--verify-rtol"),
                          ("VERIFY_ATOL", "--verify-atol")]:
        v = os.environ.get(env_key, "")
        if v:
            cmd += [flag, v]
    if skip_gen:
        cmd.append("--skip-pypto-gen")
    return cmd


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_pypto_repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent.parent
    if (here / "pyproject.toml").exists():
        return here
    return Path.cwd()


def _resolve_bench_dir() -> Path:
    yaml_cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
    yaml_bench = (yaml_cfg.get("bench_dir") or "").strip()
    if yaml_bench:
        bench_dir = Path(yaml_bench).expanduser()
    else:
        bench_dir = Path(__file__).resolve().parent / ".cache" / "KernelBench" / "KernelBench"

    if not bench_dir.is_absolute():
        repo_root = _resolve_pypto_repo_root()
        cwd_candidate = (Path.cwd() / bench_dir).resolve()
        repo_candidate = (repo_root / bench_dir).resolve()
        if cwd_candidate.exists():
            bench_dir = cwd_candidate
        elif repo_candidate.exists():
            bench_dir = repo_candidate
        else:
            bench_dir = cwd_candidate
    return bench_dir


def _resolve_requested_cases(cases: str) -> List[Dict[str, str]]:
    """把原始 --cases 解析成 run_kernelbench 实际使用的 case/op 映射."""
    yaml_cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
    level_cfg = yaml_cfg.get("level", "level1")
    if isinstance(level_cfg, (list, tuple)):
        default_levels = [str(item).strip() for item in level_cfg if str(item).strip()]
    else:
        default_levels = parse_csv_str_list(str(level_cfg))
    cases_by_level = parse_cases_by_level(cases, default_levels or ["level1"])
    bench_dir = _resolve_bench_dir()
    case_paths: List[Path] = []
    for level in cases_by_level:
        requested = cases_by_level.get(level)
        case_paths.extend(discover_cases(bench_dir / level, requested=requested, limit=None))
    resolved: List[Dict[str, str]] = []
    for path in case_paths:
        resolved.append({
            "requested": path.stem.split("_", 1)[0] if "_" in path.stem else path.stem,
            "level": path.parent.name,
            "case_id": path.stem,
            "op_name": derive_op_name(path.stem),
        })
    return resolved


# ────────────────────────────────────────────────────────────
# 报告目录扫描
# ────────────────────────────────────────────────────────────

def _scan_reports(report_dir: str, report_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """扫描 report_dir 下各算子的 result.json."""
    out: Dict[str, Dict[str, Any]] = {}
    rd = Path(report_dir)
    if not rd.exists():
        return out
    for key in report_keys:
        rf = rd / key / "result.json"
        if rf.exists():
            try:
                out[key] = json.loads(rf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def _scan_phase_states(report_dir: str, report_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """扫描 report_dir 下各算子的 phase_state.json."""
    out: Dict[str, Dict[str, Any]] = {}
    rd = Path(report_dir)
    if not rd.exists():
        return out
    for key in report_keys:
        pf = rd / key / "phase_state.json"
        if pf.exists():
            try:
                out[key] = json.loads(pf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def _empty_phase() -> Dict[str, Any]:
    return {
        "pid": None,
        "started_at": None,
        "ended_at": None,
        "duration_sec": 0.0,
        "status": None,
    }


def _ensure_phase(op: Dict[str, Any], phase: str) -> Dict[str, Any]:
    phases = op.setdefault("phases", {})
    if phase not in phases or not isinstance(phases.get(phase), dict):
        phases[phase] = _empty_phase()
    return phases[phase]


def _update_phase_pid(op: Dict[str, Any], phase: str, pid: int, first_seen: str) -> None:
    info = _ensure_phase(op, phase)
    if info.get("pid") != pid:
        info["pid"] = pid
        info["started_at"] = first_seen
        info["ended_at"] = None
        info["duration_sec"] = 0.0
    if _pid_alive(pid):
        info["duration_sec"] = round(_elapsed_since(info.get("started_at") or first_seen), 1)
        info["status"] = "running"
    elif not info.get("ended_at"):
        info["ended_at"] = _now()


# ────────────────────────────────────────────────────────────
# 守护进程
# ────────────────────────────────────────────────────────────

_bench_proc: Optional[subprocess.Popen] = None
_daemon_state: Dict[str, Any] = {}


def _sigterm_handler(_signum: int, _frame: Any) -> None:
    global _bench_proc, _daemon_state
    _kill_opencode_groups_from_state(_daemon_state)
    if _bench_proc and _bench_proc.poll() is None:
        _kill_tree(_bench_proc.pid)
    _daemon_state["main_status"] = "未完成"
    _daemon_state["updated_at"] = _now()
    try:
        _write_state(_daemon_state)
    except OSError:
        pass
    sys.exit(0)


def _run_monitor_session(cases: str, skip_gen: bool, timeout_sec: int, *, tee_stdout: bool) -> int:
    """前台启动 benchmark 并周期性采集状态."""
    global _bench_proc, _daemon_state
    signal.signal(signal.SIGTERM, _sigterm_handler)

    sd = _state_dir()
    bench_root = _benchmark_log_root()
    if bench_root is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_root = sd / f"bench_{ts}"
    report_dir = bench_root / "report"
    log_dir = bench_root / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved_cases = _resolve_requested_cases(cases)
    case_list = [item["case_id"] for item in resolved_cases]
    multi_level = len({item.get("level", "") for item in resolved_cases}) > 1
    op_names = [item["op_name"] for item in resolved_cases]
    report_keys = [
        f"{item['level']}/{item['op_name']}" if multi_level else item["op_name"]
        for item in resolved_cases
    ]

    operators: List[Dict[str, Any]] = []
    for item in resolved_cases:
        operators.append({
            "case_id": item["case_id"],
            "level": item.get("level", ""),
            "report_key": (
                f"{item['level']}/{item['op_name']}" if multi_level else item["op_name"]
            ),
            "op_name": item["op_name"],
            "phase": "pending",
            "phase_status": "pending",
            "phase_message": "",
            "opencode_pid": None,
            "phases": {
                "pypto": _empty_phase(),
                "verifier": _empty_phase(),
            },
            "started_at": None,
            "ended_at": None,
            "duration_sec": 0.0,
            "dev_status": "未开始",
            "result_status": None,
            "pypto_status": None,
        })

    cmd = _build_bench_cmd(cases, skip_gen, str(report_dir), str(log_dir), timeout_sec)

    _daemon_state = {
        "daemon_pid": os.getpid(),
        "main_pid": None,
        "main_status": "正在进行",
        "main_exit_code": None,
        "started_at": _now(),
        "updated_at": _now(),
        "cases": case_list,
        "timeout_sec": timeout_sec,
        "report_dir": str(report_dir),
        "log_dir": str(log_dir),
        "operators": operators,
    }
    _write_state(_daemon_state)

    bench_log = open(bench_root / "benchmark.log", "w", encoding="utf-8", buffering=1)
    try:
        _bench_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except Exception:
        _daemon_state["main_status"] = "未完成"
        _daemon_state["main_exit_code"] = -1
        _daemon_state["updated_at"] = _now()
        _write_state(_daemon_state)
        bench_log.close()
        return 1

    _daemon_state["main_pid"] = _bench_proc.pid
    _write_state(_daemon_state)
    logger.info("benchmark started: pid=%s report_dir=%s log_dir=%s",
                _bench_proc.pid, report_dir, log_dir)

    reader = threading.Thread(
        target=_stream_to_outputs, args=(_bench_proc, bench_log, tee_stdout), daemon=True
    )
    reader.start()

    seen_pids: Dict[Tuple[str, str], Tuple[int, str]] = {}

    try:
        while True:
            rc = _bench_proc.poll()

            if rc is None:
                oc = _find_opencode_procs(_bench_proc.pid)
                for key, pid in oc.items():
                    if key not in seen_pids or seen_pids[key][0] != pid:
                        seen_pids[key] = (pid, _now())

            reports = _scan_reports(str(report_dir), report_keys)
            phase_states = _scan_phase_states(str(report_dir), report_keys)

            for op in _daemon_state["operators"]:
                name = op["op_name"]
                report_key = op.get("report_key") or name
                if report_key in phase_states:
                    ps = phase_states[report_key]
                    op["phase"] = ps.get("phase", op.get("phase", "pending"))
                    op["phase_status"] = ps.get("status", op.get("phase_status", ""))
                    op["phase_message"] = ps.get("message", "")
                    if ps.get("updated_at") and not op["started_at"]:
                        op["started_at"] = ps["updated_at"]
                    if ps.get("pypto_status"):
                        op["pypto_status"] = ps["pypto_status"]
                    if ps.get("verifier_status"):
                        op["verifier_status"] = ps["verifier_status"]

                active_pid: Optional[int] = None
                for phase in ("pypto", "verifier"):
                    key = (name, phase)
                    if key in seen_pids:
                        pid, first = seen_pids[key]
                        _update_phase_pid(op, phase, pid, first)
                        if _pid_alive(pid):
                            active_pid = pid
                        if not op["started_at"]:
                            op["started_at"] = first
                op["opencode_pid"] = active_pid

                if report_key in reports:
                    r = reports[report_key]
                    op["result_status"] = r.get("overall_status", "")
                    op["pypto_status"] = r.get("pypto_status", "")
                    op["verifier_status"] = r.get("verifier_status", "")
                    if r.get("finished_at") and not op["ended_at"]:
                        op["ended_at"] = r["finished_at"]
                    if r.get("started_at") and not op["started_at"]:
                        op["started_at"] = r["started_at"]
                    dur = (r.get("pypto_duration_sec") or 0) + (r.get("verifier_duration_sec") or 0)
                    if dur > op["duration_sec"]:
                        op["duration_sec"] = round(dur, 1)
                    pypto_phase = _ensure_phase(op, "pypto")
                    verify_phase = _ensure_phase(op, "verifier")
                    if r.get("pypto_duration_sec") is not None:
                        pypto_phase["duration_sec"] = round(r.get("pypto_duration_sec") or 0, 1)
                    if r.get("verifier_duration_sec") is not None:
                        verify_phase["duration_sec"] = round(r.get("verifier_duration_sec") or 0, 1)
                    if r.get("pypto_status"):
                        pypto_phase["status"] = r["pypto_status"]
                    if r.get("verifier_status"):
                        verify_phase["status"] = r["verifier_status"]
                    if r.get("finished_at") and r.get("overall_status"):
                        op["phase"] = "done"
                        op["phase_status"] = r["overall_status"]

                phase_total = 0.0
                for phase in ("pypto", "verifier"):
                    info = _ensure_phase(op, phase)
                    phase_total += float(info.get("duration_sec") or 0.0)
                if phase_total > op["duration_sec"]:
                    op["duration_sec"] = round(phase_total, 1)

                op["dev_status"] = _compute_dev_status(op, timeout_sec)

            if rc is not None:
                _daemon_state["main_exit_code"] = rc
                _daemon_state["main_status"] = "已完成" if rc == 0 else "未完成"

            _daemon_state["updated_at"] = _now()
            _write_state(_daemon_state)

            if rc is not None:
                break
            time.sleep(_POLL_SEC)
    except KeyboardInterrupt:
        _kill_opencode_groups_from_state(_daemon_state)
        if _bench_proc and _bench_proc.poll() is None:
            _kill_tree(_bench_proc.pid)
        _daemon_state["main_status"] = "未完成"
        _daemon_state["updated_at"] = _now()
        _write_state(_daemon_state)
    finally:
        reader.join(timeout=5)
        bench_log.close()

    return 0


# ────────────────────────────────────────────────────────────
# 看板渲染
# ────────────────────────────────────────────────────────────

_MAIN_STATUS_LABEL = {
    "正在进行": "正在进行",
    "已完成": "已完成",
    "未完成": "未完成",
}

_PHASE_LABEL = {
    "pending": "未开始",
    "prepare": "准备",
    "pypto": "PyPTO",
    "verifier": "Verifier",
    "done": "完成",
}

_COL_W = {
    "idx": 4, "op": 24, "phase": 10, "pid": 11, "time": 20, "dur": 9, "status": 22,
}


def _cell_width(value: Any) -> int:
    width = wcswidth(str(value))
    return width if width >= 0 else len(str(value))


def _clip_cell(value: Any, width: int) -> str:
    text = str(value)
    if _cell_width(text) <= width:
        return text
    suffix = "…"
    suffix_width = _cell_width(suffix)
    out: List[str] = []
    used = 0
    for ch in text:
        ch_width = _cell_width(ch)
        if used + ch_width + suffix_width > width:
            break
        out.append(ch)
        used += ch_width
    return "".join(out) + suffix


def _pad_cell(value: Any, width: int, *, align: str = "<") -> str:
    text = _clip_cell(value, width)
    pad = max(0, width - _cell_width(text))
    if align == ">":
        return " " * pad + text
    return text + " " * pad


def _render_dashboard(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    width = 132
    lines.append("")
    lines.append("=" * width)
    lines.append("  算子运行监控看板")
    lines.append("=" * width)
    lines.append("")

    mp = state.get("main_pid") or "—"
    ms = _MAIN_STATUS_LABEL.get(state.get("main_status", ""), state.get("main_status", "—"))
    ec = state.get("main_exit_code")
    sa = state.get("started_at", "—")
    ua = state.get("updated_at", "—")

    lines.append(f"  主进程 PID: {mp}    状态: {ms}    退出码: {ec if ec is not None else '—'}")
    lines.append(f"  启动时间:   {sa}    更新时间: {ua}")
    lines.append("")

    ops = state.get("operators", [])
    if not ops:
        lines.append("  (无算子记录)")
    else:
        w = _COL_W
        hdr = (
            f"  {_pad_cell('#', w['idx'], align='>')}"
            f"  {_pad_cell('Operator', w['op'])}"
            f"  {_pad_cell('Phase', w['phase'])}"
            f"  {_pad_cell('PyPTO PID', w['pid'], align='>')}"
            f"  {_pad_cell('Verify PID', w['pid'], align='>')}"
            f"  {_pad_cell('Started', w['time'])}"
            f"  {_pad_cell('Ended', w['time'])}"
            f"  {_pad_cell('Sec', w['dur'], align='>')}"
            f"  {_pad_cell('Status', w['status'])}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (width - 2))

        for i, op in enumerate(ops, 1):
            phases = op.get("phases") or {}
            pypto_pid = "—"
            verifier_pid = "—"
            if isinstance(phases.get("pypto"), dict):
                pypto_pid = str(phases["pypto"].get("pid") or "—")
            if isinstance(phases.get("verifier"), dict):
                verifier_pid = str(phases["verifier"].get("pid") or "—")
            phase_s = _PHASE_LABEL.get(op.get("phase", ""), op.get("phase", "—"))
            st_s = op.get("started_at") or "—"
            et_s = op.get("ended_at") or "—"
            dur_s = f"{op.get('duration_sec', 0):.1f}"
            dev_s = op.get("dev_status", "未知")
            nm = op.get("op_name", "?")
            if op.get("level"):
                nm = f"{op['level']}/{nm}"
            lines.append(
                f"  {_pad_cell(i, w['idx'], align='>')}"
                f"  {_pad_cell(nm, w['op'])}"
                f"  {_pad_cell(phase_s, w['phase'])}"
                f"  {_pad_cell(pypto_pid, w['pid'], align='>')}"
                f"  {_pad_cell(verifier_pid, w['pid'], align='>')}"
                f"  {_pad_cell(st_s, w['time'])}"
                f"  {_pad_cell(et_s, w['time'])}"
                f"  {_pad_cell(dur_s, w['dur'], align='>')}"
                f"  {_pad_cell(dev_s, w['status'])}"
            )

    lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI: run / monitor
# ────────────────────────────────────────────────────────────

def _load_run_args_from_env() -> Tuple[str, bool, int]:
    cases = os.environ.get("CASES", "19_ReLU")
    skip_gen = os.environ.get("FULL", "1") == "0"
    timeout_sec = int(os.environ.get("PYPTO_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    return cases, skip_gen, timeout_sec


def cmd_run(_args: argparse.Namespace) -> int:
    sd = _state_dir()
    sd.mkdir(parents=True, exist_ok=True)

    sf = _state_file()
    if sf.exists():
        sf.unlink(missing_ok=True)

    cases, skip_gen, timeout_sec = _load_run_args_from_env()
    logger.info("foreground monitor run: cases=%s skip_gen=%s timeout_sec=%s",
                cases, skip_gen, timeout_sec)
    return _run_monitor_session(cases, skip_gen, timeout_sec, tee_stdout=True)


# ────────────────────────────────────────────────────────────
# CLI: monitor
# ────────────────────────────────────────────────────────────

def cmd_monitor(args: argparse.Namespace) -> int:
    state = _read_state()
    if state is None:
        print("无监控状态, 请确认 BENCHMARK_MONITOR_DIR 指向当前运行的 monitor_state 目录.", file=sys.stderr)
        return 1

    if args.once:
        print(_render_dashboard(state))
        return 0

    try:
        while True:
            os.system("clear")
            state = _read_state()
            if state is None:
                print("监控状态丢失.")
                break
            print(_render_dashboard(state))
            if state.get("main_status") in ("已完成", "未完成"):
                print("  主进程已结束, 看板停止刷新.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    return 0


# ────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    _setup_logging(os.environ.get("BENCHMARK_MONITOR_LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser(
        prog="python3 -m integration.benchmark.monitor",
        description="算子运行监控 CLI — 默认前台运行 benchmark; `monitor` 查看看板",
    )
    p.add_argument(
        "command",
        nargs="?",
        choices=["monitor"],
        help="留空=前台运行 benchmark 并写 state.json; `monitor`=查看监控看板",
    )

    p.add_argument("--once", action="store_true", help="仅 `monitor` 模式使用: 单次快照后退出")

    args = p.parse_args(argv)

    if args.command == "monitor":
        return cmd_monitor(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
