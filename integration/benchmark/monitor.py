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
"""算子运行监控 CLI — 后台启动、实时看板、停止控制.

子命令:
    start     后台启动 benchmark 并开始监控 (读取 CASES / FULL 等环境变量)
    monitor   查看刷新式监控看板 (--once 单次快照)
    stop      停止后台监控守护进程

用法:
    FULL=0 CASES=19_ReLU python3 -m integration.benchmark.monitor start
    python3 -m integration.benchmark.monitor monitor --once
    python3 -m integration.benchmark.monitor stop

设计要点:
- ``start`` 以子进程方式启动后台守护进程 (``_daemon``), 守护进程再启动
  ``run_kernelbench.py`` 作为受监控的主流程, 同时周期性轮询 ``/proc`` 扫描
  opencode 子进程, 把采集到的进程信息 + 报告目录结果写入共享 state.json.
- ``monitor`` 读 state.json 渲染刷新式表格; ``--once`` 单次快照.
- ``stop`` 读 PID 文件终止守护进程及其子进程树.
- 状态判断基于可核实的进程存活信息与时间戳, 不伪造完成态.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from integration.benchmark.case_loader import derive_op_name


_STATE_DIR_DEFAULT = "/tmp/pypto-benchmark-monitor"
_POLL_SEC = 3
_DEFAULT_TIMEOUT = 5400
_OVER_THRESHOLD = 5400

_OP_RE_WORKFLOW = re.compile(r"算子 `(\w+)`")
_OP_RE_VERIFIER = re.compile(r"op_name\s*=\s*(\w+)")


# ────────────────────────────────────────────────────────────
# 路径约定
# ────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    return Path(os.environ.get("BENCHMARK_MONITOR_DIR", _STATE_DIR_DEFAULT))


def _state_file() -> Path:
    return _state_dir() / "state.json"


def _pid_file() -> Path:
    return _state_dir() / "daemon.pid"


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


def _find_opencode_procs(root_pid: int) -> Dict[str, int]:
    """扫描 root_pid 的后代, 返回 {op_name: pid} (opencode 进程)."""
    out: Dict[str, int] = {}
    for pid in _descendants(root_pid):
        cmd = _read_cmdline(pid)
        if "opencode" not in cmd:
            continue
        m = _OP_RE_WORKFLOW.search(cmd) or _OP_RE_VERIFIER.search(cmd)
        if m:
            out[m.group(1)] = pid
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


# ────────────────────────────────────────────────────────────
# 开发状态判定
# ────────────────────────────────────────────────────────────

def _elapsed_since(ts: str) -> float:
    try:
        return (dt.datetime.now() - dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        return 0.0


def _compute_dev_status(op: Dict[str, Any], timeout_sec: int) -> str:
    """基于进程与报告信息计算算子开发状态."""
    rs = op.get("result_status")
    if rs == "success":
        return "已完成"
    if rs in ("pypto_failed", "verify_failed", "verify_error"):
        if op.get("pypto_status") == "timeout":
            return "超时"
        return "失败"

    pid = op.get("opencode_pid")
    started = op.get("started_at")
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


# ────────────────────────────────────────────────────────────
# 报告目录扫描
# ────────────────────────────────────────────────────────────

def _scan_reports(report_dir: str, op_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """扫描 report_dir 下各算子的 result.json."""
    out: Dict[str, Dict[str, Any]] = {}
    rd = Path(report_dir)
    if not rd.exists():
        return out
    for op in op_names:
        rf = rd / op / "result.json"
        if rf.exists():
            try:
                out[op] = json.loads(rf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return out


# ────────────────────────────────────────────────────────────
# 守护进程
# ────────────────────────────────────────────────────────────

_bench_proc: Optional[subprocess.Popen] = None
_daemon_state: Dict[str, Any] = {}


def _sigterm_handler(_signum: int, _frame: Any) -> None:
    global _bench_proc, _daemon_state
    if _bench_proc and _bench_proc.poll() is None:
        _bench_proc.terminate()
        try:
            _bench_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bench_proc.kill()
    _daemon_state["main_status"] = "未完成"
    _daemon_state["updated_at"] = _now()
    try:
        _write_state(_daemon_state)
    except OSError:
        pass
    sys.exit(0)


def _daemon_main(cases: str, skip_gen: bool, timeout_sec: int) -> int:
    """守护进程主循环: 启动 benchmark 并周期性采集状态."""
    global _bench_proc, _daemon_state
    signal.signal(signal.SIGTERM, _sigterm_handler)

    sd = _state_dir()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = sd / f"bench_{ts}"
    report_dir = bench_dir / "report"
    log_dir = bench_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    case_list = [c.strip() for c in cases.split(",") if c.strip()]
    op_names = [derive_op_name(c) for c in case_list]

    operators: List[Dict[str, Any]] = []
    for cid, op in zip(case_list, op_names):
        operators.append({
            "case_id": cid,
            "op_name": op,
            "opencode_pid": None,
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

    bench_log = open(bench_dir / "benchmark.log", "w")
    try:
        _bench_proc = subprocess.Popen(
            cmd, stdout=bench_log, stderr=subprocess.STDOUT,
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

    seen_pids: Dict[str, Tuple[int, str]] = {}

    try:
        while True:
            rc = _bench_proc.poll()

            if rc is None:
                oc = _find_opencode_procs(_bench_proc.pid)
                for op_name, pid in oc.items():
                    if op_name not in seen_pids or seen_pids[op_name][0] != pid:
                        seen_pids[op_name] = (pid, _now())

            reports = _scan_reports(str(report_dir), op_names)

            for op in _daemon_state["operators"]:
                name = op["op_name"]
                if name in seen_pids:
                    pid, first = seen_pids[name]
                    op["opencode_pid"] = pid
                    if not op["started_at"]:
                        op["started_at"] = first
                    if _pid_alive(pid):
                        op["duration_sec"] = round(_elapsed_since(first), 1)
                    elif not op["ended_at"]:
                        op["ended_at"] = _now()

                if name in reports:
                    r = reports[name]
                    op["result_status"] = r.get("overall_status", "")
                    op["pypto_status"] = r.get("pypto_status", "")
                    if r.get("finished_at") and not op["ended_at"]:
                        op["ended_at"] = r["finished_at"]
                    if r.get("started_at") and not op["started_at"]:
                        op["started_at"] = r["started_at"]
                    dur = (r.get("pypto_duration_sec") or 0) + (r.get("verifier_duration_sec") or 0)
                    if dur > op["duration_sec"]:
                        op["duration_sec"] = round(dur, 1)

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
        if _bench_proc and _bench_proc.poll() is None:
            _bench_proc.terminate()
            try:
                _bench_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _bench_proc.kill()
        _daemon_state["main_status"] = "未完成"
        _daemon_state["updated_at"] = _now()
        _write_state(_daemon_state)
    finally:
        bench_log.close()
        try:
            _pid_file().unlink(missing_ok=True)
        except OSError:
            pass

    return 0


# ────────────────────────────────────────────────────────────
# 看板渲染
# ────────────────────────────────────────────────────────────

_MAIN_STATUS_LABEL = {
    "正在进行": "正在进行",
    "已完成": "已完成",
    "未完成": "未完成",
}

_COL_W = {
    "idx": 4, "op": 16, "pid": 13, "time": 20, "dur": 9, "status": 20,
}


def _render_dashboard(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    width = 106
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
            f"  {'#':>{w['idx']}}"
            f"  {'Operator':<{w['op']}}"
            f"  {'opencode PID':>{w['pid']}}"
            f"  {'Started':<{w['time']}}"
            f"  {'Ended':<{w['time']}}"
            f"  {'Sec':>{w['dur']}}"
            f"  {'Status'}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (width - 2))

        for i, op in enumerate(ops, 1):
            pid_s = str(op.get("opencode_pid") or "—")
            st_s = op.get("started_at") or "—"
            et_s = op.get("ended_at") or "—"
            dur_s = f"{op.get('duration_sec', 0):.1f}"
            dev_s = op.get("dev_status", "未知")
            nm = op.get("op_name", "?")
            lines.append(
                f"  {i:>{w['idx']}}"
                f"  {nm:<{w['op']}}"
                f"  {pid_s:>{w['pid']}}"
                f"  {st_s:<{w['time']}}"
                f"  {et_s:<{w['time']}}"
                f"  {dur_s:>{w['dur']}}"
                f"  {dev_s}"
            )

    lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI: start
# ────────────────────────────────────────────────────────────

def cmd_start(_args: argparse.Namespace) -> int:
    sd = _state_dir()
    sd.mkdir(parents=True, exist_ok=True)

    pf = _pid_file()
    if pf.exists():
        try:
            old = int(pf.read_text().strip())
            if _pid_alive(old):
                print(f"监控已在运行 (PID={old}), 先执行 stop.", file=sys.stderr)
                return 1
        except (ValueError, OSError):
            pass
        pf.unlink(missing_ok=True)

    sf = _state_file()
    if sf.exists():
        sf.unlink(missing_ok=True)

    cases = os.environ.get("CASES", "19_ReLU")
    skip_gen = os.environ.get("FULL", "1") == "0"
    timeout_sec = os.environ.get("PYPTO_TIMEOUT", str(_DEFAULT_TIMEOUT))

    daemon_cmd = [
        sys.executable, "-m", "integration.benchmark.monitor", "_daemon",
        "--cases", cases,
        "--timeout-sec", timeout_sec,
    ]
    if skip_gen:
        daemon_cmd.append("--skip-gen")

    log_path = sd / "daemon.log"
    log_h = open(log_path, "w")
    proc = subprocess.Popen(
        daemon_cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_h,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_h.close()

    pf.write_text(str(proc.pid))

    for _ in range(50):
        if sf.exists():
            break
        time.sleep(0.1)

    st = _read_state()
    if st:
        print(f"监控守护进程已启动 (PID={proc.pid})")
        print(f"  报告目录: {st.get('report_dir', '?')}")
    else:
        print(f"监控守护进程启动中 (PID={proc.pid})")
    return 0


# ────────────────────────────────────────────────────────────
# CLI: monitor
# ────────────────────────────────────────────────────────────

def cmd_monitor(args: argparse.Namespace) -> int:
    state = _read_state()
    if state is None:
        print("无监控状态, 请先执行 start.", file=sys.stderr)
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
# CLI: stop
# ────────────────────────────────────────────────────────────

def cmd_stop(_args: argparse.Namespace) -> int:
    pf = _pid_file()
    state = _read_state()

    if not pf.exists():
        print("无运行中的监控进程.")
        return 0

    try:
        dpid = int(pf.read_text().strip())
    except (ValueError, OSError):
        pf.unlink(missing_ok=True)
        print("PID 文件无效, 已清理.")
        return 0

    if _pid_alive(dpid):
        print(f"正在停止监控 (PID={dpid})...")
        try:
            os.kill(dpid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        for _ in range(20):
            if not _pid_alive(dpid):
                break
            time.sleep(0.5)
        else:
            try:
                os.kill(dpid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        print("监控已停止.")
    else:
        print("守护进程已不在运行.")

    if state and state.get("main_pid"):
        main_pid = state["main_pid"]
        if _pid_alive(main_pid):
            _kill_tree(main_pid)

    pf.unlink(missing_ok=True)
    return 0


# ────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m integration.benchmark.monitor",
        description="算子运行监控 CLI — 后台启动、实时看板、停止控制",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="后台启动监控 (读取 CASES/FULL 等环境变量)")

    mon = sub.add_parser("monitor", help="查看监控看板")
    mon.add_argument("--once", action="store_true", help="单次快照后退出")

    sub.add_parser("stop", help="停止监控守护进程")

    dp = sub.add_parser("_daemon", help=argparse.SUPPRESS)
    dp.add_argument("--cases", required=True)
    dp.add_argument("--skip-gen", action="store_true")
    dp.add_argument("--timeout-sec", type=int, default=_DEFAULT_TIMEOUT)

    args = p.parse_args(argv)

    handlers = {
        "start": cmd_start,
        "monitor": cmd_monitor,
        "stop": cmd_stop,
        "_daemon": lambda a: _daemon_main(a.cases, a.skip_gen, a.timeout_sec),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
