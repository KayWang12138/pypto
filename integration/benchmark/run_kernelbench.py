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
"""KernelBench × pypto 端到端批处理 CLI.

使用示例:

    # 单 case MVP (默认 --bench-dir 走桥接层 .cache/KernelBench/KernelBench/)
    python -m integration.benchmark.run_kernelbench \\
        --cases 19_ReLU \\
        --level level1 \\
        --devices 0 --arch ascend910b4 --mode correctness

    # 多 case 多卡并发
    python -m integration.benchmark.run_kernelbench \\
        --cases 19:21,31,41:50 \\
        --devices 0,1,2 --concurrency 3 \\
        --report-dir benchmark_report

    # 多 level 批处理: 直接在 --cases 中写完整 level/case 坐标
    python -m integration.benchmark.run_kernelbench \\
        --cases 'level1=1:21;level2=31,41:50'

    # 仅复跑验证 (跳过 pypto 生成阶段, 要求产物已就绪)
    python -m integration.benchmark.run_kernelbench \\
        --cases 19_ReLU --skip-pypto-gen
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from integration.benchmark import case_loader, pypto_runner, verifier_runner, report
from integration.benchmark.case_loader import CaseSpec, derive_op_name
from integration.benchmark.opencode_exporter import OpencodeExportResult, append_export_result_to_log
from integration.benchmark.pypto_runner import PyptoRunResult, PyptoRunStatus, run_pypto_workflow
from integration.benchmark.report import CaseRunRecord, derive_overall_status, write_case_result, write_summary
from integration.benchmark.verifier_runner import VerifierResult, VerifierStatus, run_verifier


logger = logging.getLogger("benchmark")


# ────────────────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_csv_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_csv_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _expand_case_selector(selector: str) -> List[str]:
    selector = selector.strip()
    if not selector:
        return []
    if ":" not in selector:
        return [selector]

    start, end = (part.strip() for part in selector.split(":", 1))
    if start.isdigit() and end.isdigit():
        lo = int(start)
        hi = int(end)
        if lo > hi:
            raise ValueError(f"case range 必须递增: {selector}")
        return [str(i) for i in range(lo, hi + 1)]
    return [selector]


def parse_case_selectors(text: str) -> List[str]:
    selectors: List[str] = []
    for token in parse_csv_str_list(text):
        selectors.extend(_expand_case_selector(token))
    return selectors


def parse_cases_by_level(
    cases_text: str,
    default_levels: List[str],
) -> Dict[str, Optional[List[str]]]:
    """解析 --cases.

    单 level 兼容 ``1:21,31,41:50`` 裸 selector, 使用 ``default_levels``
    里的单个 level. 多 level 必须直接在 --cases 中写完整坐标:
    ``level1=1:21;level2=31,41:50``.
    """
    if not cases_text:
        if len(default_levels) > 1:
            raise ValueError(
                "多 level 不能通过 --level/config 单独表达. "
                "请使用 --cases 'level1=1:21;level2=31,41:50'."
            )
        return {level: None for level in default_levels}

    if "=" not in cases_text:
        if len(default_levels) > 1:
            raise ValueError(
                "多 level 下裸 --cases selector 语义不明确: "
                f"{cases_text!r}. 请改用 'level1=1:21;level2=31,41:50' "
                "这种分 level 写法."
            )
        selectors = parse_case_selectors(cases_text)
        level = default_levels[0] if default_levels else "level1"
        return {level: selectors}

    mapping: Dict[str, Optional[List[str]]] = {}
    for chunk in (part.strip() for part in cases_text.split(";") if part.strip()):
        if "=" not in chunk:
            raise ValueError(
                "分 level 指定 --cases 时必须使用 'level=cases' 片段, "
                f"收到: {chunk!r}"
            )
        level, selectors_text = (part.strip() for part in chunk.split("=", 1))
        if not level:
            raise ValueError(f"--cases 中存在空 level 片段: {chunk!r}")
        if level in mapping:
            raise ValueError(f"--cases 中重复指定 level: {level}")
        mapping[level] = parse_case_selectors(selectors_text)
    if not mapping:
        raise ValueError(f"--cases 未解析到任何 level: {cases_text!r}")
    return mapping


# ────────────────────────────────────────────────────────────
# 用例发现
# ────────────────────────────────────────────────────────────

def discover_cases(level_dir: Path, requested: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> List[Path]:
    """在 ``level_dir`` (即 ``KernelBench/<level>/``) 下查找用例.

    上游 KernelBench (github.com/ScalingIntelligence/KernelBench @ 21fbe5a)
    用例文件形如 ``KernelBench/level1/19_ReLU.py``. 每个 .py
    即一个用例, ``case_id`` = 文件名 stem (不含 ``.py``).

    Args:
        level_dir: ``KernelBench/<level>/`` 目录绝对路径.
        requested: 用户指定的 case_id 子集; ``None`` 表示全选.
            可写完整 stem (``19_ReLU``) 或仅序号前缀 (``19``).
        limit: 截断数量, 仅在 ``requested is None`` 时生效.

    Returns:
        每个用例对应的 ``.py`` 文件绝对路径列表.

    Raises:
        FileNotFoundError: ``level_dir`` 不存在.
        ValueError: ``level_dir`` 下没有任何 .py; 或
            ``requested`` 中有 case 找不到.
    """
    if not level_dir.exists():
        raise FileNotFoundError(f"level dir 不存在: {level_dir}")
    if not level_dir.is_dir():
        raise ValueError(f"level dir 不是目录: {level_dir}")

    py_files = sorted(p for p in level_dir.iterdir() if p.is_file() and p.suffix == ".py")
    if not py_files:
        raise ValueError(
            f"{level_dir} 下找不到任何 .py 用例. 检查路径是否指向 "
            f"KernelBench/<level>/ (例如 .cache/KernelBench/KernelBench/level1)."
        )

    by_stem: Dict[str, Path] = {p.stem: p for p in py_files}
    by_index_prefix: Dict[str, Path] = {}
    for stem, path in by_stem.items():
        idx = stem.split("_", 1)[0]
        if idx.isdigit():
            by_index_prefix.setdefault(idx, path)

    if requested is not None:
        resolved: List[Path] = []
        missing: List[str] = []
        for r in requested:
            if r in by_stem:
                resolved.append(by_stem[r])
            elif r in by_index_prefix:
                resolved.append(by_index_prefix[r])
            else:
                missing.append(r)
        if missing:
            raise ValueError(
                f"以下 case 在 {level_dir} 中找不到: {missing}. "
                f"可用 case (前 10 个): {sorted(by_stem)[:10]}"
            )
        return resolved

    cases = list(by_stem.values())
    if limit is not None:
        cases = cases[:limit]
    return cases


# ────────────────────────────────────────────────────────────
# 单 case 流水线
# ────────────────────────────────────────────────────────────

def _write_case_phase(
    case_report_dir: Path,
    *,
    op_name: str,
    case_id: str,
    phase: str,
    status: str,
    level: str = "",
    message: str = "",
    pypto_status: str = "",
    verifier_status: str = "",
) -> None:
    """Write a tiny per-case phase marker for the live monitor."""
    payload = {
        "op_name": op_name,
        "case_id": case_id,
        "level": level,
        "phase": phase,
        "status": status,
        "message": message,
        "pypto_status": pypto_status,
        "verifier_status": verifier_status,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    case_report_dir.mkdir(parents=True, exist_ok=True)
    out = case_report_dir / "phase_state.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


@dataclass
class _RunCfg:
    pypto_repo_root: Path
    workdir_root: str
    opencode_bin: str
    opencode_model: str
    pypto_agent: str
    pypto_timeout: int
    pypto_output_format: str
    skip_stage7_perf_tune: bool
    arch: str
    backend: str
    framework: str
    verify_timeout: int
    log_dir: Path
    report_dir: Path
    mode: str                      # correctness / performance / full
    skip_pypto_gen: bool
    force_regen: bool
    use_level_dirs: bool
    extra_verifier_config: Dict[str, Any]
    verifier_mode: str             # opencode / direct
    validator_agent: str           # opencode 模式: agent 名
    skill_timeout_sec: int         # opencode 模式: skill 子进程硬超时
    skill_retry: int               # opencode/API 层失败后的重试次数
    skill_retry_interval_sec: int  # opencode/API 层重试间隔秒数


async def run_one_case(case_path: Path, device_id: int, cfg: _RunCfg,
                       semaphore: asyncio.Semaphore) -> CaseRunRecord:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    logger.info("[%s] case start: device=%s source=%s", case_path.stem, device_id, case_path)

    logger.info("[%s] loading case + probing inputs", case_path.stem)
    case = case_loader.load_case(case_path, case_id=case_path.stem)
    op_name = case.op_name
    report_subdir = f"{case.level}/{op_name}" if cfg.use_level_dirs and case.level else op_name
    workdir_root = (
        f"{cfg.workdir_root}/{case.level}"
        if cfg.use_level_dirs and case.level else cfg.workdir_root
    )
    op_workdir = cfg.pypto_repo_root / workdir_root
    op_dir = op_workdir / op_name
    case_report_dir = cfg.report_dir / report_subdir
    case_report_dir.mkdir(parents=True, exist_ok=True)
    _write_case_phase(
        case_report_dir,
        op_name=op_name,
        case_id=case.case_id,
        phase="prepare",
        status="running",
        message="case loaded; writing SPEC/task_desc",
    )
    logger.info("[%s] case loaded: op=%s report_dir=%s", case.case_id, op_name, case_report_dir)

    # 1) 写 SPEC + task_desc
    logger.info("[%s] writing SPEC/task_desc into %s", case.case_id, op_workdir)
    case_loader.write_spec(case, op_workdir)
    case_loader.write_task_desc(case, op_workdir)

    # 2) Pypto 7-stage 工作流 (占设备号槽位)
    record = CaseRunRecord(
        op_name=op_name,
        case_id=case.case_id,
        source_file=case.source_file,
        level=case.level,
        report_subdir=report_subdir,
        started_at=started_at,
    )

    logger.info("[%s] waiting for execution slot (concurrency gate)", case.case_id)
    async with semaphore:
        logger.info("[%s] acquired execution slot", case.case_id)
        if cfg.skip_pypto_gen:
            logger.info("[%s] skip pypto generation (--skip-pypto-gen)", case.case_id)
            pypto_log = case_report_dir / "pypto_run.log"
            pypto_log.write_text(
                "[pypto workflow skipped] --skip-pypto-gen\n",
                encoding="utf-8",
            )
            session_export = OpencodeExportResult(
                status="skipped",
                message="--skip-pypto-gen, 本次没有新的 OpenCode session 可导出.",
            )
            append_export_result_to_log(pypto_log, session_export, label="pypto")
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="pypto",
                status="skipped",
                message="--skip-pypto-gen",
            )
            pypto_result = PyptoRunResult(
                op_name=op_name,
                status=PyptoRunStatus.SKIPPED,
                workdir=op_dir,
                log_file=pypto_log,
                message="--skip-pypto-gen, 跳过 pypto 生成阶段.",
                opencode_session_export_message=session_export.message,
            )
        else:
            pypto_log = case_report_dir / "pypto_run.log"
            logger.info("[%s] launching pypto workflow; log=%s", case.case_id, pypto_log)
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="pypto",
                status="running",
                message=f"log={pypto_log}",
            )
            pypto_result = await asyncio.to_thread(
                run_pypto_workflow,
                op_name=op_name,
                pypto_repo_root=cfg.pypto_repo_root,
                workdir_root=workdir_root,
                opencode_bin=cfg.opencode_bin,
                opencode_model=cfg.opencode_model,
                agent=cfg.pypto_agent,
                timeout_sec=cfg.pypto_timeout,
                device_id=device_id,
                log_file=pypto_log,
                output_format=cfg.pypto_output_format,
                skip_if_done=not cfg.force_regen,
                task_desc_rel=f"{workdir_root}/{op_name}/task_desc.py",
                case_init_args_repr=case.init_args_repr,
                case_init_source=case.init_source,
                case_forward_source=case.forward_source,
                skip_stage7_perf_tune=cfg.skip_stage7_perf_tune,
                stop_event=_stop_event,
            )
            logger.info("[%s] pypto workflow finished: status=%s duration=%.1fs message=%s",
                        case.case_id, pypto_result.status.value,
                        pypto_result.duration_sec, pypto_result.message)

        record.pypto_status = pypto_result.status.value
        record.pypto_message = pypto_result.message
        record.pypto_duration_sec = pypto_result.duration_sec
        record.pypto_log_file = str(pypto_result.log_file) if pypto_result.log_file else None
        record.pypto_session_id = pypto_result.opencode_session_id
        record.pypto_session_md_file = (
            str(pypto_result.opencode_session_md_file)
            if pypto_result.opencode_session_md_file else None
        )
        record.pypto_session_export_message = pypto_result.opencode_session_export_message
        record.pypto_artifacts = {k: str(v) for k, v in pypto_result.artifacts.items()}

        if not pypto_result.ok:
            record.overall_status = "pypto_failed"
            record.finished_at = dt.datetime.now().isoformat(timespec="seconds")
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="done",
                status="pypto_failed",
                message=pypto_result.message,
                pypto_status=pypto_result.status.value,
            )
            write_case_result(record, cfg.report_dir)
            logger.warning("[%s] case stop after pypto failure: status=%s",
                           case.case_id, record.pypto_status)
            return record

        # 3) KernelVerifier (仍占设备号槽位避免冲突)
        verifier_log = case_report_dir / "verifier.log"
        try:
            logger.info("[%s] launching verifier; mode=%s log=%s",
                        case.case_id, cfg.verifier_mode, verifier_log)
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="verifier",
                status="running",
                message=f"mode={cfg.verifier_mode}; log={verifier_log}",
                pypto_status=pypto_result.status.value,
            )
            verifier_result = await run_verifier(
                op_name=op_name,
                op_dir=op_dir,
                task_desc=case.task_desc,
                arch=cfg.arch,
                backend=cfg.backend,
                framework=cfg.framework,
                device_id=device_id,
                log_dir=cfg.log_dir,
                task_id=f"benchmark_{op_name}_{int(time.time()*1000)}",
                verify_timeout=cfg.verify_timeout,
                extra_config=cfg.extra_verifier_config,
                log_file=verifier_log,
                mode=cfg.mode,
                verifier_mode=cfg.verifier_mode,
                opencode_bin=cfg.opencode_bin,
                opencode_model=cfg.opencode_model,
                validator_agent=cfg.validator_agent,
                skill_timeout_sec=cfg.skill_timeout_sec,
                skill_retry=cfg.skill_retry,
                skill_retry_interval_sec=cfg.skill_retry_interval_sec,
            )
            logger.info("[%s] verifier finished: status=%s duration=%.1fs correctness=%s",
                        case.case_id, verifier_result.status.value,
                        verifier_result.duration_sec, verifier_result.correctness)
        except Exception as e:
            logger.exception("[%s] verifier raised exception: %s", case.case_id, e)
            verifier_result = VerifierResult(
                op_name=op_name,
                status=VerifierStatus.ERROR,
                message=f"run_verifier 抛异常: {e}",
            )

    record.verifier_status = verifier_result.status.value
    record.verifier_message = verifier_result.message
    record.verifier_duration_sec = verifier_result.duration_sec
    record.verifier_log_file = str(verifier_result.log_file) if verifier_result.log_file else None
    record.verifier_session_id = verifier_result.opencode_session_id
    record.verifier_session_md_file = (
        str(verifier_result.opencode_session_md_file)
        if verifier_result.opencode_session_md_file else None
    )
    record.verifier_session_export_message = verifier_result.opencode_session_export_message
    record.correctness = verifier_result.correctness
    record.perf_gen_time_us = verifier_result.perf_gen_time_us
    record.perf_base_time_us = verifier_result.perf_base_time_us
    record.perf_speedup = verifier_result.perf_speedup
    record.perf_roofline_time_us = verifier_result.perf_roofline_time_us
    record.perf_roofline_speedup = verifier_result.perf_roofline_speedup
    record.perf_message = verifier_result.perf_message
    record.overall_status = derive_overall_status(
        pypto_ok=pypto_result.ok,
        verifier_status=verifier_result.status.value,
        correctness=verifier_result.correctness,
    )
    record.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    _write_case_phase(
        case_report_dir,
        op_name=op_name,
        case_id=case.case_id,
        phase="done",
        status=record.overall_status,
        message=verifier_result.message,
        pypto_status=pypto_result.status.value,
        verifier_status=verifier_result.status.value,
    )
    write_case_result(record, cfg.report_dir)
    logger.info("[%s] case finished: overall=%s pypto=%s verifier=%s",
                case.case_id, record.overall_status,
                record.pypto_status, record.verifier_status)
    return record


# ────────────────────────────────────────────────────────────
# 主调度
# ────────────────────────────────────────────────────────────

async def run_batch(case_paths: List[Path], devices: List[int], concurrency: int,
                    cfg: _RunCfg) -> List[CaseRunRecord]:
    if not case_paths:
        return []
    if not devices:
        raise ValueError("devices 列表不能为空.")
    if concurrency < 1:
        raise ValueError(f"concurrency 必须 >= 1, 收到 {concurrency}")

    # 并发仅由 concurrency 控制; device 按 case 索引轮询 (idx % len(devices)),
    # 与 NPU 侧/工作流内部的占卡策略解耦, 不在此处用 len(devices) 夹逼并发度.
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapper(idx: int, path: Path) -> CaseRunRecord:
        device_id = devices[idx % len(devices)]
        try:
            logger.info("[%s] scheduled on device %s", path.stem, device_id)
            return await run_one_case(path, device_id, cfg, semaphore)
        except Exception as e:
            logger.exception(f"[{path.name}] run_one_case 异常: {e}")
            op_name = derive_op_name(path.parent.name)
            return CaseRunRecord(
                op_name=op_name,
                case_id=path.parent.name,
                source_file=str(path),
                pypto_status="exception",
                pypto_message=str(e),
                overall_status="pypto_failed",
                started_at=dt.datetime.now().isoformat(timespec="seconds"),
                finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            )

    coros = [_wrapper(i, p) for i, p in enumerate(case_paths)]
    return await asyncio.gather(*coros)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def _resolve_pypto_repo_root(repo_root: Optional[Path]) -> Path:
    if repo_root:
        return repo_root.resolve()
    here = Path(__file__).resolve().parent.parent.parent
    if (here / "pyproject.toml").exists():
        return here
    return Path.cwd()


def _build_cfg(args: argparse.Namespace, yaml_cfg: Dict[str, Any]) -> _RunCfg:
    pypto_yaml = yaml_cfg.get("pypto", {}) or {}
    verifier_yaml = yaml_cfg.get("verifier", {}) or {}
    report_yaml = yaml_cfg.get("report", {}) or {}
    extra_verifier_config: Dict[str, Any] = {}
    verify_rtol = args.verify_rtol if args.verify_rtol is not None else verifier_yaml.get("verify_rtol")
    verify_atol = args.verify_atol if args.verify_atol is not None else verifier_yaml.get("verify_atol")
    if verify_rtol is not None:
        extra_verifier_config["verify_rtol"] = float(verify_rtol)
    if verify_atol is not None:
        extra_verifier_config["verify_atol"] = float(verify_atol)
    keep_env = (os.environ.get("PYPTO_BENCH_KEEP_ARTIFACTS") or "").strip().lower()
    if args.keep_verifier_artifacts is not None:
        keep_artifacts = args.keep_verifier_artifacts
    elif keep_env:
        keep_artifacts = keep_env in ("1", "true", "yes", "on")
    else:
        keep_artifacts = bool(verifier_yaml.get("keep_artifacts", False))
    extra_verifier_config["keep_artifacts"] = bool(keep_artifacts)

    return _RunCfg(
        pypto_repo_root=_resolve_pypto_repo_root(args.repo_root),
        workdir_root=args.workdir_root or pypto_yaml.get("workdir_root", "custom"),
        opencode_bin=args.opencode_bin or pypto_yaml.get("opencode_bin", "") or "",
        opencode_model=args.opencode_model or pypto_yaml.get("opencode_model", "") or "",
        pypto_agent=args.agent or pypto_yaml.get("agent", "pypto-op-orchestrator"),
        pypto_timeout=args.timeout_sec or pypto_yaml.get("timeout_sec", 1800),
        pypto_output_format=args.opencode_format or pypto_yaml.get("output_format", "default"),
        skip_stage7_perf_tune=(
            args.skip_stage7_perf_tune
            if args.skip_stage7_perf_tune is not None
            else bool(pypto_yaml.get("skip_stage7_perf_tune", False))
        ),
        arch=args.arch or verifier_yaml.get("arch", "ascend910b4"),
        backend=args.backend or verifier_yaml.get("backend", "ascend"),
        framework=args.framework or verifier_yaml.get("framework", "torch"),
        verify_timeout=args.verify_timeout or verifier_yaml.get("verify_timeout", 900),
        log_dir=Path(args.log_dir or verifier_yaml.get("log_dir", "~/pypto_bench_logs")).expanduser(),
        report_dir=Path(args.report_dir or report_yaml.get("out_dir", "benchmark_report")).resolve(),
        mode=args.mode or verifier_yaml.get("mode", "correctness"),
        skip_pypto_gen=args.skip_pypto_gen,
        force_regen=args.force_regen,
        use_level_dirs=False,
        extra_verifier_config=extra_verifier_config,
        verifier_mode=args.verifier_mode or verifier_yaml.get("verifier_mode", "opencode"),
        validator_agent=args.validator_agent or verifier_yaml.get(
            "validator_agent", "pypto-kernel-validator"
        ),
        skill_timeout_sec=args.skill_timeout or verifier_yaml.get("skill_timeout_sec", 1800),
        skill_retry=args.skill_retry if args.skill_retry is not None else verifier_yaml.get("skill_retry", 2),
        skill_retry_interval_sec=(
            args.skill_retry_interval
            if args.skill_retry_interval is not None
            else verifier_yaml.get("skill_retry_interval_sec", 600)
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KernelBench × pypto 端到端批处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help=f"YAML 配置 (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--bench-dir", type=Path, default=None,
                   help="上游 KernelBench 根目录 (即 KernelBench/KernelBench/, "
                        "含 level1/level2/level3 子目录); 默认 .cache/KernelBench/KernelBench")
    p.add_argument("--level", type=str, default="",
                   help="单 level 默认难度子目录, 如 level1; 多 level 请直接写进 --cases")
    p.add_argument("--cases", type=str, default="",
                   help="逗号分隔的 case selector, 支持完整 stem、序号和闭区间 "
                        "(如 '19_ReLU', '19', '1:21,31,41:50'); 多 level 直接写 "
                        "'level1=1:21;level2=31,41:50'. 未传时读取 CASES "
                        "环境变量; 空表示按 --limit 跑默认 level 全部")
    p.add_argument("--limit", type=int, default=None,
                   help="未指定 --cases 时, 截取前 N 个用例")
    p.add_argument("--devices", type=str, default="",
                   help="逗号分隔的 device_id 列表 (覆盖配置)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="并发任务上限 (与 --devices 条目数无关; 默认: 配置或 len(devices))")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="pypto 仓根 (子进程 cwd)")
    p.add_argument("--workdir-root", type=str, default="",
                   help="算子产物根目录, 形成 {root}/{op}/")
    p.add_argument("--opencode-bin", type=str, default="",
                   help="opencode 可执行路径")
    p.add_argument("--opencode-model", type=str, default="",
                   help="显式传给 opencode run -m 的模型名; 未传则沿用 opencode 当前默认配置")
    p.add_argument("--agent", type=str, default="",
                   help="opencode --agent 名")
    p.add_argument("--timeout-sec", type=int, default=0,
                   help="单 case opencode 子进程超时 (秒)")
    p.add_argument("--opencode-format", type=str, default="",
                   choices=["", "default", "json"],
                   help="opencode --format")
    p.add_argument("--skip-stage7-perf-tune",
                   action=argparse.BooleanOptionalAction,
                   default=None,
                   help="是否跳过 pypto 工作流第七步的迭代性能调优; "
                        "true 时仅做 Stage 7 收尾, 不跑性能优化循环")
    p.add_argument("--arch", type=str, default="",
                   help="硬件架构 (ascend910b4 等)")
    p.add_argument("--backend", type=str, default="",
                   choices=["", "ascend", "cuda", "cpu"])
    p.add_argument("--framework", type=str, default="",
                   choices=["", "torch", "mindspore", "numpy"])
    p.add_argument("--verify-timeout", type=int, default=0,
                   help="单次 KernelVerifier 超时 (秒)")
    p.add_argument("--verify-rtol", type=float, default=None,
                   help="verify 精度比较 rtol; 未传则用配置/默认值")
    p.add_argument("--verify-atol", type=float, default=None,
                   help="verify 精度比较 atol; 未传则用配置/默认值")
    p.add_argument("--log-dir", type=str, default="",
                   help="KernelVerifier log 根目录")
    p.add_argument("--keep-verifier-artifacts",
                   action=argparse.BooleanOptionalAction,
                   default=None,
                   help="是否保留 KernelVerifier verify/profile 临时工作目录; 默认自动清理")
    p.add_argument("--report-dir", type=str, default="",
                   help="批处理报告输出目录")
    p.add_argument("--mode", type=str, default="",
                   choices=["", "correctness", "performance", "full"],
                   help="验证模式 (correctness=精度; performance/full=精度+性能)")
    p.add_argument("--verifier-mode", type=str, default="",
                   choices=["", "opencode", "direct"],
                   help="opencode=经 skill 走 LLM 语义层反作弊+精度+性能 (默认); "
                        "direct=直调 KernelVerifier, 跳过 LLM 语义层, 用于离线 dev 调试")
    p.add_argument("--validator-agent", type=str, default="",
                   help="opencode 模式 agent 名 (默认 pypto-kernel-validator)")
    p.add_argument("--skill-timeout", type=int, default=0,
                   help="opencode 模式 skill 子进程硬超时 (秒, 默认 1800)")
    p.add_argument("--skill-retry", type=int, default=None,
                   help="opencode/API 层失败后的重试次数; 业务验证失败不重试 (默认 2)")
    p.add_argument("--skill-retry-interval", type=int, default=None,
                   help="opencode/API 层重试间隔秒数 (默认 600)")
    p.add_argument("--skip-pypto-gen", action="store_true",
                   help="跳过 pypto 生成, 仅复跑验证 (要求产物已就绪)")
    p.add_argument("--force-regen", action="store_true",
                   help="即使产物齐全也强制重跑 pypto 生成")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    return p


def _setup_logging(level: str) -> None:
    norm = "WARNING" if level == "WARN" else level
    logging.basicConfig(
        level=getattr(logging, norm),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_stop_event: Optional[threading.Event] = None


def _signal_handler(signum: int, _frame: Any) -> None:
    """SIGTERM/SIGHUP 转为 KeyboardInterrupt, 并通知子线程停止."""
    global _stop_event
    if _stop_event is not None:
        _stop_event.set()
    raise KeyboardInterrupt()


def main(argv: Optional[List[str]] = None) -> int:
    global _stop_event
    _stop_event = threading.Event()

    # SIGTERM: monitor.py _kill_tree 发送; SIGHUP: shell 在 test-integration.sh
    # 退出时发送. 两者都转为 KeyboardInterrupt 让 asyncio.run 有机会取消.
    # 同时 set _stop_event, 让 asyncio.to_thread 里的子线程从 sleep 中醒来.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    # SIGINT (Ctrl+C) 默认就是 KeyboardInterrupt, 不需要注册.

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    yaml_cfg = load_yaml_config(args.config)
    cfg = _build_cfg(args, yaml_cfg)

    yaml_bench = (yaml_cfg.get("bench_dir") or "").strip()
    if args.bench_dir is not None:
        bench_dir = args.bench_dir
    elif yaml_bench:
        bench_dir = Path(yaml_bench).expanduser()
    else:
        # 默认指向桥接层 .cache/KernelBench/KernelBench (download_kernelbench.sh 落地点)
        bench_dir = Path(__file__).resolve().parent / ".cache" / "KernelBench" / "KernelBench"
    if not bench_dir.is_absolute():
        # 相对路径优先按当前 CWD 解析; 若不存在再回退到 pypto_repo_root.
        cwd_candidate = (Path.cwd() / bench_dir).resolve()
        repo_candidate = (cfg.pypto_repo_root / bench_dir).resolve()
        if cwd_candidate.exists():
            bench_dir = cwd_candidate
        elif repo_candidate.exists():
            bench_dir = repo_candidate
        else:
            bench_dir = cwd_candidate  # 报错时给个明确的路径

    level_cfg = args.level or yaml_cfg.get("level", "level1")
    if isinstance(level_cfg, (list, tuple)):
        default_levels = [str(item).strip() for item in level_cfg if str(item).strip()]
    else:
        default_levels = parse_csv_str_list(str(level_cfg))
    default_levels = default_levels or ["level1"]
    if args.level and len(default_levels) > 1:
        parser.error(
            "多 level 不再使用 --level 表达; 请直接写 "
            "--cases 'level1=1:21;level2=31,41:50'."
        )

    try:
        cases_text = args.cases or os.environ.get("CASES", "")
        cases_by_level = parse_cases_by_level(cases_text, default_levels)
    except ValueError as e:
        parser.error(str(e))
    levels = list(cases_by_level.keys())
    cfg.use_level_dirs = len(levels) > 1

    case_paths: List[Path] = []
    for level in levels:
        level_dir = bench_dir / level
        if not level_dir.exists():
            parser.error(
                f"level dir 不存在: {level_dir}\n"
                f"  bench_dir = {bench_dir}\n"
                f"  level     = {level}\n"
                f"请先运行: bash pypto/integration/benchmark/scripts/download_kernelbench.sh"
            )
        requested = cases_by_level.get(level)
        case_paths.extend(discover_cases(level_dir, requested=requested, limit=args.limit))
    if not case_paths:
        parser.error(f"未发现可执行用例 (bench_dir={bench_dir}, levels={levels})")

    devices = parse_csv_int_list(args.devices) if args.devices else (yaml_cfg.get("devices") or [0])
    concurrency = args.concurrency or yaml_cfg.get("concurrency") or len(devices)

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"pypto_repo_root  = {cfg.pypto_repo_root}")
    logger.info(f"bench_dir        = {bench_dir}")
    logger.info(f"levels           = {levels}")
    logger.info(f"cases            = {[p.stem for p in case_paths]}")
    logger.info(f"devices          = {devices}, concurrency = {concurrency}")
    logger.info(f"skip_stage7      = {cfg.skip_stage7_perf_tune}")
    logger.info(f"arch / backend   = {cfg.arch} / {cfg.backend}")
    logger.info(f"report_dir       = {cfg.report_dir}")

    records = asyncio.run(run_batch(case_paths, devices, concurrency, cfg))

    summary_paths = write_summary(
        records, cfg.report_dir,
        meta={
            "bench_dir": str(bench_dir),
            "level": ",".join(levels),
            "levels": levels,
            "kernelbench_commit": "21fbe5a642898cd60b8f60c7aefb43d475e11f33",
            "arch": cfg.arch,
            "backend": cfg.backend,
            "framework": cfg.framework,
            "mode": cfg.mode,
            "verifier_mode": cfg.verifier_mode,
            "validator_agent": cfg.validator_agent if cfg.verifier_mode == "opencode" else None,
            "skip_stage7_perf_tune": cfg.skip_stage7_perf_tune,
            "devices": devices,
            "concurrency": concurrency,
            "pypto_repo_root": str(cfg.pypto_repo_root),
        },
    )

    success_n = sum(1 for r in records if r.succeeded)
    total = len(records)
    logger.info(f"完成: {success_n}/{total} 通过")
    logger.info(f"summary: {summary_paths['md']}")

    return 0 if success_n == total else 1


if __name__ == "__main__":
    sys.exit(main())
