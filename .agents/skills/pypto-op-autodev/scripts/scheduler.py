#!/usr/bin/env python3
"""scheduler.py — PyPTO AutoDev 纯脚本调度器

权威流程定义：.agents/skills/pypto-op-autodev/SKILL.md
本文件是 SKILL.md 的薄 runner 实现，不包含独立业务逻辑。
调度层只做确定性编排，AI 工作由 opencode agent 承载。

步骤映射（scheduler.py → SKILL.md）:
    _step0_check_env          → Step 0  环境预检            (纯脚本)
    _step1_reset_stale        → Step 1  并发控制与中断恢复  (纯脚本)
    _step2_select_op          → Step 2  选择目标算子        (纯脚本 + AI)
      ├ 2a 检查用户新需求     → Step 2a                     (纯脚本)
      ├ 2b 选择下一个算子     → Step 2b                     (纯脚本)
      └ 2c 按需 discover      → Step 2c                     (AI · §A)
    _step3_prepare_task       → Step 3  标记开始+准备任务    (纯脚本)
    _step4_run_orchestrator   → Step 4  调度 orchestrator    (AI · §B 架构差异)
    _step5_verify             → Step 5  独立验证             (纯脚本)
    _step6_fracture_detection → Step 6  断裂点检测           (AI · §C)
    _step7_update_final       → Step 7  更新最终状态         (纯脚本)
    _step8_health_check       → Step 8  健康检查+进度报告    (纯脚本)

用法:
    python scheduler.py --dry-run   # 只选算子不执行
    python scheduler.py             # 执行一次完整流程
    python scheduler.py --config /path/to/scheduler.conf.json
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__version__ = "2.1.0"


# ── 配置 ──────────────────────────────────────────────────────────────
@dataclass
class SchedulerConfig:
    """调度器配置，替代原始 Dict[str, Any]，提供类型安全和默认值。"""

    pypto_root: str = "/workspace/projects/code/pypto"
    csv_path: str = "autodev/scan_results.csv"
    custom_dir: str = "autodev/custom"
    log_file: str = "autodev/logs/scheduler.log"
    pid_file: str = "autodev/run/scheduler.pid"
    scripts_dir: str = ".agents/skills/pypto-op-autodev/scripts"

    stale_timeout_hours: int = 6
    dev_timeout_sec: int = 5400
    script_timeout_sec: int = 300
    verify_test_timeout_sec: int = 600

    opencode_bin: str = "opencode"
    orchestrator_agent: str = "pypto-op-orchestrator"
    discover_agent: str = "pypto-op-discover"

    dev_strategy: str = "auto"
    enable_discover: bool = False
    enable_fracture: bool = True
    
    # CLI 参数（不从配置文件加载）
    dry_run: bool = False  # dry-run 模式（只选算子不执行）

    # 允许保留未知配置项（兼容 conf.json 中的自定义字段）
    _extra: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def _coerce_known_field(cls, key: str, value: object) -> object:
        bool_fields = {
            "enable_discover", "enable_fracture", "dry_run",
        }
        int_fields = {
            "stale_timeout_hours", "dev_timeout_sec",
            "script_timeout_sec", "verify_test_timeout_sec",
        }
        str_fields = {
            "pypto_root", "csv_path", "custom_dir", "log_file", "pid_file",
            "scripts_dir",
            "opencode_bin", "orchestrator_agent",
            "discover_agent", "dev_strategy",
        }

        if key in bool_fields:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            raise ValueError(f"invalid bool for {key}: {value!r}")

        if key in int_fields:
            if isinstance(value, bool):
                raise ValueError(f"invalid int for {key}: {value!r}")
            if not isinstance(value, (str, int, float)):
                raise ValueError(f"invalid int for {key}: {value!r}")
            try:
                ivalue = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid int for {key}: {value!r}") from exc
            if ivalue < 0:
                raise ValueError(f"negative value for {key}: {ivalue}")
            return ivalue

        if key in str_fields:
            if value is None:
                raise ValueError(f"invalid str for {key}: None")
            return str(value)

        return value

    @classmethod
    def from_file(cls, config_path: Path | None = None) -> SchedulerConfig:
        """从 JSON 配置文件加载，未知字段存入 _extra。"""
        cfg = cls()
        if config_path is None:
            # 默认从 scheduler.py 同目录加载 conf.json
            config_path = Path(__file__).resolve().parent / "scheduler.conf.json"
        if config_path.is_absolute():
            config_path = config_path.resolve()
        else:
            config_path = (Path.cwd() / config_path).resolve()
        if config_path.is_file():
            try:
                with open(config_path, encoding="utf-8") as f:
                    overrides = json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                print(f"[WARN] 配置加载失败，使用默认值: {exc}", file=sys.stderr)
                return cfg
            known_fields = {f.name for f in cls.__dataclass_fields__.values()
                           if f.name != "_extra"}
            for key, value in overrides.items():
                if key in known_fields:
                    try:
                        coerced = cls._coerce_known_field(key, value)
                    except ValueError as exc:
                        print(f"[WARN] 忽略非法配置项 {key}: {exc}", file=sys.stderr)
                        continue
                    setattr(cfg, key, coerced)
                else:
                    cfg._extra[key] = value
        return cfg


# ── PID 文件锁 ──────────────────────────────────────────────────────
class PidLock:
    """基于 flock 的进程级互斥锁，防止调度器并发运行。

    支持上下文管理器协议::

        with PidLock(path) as acquired:
            if not acquired:
                return  # 已有实例运行
            ...  # 正常业务
    """

    def __init__(self, pid_path: Path) -> None:
        self._path = pid_path
        self._fd: int | None = None

    # -- 上下文管理器 --------------------------------------------------

    def __enter__(self) -> bool:
        """进入上下文时尝试获取锁，返回是否成功。"""
        return self.acquire()

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None,
                 exc_tb: object) -> None:
        """离开上下文自动释放锁。"""
        self.release()

    # -- 核心方法 ------------------------------------------------------

    def acquire(self) -> bool:
        """尝试获取锁。成功返回 True，已有其他进程持有返回 False。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._fd)
            self._fd = None
            return False
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, f"{os.getpid()}\n".encode())
        return True

    def release(self) -> None:
        """释放锁并清理 PID 文件。"""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError as exc:
                logging.getLogger("scheduler").warning("释放 PID 锁失败: %s", exc)
            self._fd = None
            try:
                self._path.unlink(missing_ok=True)
            except OSError as exc:
                logging.getLogger("scheduler").warning("删除 PID 文件失败: %s", exc)


# ── 工具函数 ─────────────────────────────────────────────────────────
def setup_logging(log_file: Path) -> logging.Logger:
    """配置日志：同时输出到文件和 stderr。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def run_script(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    logger: logging.Logger,
    label: str = "",
) -> subprocess.CompletedProcess[str]:
    """运行确定性脚本，统一日志和超时处理。"""
    tag = "[%s] " % label if label else ""
    logger.info("%s执行: %s", tag, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        logger.error("%s超时 (%ds)", tag, timeout)
        raise
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[:20]:
            logger.info("%sstdout: %s", tag, line)
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[:10]:
            logger.info("%sstderr: %s", tag, line)
    logger.info("%s退出码: %d", tag, result.returncode)
    return result


def run_agent(
    agent: str,
    prompt: str,
    cwd: Path,
    timeout: int,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> subprocess.CompletedProcess[str]:
    """运行 opencode agent，统一日志和超时处理。"""
    cmd = [
        config.opencode_bin, "run",
        "--agent", agent,
        prompt,
    ]
    logger.info("[Agent:%s] 启动（超时 %ds）", agent, timeout)
    logger.info("[Agent:%s] prompt: %s", agent, prompt[:200])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        logger.error("[Agent:%s] 超时 (%ds)", agent, timeout)
        raise
    logger.info("[Agent:%s] 退出码: %d", agent, result.returncode)
    return result


def parse_json_output(raw: str, logger: logging.Logger) -> dict[str, object] | None:
    """安全解析子进程的 JSON 输出，失败返回 None。

    使用 ``json.JSONDecoder.raw_decode`` 从右侧搜索最后一个 ``{``
    开头的 JSON 对象，比手动逐字符扫描更健壮（能正确处理字符串
    内的花括号）。
    """
    text = raw.strip()
    if not text:
        return None
    # 快速路径：整段文本就是合法 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 慢速路径：从后向前找最后一个 '{' 开头的 JSON 对象
    decoder = json.JSONDecoder()
    last_result: dict[str, object] | None = None
    search_start = 0
    while search_start < len(text):
        idx = text.find("{", search_start)
        if idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                last_result = obj
            search_start = end_idx
        except json.JSONDecodeError:
            search_start = idx + 1
    if last_result is not None:
        return last_result
    logger.warning("无法解析 JSON 输出: %s", text[:200])
    return None


def _update_status(
    python: str,
    pypto_root: Path,
    scripts_dir: Path,
    csv_path: Path,
    op_name: str,
    status: str,
    config: SchedulerConfig,
    logger: logging.Logger,
    dev_result: str | None = None,
) -> None:
    """辅助：调用 update_op.py 更新算子状态。"""
    cmd = [
        python, str(scripts_dir / "update_op.py"),
        "--csv", str(csv_path),
        "--op", op_name,
        "--status", status,
    ]
    if dev_result:
        cmd.extend(["--dev-result", dev_result])
    try:
        run_script(
            cmd, cwd=pypto_root, timeout=config.script_timeout_sec,
            logger=logger, label="UpdateStatus",
        )
    except Exception as exc:
        logger.error("状态更新失败: %s", exc)


def _step0_check_env(
    scripts_dir: Path,
    python: str,
    pypto_root: Path,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> tuple[bool, str]:
    if not config.enable_env_check:
        logger.info("Step 0: 环境预检已禁用，跳过")
        return True, "skipped"

    logger.info("Step 0: 环境预检")
    try:
        result = run_script(
            [python, str(scripts_dir / "check_env.py"),
             "--csv", str(pypto_root / config.csv_path),
             "--work-dir", str(pypto_root / config.custom_dir)],
            cwd=pypto_root,
            timeout=config.script_timeout_sec,
            logger=logger,
            label="Step0",
        )
    except subprocess.TimeoutExpired:
        logger.error("环境预检超时")
        return False, "error"
    except Exception as exc:
        logger.error("环境预检异常: %s", exc)
        return False, "error"

    if result.returncode == 0:
        logger.info("环境预检通过")
        return True, "ok"
    if result.returncode == 1:
        logger.warning("环境预检告警，将以降级模式继续")
        return True, "degraded"
    logger.error("环境预检失败（critical）")
    return False, "critical"


def _step7_update_final_status(
    scripts_dir: Path,
    python: str,
    pypto_root: Path,
    csv_path: Path,
    op_name: str,
    dev_success: bool,
    dev_detail: str,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> str:
    logger.info("Step 7: 更新状态")
    final_status = "completed" if dev_success else "failed"
    _update_status(
        python, pypto_root, scripts_dir, csv_path, op_name, final_status,
        config, logger, dev_result=dev_detail,
    )
    return final_status


def detect_dev_success(
    op_dir: Path,
    op_name: str,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """检测算子开发是否成功。只读 .dev_result.json。"""
    result_file = op_dir / ".dev_result.json"
    if result_file.is_file():
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            status = data.get("status", "unknown")
            logger.info(".dev_result.json: status=%s", status)
            if status == "SUCCESS":
                return True, "SUCCESS"
            blocked = data.get("blocked_reason", "")
            notes = data.get("notes", "")
            detail = blocked or notes or status
            return False, detail
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("读取 .dev_result.json 失败: %s", exc)

    # fallback：.dev_result.json 不存在，检查关键产物
    impl_file = op_dir / f"{op_name}_impl.py"
    test_file = op_dir / f"test_{op_name}.py"
    if impl_file.is_file() and test_file.is_file():
        logger.info("fallback: impl + test 文件存在（无 .dev_result.json）")
        return True, "fallback: impl+test exist"

    return False, "no .dev_result.json and missing artifacts"


# ── 流程步骤 ─────────────────────────────────────────────────────────
# 将原 main_pipeline 中的每个 Step 拆为独立函数，便于测试和维护。

def _step1_reset_stale(
    scripts_dir: Path,
    python: str,
    pypto_root: Path,
    csv_path: Path,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> bool:
    """Step 1: 并发控制 — 重置 stale 任务。

    返回 True 表示可以继续，False 表示存在活跃任务，本轮应跳过。
    """
    logger.info("Step 1: 并发控制检查（重置 stale 任务）")
    result = run_script(
        [python, str(scripts_dir / "update_op.py"),
         "--csv", str(csv_path),
         "--reset-stale",
         "--timeout-hours", str(config.stale_timeout_hours)],
        cwd=pypto_root,
        timeout=config.script_timeout_sec,
        logger=logger,
        label="Step1",
    )
    if result.returncode == 1:
        logger.info("有活跃任务（未超时），本轮跳过")
        return False
    return True


def _step2_select_op(
    scripts_dir: Path,
    python: str,
    pypto_root: Path,
    csv_path: Path,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> tuple[str | None, dict[str, object]]:
    """Step 2: 选择算子。

    返回 (算子名称, selected_info)；无候选时返回 (None, {})。
    """
    logger.info("Step 2: 选择算子")
    result = run_script(
        [python, str(scripts_dir / "select_next_op.py"),
         "--csv", str(csv_path)],
        cwd=pypto_root,
        timeout=config.script_timeout_sec,
        logger=logger,
        label="Step2",
    )

    if result.returncode == 1:
        # 退出码 1 = 需要触发 discover
        if config.enable_discover:
            logger.info("Step 2b: 无候选算子，触发 discover agent")
            try:
                run_agent(
                    config.discover_agent,
                    "扫描 PyPTO 项目，发现新的待实现算子，更新 scan_results.csv",
                    cwd=pypto_root,
                    timeout=config.dev_timeout_sec,
                    config=config,
                    logger=logger,
                )
            except Exception as exc:
                logger.error("Discover agent 失败: %s", exc)
            result = run_script(
                [python, str(scripts_dir / "select_next_op.py"),
                 "--csv", str(csv_path)],
                cwd=pypto_root,
                timeout=config.script_timeout_sec,
                logger=logger,
                label="Step2-retry",
            )
            if result.returncode != 0:
                logger.info("discover 后仍无候选算子")
                return None, {}
        else:
            logger.info("无候选算子（discover 未启用）")
            return None, {}
    elif result.returncode != 0:
        raise RuntimeError(
            f"select_next_op 异常退出: {result.returncode}"
        )

    data = parse_json_output(result.stdout, logger)
    if not data or "selected" not in data:
        raise RuntimeError("select_next_op 输出格式异常")
    selected = data["selected"]
    if not isinstance(selected, dict):
        raise RuntimeError("select_next_op 输出的 'selected' 不是对象")
    op_name = str(selected.get("op_name", "")).strip()
    if not op_name:
        raise RuntimeError("select_next_op 输出缺少有效 op_name")
    logger.info("选中: %s (score=%s)", op_name, selected.get("score", "?"))
    return op_name, selected


def _step3_prepare_task(
    op_name: str,
    custom_dir: Path,
    start_time: str,
    selected_info: dict[str, object],
    logger: logging.Logger,
) -> Path:
    """Step 3: 准备任务文件。返回算子工作目录。"""
    logger.info("Step 3: 准备任务文件")
    op_dir = custom_dir / op_name
    op_dir.mkdir(parents=True, exist_ok=True)

    resume = selected_info.get("resume_from_stage", 1)
    complexity = selected_info.get("complexity", "medium")
    category = selected_info.get("category", "other")

    task_file = op_dir / "autodev-task.md"
    task_content = textwrap.dedent(f"""\
        # 开发任务
        - op_name: {op_name}
        - working_dir: {op_dir.as_posix()}/
        - resume_from_stage: {resume}
        - complexity: {complexity}
        - category: {category}
        - start_time: {start_time}
    """)
    task_file.write_text(task_content, encoding="utf-8")
    logger.info("任务文件: %s", task_file)
    return op_dir


def _resolve_dev_strategy(
    op_name: str,
    csv_path: Path,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> str:
    """按规则决定开发策略。返回 "orchestrator" 或 "workflow"。"""
    strategy = config.dev_strategy
    if strategy in ("orchestrator", "workflow"):
        logger.info("策略（配置指定）: %s", strategy)
        return strategy

    # auto 决策
    try:
        sys.path.insert(0, str(Path(config.scripts_dir) / "data"))
        from csv_ops import read_csv
        rows = read_csv(str(csv_path), op_name=op_name)
        if rows:
            row = rows[0]
            if (row.get("status") == "failed"
                    and row.get("last_strategy") == "orchestrator"):
                logger.info("策略（auto: 上次 orchestrator 失败）: workflow")
                return "workflow"
            if row.get("complexity") == "easy":
                logger.info("策略（auto: easy 算子）: workflow")
                return "workflow"
    except Exception as exc:
        logger.warning("策略决策异常，使用 orchestrator: %s", exc)

    logger.info("策略（auto 默认）: orchestrator")
    return "orchestrator"


def _build_dev_prompt(
    op_name: str,
    op_dir: Path,
    pypto_root: Path,
    selected_info: dict[str, object],
) -> str:
    """构建调用执行层的 prompt。"""
    resume = selected_info.get("resume_from_stage", 1)
    complexity = selected_info.get("complexity", "medium")
    category = selected_info.get("category", "other")

    # 读取 known-limitations
    kl_path = pypto_root / "autodev" / "known-limitations.md"
    kl_content = ""
    if kl_path.is_file():
        kl_content = kl_path.read_text(encoding="utf-8")[:3000]

    kl_section = ""
    if kl_content:
        kl_section = (
            f"\n参考已知框架限制（开发时注意规避）:\n{kl_content}\n"
        )

    return textwrap.dedent(f"""\
        开发算子 {op_name}。
        工作目录: {op_dir.as_posix()}/
        从 Stage {resume} 开始。
        复杂度: {complexity}，类别: {category}。
        {kl_section}
        完成后在工作目录写 .dev_result.json，格式:
        {{"status": "SUCCESS|FAILED|BLOCKED", "blocked_reason": "...|null",
         "precision_result": "PASS|FAIL|null", "completed_stages": [...],
         "artifacts": [...], "notes": "..."}}

        开发过程持续追加写入 dev-log.md（报错、方案变更、API 限制、workaround）。
    """)


def _step4_develop(
    op_name: str,
    op_dir: Path,
    pypto_root: Path,
    strategy: str,
    selected_info: dict[str, object],
    config: SchedulerConfig,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Step 4: 调用执行层开发算子。返回 (成功, 详情)。"""
    prompt = _build_dev_prompt(op_name, op_dir, pypto_root, selected_info)
    agent_name = config.orchestrator_agent if strategy == "orchestrator" else "pypto-op-workflow-runner"

    logger.info("Step 4: 策略=%s, agent=%s", strategy, agent_name)
    try:
        agent_result = run_agent(
            agent_name, prompt,
            cwd=pypto_root,
            timeout=config.dev_timeout_sec,
            config=config,
            logger=logger,
        )
        if agent_result.returncode == 0:
            return detect_dev_success(op_dir, op_name, logger)
        return False, f"agent exit code {agent_result.returncode}"
    except subprocess.TimeoutExpired:
        logger.error("开发超时 (%ds)", config.dev_timeout_sec)
        return False, "TIMEOUT"
    except Exception as exc:
        logger.error("开发异常: %s", exc)
        return False, f"exception: {exc}"


def _step5_verify(
    scripts_dir: Path,
    op_name: str,
    op_dir: Path,
    pypto_root: Path,
    csv_path: Path,
    python: str,
    config: SchedulerConfig,
    logger: logging.Logger,
    dev_success: bool,
) -> tuple[bool, str | None]:
    """Step 5: 验证算子。返回 (是否通过, 失败原因或 None)。"""
    if not dev_success:
        logger.info("Step 5: 跳过验证（开发未成功）")
        return dev_success, None
    if not config.enable_verify_test:
        logger.info("Step 5: 验证未启用，跳过")
        return dev_success, None

    logger.info("Step 5: 验证算子")
    try:
        verify_result = run_script(
            [python, str(scripts_dir / "verify_op.py"),
             "--csv", str(csv_path),
             "--op", op_name,
             "--op-dir", str(op_dir),
             "--run-test",
             "--timeout", str(config.verify_test_timeout_sec)],
            cwd=pypto_root,
            timeout=config.verify_test_timeout_sec + 60,
            logger=logger,
            label="Step5",
        )
        if verify_result.returncode != 0:
            logger.warning("验证未通过，标记为 failed")
            return False, "verify failed"
    except subprocess.TimeoutExpired:
        logger.error("验证超时")
        return False, "verify timeout"
    return True, None


def _step6_fracture_detection(
    op_name: str,
    op_dir: Path,
    pypto_root: Path,
    config: SchedulerConfig,
    logger: logging.Logger,
) -> None:
    """Step 6: 断裂点检测（非致命）。"""
    if not config.enable_fracture:
        logger.info("Step 6: 断裂点检测未启用，跳过")
        return
    logger.info("Step 6: 断裂点检测")
    dev_log = op_dir / "dev-log.md"
    if not dev_log.is_file():
        logger.info("无 dev-log.md，跳过断裂点检测")
        return
    try:
        run_agent(
            config.fracture_agent,
            f"分析 {str(op_dir / 'dev-log.md')}，检测断裂点",
            cwd=pypto_root,
            timeout=config.dev_timeout_sec,
            config=config,
            logger=logger,
        )
    except Exception as exc:
        logger.warning("断裂点检测失败（非致命）: %s", exc)


def _step8_health_check(
    scripts_dir: Path,
    op_name: str,
    op_dir: Path,
    pypto_root: Path,
    python: str,
    config: SchedulerConfig,
    logger: logging.Logger,
    final_status: str,
    dev_detail: str,
    start_time: str,
) -> None:
    """Step 8: 健康检查（非致命）。"""
    logger.info("Step 8: 健康检查")
    try:
        run_script(
            [python, str(scripts_dir / "health_check.py"),
             "--op", op_name,
             "--op-dir", str(op_dir),
             "--status", final_status,
             "--dev-result", dev_detail,
             "--start-time", start_time],
            cwd=pypto_root,
            timeout=config.script_timeout_sec,
            logger=logger,
            label="Step8",
        )
    except Exception as exc:
        logger.warning("健康检查失败（非致命）: %s", exc)


# ── 主流程 ───────────────────────────────────────────────────────────
def main_pipeline(config: SchedulerConfig, dry_run: bool = False) -> int:
    """8 步主流程。返回退出码。"""
    pypto_root = Path(config.pypto_root)
    csv_path = pypto_root / config.csv_path
    custom_dir = pypto_root / config.custom_dir
    scripts_dir = Path(config.scripts_dir)
    if not scripts_dir.is_absolute():
        scripts_dir = pypto_root / scripts_dir
    python = sys.executable
    log_file = pypto_root / config.log_file

    # 自动创建运行时目录
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    custom_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    Path(pypto_root / config.pid_file).parent.mkdir(parents=True, exist_ok=True)

    # 在 try 之前初始化 logger，防止 except 中 NameError
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("PyPTO AutoDev Scheduler v%s 启动", __version__)
    logger.info("dry_run=%s", dry_run)
    logger.info("=" * 60)

    op_name: str | None = None
    start_time = datetime.now(timezone.utc).isoformat()

    with PidLock(pypto_root / config.pid_file) as acquired:
        if not acquired:
            logger.info("已有调度器实例运行，退出")
            return 0

        try:
            if dry_run:
                logger.info("dry-run 模式，跳过环境预检")
            else:
                can_proceed, env_detail = _step0_check_env(
                    scripts_dir, python, pypto_root, config, logger,
                )
                if not can_proceed:
                    logger.error("环境预检未通过: %s", env_detail)
                    return 2

            # Step 1
            if not _step1_reset_stale(scripts_dir, python, pypto_root, csv_path, config, logger):
                return 0

            # Step 2
            try:
                op_name, selected_info = _step2_select_op(
                    scripts_dir, python, pypto_root, csv_path, config, logger,
                )
            except RuntimeError as exc:
                logger.error("%s", exc)
                return 2

            if op_name is None:
                return 0

            if dry_run:
                logger.info("dry-run 模式，选中 %s，不执行开发", op_name)
                return 0

            # Step 3
            _update_status(
                python, pypto_root, scripts_dir, csv_path, op_name, "in_progress",
                config, logger,
            )
            op_dir = _step3_prepare_task(
                op_name, custom_dir, start_time, selected_info, logger,
            )

            # 策略选择
            strategy = _resolve_dev_strategy(
                op_name, csv_path, config, logger,
            )

            # Step 4
            dev_success, dev_detail = _step4_develop(
                op_name, op_dir, pypto_root, strategy, selected_info,
                config, logger,
            )
            logger.info(
                "开发结果: strategy=%s, success=%s, detail=%s",
                strategy, dev_success, dev_detail,
            )

            # Step 5：仅成功时验证
            if dev_success:
                verified, fail_reason = _step5_verify(
                    scripts_dir, op_name, op_dir, pypto_root, csv_path,
                    python, config, logger, dev_success,
                )
                if fail_reason:
                    dev_success = False
                    dev_detail = fail_reason

            # Step 6：知识提取
            _step6_fracture_detection(
                op_name, op_dir, pypto_root, config, logger,
            )
            # Step 6b: 更新 known-limitations
            fracture_summary = op_dir / "fracture-summary.json"
            if fracture_summary.is_file():
                kl_path = pypto_root / "autodev" / "known-limitations.md"
                try:
                    run_script(
                        [python, str(scripts_dir / "update_known_limitations.py"),
                         "--known-limitations", str(kl_path),
                         "--fracture-summary", str(fracture_summary)],
                        cwd=pypto_root, timeout=config.script_timeout_sec,
                        logger=logger, label="Step6b",
                    )
                except Exception as exc:
                    logger.warning("known-limitations 更新失败（非致命）: %s", exc)

            # Step 7: 更新状态
            final_status = "completed" if dev_success else "failed"
            step7_cmd = [
                python, str(scripts_dir / "update_op.py"),
                "--csv", str(csv_path),
                "--op", op_name,
                "--status", final_status,
                "--dev-result", dev_detail,
                "--last-strategy", strategy,
            ]
            logger.info("Step 7: 更新状态 → %s", final_status)
            try:
                run_script(
                    step7_cmd, cwd=pypto_root,
                    timeout=config.script_timeout_sec,
                    logger=logger, label="Step7",
                )
            except Exception as exc:
                logger.error("Step 7 状态更新失败: %s", exc)

            # Step 8
            _step8_health_check(
                scripts_dir, op_name, op_dir, pypto_root, python, config, logger,
                final_status, dev_detail, start_time,
            )

            logger.info("=" * 60)
            logger.info("完成: %s -> %s (%s)", op_name, final_status, dev_detail)
            logger.info("=" * 60)

            return 0 if dev_success else 1

        except Exception as exc:
            logger.exception("主流程异常: %s", exc)
            if op_name:
                _update_status(
                    python, pypto_root, scripts_dir, csv_path, op_name, "failed",
                    config, logger,
                    dev_result=f"scheduler exception: {exc}",
                )
            return 3


# ── 入口 ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyPTO AutoDev 调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只选算子不执行开发",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (默认: 与 scheduler.py 同目录的 scheduler.conf.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = SchedulerConfig.from_file(config_path)

    sys.exit(main_pipeline(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
