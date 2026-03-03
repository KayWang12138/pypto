#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPTO Performance Autotuner - Iterative Search Strategy

Searches optimal NPU performance configurations through layered iterative tuning.
Supports Grid Search, Random Search + Grid Refinement, and Bayesian (TPE) strategies.

Usage:
    python3 autotune.py --dry-run
    python3 autotune.py --search-space search_space.json --baseline-dir /path/to/output
    python3 autotune.py --search-space search_space.json --benchmark-cmd "python3 run.py"
"""

# pyright: basic, reportDeprecated=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportUnnecessaryIsInstance=false
# pyright: reportUnusedCallResult=false

import argparse
import copy
import itertools
import json
import logging
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
SKILLS_DIR = SKILL_DIR.parent

logger = logging.getLogger(__name__)
DEFAULT_SEARCH_SPACE = (SCRIPT_DIR / "../references/search_space_schema.json").resolve()
DEFAULT_ANALYZER_SCRIPT = (
    SKILLS_DIR / "pypto-performance-analyzer" / "scripts" / "analyze.py"
).resolve()
DEFAULT_OUTPUT_DIR = Path("./autotune_results")

DEFAULT_TARGET_METRIC = "total_latency_ms"
DEFAULT_BUDGET = 200
DEFAULT_TIME_BUDGET_MIN = 120
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_IMPROVEMENT_THRESHOLD = 0.02
DEFAULT_BENCHMARK_TIMEOUT_SEC = 300
MAX_RETRIES = 3
SUMMARY_FILE_NAME = "analysis_summary.json"


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Return stable JSON for hashing/comparison."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a float between low/high bounds."""
    return max(low, min(high, value))


def flatten_layers(config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten layered config into a single key-value dict."""
    flat: Dict[str, Any] = {}
    for _, layer_cfg in config.items():
        if isinstance(layer_cfg, dict):
            flat.update(layer_cfg)
    return flat


def merge_layer_config(
    fixed_layers: Dict[str, Dict[str, Any]],
    layer_name: str,
    candidate: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Merge fixed best layers with current layer candidate."""
    merged = copy.deepcopy(fixed_layers)
    merged[layer_name] = copy.deepcopy(candidate)
    return merged


def safe_float(value: Any) -> Optional[float]:
    """Convert to float if possible, otherwise None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def find_analysis_summary(base: Optional[Path]) -> Optional[Path]:
    """Find analysis_summary.json from a file path or directory path."""
    if base is None:
        return None
    if base.is_file() and base.name == SUMMARY_FILE_NAME:
        return base
    if base.is_file():
        return None
    direct = base / SUMMARY_FILE_NAME
    if direct.exists():
        return direct
    for candidate in base.rglob(SUMMARY_FILE_NAME):
        return candidate
    return None


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON object from file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


@dataclass
class TrialOutcome:
    """Trial result payload for recorder."""

    trial_id: int
    layer: str
    config: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    status: str
    error: Optional[str]
    timestamp: str
    env: Dict[str, Any]


@dataclass
class LayerResult:
    """Per-layer execution summary."""

    layer_name: str
    strategy: str
    space_size: int
    trial_budget: int
    success_count: int
    fail_count: int
    pruned_count: int
    best_metric: Optional[float]
    stop_reason: Optional[str]


class SearchSpaceLoader:
    """Load and validate search space definitions."""

    def __init__(self, search_space_path: Path):
        self.search_space_path = search_space_path

    def load(self) -> Dict[str, Any]:
        """Load search space JSON and normalize layers."""
        if not self.search_space_path.exists():
            raise FileNotFoundError("search space not found: {}".format(self.search_space_path))

        raw = load_json_file(self.search_space_path)
        layers = raw.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("search space requires non-empty 'layers' list")

        normalized_layers: List[Dict[str, Any]] = []
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError("layer at index {} must be object".format(idx))
            normalized_layers.append(self._normalize_layer(layer))

        raw["layers"] = normalized_layers
        return raw

    def estimate_layer_size(self, layer: Dict[str, Any]) -> int:
        """Estimate Cartesian size for one layer."""
        params = layer.get("params", {})
        if not isinstance(params, dict) or not params:
            return 1

        size = 1
        for _, spec in params.items():
            values = self.expand_param_values(spec)
            if not values:
                continue
            size *= len(values)
        return max(size, 1)

    def expand_param_values(self, spec: Dict[str, Any]) -> List[Any]:
        """Expand schema entry into concrete candidate values."""
        spec_type = str(spec.get("type", "choice"))
        if spec_type == "bool":
            return [True, False]
        if spec_type == "choice":
            values = spec.get("values", [])
            if not isinstance(values, list) or not values:
                raise ValueError("choice param requires non-empty values")
            return copy.deepcopy(values)
        if spec_type == "range":
            min_v = spec.get("min")
            max_v = spec.get("max")
            step_v = spec.get("step", 1)
            if not isinstance(min_v, int) or not isinstance(max_v, int):
                raise ValueError("range param requires int min/max")
            if not isinstance(step_v, int) or step_v <= 0:
                raise ValueError("range param requires positive int step")
            values: List[int] = []
            current = min_v
            while current <= max_v:
                values.append(current)
                current += step_v
            return values

        raise ValueError("unsupported param type: {}".format(spec_type))

    def _normalize_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one layer and enforce stitch max-num primary knob."""
        name = layer.get("name")
        params = layer.get("params")
        if not isinstance(name, str) or not name:
            raise ValueError("layer requires non-empty name")
        if not isinstance(params, dict) or not params:
            raise ValueError("layer '{}' requires non-empty params".format(name))

        normalized = copy.deepcopy(layer)
        normalized.setdefault("priority", 999)
        normalized.setdefault("max_trials", DEFAULT_BUDGET)

        if name == "stitch":
            normalized["params"] = self._normalize_stitch_params(params)

        for param_name, spec in normalized["params"].items():
            if not isinstance(param_name, str) or not param_name:
                raise ValueError("invalid param name in layer '{}'".format(name))
            self.expand_param_values(spec)

        return normalized

    def _normalize_stitch_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Use stitch_function_max_num as the primary stitch knob."""
        deprecated = {
            "stitch_function_num_initial",
            "stitch_function_outcast_memory",
            "stitch_function_inner_memory",
        }
        normalized: Dict[str, Any] = {}

        if "stitch_function_max_num" in params:
            normalized["stitch_function_max_num"] = copy.deepcopy(params["stitch_function_max_num"])
        else:
            derived = [16, 32, 64, 128]
            legacy_initial = params.get("stitch_function_num_initial")
            if isinstance(legacy_initial, dict):
                try:
                    values = self.expand_param_values(legacy_initial)
                    clean_values = [int(v) for v in values if isinstance(v, (int, float))]
                    clean_values = [v for v in clean_values if 1 <= v <= 256]
                    if clean_values:
                        derived = sorted(set(clean_values))
                except Exception:
                    derived = [16, 32, 64, 128]
            normalized["stitch_function_max_num"] = {"type": "choice", "values": derived}

        for key, value in params.items():
            if key in deprecated or key == "stitch_function_max_num":
                continue
            normalized[key] = copy.deepcopy(value)

        return normalized


class CandidateGenerator:
    """Generate candidates by grid/random/Bayesian strategies."""

    def __init__(self, loader: SearchSpaceLoader, rng: random.Random):
        self.loader = loader
        self.rng = rng

    def select_strategy(self, space_size: int) -> str:
        """Auto-select strategy from search-space size."""
        if space_size < 50:
            return "grid"
        if space_size <= 500:
            return "random+grid"
        return "bayesian"

    def grid_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enumerate all combinations in Cartesian product."""
        names = sorted(params.keys())
        if not names:
            return [{}]

        expanded: List[List[Any]] = []
        for name in names:
            expanded.append(self.loader.expand_param_values(params[name]))

        candidates: List[Dict[str, Any]] = []
        for combo in itertools.product(*expanded):
            item: Dict[str, Any] = {}
            for index, param_name in enumerate(names):
                item[param_name] = copy.deepcopy(combo[index])
            candidates.append(item)
        return candidates

    def random_search(self, params: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        """Generate random unique samples from parameter space."""
        if n_trials <= 0:
            return []

        names = sorted(params.keys())
        options: Dict[str, List[Any]] = {}
        for name in names:
            options[name] = self.loader.expand_param_values(params[name])

        seen = set()
        candidates: List[Dict[str, Any]] = []
        max_attempts = max(n_trials * 25, 50)

        for _ in range(max_attempts):
            if len(candidates) >= n_trials:
                break
            candidate: Dict[str, Any] = {}
            for name in names:
                candidate[name] = copy.deepcopy(self.rng.choice(options[name]))
            marker = canonical_json(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(candidate)

        return candidates

    def bayesian_suggest(
        self,
        params: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        n_trials: int = 1,
    ) -> List[Dict[str, Any]]:
        """Suggest candidates via simplified TPE-like frequency weighting."""
        if n_trials <= 0:
            return []

        if len(history) < 10:
            seed_trials = min(max(10, n_trials), 25)
            return self.random_search(params, seed_trials)[:n_trials]

        sorted_history = sorted(history, key=lambda item: float(item.get("metric", float("inf"))))
        good_n = max(1, int(len(sorted_history) * 0.25))
        good = sorted_history[:good_n]
        bad = sorted_history[good_n:]
        if not bad:
            bad = sorted_history

        names = sorted(params.keys())
        value_cache: Dict[str, List[Any]] = {}
        for name in names:
            value_cache[name] = self.loader.expand_param_values(params[name])

        suggestions: List[Dict[str, Any]] = []
        seen = set()
        attempts = max(30, n_trials * 30)

        for _ in range(attempts):
            if len(suggestions) >= n_trials:
                break
            candidate: Dict[str, Any] = {}
            for name in names:
                options = value_cache[name]
                weights: List[float] = []
                for value in options:
                    g_cnt = self._value_count(good, name, value)
                    b_cnt = self._value_count(bad, name, value)
                    g_score = float(g_cnt + 1) / float(len(good) + len(options))
                    b_score = float(b_cnt + 1) / float(len(bad) + len(options))
                    weights.append(max(g_score / b_score, 1e-6))
                candidate[name] = copy.deepcopy(self._weighted_choice(options, weights))

            marker = canonical_json(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            suggestions.append(candidate)

        if not suggestions:
            return self.random_search(params, n_trials)
        return suggestions

    def apply_guidance(
        self,
        layer_name: str,
        candidates: List[Dict[str, Any]],
        guidance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Order candidates with analyzer guidance scores."""
        if not candidates:
            return candidates
        if not guidance:
            return candidates

        labels = [str(x).lower() for x in guidance.get("bottleneck_labels", []) if isinstance(x, str)]
        top_ops = [str(x).lower() for x in guidance.get("top_ops", []) if isinstance(x, str)]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in candidates:
            score = 0.0
            if layer_name == "stitch":
                max_num = item.get("stitch_function_max_num")
                if any("idle" in s or "bubble" in s for s in labels):
                    if max_num == 64:
                        score += 3.0
                    elif max_num in (32, 128):
                        score += 1.0
                if isinstance(max_num, (int, float)) and float(max_num) > 128:
                    score -= 1.0
            elif layer_name == "matmul":
                if any("matmul" in s or "cube" in s for s in labels + top_ops):
                    if item.get("enable_multi_data_load") is True:
                        score += 1.5
                    if item.get("cube_l1_reuse_mode") == 1:
                        score += 1.5
            elif layer_name == "vector":
                if any("vector" in s or "parallel" in s for s in labels + top_ops):
                    if item.get("vec_nbuffer_mode") in (1, 2):
                        score += 1.2
                    sg_scope = item.get("sg_set_scope")
                    if isinstance(sg_scope, int) and sg_scope >= 0:
                        score += 0.8
            elif layer_name == "scheduling":
                if item.get("device_sched_mode") == 1:
                    score += 1.0

            tie_breaker = self.rng.random() * 0.001
            scored.append((score + tie_breaker, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [x[1] for x in scored]

    def build_refinement_grid(
        self,
        params: Dict[str, Any],
        base_configs: Sequence[Dict[str, Any]],
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Build local grid around top random candidates."""
        if max_candidates <= 0:
            return []
        if not base_configs:
            return []

        all_candidates: List[Dict[str, Any]] = []
        seen = set()

        for base in base_configs:
            local_options: Dict[str, List[Any]] = {}
            for name, spec in params.items():
                values = self.loader.expand_param_values(spec)
                local_options[name] = self._local_neighbors(values, base.get(name))

            names = sorted(local_options.keys())
            if not names:
                continue

            for combo in itertools.product(*[local_options[name] for name in names]):
                candidate: Dict[str, Any] = {}
                for index, param_name in enumerate(names):
                    candidate[param_name] = copy.deepcopy(combo[index])
                marker = canonical_json(candidate)
                if marker in seen:
                    continue
                seen.add(marker)
                all_candidates.append(candidate)
                if len(all_candidates) >= max_candidates:
                    return all_candidates

        return all_candidates

    def _value_count(self, rows: Sequence[Dict[str, Any]], name: str, value: Any) -> int:
        marker = canonical_json(value)
        count = 0
        for row in rows:
            cfg = row.get("config")
            if not isinstance(cfg, dict):
                continue
            if canonical_json(cfg.get(name)) == marker:
                count += 1
        return count

    def _weighted_choice(self, options: Sequence[Any], weights: Sequence[float]) -> Any:
        total = sum(max(w, 0.0) for w in weights)
        if total <= 0.0:
            return options[self.rng.randrange(0, len(options))]
        pivot = self.rng.random() * total
        cumulative = 0.0
        for idx, value in enumerate(options):
            cumulative += max(weights[idx], 0.0)
            if cumulative >= pivot:
                return value
        return options[-1]

    def _local_neighbors(self, values: List[Any], center: Any) -> List[Any]:
        if not values:
            return []
        if len(values) <= 3:
            return values

        marker = canonical_json(center)
        center_index: Optional[int] = None
        for idx, value in enumerate(values):
            if canonical_json(value) == marker:
                center_index = idx
                break
        if center_index is None:
            return values[:3]

        start = max(0, center_index - 1)
        end = min(len(values), center_index + 2)
        return values[start:end]


class PruningEngine:
    """Domain-specific pruning rules for invalid/low-value combinations."""

    def should_prune(self, flat_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Return (should_prune, reason)."""
        cube_l1_reuse_mode = flat_config.get("cube_l1_reuse_mode")
        cube_nbuffer_mode = flat_config.get("cube_nbuffer_mode", 0)
        if cube_l1_reuse_mode == 1 and cube_nbuffer_mode not in (None, 0):
            return True, "cube_l1_reuse_mode=1 requires cube_nbuffer_mode=0"

        if flat_config.get("enable_split_k") is True:
            k_tile = flat_config.get("cube_tile_k")
            if self._is_small_k_tile(k_tile):
                return True, "enable_split_k=True but cube_tile_k is too small"

        if flat_config.get("pg_skip_partition") is True:
            if flat_config.get("pg_upper_bound") is not None or flat_config.get("pg_lower_bound") is not None:
                return True, "pg_skip_partition=True conflicts with pg bounds"

        if flat_config.get("vec_nbuffer_mode") == 0:
            setting = flat_config.get("vec_nbuffer_setting")
            if setting not in (None, {}, []):
                return True, "vec_nbuffer_mode=0 ignores vec_nbuffer_setting"

        stitch_max = safe_float(flat_config.get("stitch_function_max_num"))
        if stitch_max is not None and stitch_max > 256:
            return True, "stitch_function_max_num > 256"

        return False, None

    def _is_small_k_tile(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        numeric = [int(x) for x in value if isinstance(x, (int, float))]
        if not numeric:
            return False
        return max(numeric) <= 64


class BenchmarkRunner:
    """Run benchmark + analyzer, then parse metrics for each trial."""

    def __init__(
        self,
        benchmark_cmd: Optional[str],
        baseline_dir: Optional[Path],
        analyzer_script: Optional[Path],
        dry_run: bool,
        benchmark_timeout: int,
        target_metric: str,
        rng: random.Random,
        seed: int,
    ):
        self.benchmark_cmd = benchmark_cmd
        self.baseline_dir = baseline_dir
        self.analyzer_script = analyzer_script
        self.dry_run = dry_run
        self.benchmark_timeout = benchmark_timeout
        self.target_metric = target_metric
        self.rng = rng
        self.seed = seed

    def run_trial(
        self,
        trial_id: int,
        layer_name: str,
        full_config: Dict[str, Dict[str, Any]],
        output_dir: Path,
    ) -> TrialOutcome:
        """Execute one trial and return structured outcome."""
        env = self._build_env_payload()
        timestamp = now_iso()

        if self.dry_run:
            metrics = self._simulate_metrics(full_config)
            return TrialOutcome(
                trial_id=trial_id,
                layer=layer_name,
                config=copy.deepcopy(full_config),
                metrics=metrics,
                status="success",
                error=None,
                timestamp=timestamp,
                env=env,
            )

        last_error: Optional[str] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._execute_benchmark(full_config)
                summary = self._run_analyzer(output_dir, trial_id)
                metrics = self._extract_metrics(summary)
                return TrialOutcome(
                    trial_id=trial_id,
                    layer=layer_name,
                    config=copy.deepcopy(full_config),
                    metrics=metrics,
                    status="success",
                    error=None,
                    timestamp=timestamp,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                last_error = "timeout after {}s (attempt {}/{})".format(
                    self.benchmark_timeout,
                    attempt,
                    MAX_RETRIES,
                )
            except Exception as exc:
                last_error = "{} (attempt {}/{})".format(str(exc), attempt, MAX_RETRIES)

            if attempt < MAX_RETRIES:
                time.sleep(1.0)

        return TrialOutcome(
            trial_id=trial_id,
            layer=layer_name,
            config=copy.deepcopy(full_config),
            metrics={},
            status="failed",
            error=last_error,
            timestamp=timestamp,
            env=env,
        )

    def _execute_benchmark(self, full_config: Dict[str, Dict[str, Any]]) -> None:
        if not self.benchmark_cmd:
            raise RuntimeError("benchmark command is required when not in dry-run")

        env = os.environ.copy()
        env["PYPTO_AUTOTUNE_CONFIG"] = canonical_json(full_config)

        completed = subprocess.run(
            self.benchmark_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.benchmark_timeout,
            env=env,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip().splitlines()
            stderr_tail = stderr_text[-3:] if stderr_text else []
            detail = " | ".join(stderr_tail)
            raise RuntimeError("benchmark crash (exit={}): {}".format(completed.returncode, detail))

    def _run_analyzer(self, output_dir: Path, trial_id: int) -> Dict[str, Any]:
        if self.analyzer_script is None or not self.analyzer_script.exists():
            raise RuntimeError("analyzer script not found")

        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "analysis_trial_{}.md".format(trial_id)

        cmd: List[str] = [
            sys.executable,
            str(self.analyzer_script),
            "--report-out",
            str(report_path),
        ]
        if self.baseline_dir is not None:
            cmd.extend(["--output-dir", str(self.baseline_dir)])

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.benchmark_timeout,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip().splitlines()
            stderr_tail = stderr_text[-3:] if stderr_text else []
            detail = " | ".join(stderr_tail)
            raise RuntimeError("analyzer failed (exit={}): {}".format(completed.returncode, detail))

        summary_path = self._resolve_summary_path(output_dir)
        if summary_path is None:
            raise RuntimeError("analysis_summary.json not found after analyzer")
        return load_json_file(summary_path)

    def _resolve_summary_path(self, output_dir: Path) -> Optional[Path]:
        candidates: List[Optional[Path]] = []
        if self.baseline_dir is not None:
            candidates.append(find_analysis_summary(self.baseline_dir))
        candidates.append(find_analysis_summary(output_dir))
        for maybe_path in candidates:
            if maybe_path is not None and maybe_path.exists():
                return maybe_path
        return None

    def _extract_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract required metrics from analysis_summary.json."""
        pools: List[Dict[str, Any]] = [summary]
        for key in ("metrics", "performance", "summary", "stats"):
            value = summary.get(key)
            if isinstance(value, dict):
                pools.append(value)

        latency = self._first_float(
            pools,
            ["total_latency_ms", "latency_ms", "end_to_end_latency_ms", "total_time_ms"],
        )
        kernel = self._first_float(
            pools,
            ["kernel_time_ms", "kernel_total_ms", "kernel_total_time_ms"],
        )
        idle = self._first_float(
            pools,
            ["idle_ratio", "npu_idle_ratio", "bubble_ratio"],
        )

        if latency is None:
            timeline_us = self._first_float(pools, ["timeline_us", "timeline_length_us"])
            if timeline_us is not None:
                latency = timeline_us / 1000.0

        if latency is None:
            raise RuntimeError("analysis_summary missing total_latency_ms")
        if kernel is None:
            kernel = latency * 0.75
        if idle is None:
            idle = 0.2

        return {
            "total_latency_ms": float(latency),
            "kernel_time_ms": float(kernel),
            "idle_ratio": clamp(float(idle), 0.0, 1.0),
        }

    def _first_float(self, pools: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Optional[float]:
        for pool in pools:
            for key in keys:
                if key in pool:
                    value = safe_float(pool.get(key))
                    if value is not None:
                        return value
        return None

    def _simulate_metrics(self, full_config: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        flat = flatten_layers(full_config)
        latency = 12.5 * (1.0 + self.rng.uniform(-0.30, 0.30))

        stitch_num = flat.get("stitch_function_max_num")
        if stitch_num == 64:
            latency *= 0.72
        elif stitch_num == 32:
            latency *= 0.84
        elif stitch_num == 128:
            latency *= 0.90
        elif isinstance(stitch_num, (int, float)) and stitch_num > 128:
            latency *= 1.05

        if flat.get("enable_multi_data_load") is True:
            latency *= 0.94
        if flat.get("cube_l1_reuse_mode") == 1:
            latency *= 0.92
        if flat.get("enable_split_k") is True:
            k_tile = flat.get("cube_tile_k")
            if isinstance(k_tile, (list, tuple)) and any(isinstance(v, (int, float)) and v >= 128 for v in k_tile):
                latency *= 0.97
            else:
                latency *= 1.03
        if flat.get("vec_nbuffer_mode") in (1, 2):
            latency *= 0.95
        if flat.get("device_sched_mode") == 1:
            latency *= 0.97

        latency += self.rng.uniform(-0.20, 0.20)
        latency = max(3.0, latency)

        kernel = latency * (0.68 + self.rng.uniform(-0.05, 0.05))
        idle = 0.35 + self.rng.uniform(-0.08, 0.08)
        if stitch_num == 64:
            idle -= 0.18
        if flat.get("device_sched_mode") == 1:
            idle -= 0.03
        if flat.get("vec_nbuffer_mode") in (1, 2):
            idle -= 0.02

        return {
            "total_latency_ms": round(latency, 6),
            "kernel_time_ms": round(max(0.1, kernel), 6),
            "idle_ratio": round(clamp(idle, 0.01, 0.95), 6),
        }

    def _build_env_payload(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "timestamp": now_iso(),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "argv": shlex.join(sys.argv),
        }


class ResultRecorder:
    """Persist trial stream and best configuration artifacts."""

    def __init__(self, output_dir: Path, target_metric: str):
        self.output_dir = output_dir
        self.target_metric = target_metric
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "tuning_results.jsonl"
        self.best_path = self.output_dir / "best_config.json"
        self.summary_path = self.output_dir / "autotune_summary.md"
        self._stream = self.results_path.open("w", encoding="utf-8")

    def record_trial(self, outcome: TrialOutcome) -> None:
        """Append one trial record to JSONL."""
        row = {
            "trial_id": outcome.trial_id,
            "layer": outcome.layer,
            "config": outcome.config,
            "metrics": outcome.metrics,
            "timestamp": outcome.timestamp,
            "status": outcome.status,
            "error": outcome.error,
            "env": outcome.env,
        }
        self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stream.flush()

    def update_best(
        self,
        best_config: Dict[str, Dict[str, Any]],
        best_metric: float,
        metrics: Dict[str, float],
        trial_id: int,
        layer_name: str,
    ) -> None:
        """Write best_config.json immediately on improvement."""
        payload = {
            "target_metric": self.target_metric,
            "best_metric": best_metric,
            "metrics": metrics,
            "trial_id": trial_id,
            "layer": layer_name,
            "updated_at": now_iso(),
            "best_config": best_config,
        }
        self.best_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_summary(self, content: str) -> None:
        """Write human-readable markdown summary."""
        self.summary_path.write_text(content, encoding="utf-8")

    def close(self) -> None:
        """Close recorder stream."""
        self._stream.close()


class EarlyStopChecker:
    """Evaluate global and per-layer early stopping conditions."""

    def __init__(self, patience: int, threshold: float, trial_budget: int, time_budget_min: int):
        self.patience = max(1, patience)
        self.threshold = max(0.0, threshold)
        self.trial_budget = max(1, trial_budget)
        self.time_budget_min = max(1, time_budget_min)

    def is_improvement(self, new_value: float, best_value: float) -> bool:
        """Return True if new value improves by threshold."""
        if best_value == float("inf"):
            return True
        return new_value < best_value * (1.0 - self.threshold)

    def check_global(self, total_trials: int, start_time: float) -> Optional[str]:
        """Check global budget/time stopping conditions."""
        if total_trials >= self.trial_budget:
            return "trial budget exceeded"
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min >= float(self.time_budget_min):
            return "time budget exceeded"
        return None

    def check_layer(self, no_improve_count: int) -> Optional[str]:
        """Check layer-level patience stopping condition."""
        if no_improve_count >= self.patience:
            return "no improvement for {} consecutive trials".format(self.patience)
        return None


def load_guidance_summary(
    baseline_dir: Optional[Path],
    analyzer_script: Optional[Path],
    timeout_sec: int,
) -> Dict[str, Any]:
    """Load analysis_summary.json and convert to guidance payload."""
    if baseline_dir is None:
        return {}

    summary_path = find_analysis_summary(baseline_dir)
    if summary_path is None and analyzer_script is not None and analyzer_script.exists():
        cmd = [
            sys.executable,
            str(analyzer_script),
            "--output-dir",
            str(baseline_dir),
            "--report-out",
            str(baseline_dir / "autotune_baseline_report.md"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        except Exception:
            pass
        summary_path = find_analysis_summary(baseline_dir)

    if summary_path is None:
        return {}

    try:
        summary = load_json_file(summary_path)
    except Exception:
        return {}

    labels: List[str] = []
    top_ops: List[str] = []
    if isinstance(summary.get("bottleneck_labels"), list):
        labels = [str(x) for x in summary.get("bottleneck_labels", [])]

    top_ops_raw = summary.get("top_ops")
    if isinstance(top_ops_raw, list):
        for item in top_ops_raw[:8]:
            if isinstance(item, dict) and "name" in item:
                top_ops.append(str(item.get("name")))
            elif isinstance(item, str):
                top_ops.append(item)

    return {
        "source": str(summary_path),
        "bottleneck_labels": labels,
        "top_ops": top_ops,
    }


def metric_of(metrics: Dict[str, float], target: str) -> float:
    """Get target metric value from metrics dict."""
    value = metrics.get(target)
    if value is None:
        return float("inf")
    return float(value)


def run_autotune(args: argparse.Namespace) -> int:
    """Main iterative layered search loop."""
    rng = random.Random(args.seed)

    search_loader = SearchSpaceLoader(Path(args.search_space).resolve())
    search_space = search_loader.load()
    layers = sorted(search_space["layers"], key=lambda item: int(item.get("priority", 999)))

    analyzer_script: Optional[Path] = None
    if args.analyzer_script:
        analyzer_script = Path(args.analyzer_script).resolve()
    elif DEFAULT_ANALYZER_SCRIPT.exists():
        analyzer_script = DEFAULT_ANALYZER_SCRIPT

    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else None
    output_dir = Path(args.output_dir).resolve()

    guidance = {}
    if not args.dry_run:
        guidance = load_guidance_summary(baseline_dir, analyzer_script, args.benchmark_timeout)

    recorder = ResultRecorder(output_dir, args.target_metric)
    generator = CandidateGenerator(search_loader, rng)
    pruning = PruningEngine()
    stopper = EarlyStopChecker(
        patience=args.early_stop_patience,
        threshold=args.improvement_threshold,
        trial_budget=args.budget,
        time_budget_min=args.time_budget,
    )
    runner = BenchmarkRunner(
        benchmark_cmd=args.benchmark_cmd,
        baseline_dir=baseline_dir,
        analyzer_script=analyzer_script,
        dry_run=args.dry_run,
        benchmark_timeout=args.benchmark_timeout,
        target_metric=args.target_metric,
        rng=rng,
        seed=args.seed,
    )

    logger.info("[autotune] layers=%d, target_metric=%s, dry_run=%s", len(layers), args.target_metric, args.dry_run)
    if guidance.get("source"):
        logger.info("[autotune] guidance summary: %s", guidance["source"])

    fixed_best_layers: Dict[str, Dict[str, Any]] = {}
    global_best_metric = float("inf")
    global_best_config: Dict[str, Dict[str, Any]] = {}
    total_trials = 0
    start_time = time.time()
    layer_reports: List[LayerResult] = []
    global_stop_reason: Optional[str] = None

    try:
        for layer in layers:
            layer_name = str(layer.get("name"))
            params = layer.get("params", {})
            if not isinstance(params, dict):
                continue

            space_size = search_loader.estimate_layer_size(layer)
            strategy = generator.select_strategy(space_size)

            remain_budget = max(0, int(args.budget) - total_trials)
            if remain_budget <= 0:
                global_stop_reason = "trial budget exceeded"
                break

            layer_max_trials = int(layer.get("max_trials", remain_budget))
            layer_budget = min(layer_max_trials, remain_budget)
            if args.dry_run:
                layer_budget = min(layer_budget, 3)
            if layer_budget <= 0:
                continue

            logger.info(
                "[layer:%s] priority=%s strategy=%s space=%d budget=%d",
                layer_name,
                layer.get("priority"),
                strategy,
                space_size,
                layer_budget,
            )

            layer_history: List[Dict[str, Any]] = []
            seen_candidates = set()
            layer_best_metric = float("inf")
            layer_best_candidate: Optional[Dict[str, Any]] = None
            no_improve_count = 0
            success_count = 0
            fail_count = 0
            pruned_count = 0
            layer_trial_count = 0
            layer_stop_reason: Optional[str] = None

            def evaluate_candidate(candidate: Dict[str, Any]) -> str:
                nonlocal total_trials
                nonlocal layer_trial_count
                nonlocal layer_best_metric
                nonlocal layer_best_candidate
                nonlocal global_best_metric
                nonlocal global_best_config
                nonlocal no_improve_count
                nonlocal success_count
                nonlocal fail_count
                nonlocal pruned_count

                marker = canonical_json(candidate)
                if marker in seen_candidates:
                    return "continue"
                seen_candidates.add(marker)

                full_config = merge_layer_config(fixed_best_layers, layer_name, candidate)
                flat_cfg = flatten_layers(full_config)
                should_prune, reason = pruning.should_prune(flat_cfg)
                if should_prune:
                    pruned_count += 1
                    if reason:
                        logger.info("  [prune] %s", reason)
                    return "continue"

                global_reason = stopper.check_global(total_trials, start_time)
                if global_reason:
                    return "stop_all"

                total_trials += 1
                layer_trial_count += 1
                trial_id = total_trials

                logger.info("  [trial %03d] layer=%s config=%s", trial_id, layer_name, canonical_json(candidate))
                outcome = runner.run_trial(trial_id, layer_name, full_config, output_dir)
                recorder.record_trial(outcome)

                if outcome.status == "success":
                    success_count += 1
                    value = metric_of(outcome.metrics, args.target_metric)
                    layer_history.append({"config": copy.deepcopy(candidate), "metric": value})
                    if stopper.is_improvement(value, layer_best_metric):
                        layer_best_metric = value
                        layer_best_candidate = copy.deepcopy(candidate)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if stopper.is_improvement(value, global_best_metric):
                        global_best_metric = value
                        global_best_config = copy.deepcopy(full_config)
                        recorder.update_best(
                            best_config=global_best_config,
                            best_metric=global_best_metric,
                            metrics=outcome.metrics,
                            trial_id=trial_id,
                            layer_name=layer_name,
                        )
                        logger.info(
                            "  [best] trial=%d %s=%.6f",
                            trial_id,
                            args.target_metric,
                            global_best_metric,
                        )
                else:
                    fail_count += 1
                    no_improve_count += 1
                    logger.warning("  [failed] trial=%d error=%s", trial_id, outcome.error)

                global_reason = stopper.check_global(total_trials, start_time)
                if global_reason:
                    return "stop_all"
                layer_reason = stopper.check_layer(no_improve_count)
                if layer_reason:
                    return "stop_layer"
                return "continue"

            if strategy == "grid":
                candidates = generator.grid_search(params)
                candidates = generator.apply_guidance(layer_name, candidates, guidance)
                for candidate in candidates:
                    if layer_trial_count >= layer_budget:
                        break
                    action = evaluate_candidate(candidate)
                    if action == "stop_all":
                        global_stop_reason = stopper.check_global(total_trials, start_time)
                        layer_stop_reason = global_stop_reason
                        break
                    if action == "stop_layer":
                        layer_stop_reason = stopper.check_layer(no_improve_count)
                        break

            elif strategy == "random+grid":
                random_budget = max(1, int(layer_budget * 0.7))
                random_candidates = generator.random_search(params, random_budget)
                random_candidates = generator.apply_guidance(layer_name, random_candidates, guidance)

                for candidate in random_candidates:
                    if layer_trial_count >= layer_budget:
                        break
                    action = evaluate_candidate(candidate)
                    if action == "stop_all":
                        global_stop_reason = stopper.check_global(total_trials, start_time)
                        layer_stop_reason = global_stop_reason
                        break
                    if action == "stop_layer":
                        layer_stop_reason = stopper.check_layer(no_improve_count)
                        break

                if layer_stop_reason is None and layer_history and layer_trial_count < layer_budget:
                    top_sorted = sorted(layer_history, key=lambda item: float(item.get("metric", float("inf"))))
                    top_configs = [item["config"] for item in top_sorted[:3] if isinstance(item.get("config"), dict)]
                    refine_budget = layer_budget - layer_trial_count
                    max_cands = max(10, refine_budget * 3)
                    refine_candidates = generator.build_refinement_grid(
                        params, top_configs, max_candidates=max_cands
                    )
                    refine_candidates = generator.apply_guidance(layer_name, refine_candidates, guidance)
                    for candidate in refine_candidates:
                        if layer_trial_count >= layer_budget:
                            break
                        action = evaluate_candidate(candidate)
                        if action == "stop_all":
                            global_stop_reason = stopper.check_global(total_trials, start_time)
                            layer_stop_reason = global_stop_reason
                            break
                        if action == "stop_layer":
                            layer_stop_reason = stopper.check_layer(no_improve_count)
                            break

            else:
                init_count = min(10, layer_budget)
                init_candidates = generator.random_search(params, init_count)
                init_candidates = generator.apply_guidance(layer_name, init_candidates, guidance)

                for candidate in init_candidates:
                    if layer_trial_count >= layer_budget:
                        break
                    action = evaluate_candidate(candidate)
                    if action == "stop_all":
                        global_stop_reason = stopper.check_global(total_trials, start_time)
                        layer_stop_reason = global_stop_reason
                        break
                    if action == "stop_layer":
                        layer_stop_reason = stopper.check_layer(no_improve_count)
                        break

                while layer_stop_reason is None and layer_trial_count < layer_budget:
                    suggestions = generator.bayesian_suggest(params, layer_history, n_trials=1)
                    if not suggestions:
                        break
                    action = evaluate_candidate(suggestions[0])
                    if action == "stop_all":
                        global_stop_reason = stopper.check_global(total_trials, start_time)
                        layer_stop_reason = global_stop_reason
                        break
                    if action == "stop_layer":
                        layer_stop_reason = stopper.check_layer(no_improve_count)
                        break

            if layer_history:
                top_row = min(layer_history, key=lambda item: float(item.get("metric", float("inf"))))
                top_cfg = top_row.get("config")
                if isinstance(top_cfg, dict):
                    layer_best_candidate = copy.deepcopy(top_cfg)
                    layer_best_metric = float(top_row.get("metric", float("inf")))
                    fixed_best_layers[layer_name] = copy.deepcopy(layer_best_candidate)
            elif success_count == 0:
                logger.warning("[layer:%s] warning: all trials failed, skip to next layer", layer_name)

            layer_reports.append(
                LayerResult(
                    layer_name=layer_name,
                    strategy=strategy,
                    space_size=space_size,
                    trial_budget=layer_budget,
                    success_count=success_count,
                    fail_count=fail_count,
                    pruned_count=pruned_count,
                    best_metric=None if layer_best_metric == float("inf") else layer_best_metric,
                    stop_reason=layer_stop_reason,
                )
            )

            if global_stop_reason is not None:
                break

        if not global_best_config:
            global_best_config = copy.deepcopy(fixed_best_layers)

        summary_lines: List[str] = []
        summary_lines.append("# Autotune Summary")
        summary_lines.append("")
        summary_lines.append("- Generated at: `{}`".format(now_iso()))
        summary_lines.append("- Target metric: `{}` (lower is better)".format(args.target_metric))
        summary_lines.append("- Dry run: `{}`".format(args.dry_run))
        summary_lines.append("- Total trials: `{}`".format(total_trials))
        elapsed_min = (time.time() - start_time) / 60.0
        summary_lines.append("- Elapsed time: `{:.2f}` min".format(elapsed_min))
        if global_best_metric != float("inf"):
            summary_lines.append("- Best {}: `{:.6f}`".format(args.target_metric, global_best_metric))
        else:
            summary_lines.append("- Best {}: `N/A`".format(args.target_metric))
        if global_stop_reason:
            summary_lines.append("- Global stop reason: `{}`".format(global_stop_reason))

        summary_lines.append("")
        summary_lines.append("## Layer Results")
        summary_lines.append("")
        summary_lines.append("| Layer | Strategy | Space | Budget | Success | Failed | Pruned | Best Metric | Stop |")
        summary_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in layer_reports:
            best_text = "N/A" if row.best_metric is None else "{:.6f}".format(row.best_metric)
            stop_text = row.stop_reason if row.stop_reason else "-"
            summary_lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    row.layer_name,
                    row.strategy,
                    row.space_size,
                    row.trial_budget,
                    row.success_count,
                    row.fail_count,
                    row.pruned_count,
                    best_text,
                    stop_text,
                )
            )

        summary_lines.append("")
        summary_lines.append("## Final Best Config")
        summary_lines.append("")
        summary_lines.append("```json")
        summary_lines.append(json.dumps(global_best_config, indent=2, ensure_ascii=False))
        summary_lines.append("```")

        recorder.write_summary("\n".join(summary_lines) + "\n")

        if global_best_metric != float("inf"):
            recorder.update_best(
                best_config=global_best_config,
                best_metric=global_best_metric,
                metrics={args.target_metric: global_best_metric},
                trial_id=total_trials,
                layer_name="final",
            )
        else:
            payload = {
                "target_metric": args.target_metric,
                "best_metric": None,
                "metrics": {},
                "trial_id": total_trials,
                "layer": "final",
                "updated_at": now_iso(),
                "best_config": global_best_config,
            }
            recorder.best_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info("[autotune] done. results: %s", recorder.results_path)
        logger.info("[autotune] best: %s", recorder.best_path)
        logger.info("[autotune] summary: %s", recorder.summary_path)
        return 0
    finally:
        recorder.close()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for autotune script."""
    parser = argparse.ArgumentParser(description="PyPTO Performance Autotuner")
    parser.add_argument(
        "--search-space",
        default=str(DEFAULT_SEARCH_SPACE),
        help="Path to search_space.json",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Path to baseline profiler output directory",
    )
    parser.add_argument(
        "--benchmark-cmd",
        default=None,
        help="Command to run benchmark, e.g. 'python3 run.py'",
    )
    parser.add_argument(
        "--analyzer-script",
        default=None,
        help="Path to analyze.py (auto-detect sibling skill by default)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Maximum total trials",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=DEFAULT_TIME_BUDGET_MIN,
        help="Maximum total time budget in minutes",
    )
    parser.add_argument(
        "--target-metric",
        default=DEFAULT_TARGET_METRIC,
        choices=["total_latency_ms", "kernel_time_ms", "idle_ratio"],
        help="Metric to optimize",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate search rounds without NPU",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Consecutive no-improvement trials before stopping layer",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=DEFAULT_IMPROVEMENT_THRESHOLD,
        help="Minimum relative improvement ratio to count as improvement",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=DEFAULT_BENCHMARK_TIMEOUT_SEC,
        help="Benchmark/analyzer timeout in seconds",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return run_autotune(args)


if __name__ == "__main__":
    sys.exit(main())
