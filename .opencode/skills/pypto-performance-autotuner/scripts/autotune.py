#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


@dataclass
class AutotuneSummaryParams:
    """Parameters for building autotune summary report."""

    target_metric: str
    dry_run: bool
    total_trials: int
    start_time: float
    global_best_metric: float
    global_stop_reason: Optional[str]
    layer_reports: List[LayerResult]
    global_best_config: Dict[str, Dict[str, Any]]


@dataclass
class AutotuneFinalizationParams:
    """Parameters for finalizing autotune results."""

    recorder: Any  # ResultRecorder
    target_metric: str
    global_best_config: Dict[str, Dict[str, Any]]
    global_best_metric: float
    total_trials: int
    summary_text: str


@dataclass
class SearchDispatchContext:
    """Shared context for strategy dispatch functions."""

    generator: Any  # CandidateGenerator
    params: Dict[str, Any]
    layer_name: str
    guidance: Dict[str, Any]
    layer_budget: int
    get_layer_trial_count: Any  # Callable
    layer_history: List[Dict[str, Any]]
    get_layer_stop_reason: Any  # Callable
    run_candidates: Any  # Callable
    evaluate_candidate: Any  # Callable
    stopper: Any  # EarlyStopChecker
    get_total_trials: Any  # Callable
    start_time: float
    get_no_improve_count: Any  # Callable


@dataclass
class AutotuneServices:
    """Service objects for autotune search."""

    generator: "CandidateGenerator"
    stopper: "EarlyStopChecker"
    runner: "BenchmarkRunner"
    recorder: "ResultRecorder"


@dataclass
class AutotuneGlobalState:
    """Mutable global state across autotune search layers."""

    total_trials: int
    start_time: float
    global_best_metric: float
    global_best_config: Dict[str, Dict[str, Any]]


@dataclass
class LayerSearchParams:
    """Parameters for searching one layer."""

    layer: Dict[str, Any]
    args: Any  # argparse.Namespace
    guidance: Dict[str, Any]
    fixed_best_layers: Dict[str, Dict[str, Any]]
    output_dir: Any  # Path


@dataclass
class LocalComboContext:
    """Mutable accumulator for local combo generation."""

    seen: set
    all_candidates: List[Dict[str, Any]]
    max_candidates: int


@dataclass
class TrialRetryInput:
    """Parameters for trial execution with retries."""

    trial_id: int
    layer_name: str
    full_config: Dict[str, Dict[str, Any]]
    output_dir: Path
    timestamp: str
    env: Dict[str, Any]


@dataclass
class LayerFinalizationInput:
    """Parameters for building final layer result."""

    layer_name: str
    strategy: str
    space_size: int
    layer_budget: int
    layer_history: List[Dict[str, Any]]
    success_count: int
    fail_count: int
    pruned_count: int
    layer_best_metric: float
    layer_stop_reason: Optional[str]
    fixed_best_layers: Dict[str, Dict[str, Any]]


@dataclass
class TrialSuccessInput:
    """Parameters for processing a successful trial outcome."""

    value: float
    candidate: Dict[str, Any]
    full_config: Dict[str, Dict[str, Any]]
    outcome: TrialOutcome
    trial_id: int
    layer_name: str
    layer_history_local: List[Dict[str, Any]]
    stopper: Any  # EarlyStopChecker
    state: AutotuneGlobalState
    recorder: Any  # ResultRecorder
    args: Any  # argparse.Namespace
    layer_best_metric: float


@dataclass
class AutotuneLoopInput:
    """Parameters for the autotune layer search loop."""

    args: Any  # argparse.Namespace
    layers: List[Dict[str, Any]]
    guidance: Dict[str, Any]
    services: AutotuneServices
    state: AutotuneGlobalState
    output_dir: Any  # Path


@dataclass
class LayerContext:
    """Unpacked local variables for one layer's search iteration."""

    layer_name: str
    layer_params: Dict[str, Any]
    space_size: int
    strategy: str
    layer_budget: int
    generator: Any  # CandidateGenerator
    stopper: Any  # EarlyStopChecker
    runner: Any  # BenchmarkRunner
    recorder: Any  # ResultRecorder
    guidance: Dict[str, Any]
    fixed_best_layers: Dict[str, Dict[str, Any]]
    output_dir: Any  # Path
    args: Any  # argparse.Namespace


@dataclass
class LayerSearchMutableState:
    """Mutable counters and results tracked during one layer's search."""

    layer_history: List[Dict[str, Any]]
    seen_candidates: set
    layer_best_metric: float
    layer_best_candidate: Optional[Dict[str, Any]]
    no_improve_count: int
    success_count: int
    fail_count: int
    pruned_count: int
    layer_trial_count: int
    layer_stop_reason: Optional[str]


def _derive_stitch_from_legacy(params: Dict[str, Any]) -> List[int]:
    """Derive stitch_function_max_num values from legacy initial param."""
    derived = [16, 32, 64, 128]
    legacy_initial = params.get("stitch_function_num_initial")
    if not isinstance(legacy_initial, dict):
        return derived
    try:
        values = SearchSpaceLoader.expand_param_values(legacy_initial)
        clean_values = [int(v) for v in values if isinstance(v, (int, float))]
        clean_values = [v for v in clean_values if 1 <= v <= 256]
        if clean_values:
            derived = sorted(set(clean_values))
    except (ValueError, TypeError, KeyError):
        derived = [16, 32, 64, 128]
    return derived


def _expand_range_values(spec: Dict[str, Any]) -> List[int]:
    """Expand a range-type param spec into a list of integers."""
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


class SearchSpaceLoader:
    """Load and validate search space definitions."""

    def __init__(self, search_space_path: Path):
        """Initialize search space loader with file path."""
        self.search_space_path = search_space_path

    @staticmethod
    def normalize_layer(layer: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one layer and enforce stitch max-num primary knob."""
        name = layer.get("name")
        params = layer.get("params")
        if not isinstance(name, str) or not name:
            raise ValueError("layer requires non-empty name")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"layer '{name}' requires non-empty params")

        normalized = copy.deepcopy(layer)
        normalized.setdefault("priority", 999)
        normalized.setdefault("max_trials", DEFAULT_BUDGET)

        if name == "stitch":
            normalized["params"] = SearchSpaceLoader.normalize_stitch_params(params)

        for param_name, spec in normalized["params"].items():
            if not isinstance(param_name, str) or not param_name:
                raise ValueError(f"invalid param name in layer '{name}'")
            SearchSpaceLoader.expand_param_values(spec)

        return normalized

    @staticmethod
    def normalize_stitch_params(params: Dict[str, Any]) -> Dict[str, Any]:
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
            derived = _derive_stitch_from_legacy(params)
            normalized["stitch_function_max_num"] = {"type": "choice", "values": derived}

        for key, value in params.items():
            if key in deprecated or key == "stitch_function_max_num":
                continue
            normalized[key] = copy.deepcopy(value)

        return normalized

    @staticmethod
    def estimate_layer_size(layer: Dict[str, Any]) -> int:
        """Estimate Cartesian size for one layer."""
        params = layer.get("params", {})
        if not isinstance(params, dict) or not params:
            return 1

        size = 1
        for _, spec in params.items():
            values = SearchSpaceLoader.expand_param_values(spec)
            if not values:
                continue
            size *= len(values)
        return max(size, 1)

    @staticmethod
    def expand_param_values(spec: Dict[str, Any]) -> List[Any]:
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
            return _expand_range_values(spec)

        raise ValueError(f"unsupported param type: {spec_type}")

    def load(self) -> Dict[str, Any]:
        """Load search space JSON and normalize layers."""
        if not self.search_space_path.exists():
            raise FileNotFoundError(f"search space not found: {self.search_space_path}")

        raw = load_json_file(self.search_space_path)
        layers = raw.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("search space requires non-empty 'layers' list")

        normalized_layers: List[Dict[str, Any]] = []
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"layer at index {idx} must be object")
            normalized_layers.append(SearchSpaceLoader.normalize_layer(layer))

        raw["layers"] = normalized_layers
        return raw


def _compute_tpe_weights(
    options: List[Any],
    name: str,
    good: Sequence[Dict[str, Any]],
    bad: Sequence[Dict[str, Any]],
) -> List[float]:
    """Compute TPE frequency weights for one parameter's options."""
    weights: List[float] = []
    for value in options:
        g_cnt = CandidateGenerator.value_count(good, name, value)
        b_cnt = CandidateGenerator.value_count(bad, name, value)
        g_score = float(g_cnt + 1) / float(len(good) + len(options))
        b_score = float(b_cnt + 1) / float(len(bad) + len(options))
        weights.append(max(g_score / b_score, 1e-6))
    return weights


def _generate_local_combos(
    base: Dict[str, Any],
    params: Dict[str, Any],
    loader: "SearchSpaceLoader",
    ctx: LocalComboContext,
) -> bool:
    """Generate local neighbor combos for one base config. Return True if capacity reached."""
    local_options: Dict[str, List[Any]] = {}
    for name, spec in params.items():
        values = loader.expand_param_values(spec)
        local_options[name] = CandidateGenerator.local_neighbors(values, base.get(name))

    names = sorted(local_options.keys())
    if not names:
        return False

    for combo in itertools.product(*[local_options[n] for n in names]):
        candidate: Dict[str, Any] = {}
        for index, param_name in enumerate(names):
            candidate[param_name] = copy.deepcopy(combo[index])
        marker = canonical_json(candidate)
        if marker in ctx.seen:
            continue
        ctx.seen.add(marker)
        ctx.all_candidates.append(candidate)
        if len(ctx.all_candidates) >= ctx.max_candidates:
            return True
    return False


class CandidateGenerator:
    """Generate candidates by grid/random/Bayesian strategies."""

    def __init__(self, loader: SearchSpaceLoader, rng: random.Random):
        """Initialize candidate generator with loader and RNG."""
        self.loader = loader
        self.rng = rng

    @staticmethod
    def value_count(rows: Sequence[Dict[str, Any]], name: str, value: Any) -> int:
        """Count distinct values for a parameter key in search space."""
        marker = canonical_json(value)
        count = 0
        for row in rows:
            cfg = row.get("config")
            if not isinstance(cfg, dict):
                continue
            if canonical_json(cfg.get(name)) == marker:
                count += 1
        return count

    @staticmethod
    def local_neighbors(values: List[Any], center: Any) -> List[Any]:
        """Generate neighbor candidates around a base configuration."""
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

    @staticmethod
    def select_strategy(space_size: int) -> str:
        """Auto-select strategy from search-space size."""
        if space_size < 50:
            return "grid"
        if space_size <= 500:
            return "random+grid"
        return "bayesian"

    @staticmethod
    def _score_stitch(item: Dict[str, Any], labels: List[str]) -> float:
        """Score a stitch-layer candidate based on guidance labels."""
        score = 0.0
        max_num = item.get("stitch_function_max_num")
        has_idle_or_bubble = any("idle" in s or "bubble" in s for s in labels)
        if has_idle_or_bubble:
            if max_num == 64:
                score += 3.0
            elif max_num in (32, 128):
                score += 1.0
        if isinstance(max_num, (int, float)) and float(max_num) > 128:
            score -= 1.0
        return score

    @staticmethod
    def _score_matmul(item: Dict[str, Any], labels: List[str], top_ops: List[str]) -> float:
        """Score a matmul-layer candidate based on guidance."""
        score = 0.0
        if any("matmul" in s or "cube" in s for s in labels + top_ops):
            if item.get("enable_multi_data_load") is True:
                score += 1.5
            if item.get("cube_l1_reuse_mode") == 1:
                score += 1.5
        return score

    @staticmethod
    def _score_vector(item: Dict[str, Any], labels: List[str], top_ops: List[str]) -> float:
        """Score a vector-layer candidate based on guidance."""
        score = 0.0
        if any("vector" in s or "parallel" in s for s in labels + top_ops):
            if item.get("vec_nbuffer_mode") in (1, 2):
                score += 1.2
            sg_scope = item.get("sg_set_scope")
            if isinstance(sg_scope, int) and sg_scope >= 0:
                score += 0.8
        return score

    @staticmethod
    def _score_scheduling(item: Dict[str, Any]) -> float:
        """Score a scheduling-layer candidate."""
        if item.get("device_sched_mode") == 1:
            return 1.0
        return 0.0

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
            candidate = self._bayesian_build_candidate(names, value_cache, good, bad)
            marker = canonical_json(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            suggestions.append(candidate)

        if not suggestions:
            return self.random_search(params, n_trials)
        return suggestions

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

    def apply_guidance(
        self,
        layer_name: str,
        candidates: List[Dict[str, Any]],
        guidance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Order candidates with analyzer guidance scores."""
        if not candidates or not guidance:
            return candidates

        labels = [str(x).lower() for x in guidance.get("bottleneck_labels", []) if isinstance(x, str)]
        top_ops = [str(x).lower() for x in guidance.get("top_ops", []) if isinstance(x, str)]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in candidates:
            score = self._score_candidate(layer_name, item, labels, top_ops)
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
        if max_candidates <= 0 or not base_configs:
            return []

        all_candidates: List[Dict[str, Any]] = []
        seen: set = set()

        for base in base_configs:
            ctx = LocalComboContext(seen=seen, all_candidates=all_candidates, max_candidates=max_candidates)
            reached = _generate_local_combos(
                base, params, self.loader, ctx
            )
            if reached:
                break

        return all_candidates

    def _bayesian_build_candidate(
        self,
        names: List[str],
        value_cache: Dict[str, List[Any]],
        good: Sequence[Dict[str, Any]],
        bad: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build one TPE-weighted candidate from parameter space."""
        candidate: Dict[str, Any] = {}
        for name in names:
            options = value_cache[name]
            weights = _compute_tpe_weights(options, name, good, bad)
            candidate[name] = copy.deepcopy(self._weighted_choice(options, weights))
        return candidate

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

    def _score_candidate(
        self,
        layer_name: str,
        item: Dict[str, Any],
        labels: List[str],
        top_ops: List[str],
    ) -> float:
        """Compute guidance score for a candidate using layer-specific scorer."""
        if layer_name == "stitch":
            return self._score_stitch(item, labels)
        if layer_name == "matmul":
            return self._score_matmul(item, labels, top_ops)
        if layer_name == "vector":
            return self._score_vector(item, labels, top_ops)
        if layer_name == "scheduling":
            return self._score_scheduling(item)
        return 0.0


def _check_prune_cube_reuse(flat: Dict[str, Any]) -> Optional[str]:
    """Check cube L1 reuse mode pruning rule."""
    cube_l1 = flat.get("cube_l1_reuse_mode")
    cube_nb = flat.get("cube_nbuffer_mode", 0)
    if cube_l1 == 1 and cube_nb not in (None, 0):
        return "cube_l1_reuse_mode=1 requires cube_nbuffer_mode=0"
    return None


def _check_prune_split_k(flat: Dict[str, Any]) -> Optional[str]:
    """Check split-K pruning rule."""
    if flat.get("enable_split_k") is not True:
        return None
    k_tile = flat.get("cube_tile_k")
    if PruningEngine.is_small_k_tile(k_tile):
        return "enable_split_k=True but cube_tile_k is too small"
    return None


def _check_prune_pg_partition(flat: Dict[str, Any]) -> Optional[str]:
    """Check partition skip pruning rule."""
    if flat.get("pg_skip_partition") is not True:
        return None
    has_bound = flat.get("pg_upper_bound") is not None
    has_lower = flat.get("pg_lower_bound") is not None
    if has_bound or has_lower:
        return "pg_skip_partition=True conflicts with pg bounds"
    return None


def _check_prune_vec_nbuffer(flat: Dict[str, Any]) -> Optional[str]:
    """Check vec nbuffer mode pruning rule."""
    if flat.get("vec_nbuffer_mode") != 0:
        return None
    setting = flat.get("vec_nbuffer_setting")
    if setting not in (None, {}, []):
        return "vec_nbuffer_mode=0 ignores vec_nbuffer_setting"
    return None


def _check_prune_stitch_max(flat: Dict[str, Any]) -> Optional[str]:
    """Check stitch max num pruning rule."""
    stitch_max = safe_float(flat.get("stitch_function_max_num"))
    if stitch_max is not None and stitch_max > 256:
        return "stitch_function_max_num > 256"
    return None


_PRUNE_CHECKERS: List[Callable[[Dict[str, Any]], Optional[str]]] = [
    _check_prune_cube_reuse,
    _check_prune_split_k,
    _check_prune_pg_partition,
    _check_prune_vec_nbuffer,
    _check_prune_stitch_max,
]


class PruningEngine:
    """Domain-specific pruning rules for invalid/low-value combinations."""

    @staticmethod
    def is_small_k_tile(value: Any) -> bool:
        """Check whether cube_tile_k values are all below the given threshold."""
        if not isinstance(value, (list, tuple)):
            return False
        numeric = [int(x) for x in value if isinstance(x, (int, float))]
        if not numeric:
            return False
        return max(numeric) <= 64

    @staticmethod
    def should_prune(flat_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Return (should_prune, reason)."""
        for checker in _PRUNE_CHECKERS:
            reason = checker(flat_config)
            if reason is not None:
                return True, reason
        return False, None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmark_cmd: Optional[str]
    baseline_dir: Optional[Path]
    analyzer_script: Optional[Path]
    dry_run: bool
    benchmark_timeout: int
    target_metric: str
    rng: random.Random
    seed: int


def _apply_knob_multipliers(flat: Dict[str, Any], latency: float) -> float:
    """Apply all knob-based multipliers to simulated latency."""
    stitch_num = flat.get("stitch_function_max_num")
    stitch_table: Dict[Any, float] = {64: 0.72, 32: 0.84, 128: 0.90}
    if stitch_num in stitch_table:
        latency *= stitch_table[stitch_num]
    elif isinstance(stitch_num, (int, float)) and float(stitch_num) > 128:
        latency *= 1.05

    if flat.get("enable_multi_data_load") is True:
        latency *= 0.94
    if flat.get("cube_l1_reuse_mode") == 1:
        latency *= 0.92
    if flat.get("vec_nbuffer_mode") in (1, 2):
        latency *= 0.95
    if flat.get("device_sched_mode") == 1:
        latency *= 0.97

    if flat.get("enable_split_k") is True:
        k_tile = flat.get("cube_tile_k")
        has_large_k = isinstance(k_tile, (list, tuple)) and any(
            isinstance(v, (int, float)) and v >= 128 for v in k_tile
        )
        latency *= 0.97 if has_large_k else 1.03

    return latency


def _make_trial_outcome(
    inp: TrialRetryInput, metrics: Dict[str, float],
    status: str, error: Optional[str],
) -> TrialOutcome:
    """Create a TrialOutcome with deep-copied config."""
    return TrialOutcome(
        trial_id=inp.trial_id,
        layer=inp.layer_name,
        config=copy.deepcopy(inp.full_config),
        metrics=metrics,
        status=status,
        error=error,
        timestamp=inp.timestamp,
        env=inp.env,
    )


class BenchmarkRunner:
    """Run benchmark + analyzer, then parse metrics for each trial."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner from config."""
        self.benchmark_cmd = config.benchmark_cmd
        self.baseline_dir = config.baseline_dir
        self.analyzer_script = config.analyzer_script
        self.dry_run = config.dry_run
        self.benchmark_timeout = config.benchmark_timeout
        self.target_metric = config.target_metric
        self.rng = config.rng
        self.seed = config.seed

    @staticmethod
    def extract_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract required metrics from analysis_summary.json."""
        pools: List[Dict[str, Any]] = [summary]
        for key in ("metrics", "performance", "summary", "stats"):
            value = summary.get(key)
            if isinstance(value, dict):
                pools.append(value)

        latency = BenchmarkRunner.first_float(
            pools,
            ["total_latency_ms", "latency_ms", "end_to_end_latency_ms", "total_time_ms"],
        )
        kernel = BenchmarkRunner.first_float(
            pools,
            ["kernel_time_ms", "kernel_total_ms", "kernel_total_time_ms"],
        )
        idle = BenchmarkRunner.first_float(
            pools,
            ["idle_ratio", "npu_idle_ratio", "bubble_ratio"],
        )

        if latency is None:
            timeline_us = BenchmarkRunner.first_float(pools, ["timeline_us", "timeline_length_us"])
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

    @staticmethod
    def first_float(pools: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Optional[float]:
        """Extract the first float-like value from a metrics dictionary."""
        for pool in pools:
            for key in keys:
                if key in pool:
                    value = safe_float(pool.get(key))
                    if value is not None:
                        return value
        return None

    @staticmethod
    def _simulate_latency(flat: Dict[str, Any], rng: random.Random) -> float:
        """Compute simulated latency from flattened config."""
        latency = 12.5 * (1.0 + rng.uniform(-0.30, 0.30))
        latency = _apply_knob_multipliers(flat, latency)
        latency += rng.uniform(-0.20, 0.20)
        return max(3.0, latency)

    @staticmethod
    def _simulate_idle(flat: Dict[str, Any], rng: random.Random) -> float:
        """Compute simulated idle ratio from flattened config."""
        idle = 0.35 + rng.uniform(-0.08, 0.08)
        if flat.get("stitch_function_max_num") == 64:
            idle -= 0.18
        if flat.get("device_sched_mode") == 1:
            idle -= 0.03
        if flat.get("vec_nbuffer_mode") in (1, 2):
            idle -= 0.02
        return idle

    def run_trial(
        self,
        trial_id: int,
        layer_name: str,
        full_config: Dict[str, Dict[str, Any]],
        output_dir: Path,
    ) -> TrialOutcome:
        """Execute one trial and return structured outcome."""
        timestamp = now_iso()
        env = self._build_env_payload()
        inp = TrialRetryInput(
            trial_id=trial_id, layer_name=layer_name,
            full_config=full_config, output_dir=output_dir,
            timestamp=timestamp, env=env,
        )

        if self.dry_run:
            metrics = self._simulate_metrics(full_config)
            return _make_trial_outcome(inp, metrics, "success", None)

        return self._run_trial_with_retries(inp)

    def _simulate_metrics(self, full_config: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Simulate benchmark metrics for dry-run mode."""
        flat = flatten_layers(full_config)
        latency = self._simulate_latency(flat, self.rng)
        kernel = latency * (0.68 + self.rng.uniform(-0.05, 0.05))
        idle = self._simulate_idle(flat, self.rng)

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

    def _run_trial_with_retries(
        self,
        inp: TrialRetryInput,
    ) -> TrialOutcome:
        """Execute benchmark with retries and return outcome."""
        full_config = inp.full_config
        output_dir = inp.output_dir
        trial_id = inp.trial_id
        last_error: Optional[str] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._execute_benchmark(full_config)
                summary = self._run_analyzer(output_dir, trial_id)
                metrics = BenchmarkRunner.extract_metrics(summary)
                return _make_trial_outcome(
                    inp, metrics, "success", None
                )
            except (
                RuntimeError, ValueError, KeyError, TypeError, OSError,
                subprocess.SubprocessError,
            ) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    last_error = (
                        f"timeout after {self.benchmark_timeout}s"
                        f" (attempt {attempt}/{MAX_RETRIES})"
                    )
                else:
                    last_error = f"{exc!s} (attempt {attempt}/{MAX_RETRIES})"

            if attempt < MAX_RETRIES:
                time.sleep(1.0)

        return _make_trial_outcome(
            inp, {}, "failed", last_error
        )

    def _execute_benchmark(self, full_config: Dict[str, Dict[str, Any]]) -> None:
        if not self.benchmark_cmd:
            raise RuntimeError("benchmark command is required when not in dry-run")

        env = os.environ.copy()
        env["PYPTO_AUTOTUNE_CONFIG"] = canonical_json(full_config)

        completed = subprocess.run(
            shlex.split(self.benchmark_cmd),
            shell=False,
            capture_output=True,
            text=True,
            timeout=self.benchmark_timeout,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip().splitlines()
            stderr_tail = stderr_text[-3:] if stderr_text else []
            detail = " | ".join(stderr_tail)
            raise RuntimeError(f"benchmark crash (exit={completed.returncode}): {detail}")

    def _run_analyzer(self, output_dir: Path, trial_id: int) -> Dict[str, Any]:
        if self.analyzer_script is None or not self.analyzer_script.exists():
            raise RuntimeError("analyzer script not found")

        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"analysis_trial_{trial_id}.md"

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
            check=False,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip().splitlines()
            stderr_tail = stderr_text[-3:] if stderr_text else []
            detail = " | ".join(stderr_tail)
            raise RuntimeError(f"analyzer failed (exit={completed.returncode}): {detail}")

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


class ResultRecorder:
    """Persist trial stream and best configuration artifacts."""

    def __init__(self, output_dir: Path, target_metric: str):
        """Initialize result recorder with output directory."""
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
        """Initialize early stop checker with stopping criteria."""
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
            return f"no improvement for {self.patience} consecutive trials"
        return None


def _invoke_analyzer_for_summary(
    baseline_dir: Path,
    analyzer_script: Path,
    timeout_sec: int,
) -> None:
    """Run analyzer to generate baseline summary file."""
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
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        pass


def _parse_guidance_from_summary(summary: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Extract bottleneck labels and top ops from analysis summary."""
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
        "source": source,
        "bottleneck_labels": labels,
        "top_ops": top_ops,
    }


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
        _invoke_analyzer_for_summary(baseline_dir, analyzer_script, timeout_sec)
        summary_path = find_analysis_summary(baseline_dir)

    if summary_path is None:
        return {}

    try:
        summary = load_json_file(summary_path)
    except (OSError, ValueError, TypeError):
        return {}

    return _parse_guidance_from_summary(summary, str(summary_path))


def metric_of(metrics: Dict[str, float], target: str) -> float:
    """Get target metric value from metrics dict."""
    value = metrics.get(target)
    if value is None:
        return float("inf")
    return float(value)


def _build_autotune_summary(params: AutotuneSummaryParams) -> str:
    """Build the markdown summary for an autotune run."""
    lines: List[str] = []
    lines.append("# Autotune Summary")
    lines.append("")
    lines.append(f"- Generated at: `{now_iso()}`")
    lines.append(f"- Target metric: `{params.target_metric}` (lower is better)")
    lines.append(f"- Dry run: `{params.dry_run}`")
    lines.append(f"- Total trials: `{params.total_trials}`")
    elapsed_min = (time.time() - params.start_time) / 60.0
    lines.append(f"- Elapsed time: `{elapsed_min:.2f}` min")
    if params.global_best_metric != float("inf"):
        lines.append(f"- Best {params.target_metric}: `{params.global_best_metric:.6f}`")
    else:
        lines.append(f"- Best {params.target_metric}: `N/A`")
    if params.global_stop_reason:
        lines.append(f"- Global stop reason: `{params.global_stop_reason}`")

    lines.append("")
    lines.append("## Layer Results")
    lines.append("")
    lines.append("| Layer | Strategy | Space | Budget | Success | Failed | Pruned | Best Metric | Stop |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in params.layer_reports:
        best_text = "N/A" if row.best_metric is None else f"{row.best_metric:.6f}"
        stop_text = row.stop_reason if row.stop_reason else "-"
        lines.append(
            f"| {row.layer_name} | {row.strategy} | {row.space_size} | {row.trial_budget} | "
            f"{row.success_count} | {row.fail_count} | {row.pruned_count} | {best_text} | {stop_text} |"
        )

    lines.append("")
    lines.append("## Final Best Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(params.global_best_config, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines) + "\n"


def _finalize_autotune_results(params: AutotuneFinalizationParams) -> None:
    """Write final summary and best-config files."""
    params.recorder.write_summary(params.summary_text)
    if params.global_best_metric != float("inf"):
        params.recorder.update_best(
            best_config=params.global_best_config,
            best_metric=params.global_best_metric,
            metrics={params.target_metric: params.global_best_metric},
            trial_id=params.total_trials,
            layer_name="final",
        )
    else:
        payload = {
            "target_metric": params.target_metric,
            "best_metric": None,
            "metrics": {},
            "trial_id": params.total_trials,
            "layer": "final",
            "updated_at": now_iso(),
            "best_config": params.global_best_config,
        }
        params.recorder.best_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _dispatch_random_grid(ctx: SearchDispatchContext) -> Optional[str]:
    """Run random+grid refinement strategy."""
    random_budget = max(1, int(ctx.layer_budget * 0.7))
    random_candidates = ctx.generator.random_search(ctx.params, random_budget)
    random_candidates = ctx.generator.apply_guidance(
        ctx.layer_name, random_candidates, ctx.guidance
    )
    global_stop_reason = ctx.run_candidates(random_candidates)

    can_refine = global_stop_reason is None and ctx.get_layer_stop_reason() is None
    has_history = bool(ctx.layer_history)
    has_budget = ctx.get_layer_trial_count() < ctx.layer_budget
    if not (can_refine and has_history and has_budget):
        return global_stop_reason

    top_sorted = sorted(
        ctx.layer_history,
        key=lambda item: float(item.get("metric", float("inf"))),
    )
    top_configs = [
        item["config"]
        for item in top_sorted[:3]
        if isinstance(item.get("config"), dict)
    ]
    refine_budget = ctx.layer_budget - ctx.get_layer_trial_count()
    max_cands = max(10, refine_budget * 3)
    refine_candidates = ctx.generator.build_refinement_grid(
        ctx.params, top_configs, max_candidates=max_cands
    )
    refine_candidates = ctx.generator.apply_guidance(
        ctx.layer_name, refine_candidates, ctx.guidance
    )
    return ctx.run_candidates(refine_candidates)


def _dispatch_strategy(strategy: str, ctx: SearchDispatchContext) -> Optional[str]:
    """Dispatch strategy-specific search logic and return global stop reason."""
    if strategy == "grid":
        candidates = ctx.generator.grid_search(ctx.params)
        candidates = ctx.generator.apply_guidance(ctx.layer_name, candidates, ctx.guidance)
        return ctx.run_candidates(candidates)
    if strategy == "random+grid":
        return _dispatch_random_grid(ctx)
    return _dispatch_bayesian(ctx)


def _dispatch_bayesian(ctx: SearchDispatchContext) -> Optional[str]:
    """Run bayesian search strategy with initial random seed."""
    init_count = min(10, ctx.layer_budget)
    init_candidates = ctx.generator.random_search(ctx.params, init_count)
    init_candidates = ctx.generator.apply_guidance(ctx.layer_name, init_candidates, ctx.guidance)
    global_stop_reason = ctx.run_candidates(init_candidates)
    should_continue = global_stop_reason is None and ctx.get_layer_stop_reason() is None
    while should_continue and ctx.get_layer_trial_count() < ctx.layer_budget:
        suggestions = ctx.generator.bayesian_suggest(ctx.params, ctx.layer_history, n_trials=1)
        if not suggestions:
            break
        action = ctx.evaluate_candidate(suggestions[0])
        if action == "stop_all":
            global_stop_reason = ctx.stopper.check_global(ctx.get_total_trials(), ctx.start_time)
            break
        if action == "stop_layer":
            break
    return global_stop_reason


def _init_layer_search(
    params: LayerSearchParams,
    services: AutotuneServices,
    state: AutotuneGlobalState,
) -> Optional[Tuple[LayerResult, Optional[str]]]:
    """Validate layer and compute budget. Return early result if layer should be skipped."""
    layer = params.layer
    args = params.args

    layer_name = str(layer.get("name"))
    layer_params = layer.get("params", {})
    if not isinstance(layer_params, dict):
        return LayerResult(layer_name, "grid", 0, 0, 0, 0, 0, None, "invalid params"), None

    space_size = SearchSpaceLoader.estimate_layer_size(layer)
    strategy = services.generator.select_strategy(space_size)

    remain_budget = max(0, int(args.budget) - state.total_trials)
    if remain_budget <= 0:
        row = LayerResult(layer_name, strategy, space_size, 0, 0, 0, 0, None, "trial budget exceeded")
        return row, "trial budget exceeded"

    layer_max_trials = int(layer.get("max_trials", remain_budget))
    layer_budget = min(layer_max_trials, remain_budget)
    if args.dry_run:
        layer_budget = min(layer_budget, 3)
    if layer_budget <= 0:
        row = LayerResult(layer_name, strategy, space_size, layer_budget, 0, 0, 0, None, "budget=0")
        return row, None

    return None


def _finalize_layer_result(inp: LayerFinalizationInput) -> LayerResult:
    """Build final LayerResult and update fixed_best_layers."""
    best_metric = inp.layer_best_metric
    if inp.layer_history:
        top_row = min(inp.layer_history, key=lambda item: float(item.get("metric", float("inf"))))
        top_cfg = top_row.get("config")
        if isinstance(top_cfg, dict):
            best_metric = float(top_row.get("metric", float("inf")))
            inp.fixed_best_layers[inp.layer_name] = copy.deepcopy(top_cfg)
    elif inp.success_count == 0:
        logger.warning("[layer:%s] warning: all trials failed, skip to next layer", inp.layer_name)

    return LayerResult(
        layer_name=inp.layer_name,
        strategy=inp.strategy,
        space_size=inp.space_size,
        trial_budget=inp.layer_budget,
        success_count=inp.success_count,
        fail_count=inp.fail_count,
        pruned_count=inp.pruned_count,
        best_metric=None if best_metric == float("inf") else best_metric,
        stop_reason=inp.layer_stop_reason,
    )


def _prepare_layer_context(
    params: LayerSearchParams,
    services: AutotuneServices,
    state: AutotuneGlobalState,
) -> LayerContext:
    """Unpack layer search parameters into a LayerContext for the search loop."""
    layer = params.layer
    args = params.args
    layer_name = str(layer.get("name"))
    layer_params = layer.get("params", {})
    space_size = SearchSpaceLoader.estimate_layer_size(layer)
    strategy = services.generator.select_strategy(space_size)

    remain_budget = max(0, int(args.budget) - state.total_trials)
    layer_max_trials = int(layer.get("max_trials", remain_budget))
    layer_budget = min(layer_max_trials, remain_budget)
    if args.dry_run:
        layer_budget = min(layer_budget, 3)

    logger.info(
        "[layer:%s] priority=%s strategy=%s space=%d budget=%d",
        layer_name, layer.get("priority"), strategy, space_size, layer_budget,
    )

    return LayerContext(
        layer_name=layer_name, layer_params=layer_params,
        space_size=space_size, strategy=strategy, layer_budget=layer_budget,
        generator=services.generator, stopper=services.stopper,
        runner=services.runner, recorder=services.recorder,
        guidance=params.guidance,
        fixed_best_layers=params.fixed_best_layers,
        output_dir=params.output_dir, args=args,
    )


def _check_candidate_eligible(
    candidate: Dict[str, Any],
    seen_candidates: set,
    fixed_best_layers: Dict[str, Dict[str, Any]],
    layer_name: str,
) -> Tuple[bool, int]:
    """Check dedup and pruning for a candidate. Return (eligible, pruned_delta)."""
    marker = canonical_json(candidate)
    if marker in seen_candidates:
        return False, 0
    seen_candidates.add(marker)

    full_config = merge_layer_config(fixed_best_layers, layer_name, candidate)
    flat_cfg = flatten_layers(full_config)
    do_prune, reason = PruningEngine.should_prune(flat_cfg)
    if do_prune:
        if reason:
            logger.info("  [prune] %s", reason)
        return False, 1

    return True, 0


def _handle_successful_trial(inp: TrialSuccessInput) -> Tuple[float, Optional[Dict[str, Any]], int]:
    """Process a successful trial outcome. Return (new_best_metric, new_best_candidate, no_improve_delta)."""
    inp.layer_history_local.append({"config": copy.deepcopy(inp.candidate), "metric": inp.value})
    new_best = inp.layer_best_metric
    new_candidate: Optional[Dict[str, Any]] = None
    no_improve_delta = 1

    if inp.stopper.is_improvement(inp.value, inp.layer_best_metric):
        new_best = inp.value
        new_candidate = copy.deepcopy(inp.candidate)
        no_improve_delta = 0

    if inp.stopper.is_improvement(inp.value, inp.state.global_best_metric):
        inp.state.global_best_metric = inp.value
        inp.state.global_best_config = copy.deepcopy(inp.full_config)
        inp.recorder.update_best(
            best_config=inp.state.global_best_config,
            best_metric=inp.state.global_best_metric,
            metrics=inp.outcome.metrics,
            trial_id=inp.trial_id,
            layer_name=inp.layer_name,
        )
        logger.info(
            "  [best] trial=%d %s=%.6f",
            inp.trial_id,
            inp.args.target_metric,
            inp.state.global_best_metric,
        )

    return new_best, new_candidate, no_improve_delta


def _evaluate_one_candidate(
    candidate: Dict[str, Any],
    ctx: LayerContext,
    ms: LayerSearchMutableState,
    state: AutotuneGlobalState,
) -> str:
    """Evaluate one candidate: dedup, prune, run trial, update mutable state. Return action."""
    eligible, pruned_delta = _check_candidate_eligible(
        candidate, ms.seen_candidates, ctx.fixed_best_layers, ctx.layer_name,
    )
    if not eligible:
        ms.pruned_count += pruned_delta
        return "continue"
    global_reason = ctx.stopper.check_global(state.total_trials, state.start_time)
    if global_reason:
        return "stop_all"
    state.total_trials += 1
    ms.layer_trial_count += 1
    trial_id = state.total_trials
    full_config = merge_layer_config(ctx.fixed_best_layers, ctx.layer_name, candidate)
    logger.info("  [trial %03d] layer=%s config=%s", trial_id, ctx.layer_name, canonical_json(candidate))
    outcome = ctx.runner.run_trial(trial_id, ctx.layer_name, full_config, ctx.output_dir)
    ctx.recorder.record_trial(outcome)
    if outcome.status == "success":
        ms.success_count += 1
        value = metric_of(outcome.metrics, ctx.args.target_metric)
        new_best, new_cand, delta = _handle_successful_trial(TrialSuccessInput(
            value=value, candidate=candidate, full_config=full_config,
            outcome=outcome, trial_id=trial_id, layer_name=ctx.layer_name,
            layer_history_local=ms.layer_history, stopper=ctx.stopper,
            state=state, recorder=ctx.recorder, args=ctx.args,
            layer_best_metric=ms.layer_best_metric,
        ))
        if delta == 0:
            ms.layer_best_metric = new_best
            ms.layer_best_candidate = new_cand
            ms.no_improve_count = 0
        else:
            ms.no_improve_count += 1
    else:
        ms.fail_count += 1
        ms.no_improve_count += 1
        logger.warning("  [failed] trial=%d error=%s", trial_id, outcome.error)
    global_reason = ctx.stopper.check_global(state.total_trials, state.start_time)
    if global_reason:
        return "stop_all"
    layer_reason = ctx.stopper.check_layer(ms.no_improve_count)
    if layer_reason:
        return "stop_layer"
    return "continue"


def _run_candidate_batch(
    candidates: List[Dict[str, Any]],
    ctx: LayerContext,
    ms: LayerSearchMutableState,
    state: AutotuneGlobalState,
) -> Optional[str]:
    """Run evaluate loop over candidates and return global stop reason."""
    for candidate in candidates:
        if ms.layer_trial_count >= ctx.layer_budget:
            break
        action = _evaluate_one_candidate(candidate, ctx, ms, state)
        if action == "stop_all":
            return ctx.stopper.check_global(state.total_trials, state.start_time)
        if action == "stop_layer":
            ms.layer_stop_reason = ctx.stopper.check_layer(ms.no_improve_count)
            return None
    return None


def _search_one_layer(
    params: LayerSearchParams,
    services: AutotuneServices,
    state: AutotuneGlobalState,
) -> Tuple[LayerResult, Optional[str]]:
    """Run autotune search for one layer and return layer result and global stop reason."""
    early = _init_layer_search(params, services, state)
    if early is not None:
        return early

    ctx = _prepare_layer_context(params, services, state)
    ms = LayerSearchMutableState(
        layer_history=[], seen_candidates=set(),
        layer_best_metric=float("inf"), layer_best_candidate=None,
        no_improve_count=0, success_count=0, fail_count=0,
        pruned_count=0, layer_trial_count=0, layer_stop_reason=None,
    )

    def eval_fn(candidate: Dict[str, Any]) -> str:
        return _evaluate_one_candidate(candidate, ctx, ms, state)

    def batch_fn(candidates: List[Dict[str, Any]]) -> Optional[str]:
        return _run_candidate_batch(candidates, ctx, ms, state)

    global_stop_reason = _dispatch_strategy(
        ctx.strategy,
        SearchDispatchContext(
            generator=ctx.generator, params=ctx.layer_params,
            layer_name=ctx.layer_name, guidance=ctx.guidance,
            layer_budget=ctx.layer_budget,
            get_layer_trial_count=lambda: ms.layer_trial_count,
            layer_history=ms.layer_history,
            get_layer_stop_reason=lambda: ms.layer_stop_reason,
            run_candidates=batch_fn,
            evaluate_candidate=eval_fn,
            stopper=ctx.stopper,
            get_total_trials=lambda: state.total_trials,
            start_time=state.start_time,
            get_no_improve_count=lambda: ms.no_improve_count,
        ),
    )

    row = _finalize_layer_result(LayerFinalizationInput(
        layer_name=ctx.layer_name, strategy=ctx.strategy,
        space_size=ctx.space_size,
        layer_budget=ctx.layer_budget, layer_history=ms.layer_history,
        success_count=ms.success_count, fail_count=ms.fail_count,
        pruned_count=ms.pruned_count, layer_best_metric=ms.layer_best_metric,
        layer_stop_reason=ms.layer_stop_reason,
        fixed_best_layers=ctx.fixed_best_layers,
    ))
    return row, global_stop_reason


def _init_autotune_services(
    args: argparse.Namespace,
) -> Tuple[SearchSpaceLoader, List[Dict[str, Any]], Dict[str, Any], AutotuneServices, Path]:
    """Initialize search loader, layers, guidance, and services for autotune."""
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

    guidance: Dict[str, Any] = {}
    if not args.dry_run:
        guidance = load_guidance_summary(baseline_dir, analyzer_script, args.benchmark_timeout)

    recorder = ResultRecorder(output_dir, args.target_metric)
    generator = CandidateGenerator(search_loader, rng)
    stopper = EarlyStopChecker(
        patience=args.early_stop_patience,
        threshold=args.improvement_threshold,
        trial_budget=args.budget,
        time_budget_min=args.time_budget,
    )
    runner = BenchmarkRunner(
        BenchmarkConfig(
            benchmark_cmd=args.benchmark_cmd,
            baseline_dir=baseline_dir,
            analyzer_script=analyzer_script,
            dry_run=args.dry_run,
            benchmark_timeout=args.benchmark_timeout,
            target_metric=args.target_metric,
            rng=rng,
            seed=args.seed,
        )
    )

    services = AutotuneServices(generator=generator, stopper=stopper, runner=runner, recorder=recorder)
    return search_loader, layers, guidance, services, output_dir


def _execute_autotune_loop(inp: AutotuneLoopInput) -> Tuple[List[LayerResult], Optional[str]]:
    """Execute the layer search loop and return reports and stop reason."""
    layer_reports: List[LayerResult] = []
    global_stop_reason: Optional[str] = None
    fixed_best_layers: Dict[str, Dict[str, Any]] = {}

    for layer in inp.layers:
        row, layer_stop = _search_one_layer(
            params=LayerSearchParams(
                layer=layer,
                args=inp.args,
                guidance=inp.guidance,
                fixed_best_layers=fixed_best_layers,
                output_dir=inp.output_dir,
            ),
            services=inp.services,
            state=inp.state,
        )
        layer_reports.append(row)
        if layer_stop is not None:
            global_stop_reason = layer_stop
            break

    if not inp.state.global_best_config:
        inp.state.global_best_config = copy.deepcopy(fixed_best_layers)

    return layer_reports, global_stop_reason


def run_autotune(args: argparse.Namespace) -> int:
    """Main iterative layered search loop."""
    _, layers, guidance, services, output_dir = _init_autotune_services(args)

    logger.info("[autotune] layers=%d, target_metric=%s, dry_run=%s", len(layers), args.target_metric, args.dry_run)
    if guidance.get("source"):
        logger.info("[autotune] guidance summary: %s", guidance.get("source", ""))

    state = AutotuneGlobalState(
        total_trials=0,
        start_time=time.time(),
        global_best_metric=float("inf"),
        global_best_config={},
    )

    try:
        layer_reports, global_stop_reason = _execute_autotune_loop(AutotuneLoopInput(
            args=args, layers=layers, guidance=guidance,
            services=services, state=state, output_dir=output_dir,
        ))

        summary_text = _build_autotune_summary(
            AutotuneSummaryParams(
                target_metric=args.target_metric,
                dry_run=args.dry_run,
                total_trials=state.total_trials,
                start_time=state.start_time,
                global_best_metric=state.global_best_metric,
                global_stop_reason=global_stop_reason,
                layer_reports=layer_reports,
                global_best_config=state.global_best_config,
            )
        )
        _finalize_autotune_results(
            AutotuneFinalizationParams(
                recorder=services.recorder,
                target_metric=args.target_metric,
                global_best_config=state.global_best_config,
                global_best_metric=state.global_best_metric,
                total_trials=state.total_trials,
                summary_text=summary_text,
            )
        )

        logger.info("[autotune] done. results: %s", services.recorder.results_path)
        logger.info("[autotune] best: %s", services.recorder.best_path)
        logger.info("[autotune] summary: %s", services.recorder.summary_path)
        return 0
    finally:
        services.recorder.close()


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add input/output related arguments to parser."""
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
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for results",
    )


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    """Add budget and stopping related arguments to parser."""
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


def _add_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add tuning control arguments to parser."""
    parser.add_argument(
        "--target-metric",
        default=DEFAULT_TARGET_METRIC,
        choices=["total_latency_ms", "kernel_time_ms", "idle_ratio"],
        help="Metric to optimize",
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


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for autotune script."""
    parser = argparse.ArgumentParser(description="PyPTO Performance Autotuner")
    _add_io_arguments(parser)
    _add_budget_arguments(parser)
    _add_tuning_arguments(parser)
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return run_autotune(args)


if __name__ == "__main__":
    sys.exit(main())
