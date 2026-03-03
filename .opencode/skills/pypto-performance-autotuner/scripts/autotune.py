#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPTO Performance Autotuner - NPU-only Configuration Search

Searches optimal configurations by toggling tile/pass/runtime knobs with proper
measurement hygiene (warmup, NPU synchronize, median aggregation, regression tolerance).

Usage:
    python3 autotune.py --dry-run
    python3 autotune.py --bench preset:softmax_npu --trials 30 --warmup 10 --measure 30

IMPORTANT: All tuning MUST use NPU mode. SIM output cannot be used as basis for best_config.
"""
import argparse
import json
import os
import sys
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "references"

# Default paths
DEFAULT_PYPTO_REPO = "/workspace/code/pypto"
DEFAULT_SPACE_FILE = str(REFERENCES_DIR / "space_softmax_npu.json")
DEFAULT_OUT_FILE = "/workspace/code/.sisyphus/evidence/autotune-results.json"

# Best update threshold (2% improvement)
BEST_UPDATE_THRESHOLD = 0.98
# Early stop after N consecutive trials without improvement
EARLY_STOP_COUNT = 5


def validate_space_schema(space: Dict[str, Any]) -> List[str]:
    """Validate search space JSON schema."""
    errors = []
    
    required_sections = ["stage_a", "stage_b", "stage_c"]
    for section in required_sections:
        if section not in space:
            errors.append(f"Missing required section: {section}")
    
    # Validate device_sched_mode range
    if "stage_c" in space and "runtime_options" in space["stage_c"]:
        rt_opts = space["stage_c"]["runtime_options"]
        if "device_sched_mode" in rt_opts:
            for val in rt_opts["device_sched_mode"]:
                if val not in [0, 1, 2, 3]:
                    errors.append(f"Invalid device_sched_mode value: {val}. Must be 0/1/2/3")
    
    # Check for deprecated parameters
    deprecated = ["stitch_function_inner_memory", "stitch_function_outcast_memory", "stitch_function_num_initial"]
    for section in ["stage_b", "stage_c"]:
        if section in space:
            for key in deprecated:
                if key in str(space[section]):
                    errors.append(f"Deprecated parameter found: {key}. Use stitch_function_max_num instead")
    
    return errors


def load_search_space(space_path: str) -> Dict[str, Any]:
    """Load and validate search space JSON."""
    with open(space_path, "r", encoding="utf-8") as f:
        space = json.load(f)
    
    errors = validate_space_schema(space)
    if errors:
        print("ERROR: Search space validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    
    return space


def get_reproducibility_info() -> Dict[str, Any]:
    """Collect reproducibility information."""
    import platform
    
    pypto_commit = "unknown"
    pypto_repo = Path("/workspace/code/pypto")
    if pypto_repo.exists() and (pypto_repo / ".git").exists():
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(pypto_repo),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pypto_commit = result.stdout.strip()[:12]
        except Exception:
            pass
    
    return {
        "pypto_commit": pypto_commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ascend_env_exists": bool(os.environ.get("ASCEND_HOME_PATH")),
        "device_id": os.environ.get("TILE_FWK_DEVICE_ID", "not_set"),
        "timestamp": datetime.now().isoformat()
    }


def generate_candidates(space: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    """Generate all candidate configurations for a stage."""
    candidates = []
    stage_data = space.get(stage, {})
    
    if stage == "stage_a":
        # Tile shapes - from space
        tile_configs = stage_data.get("vec_tile_shapes", [[64, 512]])
        for cfg in tile_configs:
            candidates.append({"vec_tile_shapes": cfg})
    
    elif stage == "stage_b":
        # Pass options - cartesian product
        pass_opts = stage_data.get("pass_options", {})
        if not pass_opts:
            return [{}]
        
        # Generate simple candidates (one param at a time for now)
        for key, values in pass_opts.items():
            for val in values:
                candidates.append({key: val})
    
    elif stage == "stage_c":
        # Runtime options - cartesian product
        rt_opts = stage_data.get("runtime_options", {})
        if not rt_opts:
            return [{}]
        
        for key, values in rt_opts.items():
            for val in values:
                candidates.append({key: val})
    
    return candidates if candidates else [{}]


def apply_config(config: Dict[str, Any], stage: str) -> None:
    """Apply configuration to PyPTO (placeholder for actual implementation)."""
    # This would be implemented with actual PyPTO calls
    # For dry-run, just print the config
    print(f"  Applying {stage} config: {config}")


def measure_latency(
    bench_fn,
    config: Dict[str, Any],
    warmup: int,
    measure: int,
    dry_run: bool = False
) -> Optional[float]:
    """
    Measure latency with proper NPU synchronization.
    
    Protocol:
    1. Warmup iterations (discarded)
    2. Measure iterations with NPU synchronize before/after each
    3. IQR 1.5x outlier removal
    4. Return median
    
    Returns median latency in milliseconds, or None if dry_run.
    """
    if dry_run:
        return None
    
    import torch_npu
    
    # Warmup
    for _ in range(warmup):
        bench_fn(**config)
        torch_npu.npu.synchronize()
    
    # Measure
    times_ms = []
    for _ in range(measure):
        torch_npu.npu.synchronize()
        t0 = time.perf_counter()
        bench_fn(**config)
        torch_npu.npu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    
    # IQR outlier removal
    if len(times_ms) < 4:
        return sorted(times_ms)[len(times_ms) // 2]
    
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    q1 = sorted_times[n // 4]
    q3 = sorted_times[3 * n // 4]
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered = [t for t in sorted_times if lower_bound <= t <= upper_bound]
    
    if not filtered:
        return sorted_times[n // 2]
    
    return sorted(filtered)[len(filtered) // 2]


def run_autotune(
    space: Dict[str, Any],
    bench_fn,
    trials: int,
    warmup: int,
    measure: int,
    dry_run: bool
) -> Dict[str, Any]:
    """Run autotuning search across all stages."""
    results = {
        "best_config": {
            "stage_a": {},
            "stage_b": {},
            "stage_c": {}
        },
        "best_latency_ms": None,
        "trials": [],
        "reproducibility": get_reproducibility_info()
    }
    
    best_latency = float('inf')
    no_improvement_count = 0
    
    stages = ["stage_a", "stage_b", "stage_c"]
    current_best_config = {}
    
    for stage in stages:
        print(f"\n=== Searching {stage} ===")
        candidates = generate_candidates(space, stage)
        stage_best_latency = float('inf')
        stage_best_config = {}
        
        for i, candidate in enumerate(candidates[:trials]):
            print(f"\nTrial {i+1}/{min(len(candidates), trials)}: {candidate}")
            
            # Combine with previous best configs
            full_config = {**current_best_config, **candidate}
            
            if dry_run:
                latency_ms = None
                print(f"  [DRY-RUN] Would measure with config: {full_config}")
            else:
                apply_config(candidate, stage)
                latency_ms = measure_latency(bench_fn, full_config, warmup, measure, dry_run)
                print(f"  Latency: {latency_ms:.3f} ms" if latency_ms else "  Latency: N/A")
            
            trial_record = {
                "stage": stage,
                "config": candidate,
                "full_config": full_config,
                "metric": {"median_latency_ms": latency_ms},
                "timestamp": datetime.now().isoformat()
            }
            results["trials"].append(trial_record)
            
            # Update best if improvement > 2%
            if latency_ms is not None:
                if latency_ms < best_latency * BEST_UPDATE_THRESHOLD:
                    best_latency = latency_ms
                    current_best_config = copy.deepcopy(full_config)
                    results["best_config"][stage] = copy.deepcopy(candidate)
                    stage_best_latency = latency_ms
                    stage_best_config = copy.deepcopy(candidate)
                    no_improvement_count = 0
                    print(f"  *** NEW BEST: {latency_ms:.3f} ms ***")
                else:
                    no_improvement_count += 1
            
            # Early stop check
            if no_improvement_count >= EARLY_STOP_COUNT:
                print(f"\nEarly stopping after {EARLY_STOP_COUNT} trials without improvement")
                break
        
        # Update current best for next stage
        if stage_best_config:
            current_best_config.update(stage_best_config)
    
    results["best_latency_ms"] = best_latency if best_latency != float('inf') else None
    
    return results


def create_default_space_file(path: str) -> None:
    """Create default search space file."""
    default_space = {
        "preset": "softmax_npu",
        "description": "Search space for softmax operator on NPU",
        "stage_a": {
            "vec_tile_shapes": [
                [64, 512],
                [32, 256],
                [128, 1024]
            ]
        },
        "stage_b": {
            "pass_options": {
                "pg_lower_bound": [128, 256, 512, 1024],
                "pg_parallel_lower_bound": [8, 16, 20, 24, 32],
                "vec_nbuffer_mode": [0, 1, 2],
                "cube_l1_reuse_mode": [0, 1]
            }
        },
        "stage_c": {
            "runtime_options": {
                "device_sched_mode": [0, 1, 2, 3],
                "stitch_function_max_num": [16, 32, 64, 128]
            }
        }
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_space, f, indent=2)
    
    print(f"Created default space file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Performance Autotuner - NPU-only Configuration Search"
    )
    parser.add_argument(
        "--pypto-repo",
        default=DEFAULT_PYPTO_REPO,
        help="Path to PyPTO repository"
    )
    parser.add_argument(
        "--bench",
        default="preset:softmax_npu",
        help="Benchmark to run (preset:softmax_npu)"
    )
    parser.add_argument(
        "--space",
        default=DEFAULT_SPACE_FILE,
        help="Path to search space JSON file"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Maximum trials per stage"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per trial"
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=30,
        help="Measurement iterations per trial"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate space schema and print plan, do not run"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_FILE,
        help="Output results JSON file"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="NPU device ID (default: from TILE_FWK_DEVICE_ID env)"
    )
    parser.add_argument(
        "--create-space",
        action="store_true",
        help="Create default space file and exit"
    )
    
    args = parser.parse_args()
    
    # Create default space file if requested
    if args.create_space:
        create_default_space_file(args.space)
        return 0
    
    # Check space file exists
    if not Path(args.space).exists():
        print(f"Space file not found: {args.space}", file=sys.stderr)
        print(f"Creating default space file...", file=sys.stderr)
        create_default_space_file(args.space)
    
    # Load search space
    print(f"Loading search space: {args.space}")
    space = load_search_space(args.space)
    
    print(f"\n{'='*60}")
    print("AUTOTUNE CONFIGURATION")
    print('='*60)
    print(f"Preset: {args.bench}")
    print(f"Max trials per stage: {args.trials}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measure iterations: {args.measure}")
    print(f"Dry-run mode: {args.dry_run}")
    
    # Print space summary
    print(f"\n{'='*60}")
    print("SEARCH SPACE SUMMARY")
    print('='*60)
    for stage in ["stage_a", "stage_b", "stage_c"]:
        if stage in space:
            print(f"\n{stage.upper()}:")
            stage_data = space[stage]
            for key, values in stage_data.items():
                if isinstance(values, list):
                    print(f"  {key}: {len(values)} candidates")
    
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY-RUN COMPLETE")
        print('='*60)
        print("Space schema validated successfully.")
        print("Run without --dry-run to execute autotuning.")
        
        # Save dry-run results
        results = {
            "best_config": {"stage_a": {}, "stage_b": {}, "stage_c": {}},
            "best_latency_ms": None,
            "trials": [],
            "reproducibility": get_reproducibility_info(),
            "dry_run": True
        }
        
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.out}")
        return 0
    
    # Check NPU environment
    device_id = args.device_id or int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    print(f"\nNPU Device ID: {device_id}")
    
    if not os.environ.get("ASCEND_HOME_PATH"):
        print("ERROR: ASCEND_HOME_PATH not set. NPU environment required.", file=sys.stderr)
        sys.exit(1)
    
    # Import torch_npu for actual execution
    try:
        import torch_npu
        import torch
    except ImportError as e:
        print(f"ERROR: Failed to import torch_npu: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Define benchmark function (placeholder - would be customized per preset)
    def softmax_benchmark(**config):
        # This would be implemented with actual PyPTO benchmark
        # Placeholder for demonstration
        x = torch.randn(32, 32, 1, 256, device=f"npu:{device_id}")
        # Apply config and run softmax...
        return x
    
    # Run autotuning
    bench_fn = softmax_benchmark  # Would be selected based on args.bench
    
    results = run_autotune(
        space=space,
        bench_fn=bench_fn,
        trials=args.trials,
        warmup=args.warmup,
        measure=args.measure,
        dry_run=args.dry_run
    )
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("AUTOTUNE COMPLETE")
    print('='*60)
    if results["best_latency_ms"]:
        print(f"Best latency: {results['best_latency_ms']:.3f} ms")
    print(f"Best config: {results['best_config']}")
    print(f"\nResults saved to: {args.out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
