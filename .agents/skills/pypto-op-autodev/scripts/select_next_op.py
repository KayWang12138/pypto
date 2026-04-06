#!/usr/bin/env python3
"""select_next_op.py — 优先级分数计算 + 算子选择 + 断点续跑 + 依赖检查 + 智能重试。

用法:
    python select_next_op.py --csv {csv_path} [--dry-run]

退出码:
    0: 成功选择算子（score >= DISCOVERY_THRESHOLD）
    1: 需要触发 discover（无候选 或 最高分 < DISCOVERY_THRESHOLD）
    2: 参数错误
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from csv_ops import read_csv

DISCOVERY_THRESHOLD = 25

COMPLEXITY_PENALTY = {"easy": 0, "medium": 5, "hard": 10}

# 不同失败类型的额外惩罚 — BLOCKED 类应等待修复，惩罚更重
RESULT_PENALTY = {
    "BLOCKED_API": 40,
    "BLOCKED_IMPL": 35,
    "BLOCKED_DESIGN": 35,
    "BLOCKED_GOLDEN": 35,
    "BLOCKED_ACCURACY": 35,
    "BLOCKED_ENVIRONMENT": 35,
    "TIMEOUT": 5,
    "PRECISION": 0,
}


def now():
    return datetime.now(timezone.utc)


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def get_category_counts(all_rows):
    """统计每个类别已完成的算子数量。"""
    counts = {}
    for r in all_rows:
        if r.get("status") == "completed":
            cat = r.get("category", "other") or "other"
            counts[cat] = counts.get(cat, 0) + 1
    return counts


def get_complexity_counts(all_rows):
    """统计各复杂度已完成的算子数量。"""
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for r in all_rows:
        if r.get("status") == "completed":
            c = r.get("complexity", "easy") or "easy"
            counts[c] = counts.get(c, 0) + 1
    return counts


def detect_resume_stage(op_name, custom_dir):
    """检测已有产物，返回可续跑的阶段号（从该阶段开始）。

    返回:
        (resume_from_stage, existing_artifacts): 元组
        resume_from_stage=1 表示没有可复用产物，从头开始。
    """
    op_dir = custom_dir / op_name
    if not op_dir.exists():
        return 1, []

    existing = []
    last_completed_stage = 0

    # Stage 1: spec.md
    if (op_dir / "spec.md").exists():
        existing.append("spec.md")
        last_completed_stage = 1

    # Stage 2: api_report.md
    if (op_dir / "api_report.md").exists():
        existing.append("api_report.md")
        last_completed_stage = 2

    # Stage 3: {op}_golden.py
    golden = op_dir / f"{op_name}_golden.py"
    if golden.exists():
        existing.append(golden.name)
        last_completed_stage = 3

    # Stage 4: design.md
    if (op_dir / "design.md").exists():
        existing.append("design.md")
        last_completed_stage = 4

    # Stage 5: {op}_impl.py + test_{op}.py
    impl = op_dir / f"{op_name}_impl.py"
    test = op_dir / f"test_{op_name}.py"
    if impl.exists() and test.exists():
        existing.append(impl.name)
        existing.append(test.name)
        last_completed_stage = 5

    resume_from = last_completed_stage + 1 if last_completed_stage > 0 else 1
    return resume_from, existing


def check_dependencies(row, all_rows):
    """检查算子的前置依赖是否全部已完成。

    返回:
        (deps_met, unmet_deps): 依赖是否满足，以及未满足的依赖列表
    """
    depends_on = row.get("depends_on", "").strip()
    if not depends_on:
        return True, []

    deps = [d.strip() for d in depends_on.split("|") if d.strip()]
    completed_ops = {r["op_name"] for r in all_rows if r.get("status") == "completed"}

    unmet = [d for d in deps if d not in completed_ops]
    return len(unmet) == 0, unmet


def get_retry_strategy(row):
    """根据失败类型返回重试策略建议。"""
    dev_result = row.get("dev_result", "")
    strategies = {
        "BLOCKED_API": {"action": "wait", "reason": "API 不支持，等待框架更新"},
        "BLOCKED_IMPL": {"action": "retry_full", "reason": "实现阶段阻塞"},
        "BLOCKED_DESIGN": {"action": "retry_full", "reason": "设计阶段阻塞"},
        "BLOCKED_GOLDEN": {"action": "retry_full", "reason": "Golden 生成阻塞"},
        "BLOCKED_ACCURACY": {"action": "retry_from_stage6", "reason": "精度修复阻塞"},
        "BLOCKED_ENVIRONMENT": {"action": "fix_env", "reason": "环境问题，需先修复环境"},
        "PRECISION": {"action": "retry_from_stage6", "reason": "精度问题，从精度调试阶段重试"},
        "TIMEOUT": {"action": "retry_full", "reason": "超时，增加超时后重试"},
    }
    return strategies.get(dev_result, {"action": "retry_full", "reason": "常规重试"})


def score(row, recent_categories, category_counts=None, complexity_counts=None):
    base = 100

    # 复杂度提升机制：随着简单算子完成越多，medium/hard 获得加分
    complexity = row.get("complexity", "easy") or "easy"
    if complexity_counts is not None:
        easy_done = complexity_counts.get("easy", 0)
        medium_done = complexity_counts.get("medium", 0)
        total_done = easy_done + medium_done + complexity_counts.get("hard", 0)

        if complexity == "easy":
            # easy 完成越多，新 easy 的惩罚越重（每完成 3 个 easy 多扣 5 分）
            c_pen = min(easy_done // 3 * 5, 20)
        elif complexity == "medium":
            # easy 完成 >= 5 个后，medium 开始获得加分
            c_pen = max(-15, 5 - easy_done * 2)  # 负值 = 加分
        else:  # hard
            # easy >= 3 且 medium >= 2 后，hard 开始获得加分
            if easy_done >= 3 and medium_done >= 2:
                c_pen = max(-25, 10 - total_done * 3)
            else:
                c_pen = 10
    else:
        c_pen = COMPLEXITY_PENALTY.get(complexity, 0)

    # 失败惩罚（封顶 45，避免失败算子被永久淘汰）
    fail_count = int(row.get("fail_count") or 0)
    f_pen = min(fail_count * 15, 45)

    # 失败类型额外惩罚
    dev_result = row.get("dev_result", "")
    r_pen = RESULT_PENALTY.get(dev_result, 0)

    # 近期失败冷却衰减
    decay = 0
    if row.get("status") == "failed":
        end = parse_dt(row.get("end_time"))
        if end:
            hours_since = (now() - end).total_seconds() / 3600
            decay = max(0, 30 - hours_since)

    # 年龄奖励
    create = parse_dt(row.get("create_time"))
    if create:
        days = (now() - create).total_seconds() / 86400
        age_b = min(20, days * 2)
    else:
        age_b = 0

    # 类别多样性奖励
    cat = row.get("category", "other") or "other"
    if category_counts is not None:
        cat_count = category_counts.get(cat, 0)
        diversity_bonus = max(0, 25 - cat_count * 5)
    else:
        diversity_bonus = 0

    # 近期重复类别惩罚
    recent_pen = 15 if cat in recent_categories else 0

    return base - c_pen - f_pen - r_pen - decay + age_b + diversity_bonus - recent_pen


def is_candidate(row, custom_dir, all_rows):
    """检查是否为有效候选。

    排除条件：
    1. 状态不是 pending 或 failed
    2. dev_result 为 NOT_IMPLEMENTABLE
    3. {work_dir}/{op_name}/ 目录已存在且有完整工件（impl + test 都存在）
    4. 前置依赖未满足
    """
    status = row.get("status", "")
    dev_result = row.get("dev_result", "")
    if status not in ("pending", "failed"):
        return False
    if dev_result == "NOT_IMPLEMENTABLE":
        return False

    # 检查目录中是否有完整工件（仅 pending 状态排除，failed 算子仍可重试）
    op_name = row["op_name"]
    op_dir = custom_dir / op_name
    if status == "pending" and op_dir.exists():
        has_impl = (op_dir / f"{op_name}_impl.py").exists()
        has_test = (op_dir / f"test_{op_name}.py").exists()
        if has_impl and has_test:
            return False

    # 检查依赖
    deps_met, _ = check_dependencies(row, all_rows)
    if not deps_met:
        return False

    return True


def get_recent_categories(all_rows, n=3):
    completed = [r for r in all_rows if r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("end_time", ""), reverse=True)
    return {r["category"] for r in completed[:n]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    custom_dir = csv_path.parent / "custom"  # {csv_path}/../custom/

    all_rows = read_csv(args.csv)
    recent_cats = get_recent_categories(all_rows)
    cat_counts = get_category_counts(all_rows)
    comp_counts = get_complexity_counts(all_rows)
    candidates = [r for r in all_rows if is_candidate(r, custom_dir, all_rows)]

    if not candidates:
        dep_blocked = []
        for r in all_rows:
            if r.get("status") in ("pending", "failed") and r.get("dev_result") != "NOT_IMPLEMENTABLE":
                deps_met, unmet = check_dependencies(r, all_rows)
                if not deps_met:
                    dep_blocked.append({"op_name": r["op_name"], "unmet_deps": unmet})
        print(json.dumps({
            "action": "trigger_discovery",
            "selected": None,
            "need_discovery": True,
            "dep_blocked": dep_blocked,
        }))
        sys.exit(1)

    scored = [(score(r, recent_cats, cat_counts, comp_counts), r) for r in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_row = scored[0]

    if best_score < DISCOVERY_THRESHOLD:
        print(json.dumps({"action": "trigger_discovery", "selected": None, "need_discovery": True}))
        sys.exit(1)

    op_name = best_row["op_name"]

    # 断点续跑检测
    resume_from, existing_artifacts = detect_resume_stage(op_name, custom_dir)

    # 智能重试策略
    retry_strategy = None
    if best_row.get("status") == "failed":
        retry_strategy = get_retry_strategy(best_row)
        if retry_strategy["action"] == "retry_from_stage6" and resume_from <= 6:
            resume_from = 6

    result = {
        "action": "select_from_csv",
        "selected": {
            "op_name": op_name,
            "complexity": best_row.get("complexity", ""),
            "category": best_row.get("category", ""),
            "fail_count": int(best_row.get("fail_count") or 0),
            "score": round(best_score, 2),
            "resume_from_stage": resume_from,
            "existing_artifacts": existing_artifacts,
        },
        "retry_strategy": retry_strategy,
        "need_discovery": False,
    }
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
