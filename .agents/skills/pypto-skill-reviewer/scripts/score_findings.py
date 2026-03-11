#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


STATUS_PRIORITY = {"FAIL": 4, "WARN": 3, "PASS": 2, "SKIP": 1}
GRADE_THRESHOLDS = [(95, "A"), (85, "B"), (70, "C"), (55, "D"), (0, "F")]
S0_VETO_CAP = 59.9


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_status(value):
    if not isinstance(value, str):
        return "SKIP"
    upper = value.upper()
    if upper in STATUS_PRIORITY:
        return upper
    mapping = {
        "通过": "PASS",
        "失败": "FAIL",
        "跳过": "SKIP",
        "警告": "WARN",
    }
    return mapping.get(value, "SKIP")


def merge_findings(existing, incoming):
    if existing is None:
        return incoming
    left = STATUS_PRIORITY.get(normalize_status(existing.get("status")), 0)
    right = STATUS_PRIORITY.get(normalize_status(incoming.get("status")), 0)
    return incoming if right >= left else existing


def get_dimension_weights(rules_data):
    out = {}
    dims = rules_data.get("dimensions", {})
    for key, value in dims.items():
        if isinstance(value, dict):
            weight = value.get("weight", 0)
        else:
            weight = value
        weight = float(weight)
        out[key] = weight / 100.0 if weight > 1 else weight
    return out


def pick_grade(total, s0_veto):
    grade = "F"
    for threshold, label in GRADE_THRESHOLDS:
        if total >= threshold:
            grade = label
            break
    if s0_veto and grade in {"A", "B", "C"}:
        return "D"
    return grade


def score(rules_data, findings, skill_path):
    rules = rules_data.get("rules", [])
    severity_deductions = rules_data.get("severity_deductions", {})
    dimension_weights = get_dimension_weights(rules_data)

    by_rule = {}
    for rule in rules:
        rule_id = rule["id"]
        by_rule[rule_id] = {
            "rule_id": rule_id,
            "severity": rule.get("severity"),
            "dimension": rule.get("dimension"),
            "type": rule.get("type"),
            "status": "SKIP",
            "message": "",
            "evidence": {"file": "", "line": 0, "snippet": ""},
        }

    for item in findings:
        rule_id = item.get("rule_id")
        if rule_id not in by_rule:
            continue
        merged = dict(by_rule[rule_id])
        merged.update(item)
        merged["status"] = normalize_status(item.get("status"))
        merged["severity"] = by_rule[rule_id]["severity"]
        merged["dimension"] = by_rule[rule_id]["dimension"]
        merged["type"] = by_rule[rule_id]["type"]
        by_rule[rule_id] = merge_findings(by_rule[rule_id], merged)

    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    deductions_by_dim = {f"D{i}": 0 for i in range(1, 10)}
    s0_veto = False

    for item in by_rule.values():
        status = normalize_status(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status != "FAIL":
            continue
        sev = item.get("severity")
        dim = item.get("dimension")
        deductions_by_dim[dim] = deductions_by_dim.get(dim, 0) + int(severity_deductions.get(sev, 0))
        if sev == "S0":
            s0_veto = True

    scripts_dir = os.path.join(skill_path, "scripts")
    d9_no_scripts = not os.path.isdir(scripts_dir)

    dimensions = {}
    for dim, weight in dimension_weights.items():
        if dim == "D9" and d9_no_scripts:
            raw = 100
            deductions = 0
        else:
            deductions = deductions_by_dim.get(dim, 0)
            raw = max(0, 100 - deductions)
        dimensions[dim] = {
            "raw": raw,
            "weight": weight,
            "deductions": deductions,
            "score": round(raw * weight, 2),
        }

    total = round(sum(item["score"] for item in dimensions.values()), 2)
    if s0_veto and total > S0_VETO_CAP:
        total = S0_VETO_CAP
    grade = pick_grade(total, s0_veto)

    expected = len(rules)
    evaluated = counts["PASS"] + counts["FAIL"] + counts["WARN"] + counts["SKIP"]
    coverage = round((evaluated / expected) * 100, 2) if expected else 0.0

    return {
        "total": total,
        "grade": grade,
        "s0_veto": s0_veto,
        "d9_no_scripts": d9_no_scripts,
        "counts": {
            "pass_count": counts["PASS"],
            "fail_count": counts["FAIL"],
            "warn_count": counts["WARN"],
            "skip_count": counts["SKIP"],
        },
        "coverage": {
            "expected": expected,
            "evaluated": evaluated,
            "coverage_percent": coverage,
        },
        "dimensions": dimensions,
        "findings": list(by_rule.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Score skill findings deterministically")
    parser.add_argument("--rules", required=True, help="Path to rules.json")
    parser.add_argument("--skill-path", required=True, help="Target skill path")
    parser.add_argument("--findings", help="Path to merged findings JSON array")
    parser.add_argument("--static", help="Path to static findings JSON array")
    parser.add_argument("--semantic", help="Path to semantic findings JSON array")
    parser.add_argument("--out", help="Output file path (defaults to stdout)")
    return parser.parse_args()


def main():
    args = parse_args()
    rules_data = load_json(args.rules)

    findings = []
    if args.findings:
        findings.extend(load_json(args.findings))
    if args.static:
        findings.extend(load_json(args.static))
    if args.semantic:
        findings.extend(load_json(args.semantic))

    result = score(rules_data, findings, args.skill_path)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
