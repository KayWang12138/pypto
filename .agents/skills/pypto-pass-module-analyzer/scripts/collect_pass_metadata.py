#!/usr/bin/env python3
import argparse
import difflib
import json
import re
from pathlib import Path


def to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def normalize_pass_name(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    if "_" in text and text.upper() == text:
        return "".join(part.title() for part in text.lower().split("_"))
    if "_" in text:
        return "".join(part.title() for part in text.split("_"))
    return text[0].upper() + text[1:]


def parse_pvc2_ooo(pass_manager_cpp: Path):
    text = pass_manager_cpp.read_text(encoding="utf-8", errors="ignore")
    block_match = re.search(
        r'RegisterStrategy\(\s*"PVC2_OOO"\s*,\s*\{(.*?)\}\s*\);',
        text,
        flags=re.S,
    )
    if not block_match:
        return []
    block = block_match.group(1)
    pairs = re.findall(r'\{"([A-Za-z0-9_]+)"\s*,\s*PassName::([A-Z0-9_]+)\}', block)
    result = []
    for idx, (name, enum_name) in enumerate(pairs):
        result.append({"index": idx, "name": name, "enum": enum_name})
    return result


def find_candidates(repo_root: Path, canonical_name: str):
    passes_root = repo_root / "framework/src/passes"
    if not passes_root.exists():
        return []
    stem = to_snake(canonical_name)
    candidates = []
    for path in passes_root.rglob("*.cpp"):
        rel = path.relative_to(repo_root)
        rel_s = str(rel)
        score = 0
        if path.stem == stem:
            score += 4
        if stem in path.stem:
            score += 2
        content = path.read_text(encoding="utf-8", errors="ignore")
        if canonical_name in content:
            score += 2
        if f"REG_PASS({canonical_name})" in content:
            score += 3
        if score > 0:
            candidates.append((score, rel_s))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [item[1] for item in candidates[:10]]


def main():
    parser = argparse.ArgumentParser(description="Collect pass metadata for report generation.")
    parser.add_argument("--pass", dest="pass_name", required=True, help="Target pass name")
    parser.add_argument("--repo-root", default=".", help="Repo root path")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pass_manager_cpp = repo_root / "framework/src/passes/pass_mgr/pass_manager.cpp"

    if not pass_manager_cpp.exists():
        print(json.dumps({
            "error": "pass_manager.cpp not found",
            "path": str(pass_manager_cpp),
        }, ensure_ascii=False, indent=2))
        return 1

    canonical_name = normalize_pass_name(args.pass_name)
    pvc2_ooo = parse_pvc2_ooo(pass_manager_cpp)

    index = None
    enum_name = None
    for item in pvc2_ooo:
        if item["name"].lower() == canonical_name.lower():
            index = item["index"]
            enum_name = item["enum"]
            break

    if index is None:
        out_name = f"99_{to_snake(canonical_name).upper()}.md"
    else:
        out_name = f"{index:02d}_{to_snake(canonical_name).upper()}.md"

    candidates = find_candidates(repo_root, canonical_name)

    available_names = [item["name"] for item in pvc2_ooo]
    closest = difflib.get_close_matches(canonical_name, available_names, n=10, cutoff=0.0)

    output = {
        "canonical_name": canonical_name,
        "pvc2_ooo_index": index,
        "pass_enum": enum_name,
        "output_file_name": out_name,
        "candidate_files": candidates,
        "closest_passes": closest,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
