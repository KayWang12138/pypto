#!/usr/bin/env python3
"""update_known_limitations.py — 从断裂点检测结果中提取新发现，追加到 known-limitations.md。

用法:
    python update_known_limitations.py \
      --known-limitations autodev/known-limitations.md \
      --fracture-summary autodev/custom/relu/fracture-summary.json

逻辑:
    1. 读取 fracture-summary.json 中 confidence=high 的断裂点
    2. 提取涉及的 PyPTO API/功能、错误描述、workaround
    3. 与 known-limitations.md 现有内容去重（按关键词匹配）
    4. 新条目追加到文件末尾
    5. 无新条目则不改文件

退出码: 0=成功（含无新条目）, 1=输入文件不存在, 2=解析错误
"""
import argparse
import json
import sys
from pathlib import Path


def load_fracture_summary(path: str) -> list[dict]:
    """加载 fracture-summary.json，返回高置信断裂点列表。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []

    fps = data.get("fracture_points", data.get("fps", []))
    if not isinstance(fps, list):
        return []

    return [fp for fp in fps if fp.get("confidence") == "high"]


def load_existing_content(path: str) -> str:
    """读取现有 known-limitations.md 内容。"""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def is_duplicate(fp: dict, existing_content: str) -> bool:
    """简单去重：检查断裂点的关键实体是否已在文件中提及。"""
    entity = fp.get("entity", "")
    if not entity:
        return True  # 无实体信息的断裂点无法入库
    # 检查实体名是否已出现在 known-limitations 中
    return entity.lower() in existing_content.lower()


def format_entry(fp: dict, op_name: str) -> str:
    """将断裂点格式化为 known-limitations 条目。"""
    entity = fp.get("entity", "未知")
    description = fp.get("description", fp.get("root_cause", ""))
    workaround = fp.get("workaround", fp.get("suggestion", "暂无"))
    impact = fp.get("impact", "")
    fp_type = fp.get("type", "")

    lines = [f"### {entity} {fp_type}"]
    if description:
        lines.append(f"- {description}")
    if workaround and workaround != "暂无":
        lines.append(f"- **Workaround**: {workaround}")
    if impact:
        lines.append(f"- **影响**: {impact}")
    lines.append(f"- **来源**: {op_name}（autodev 自动发现）")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="更新 known-limitations.md")
    parser.add_argument("--known-limitations", required=True, help="known-limitations.md 路径")
    parser.add_argument("--fracture-summary", required=True, help="fracture-summary.json 路径")
    args = parser.parse_args()

    # 加载
    fracture_path = Path(args.fracture_summary)
    if not fracture_path.exists():
        print(json.dumps({"updated": False, "reason": "fracture-summary not found"}))
        sys.exit(0)  # 不是错误，只是没有输入

    high_fps = load_fracture_summary(args.fracture_summary)
    if not high_fps:
        print(json.dumps({"updated": False, "reason": "no high-confidence fracture points"}))
        sys.exit(0)

    existing = load_existing_content(args.known_limitations)

    # 从 fracture-summary 推断 op_name
    try:
        summary_data = json.loads(fracture_path.read_text(encoding="utf-8"))
        op_name = summary_data.get("op_name", fracture_path.parent.name)
    except Exception:
        op_name = fracture_path.parent.name

    # 去重并收集新条目
    new_entries = []
    for fp in high_fps:
        if not is_duplicate(fp, existing):
            new_entries.append(format_entry(fp, op_name))

    if not new_entries:
        print(json.dumps({"updated": False, "reason": "all entries already exist", "checked": len(high_fps)}))
        sys.exit(0)

    # 追加到文件
    kl_path = Path(args.known_limitations)
    kl_path.parent.mkdir(parents=True, exist_ok=True)

    separator = "\n\n" if existing.strip() else ""
    with open(kl_path, "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(f"{separator}{entry}\n")
            separator = "\n"

    print(json.dumps({
        "updated": True,
        "new_entries": len(new_entries),
        "total_checked": len(high_fps),
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
