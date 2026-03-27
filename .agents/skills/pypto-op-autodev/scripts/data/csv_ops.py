"""
csv_ops.py — 数据层：纯 CSV CRUD，不含业务逻辑。

公开接口:
    read_csv(path, **filters) -> list[dict]
    upsert_row(path, op_name, **fields) -> dict
"""
import csv
import shutil
from pathlib import Path

FIELDS = [
    "op_name", "source", "status", "complexity", "category",
    "dev_result", "fail_count", "fps_total", "fps_confirmed",
    "create_time", "start_time", "end_time", "note",
]


def read_csv(path: str, **filters) -> list:
    """读取 CSV，返回所有满足 filters 条件的行（list of dict）。
    filters 为字段名=值的关键字参数，全部满足才返回（AND 语义）。
    """
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not filters:
        return rows
    return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


def upsert_row(path: str, op_name: str, **fields) -> dict:
    """按 op_name 做 upsert。写入前自动备份为 .bak。
    返回写入后的完整行 dict。
    """
    p = Path(path)
    rows = []
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # 找到已有行或新建空行
    target = None
    for r in rows:
        if r["op_name"] == op_name:
            target = r
            break
    if target is None:
        target = {f: "" for f in FIELDS}
        target["op_name"] = op_name
        rows.append(target)

    # 更新字段
    for k, v in fields.items():
        if k in FIELDS:
            target[k] = str(v) if v is not None else ""

    # 备份
    if p.exists():
        shutil.copy2(p, str(p) + ".bak")

    # 确保父目录存在
    p.parent.mkdir(parents=True, exist_ok=True)

    # 写回
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return dict(target)
