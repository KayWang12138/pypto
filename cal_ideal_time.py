#!/usr/bin/env python3
# coding: utf-8
"""
基于 merged_swimlane.json 和 dyn_topo.txt 计算 ideal time。
逻辑与 extract_esl_v3.py 中的 calculate_total_time 一致：
保持每个 task 的原始核分配和执行时长，去除调度开销（气泡），
通过拓扑依赖进行 BFS 仿真得到理想端到端执行时间。

Fake/HUB tasks（虚拟依赖节点）每个分配独立的虚拟核心，
确保依赖传递不受共享核心排序约束。

用法:
  单文件:   python3 calc_ideal_time.py <merged_swimlane.json> <dyn_topo.txt>
  批量目录: python3 calc_ideal_time.py --dir <json_dir> <dyn_topo.txt>
  画ideal_time 图: python3 calc_ideal_time.py --plot <merged_swimlane.json> <dyn_topo.txt>
"""
import json
import sys
import os
import math
import importlib
from collections import defaultdict, deque

HUB_CORE_TYPE = 4
SEQNO_SHFT_BITS = 32
TASK_ID_MASK = 0xFFFFFFFF


def make_task_uid(task_id, seq_no, encode_seqno):
    """按需将 (seqNo, taskId) 组合成唯一 uid。"""
    if not encode_seqno:
        return task_id
    return (seq_no << SEQNO_SHFT_BITS) | (task_id & TASK_ID_MASK)


def parse_swimlane(swimlane_path, encode_seqno=False):
    """解析 merged_swimlane.json，返回 (tid_name, tid_to_blk, core_tasks, hub_blk_set)

    每个 HUB/fake task 分配独立的虚拟核心索引，避免共享核心的排序问题。
    即使某个 tid 同时包含 real/fake task，也会把 fake task 单独拆出。
    返回的 hub_blk_set 包含所有 HUB 虚拟核心的 blk 编号。
    """
    with open(swimlane_path, "r") as f:
        trace = json.load(f)
    events = trace["traceEvents"]

    tid_name = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid_name[ev["tid"]] = ev["args"]["name"]

    tid_tasks = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        ev_name = ev.get("name", "")
        thread_name = tid_name.get(ev["tid"], "")
        is_fake = (
            ev.get("args", {}).get("color") == "fake"
            or "(fake)" in ev_name
            or thread_name.startswith("Fake Core")
        )
        seq_no = ev["args"].get("seqNo", 0)
        raw_task_id = ev["args"]["taskId"]
        tid_tasks[ev["tid"]].append({
            "taskId": make_task_uid(raw_task_id, seq_no, encode_seqno),
            "rawTaskId": raw_task_id,
            "seqNo": seq_no,
            "execStart": ev["ts"],
            "execEnd": ev["ts"] + ev["dur"],
            "execTime": ev["dur"],
            "label": ev.get("name", ""),
            "is_hub": is_fake,
        })

    tid_to_blk = {}
    core_tasks = {}
    hub_blk_set = set()
    next_blk = 0

    for tid in sorted(tid_tasks):
        real_tasks = [t for t in tid_tasks[tid] if not t["is_hub"]]
        fake_tasks = [t for t in tid_tasks[tid] if t["is_hub"]]

        if real_tasks:
            blk = next_blk
            next_blk += 1
            tid_to_blk[tid] = blk
            for t in real_tasks:
                t["blockIdx"] = blk
            real_tasks.sort(key=lambda x: x["execStart"])
            core_tasks[blk] = real_tasks
        else:
            tid_to_blk[tid] = None

        for t in fake_tasks:
            blk = next_blk
            next_blk += 1
            t["blockIdx"] = blk
            core_tasks[blk] = [t]
            hub_blk_set.add(blk)

    return tid_name, tid_to_blk, core_tasks, hub_blk_set


def parse_topo(topo_path):
    """解析 topo，必要时按 seqNo 组合唯一 task uid。

    返回:
      topo_list: [{taskId, seqNo, coreType, successors}, ...]
      encode_seqno: 是否启用了 seqNo + taskId 的 uid 编码
    """
    raw_topo_list = []
    seen_uids = set()
    with open(topo_path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            seq_no = int(parts[0])
            task_id = int(parts[1])
            core_type = int(parts[7])
            succs_raw = parts[9:]
            successors = [int(x) for x in succs_raw if x.strip() != ""]

            uid = (seq_no, task_id)
            if uid in seen_uids:
                break
            seen_uids.add(uid)
            raw_topo_list.append({
                "taskId": task_id,
                "seqNo": seq_no,
                "coreType": core_type,
                "successors": successors,
            })

    # 原始 extract_esl_v3.py 会把 seqNo 编进 task uid。
    # 这里仅在检测到裸 taskId 跨 seqNo 重复时启用，避免破坏已是全局唯一 id 的输入。
    task_id_to_seqnos = defaultdict(set)
    for task in raw_topo_list:
        task_id_to_seqnos[task["taskId"]].add(task["seqNo"])
    encode_seqno = any(len(seqnos) > 1 for seqnos in task_id_to_seqnos.values())

    topo_list = []
    for task in raw_topo_list:
        seq_no = task["seqNo"]
        topo_list.append({
            "taskId": make_task_uid(task["taskId"], seq_no, encode_seqno),
            "seqNo": seq_no,
            "coreType": task["coreType"],
            "successors": [
                make_task_uid(succ, seq_no, encode_seqno)
                for succ in task["successors"]
            ],
        })
    return topo_list, encode_seqno


def calculate_ideal_time(core_tasks, topo_list, hub_blk_set=None):
    """与 extract_esl_v3.py 的 calculate_total_time 逻辑一致。

    每个 HUB task 在独立虚拟核心上，不受共享核心排序约束。

    返回: (ideal_time, core_finish_time, task_start_end)
    """
    if hub_blk_set is None:
        hub_blk_set = set()

    uid2suid = {}
    uid2puid = defaultdict(list)
    for topo_task in topo_list:
        tid = topo_task["taskId"]
        uid2suid[tid] = topo_task["successors"]
        uid2puid[tid] = uid2puid.get(tid, [])
        for suid in topo_task["successors"]:
            uid2puid[suid] = uid2puid.get(suid, [])
            uid2puid[suid].append(tid)

    all_tasks = {}
    for blk_idx, tasks in core_tasks.items():
        for t in tasks:
            t["successors"] = uid2suid.get(t["taskId"], [])
            all_tasks[t["taskId"]] = t

    blk_ids = list(core_tasks.keys())
    blk_rng = max(blk_ids) + 1

    task_exec_time = {}
    task_in_degree = defaultdict(int)
    task_pred_finish = defaultdict(list)
    core_finish_time = [0] * blk_rng
    task_start_end = {}
    core_current_idx = [0] * blk_rng
    seq_cnt = 0

    for blk_idx, tasks in core_tasks.items():
        for task in tasks:
            seq_cnt = max(seq_cnt, task["seqNo"] + 1)
            task_exec_time[task["taskId"]] = task["execTime"]
            for succ in task["successors"]:
                task_in_degree[succ] += 1

    device_task_start_time = 0

    for i in range(seq_cnt):
        queue = deque()
        for blk_idx, tasks in core_tasks.items():
            for task in tasks:
                t_id = task["taskId"]
                if task_in_degree[t_id] == 0 and task["seqNo"] == i:
                    queue.append((t_id, blk_idx))
                    core_current_idx[blk_idx] += 1

        while queue:
            tid, blk_idx = queue.popleft()
            pred_times = task_pred_finish.get(tid, [0])
            start_time = max(pred_times)
            start_time = max(start_time, device_task_start_time)
            start_time = max(start_time, core_finish_time[blk_idx])
            finish_time = start_time + task_exec_time[tid]
            core_finish_time[blk_idx] = finish_time
            task_start_end[tid] = (start_time, finish_time)

            for succ in uid2suid.get(tid, []):
                task_in_degree[succ] -= 1
                task_pred_finish[succ].append(finish_time)

            for _blk in blk_ids:
                if core_current_idx[_blk] < len(core_tasks[_blk]):
                    next_task = core_tasks[_blk][core_current_idx[_blk]]
                    next_task_id = next_task["taskId"]
                    if task_in_degree[next_task_id] == 0 and next_task["seqNo"] == i:
                        queue.append((next_task_id, _blk))
                        core_current_idx[_blk] += 1

        device_task_start_time = max(core_finish_time)

    ideal_time = max(core_finish_time)
    return ideal_time, core_finish_time, task_start_end


def _format_task_label(task):
    """构造泳道图中的 task 标签。"""
    raw_task_id = task.get("rawTaskId", task["taskId"])
    seq_no = task.get("seqNo", 0)
    if seq_no > 0:
        return f"{seq_no}:{raw_task_id:x}"
    return hex(raw_task_id).replace("0x", "")


def plot_ideal_swimlane(core_tasks, task_start_end, blk_to_name, output_path):
    """根据 ideal start/end 结果绘制去掉调度空隙后的泳道图。"""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前环境缺少 matplotlib，无法生成泳道图。请先安装 matplotlib，或不带 --plot 运行。"
        ) from exc

    ordered_blks = sorted(core_tasks.keys())
    num_cores = len(ordered_blks)
    fig_height = max(6, min(24, num_cores * 0.35 + 2))
    fig, ax = plt.subplots(figsize=(20, fig_height))
    colors = ["r", "g", "b", "c", "m", "y"]

    bar_height = 0.8
    bar_gap = 0.2
    last_tick = 0
    total_tasks = sum(len(tasks) for tasks in core_tasks.values())
    show_text = total_tasks <= 400

    for core_pos, blk in enumerate(ordered_blks):
        tasks = core_tasks[blk]
        y = core_pos * (bar_height + bar_gap)
        for j, task in enumerate(tasks):
            task_id = task["taskId"]
            if task_id not in task_start_end:
                continue
            start_time, end_time = task_start_end[task_id]
            color = colors[j % len(colors)]
            ax.barh(
                y,
                end_time - start_time,
                left=start_time,
                height=bar_height,
                color=color,
            )
            if show_text and end_time > start_time:
                ax.text(
                    start_time + (end_time - start_time) / 2,
                    y,
                    _format_task_label(task),
                    ha="center",
                    va="center",
                    fontsize=6,
                )
            last_tick = max(end_time, last_tick)

    ax.set_yticks([i * (bar_height + bar_gap) for i in range(num_cores)])
    ax.set_yticklabels(
        [blk_to_name.get(blk, f"Blk {blk}") for blk in ordered_blks],
        fontsize=10,
    )
    ax.set_ylabel("Blks")

    if last_tick <= 0:
        x_ticks = [0]
        x_labels = ["0"]
        tick_log_scale = 0
    else:
        tick_cnt = 100
        tick_log_scale = max(
            round(math.log10(last_tick)) - math.ceil(math.log10(tick_cnt)),
            0,
        )
        tick_step = max(1, int(10 ** tick_log_scale))
        x_ticks = list(range(0, int(last_tick) + tick_step, tick_step))
        if len(x_ticks) == 1:
            x_ticks.append(tick_step)
        tick_label_scale = 10 ** tick_log_scale
        label_every_n_tick = max(
            1,
            int(2 ** math.floor(math.log10(max(x_ticks[-1] / tick_step, 1)))),
        )
        x_labels = [
            "%d" % (tick / tick_label_scale) if (i % label_every_n_tick == 0) else ""
            for i, tick in enumerate(x_ticks)
        ]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f"Time in Ticks [10^{tick_log_scale}]")
    ax.set_title("Ideal Task Execution Timeline")
    fig.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


def process_single(swimlane_path, topo_path, verbose=True, plot_path=None):
    """处理单个 swimlane json，返回结果字典。"""
    topo_list, encode_seqno = parse_topo(topo_path)
    tid_name, tid_to_blk, core_tasks, hub_blk_set = parse_swimlane(
        swimlane_path, encode_seqno=encode_seqno
    )

    total_tasks = sum(len(v) for v in core_tasks.values())
    hub_tasks = sum(1 for tasks in core_tasks.values() for t in tasks if t["is_hub"])
    real_tasks = total_tasks - hub_tasks
    num_real_cores = sum(1 for blk in core_tasks if blk not in hub_blk_set)

    if len(topo_list) != total_tasks and verbose:
        print(f"  [Warning] swimlane task 数 ({total_tasks}) != topo task 数 ({len(topo_list)})")

    ideal_time, core_finish_time, task_start_end = calculate_ideal_time(
        core_tasks, topo_list, hub_blk_set
    )

    scheduled = len(task_start_end)
    if scheduled != total_tasks and verbose:
        print(f"  [Warning] BFS 仅调度了 {scheduled}/{total_tasks} 个 task")

    # 瓶颈核心名称映射（只考虑非 HUB 核心）
    blk_to_name = {}
    for tid_val, blk in tid_to_blk.items():
        if blk is not None:
            blk_to_name[blk] = tid_name.get(tid_val, f"Core_{blk}")
    for blk in hub_blk_set:
        hub_tasks_for_blk = core_tasks.get(blk, [])
        if hub_tasks_for_blk:
            blk_to_name[blk] = f"Fake {hub_tasks_for_blk[0].get('rawTaskId', blk)}"
        else:
            blk_to_name[blk] = f"Fake {blk}"

    non_hub_blks = [b for b in core_tasks.keys() if b not in hub_blk_set]
    if non_hub_blks:
        bottleneck_blk = max(non_hub_blks, key=lambda b: core_finish_time[b])
    else:
        bottleneck_blk = max(core_tasks.keys(), key=lambda b: core_finish_time[b])
    bottleneck_core = blk_to_name.get(bottleneck_blk, f"Core_{bottleneck_blk}")
    ideal_time_real = core_finish_time[bottleneck_blk]
    ideal_time_all = ideal_time

    # actual time 只统计 real（非 fake）tasks 的时间跨度
    real_starts = []
    real_ends = []
    for tasks in core_tasks.values():
        for t in tasks:
            if not t["is_hub"]:
                real_starts.append(t["execStart"])
                real_ends.append(t["execEnd"])
    actual_time = max(real_ends) - min(real_starts) if real_starts else 0

    result = {
        "file": os.path.basename(swimlane_path),
        "num_cores": num_real_cores,
        "num_tasks": real_tasks,
        "num_hub_tasks": hub_tasks,
        # ideal_time 与 actual_time 统一口径，仅统计 real core 的端到端时间。
        "ideal_time": ideal_time_real,
        "ideal_time_with_hub": ideal_time_all,
        "actual_time": actual_time,
        "speedup": actual_time / ideal_time_real if ideal_time_real > 0 else 0,
        "overhead_ratio": (actual_time - ideal_time_real) / actual_time * 100 if actual_time > 0 else 0,
        "bottleneck_core": bottleneck_core,
    }

    if verbose:
        print(f"  核心数: {result['num_cores']}, task 数: {result['num_tasks']} (+ {hub_tasks} hub/fake)")
        print(f"  Ideal time: {result['ideal_time']}  (瓶颈核心: {bottleneck_core})")
        if ideal_time_all != ideal_time_real:
            print(f"  Ideal time incl. hub/fake: {ideal_time_all}")
        print(f"  Actual time: {result['actual_time']}")
        if result['actual_time'] > 0:
            print(f"  Speedup: {result['speedup']:.4f}x, Overhead: {result['overhead_ratio']:.2f}%")

    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_ideal_swimlane(core_tasks, task_start_end, blk_to_name, plot_path)
        if verbose:
            print(f"  Ideal 泳道图已保存到: {plot_path}")

    return result


def main():
    args = sys.argv[1:]
    plot_enabled = False

    if "--plot" in args:
        plot_enabled = True
        args.remove("--plot")

    if len(args) < 2:
        print("用法:")
        print("  单文件:   python3 calc_ideal_time.py [--plot] <merged_swimlane.json> <dyn_topo.txt>")
        print("  批量目录: python3 calc_ideal_time.py [--plot] --dir <json_dir> <dyn_topo.txt>")
        sys.exit(1)

    if args[0] == "--dir":
        if len(args) < 3:
            print("用法: python3 calc_ideal_time.py [--plot] --dir <json_dir> <dyn_topo.txt>")
            sys.exit(1)
        json_dir = args[1]
        topo_path = args[2]

        json_files = sorted([
            f for f in os.listdir(json_dir)
            if f.endswith(".json")
        ])
        if not json_files:
            print(f"目录 {json_dir} 下未找到 .json 文件")
            sys.exit(1)

        print(f"找到 {len(json_files)} 个 .json 文件，topo: {topo_path}")
        print("=" * 100)

        results = []
        for jf in json_files:
            json_path = os.path.join(json_dir, jf)
            print(f"\n[{jf}]")
            try:
                plot_path = None
                if plot_enabled:
                    stem = os.path.splitext(jf)[0]
                    plot_path = os.path.join(json_dir, f"{stem}.ideal_swimlane.png")
                r = process_single(json_path, topo_path, verbose=True, plot_path=plot_path)
                results.append(r)
            except Exception as e:
                print(f"  [Error] 处理失败: {e}")
                results.append({"file": jf, "error": str(e)})

        print("\n" + "=" * 100)
        print("汇总结果:")
        print(f"{'文件名':<40} {'Ideal Time':>12} {'Actual Time':>12} {'Speedup':>10} {'Overhead':>10} {'瓶颈核心':<12}")
        print("-" * 100)
        for r in results:
            if "error" in r:
                print(f"{r['file']:<40} {'ERROR':>12}  {r['error']}")
            else:
                print(f"{r['file']:<40} {r['ideal_time']:>12.4f} {r['actual_time']:>12.4f} {r['speedup']:>9.4f}x {r['overhead_ratio']:>9.2f}% {r['bottleneck_core']:<12}")
        print("=" * 100)

    else:
        swimlane_path = args[0]
        topo_path = args[1]
        plot_path = None
        if plot_enabled:
            stem = os.path.splitext(os.path.basename(swimlane_path))[0]
            plot_dir = os.path.dirname(os.path.abspath(swimlane_path)) or "."
            plot_path = os.path.join(plot_dir, f"{stem}.ideal_swimlane.png")

        print("=" * 60)
        print(f"解析 {os.path.basename(swimlane_path)} ...")
        try:
            r = process_single(swimlane_path, topo_path, verbose=True, plot_path=plot_path)
        except Exception as e:
            print(f"[Error] {e}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("结果:")
        print(f"  Projected ideal e2e time: {r['ideal_time']}  (瓶颈核心: {r['bottleneck_core']})")
        print(f"  Actual e2e time (from swimlane): {r['actual_time']}")
        if r['actual_time'] > 0:
            print(f"  Speedup (actual / ideal): {r['speedup']:.4f}x")
            print(f"  Scheduling overhead ratio: {r['overhead_ratio']:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()