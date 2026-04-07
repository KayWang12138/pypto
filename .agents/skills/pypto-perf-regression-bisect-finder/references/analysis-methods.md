# 性能分析方法参考

本文档提供了常用的性能分析方法从泳道图中提取指标的详细说明。

## 内置分析方法

### 方法 1：aic_last_to_aiv_first

**描述**：计算最后一个 AIC 完成时间和第一个 AIV 开始时间的差值

**适用场景**：
- AIC 任务和 AIV 任务之间存在明显间隙
- 关注 AIC 全部完成到 AIV 开始执行的等待时间

**计算逻辑**：
1. 提取所有 AIC 线程的事件
2. 找到所有 AIC 事件的最大 `end` 时间戳
3. 提取所有 AIV 线程的事件
4. 找到所有 AIV 事件的最小 `start` 时间戳
5. 计算差值：`gap = aiv_first_start - aic_latest_end`

**示例代码**：
```python
def aic_last_to_aiv_first(swimlane_file):
    with open(swimlane_file, 'r') as f:
        data = json.load(f)

    trace = data.get('traceEvents', [])

    # 获取线程名称映射
    thread_names = {}
    for e in trace:
        if e.get('name') == 'thread_name':
            tid = e.get('tid')
            thread_names[tid] = e.get('args', {}).get('name', '')

    # 提取 AIC 和 AIV 事件
    aic_events = []
    aiv_events = []

    for e in trace:
        if e.get('ph') in ['X', 'B', 'E']:
            tid = e.get('tid')
            thread_name = thread_names.get(tid, '')
            ts = e.get('ts', 0)
            dur = e.get('dur', 0)

            if thread_name.startswith('AIC'):
                aic_events.append({
                    'start': ts,
                    'end': ts + dur,
                    'dur': dur
                })
            elif thread_name.startswith('AIV'):
                aiv_events.append({
                    'start': ts,
                    'end': ts + dur,
                    'dur': dur
                })

    if aic_events and aiv_events:
        aic_latest_end = max([e['end'] for e in aic_events])
        aiv_first_start = min([e['start'] for e in aiv_events])
])
        gap = aiv_first_start_start - aic_latest_end
        return gap

    return None
```

### 方法 2：aic_first_to_aiv_first

**描述**：计算第一个 AIC 完成时间和第一个 AIV 开始时间的差值

**适用场景**：
- 关注 AIC 任务启动到 AIV 任务启动的等待时间
- 适用于 AIC 任务按顺序执行的场景

**计算逻辑**：
1. 提取所有 AIC 线程的事件
2. 找到所有 AIC 事件的最小 `end` 时间戳
3. 提取所有 AIV 线程的事件
4. 找到所有 AIV 事件的最小 `start` 时间戳
5. 计算差值：`gap = aiv_first_start - aic_first_end`

### 方法 3：total_time

**描述**：计算泳道图中所有事件的总时间跨度

**适用场景**：
- 关注整个测试用例的执行时间
- 适用于性能整体退化的场景

**计算逻辑**：
1. 提取所有事件的时间戳
2. 找到最小的 `start` 时间戳
3. 找到最大的 `end` 时间戳
4. 计算差值：`total_time = last_end - first_start`

## 自定义分析方法

### 气泡率分析

**描述**：计算气泡等待时间占总时间的比例

**适用场景**：
- 关注核心利用率
- 识别调度策略问题

**计算逻辑**：
```python
def bubble_rate(swimlane_file):
    with open(swimlane_file, 'r') as f:
        data = json.load(f)

    trace = data.get('traceEvents', [])

    bubble_time = 0
    aicore_time = 0

    for e in trace:
        if e.get('name') == 'Bubble':
            bubble_time += e.get('dur', 0)
        elif e.get('name') == 'AICore':
            aicore_time += e.get('dur', 0)

    if aicore_time + bubble_time > 0:
        return bubble_time / (aicore_time + bubble_time)

    return None
```

### 核心利用率分析

**描述**：计算所有核心的平均利用率

**适用场景**：
- 关注多核性能
- 识别负载均衡问题

**计算逻辑**：
```python
def core_utilization(swimlane_file):
    with open(swimlane_file, 'r') as f:
        data = json.load(f)

    trace = data.get('traceEvents', [])

    # 按核心分组
    core_times = {}

    for e in trace:
        if e.get('ph') in ['X', 'B', 'E']:
            tid = e.get('tid')
            dur = e.get('dur', 0)

            if tid not in core_times:
                core_times[tid] = 0
            core_times[tid] += dur

    # 计算平均利用率
    if core_times:
        total_time = sum(core_times.values())
        avg_time = total_time / len(core_times)
        return avg_time

    return None
```

## 选择建议

| 性能现象 | 推荐分析方法 | 判断标准示例 |
|---------|-------------|-------------|
| AIC-AIV 间隙变大 | `aic_last_to_aiv_first` 或 `aic_first_to_aiv_first` | Good: < 8.0us, Bad: >= 10.0us |
| 总执行时间变长 | `total_time` | Good: < 1000.0us, Bad: >= 1200.0us |
| 核心利用率下降 | 自定义 `core_utilization` | Good: > 80%, Bad: <= 60% |
| 气泡率增加 | 自定义 `bubble_rate` | Good: < 10%, Bad: >= 20% |
