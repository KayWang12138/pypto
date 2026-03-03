# PyPTO 产物文件说明

## 输出目录结构

运行 PyPTO 程序后，产物生成在 `{work_dir}/output/output_{timestamp}/` 目录下。

```
output_20260302_234138_092971_59932/
├── program.json              # 程序描述文件
├── topo.json                 # 拓扑信息
├── merged_swimlane.json      # 泳道图数据（需要 runtime_debug_mode=1）
├── Pass_*/                   # 编译阶段产物（需要 compile_debug_mode=1）
│   ├── *.json               # 计算图 JSON
│   └── ...
├── kernel_aicore/           # AICore 内核
├── kernel_aicpu/            # AICpu 内核
└── built_in/
    └── pypto_op_info.json   # 算子信息
```

## 关键文件

### merged_swimlane.json
- **格式**: Chrome Trace Format (Perfetto 兼容)
- **生成条件**: `debug_options={"runtime_debug_mode": 1}`
- **用途**: 性能分析、利用率计算、瓶颈识别

#### 结构说明
```json
{
  "traceEvents": [
    {"ph": "M", "name": "process_name", ...},     // 元数据
    {"ph": "M", "name": "thread_name", ...},      // 线程名 (AIC_0, AIV_0)
    {"ph": "X", "name": "task_name", "ts": ..., "dur": ..., ...},  // 任务事件
    {"ph": "C", "name": "ReadyCount_AIC", ...}    // 计数器
  ]
}
```

#### 关键字段
- `ph`: 事件类型 (M=metadata, X=task, C=counter)
- `ts`: 开始时间戳（微秒）
- `dur`: 持续时间（微秒）
- `tid`: 线程 ID（对应 AIC/AIV 核心）
- `args.event-hint`: 包含 rootHash, callOpMagic, leafHash

### program.json
- **用途**: 程序结构描述
- **内容**: 算子信息、子图划分等

### Pass_* 计算图 JSON
- **生成条件**: `debug_options={"compile_debug_mode": 1}`
- **用途**: 查看编译阶段子图划分、节点数量

## 定位最新输出目录

```bash
# 方法1: 按时间排序
ls -lt /workspace/code/pypto/output/ | head -5

# 方法2: 使用脚本自动检测
python3 scripts/analyze.py --pypto-repo /workspace/code/pypto
```
