# PyPTO Autodev 优化方案

**日期**: 2026-03-27  
**版本**: v1.0  
**作者**: Claude (基于 softmax 算子开发实践)  
**状态**: 待实施

---

## 一、背景

在执行 softmax 算子自动开发流程时，发现 `pypto-op-autodev` skill 存在以下问题：

1. **环境假设过强**: 假设 CSV 文件、requirements.md 等文件已存在
2. **错误信息不透明**: 脚本失败时难以定位问题
3. **缺少前置检查**: PyPTO 环境问题在开发中途才发现
4. **流程不够健壮**: 缺少初始化和容错机制

本文档基于实际执行经验，提出系统性的优化方案。

---

## 二、流程报错总结

### 2.1 Step 2a: requirements.md 文件不存在

**错误信息**:
```
ENOENT: no such file or directory, open '/workspace/code/pypto/autodev/sources/requirements.md'
```

**影响**: 跳过用户需求检查  
**当前处理**: 继续执行后续步骤  
**问题**: 未明确说明跳过原因，可能误导用户

### 2.2 Step 2b: CSV 文件不存在

**错误信息**:
```
Command failed: python .agents/skills/pypto-op-autodev/scripts/select_next_op.py --csv autodev/scan_results.csv
```

**根本原因**: 
- `autodev/` 目录不存在
- `autodev/scan_results.csv` 文件不存在

**当前处理**: 触发 discover 补充候选  
**问题**: 
- 错误信息不透明（stderr 不可见）
- 浪费一次脚本调用

### 2.3 Step 3: PyPTO 框架环境问题（非 autodev 问题）

**错误信息**:
```
AttributeError: module 'pypto.pypto_impl' has no attribute 'ShmemTensor'
```

**影响**: 阻塞运行时验证，但算子工件全部生成成功  
**当前处理**: 继续完成流程，在断裂点报告中记录  
**问题**: 
- 1 小时开发后才发现无法验证
- 应提前检查环境

---

## 三、优化方案详细设计

### 3.1 优化 1: 前置检查增强（Step 1）

**目标**: 在 Step 1 阶段完成所有环境检查，避免后续步骤失败

**实施方案**:

#### 3.1.1 修改 `update_op.py`

新增参数和功能：

```python
# update_op.py 新增参数
parser.add_argument('--check-env', action='store_true',
                    help='Check environment before proceeding')
parser.add_argument('--init-csv', action='store_true',
                    help='Initialize CSV if not exists')

# 新增函数
def check_environment():
    """检查 PyPTO 环境"""
    checks = {
        'csv_exists': Path(args.csv).exists(),
        'pypto_import': check_pypto_import(),
        'npu_available': check_npu_available(),
    }
    return checks

def init_csv_if_needed():
    """初始化 CSV 文件"""
    if not Path(args.csv).exists():
        # 创建目录
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        # 创建空 CSV
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
        return True
    return False
```

#### 3.1.2 修改 SKILL.md

更新 Step 1 流程：

```markdown
### Step 1：前置检查（环境 + 并发控制 + 中断恢复）

```bash
# 1a. 环境检查
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --check-env

# 处理退出码：
# exit 0: 环境正常 → 继续
# exit 3: CSV 不存在 → 执行 1b
# exit 4: PyPTO 环境不完整 → 警告并记录，继续
# exit 5+: 其他错误 → 终止

# 1b. 初始化 CSV（如果需要）
if [ $? -eq 3 ]; then
    python .agents/skills/pypto-op-autodev/scripts/update_op.py \
      --init-csv --csv autodev/scan_results.csv
fi

# 1c. 重置过期任务
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --reset-stale --timeout-hours 6
```
```

**预期效果**:
- 提前发现环境问题
- 自动初始化 CSV
- 减少后续步骤失败

**工作量**: 1-2 小时

---

### 3.2 优化 2: requirements.md 容错处理

**目标**: 优雅处理 requirements.md 不存在的情况

**实施方案**:

#### 3.2.1 修改 SKILL.md

```markdown
### Step 2a：检查用户新需求（可选）

**前置条件**: 检查 `autodev/sources/requirements.md` 是否存在

```bash
if [ -f "autodev/sources/requirements.md" ]; then
    # 读取并处理需求
    python .agents/skills/pypto-op-autodev/scripts/process_requirements.py \
      --csv autodev/scan_results.csv \
      --requirements autodev/sources/requirements.md
else
    echo "ℹ️  未发现 requirements.md，跳过用户需求检查"
    echo "   提示: 可创建 autodev/sources/requirements.md 添加自定义需求"
fi
```

**requirements.md 格式示例**:

```markdown
# 自定义算子需求

## 1. 算子名称: my_custom_op
- 复杂度: medium
- 类别: activation
- 来源: manual
- 备注: 用于特定模型优化

## 2. 算子名称: fused_attention
- 复杂度: hard
- 类别: attention
- 来源: manual
```
```

#### 3.2.2 创建示例文件

```bash
# 创建示例文件
cat > autodev/sources/requirements.md.example << 'EOF'
# 自定义算子需求

此文件用于手动添加算子开发需求。
格式：每个需求以 "## N. 算子名称: {op_name}" 开头

示例：

## 1. 算子名称: my_custom_op
- 复杂度: medium
- 类别: activation
- 来源: manual
- 备注: 用于特定模型优化
EOF
```

**预期效果**:
- 明确说明跳过原因
- 提供使用指引
- 减少用户困惑

**工作量**: 0.5 小时

---

### 3.3 优化 3: 脚本错误信息标准化

**目标**: 所有脚本输出统一格式的错误信息

**实施方案**:

#### 3.3.1 定义错误码规范

在 `scripts/data/error_codes.py` 中定义：

```python
"""错误码规范"""

ERROR_CODES = {
    0: "SUCCESS",
    1: "BUSY",              # 有算子正在开发中
    2: "PARAM_ERROR",       # 参数错误
    3: "CSV_NOT_FOUND",     # CSV 文件不存在
    4: "PYPTO_ENV_ERROR",   # PyPTO 环境不完整
    5: "NETWORK_ERROR",     # 网络错误
    6: "PERMISSION_ERROR",  # 权限错误
    99: "UNKNOWN_ERROR",    # 未知错误
}

ERROR_MESSAGES = {
    "CSV_NOT_FOUND": {
        "message": "CSV file not found",
        "suggestion": "Run 'update_op.py --init-csv' to initialize",
        "doc_link": "https://pypto.readthedocs.io/en/latest/autodev.html#csv-init"
    },
    "PYPTO_ENV_ERROR": {
        "message": "PyPTO environment check failed",
        "suggestion": "Run 'pypto-diagnose' to check installation",
        "doc_link": "https://pypto.readthedocs.io/en/latest/installation.html"
    }
}
```

#### 3.3.2 统一错误输出格式

所有脚本在失败时输出 JSON：

```python
def output_error(error_code, details=None):
    """统一错误输出"""
    error_name = ERROR_CODES.get(error_code, "UNKNOWN_ERROR")
    error_info = ERROR_MESSAGES.get(error_name, {})
    
    output = {
        "error": True,
        "error_code": error_name,
        "exit_code": error_code,
        "message": error_info.get("message", "Unknown error"),
        "suggestion": error_info.get("suggestion", "Check documentation"),
        "doc_link": error_info.get("doc_link"),
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    print(json.dumps(output, indent=2), file=sys.stderr)
    sys.exit(error_code)
```

#### 3.3.3 更新所有脚本

在 `select_next_op.py`, `add_op.py`, `update_op.py` 等脚本中使用统一错误输出：

```python
# 示例：select_next_op.py
if not Path(args.csv).exists():
    output_error(
        error_code=3,  # CSV_NOT_FOUND
        details={
            "csv_path": args.csv,
"expected_location": "autodev/scan_results.csv"
        }
    )
```

**预期效果**:
- 错误信息结构化
- 便于自动化处理
- 提供解决建议

**工作量**: 1-2 小时

---

### 3.4 优化 4: discover 触发时机前置

**目标**: 避免 Step 2b 失败后才触发 discover

**实施方案**:

#### 3.4.1 修改 SKILL.md

```markdown
### Step 1d：CSV 状态检查（新增）

在 Step 1 最后，检查 CSV 状态：

```bash
# 检查 CSV 是否有候选算子
python .agents/skills/pypto-op-autodev/scripts/check_csv_status.py \
  --csv autodev/scan_results.csv

# 处理退出码：
# exit 0: 有足够候选 → 继续 Step 2
# exit 1: 候选不足 → 触发 discover
# exit 3: CSV 不存在 → 先初始化，再 discover

if [ $? -eq 1 ] || [ $? -eq 3 ]; then
    echo "📥 候选算子不足，触发 discover..."
    # 调用 discover
fi
```

### Step 2：选择算子（简化）

**前置条件**: CSV 已存在且有候选

```bash
python .agents/skills/pypto-op-autodev/scripts/select_next_op.py \
  --csv autodev/scan_results.csv

# 此时 CSV 必然存在且有候选，exit 1 仅表示需要 discover
# 但已在 Step 1d 处理，此处 exit 1 应视为异常
```
```

#### 3.4.2 创建 `check_csv_status.py`

```python
#!/usr/bin/env python3
"""check_csv_status.py - 检查 CSV 状态"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.csv_ops import read_csv

MIN_CANDIDATES = 3  # 最少候选数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(json.dumps({"status": "csv_not_found", "candidates": 0}))
        sys.exit(3)

    rows = read_csv(args.csv)
    candidates = [r for r in rows if r.get("status") in ("pending", "failed")]

    if len(candidates) < MIN_CANDIDATES:
        print(json.dumps({
            "status": "insufficient",
            "candidates": len(candidates),
            "min_required": MIN_CANDIDATES
        }))
        sys.exit(1)

    print(json.dumps({
        "status": "ok",
        "candidates": len(candidates)
    }))
    sys.exit(0)
```

**预期效果**:
- 减少无效调用
- 流程更清晰
- 便于调试

**工作量**: 0.5-1 小时

---

### 3.5 优化 5: PyPTO 环境预检查

**目标**: 在开发前检测 PyPTO 环境问题

**实施方案**:

#### 3.5.1 创建环境检查脚本

`scripts/check_pypto_env.py`:

```python
#!/usr/bin/env python3
"""check_pypto_env.py - PyPTO 环境检查"""

import sys
import subprocess
from pathlib import Path

CHECKS = [
    {
        "name": "pypto_import",
        "description": "Import pypto module",
        "command": "import pypto",
        "critical": True
    },
    {
        "name": "shmem_tensor",
        "description": "Import ShmemTensor",
        "command": "from pypto.pypto_impl import ShmemTensor",
        "critical": True
    },
    {
        "name": "cann_version",
        "description": "Check CANN version",
        "command": "import os; print(os.environ.get('ASCEND_HOME_PATH', 'Not set'))",
        "critical": False
    },
    {
        "name": "npu_available",
        "description": "Check NPU availability",
        "command": "import subprocess; result = subprocess.run(['npu-smi', 'info'], capture_output=True); exit(result.returncode)",
        "critical": False
    }
]

def run_check(check):
    """执行单个检查"""
    try:
        result = subprocess.run(
            ["python", "-c", check["command"]],
            capture_output=True,
            timeout=5
        )
        return {
            "name": check["name"],
            "description": check["description"],
            "passed": result.returncode == 0,
            "critical": check["critical"],
            "output": result.stdout.decode() if result.stdout else None,
            "error": result.stderr.decode() if result.stderr else None
        }
    except Exception as e:
        return {
            "name": check["name"],
            "description": check["description"],
            "passed": False,
            "critical": check["critical"],
            "error": str(e)
        }

def main():
    results = [run_check(check) for check in CHECKS]
    
    critical_failures = [r for r in results if not r["passed"] and r["critical"]]
    
    output = {
        "all_passed": len(critical_failures) == 0,
        "critical_failures": len(critical_failures),
        "results": results,
        "recommendation": None
    }
    
    if critical_failures:
        output["recommendation"] = (
            "PyPTO 环境不完整，运行时验证将被跳过。"
            "建议: 修复 PyPTO 安装或使用 sim 模式。"
        )
    
    print(json.dumps(output, indent=2))
    
    if critical_failures:
        sys.exit(4)  # PYPTO_ENV_ERROR
    sys.exit(0)

if __name__ == "__main__":
    main()
```

#### 3.5.2 集成到 SKILL.md

```markdown
### Step 1e：PyPTO 环境检查（新增）

```bash
python .agents/skills/pypto-op-autodev/scripts/check_pypto_env.py

# 处理退出码：
# exit 0: 环境正常 → 继续
# exit 4: 环境不完整 → 记录警告，设置 ENV_MODE=sim

if [ $? -eq 4 ]; then
    echo "⚠️  PyPTO 环境不完整，运行时验证将被跳过"
    export AUTODEV_SKIP_RUNTIME_VERIFY=1
    echo "   已设置 AUTODEV_SKIP_RUNTIME_VERIFY=1"
fi
```
```

**预期效果**:
- 提前发现问题
- 避免 1 小时开发后失败
- 自动降级到 sim 模式

**工作量**: 1 小时

---

### 3.6 优化 6: CSV 初始化脚本

**目标**: 提供独立的 CSV 初始化工具

**实施方案**:

#### 3.6.1 创建 `init_csv.py`

```python
#!/usr/bin/env python3
"""init_csv.py - 初始化 CSV 文件"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.csv_ops import FIELDS

DEFAULT_OPS = [
    # 可选：预置一些基础算子
    # {"op_name": "add", "complexity": "easy", "category": "binary", "source": "builtin"},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--with-defaults", action="store_true",
                       help="Include default operators")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    
    if csv_path.exists() and not args.force:
        print(json.dumps({
            "status": "already_exists",
            "csv_path": str(csv_path),
            "message": "CSV already exists. Use --force to overwrite."
        }))
        sys.exit(0)

    # 创建目录
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 CSV
    rows = []
    if args.with_defaults:
        for op in DEFAULT_OPS:
            row = {f: "" for f in FIELDS}
            row.update(op)
            row["status"] = "pending"
            row["create_time"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "initialized",
        "csv_path": str(csv_path),
        "rows_added": len(rows),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**预期效果**:
- 独立初始化工具
- 支持预置算子
- 便于调试

**工作量**: 1 小时

---

### 3.7 优化 7: 流程状态持久化

**目标**: 支持断点续传和执行监控

**实施方案**:

#### 3.7.1 定义状态文件格式

`operators/.autodev_state.json`:

```json
{
  "version": "1.0",
  "session_id": "autodev-20260327-102958",
  "current_op": "softmax",
  "start_time": "2026-03-27T02:29:58",
  "last_update": "2026-03-27T02:47:00",
  "status": "completed",
  "checkpoints": {
    "step_1": {
      "status": "completed",
      "timestamp": "2026-03-27T02:29:58",
      "output": {"reset_ops": [], "has_active": false}
    },
    "step_2a": {
      "status": "skipped",
      "timestamp": "2026-03-27T02:30:00",
      "reason": "requirements.md not found"
    },
    "step_2b": {
      "status": "completed",
      "timestamp": "2026-03-27T02:30:05",
      "output": {"op_name": "softmax", "score": 85.0}
    },
    "step_3": {
      "status": "completed",
      "timestamp": "2026-03-27T02:42:00",
      "dev_result": "SUCCESS",
      "artifacts": ["spec.md", "design.md", "softmax_impl.py", ...]
    },
    "step_4": {
      "status": "completed",
      "timestamp": "2026-03-27T02:45:00",
      "fps_total": 2,
      "fps_confirmed": 1
    },
    "step_5": {
      "status": "completed",
      "timestamp": "2026-03-27T02:46:00",
      "issues_generated": 2
    },
    "step_6": {
      "status": "completed",
      "timestamp": "2026-03-27T02:47:00"
    }
  },
  "environment": {
    "pypto_ok": false,
    "skip_runtime_verify": true,
    "cann_version": "8.5.0",
    "server_type": "A3"
  },
  "errors": [
    {
      "step": "step_3",
      "type": "runtime_error",
      "message": "ShmemTensor not found",
      "handled": true
    }
  ]
}
```

#### 3.7.2 创建状态管理脚本

`scripts/state_manager.py`:

```python
#!/usr/bin/env python3
"""state_manager.py - 流程状态管理"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("operators/.autodev_state.json")

def load_state():
    """加载状态"""
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        return json.load(f)

def save_state(state):
    """保存状态"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def init_state(session_id):
    """初始化状态"""
    return {
        "version": "1.0",
        "session_id": session_id,
        "current_op": None,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checkpoints": {},
        "environment": {},
        "errors": []
    }

def update_checkpoint(step, status, output=None, reason=None):
    """更新检查点"""
    state = load_state() or init_state(f"autodev-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    state["checkpoints"][step] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": output,
        "reason": reason
    }
    save_state(state)

def add_error(step, error_type, message, handled=True):
    """记录错误"""
    state = load_state()
    if state:
        state["errors"].append({
            "step": step,
            "type": error_type,
            "message": message,
            "handled": handled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        save_state(state)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True,
                       choices=["init", "checkpoint", "error", "status"])
    parser.add_argument("--step", help="Step name")
    parser.add_argument("--status", help="Step status")
    parser.add_argument("--output", help="JSON output")
    parser.add_argument("--reason", help="Skip reason")
    args = parser.parse_args()

    if args.action == "init":
        state = init_state(f"autodev-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        save_state(state)
        print(json.dumps({"status": "initialized", "state_file": str(STATE_FILE)}))
    
    elif args.action == "checkpoint":
        output = json.loads(args.output) if args.output else None
        update_checkpoint(args.step, args.status, output, args.reason)
        print(json.dumps({"status": "updated"}))
    
    elif args.action == "error":
        add_error(args.step, "error", args.output or "Unknown error")
        print(json.dumps({"status": "recorded"}))
    
    elif args.action == "status":
        state = load_state()
        print(json.dumps(state, indent=2))

if __name__ == "__main__":
    main()
```

#### 3.7.3 集成到 SKILL.md

```markdown
### Step 0：状态初始化（新增）

```bash
python .agents/skills/pypto-op-autodev/scripts/state_manager.py --action init
```

### Step 1-6：每个步骤后更新状态

```bash
# 示例：Step 2b 完成后
python .agents/skills/pypto-op-autodev/scripts/state_manager.py \
  --action checkpoint \
  --step step_2b \
  --status completed \
  --output '{"op_name": "softmax", "score": 85.0}'
```

### 查询当前状态

```bash
python .agents/skills/pypto-op-autodev/scripts/state_manager.py --action status
```
```

**预期效果**:
- 支持断点续传
- 便于监控和调试
- 可生成执行报告

**工作量**: 2-3 小时

---

## 四、实施优先级

| 优化项 | 优先级 | 理由 | 预计工作量 | 实施顺序 |
|--------|--------|------|-----------|---------|
| 优化 3: 错误信息标准化 | **P0** | 直接影响调试效率，阻塞问题定位 | 1-2h | 1 |
| 优化 4: discover 前置 | **P0** | 避免无效调用，提升效率 | 0.5-1h | 2 |
| 优化 2: requirements.md 容错 | **P1** | 提升健壮性，减少用户困惑 | 0.5h | 3 |
| 优化 5: PyPTO 环境预检查 | **P1** | 提前发现问题，避免时间浪费 | 1h | 4 |
| 优化 1: 前置检查增强 | **P1** | 避免后续步骤失败 | 1-2h | 5 |
| 优化 6: CSV 初始化脚本 | **P2** | 优化初始化流程，非必须 | 1h | 6 |
| 优化 7: 流程状态持久化 | **P2** | 支持断点续传，增强功能 | 2-3h | 7 |

**总工作量**: 7-11 小时

---

## 五、实施建议

### 5.1 分阶段实施

**阶段 1（立即修复，1-2 天）**:
- 优化 3: 错误信息标准化
- 优化 4: discover 前置

**阶段 2（短期优化，3-5 天）**:
- 优化 2: requirements.md 容错
- 优化 5: PyPTO 环境预检查
- 优化 1: 前置检查增强

**阶段 3（长期增强，1-2 周）**:
- 优化 6: CSV 初始化脚本
- 优化 7: 流程状态持久化

### 5.2 测试策略

每个优化实施后，应测试以下场景：

1. **正常流程**: CSV 存在，环境正常
2. **CSV 不存在**: 从零开始初始化
3. **requirements.md 不存在**: 跳过用户需求
4. **PyPTO 环境不完整**: 自动降级
5. **网络错误**: discover 失败
6. **并发执行**: 多个 autodev 同时运行

### 5.3 文档更新

实施后需更新以下文档：

1. **SKILL.md**: 更新流程描述
2. **README.md**: 添加故障排查指南
3. **新增文档**:
   - `docs/autodev-troubleshooting.md`: 故障排查
   - `docs/autodev-state-format.md`: 状态文件格式

---

## 六、验收标准

### 6.1 功能验收

- [ ] CSV 不存在时自动初始化
- [ ] requirements.md 不存在时优雅跳过
- [ ] PyPTO 环境不完整时自动降级
- [ ] 所有脚本输出标准化错误信息
- [ ] 支持断点续传（可选）

### 6.2 性能验收

- [ ] Step 1 完成时间 < 5 秒
- [ ] 环境检查时间 < 3 秒
- [ ] 错误定位时间减少 50%

### 6.3 用户体验验收

- [ ] 错误信息清晰易懂
- [ ] 提供明确的解决建议
- [ ] 无需手动创建文件或目录

---

## 七、风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 脚本兼容性破坏 | 中 | 低 | 保持向后兼容，增加参数而非修改现有行为 |
| 环境检查误报 | 中 | 中 | 提供跳过检查的选项，记录检查结果 |
| 状态文件损坏 | 低 | 低 | 定期备份，提供恢复工具 |

---

## 八、总结

**核心问题**: autodev 流程假设环境已就绪，缺少初始化和容错机制

**根本原因**:
1. CSV 文件不存在时无降级策略
2. 脚本错误信息不透明
3. 环境检查滞后

**优化目标**:
1. **健壮性**: 自动处理各种异常情况
2. **透明性**: 提供清晰的错误信息和解决建议
3. **效率**: 提前检查，避免无效操作
4. **可观测性**: 支持状态监控和断点续传

**预期收益**:
- 减少 50% 的调试时间
- 提升用户体验
- 降低新手上手难度
- 减少重复问题报告

---

## 附录 A: 错误码完整列表

| 错误码 | 名称 | 含义 | 处理建议 |
|--------|------|------|---------|
| 0 | SUCCESS | 成功 | - |
| 1 | BUSY | 有算子正在开发中 | 等待或重置 |
| 2 | PARAM_ERROR | 参数错误 | 检查命令参数 |
| 3 | CSV_NOT_FOUND | CSV 文件不存在 | 运行 --init-csv |
| 4 | PYPTO_ENV_ERROR | PyPTO 环境不完整 | 检查 PyPTO 安装 |
| 5 | NETWORK_ERROR | 网络错误 | 检查网络连接 |
| 6 | PERMISSION_ERROR | 权限错误 | 检查文件权限 |
| 99 | UNKNOWN_ERROR | 未知错误 | 查看日志或提交 Issue |

---

## 附录 B: 状态文件检查点列表

| 检查点 | 含义 | 必需输出 |
|--------|------|---------|
| step_1 | 前置检查 | `{has_active: bool}` |
| step_2a | 用户需求检查 | `{requirements: []}` |
| step_2b | 选择算子 | `{op_name: string, score: number}` |
| step_2c | discover 补充 | `{ops_added: number}` |
| step_3 | 算子开发 | `{dev_result: string, artifacts: []}` |
| step_4 | 断裂点检测 | `{fps_total: number, fps_confirmed: number}` |
| step_5 | Issue 生成 | `{issues_generated: number}` |
| step_6 | 状态更新 | `{final_status: string}` |

---

## 附录 C: 参考资料

1. [PyPTO 官方文档](https://pypto.readthedocs.io/)
2. [GitCode Issue 规范](https://gitcode.com/cann/pypto/blob/master/CONTRIBUTING.md)
3. [断裂点检测设计文档](../../docs/plans/2026-03-26-fracture-point-scanner-design.md)

---

**文档版本历史**:

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v1.0 | 2026-03-27 | Claude | 初始版本，基于 softmax 算子开发实践 |
