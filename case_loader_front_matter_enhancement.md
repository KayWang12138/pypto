# 需求文档：case_loader.py SPEC.md Front Matter 增强

**需求编号**: REQ-2026-04-001  
**优先级**: P0（最高）  
**目标文件**: `integration/benchmark/case_loader.py`  

---

## 1. 问题概述

### 1.1 问题现象

KernelBench 批量算子开发测试中，多个算子因 SPEC.md 缺少 YAML front matter 触发门禁阻断（OL39/40），导致流程无法推进。

**受影响算子**：
- GroupNorm（Stage 5 阻断，retry 4次后超时）
- Softsign（Stage 5 阻断，触发 SPEC.md 冻结冲突）
- Product_reduction、LayerNorm、Average_Pooling_2D、Softmax（Stage 1 阻断）

**成功率影响**：10个算子中4个失败，失败率40%。

### 1.2 根因分析

**核心问题**：KernelBench 自动生成的 SPEC.md 与 PyPTO 门禁标准不一致。

| 对比维度 | KernelBench case_loader.py 输出 | PyPTO 门禁要求 | 冲突结果 |
|---------|--------------------------------|---------------|---------|
| front matter | **无**（只有普通 markdown） | **必须有**（OL39/40） | Stage 5 阻断 |
| 检查时序 | Stage 1-2 只检查章节结构（OL09） | Stage 5 才检查 front matter | 时序不一致 |
| SPEC 冻结 | Stage 1 记录 hash（无 front matter） | Stage 5 添加 front matter → hash 变化 → 冻结违规 | 无法修复 |

**制度性缺陷**：
- SPEC.md 在 `complete_stage(1)` 时记录 hash（无 front matter）
- Stage 5 门禁要求 front matter → Subagent 添加 → hash 变化 → 触发冻结违规
- 导致算子 BLOCKED，无法通过 retry 解决

### 1.3 OL39/40 检查范围

门禁检查三个文档的 front matter：**SPEC.md、DESIGN.md、API_REPORT.md**。

实际触发报错的只有 **SPEC.md**：
- DESIGN.md：由 pypto-op-analyst skill 生成，**已包含 front matter** ✓
- API_REPORT.md：由 pypto-api-explore skill 生成，**已包含 front matter** ✓
- SPEC.md：由 case_loader.py 生成，**无 front matter** ❌

**结论**：只需修改 case_loader.py，无需修改其他 skill 模板。

---

## 2. 需求内容

### 2.1 核心需求

修改 `integration/benchmark/case_loader.py`，使生成的 SPEC.md 自动包含完整的 YAML front matter。

### 2.2 Front Matter 必需字段

```yaml
---
schema_version: 1
op_name: {op_name}
supported_dtypes: [{dtype_list}]
p0_shapes: [{shape_list}]
tolerance: {rtol: xxx, atol: xxx}
---
```

| 字段 | 来源 | 说明 |
|------|------|------|
| `schema_version` | 固定值 `1` | 文档版本号 |
| `op_name` | CaseSpec.op_name | 算子名称 |
| `supported_dtypes` | 从 inputs.dtype 提取 | 如 `["float32"]` |
| `p0_shapes` | 从 inputs.shape 提取 | 如 `[[16, 256, 256]]` |
| `tolerance` | 根据 dtype 计算 | FP32: `{rtol: 1e-3, atol: 1e-3}` |

### 2.3 字段提取规则

**supported_dtypes**：
```python
# 从 inputs 提取 dtype（去重）
supported_dtypes = list(set(spec.dtype for spec in case.inputs if spec.dtype))
# 降级：探针失败时默认 ["float32"]
```

**p0_shapes**：
```python
# 从 inputs 提取 shape
p0_shapes = [list(spec.shape) for spec in case.inputs if spec.shape]
# 降级：探针失败时默认 []
```

**tolerance**：
```python
# 根据 dtype 设置
if any(dt in ["float16", "bfloat16"] for dt in supported_dtypes):
    tolerance = {"rtol": 4e-3, "atol": 4e-3}  # FP16/BF16 放宽精度
else:
    tolerance = {"rtol": 1e-3, "atol": 1e-3}  # FP32 标准精度
```

---

## 3. 实现方案

### 3.1 修改位置

| 位置 | 行号 | 修改内容 |
|------|------|---------|
| CaseSpec | 66-77 | 新增字段：supported_dtypes, p0_shapes, tolerance |
| _SPEC_TEMPLATE | 300-372 | 开头插入 YAML front matter |
| render_spec_md() | 386-398 | 扩展渲染逻辑，填充 front matter 字段 |

### 3.2 CaseSpec 扩展

```python
class CaseSpec:
    """KernelBench 用例派生的结构化规格."""
    op_name: str
    case_id: str
    source_file: str
    task_desc: str
    framework_module: str = "torch"
    init_source: str = ""
    forward_source: str = ""
    init_args_repr: str = "[]"
    inputs: List[TensorSpec] = field(default_factory=list)
    # 新增字段
    supported_dtypes: List[str] = field(default_factory=lambda: ["float32"])
    p0_shapes: List[List[int]] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=lambda: {"rtol": 1e-3, "atol": 1e-3})
```

### 3.3 _SPEC_TEMPLATE 修改

```python
_SPEC_TEMPLATE = """\
---
schema_version: 1
op_name: {op_name}
supported_dtypes: {supported_dtypes_json}
p0_shapes: {p0_shapes_json}
tolerance: {tolerance_json}
---

# {op_name} 算子需求规格 (派生自上游 KernelBench)

> 本 SPEC 由 ``integration.benchmark.case_loader`` 自动生成...
"""
```

### 3.4 render_spec_md() 扩展

```python
import json  # 需在文件开头导入

def render_spec_md(case: CaseSpec) -> str:
    """把 CaseSpec 渲染成 SPEC.md 文本."""
    # 提取 front matter 字段
    supported_dtypes = list(set(spec.dtype for spec in case.inputs if spec.dtype)) or ["float32"]
    p0_shapes = [list(spec.shape) for spec in case.inputs if spec.shape] if case.inputs else []
    
    # 计算 tolerance
    has_low_precision = any(dt in ["float16", "bfloat16"] for dt in supported_dtypes)
    tolerance = {"rtol": 4e-3, "atol": 4e-3} if has_low_precision else {"rtol": 1e-3, "atol": 1e-3}
    
    return _SPEC_TEMPLATE.format(
        supported_dtypes_json=json.dumps(supported_dtypes),
        p0_shapes_json=json.dumps(p0_shapes),
        tolerance_json=json.dumps(tolerance),
        op_name=case.op_name,
        # ... 其他原有字段
    )
```

---

## 4. 验收标准

### 4.1 功能验收

| 验收项 | 验收标准 |
|--------|---------|
| SPEC.md 包含 front matter | 生成的文件以 `---` 开头和结尾 |
| 字段完整 | 包含 schema_version, op_name, supported_dtypes, p0_shapes, tolerance |
| dtype 提取正确 | supported_dtypes 与 inputs.dtype 一致 |
| shape 提取正确 | p0_shapes 与 inputs.shape 一致 |
| tolerance 设置正确 | FP32: rtol=1e-3; FP16/BF16: rtol=4e-3 |
| 降级逻辑正确 | 探针失败时：supported_dtypes=["float32"], p0_shapes=[] |

### 4.2 门禁验收

| 验收项 | 验收标准 |
|--------|---------|
| OL09 通过 | Stage 1 无阻断 |
| OL39 通过 | Stage 5 front matter 检查通过 |
| OL40 通过 | Stage 5 字段完整性检查通过 |
| SPEC 冻结不冲突 | Stage 1 → Stage 5 流程正常推进 |

### 4.3 集成验收

- 已验证的 6 个算子（ReLU, Softmax 等）重新生成 SPEC.md 后流程正常
- 失败算子（GroupNorm, Softsign）修复后 Stage 5 门禁通过

---

## 5. 生成的 SPEC.md 示例

```yaml
---
schema_version: 1
op_name: Softmax
supported_dtypes: ["float32"]
p0_shapes: [[16, 256, 256]]
tolerance: {"rtol": 0.001, "atol": 0.001}
---

# Softmax 算子需求规格 (派生自上游 KernelBench)

> 本 SPEC 由 ``integration.benchmark.case_loader`` 自动生成...
```