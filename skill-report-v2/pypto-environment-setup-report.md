# 技能评审报告

## 1. 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | `pypto-environment-setup` |
| 评审时间 | 2026-03-11 |
| 总分 | **99.30 / 100** |
| 等级 | **A** |
| S0 否决 | 否 |
| 规则统计 | PASS 48 / FAIL 2 / WARN 0 / SKIP 1 |

## 2. 维度评分表

| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细（仅 FAIL） |
|------|---------------|------|--------|----------------------|
| D1 Frontmatter 元数据 | 100 | 0.25 | 25.00 | 无 |
| D2 简洁性与效率 | 100 | 0.15 | 15.00 | 无 |
| D3 文件结构与导航 | 100 | 0.10 | 10.00 | 无 |
| D4 语言与表达 | 98 | 0.10 | 9.80 | R46(-2, S3) |
| D5 精确性与可执行性 | 95 | 0.10 | 9.50 | R25(-5, S2) |
| D6 工作流完整性 | 100 | 0.10 | 10.00 | 无 |
| D7 模式与最佳实践 | 100 | 0.05 | 5.00 | 无 |
| D8 反模式检测 | 100 | 0.10 | 10.00 | 无 |
| D9 脚本与代码质量 | 100 | 0.05 | 5.00 | 无 |

> 总分校验：25.00 + 15.00 + 10.00 + 9.80 + 9.50 + 10.00 + 5.00 + 10.00 + 5.00 = **99.30**

## 3. 规则覆盖率

- 静态规则：PASS 28 + FAIL 1 + SKIP 0 = **29**
- 语义规则：PASS 20 + FAIL 1 + SKIP 1 = **22**
- 总评估数：29 + 22 = **51**
- 覆盖率：51 / 51 × 100% = **100.00%**

状态拆分（全量 51 条）：

| 状态 | 数量 |
|------|------|
| PASS | 48 |
| FAIL | 2 |
| WARN | 0 |
| SKIP | 1 |

SKIP 规则：`R44`（原因：该技能 frontmatter 未设置 `context: fork`，规则不适用）

## 4. 质量门禁

| 过滤类型 | 数量 | 说明 |
|----------|------|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身而非目标 skill 的误绑证据 |
| low_information_snippet | 0 | 未发现无法在目标文件逐字匹配的语义证据 |

本次无被过滤条目。

## 5. 问题列表

### S2

#### 问题 1：命令存在未解析占位符，影响直接可执行性

- 命中规则：`[S2] R25`
- 位置：`.agents/skills/pypto-environment-setup/SKILL.md:70`
- 证据：`cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2|a3>`
- 问题描述：步骤 3 的关键安装命令使用 `<a2|a3>` 占位符，且未在该命令处给出“如何直接替换为实值”的执行写法，导致操作者复制命令后仍需二次人工改写。
- 修复建议（前后对比）：

```diff
- cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2|a3>
+ DEVICE_TYPE=$(python3 scripts/diagnose_env.py --json | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('npu_env',{}).get('device_type') or 'a3').replace('a3+','a3'))")
+ cd "$PYPTO_REPO" && bash tools/prepare_env.sh --quiet --type=deps --device-type="$DEVICE_TYPE"
```

### S3

#### 问题 2：存在未标注语言的围栏代码块

- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-environment-setup/SKILL.md:141`
- 证据：`**报告模板**：` 后紧跟 ````` ``（未带语言标识）
- 问题描述：报告模板代码块使用裸围栏，未声明语言类型，降低了渲染可读性和静态检查一致性。
- 修复建议（前后对比）：

```diff
- ```
+ ```text
```

## 6. 通过规则汇总

| 维度 | 通过规则 |
|------|----------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R51 |
| D8 | R34, R35, R36, R37, R38, R47 |
| D9 | R39, R40, R41, R42, R50 |
| D0 | R43, R43 |
