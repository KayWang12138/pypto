# PyPTO Pass 精度验证 - 配置指南

本文档介绍 PyPTO Pass 精度验证所需的配置项及其设置方法。

---

## 配置对比总览

| 配置类型 | 配置文件/参数 | 关键配置项 | 精度问题时设置 | 说明 |
|---------|-------------|----------|--------------|------|
| **verify_options** | 算子实现文件 | `enable_pass_verify` | **True** | 启用Pass验证（必须） |
| | | `pass_verify_pass_filter` | "all" | 验证所有Pass |
| | | `pass_verify_save_tensor` | **True** | 保存中间数据 |
| **tile_fwk_config.json** | framework配置文件 | `pre_check` | **true** | Pass前校验 |
| | | `post_check` | **true** | Pass后校验 |
| | | `dump_graph` | true | 保存IR图 |
| | | `print_graph` | true | 打印IR图 |
| **环境变量** | Shell环境 | `ASCEND_WORK_PATH` | 必须设置 | 工作目录路径 |
| | | `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为0 | DEBUG级别日志 |

---

## verify_options 配置

在 PyPTO 算子实现文件中配置：

### 配置示例

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_pass_filter": "all",      # 可选：Pass 侧有精度问题时设置
    "pass_verify_save_tensor": True,      # 可选：Pass 侧存在精度问题时设置
}
@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    output: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1
```

### 配置项说明

| 配置项 | 说明 | 使用时机 | 默认值 |
|-------|------|---------|-------|
| `enable_pass_verify` | 启用 Pass 验证 | 必须设置 | `False` |
| `pass_verify_pass_filter` | 过滤要验证的 Pass | Pass 侧有精度问题时设置为 `"all"` | `[]` |
| `pass_verify_save_tensor` | 保存 Pass 中间数据 | Pass 侧存在精度问题时设置为 `True` | `False` |

### 注意事项

1. **Shape 必须具体**：必须使用具体的 shape 值，不能使用占位符
2. **输出参数形式**：返回类型不能使用 `-> pypto.Tensor(...)` 语法，必须使用输出参数
3. **Golden 设置时机**：`pypto.set_verify_golden_data()` 必须在算子调用**之前**执行

---

## tile_fwk_config.json 配置

用于启用 PreCheck 和 PostCheck。

### 配置文件位置

`framework/src/interface/configs/tile_fwk_config.json`

### 配置示例

```json
{
    "pass": {
        "enable_binary_cache": false,
        "enable_pass_configs": true,
        "enable_cv_fuse": false,
        "pass_thread_num": 1,
        "vf_opt_mark_for": false,
        "enable_vf": true,
        "default_pass_configs": {
            "print_graph": true,
            "print_program": false,
            "dump_graph": true,
            "dump_pass_time_cost": true,
            "pre_check": false,     // Pass 精度问题时设为 true
            "post_check": false,    // Pass 精度问题时设为 true
            "expected_value_check": false,
            "disable_pass": false,
            "health_check": false,
            "use_max_freq_label": false
        }
    }
}
```

### PreCheck/PostCheck 说明

| 配置项 | 说明 | 使用场景 |
|-------|------|---------|
| `pre_check` | Pass 执行前校验 | Pass 侧有精度问题时开启 |
| `post_check` | Pass 执行后校验 | Pass 侧有精度问题时开启 |
| `dump_graph` | 保存 IR 图 | 帮助分析 Pass 处理结果 |
| `print_graph` | 打印 IR 图 | 快速查看图结构变化 |

---

## 二分 CCE 调试配置

使用二分 CCE 方法时，需要额外配置：

### tile_fwk_config.json 二分配置

```json
{
    "codegen": {
        "fixed_output_path": true,      // 固定CCE输出路径
        "force_overwrite": false,       // 不覆盖已修改的CCE文件
        "parallel_compile": 1           // 单线程编译
    }
}
```

### 配置说明

| 配置项 | 正确值 | 说明 |
|-------|-------|------|
| `fixed_output_path` | `true` | CCE 固定生成在 `./kernel_aicore/` |
| `force_overwrite` | `false` | 不覆盖手动修改的 CCE 文件 |
| `parallel_compile` | `1` | 单线程编译，便于调试 |

### aicore_print.h 打印开关

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：

```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

---

## 配置备份与恢复

### 配置备份

在修改配置前，建议先备份原始配置：

```bash
# 备份 tile_fwk_config.json
cp framework/src/interface/configs/tile_fwk_config.json \
   framework/src/interface/configs/tile_fwk_config.json.backup

# 验证备份文件
ls -lh framework/src/interface/configs/tile_fwk_config.json.backup
```

### 配置恢复

调试完成后，恢复原始配置：

```bash
# 恢复 tile_fwk_config.json
cp framework/src/interface/configs/tile_fwk_config.json.backup \
   framework/src/interface/configs/tile_fwk_config.json

# 验证恢复
diff framework/src/interface/configs/tile_fwk_config.json.backup \
     framework/src/interface/configs/tile_fwk_config.json
```

### 配置修改建议

| 建议 | 说明 |
|------|------|
| 每次修改前备份 | 避免配置丢失 |
| 记录修改内容 | 便于问题排查 |
| 测试完成后恢复 | 避免影响后续工作 |
| 使用版本控制 | 推荐使用 git 管理配置文件 |

---

## 快速配置脚本

一键配置所有精度验证开关：

```bash
#!/bin/bash
CONFIG_FILE="framework/src/interface/configs/tile_fwk_config.json"

# 备份原配置
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

# 开启 PreCheck/PostCheck（需要 jq 工具）
jq '.global.pass.default_pass_configs.pre_check = true' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
jq '.global.pass.default_pass_configs.post_check = true' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"

echo "配置已更新，原配置已备份到 ${CONFIG_FILE}.backup"
```

---

## 相关文档

- 主流程文档：[../SKILL.md](../SKILL.md)
- 环境配置：[environment_setup.md](./environment_setup.md)
- CCE打印指南：[binary_cce.md](./binary_cce.md)