# 框架配置与调试 (Framework Configuration)

本目录包含 PyPTO 框架层面的配置示例，主要涉及运行时调试和日志打印。

## 总览介绍

在算子开发过程中，合理的调试信息输出对于排查逻辑错误至关重要。本样例展示了如何配置 PyPTO 的全局打印选项。

## 关键文件说明

- **`utils/print.py`**: 展示了 `pypto.set_print_options` 的用法。
  - `edge_items`: 张量打印时边缘显示的元素数量。
  - `precision`: 浮点数显示的精度。
  - `threshold`: 张量元素总数超过该阈值时会进行截断显示。
  - `linewidth`: 每行打印的字符宽度限制。

## 运行方法

### 执行脚本

```bash
cd utils
python3 print.py
```

## 注意事项
- `set_print_options` 主要影响 PyPTO 运行时的内部调试输出和 IR（中间表示）层的信息，不直接改变 Python 层面的 `torch.Tensor` 打印行为。
- 在生产环境中，建议调低打印选项以减少不必要的日志开销。

