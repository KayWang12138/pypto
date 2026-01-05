# 测试结果与定位建议

本页用于说明：当你在本地或 CI 中运行 `python/tests` 之后，应该如何查看“结果是否通过”、以及失败时从哪里开始定位。

---

## 1. 本地运行（推荐先从这里开始）

```bash
cd /path/to/pypto
python3 -m pytest -q python/tests
```

常见输出含义：

- **全部通过**：pytest 返回码为 0
- **存在失败/错误**：pytest 会列出失败用例与 traceback

定位建议（从易到难）：

1. 先确认环境一致性（Python / PyTorch / torch_npu / CANN）
2. 固定输出目录并查看 `run.log`（见：`docs/note/03-mechanisms/output-files/run-log.md`）
3. 如果是崩溃类问题，按调试流程走（见：`docs/note/05-debugging/00-complete-guide.md`）

---

## 2. CI 结果怎么看？

不同仓库托管平台的 CI 展示略有差异，但通用信息包括：

- 失败的 job 名称（例如 build/test/lint）
- 失败的命令（通常是 `pytest ...` 或构建脚本）
- 失败日志（包含报错栈、运行参数、部分环境信息）

建议做法：

- 先把失败用例在本地**最小化复现**（固定输入/固定 dtype/固定输出目录）
- 再对照 `run.log` 与产物（见：[输出目录与产物总览](../03-mechanisms/output-files/README.md)）

---

## 3. 与其它文档的关系

- 测试方法论：见 [测试和验证方法](00-methodology.md)
- 常见问题：见 [常见问题与已知问题库](../05-debugging/03-troubleshooting-and-known-issues.md)

