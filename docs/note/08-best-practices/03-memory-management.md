# 内存管理建议（工程化）

本页补充一些“工程上最常踩坑”的内存与生命周期相关建议，避免把问题留到运行期崩溃或精度异常才暴露。

---

## 1. 先用产物定位，再谈优化

- 先固定输出目录，查看 `run.log` 与相关产物：见 [输出文件说明](../03-mechanisms/output-files/README.md)（包含 [run.log 详细说明](../03-mechanisms/output-files/run-log.md)）
- 如果出现崩溃/内存问题，优先按调试流程走：见 [完整调试指南](../05-debugging/00-complete-guide.md)

---

## 2. 常见问题

- 张量生命周期不清晰导致的“使用未初始化/越界写”
- view/assemble 边界处理不当导致的边界 tile 覆盖写回


