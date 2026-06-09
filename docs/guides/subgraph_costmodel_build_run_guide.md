# 子图 CostModel 编译运行指南

> 本文档提供**思路和方法**，不提供具体命令。每个环境构建方式不同，需要先摸清再动手。

---

## 一、前提：搞清楚对方的构建方式

对方的构建流程大概率跟你的环境不同。先回答以下问题：

| 问题 | 怎么查 |
|------|--------|
| 构建入口是什么？ | 找 `build*.py`、`build*.sh`、`Makefile`，或问对方开发者 |
| cmake 有哪些必要参数？ | 看已有构建脚本里的 cmake 命令，特别是 toolchain、CANN 路径等 |
| Python 绑定 target 是否需要开关启用？ | 查看是否有类似 `ENABLE_FEATURE_PYTHON_FRONT_END` 的 cmake 选项 |
| 编译产物放在哪？ | 找已有的 `.so` 文件位置，反推输出目录 |
| 对方平时怎么编译整图的？ | 看对方的历史操作或文档，照着来 |

---

## 二、只需要编两个东西

子图 costmodel 的代码改动涉及的编译目标只有两个：

| 源码改动 | 对应编译产物 | 作用 |
|---|---|---|
| backend.h/cpp、cost_model_launcher.h | simulation 动态库（`.so`） | 仿真核心，包含子图过滤逻辑 |
| cost_model.cpp（Python 绑定） | Python 绑定模块（`*_impl.so`） | Python 调 C++ 的桥梁 |

**不要编全部。** 找到对方环境里这两个 target 叫什么，只编它们。

查找方法：
- 看 cmake 定义：`grep -rn "add_library.*simulation" framework/` 和 `grep -rn "pybind11_add_module\|add_library.*impl" python/src/`
- 看已有产物：`find <编译输出> -name "*simulation*.so"` 和 `find <编译输出> -name "*_impl*.so"`

---

## 三、编译后做三件事

### 3.1 动态库路径

编译出的 `.so` 必须在运行时能被找到。把编译输出目录下所有含 `.so` 的子目录加入 `LD_LIBRARY_PATH`。

思路：`find <编译输出目录> -name "*.so"` 找到所有 `.so`，把它们所在目录全部加进去。不要猜，直接搜。

如果运行时报 `undefined symbol`，说明缺了某个 `.so` 目录。用 `c++filt` 解码符号名，再 `find + nm -C` 找到该符号在哪个 `.so` 里，把那个目录补上。

### 3.2 Python 绑定安装

编译出的 `*_impl.so` 不能只靠 `PYTHONPATH`，必须复制到 Python 包目录内部。

思路：找到 `python/<包名>/` 目录，把 `*_impl.so` 复制进去。验证方法是 `python3 -c "from <包名> import <包名>_impl; print('OK')"`。

### 3.3 环境变量

至少需要：
- **CANN 路径**：指向目标环境的 CANN 安装目录
- **设备 ID**：仿真模式也需要设置
- **动态库路径**：上面 3.1 说的所有 `.so` 目录
- **Python 路径**：`python/` 目录加入 `PYTHONPATH`

---

## 四、验证流程

编译和运行环境配好后，按顺序验证：

1. **import 验证**：`python3 -c "from <包名> import <包名>_impl"` → 不报错
2. **整图先跑通**：确保整图 costmodel 能正常工作（子图依赖整图的产物）
3. **子图测试**：调 `_cost_model_run_subgraph_line(inputs, outputs, p_sg_id=0)`，检查返回 `status == "success"`

子图 costmodel 必须在整图之后运行，因为它依赖整图生成的 `dyn_topo.txt` 和已填充的 `ProgramData`。

---

## 五、常见问题排查思路

| 现象 | 排查方向 |
|------|---------|
| `No rule to make target 'pto_impl'` | cmake 缺少启用 Python 前端的参数，重新配置 |
| 编译报头文件找不到 | 检查 `target_include_directories` 是否包含 `${CMAKE_SOURCE_DIR}/framework/src/cost_model` |
| `undefined symbol` | `LD_LIBRARY_PATH` 缺目录，按符号找 `.so` 补上 |
| `cannot import name` | `*_impl.so` 没复制到 Python 包目录 |
| `ConfigManagerNg` 崩溃 | 缺少编译配置文件，需要用对方的构建脚本生成 |
| `mismatch input/output` | `InitInputOutputData` 被误调了，确认已跳过 |
| `no compiled function found` | 没有先运行整图编译，子图依赖整图的编译产物 |

---

## 六、总结

**核心思路**：找到对方的构建方式 → 只编 simulation 库 + Python 绑定 → .so 找得到 + 装到位 → 先整图后子图。

不需要理解对方的完整构建系统，只需要找到两个 target 名和编译产物的位置。
