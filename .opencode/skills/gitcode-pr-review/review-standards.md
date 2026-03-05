# pypto 代码检视标准

## 通用检查项

### 必须检查（🔴 必须修改）

- [ ] 编译错误或明显的运行时错误
- [ ] 空指针/空引用解引用
- [ ] 内存泄漏（C++ 中 new 无对应 delete，文件/资源未关闭）
- [ ] 数组/容器越界访问
- [ ] 死锁或竞态条件（多线程场景）
- [ ] 逻辑错误（条件判断取反、循环边界错误）

### 建议检查（🟡 建议修改）

- [ ] 函数过长（C++ 超过 80 行，Python 超过 60 行）
- [ ] 重复代码超过 3 处
- [ ] 魔法数字（未命名的常量）
- [ ] 异常处理不完整（catch 过于宽泛或完全没有）
- [ ] 入参未做校验
- [ ] 性能问题：O(n²) 以上且 n 可能较大

### 可选优化（🟢 可选优化）

- [ ] 变量/函数命名不够语义化
- [ ] 注释缺失或过时
- [ ] 可以用标准库简化的代码

---

## C++ 专项（framework/ 目录）

### 内存管理

- 优先使用智能指针（`std::unique_ptr`、`std::shared_ptr`）而非裸指针
- 禁止使用 `delete[]` 配合手动 `new[]`，改用 `std::vector`
- 类成员中如有裸指针，检查 Rule of Five 是否完整实现

### 类型安全

- 避免 C 风格 cast（`(int)x`），使用 `static_cast`/`reinterpret_cast`
- 检查 `const` 正确性：不修改成员的成员函数应标记 `const`
- 模板参数需有约束或 `static_assert`

### 并发

- 共享数据必须有锁保护（`std::mutex` 或 `std::atomic`）
- 避免在持锁时调用外部函数（死锁风险）
- 优先使用 `std::lock_guard` / `std::unique_lock` 而非手动 lock/unlock

### 接口设计

- 公共 API 必须有头文件注释（参数含义、返回值、异常）
- 禁止在头文件中使用 `using namespace std`

### 编译确定性（pass 阶段）🔴

编译过程（尤其是 pass 阶段）必须产生**确定性结果**，禁止引入任何依赖指针地址或哈希随机化的遍历顺序。

**禁止使用指针作为有序容器的 key**：

- `std::set<T*>` 和 `std::map<T*, V>` 以指针值（内存地址）作为排序依据，相同逻辑在不同运行中地址不同，导致遍历顺序随机化
- 替代方案：使用稳定的标识符（如节点名称、ID、序号）作为 key，或使用自定义比较器基于对象内容排序
  ```cpp
  // ❌ 禁止
  std::set<Node*> visited;
  std::map<Value*, int> valueMap;

  // ✅ 替代
  std::set<NodeId> visited;               // 用稳定 ID
  std::map<std::string, int> valueMap;    // 用名称/字符串
  std::map<Value*, int, ValuePtrCmp> m;   // 或自定义比较器（基于内容）
  ```

**禁止遍历 `unordered_map` / `unordered_set` 的元素**：

- `std::unordered_map` / `std::unordered_set` 的迭代顺序由哈希值决定，哈希种子在不同进程/编译中可能不同（Address Space Layout Randomization 等），遍历结果不确定
- **读取单个元素（`find`/`count`/`[]`）不受影响**，有问题的是遍历（`for (auto& kv : map)`、`begin()`/`end()` 迭代）
- 替代方案：
  - 将元素收集到 `std::vector` 后排序再遍历
  - 改用 `std::map` / `std::set`（有序容器，遍历顺序稳定）
  ```cpp
  // ❌ 禁止：直接遍历 unordered 容器
  std::unordered_map<std::string, int> m;
  for (auto& [k, v] : m) { /* 遍历顺序不确定 */ }

  // ✅ 替代方案一：改用有序容器
  std::map<std::string, int> m;
  for (auto& [k, v] : m) { /* 按 key 字典序稳定遍历 */ }

  // ✅ 替代方案二：排序后遍历（保留 unordered 查找性能）
  std::vector<std::pair<std::string, int>> items(m.begin(), m.end());
  std::sort(items.begin(), items.end());
  for (auto& [k, v] : items) { /* 顺序确定 */ }
  ```

**判断依据**：凡是在 pass 阶段（IR 变换、图优化、代码生成等）出现上述用法，且遍历结果会影响输出顺序、寄存器分配、指令排布等，均属于 🔴 必须修改。

---

## Python 专项（python/ 和 models/ 目录）

### 类型与安全

- 公共函数必须有类型注解（`def foo(x: int) -> str:`）
- 不应使用裸 `except:` 或 `except Exception:`，应捕获具体异常
- 避免可变默认参数（`def f(lst=[]):`）

### 深度学习相关（models/ 目录）

- Tensor 操作检查 device 一致性（避免 CPU/GPU 混用）
- 矩阵维度注释（`# [batch, seq_len, hidden_dim]`）
- 避免在训练循环中频繁在 CPU/GPU 间搬运数据

### 代码风格

- 遵循 PEP8，行长度不超过 120 字符
- 模块级 docstring 说明模块用途
- 复杂算法必须有注释说明思路

---

## 仓库风格一致性

在检视变更文件时，需对照**同目录或同模块**的已有代码，检查新代码是否与仓库现有风格保持一致。具体方法：通过 `git show` 或读取仓库中相邻文件来了解周边代码惯例，而非仅依赖通用规范。

### 命名惯例

- **C++ 类名**：对照 `framework/src/` 中已有类的命名风格（如 `HostMachine`、`PassManager` 均为 PascalCase）
- **C++ 成员变量**：检查是否沿用尾部下划线惯例（`foo_`）
- **C++ 函数名**：对照相邻文件（如 `StartStage` vs `start_stage`，仓库使用哪种）
- **Python 模块/函数**：对照 `python/pypto/` 中已有模块的命名风格（snake_case）
- **CMake target 命名**：对照现有 target 名称前缀（如 `tile_fwk_*`）

### 文件与目录结构

- 新增文件的目录层级是否符合仓库惯例（如 C++ 接口头文件放 `interface/` 而非根目录）
- 头文件保护是否统一（仓库使用 `#pragma once` 还是 include guard）
- `.h`/`.cpp` 拆分方式是否与同模块一致（内联实现还是独立 `.cpp`）

### 错误处理与日志

- C++ 错误处理：对照周边代码使用 `ASSERT`/`MACHINE_ASSERT` 还是返回错误码，新代码应保持一致
- 日志输出：对照仓库使用 `ALOG_INFO_F`/`ALOG_DEBUG_F` 还是直接 `fprintf`，monitor 相关代码若用 `fprintf(stderr, ...)` 而其他模块均用 `ALOG_*`，需说明理由
- Python 异常类型：对照 `python/pypto/` 中已有异常的基类和命名规范

### 初始化与生命周期模式

- 单例实现方式是否与仓库其他单例一致（Meyers 单例 vs `std::call_once` vs 静态成员）
- RAII wrapper 的写法是否与同目录已有 Scope/Guard 类一致
- 模块初始化入口（如 Python `__init__.py` 中的导入顺序）是否遵循已有约定

### 检查方法

对于同目录的参照文件，可通过以下方式快速获取：
```bash
# 查看同目录其他文件了解命名惯例
ls framework/src/interface/machine/host/
# 读取同类文件对比风格
git show HEAD:framework/src/interface/machine/host/host_machine.h | head -50
```

---

## PR 描述检查

- PR 标题应说明「做了什么」，而非「修改了哪个文件」
- PR 描述应包含：背景、变更内容、测试方法
- 如有 Breaking Change，必须在描述中明确标注

---

## 检视输出建议

检视完成后，在整体审查意见中包含：

```
## 检视总结

**变更范围**：涉及 X 个文件，+Y -Z 行

**主要问题**：
- 🔴 必须修改 N 条
- 🟡 建议修改 N 条
- 🟢 可选优化 N 条

**整体评价**：[APPROVE / REQUEST_CHANGES / COMMENT]

具体意见已通过行内评论标注在对应位置。
```