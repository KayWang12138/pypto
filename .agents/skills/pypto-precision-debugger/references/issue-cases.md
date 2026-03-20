# Issue 典型案例

> 以下案例来自 GitCode cann/pypto 仓库已关闭的精度 Issue（2026-02-01 至 2026-03-23）

## 问题匹配指南

根据用户描述的问题现象，快速匹配相似的 Issue 案例：

> **注意**：以下 Issue 大部分已修复，如使用最新版本仍遇到类似问题，可能是新 bug 或其他原因。

| 用户问题关键词 | 可能的问题类型 | 相关 Issue | 修复状态 | 推荐排查方法 |
|--------------|--------------|-----------|---------|-------------|
| reshape 后精度异常 | valid_shape 推导错误 | #498, #292, #813 | ✅ 已修复 | 检查 reshape 前后 valid_shape |
| view + reshape 组合 | inplace 元数据问题 | #343 | - | 避免 inplace=True |
| unroll 后精度失败 | RegisterCopy pass 错误 | #223, #341, #653 | ✅ 已修复 | 设置 unroll_list=[1] 测试 |
| 非整除场景精度失败 | valid_shape 推导 | #498, #341 | ✅ 已修复 | 检查整除/非整除差异 |
| loop 外 tensor 问题 | tensor 作用域问题 | #340, #626 | ✅ 已修复 | 在 loop 内创建 tensor |
| 动态轴结果不一致 | JIT 缓存问题 | #273, #529, #533 | ✅ 已修复 | 检查动态轴缓存 |
| 嵌套循环精度异常 | 并行执行内存覆盖 | #4 (案例4) | - | 添加 submit_before_loop=True |
| 特定 shape 精度失败 | buffer 限制/对齐问题 | #787, #724, #355 | ✅ 已修复 | 检查 buffer 大小限制 |
| mix 场景精度失败 | UB2L1 脏数据 | #799 | ✅ 已修复 | 检查 viewShape/validShape |
| 多 kernel 配置干扰 | 配置作用域泄漏 | #712 | ✅ 已修复 | 检查配置栈清理 |
| 精度工具通过但上板失败 | 仿真不覆盖 | #539 | - | 二分定位问题 op |
| assemble 相关问题 | Memtype 约束缺失 | #260 | ✅ 已修复 | 检查 Assemble Memtype |
| bf16 精度失败 | 舍入/对齐问题 | #355, #354 | ✅ 已修复 | 检查尾轴对齐 |
| 多核切 K 精度不确定 | GM 原子累加 | #795 | ✅ 已修复 | 检查多核累加逻辑 |
| inplace tensor 问题 | 内存处理错误 | #481, #468 | ✅ 已修复 | 检查 inplace 内存管理 |
| 精度工具粗检失败 | GC 内存回收 | #680 | ✅ 已修复 | 保存 tensor 引用 |

**使用建议**：
- 如使用最新版本代码，上述已修复问题不应再出现
- 如仍遇到类似问题，建议先确认代码版本，或考虑是新 bug
- 案例中的排查方法仍可作为定位思路参考

---

## Issue #799：UB2L1 脏数据污染

**修复状态**：✅ 已修复 (PR !1775, !1781)

**问题现象**：DeepSeek V3 SparseAttention antiquant 算子，打开 mix 场景后精度失败。

**问题原因**：UB2L1 搬运在 mix 场景下，当 `viewShape > align(validShape)` 时，搬运了多余的脏数据区域（UB 中未初始化的区域）混入 L1，导致后续计算结果错误。

**定位方法**：
1. 使用精度工具对比仿真与上板结果
2. 检查 viewShape 与 validShape 的关系
3. 定位到 UB2L1 操作的边界判断逻辑

**修复方法**：修复 UB2L1 操作的边界判断逻辑，正确处理 `viewShape > align(validShape)` 情况。

---

## Issue #795：GM 非确定性原子累加精度报错

**修复状态**：✅ 已修复

**问题现象**：多核切 K 场景下，GM 原子累加结果不确定，精度偶发失败。

**问题原因**：多核并发执行时，GM（Global Memory）原子累加操作存在竞争条件，不同核的累加顺序不确定，导致结果非确定性。

**定位方法**：
1. 多次运行同一 case，检查结果是否一致
2. 检查是否使用多核切 K
3. 定位 GM 累加逻辑

**修复方法**：修复 GM 原子累加的同步机制，确保多核累加结果确定性。

---

## Issue #787：Log1p/PReLU 超 UB 精度错误

**修复状态**：✅ 已修复 (PR !1716, !1747)

**问题现象**：Log1p/PReLU 算子运行报错 `exception aicore error`，或精度静默失败。

**问题原因**：Compare/TCMP 操作的临时缓冲区存在 **4096 字节硬限制**，当实际 shape 所需缓冲区超过 4096 字节时，发生 UB 越界。

**定位方法**：
1. 检查算子内部临时 buffer 大小计算
2. 对比 shape 与 4096 字节限制
3. 使用精度工具检测静默越界

**修复方法**：移除 4096 字节硬限制，改为根据实际 shape 动态计算 buffer 大小。

---

## Issue #712：多 kernel 配置项相互干扰

**修复状态**：✅ 已修复 (PR !1662)

**问题现象**：同一进程中依次定义/调用多个 pypto kernel 时，前一个 kernel 的配置影响后一个 kernel 的结果。

**问题原因**：`combine_axis` 等配置选项使用全局配置作用域栈，但在 kernel 结束时作用域栈未被正确清理，导致后续 kernel 的配置环境被污染。

**定位方法**：
1. 检查多 kernel 场景的配置项
2. 对比单 kernel 与多 kernel 的精度差异
3. 定位配置作用域栈

**修复方法**：修复配置作用域栈，在 kernel 调用结束时正确回收作用域，防止配置项跨 kernel 泄漏。

---

## Issue #680：精度工具粗检结果不一致

**修复状态**：✅ 已修复 (PR !1610)

**问题现象**：`set_verify_golden_data` 不同传参方式影响粗检结果。先保存到变量再传入 → 通过；直接传入 tensor → 失败。

**问题原因**：接口内部只保存了设备端数据引用，未保存原始 tensor 对象。直接传入的 tensor 没有外部变量持有，被 Python GC 立即回收，后续 verify 阶段读到脏内存。

**定位方法**：
1. 对比不同传参方式的粗检结果
2. 检查 golden tensor 的生命周期
3. 定位 GC 回收时机

**修复方法**：在接口内部额外保存原始 tensor 的强引用，防止 GC 过早回收。

---

## Issue #653：循环展开精度问题

**修复状态**：✅ 已修复

**问题现象**：将循环从简洁写法改为手动展开后，精度失败。两种数学等价的写法，上板结果不同。

**问题原因**：前端在处理手动展开的循环体时，写法的微小差异导致 trace 阶段生成不等价的图。

**定位方法**：
1. 对比简洁写法与展开写法的精度差异
2. 检查 trace 阶段生成的计算图
3. 定位前端循环展开处理逻辑

**修复方法**：修复前端对循环展开写法的 trace 逻辑，保证手动展开循环与简洁写法生成等价的计算图。

---

## Issue #626：loop 外 full tensor 精度问题

**修复状态**：✅ 已修复

**问题现象**：`pypto.full` 创建的 tensor 放在 loop 外有精度问题，放在 loop 里则精度正确。

**问题原因**：在 `pypto.loop` 内部，赋值语句 `a = a + 1` 在 trace 阶段 `=` 前后的 `a` 被识别为不同的张量，导致每次循环的累加结果未能正确写回原张量。

**定位方法**：
1. 对比 loop 内/外创建 tensor 的精度差异
2. 检查 loop 内赋值语句的 trace 结果
3. 定位 TensorGraph 仿真阶段的 view 状态传播

**修复方法**：在 loop 内对 tensor 做累加时，使用原地赋值语法 `a[:] = a + 1`。

---

## Issue #539：精度工具通过但上板失败

**问题现象**：所有 pass 精度工具校验均通过，但上板实测存在严重精度偏差（74%+ 元素超容差）。

**问题原因**：精度工具的 pass 验证覆盖范围存在盲区——某些 pass 的变换在仿真中通过，但在实际硬件执行时引入了精度偏差。

**定位方法**：
1. 使用二分定位问题 op
2. 对比仿真与上板的中间结果
3. 检查 pass 变换路径

**修复方法**：修复精度工具仿真逻辑，确保 pass 校验覆盖与实际上板执行路径一致。

---

## Issue #533：动态轴 tensor 需手动取出

**修复状态**：✅ 已修复 (PR !1490)

**问题现象**：jit 函数内部定义的 tensor 如果使用动态轴，需要手动取出动态轴值，否则功能错误。

**问题原因**：Parser 在处理 jit 函数内部定义的 tensor（非入参）时，未正确生成 outcast 动态轴的推导表达式，导致编译出的 CCE 代码中输出 tensor 的动态 shape 推导缺失。

**定位方法**：
1. 检查内部定义 tensor 的动态轴处理
2. 检查生成的 CCE 代码中 outcast 表达式
3. 定位 Parser 动态轴处理逻辑

**修复方法**：修复 Parser 中对 kernel 内部定义 tensor 的动态轴处理逻辑，自动生成正确的 outcast 动态轴推导表达式。

---

## Issue #498：pypto.reshape 丢失 valid_shape

**修复状态**：✅ 已修复 (PR !1385)

**问题现象**：reshape + matmul 场景，b 轴不能被 tile_b 整除时精度失败。

**问题原因**：`pypto.reshape` 在进行形状变换时，**未将上游 tensor 的 valid_shape 正确传播**到输出 tensor，导致 valid_shape 从动态符号退化为静态常量，后续 MTE 指令使用错误的 valid_shape 访问越界。

**定位方法**：
1. 对比整除与非整除场景的精度差异
2. 检查 reshape 前后 valid_shape 的值
3. 定位 MTE 越界问题

**修复方法**：修复 reshape pass 中的 valid_shape 推导逻辑，确保 reshape 操作正确传播上游 valid_shape 的动态符号信息。

---

## Issue #468：新前端返回值写法精度有误

**修复状态**：✅ 已修复

**问题现象**：output_tensor 不作入参带返回值时精度有误；output_tensor 作入参不带返回值时精度正确。

**问题原因**：output_tensor 在函数内部新建（非入参），框架在处理带返回值的 output tensor 时，内存分配与 host 侧对接存在差异，导致返回的 tensor 数据不完整或地址错误。

**定位方法**：
1. 对比两种写法的精度差异
2. 检查返回值 tensor 的内存分配
3. 定位前端返回值处理逻辑

**修复方法**：修复新前端对带返回值 output tensor 的内存处理逻辑，使两种写法的精度行为一致。

---

## Issue #355：Round Operation bf16 精度失败

**修复状态**：✅ 已修复 (PR !908)

**问题现象**：Round 操作部分 bf16 case 精度失败，四舍五入边界值（如 2.5、8.5、-4.5）计算结果与预期不符。

**问题原因**：Round 操作对 bf16 类型四舍五入边界值的处理逻辑存在精度 bug，在尾轴非对齐场景下未能正确处理边界值的舍入。

**定位方法**：
1. 检查 bf16 类型的四舍五入边界值
2. 对比不同尾轴对齐情况
3. 定位 Round 操作的舍入逻辑

**修复方法**：修复 Round 操作对 bf16 类型尾轴非对齐场景的处理，正确支持四舍五入边界值。

---

## Issue #343：Reshape inplace=True 大偏差

**注意**：这是前端语法问题。

**问题现象**：`pypto.view` 后再做 `pypto.reshape(inplace=True)`，精度严重偏差（96%+ 元素超容差）。

**问题原因**：inplace=True 的 reshape 试图在原 tensor 上就地修改，但 view 的元数据（offset、valid_shape）未被正确传播，导致 reshape 结果指向错误的内存区域。

**定位方法**：
1. 对比 `inplace=False` 与 `inplace=True` 的精度差异
2. 检查 view + reshape 组合的元数据传播
3. 使用精度工具定位偏差来源

**修复方法**：避免 view 后使用 `reshape(inplace=True)`，或修复 inplace reshape 的元数据传播逻辑。

---

## Issue #341：非整除场景精度问题

**修复状态**：✅ 已修复

**问题现象**：gdr 算子非整除场景精度失败。`unroll_list=[16,1]` 精度失败，`[32,1]` 或 `[32,16,1]` 精度正常。

**问题原因**：RegisterCopy pass 在处理非整除场景时，unroll_list 配置与非整除尾部处理存在 bug——某些 unroll 配置下 tensor 的 Copy 路径选择错误。

**定位方法**：
1. 对比不同 unroll_list 配置的精度差异
2. 检查整除与非整除场景
3. 定位 RegisterCopy pass 的 Copy 路径选择

**修复方法**：修复 RegisterCopy pass，正确处理 unroll_list 与非整除尾部场景的 Copy 逻辑。

---

## Issue #340：pypto.full 在 loop 外创建导致精度问题

**修复状态**：✅ 已修复 (PR !1069)

**问题现象**：`pypto.full` 在循环外创建的 tensor 放在 assemble 中使用时精度失败，循环内创建则精度正确。

**问题原因**：`pypto.full` 在循环外创建的 tensor 在编译器的 RemoveRedundantOp/validShape 处理中，tensor 的 view attr 未被正确维护，在 unroll 场景下 tensor 被错误复用或 view 映射错误。

**定位方法**：
1. 对比 loop 内/外创建 tensor 的精度差异
2. 检查 unroll 场景下的 tensor 复用情况
3. 使用精度工具验证 view attr

**修复方法**：在 loop 内创建 full tensor，或修复 RemoveRedundantOp pass 对 validShape 的重用逻辑。

---

## Issue #273：动态轴第二次 call 按第一次 shape 执行

**修复状态**：✅ 已修复 (PR !1135)

**问题现象**：创建一次 kernel，两次 call，第二次 call 传入不同的动态轴 shape，但仍按第一次 compile 的 shape 执行，精度错误。

**问题原因**：新前端对动态轴的 JIT 缓存逻辑存在 bug：第二次调用时未重新编译，复用了第一次的编译产物（不含新动态轴的正确推导）。

**定位方法**：
1. 对比两次 call 的 shape 和输出
2. 检查 JIT 缓存是否正确失效
3. 打印编译日志确认是否重新编译

**修复方法**：修复 Parser 对动态 dimension 的缓存与重编译逻辑，确保动态轴变化时触发重新编译。

---

## Issue #260：Assemble Memtype 约束缺失

**修复状态**：✅ 已修复

**问题现象**：GDR 算子整网精度失败，整网数据出现 nan。

**问题原因**：PR 修改了 `insert_op_for_viewassemble` pass，在处理 ASSEMBLE op 时未加入 Memtype 约束条件，导致某些内存类型的 assemble 在 PreGraphProcess 时处理错误，引发精度失败和 nan。

**定位方法**：
1. 使用二分定位问题 commit
2. 检查 Assemble 操作的 Memtype 约束
3. 定位 PreGraphProcess 处理逻辑

**修复方法**：在 `insert_op_for_viewassemble` pass 中加入 Memtype 约束条件，确保 Assemble 操作正确处理。
