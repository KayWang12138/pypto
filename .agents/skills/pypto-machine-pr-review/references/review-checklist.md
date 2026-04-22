# PR Review Checklist（PyPTO）

## 1) 审查规则

- 正确性：逻辑、边界、异常路径是否闭合。
- 回归风险：默认行为、接口契约、兼容性是否变化。
- 资源与稳定性：并发、时序、内存、释放路径是否安全。
- 可维护性：复杂度、命名、重复实现、可读性。
- 测试充分性：正常路径、异常路径、边界场景是否覆盖。

## 1.1) Machine 业务专项规则（必须检查）

- 调用链闭合：`host -> compile -> runtime -> device` 链路是否完整且行为一致。
- 返回码与异常态：关键分支必须处理返回码，禁止静默吞错。
- 并发与时序：重点检查线程/队列协作（如 compile/agent 线程）与多流同步点是否存在竞态。
- 内存生命周期：`AllocDevAddr/FreeDevAddr` 等分配释放是否成对，失败路径是否释放临时资源。
- 缓存一致性：CacheManager/RuntimeAgentMemory 相关读写是否一致，是否引入脏读或过期状态。
- ABI/接口行为：machine 对外接口或默认参数变更是否引入兼容性风险。
- 可观测性：关键路径是否保留必要日志/trace，问题是否可通过现有脚本定位。
- 测试映射：UT/ST 是否覆盖 machine 正常链路 + 异常链路 + 回归场景。
- 风险归因：结论必须绑定具体路径与符号（类/函数/命名空间），并区分事实与推断。

---

## 2) Finding 模板

每条问题建议使用以下结构：

- Severity: `High|Medium|Low`
- Fact（已确认事实）:
- Risk（风险）:
- Suggestion（修复建议）:
- Anchor: `path + position`（或暂定锚点）

---

## 3) 提交前确认模板（强制）

在发布评论前，必须向用户展示：

```
拟提交评论清单：
1) [High] path=..., position=..., need_to_resolve=true
   body=...
2) [Medium] path=..., position=..., need_to_resolve=false
   body=...
```

并明确询问：

`请确认是否提交以上评论到 PR（回复：确认提交 / 暂不提交 / 修改后再提交）`

只有收到“确认提交”后，才允许执行远程提交。

---

## 4) 提交后回执模板

```
已提交：
- note_id=..., path=..., position=...

提交失败：
- path=..., position=..., error=...
```

若存在失败项，需给出可执行重试建议。
