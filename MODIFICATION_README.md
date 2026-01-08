# 修改说明：直接在 args_action.py 中重排序耗时测试

## 问题背景
用户希望将耗时最长的测试用例在 `--gtest_list_tests` 中排在最前面，以减少多卡并行测试的拖尾现象。

## 解决方案
直接修改 `framework/tests/cmake/scripts/python/utils/args_action.py` 中的 `ArgsGTestFilterListAction.parse_all_cases()` 方法，在获取测试列表后进行重排序。

## 修改内容

### 1. 添加耗时测试列表
在 `ArgsGTestFilterListAction` 类中添加了 `SLOW_TESTS` 常量：

```python
SLOW_TESTS = [
    "DeepSeekIndexerAttentionQuantUTest.4B_mtp_ut",
    "DynamicQuantLightningIndexerPrologUtest.b4_s1_2_s2_64k",
    "DynamicOpsTest.Cube",
    "DecodeIndexerAttentionUtest.utest_decode_indexer_attention",
    "DynamicPATest.dynamic_pa_low_lantency_manual_unroll",
    "FunctionTest.Test_quantMM",
    "DynamicGenGatedScoreUtest.utest_gen_gated_score_dyn",
    "DynamicGenGatedScoreUtest.utest_gen_gated_score_plus_dyn",
    "DynamicLightningIndexerPrologUtest.utest_lightning_indexer_prolog",
    "AttentionPostUTest.b32_s1_nz_fp16_quant",
    "TestLightningIndexerUtest.lightning_indexer_b_4_s1_2_s2_64k_quant",
    "DynamicAttentionUtTest.dynamic_attention_low_nz",
    "GenAttnUtTest.TestDynamicGenAttenTest_FP16_ut",
    "DynamicAttentionUtTest.dynamic_attention_low",
    "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4_int8",
    "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4"
]
```

### 2. 修改 parse_all_cases 方法
在返回测试列表之前添加重排序逻辑：

```python
@staticmethod
def parse_all_cases(binary: str) -> List[str]:
    """获取gtest ut测试用例，并将耗时测试排在前面
    """
    # ... 原有的解析逻辑 ...

    # 重新排序：将耗时测试排在前面
    slow_tests_in_list = []
    regular_tests = []

    for test in cases:
        if test in ArgsGTestFilterListAction.SLOW_TESTS:
            slow_tests_in_list.append(test)
        else:
            regular_tests.append(test)

    # 保持耗时测试的原有顺序（已经是按耗时降序）
    ordered_slow_tests = []
    for test in ArgsGTestFilterListAction.SLOW_TESTS:
        if test in slow_tests_in_list:
            ordered_slow_tests.append(test)

    # 返回重排序后的列表：耗时测试在前，普通测试在后
    return ordered_slow_tests + regular_tests
```

## 工作原理

1. **拦截获取**: 当并行测试框架调用 `--gtest_filter=*` 时，会通过 `ArgsGTestFilterListAction.parse_all_cases()` 获取所有测试用例
2. **重排序**: 在返回测试列表之前，按照内置的耗时测试名单重新排序
3. **优先调度**: 并行测试框架会按照返回的顺序分配测试到不同的卡上

## 优势

### 1. 无侵入性
- 不需要创建额外的包装脚本
- 直接在现有代码中修改
- 保持原有接口不变

### 2. 全局生效
- 所有使用 `ArgsGTestFilterListAction` 的地方都会自动应用重排序
- 不需要在每个调用处单独处理

### 3. 易于维护
- 耗时测试列表集中在一個地方管理
- 可以轻松添加或删除测试用例

## 使用方式

修改后，你的并行测试框架无需任何改变，原来的命令：

```bash
# 原来的调用方式保持不变
./your_test_executable --gtest_filter=*
```

会自动返回重排序后的测试列表，耗时测试排在前面。

## 验证方法

使用提供的 `test_args_action.py` 脚本验证：

```bash
python test_args_action.py
```

会创建模拟的gtest程序并验证重排序功能是否正常工作。

## 自定义配置

如果需要添加更多耗时测试，只需修改 `SLOW_TESTS` 列表：

```python
SLOW_TESTS = [
    "YourNewSlowTest.TestCase",
    # 添加更多...
]
```

## 注意事项

1. **测试名称匹配**: 确保测试名称与列表中的完全匹配
2. **顺序保持**: 列表中的顺序决定了测试的优先级（耗时从高到低）
3. **向后兼容**: 不影响没有在列表中的测试用例
4. **性能影响**: 重排序逻辑开销很小，不影响测试执行性能

这个修改确保了耗时最长的测试用例会被优先调度到并行测试环境中，减少拖尾现象，提高整体测试效率。
