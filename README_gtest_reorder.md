# gtest 测试用例重排序工具

## 问题描述
在多卡并行测试环境中，耗时长的测试用例会导致拖尾现象，影响整体测试效率。

## 解决方案
使用 `gtest_reorder.py` 脚本包装 gtest 可执行文件，将指定的耗时测试用例排在 `--gtest_list_tests` 输出列表的最前面。

## 耗时测试用例列表
脚本内置了以下16个耗时最长的测试用例（按耗时降序排列）：

1. `DeepSeekIndexerAttentionQuantUTest.4B_mtp_ut` (189.20s)
2. `DynamicQuantLightningIndexerPrologUtest.b4_s1_2_s2_64k` (158.51s)
3. `DynamicOpsTest.Cube` (155.60s)
4. `DecodeIndexerAttentionUtest.utest_decode_indexer_attention` (148.72s)
5. `DynamicPATest.dynamic_pa_low_lantency_manual_unroll` (124.40s)
6. `FunctionTest.Test_quantMM` (92.51s)
7. `DynamicGenGatedScoreUtest.utest_gen_gated_score_dyn` (81.44s)
8. `DynamicGenGatedScoreUtest.utest_gen_gated_score_plus_dyn` (43.59s)
9. `DynamicLightningIndexerPrologUtest.utest_lightning_indexer_prolog` (41.61s)
10. `AttentionPostUTest.b32_s1_nz_fp16_quant` (41.37s)
11. `TestLightningIndexerUtest.lightning_indexer_b_4_s1_2_s2_64k_quant` (39.66s)
12. `DynamicAttentionUtTest.dynamic_attention_low_nz` (36.41s)
13. `GenAttnUtTest.TestDynamicGenAttenTest_FP16_ut` (35.37s)
14. `DynamicAttentionUtTest.dynamic_attention_low` (34.65s)
15. `DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4_int8` (34.45s)
16. `DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4` (31.19s)

## 使用方法

### 1. 替换 gtest 调用
将原来的：
```bash
./your_test_executable --gtest_list_tests
```

替换为：
```bash
python gtest_reorder.py ./your_test_executable --gtest_list_tests
```

### 2. 执行测试时也使用包装器
```bash
# 原来的命令
./your_test_executable --gtest_filter=*

# 替换为
python gtest_reorder.py ./your_test_executable --gtest_filter=*
```

### 3. 在脚本中使用
```bash
#!/bin/bash
TEST_CMD="python gtest_reorder.py ./your_test_executable"

# 查看重排序的测试列表
$TEST_CMD --gtest_list_tests

# 运行所有测试
$TEST_CMD --gtest_filter=*
```

## 输出示例

### 重排序前的输出
```
TestSuite1.
  test_case_1
  test_case_2
TestSuite2.
  slow_test_1
  test_case_3
```

### 重排序后的输出
```
TestSuite2.
  slow_test_1  [SLOW_TEST]
TestSuite1.
  test_case_1
  test_case_2
TestSuite2.
  test_case_3
```

## 集成到并行测试框架

在你的多卡并行测试框架中，可以这样集成：

```python
class ParallelTestRunner:
    def __init__(self):
        self.slow_tests = [
            "DeepSeekIndexerAttentionQuantUTest.4B_mtp_ut",
            "DynamicQuantLightningIndexerPrologUtest.b4_s1_2_s2_64k",
            # ... 其他耗时测试
        ]

    def schedule_test(self, test_name: str) -> dict:
        """调度测试到合适的卡"""
        if any(slow_test in test_name for slow_test in self.slow_tests):
            return {
                'gpu_id': 0,        # 最好的GPU
                'timeout': 300,     # 5分钟超时
                'priority': 'HIGH'
            }
        else:
            return {
                'gpu_id': self.get_available_gpu(),
                'timeout': 60,
                'priority': 'NORMAL'
            }
```

## 脚本原理

1. **拦截命令**: 检测是否为 `--gtest_list_tests` 命令
2. **获取列表**: 执行原始的 gtest 获取测试列表
3. **重新排序**: 将耗时测试移到列表前面，并添加 `[SLOW_TEST]` 标记
4. **格式化输出**: 保持 gtest 原有的输出格式

## 注意事项

1. **测试名称匹配**: 确保测试名称与脚本中定义的完全匹配
2. **更新列表**: 如果有新的耗时测试，可以直接修改脚本中的 `SLOW_TESTS` 列表
3. **兼容性**: 脚本完全兼容原有的 gtest 参数和选项
4. **性能**: 脚本本身开销很小，不会影响测试执行性能

## 优化效果

通过将耗时测试排在前面，你的并行测试框架可以：
- 优先调度耗时测试到性能最好的资源
- 减少整体测试时间的拖尾现象
- 提高多卡并行测试的效率
