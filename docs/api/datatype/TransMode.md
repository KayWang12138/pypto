# TransMode

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950 PR/Ascend 950 DT |    √     |

## 功能说明

TransMode定义了FP32转换TF32时的舍入模式，用于控制浮点数转换时的精度处理方式，确保转换结果的准确性。

## 原型定义

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # 不进行TF32转换
     CAST_RINT = ...   # 舍入到最近整数，平局时舍入到偶数
     CAST_ROUND = ...  # 舍入到最近整数，平局时远离零舍入
```

