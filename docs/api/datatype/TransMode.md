# TransMode

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950 PR/Ascend 950 DT |    √     |

## 功能说明

TransMode定义了在矩阵运算时，float数据类型转换为TF32数据类型时的舍入模式。

## 原型定义

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # 不使能float数据类型转换为TF32数据类型
     CAST_RINT = ...   # 舍入到最近整数，平局时舍入到偶数
     CAST_ROUND = ...  # 舍入到最近整数，平局时远离零舍入
```

