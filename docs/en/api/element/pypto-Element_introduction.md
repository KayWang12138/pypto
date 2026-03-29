# pypto.Element Introduction

In the PyPTO framework, the Element type is used to store scalar constants, representing constant expressions in computational operations. Python's built-in int type is typically mapped to the DT\_INT64 type, while the float type is mapped to DT\_FP32. In scenarios where operand types include both Tensor and Scalar, int and float values are usually converted to the corresponding Tensor's data type.

