# PyPTO编程指南

-   [简介](简介.md)
-   [快速入门](快速入门.md)
-   [编程范式](编程范式.md)
-   [算子开发](算子开发.md)
    -   [Tensor的创建](Tensor的创建.md)
    -   [Tensor的操作](Tensor的操作.md)
        -   [支持数学运算](支持数学运算.md)
        -   [支持逻辑结构变换](支持逻辑结构变换.md)

    -   [Tiling配置](Tiling配置.md)
    -   [编译与执行](编译与执行.md)
    -   [循环和数据切分](循环和数据切分.md)
    -   [条件与分支](条件与分支.md)

-   [功能调试](功能调试.md)
    -   [编译与执行流程](编译与执行流程.md)
    -   [NPU上板调试](NPU上板调试.md)
    -   [CPU仿真调试](CPU仿真调试.md)
    -   [ffn\_shared\_expert\_quant算子NPU上板调试案例](ffn_shared_expert_quant算子NPU上板调试案例.md)

-   [精度调试](精度调试.md)
    -   [精度调试流程](精度调试流程.md)
    -   [精度调试工具](精度调试工具.md)

-   [性能调优](性能调优.md)
    -   [性能调优流程](性能调优流程.md)
    -   [性能数据采集与分析](性能数据采集与分析.md)
    -   [QuantIndexerProlog算子性能优化案例](QuantIndexerProlog算子性能优化案例.md)

-   [PyTorch集成和接入](PyTorch集成和接入.md)
-   [常见问题](常见问题.md)
    -   [kernel函数出参未写回导致计算不生效](kernel函数出参未写回导致计算不生效.md)
    -   [未设置执行算子的设备id](未设置执行算子的设备id.md)
    -   [CANN包不兼容](CANN包不兼容.md)
    -   [TileShape与Tensor维度不匹配](TileShape与Tensor维度不匹配.md)
    -   [view未传入valid\_shape导致精度问题](view未传入valid_shape导致精度问题.md)
    -   [set\_xxx\_tile\_shapes最后一维没有32字节对齐校验报错](set_xxx_tile_shapes最后一维没有32字节对齐校验报错.md)
    -   [算子编译报堆栈溢出错误](算子编译报堆栈溢出错误.md)
    -   [循环中使用Python print函数打印](循环中使用Python-print函数打印.md)

-   [已知问题](已知问题.md)
    -   [使用未初始化的Tensor](使用未初始化的Tensor.md)
    -   [同一个Tensor进行View和Assemble导致图成环报错](同一个Tensor进行View和Assemble导致图成环报错.md)
    -   [同一个算子多次执行时，静态轴传入不同的运行时值（或者动态轴缺少标注）](同一个算子多次执行时-静态轴传入不同的运行时值（或者动态轴缺少标注）.md)
    -   [父循环内跨多个子循环的Tensor内存不支持在每次父循环迭代中分配](父循环内跨多个子循环的Tensor内存不支持在每次父循环迭代中分配.md)
    -   [SymbolicScalar不支持循环内自增](SymbolicScalar不支持循环内自增.md)
    -   [已安装torch\_npu，但未安装cann时，执行仿真异常](已安装torch_npu-但未安装cann时-执行仿真异常.md)

-   [附录](附录.md)
    -   [术语表](术语表.md)

