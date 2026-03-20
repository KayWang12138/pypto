/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.h
 * \brief 量化Tile算子 - 支持INT8对称量化和非对称量化
 *
 * 本文件实现了基于Tile的量化操作，用于将FP32浮点张量转换为INT8/UINT8量化张量。
 * 量化是深度学习模型压缩和加速的关键技术，通过降低数值精度来减少内存占用和计算开销。
 *
 * 支持的量化类型:
 * - INT8_SYM (INT8对称量化): FP32 -> INT8，输出范围 [-128, 127]
 * - INT8_ASYM (INT8非对称量化): FP32 -> UINT8，输出范围 [0, 255]
 *
 * 注意: 本文件只实现逐行量化 (axis=-1)，逐列量化 (axis=-2) 在 Operation 层通过 Transpose 实现
 *
 * 量化公式:
 * - 对称量化: int8 = round(fp32 * scale)，结果截断到 [-128, 127]
 * - 非对称量化: uint8 = round(fp32 * scale + offset)，结果截断到 [0, 255]
 *
 * 5D张量布局说明:
 * - 第0维 (N0): 批次维度的第一层拆分
 * - 第1维 (N1): 批次维度的第二层拆分
 * - 第2维 (N2): 批次维度的第三层拆分
 * - 第3维 (H): 高度/行维度
 * - 第4维 (W): 宽度/列维度
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

// =============================================================================
// INT8 对称量化 (INT8 Symmetric Quantization)
// =============================================================================
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))
#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8对称量化算子 (逐行量化)
 *
 * 对称量化将浮点数映射到对称的整数范围 [-128, 127]，零点恰好对应0。
 * 这是最简单的量化方式，只需要一个缩放因子(scale)，不需要零点偏移。
 *
 * 数学公式:
 *   int8_value = round(fp32_value * scale)
 *   int8_value = clamp(int8_value, -128, 127)  // 截断到有效范围
 *
 * 其中 scale = 127 / max(|fp32_values|)，即目标范围的最大值除以输入绝对值的最大值
 *
 * @param dst 输出张量（INT8类型），存储量化后的结果
 * @param src 输入张量（FP32类型），待量化的浮点数据
 * @param scale 缩放因子张量，每行对应一个scale值，形状为 [rows, 1]
 *
 * 使用示例:
 *   // 逐行量化: 输入 [M, N], scale [M, 1], 输出 [M, N]
 *   TQuantInt8Sym(dst, src, scale);
 */
template <typename T0, typename T1, typename T2>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale) {
    // 期望的5D张量维度数，所有张量都将被统一转换为5D表示进行处理
    constexpr size_t expectSize = 5;

    // 获取各张量的布局信息（Layout包含形状、步长等元数据）
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // =========================================================================
    // 步骤1: 获取输出张量的形状信息
    // =========================================================================
    // 5D张量的各维度大小: [N0, N1, N2, H, W]
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();  // 第0维大小
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();  // 第1维大小
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();  // 第2维大小
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();  // 第3维大小（H: 行数）
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();  // 第4维大小（W: 列数）

    // 边界检查: 如果H或W为0，直接返回，无需处理
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    // 获取输入张量的H和W维度大小（通常与输出相同）
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // =========================================================================
    // 步骤2: 获取各张量的步长信息
    // =========================================================================
    // 步长用于计算在内存中的偏移量，支持非连续内存布局
    // 步长单位为元素个数（不是字节）
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride0 = scaleLayout.template GetStrideDim<0, expectSize>();
    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();

    // =========================================================================
    // 步骤3: 获取Tile级别的形状信息（编译期常量）
    // =========================================================================
    // Tile是硬件级别的基本处理单元，Tile形状在编译时确定
    // 这些值决定了每次迭代处理的数据块大小
    // dst rowwise的 W 要按int8做32对齐
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();   // 输出Tile高度
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();   // 输出Tile宽度
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, 32 / sizeof(int8_t));
    // src rowwise的 W 要按half做32对齐(TQuant中间fp32->s32->fp16)
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();   // 输入Tile高度
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();   // 输入Tile宽度
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, 32 / sizeof(half));
    // scale colwise的 H 需要32对齐
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>(); // scale Tile高度
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileH, 32 / sizeof(float));
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>(); // scale Tile宽度

    // =========================================================================
    // 步骤4: 提取数据类型信息
    // =========================================================================
    using DstDtype = typename T0::Type;    // 输出数据类型（通常是int8_t）
    using SrcDtype = typename T1::Type;    // 输入数据类型（通常是float）
    using ScaleDtype = typename T2::Type;  // scale数据类型（通常是float）

    // =========================================================================
    // 逐行量化 (axis = -1)
    // =========================================================================
    // 逐行量化: 每行使用一个独立的scale值
    // 输入形状: [..., H, W]
    // scale形状: [..., H, 1] (每行一个scale)
    // 输出形状: [..., H, W]
    {
        // 定义Tile类型:
        // - DstTileDefine: 输出Tile，行主序布局
        // - SrcTileDefine: 输入Tile，行主序布局
        // - ScaleTileDefine: scale Tile，列主序布局（因为scale形状是[H,1]）
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst,
                                        pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src,
                                        pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, scaleTileW,
                                        pto::BLayout::ColMajor, -1, -1>;

        // 四层嵌套循环遍历所有Tile
        // n0, n1, n2: 遍历批次维度
        // n3: 遍历H维度（行），每个n3对应一组独立的scale
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    // 创建Tile对象，指定当前Tile要处理的数据形状
                    // dstTile和srcTile: 处理整个 [H, W] 块
                    // scaleTile: 只有一列 [H, 1]，每个H对应一个scale值
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    SrcTileDefine srcTile(srcShape3, srcShape4);
                    ScaleTileDefine scaleTile(srcShape3, 1);

                    // 计算各张量在当前Tile的内存偏移量
                    // 偏移量 = 各维度索引 * 对应维度的步长 之和
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 + n2Index * scaleStride2;

                    // 将Tile绑定到实际内存地址
                    // TASSIGN: Tile Address ASSIGNment，建立Tile与内存地址的映射
                    // 注意: 地址需要转换为字节地址（乘以sizeof(类型)）
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));

                    // 执行INT8对称量化操作
                    // TQUANT<INT8_SYM>: 底层量化算子，执行 FP32 -> INT8 转换
                    pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile);
                }
            }
        }
    }
}

// =============================================================================
// INT8 非对称量化 (INT8 Asymmetric Quantization)
// =============================================================================

#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8非对称量化算子 (逐行量化)
 *
 * 非对称量化将浮点数映射到非对称的整数范围 [0, 255]（即UINT8），
 * 除了缩放因子(scale)外，还需要一个零点偏移(offset/zero_point)。
 *
 * 数学公式:
 *   uint8_value = round(fp32_value * scale + offset)
 *   uint8_value = clamp(uint8_value, 0, 255)  // 截断到有效范围
 *
 * 其中:
 *   scale = 255 / (max(fp32_values) - min(fp32_values))
 *   offset = round(0 - min(fp32_values) * scale)
 *
 * 非对称量化可以更精确地表示偏斜分布的数据，但计算开销略大。
 *
 * @param dst 输出张量（UINT8类型），存储量化后的结果
 * @param src 输入张量（FP32类型），待量化的浮点数据
 * @param scale 缩放因子张量，每行对应一个scale值，形状为 [rows, 1]
 * @param offset 零点偏移张量，与scale形状相同
 *
 * 使用示例:
 *   // 逐行量化: 输入 [M, N], scale [M, 1], offset [M, 1], 输出 [M, N]
 *   TQuantInt8Asym(dst, src, scale, offset);
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset) {
    // 期望的5D张量维度数
    constexpr size_t expectSize = 5;

    // 获取各张量的布局信息
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // =========================================================================
    // 步骤1: 获取输出张量的形状信息
    // =========================================================================
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    // 边界检查
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    // 获取输入张量的H和W维度大小
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // =========================================================================
    // 步骤2: 获取各张量的步长信息
    // =========================================================================
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride0 = scaleLayout.template GetStrideDim<0, expectSize>();
    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();

    // offset的步长信息
    auto offsetStride0 = offsetLayout.template GetStrideDim<0, expectSize>();
    auto offsetStride1 = offsetLayout.template GetStrideDim<1, expectSize>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<2, expectSize>();

    // =========================================================================
    // 步骤3: 获取Tile级别的形状信息（编译期常量）
    // =========================================================================
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr auto offsetTileH = TileOp::GetTensorTileShapeDim<T3, 3, expectSize>();
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();

    // =========================================================================
    // 步骤4: 提取数据类型信息
    // =========================================================================
    using DstDtype = typename T0::Type;      // 输出数据类型（通常是uint8_t）
    using SrcDtype = typename T1::Type;      // 输入数据类型（通常是float）
    using ScaleDtype = typename T2::Type;    // scale数据类型（通常是float）
    using OffsetDtype = typename T3::Type;   // offset数据类型（通常是float或int）

    // =========================================================================
    // 逐行量化 (axis = -1)
    // =========================================================================
    // 逐行量化: 每行使用独立的scale和offset值
    // 输入形状: [..., H, W]
    // scale/offset形状: [..., H, 1]
    // 输出形状: [..., H, W]
    {
        // 定义Tile类型
        // DstTileDefine/SrcTileDefine: 行主序布局
        // ScaleTileDefine/OffsetTileDefine: 列主序布局（因为形状是[H,1]）
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, scaleTileH, scaleTileW, pto::BLayout::ColMajor, -1, -1>;
        using OffsetTileDefine = pto::Tile<pto::TileType::Vec, OffsetDtype, offsetTileH, offsetTileW, pto::BLayout::ColMajor, -1, -1>;

        // 四层嵌套循环遍历所有Tile
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    // 创建Tile对象
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    SrcTileDefine srcTile(srcShape3, dstShape4);
                    ScaleTileDefine scaleTile(srcShape3, 1);
                    OffsetTileDefine offsetTile(srcShape3, 1);

                    // 计算内存偏移量
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 + n2Index * scaleStride2;
                    auto offsetOffset = n0Index * offsetStride0 + n1Index * offsetStride1 + n2Index * offsetStride2;

                    // 将Tile绑定到实际内存地址
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                    // 执行INT8非对称量化操作
                    // 注意: 非对称量化需要传入offset的指针
                    pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief 统一的量化接口
 *
 * 提供统一的模板接口，根据量化类型(quantType)自动选择对应的量化实现。
 * 这个接口简化了用户的使用，隐藏了底层实现的复杂性。
 *
 * @tparam quantType 量化类型:
 *         - pto::QuantType::INT8_SYM: INT8对称量化（3参数版本）
 *         - pto::QuantType::INT8_ASYM: INT8非对称量化（4参数版本）
 *
 * 使用示例:
 *   // INT8对称量化（3参数）
 *   TQuant<pto::QuantType::INT8_SYM>(dst, src, scale);
 *
 *   // INT8非对称量化（4参数）
 *   TQuant<pto::QuantType::INT8_ASYM>(dst, src, scale, offset);
 */

// ---------------------------------------------------------------------------
// INT8_SYM: 3参数版本（对称量化，不需要offset）
// ---------------------------------------------------------------------------
template <pto::QuantType quantType, typename T0, typename T1, typename T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale) {
    // 如果用户传入INT8_ASYM但只提供3个参数，会在编译时报错
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 3 parameters only supports INT8_SYM. For INT8_ASYM, provide offset parameter.");
    TQuantInt8Sym(dst, src, scale);
}

// ---------------------------------------------------------------------------
// INT8_ASYM: 4参数版本（非对称量化，需要offset）
// ---------------------------------------------------------------------------
template <pto::QuantType quantType, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset) {
    // 如果用户传入INT8_SYM但提供了4个参数，会在编译时报错
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 4 parameters only supports INT8_ASYM. For INT8_SYM, use 3 parameters.");
    TQuantInt8Asym(dst, src, scale, offset);
}

#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
