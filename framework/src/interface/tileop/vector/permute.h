/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "mte.h"
#include "trans.h"

namespace {

template <typename T>
__aicore__ inline void swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename DType>
__aicore__ void UBTransposeAxis(
    uint64_t srcUbAddr, uint64_t dstUbAddr, uint64_t tmpUbAddr,
    const size_t srcShape[5], const size_t srcStride[5],
    unsigned axis0, unsigned axis1,
    size_t dstShape[5], size_t dstStride[5]) {

    // 1. 计算新形状和步长
    for (unsigned d = 0; d < 5; ++d) {
        dstShape[d] = srcShape[d];
        dstStride[d] = srcStride[d];
    }
    swap(dstShape[axis0], dstShape[axis1]);
    swap(dstStride[axis0], dstStride[axis1]);

    // 2. 确定循环维度（除 axis0 和 axis1 外的三个轴）
    unsigned loopDims[3];
    int idx = 0;
    for (unsigned d = 0; d < 5; ++d) {
        if (d != axis0 && d != axis1) {
            loopDims[idx++] = d;
        }
    }
    // 如果实际维数少于 5，缺失的维度长度为 1，循环仍正确

    // 3. 循环遍历所有其他维度
    int index[3] = {0};
    bool done = false;
    while (!done) {
        // 计算源地址偏移
        size_t srcOffset = 0;
        for (int k = 0; k < 3; ++k) {
            srcOffset += index[k] * srcStride[loopDims[k]];
        }
        // 计算目标地址偏移（使用目标步长）
        size_t dstOffset = 0;
        for (int k = 0; k < 3; ++k) {
            dstOffset += index[k] * dstStride[loopDims[k]];
        }

        size_t rows = srcShape[axis0];
        size_t cols = srcShape[axis1];
        size_t rowStride = srcStride[axis0];
        size_t colStride = srcStride[axis1];

        // 构造源 Tile
        using SrcTile = pto::Tile<pto::TileType::Vec, DType, -1, -1, pto::BLayout::RowMajor, -1, -1>;
        SrcTile srcTile(rows, cols);
        pto::TASSIGN(srcTile, srcUbAddr + srcOffset * sizeof(DType));

        // 构造目标 Tile
        using DstTile = pto::Tile<pto::TileType::Vec, DType, -1, -1, pto::BLayout::RowMajor, -1, -1>;
        DstTile dstTile(rows, cols);
        pto::TASSIGN(dstTile, dstUbAddr + dstOffset * sizeof(DType));

        // 临时 Tile（大小与子矩阵相同）
        using TmpTile = pto::Tile<pto::TileType::Vec, DType, -1, -1, pto::BLayout::RowMajor, -1, -1>;
        TmpTile tmpTile(rows, cols);
        pto::TASSIGN(tmpTile, tmpUbAddr);

        // 执行转置
        pto::TTRANS(dstTile, srcTile, tmpTile);

        // 更新循环变量
        for (int k = 2; k >= 0; --k) {
            index[k]++;
            if (index[k] < static_cast<int>(srcShape[loopDims[k]])) {
                break;
            } else {
                index[k] = 0;
                if (k == 0) done = true;
            }
        }
    }
}

/**
 * @brief 简单的 UB 张量包装，用于 TLoad/TStore。
 */
template <typename T>
struct UbTensor {
    uint64_t addr;
    Layout layout;
    static constexpr auto FORMAT = Hardware::UB;
    using Type = T;
    uint64_t GetAddr() const { return addr; }
    Layout GetLayout() const { return layout; }
};

} // anonymous namespace

/**
 * @brief 对 GM 张量进行 permute，在 UB 内完成所有轴交换。
 * @param dst        目标 GM 张量（输出）
 * @param src        源 GM 张量（输入）
 * @param perm       目标排列，长度为 n，表示新顺序中每个位置对应的原维度索引
 * @param n          维度数 (2 ≤ n ≤ 5)
 * @param coordinate 当前核坐标（用于多核分片）
 * @param ubAddr     用户分配的 UB 缓冲区基地址
 * @param ubSize     UB 缓冲区大小（字节）。需要至少 3 * 张量大小（两个副本 + 一个临时区）
 */
template <typename DST, typename SRC, typename C>
__aicore__ void TPermute(DST dst, SRC src, const size_t perm[], size_t n, C coordinate,
                         uint64_t ubAddr, size_t ubSize) {
    static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::GM);

    using DType = typename DST::Type;
    // 处理 bool 类型
    using ActualType = std::conditional_t<std::is_same_v<DType, bool>, uint8_t, DType>;
    constexpr size_t elemSize = sizeof(ActualType);

    // 获取源张量布局
    const auto srcLayout = src.GetLayout();
    size_t srcShape[5] = {0};
    size_t srcStride[5] = {0};
    for (unsigned d = 0; d < 5; ++d) {
        srcShape[d] = srcLayout.template GetShapeDim<d, 5>();
        srcStride[d] = srcLayout.template GetStrideDim<d, 5>();
    }

    // 计算总元素个数
    size_t totalElems = 1;
    for (unsigned d = 0; d < n; ++d) totalElems *= srcShape[d];
    size_t totalBytes = totalElems * elemSize;
    // 确保 UB 缓冲区足够大
    // 注意：实际只需 2 * totalBytes + maxSubmatrixBytes，这里简化要求 3 * totalBytes
    (void)ubSize; // 避免未使用警告，实际可添加检查

    // 分配 UB 缓冲区：A, B, 临时区
    uint64_t ubAddrA = ubAddr;
    uint64_t ubAddrB = ubAddr + totalBytes;
    uint64_t tmpUbAddr = ubAddr + 2 * totalBytes; // 临时区大小至少等于一个子矩阵，这里用整个张量大小

    // 计算 GM 偏移（多核并行）
    size_t gmOffset = srcLayout.template GetGmOffset<C, 5>(coordinate);

    // 第一步：从 GM 加载到 UB A
    UbTensor<ActualType> ubTensorA;
    ubTensorA.addr = ubAddrA;
    ubTensorA.layout = srcLayout; // 初始布局与源相同
    TLoad(ubTensorA, src, coordinate);

    // 当前活动缓冲区地址和布局
    uint64_t activeUbAddr = ubAddrA;
    size_t currentShape[5];
    size_t currentStride[5];
    for (unsigned d = 0; d < 5; ++d) {
        currentShape[d] = srcShape[d];
        currentStride[d] = srcStride[d];
    }

    // 计算逆排列 inv，inv[i] 表示原维度 i 当前所在的位置（初始即 i）
    size_t inv[5];
    for (size_t i = 0; i < n; ++i) {
        inv[perm[i]] = i;
    }

    // 迭代归位算法
    for (size_t i = 0; i < n; ++i) {
        if (inv[i] != i) {
            // 找到 j 使得 inv[j] == i
            size_t j = i;
            while (j < n && inv[j] != i) ++j;
            // 交换轴 i 和 j
            uint64_t otherUbAddr = (activeUbAddr == ubAddrA) ? ubAddrB : ubAddrA;
            size_t newShape[5], newStride[5];
            UBTransposeAxis<ActualType>(activeUbAddr, otherUbAddr, tmpUbAddr,
                                         currentShape, currentStride, i, j,
                                         newShape, newStride);
            // 更新活动缓冲区
            activeUbAddr = otherUbAddr;
            for (unsigned d = 0; d < 5; ++d) {
                currentShape[d] = newShape[d];
                currentStride[d] = newStride[d];
            }
            // 更新逆排列
            swap(inv[i], inv[j]);
        }
    }

    // 最后，将活动缓冲区中的数据存回 GM
    // 构造 UB 张量对象，其布局为最终形状
    UbTensor<ActualType> ubTensorResult;
    ubTensorResult.addr = activeUbAddr;
    // 使用 MakeShape 和 MakeStride 从数组创建 tuple，使用 MakeLayout 创建 Layout
    auto resultShape = MakeShape(currentShape[0], currentShape[1], currentShape[2], currentShape[3], currentShape[4]);
    auto resultStride = MakeStride(currentStride[0], currentStride[1], currentStride[2], currentStride[3], currentStride[4]);
    auto resultTileShape = MakeTileShape(currentShape[0], currentShape[1], currentShape[2], currentShape[3], currentShape[4]);
    Layout resultLayout = MakeLayout(resultShape, resultStride, resultTileShape);
    ubTensorResult.layout = resultLayout;

    // 存储到 GM
    TStore(dst, ubTensorResult, coordinate);
}

#endif // TILEOP_TILE_OPERATOR_PERMUTE__H