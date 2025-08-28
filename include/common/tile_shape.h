/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_shape.h
 * \brief
 */

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "tilefwk/hash_buffer.h"

namespace npu::tile_fwk {

constexpr int MAX_MDIM_SIZE = 2;
constexpr int MAX_KDIM_SIZE = 3;
constexpr int MAX_NDIM_SIZE = 2;
constexpr int MAX_DIST_DIM_SIZE = 3;

enum class TileShapeType {
    NORMAL,
    K,
    M,
    N,
    TYPE_NUM,
};

using VecTileShapes = std::vector<int64_t>;

class CubeTileShapes {
public:
    CubeTileShapes() = default;
    CubeTileShapes(std::array<int64_t, MAX_MDIM_SIZE> mShape, std::array<int64_t, MAX_KDIM_SIZE> kShape,
        std::array<int64_t, MAX_NDIM_SIZE> nShape, bool setL1TileParam = false)
        : m(mShape), k(kShape), n(nShape), setL1Tile(setL1TileParam) {}

    template <TileShapeType T>
    void SetTileShape(int index, int value) {
        if constexpr (T == TileShapeType::M || T == TileShapeType::N) {
            assert(index == 0 || index == 1);
        } else if constexpr (T == TileShapeType::K) {
            assert(index >= 0 && index < static_cast<int>(MAX_KDIM_SIZE));
        }
        assert(value > 0);
        if constexpr (T == TileShapeType::M) {
            m[index] = value;
        } else if constexpr (T == TileShapeType::K) {
            k[index] = value;
        } else if constexpr (T == TileShapeType::N) {
            n[index] = value;
        } else {
            assert(false);
        }
    }

    template <TileShapeType T>
    [[nodiscard]] int GetTileShape(int index) const {
        if constexpr (T == TileShapeType::M || T == TileShapeType::N) {
            assert(index == 0 || index == 1);
        } else if constexpr (T == TileShapeType::K) {
            assert(index >= 0 && index < static_cast<int>(MAX_KDIM_SIZE));
        }
        int v;
        if constexpr (T == TileShapeType::M) {
            v = m[index];
        } else if constexpr (T == TileShapeType::K) {
            v = k[index];
        } else if constexpr (T == TileShapeType::N) {
            v = n[index];
        } else {
            assert(false);
        }
        assert(v > 0);
        return v;
    }

    int GetTileShape(TileShapeType type, int index) const {
        if (type == TileShapeType::M || type == TileShapeType::N) {
            assert(index == 0 || index == 1);
        } else if (type == TileShapeType::K) {
            assert(index >= 0 && index < static_cast<int>(MAX_KDIM_SIZE));
        }
        int v = 0;
        if (type == TileShapeType::M) {
            v = m[index];
        } else if (type == TileShapeType::K) {
            v = k[index];
        } else if (type == TileShapeType::N) {
            v = n[index];
        } else {
            assert(false);
        }
        assert(v > 0);
        return v;
    }

    bool SetL1Tile() const {
        return setL1Tile;
    }

    void SerializeTo(HashBuffer &buffer) const {
        for (size_t i = 0; i < 0x2; i++) {
            buffer.Append(m[i]);
        }
        for (size_t i = 0; i < 0x3; i++) {
            buffer.Append(k[i]);
        }
        for (size_t i = 0; i < 0x2; i++) {
            buffer.Append(n[i]);
        }
    }

    static CubeTileShapes DeserializeFrom(const HashBuffer &buffer) {
        CubeTileShapes result;
        result.m[0] = buffer.Get<int64_t>(0);                 // offset 0.  m[0]
        result.m[1] = buffer.Get<int64_t>(2);                 // offset 2,  m[1]
        result.k[0] = buffer.Get<int64_t>(4);                 // offset 4,  k[0]
        result.k[1] = buffer.Get<int64_t>(6);                 // offset 6,  k[1]
        result.k[MAX_KDIM_SIZE - 1] = buffer.Get<int64_t>(8); // offset 8,  k[2]
        result.n[0] = buffer.Get<int64_t>(10);                // offset 10, n[0]
        result.n[1] = buffer.Get<int64_t>(12);                // offset 12, n[1]
        return result;
    }

    bool TileShapeAvaliable() {
        bool mZero = std::all_of(m.begin(), m.end(), [](int x) { return x == 0; });
        bool kZero = std::all_of(n.begin(), n.end(), [](int x) { return x == 0; });
        bool nZero = std::all_of(k.begin(), k.end(), [](int x) { return x == 0; });
        return (mZero == false) && (kZero == false) && (nZero == false);
    }

private:
    std::array<int64_t, MAX_MDIM_SIZE> m{0, 0};
    std::array<int64_t, MAX_KDIM_SIZE> k{0, 0};
    std::array<int64_t, MAX_NDIM_SIZE> n{0, 0};
    bool setL1Tile;
};

class DistTileShapes {
public:
    DistTileShapes() = default;
    DistTileShapes(std::array<int, MAX_DIST_DIM_SIZE> row, std::array<int, MAX_DIST_DIM_SIZE> col,
                   std::array<int, MAX_DIST_DIM_SIZE> rank):
        row_(row), col_(col), rank_(rank) {}

    void SetRowTileShapes(std::array<int, MAX_DIST_DIM_SIZE> row) {
        row_ = row;
    }

    void SetColTileShapes(std::array<int, MAX_DIST_DIM_SIZE> col) {
        col_ = col;
    }

    void SetRankTileShapes(std::array<int, MAX_DIST_DIM_SIZE> rank) {
        rank_ = rank;
    }

    void SpecifyStaticRankId(int rankId) {
       rankId_ = rankId;
    }

    void SerializeTo(HashBuffer &buffer) const {
        buffer.Append(row_);
        buffer.Append(col_);
        buffer.Append(rank_);
        buffer.Append(rankId_);
    }

    static DistTileShapes DeserializeFrom(const HashBuffer &buffer) {
        DistTileShapes result;
        result.row_[0x0] = buffer[0x0];
        result.row_[0x1] = buffer[0x1];
        result.row_[0x2] = buffer[0x2];
        result.col_[0x0] = buffer[0x3];
        result.col_[0x1] = buffer[0x4];
        result.col_[0x2] = buffer[0x5];
        result.rank_[0x0] = buffer[0x6];
        result.rank_[0x1] = buffer[0x7];
        result.rank_[0x2] = buffer[0x8];
        result.rankId_ = buffer[0x9];
        return result;
    }

    bool TileShapeAvaliable() {
        bool rowZero = std::all_of(row_.begin(), row_.end(), [](int x) { return x == 0; });
        bool colZero = std::all_of(col_.begin(), col_.end(), [](int x) { return x == 0; });
        bool rankZero = std::all_of(rank_.begin(), rank_.end(), [](int x) { return x == 0; });
        return (rowZero == false) && (colZero == false) && (rankZero == false);
    }

    // row/col/rank轴切分，格式[size, count, tail]，表示：
    // 该轴从头开始，按照size依次切分为count块，剩余tail大小的尾块
    // 该轴的总size等于size * count + tail的值
    std::array<int, MAX_DIST_DIM_SIZE> row_{0, 0, 0};
    std::array<int, MAX_DIST_DIM_SIZE> col_{0, 0, 0};
    std::array<int, MAX_DIST_DIM_SIZE> rank_{0, 0, 0};
    int rankId_ {INT16_MAX}; // 指定本卡rankId，根据rankId做静态切分，不能为-1，JSON解析不支持
};

class TileShape {
public:
    template <typename... Args>
    void SetVecTileShapes(Args &&...args) {
        vecTileShapes_ = std::vector<int64_t>{args...};
    }

    void SetVecTileShapes(const std::vector<int64_t>& tileShape) {
        vecTileShapes_ = tileShape;
    }

    void SetCubeTileShapes(std::array<int64_t, MAX_MDIM_SIZE> m, std::vector<int64_t> k, std::array<int64_t, MAX_NDIM_SIZE> n,
        bool setL1Tile = false) {
        if (k.size() == MAX_KDIM_SIZE) {
            std::array<int64_t, MAX_KDIM_SIZE> kc = {k[0], k[1], k[2]};
            cubeTileShapes_ = CubeTileShapes(m, kc, n, setL1Tile);
        } else {
            std::array<int64_t, MAX_KDIM_SIZE> kc = {k[0], k[1], k[1]};
            cubeTileShapes_ = CubeTileShapes(m, kc, n, setL1Tile);
        }
    }

    void SetDistTileShapes(std::array<int, MAX_DIST_DIM_SIZE> row,
                           std::array<int, MAX_DIST_DIM_SIZE> col,
                           std::array<int, MAX_DIST_DIM_SIZE> rank) {
        distTileShapes_.SetRowTileShapes(row);
        distTileShapes_.SetColTileShapes(col);
        distTileShapes_.SetRankTileShapes(rank);
    }

    void SetDistTileShapes(std::array<int, MAX_DIST_DIM_SIZE> row) {
        distTileShapes_.SetRowTileShapes(row);
    }

    void SpecifyStaticRankId(int rankId) {
        distTileShapes_.SpecifyStaticRankId(rankId);
    }

    void Reset() {
        vecTileShapes_.clear();
        cubeTileShapes_ = CubeTileShapes({0, 0}, {0, 0, 0}, {0, 0});
        distTileShapes_ = DistTileShapes({0, 0, 0}, {0, 0, 0}, {0, 0, 0});
    }
    [[nodiscard]] std::string Dump(bool dumpCube = false) const {
        if (vecTileShapes_.empty()) {
            return "";
        }
        std::ostringstream oss;

        if (dumpCube) {
            oss << "Tile Shapes: ";
            oss << " m: [" << cubeTileShapes_.GetTileShape<TileShapeType::M>(0) << ", " << cubeTileShapes_.GetTileShape<TileShapeType::M>(1) << "], ";
            oss << " k: [" << cubeTileShapes_.GetTileShape<TileShapeType::K>(0) << ", " << cubeTileShapes_.GetTileShape<TileShapeType::K>(1) << ", " << cubeTileShapes_.GetTileShape<TileShapeType::K>(MAX_KDIM_SIZE - 1) << "], ";
            oss << " n: [" << cubeTileShapes_.GetTileShape<TileShapeType::N>(0) << ", " << cubeTileShapes_.GetTileShape<TileShapeType::N>(1) << "], ";
        } else {
            oss << "Tile Shapes: [";
            for (size_t i = 0; i < vecTileShapes_.size(); i++) {
                oss << vecTileShapes_[i];
                if (i != vecTileShapes_.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
        }

        return oss.str();
    }

    template <TileShapeType T = TileShapeType::NORMAL>
    void SetTileShape(size_t index, int value) {
        assert(value > 0);
        static_assert(T < TileShapeType::TYPE_NUM);
        if constexpr (T == TileShapeType::NORMAL) {
            assert(index < vecTileShapes_.size());
            vecTileShapes_[index] = value;
            return;
        }
        assert(index < MAX_KDIM_SIZE);
        cubeTileShapes_.SetTileShape<T>(index, value);
    }

    [[nodiscard]] int64_t V(size_t index) const {
        assert(index < vecTileShapes_.size());
        auto value = vecTileShapes_[index];
        return value;
    }

    [[nodiscard]] int M(int index) const { return cubeTileShapes_.GetTileShape<TileShapeType::M>(index); }

    [[nodiscard]] int K(int index) const { return cubeTileShapes_.GetTileShape<TileShapeType::K>(index); }

    [[nodiscard]] int N(int index) const { return cubeTileShapes_.GetTileShape<TileShapeType::N>(index); }

    [[nodiscard]] bool SetL1Tile() const { return cubeTileShapes_.SetL1Tile(); }

    [[nodiscard]] const VecTileShapes &GetVecTileShapes() const { return vecTileShapes_; }

    [[nodiscard]] const CubeTileShapes &GetCubeTileShapes() const { return cubeTileShapes_; }

    [[nodiscard]] const std::array<int, MAX_DIST_DIM_SIZE>& GetDistTileRow() const { return distTileShapes_.row_; }

    [[nodiscard]] const std::array<int, MAX_DIST_DIM_SIZE>& GetDistTileCol() const { return distTileShapes_.col_; }

    [[nodiscard]] const std::array<int, MAX_DIST_DIM_SIZE>& GetDistTileRank() const { return distTileShapes_.rank_; }

    [[nodiscard]] int GetDistRankId() const { return distTileShapes_.rankId_; }

    void SerializeTo(HashBuffer &vecBuffer, HashBuffer &cubeBuffer, HashBuffer &distBuffer) const {
        vecBuffer.assign(vecTileShapes_.begin(), vecTileShapes_.end());
        cubeTileShapes_.SerializeTo(cubeBuffer);
        distTileShapes_.SerializeTo(distBuffer);
    }

    static TileShape DeserializeFrom(const HashBuffer &vecBuffer, const HashBuffer &cubeBuffer, const HashBuffer &distBuffer) {
        TileShape result;
        result.vecTileShapes_.assign(vecBuffer.begin(), vecBuffer.end());
        result.cubeTileShapes_ = CubeTileShapes::DeserializeFrom(cubeBuffer);
        result.distTileShapes_ = DistTileShapes::DeserializeFrom(distBuffer);
        return result;
    }

    bool TileShapeAvaliable() {
        return cubeTileShapes_.TileShapeAvaliable() ||
            vecTileShapes_.size() > 0 ||
            distTileShapes_.TileShapeAvaliable();
    }

    static TileShape &Current();

private:
    VecTileShapes vecTileShapes_;
    CubeTileShapes cubeTileShapes_;
    DistTileShapes distTileShapes_;
};

class MatrixSize {
 public:
    int Size() { return matrixSize_.size(); }
    void SetMatrixSize(const std::vector<int64_t>& size) {
        matrixSize_ = size;
    }
    [[nodiscard]] int V(size_t index) const {
        assert(index < matrixSize_.size());
        return matrixSize_[index];
    }

    static MatrixSize &Current();
 private:
    std::vector<int64_t> matrixSize_;
};

} // namespace npu::tile_fwk
