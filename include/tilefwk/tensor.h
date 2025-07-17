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
 * \file tensor.h
 * \brief
 */

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "common/pre_def.h"
#include "common/data_type.h"

namespace npu::tile_fwk {
class Tensor {
public:
    Tensor();
    ~Tensor();

    Tensor(std::shared_ptr<LogicalTensor> s);

    Tensor(DataType t, std::vector<int> tshape, std::string tname = "", NodeType tnodetype = NodeType::LOCAL,
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    Tensor(DataType t, std::vector<int> tshape, uint8_t *data, std::string tname, NodeType tnodetype = NodeType::LOCAL,
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND)
        : Tensor(t, tshape, tname, tnodetype, tensorfmt) {
        SetData(data);
    }

    Tensor(DataType t, std::vector<int> tshape, std::vector<int> dynDims, std::string tname = "",
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    Tensor(std::shared_ptr<RawTensor> rawtensor, std::vector<int> toffset, std::vector<int> tshape,
        NodeType tnodetype = NodeType::LOCAL, TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    Tensor &operator=(const Tensor &rhs);
    Tensor &operator=(Tensor &&rhs) noexcept;
    Tensor(const Tensor &rhs);
    Tensor(Tensor &&rhs);

    const LogicalTensor *operator->() const;

    LogicalTensor *operator->();

    const LogicalTensor &operator*() const;

    LogicalTensor &operator*();

    const std::shared_ptr<LogicalTensor> &GetStorage(bool readSlot = true) const;
    std::shared_ptr<LogicalTensor> &GetStorage(bool readSlot = true);

    // Mark this tensor do L2 Prefetch. (Now max prefetch num is 4.)
    void Prefetch(int preloadDep = 0);
    DataType GetDataType() const;
    const std::vector<int> &GetShape() const;
    int GetShape(int axis) const;
    int Id() const { return index_; }
    void SetData(BinDataPtr data);
    auto GetData() const { return data_; }

private:
    std::shared_ptr<LogicalTensor> storage;
    int index_{-1};
    BinDataPtr data_{};
};

SymbolicScalar GetInputShapeDimSize(const Tensor &t);
SymbolicScalar GetInputShapeDim(const Tensor &t, int n);
SymbolicScalar GetInputDataInt32Dim1(const Tensor &t, SymbolicScalar off0);
SymbolicScalar GetInputDataInt32Dim2(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1);
SymbolicScalar GetInputDataInt32Dim3(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1, SymbolicScalar off2);
SymbolicScalar IsLoopBegin(const SymbolicScalar &symbol, const SymbolicScalar &begin);
SymbolicScalar IsLoopEnd(const SymbolicScalar &symbol, const SymbolicScalar &end);
} // namespace npu::tile_fwk
