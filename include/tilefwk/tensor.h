/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file tensor.h
 * \brief Basic methods of the Tensor class.
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
    /**
     * \brief Constructs a new default Tensor object
     * 
     */
    Tensor();
    /**
     * \brief Destroy the Tensor object
     * 
     */
    ~Tensor();

    /**
     * \brief Constructs a new Tensor object with one input parameter
     * 
     * \param s : a shared pointer to a LogicalTensor object
     */
    Tensor(std::shared_ptr<LogicalTensor> s);

    /**
     * \brief Construct a new Tensor object with 5 input parameters
     * 
     * \param t : Data type of the tensor.
     * \param tshape : A vector that stores the shape of the tensor.
     * \param tname : Name of the tensor. The default value is "".
     * \param tnodetype : The type of the node. The default value is NodeType::LOCAL.
     * \param tensorfmt : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     * \attention : The parameters tand tshape are required parameters.
     */
    Tensor(DataType t, std::vector<int> tshape, std::string tname = "", NodeType tnodetype = NodeType::LOCAL,
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    /**
     * \brief Construct a new Tensor object with 6 input parameters
     * 
     * \param t : Data type of the tensor.
     * \param tshape : A vector that stores the shape of the tensor.
     * \param data : Pointer to the data of the tensor.
     * \param tname : Name of the tensor.
     * \param tnodetype : Type of the node. The default value is NodeType::LOCAL.
     * \param tensorfmt : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     * \attention : The parameters t,tshape,data and tname are required parameters.
     */
    Tensor(DataType t, std::vector<int> tshape, uint8_t *data, std::string tname, NodeType tnodetype = NodeType::LOCAL,
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND)
        : Tensor(t, tshape, tname, tnodetype, tensorfmt) {
        SetData(data);
    }

    /**
     * \brief Construct a new Tensor object with 5 input parameters
     * 
     * \param t : Data type of the tensor.
     * \param tshape : A vector that stores the shape of the tensor.
     * \param dynDims : A vector that stores the dynamic dimensions of the tensor.
     * \param tname : Name of the tensor. The default value is "".
     * \param tensorfmt : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     */
    Tensor(DataType t, std::vector<int> tshape, std::vector<int> dynDims, std::string tname = "",
        TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    /**
     * \brief Construct a new Tensor object with 5 input parameters
     * 
     * \param rawtensor : A shared pointer to a RawTensor object.
     * \param toffset : A vector that stores the offset of the tensor.
     * \param tshape : A vector that stores the shape of the tensor.
     * \param tnodetype : The type of the node. The default value is NodeType::LOCAL.
     * \param tensorfmt : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     * \attention : The parameters rawtensor,toffset and tshape are required parameters.
     */
    Tensor(std::shared_ptr<RawTensor> rawtensor, std::vector<int> toffset, std::vector<int> tshape,
        NodeType tnodetype = NodeType::LOCAL, TileOpFormat tensorfmt = TileOpFormat::TILEOP_ND);

    /**
     * \brief Overload the assignment operator to assign the value of another Tensor object to the current Tensor object.
     * 
     * \param rhs : A constant reference to another Tensor object.
     * \return Tensor& : A reference to the current Tensor object.
     */
    Tensor &operator=(const Tensor &rhs);

    /**
     * \brief Move assignment operator.
     * 
     * \param rhs : Rvalue reference to another Tensor object.
     * \return Tensor& : A reference to the current Tensor object.
     * \attention : The noexcept declaration indicates that the function will not throw exceptions.
     */
    Tensor &operator=(Tensor &&rhs) noexcept;

    /**
     * \brief Construct a new Tensor object by copying another Tensor object.
     * 
     * \param rhs : A constant reference to another Tensor object.
     */
    Tensor(const Tensor &rhs);

    /**
     * \brief Construct a new Tensor object by moving another Tensor object.
     * 
     * \param rhs : Rvalue reference to another Tensor object.
     */
    Tensor(Tensor &&rhs);

    /**
     * \brief Overload the -> operator to access the members of the LogicalTensor object.
     * 
     * \return const LogicalTensor* : A pointer to the LogicalTensor object.
     * \attention : The const keyword indicates that the function does not modify the object.
     */
    const LogicalTensor *operator->() const;

    /**
     * \brief Overload the -> operator to access the members of the LogicalTensor object.
     * 
     * \return LogicalTensor* : A pointer to the LogicalTensor object.
     */
    LogicalTensor *operator->();

    /**
     * \brief Overload the * operator to access the LogicalTensor object.
     * 
     * \return const LogicalTensor& : A reference to the LogicalTensor object.
     * \attention : The const keyword indicates that the function does not modify the object.
     */
    const LogicalTensor &operator*() const;

    /**
     * \brief Overload the * operator to access the LogicalTensor object.
     *
     * \return LogicalTensor& : A reference to the LogicalTensor object.
     */
    LogicalTensor &operator*();

    /**
     * \brief Get the const Storage object.
     * 
     * \param readSlot : This parameter indicates whether slot reading is required. The default value is true.
     * \return const std::shared_ptr<LogicalTensor>& : A constant reference to the storage object.
     * \attention : The const keyword indicates that the function does not modify the object.
     */
    const std::shared_ptr<LogicalTensor> &GetStorage(bool readSlot = true) const;

    /**
     * \brief Get the Storage object.
     * 
     * \param readSlot : This parameter indicates whether slot reading is required. The default value is true.
     * \return std::shared_ptr<LogicalTensor>& : A reference to the storage object.
     */
    std::shared_ptr<LogicalTensor> &GetStorage(bool readSlot = true);

    // Mark this tensor do L2 Prefetch. (Now max prefetch num is 4.)
    /**
     * \brief Prefetch the tensor to L2 cache.
     * 
     * \param preloadDep : This parameter is used to control the timing of L2 prefetching for this Tensor. The default value is 0.
     * \attention : This parameter is still in its infancy and has no actual function.
     */
    void Prefetch(int preloadDep = 0);

    /**
     * \brief Get the Data Type object
     * 
     * \return DataType : The data type of the tensor.
     */
    DataType GetDataType() const;

    /**
     * \brief Get the shape of a tensor.
     * 
     * \return const std::vector<int>& : A constant reference to the shape of the tensor.
     */
    const std::vector<int> &GetShape() const;

    /**
     * \brief Get the shape information of the specified axis of Tensor.
     * 
     * \param axis : The axis of the shape to be obtained.
     * \return int : The shape of the specified axis.
     */
    int GetShape(int axis) const;

    /**
     * \brief Get the Id information of the Tensor.
     * 
     * \return int : The Id information of the Tensor.
     */
    int Id() const { return index_; }

    /**
     * \brief Set the data of Tensor.
     * 
     * \param data : Pointer to the data of the tensor. The data type is uint8_t.
     */
    void SetData(BinDataPtr data);

    /**
     * \brief Get the data of Tensor.
     * 
     * \return auto : A pointer to the data of the tensor.
     */
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
