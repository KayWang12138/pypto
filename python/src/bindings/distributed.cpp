/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file distributed.cpp
* \brief
*/

#include "pybind_common.h"
#include "machine/runtime/distributed/distributed_context.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindDistributed(py::module& m) {
    m.def(
        "CreateShmemData",
        [](const Tensor& commTensor, int64_t worldSize, DataType dataType, const Shape& shape, Tensor& shmemTensor,
            uint64_t memType = 0) {
            return Distributed::CreateShmemData(commTensor, worldSize, dataType, shape, shmemTensor, memType);
        },
        py::arg("commTensor"), py::arg("worldSize"), py::arg("dataType"), py::arg("shape"), py::arg("shmemTensor"),
        py::arg("memType") = 0, "Create shmem data.");

    m.def(
        "CreateShmemSignal",
        [](const Tensor& commTensor, Tensor& shmemData, Tensor& shmemSignal) {
            return Distributed::CreateShmemSignal(commTensor, shmemData, shmemSignal);
        },
        py::arg("commTensor"), py::arg("shmemData"), py::arg("shmemSignal"), "Create shmem signal data.");

    m.def(
        "ShmemBarrier",
        [](const Tensor& predToken, Tensor& shmemSignal, const Tensor& commTensor, uint32_t worldSize) {
            return Distributed::ShmemBarrier(predToken, shmemSignal, commTensor, worldSize);
        },
        py::arg("predToken"), py::arg("shmemSignal"), py::arg("commTensor"), py::arg("worldSize"),
        "Sync shared memory acess between comm operators.");

    m.def(
        "ShmemDataSet",
        [](const Tensor& predToken, const Tensor& shmemData) {
            return Distributed::ShmemDataSet(predToken, shmemData);
        },
        py::arg("predToken"), py::arg("shmemData"), "Clear shmem data.");
    
    m.def(
        "ShmemSignalSet",
        [](const Tensor& predToken, const Tensor& shmemSignal) {
            return Distributed::ShmemSignalSet(predToken, shmemSignal);
        },
        py::arg("predToken"), py::arg("shmemSignal"), "Clear shmem signal.");

    m.def(
        "ShmemPut",
        [](const Tensor& predToken, const Tensor& in, const Tensor& shmemData, 
            Distributed::AtomicType atomicType = Distributed::AtomicType::SET) {
            return Distributed::ShmemPut(predToken, in, shmemData, atomicType);
        },
        py::arg("predToken"), py::arg("in"), py::arg("shmemData"),
        py::arg("atomicType") = Distributed::AtomicType::SET, "Put gm data to shmem.");
    
    m.def(
        "ShmemPutUb2Gm",
        [](const Tensor& in, const Tensor& shmemDataTile, const Tensor& barrierDummy,
            Distributed::AtomicType atomicType = Distributed::AtomicType::SET) {
            return Distributed::ShmemPutUb2Gm(in, shmemDataTile, barrierDummy, atomicType);
        },
        py::arg("in"), py::arg("shmemDataTile"), py::arg("barrierDummy"),
        py::arg("atomicType") = Distributed::AtomicType::SET, "Put gm data to shmem.");

    m.def(
        "ShmemGet",
        [](const Tensor& predToken, const Tensor& shmemData, DataType nonShmemDataType = DataType::DT_BOTTOM,
            Distributed::AtomicType atomicType = Distributed::AtomicType::SET) {
            return Distributed::ShmemGet(predToken, shmemData, nonShmemDataType, atomicType);
        },
        py::arg("predToken"), py::arg("shmemData"), py::arg("nonShmemDataType") = DataType::DT_BOTTOM,
        py::arg("atomicType") = Distributed::AtomicType::SET, "Get shmem data to gm.");

    m.def(
        "ShmemGetGm2Ub",
        [](const Tensor& dummy, const Tensor& shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
            Distributed::AtomicType atomicType = Distributed::AtomicType::SET) {
            return Distributed::ShmemGetGm2Ub(dummy, shmemDataTile, nonShmemDataType, atomicType);
        },
        py::arg("dummy"), py::arg("shmemDataTile"), py::arg("nonShmemDataType") = DataType::DT_BOTTOM,
        py::arg("atomicType") = Distributed::AtomicType::SET, "Get shmem data to ub.");

    m.def(
        "ShmemSignal",
        [](const Tensor& predToken, const Tensor& shmemSignal, Distributed::AtomicType atomicType) {
            return Distributed::ShmemSignal(predToken, shmemSignal, atomicType);
        },
        py::arg("predToken"), py::arg("shmemSignal"), py::arg("atomicType"), "Set shmem signal data.");

    m.def(
        "WaitUntil",
        [](const Tensor& predToken, const Tensor& shmemSignal, int32_t cmpValue, bool resetSignal = false) {
            return Distributed::WaitUntil(predToken, shmemSignal, cmpValue, resetSignal);
        },
        py::arg("predToken"), py::arg("shmemSignal"), py::arg("cmpValue"), py::arg("resetSignal") = false,
        "Wait signal data.");

    m.def(
        "GetSymbolicScalarPeId", [](const Tensor& commTensor) { return GetHcclRankIdV2(commTensor); }, py::arg("commTensor"),
        "Get local rank id by groupname.");
    m.def(
        "GetCommContext", 
        [](const std::string& group) -> std::pair<int64_t, uint64_t> { 
            std::vector<uint64_t> hcclContexts = dynamic::DistributedContext::GetCommContextToHost(std::vector<std::string>{group});
            auto hcclOpParam = (TileOp::CommContext*)hcclContexts[0];
            auto rankNum = hcclOpParam->rankNum;
            int64_t ctxSize = static_cast<int64_t>(sizeof(TileOp::CommContext)) + 
                static_cast<int64_t>(sizeof(uint64_t)) * rankNum * dynamic::WIN_TYPE_NUM;
            auto commContext = dynamic::DistributedContext::GetCommContext(std::vector<std::string>{group});
            return {ctxSize, commContext[0]}; 
        }, 
        py::arg("group"),
        "Get CommContextHost addr");
}

} // namespace pypto