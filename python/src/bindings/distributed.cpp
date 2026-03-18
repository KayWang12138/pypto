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

using namespace npu::tile_fwk;

namespace pypto {
void BindDistributed(py::module& m) {
    m.def(
        "CreateShmemData",
        [](const char* group, int64_t worldSize, DataType dataType, const Shape& shape, Tensor& shmemTensor,
            uint64_t memType = 0) {
            return Distributed::CreateShmemData(group, worldSize, dataType, shape, shmemTensor, memType);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("dataType"), py::arg("shape"), py::arg("shmemTensor"),
        py::arg("memType") = 0, "Create shmem data.");

    m.def(
        "CreateShmemSignal",
        [](const char* group, Tensor& shmemData, Tensor& shmemSignal) {
            return Distributed::CreateShmemSignal(group, shmemData, shmemSignal);
        },
        py::arg("group"), py::arg("shmemData"), py::arg("shmemSignal"), "Create shmem signal data.");

    m.def(
        "ShmemBarrier",
        [](const Tensor& predToken, Tensor& shmemSignal, const char* group, uint32_t worldSize) {
            return Distributed::ShmemBarrier(predToken, shmemSignal, group, worldSize);
        },
        py::arg("predToken"), py::arg("shmemSignal"), py::arg("group"), py::arg("worldSize"),
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
        "GetSymbolicScalarPeId", [](std::string group) { return GetHcclRankId(group); }, py::arg("group"),
        "Get local rank id by groupname.");

    py::class_<ShmemTensor>(m, "ShmemTensor")
        .def(py::init<>())
        .def_readwrite("group", &ShmemTensor::group)
        .def_readwrite("worldSize", &ShmemTensor::worldSize)
        .def_readwrite("data", &ShmemTensor::data)
        .def_readwrite("signal", &ShmemTensor::signal);

    m.def(
        "CreateShmemData",
        [](const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor &t) {
            return Distributed::CreateShmemData(group, worldSize, dataType, shape, t);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("dataType"), py::arg("shape"), py::arg("t"), "Create shmem data.");

    m.def(
        "CreateShmemSignal",
        [](const char* group, int64_t worldSize, ShmemTensor &t) {
            return Distributed::CreateShmemSignal(group, worldSize, t);
        },
        py::arg("group"), py::arg("worldSize"),  py::arg("t"), "Create shmem signal data.");

    m.def(
        "ShmemView",
        [](const ShmemTensor &operand, const std::vector<int64_t> &shapes, const py::sequence &offsets) {
            bool has_symbolic = false;
            for (const auto &item : offsets) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_offsets;
                symbolic_offsets.reserve(py::len(offsets));
                for (const auto &item : offsets) {
                    symbolic_offsets.push_back(item.cast<SymbolicScalar>());
                }
                return Distributed::ShmemView(operand, shapes, symbolic_offsets);
            } else {
                std::vector<int64_t> int_offsets;
                int_offsets.reserve(py::len(offsets));
                for (const auto &item : offsets) {
                    int_offsets.push_back(item.cast<int64_t>());
                }
                return Distributed::ShmemView(operand, shapes, int_offsets);
            }
        },
        py::arg("operand"), py::arg("shapes"), py::arg("offsets"), "Create shmem view.");

    m.def(
        "ShmemView",
        [](const ShmemTensor &operand, const std::vector<int64_t> &shapes, const py::sequence &newValidShapes,
            const py::sequence &newOffsets) {
            std::vector<SymbolicScalar> symbolic_newValidShapes;
            symbolic_newValidShapes.reserve(py::len(newValidShapes));
            for (const auto &item : newValidShapes) {
                symbolic_newValidShapes.push_back(item.cast<SymbolicScalar>());
            }
            std::vector<SymbolicScalar> symbolic_offsets;
            symbolic_offsets.reserve(py::len(newOffsets));
            for (const auto &item : newOffsets) {
                symbolic_offsets.push_back(item.cast<SymbolicScalar>());
            }
            return Distributed::ShmemView(operand, shapes, symbolic_newValidShapes, symbolic_offsets);
        },
        py::arg("operand"), py::arg("shapes"), py::arg("newValidShapes"), py::arg("newOffsets"),
        "Create shmem view with valid shapes and offsets.");

    m.def(
        "ShmemPut",
        [](const Tensor &src, const ShmemTensor &dst, const SymbolicScalar &dstRank, Distributed::AtomicType putOp,
            const Tensor& pred) {
            return Distributed::ShmemPut(src, dst, dstRank, putOp, pred);
        },
        py::arg("src"), py::arg("dst"), py::arg("dstRank"), py::arg("putOp"), py::arg("pred"),
        "Put tensor to shmem with rank.");

    m.def(
        "ShmemGet",
        [](const ShmemTensor &src, const SymbolicScalar &srcRank, const Tensor& pred,
            DataType targetDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemGet(src, srcRank, pred, targetDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("targetDataType") = DataType::DT_BOTTOM,
        "Get shmem data with rank.");

    m.def(
        "ShmemSignal",
        [](const ShmemTensor& dst, const SymbolicScalar &dstRank, int32_t signal, Distributed::AtomicType sigOp,
            const Tensor& pred) {
            return Distributed::ShmemSignal(dst, dstRank, signal, sigOp, pred);
        },
        py::arg("dst"), py::arg("dstRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal shmem with rank.");

    m.def(
        "ShmemSignal",
        [](const ShmemTensor& dst, const SymbolicScalar &dstRank, const SymbolicScalar &consumerRank, int32_t signal,
            Distributed::AtomicType sigOp, const Tensor& pred) {
            return Distributed::ShmemSignal(dst, dstRank, consumerRank, signal, sigOp, pred);
        },
        py::arg("dst"), py::arg("dstRank"), py::arg("consumerRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal shmem with consumer rank.");

    m.def(
        "ShmemSignalAll",
        [](const ShmemTensor& dst, const SymbolicScalar &dstRank, int32_t signal, Distributed::AtomicType sigOp,
            const Tensor& pred) {
            return Distributed::ShmemSignalAll(dst, dstRank, signal, sigOp, pred);
        },
        py::arg("dst"), py::arg("dstRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal all ranks in shmem.");

    m.def(
        "ShmemWaitUntil",
        [](const ShmemTensor& src, const SymbolicScalar &srcRank, OpType cmp, int32_t cmpValue,
            bool clearSignal, const Tensor &pred) {
            return Distributed::ShmemWaitUntil(src, srcRank, cmp, cmpValue, clearSignal, pred);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("cmp"), py::arg("cmpValue"),
        py::arg("clearSignal"), py::arg("pred"), "Wait shmem signal.");

    m.def(
        "ShmemClearData",
        [](const ShmemTensor& src, Tensor &pred) {
            return Distributed::ShmemClearData(src, pred);
        },
        py::arg("src"), py::arg("pred"), "Clear shmem data.");

    m.def(
        "ShmemClearSignal",
        [](const ShmemTensor& src, Tensor &pred) {
            return Distributed::ShmemClearSignal(src, pred);
        },
        py::arg("src"), py::arg("pred"), "Clear shmem signal.");

    m.def(
        "ShmemBarrier",
        [](const ShmemTensor &src, const Tensor &pred) {
            return Distributed::ShmemBarrier(src, pred);
        },
        py::arg("src"), py::arg("pred"), "Barrier on shmem.");

    m.def(
        "ShmemLoad",
        [](const ShmemTensor &src, const SymbolicScalar &srcRank, const Tensor& pred,
            DataType nonShmemDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemLoad(src, srcRank, pred, nonShmemDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("nonShmemDataType") = DataType::DT_BOTTOM,
        "Load shmem data.");
}

} // namespace pypto