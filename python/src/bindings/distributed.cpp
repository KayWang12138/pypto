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
#include <pybind11/numpy.h>
#include "acl/acl.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::Distributed;

namespace pypto {
void BindDistributed(py::module& m)
{
    py::class_<ShmemTensor>(m, "ShmemTensor")
        .def(py::init<>())
        .def_readwrite("group", &ShmemTensor::group)
        .def_readwrite("worldSize", &ShmemTensor::worldSize)
        .def_readwrite("data", &ShmemTensor::data)
        .def_readwrite("signal", &ShmemTensor::signal);

    m.def(
        "CreateShmemTensor",
        [](const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor& t) {
            return Distributed::CreateShmemTensor(group, worldSize, dataType, shape, t);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("dataType"), py::arg("shape"), py::arg("t"),
        "Create shmem data.");

    m.def(
        "CreateShmemSignal",
        [](const char* group, int64_t worldSize, ShmemTensor& t) {
            return Distributed::CreateShmemSignal(group, worldSize, t);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("t"), "Create shmem signal data.");

    m.def(
        "ShmemView",
        [](const ShmemTensor& operand, const std::vector<int64_t>& shapes, const py::sequence& offsets) {
            bool has_symbolic = false;
            for (const auto& item : offsets) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_offsets;
                symbolic_offsets.reserve(py::len(offsets));
                for (const auto& item : offsets) {
                    symbolic_offsets.push_back(item.cast<SymbolicScalar>());
                }
                return Distributed::ShmemView(operand, shapes, symbolic_offsets);
            } else {
                std::vector<int64_t> int_offsets;
                int_offsets.reserve(py::len(offsets));
                for (const auto& item : offsets) {
                    int_offsets.push_back(item.cast<int64_t>());
                }
                return Distributed::ShmemView(operand, shapes, int_offsets);
            }
        },
        py::arg("operand"), py::arg("shapes"), py::arg("offsets"), "Create shmem view.");

    m.def(
        "ShmemView",
        [](const ShmemTensor& operand, const std::vector<int64_t>& shapes, const py::sequence& newValidShapes,
           const py::sequence& newOffsets) {
            std::vector<SymbolicScalar> symbolic_newValidShapes;
            symbolic_newValidShapes.reserve(py::len(newValidShapes));
            for (const auto& item : newValidShapes) {
                symbolic_newValidShapes.push_back(item.cast<SymbolicScalar>());
            }
            std::vector<SymbolicScalar> symbolic_offsets;
            symbolic_offsets.reserve(py::len(newOffsets));
            for (const auto& item : newOffsets) {
                symbolic_offsets.push_back(item.cast<SymbolicScalar>());
            }
            return Distributed::ShmemView(operand, shapes, symbolic_newValidShapes, symbolic_offsets);
        },
        py::arg("operand"), py::arg("shapes"), py::arg("newValidShapes"), py::arg("newOffsets"),
        "Create shmem view with valid shapes and offsets.");

    m.def(
        "ShmemPut",
        [](const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, Distributed::AtomicType putOp,
           const Tensor& pred) { return Distributed::ShmemPut(src, dst, dstRank, putOp, pred); },
        py::arg("src"), py::arg("dst"), py::arg("dstRank"), py::arg("putOp"), py::arg("pred"),
        "Put tensor to shmem with rank.");

    m.def(
        "ShmemGet",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
           DataType targetDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemGet(src, srcRank, pred, targetDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("targetDataType") = DataType::DT_BOTTOM,
        "Get shmem data with rank.");

    m.def(
        "ShmemSignal",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const SymbolicScalar& targetRank, int32_t signal,
           Distributed::AtomicType sigOp,
           const Tensor& pred) { return Distributed::ShmemSignal(src, srcRank, targetRank, signal, sigOp, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("targetRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal shmem with consumer rank.");

    m.def(
        "ShmemSignalAll",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, int32_t signal, Distributed::AtomicType sigOp,
           const Tensor& pred) { return Distributed::ShmemSignalAll(src, srcRank, signal, sigOp, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal all ranks in shmem.");

    m.def(
        "ShmemWaitUntil",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, OpType cmp, int32_t cmpValue, bool clearSignal,
           const Tensor& pred) { return Distributed::ShmemWaitUntil(src, srcRank, cmp, cmpValue, clearSignal, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("cmp"), py::arg("cmpValue"), py::arg("clearSignal"),
        py::arg("pred"), "Wait shmem signal.");

    m.def(
        "ShmemClearData", [](const ShmemTensor& src, Tensor& pred) { return Distributed::ShmemClearData(src, pred); },
        py::arg("src"), py::arg("pred"), "Clear shmem data.");

    m.def(
        "ShmemClearSignal",
        [](const ShmemTensor& src, Tensor& pred) { return Distributed::ShmemClearSignal(src, pred); }, py::arg("src"),
        py::arg("pred"), "Clear shmem signal.");

    m.def(
        "ShmemBarrier", [](const ShmemTensor& src, const Tensor& pred) { return Distributed::ShmemBarrier(src, pred); },
        py::arg("src"), py::arg("pred"), "Barrier on shmem.");

    m.def(
        "ShmemLoad",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
           DataType nonShmemDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemLoad(src, srcRank, pred, nonShmemDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("nonShmemDataType") = DataType::DT_BOTTOM,
        "Load shmem data to local.");

    m.def(
        "ShmemStore",
        [](const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, Distributed::AtomicType putOp,
           const Tensor& pred) { return Distributed::ShmemStore(src, dst, dstRank, putOp, pred); },
        py::arg("src"), py::arg("dst"), py::arg("dstRank"), py::arg("putOp"), py::arg("pred"),
        "Store local tensor to shmem.");

    m.def(
        "GetSymbolicScalarPeId", [](std::string group) { return GetHcclRankId(group); }, py::arg("group"),
        "Get local rank id by groupname.");
    m.def(
    "GetCommContext", 
    [](const std::string& group) -> std::pair<int64_t, uint64_t> { 
        std::vector<uint64_t> hcclContexts = dynamic::DistributedContext::GetCommContextToHost(std::vector<std::string>{testParam.group});
        auto hcclOpParam = (TileOp::CommContext*)hcclContexts[0];
        auto rankNum = hcclOpParam->rankNum;
        int64_t ctxSize = static_cast<int64_t>(sizeof(TileOp::CommContext)) + 
            static_cast<int64_t>(sizeof(uint64_t)) * rankNum * dynamic::WIN_TYPE_NUM;
        auto commContext = dynamic::DistributedContext::GetCommContext(std::vector<string>{group});
        return {ctxSize, commContext[0]}; 
    }, 
    py::arg("group"),
    "Get CommContextHost addr");

    py::enum_<WinType>(m, "WinType")
        .value("WIN_EXP", WinType::WIN_EXP)
        .value("WIN_OUT", WinType::WIN_OUT)
        .value("WIN_IN", WinType::WIN_IN)

class HcclWin {
    public:
        HcclWin(uint64_t addr)
        {
            (void)rtMemcpy(&param_, sizeof(param_), (uint8_t*)addr, sizeof(param_), RT_MEMCPY_DEVICE_TO_HOST);
        }

        std::vector<uint8_t> GetWinValue(WinType winType, size_t byteSize, size_t count = 0UL, size_t offset = 0UL)
        {
            CHECK(byteSize == 0 !) << "byteSize must not be 0";
            auto [devAddr, winSize] = GetWinAddrAndSize(winType);
            CHECK((devAddr != 0) && (winSize != 0)) << "devAddr and winSize must not be 0";
            auto maxDataCnt = winSize / sizeof(T);
            offset = offset % maxDataCnt;
            if ((count == 0UL) || (count > maxDataCnt - offset)) {
                count = maxDataCnt - offset;
            }
            std::vector<uint8_t> result(count * byteSize, 0);
            (void)rtMemcpy(
                result.data(), result.size(), 
                reinterpret_cast<uint8_t*>(devAddr) + offset * byteSize, 
                result.size(), 
                RT_MEMCPY_DEVICE_TO_HOST);
            return result;
        }

    private:
        std::tuple<uint64_t, uint64_t> GetWinAddrAndSize(WinType winType)
        {
            uint64_t devAddr = 0UL;
            uint64_t winSize = 0UL;
            switch (winType) {
                case WinType::WIN_EXP:
                    devAddr = param_.winAddr[param_.statusIndex + param_.rankId];
                    winSize = param_.winStatusSize;
                    break;
                case WinType::WIN_OUT:
                    devAddr = param_.winAddr[param_.debugIndex + param_.rankId];
                    winSize = param_.winDebugSize;
                    break;
                case WinType::WIN_IN:
                    devAddr = param_.winAddr[param_.rankId];
                    winSize = param_.winDataSize;
                    break;
                default:
                    break;
            }
            return std::tie(devAddr, winSize);
        }

    private:
        TileOp::CommContext param_;
};

    py::class_<HcclWin>(m, "HcclWin")
        .def(py::init<uint64_t>(), py::arg("addr"))

        .def("get_win_value", [](HcclWin& self, WinType winType, size_t byte_size, size_t count, size_t offset) {
            std::vector<uint8_t> raw_data = self.GetWinValue(winType, byte_size, count, offset);
            
            // 将原始字节包装为 numpy 的 uint8 数组（零拷贝）
            py::array_t<uint8_t> py_array(raw_data.size(), raw_data.data());
            
            // 由于 vector 生命周期在函数结束销毁，必须转移内存所有权给 Python，
            // 否则会变成悬空指针引发段错误
            return py_array;
        }, 
        py::arg("win_type"), 
        py::arg("byte_size"), 
        py::arg("count") = 0, 
        py::arg("offset") = 0,
        py::return_value_policy::move, // 关键：让 Python 接管底层 vector 的内存
        R"doc(
            Read raw bytes from window memory.
            
            Args:
                win_type: Window type (WinType.WIN_IN, etc.)
                byte_size: Bytes per element (e.g., 4 for int32/float32, 8 for int64/float64)
                count: Number of elements to read (0 means read all remaining)
                offset: Element offset (not byte offset)
                
            Returns:
                numpy.ndarray of dtype uint8
        )doc");
}

} // namespace pypto
