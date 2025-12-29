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
 * \file test_allreduce_add_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"
#include "tilefwk/symbolic_distributed.h"

namespace npu::tile_fwk::Distributed {

Tensor Nop(const std::vector<Tensor>& inTensors)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    LogicalTensors iOperands;
    for (const Tensor& inTensor : inTensors) {
        iOperands.emplace_back(inTensor.GetStorage());
    }
    function.AddOperation(Opcode::OP_NOP, iOperands, {out});
    return out;
}

template<typename T>
void TestShmemAllReduceAddAllReduce(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    Shape shape{row, col};
    DataType dType = GetDataTypeNum(typeNum);

    Tensor in(dType, shape, "in");
    Tensor allReduceOut(dType, shape, "allReduceOut");
    Tensor addOut(dType, shape, "addOut");
    Tensor out(dType, shape, "out");
    Tensor barrier1Out(DT_INT32, {1, 1}, "barrier1Out");
    Tensor memSetOut(DT_INT32, {1, 1}, "memSetOut");
    Tensor barrier2Out(DT_INT32, {1, 1}, "barrier2Out");

    std::vector<T> inPtr = ReadToVector<T>(GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin",
        shape);

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<T>(in, inPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    FUNCTION("AllReduceAddAllReduce", {in}, {out}) {
        LOOP("AllReduce1", FunctionType::DYNAMIC_LOOP, allReduce1Index, LoopRange(0, 1, 1)) {
            (void)allReduce1Index;
            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            OneShotShmemAllReduce(in, testParam.group, allReduceOut);
        }
        LOOP("Add", FunctionType::DYNAMIC_LOOP, index, LoopRange(0, 1, 1)) {
            (void)index;
            TileShape::Current().SetVecTile({128, 256});
            addOut = npu::tile_fwk::Add(allReduceOut, allReduceOut);
        }
        Tensor shmemBarrier1ShmemSignal;
        Tensor shmemBarrier2ShmemSignal;
        Tensor allReduce2ShmemData;
        Tensor allReduce2ShmemSignal;
        int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(testParam.group)));
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(shmemBarrier1ShmemSignal, testParam.rankSize, hcclGroupIndex, DT_INT32, Shape{1, 1, 8},
                1);
            CreateShmemTensor(shmemBarrier2ShmemSignal, testParam.rankSize, hcclGroupIndex, DT_INT32, Shape{1, 1, 8},
                1);
            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            int32_t tileCount = 1;
            Shape allReduce2ShmemDataShape = {1, addOut.GetShape(0), addOut.GetShape(1)};
            Shape allReduce2ShmemSignalShape = {1, tileCount, 8};
            DataType allReduce2ShmemDataType = in.GetDataType();
            if ((allReduce2ShmemDataType == DT_BF16) || (allReduce2ShmemDataType == DT_FP16)) {
                allReduce2ShmemDataType = DT_FP32;
            }
            CreateShmemTensor(allReduce2ShmemData, testParam.rankSize, hcclGroupIndex, allReduce2ShmemDataType,
                allReduce2ShmemDataShape);
            CreateShmemTensor(allReduce2ShmemSignal, testParam.rankSize, hcclGroupIndex, DT_INT32,
                allReduce2ShmemSignalShape);
        }
        LOOP("AllReduce2", FunctionType::DYNAMIC_LOOP, allReduce2Index, LoopRange(0, 1, 1)) {
            (void)allReduce2Index;

            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            int32_t tileCount = 1;
            SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

            ShmemBarrier(addOut, shmemBarrier1ShmemSignal, testParam.group, barrier1Out);
            LOOP("ShmemSet", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
                (void)index;
                auto allReduce2ShmemDataTile = View(allReduce2ShmemData, {1, 1, addOut.GetShape(0), addOut.GetShape(1)},
                    std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
                auto memSetDataOut = ShmemSet(barrier1Out, allReduce2ShmemDataTile);
                auto allReduce2ShmemSignalTile = View(allReduce2ShmemSignal, {1, 1, tileCount, 8},
                    std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
                auto memSetSignalOut = ShmemSet(barrier1Out, allReduce2ShmemSignalTile);
                memSetOut = Nop({memSetDataOut, memSetSignalOut});
            }
            ShmemBarrier(memSetOut, shmemBarrier2ShmemSignal, testParam.group, barrier2Out);
            OneShotShmemAllReduce(barrier2Out, addOut, allReduce2ShmemData, allReduce2ShmemSignal, testParam.group,
                out);
        }
    }

    auto hcclContext = GetHcclContext({std::string(testParam.group)});
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto output = ProgramData::GetInstance().GetOutputData(0);
    int32_t outSize = row * col;
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/out_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestShmemAllReduceAddAllReduce<int32_t>(OpTestParam& testParam);
template void TestShmemAllReduceAddAllReduce<float>(OpTestParam& testParam);
template void TestShmemAllReduceAddAllReduce<float16>(OpTestParam& testParam);
template void TestShmemAllReduceAddAllReduce<bfloat16>(OpTestParam& testParam);

} // namespace npu::tile_fwk::Distributed