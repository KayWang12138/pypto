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
 * \file set_heuristic_tile_shapes.cpp
 * \brief
 */

#include <climits>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "set_heuristic_tile_shapes.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Status SetHeuristicTileShapes::RunOnFunction(Function &function) {
    SetHeuristicTileShapesFunc(function);
    return SUCCESS;
}

const std::unordered_map<DataType, int64_t> LATENCY{
    {DataType::DT_FP16, 200},
    {DataType::DT_FP32, 200},
    {DataType::DT_INT32, 0},
    {DataType::DT_INT16, 0}
};

const std::unordered_map<DataType, int64_t> PARALLELISM{
    {DataType::DT_FP16, 64}, // 128B/cycle
    {DataType::DT_FP32, 32},
    {DataType::DT_INT32, 32},
    {DataType::DT_INT16, 64}
};

uint64_t GetParallelism(DataType dtype) {
    auto iterDtype = PARALLELISM.find(dtype);
    if (iterDtype == PARALLELISM.end()) {
        return DEFAULT_MAX_PARALLELISM;
    }
    return iterDtype->second;
}

uint64_t GetLatency(DataType dtype) {
    auto iterDtype = LATENCY.find(dtype);
    if (iterDtype == LATENCY.end()) {
        return DEFAULT_LATENCY;
    }
    return iterDtype->second;
}

bool IsFloat(const std::shared_ptr<LogicalTensor> tensor) {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_FP16) || (dataType == DT_FP32) || (dataType == DT_BF16)) {
        return true;
    }
    return false;
}

double CalculateGeometricMean(const std::vector<double>& vectorRatio) {
    double product = 1.0;
    for (double num : vectorRatio) {
        product *= num;
    }
    return std::pow(product, 1.0 / vectorRatio.size());
}

void SetTileFilling(std::map<std::vector<int64_t>, double>& setOfTiles, int64_t m, int64_t k, int64_t n, int64_t inputTypeSize, int64_t outputTypeSize) {
    std::vector<int64_t> tmpTile = {0,0,0}; // m,k,n
    if (((m * k * inputTypeSize) <= (L0A_MAX_SIZE / DOUBLE_BUFFER)) && ((k * n * inputTypeSize) <= (L0B_MAX_SIZE / DOUBLE_BUFFER)) && ((m * n * outputTypeSize) <= (L0C_MAX_SIZE / DOUBLE_BUFFER))) {
        tmpTile[M_DIM] = m;
        tmpTile[K_DIM] = k;
        tmpTile[N_DIM] = n;
        setOfTiles[tmpTile] = 0.f; // set initial score = 0
    }
}

void SetPossibleTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfTiles) {
    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];

    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);

    int64_t newM = int64_t(std::pow(2, int64_t(std::ceil(std::log2(std::max(M, MIN_MKN))))));
    int64_t newK = int64_t(std::pow(2, int64_t(std::ceil(std::log2(std::max(K, MIN_MKN))))));
    int64_t newN = int64_t(std::pow(2, int64_t(std::ceil(std::log2(std::max(N, MIN_MKN))))));

    for (int64_t m = MIN_MKN; m <= newM; m *= FACTOR) {
        m = m > std::max(M, MIN_MKN) ? std::max(M, MIN_MKN) : m;
        for (int64_t k = MIN_MKN; k <= newK; k *= FACTOR) {
            k = k > std::max(K, MIN_MKN) ? std::max(K, MIN_MKN) : k;
            for (int64_t n = MIN_MKN; n <= newN; n *= FACTOR) {
                n = n > std::max(N, MIN_MKN) ? std::max(N, MIN_MKN) : n;
                SetTileFilling(setOfTiles, m, k, n, inputTypeSize, outputTypeSize);
            }
        }
    }
}

void FindScoreForTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfTiles,
                       int64_t l1Reuse, int64_t cubeNBuffer, int64_t numOfMatmuls) {
    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];

    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);

    uint64_t inputMKN = std::max(M, MIN_MKN) * std::max(K, MIN_MKN) * std::max(N, MIN_MKN);
    std::vector<double> vectorRatio = {0.f, 0.f, 0.f};
    for (auto & [tile, score] : setOfTiles) {
        uint64_t mkn = tile[M_DIM] * tile[K_DIM] * tile[N_DIM];
        double m = tile[M_DIM];
        double k = tile[K_DIM];
        double n = tile[N_DIM];

        // If the tiling size = shape size -> the preferred option
        score = (tile[M_DIM] == std::max(M, MIN_MKN)) ? (score + WHOLE_M_SCORE) : score;
        score = (tile[K_DIM] == std::max(K, MIN_MKN)) ? (score + WHOLE_K_SCORE) : score;
        score = (tile[N_DIM] == std::max(N, MIN_MKN)) ? (score + WHOLE_N_SCORE) : score;

        // The more filled L0A, L0B, L0C is better
        double utilizationL0A = (m * k * inputTypeSize) / (L0A_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0B = (k * n * inputTypeSize) / (L0B_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0C = (m * n * outputTypeSize) / (L0C_MAX_SIZE / DOUBLE_BUFFER);
        vectorRatio = {utilizationL0A, utilizationL0B, utilizationL0C};
        double geomeanUtilizationL0 = CalculateGeometricMean(vectorRatio);
        score += WEIGHT_L0 * geomeanUtilizationL0;

        // The closer the tasksRatio is to 1, the better
        double tasks = numOfMatmuls * (std::max(M, MIN_MKN) / m) * (std::max(N, MIN_MKN) / n) / (l1Reuse * cubeNBuffer);
        double tasksRatioLess = (tasks < CUBE_CORES) ? (1 - tasks / CUBE_CORES) : 0;
        double tasksRatioMore = (tasks > 2 * CUBE_CORES) ? (tasks / (2 * CUBE_CORES) - 1) : 0;

        // Penalty for tasks < CUBE_CORES & tasks > 2 * CUBE_CORES
        score -= TASKS_WEIGHT * (tasksRatioLess + tasksRatioMore);

        // The more residualTasks the better
        int64_t residualTasks = (int64_t(std::ceil(tasks)) % int64_t(CUBE_CORES) == 0) ? int64_t(CUBE_CORES) : (int64_t(std::ceil(tasks)) % int64_t(CUBE_CORES));
        score += RESIDUAL_TASKS_WEIGHT * residualTasks;

        //The closer the ratio's is to 1, the better
        double ratioMK = (m > k) ? (m / k) : (k / m);
        double ratioKN = (k > n) ? (k / n) : (n / k);
        double ratioMN = (m > n) ? (m / n) : (n / m);
        vectorRatio = {ratioMK, ratioKN, ratioMN};
        double ratioMKN = CalculateGeometricMean(vectorRatio);

        // Penalty for bad balance
        score -= BALANCE_WEIGHT * ratioMKN;

        // Consider num of L1CopyIn cycles
        uint64_t numL1CopyInL1A = inputMKN / (mkn * l1Reuse * cubeNBuffer); // Num of L1CopyIn instructions for A
        uint64_t numL1CopyInL1B = inputMKN / mkn; // Num of L1CopyIn instructions for B

        uint64_t elePerRepeat = BYTES_PER_REPEAT / BytesOf(inputType);
        uint64_t parallelism = GetParallelism(inputType) == 0 ? 1 : GetParallelism(inputType);
        uint64_t cyclePerRepeat = elePerRepeat / parallelism;
        uint64_t latency = GetLatency(inputType);

        uint64_t repeatCountL1A = (tile[M_DIM] * tile[K_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;
        uint64_t repeatCountL1B = (tile[K_DIM] * tile[N_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;

        uint64_t cyclesL1A = numL1CopyInL1A * (latency + cyclePerRepeat * repeatCountL1A - 1);
        uint64_t cyclesL1B = numL1CopyInL1B * (latency + cyclePerRepeat * repeatCountL1B - 1);

        // Overall cycles of L1CopyIn for {m, k, n} tiles (the less the better)
        uint64_t cycles = cyclesL1A + cyclesL1B;
        double cyclesLog = std::log2(cycles);

        // Penalty for large num of cycles
        score -= CYCLES_WEIGHT * cyclesLog;
    }
}

void PrintTiles(std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, std::vector<int64_t>>& resultTilesAndInfo) {
    for (auto & [shapeAndTypeInfo, tiles] : resultTilesAndInfo) {
        std::cout << "shapeAndTypeInfo = [";
        for (size_t i = 0; i < shapeAndTypeInfo.first.size(); i++) {
            std::cout << shapeAndTypeInfo.first[i] << " ";
        }
        for (size_t i = 0; i < shapeAndTypeInfo.second.size(); i++) {
            std::cout << BytesOf(shapeAndTypeInfo.second[i]) << " ";
        }
        std::cout << "] -> Tiles = [";
        for (size_t i = 0; i < tiles.size(); i++) {
            std::cout << tiles[i] << " ";
        }
        std::cout << "]\n" << std::endl;
    }
}

std::vector<int64_t> FindAndSetCubeTileShapes(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, int64_t numOfMatmuls, int64_t l1Reuse, int64_t cubeNBuffer) {
    // Define set of possible tiles
    std::map<std::vector<int64_t>, double> setOfTiles;
    SetPossibleTiles(shapeAndTypeInfo, setOfTiles);

    // Find score for each set of tiles
    FindScoreForTiles(shapeAndTypeInfo, setOfTiles, l1Reuse, cubeNBuffer, numOfMatmuls);

    // Find set of tiles with max Score
    double maxScore = -std::numeric_limits<double>::max();
    int64_t mFinal = 0;
    int64_t kFinal = 0;
    int64_t nFinal = 0;
    for (auto & [tile, score] : setOfTiles) {
        if (maxScore < score) {
            mFinal = tile[M_DIM];
            kFinal = tile[K_DIM];
            nFinal = tile[N_DIM];
            maxScore = score;
        }
    }

    std::vector<int64_t> resultTiles;
    resultTiles.push_back(mFinal);
    resultTiles.push_back(kFinal);
    resultTiles.push_back(nFinal);
    return resultTiles;
}

void SetHeuristicTileShapes::SetHeuristicTileShapesFunc(Function &function) const {
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, int64_t> uniqueTiles;

    std::unordered_set<Operation *> cubeOperations;

    int64_t l1Reuse = (function.paramConfigs_.l1ReuseNum == 0) ? 1 : function.paramConfigs_.l1ReuseNum;
    int64_t cubeNBuffer = (function.paramConfigs_.cubeNBufferNum == 0) ? 1 : function.paramConfigs_.cubeNBufferNum;

    std::pair<std::vector<int64_t>, std::vector<DataType>> curShapeAndType = {{0, 0, 0}, {DataType::DT_FP16, DataType::DT_FP16}}; // shapeM, shapeK, shapeN, InputType, OutputType
    for (auto &op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC") {
            int64_t shapeM = op.GetIOperands()[0]->shape[0];
            int64_t shapeK = op.GetIOperands()[0]->shape[1];
            int64_t shapeN = (op.GetIOperands()[0]->shape[1] == op.GetIOperands()[1]->shape[0]) ? op.GetIOperands()[1]->shape[1] : op.GetIOperands()[1]->shape[0];

            DataType inputType = op.GetIOperands()[0]->tensor->GetDataType();
            DataType outputType = (IsFloat(op.GetOOperands()[0])) ? DataType::DT_FP32 : DataType::DT_INT32;

            curShapeAndType = {{shapeM, shapeK, shapeN}, {inputType, outputType}};

            // Find set of tiles by key
            auto it = uniqueTiles.find(curShapeAndType);
            if (it != uniqueTiles.end()) {
                uniqueTiles[curShapeAndType]++;
            } else {
                uniqueTiles[curShapeAndType] = 1;
            }
            cubeOperations.insert(&op);
        }
    }

    // Find and set heuristic cube tile shapes
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, std::vector<int64_t>> resultTilesAndInfo;
    for (auto & [shapeAndTypeInfo, numOfMatmuls] : uniqueTiles) {
        std::vector<int64_t> resultTiles = FindAndSetCubeTileShapes(shapeAndTypeInfo, numOfMatmuls, l1Reuse, cubeNBuffer);
        resultTilesAndInfo[shapeAndTypeInfo] = resultTiles;
    }

    std::array<int64_t, MAX_MDIM> m = {0,0};
    std::array<int64_t, MAX_KDIM> k = {0, 0, 0};
    std::array<int64_t, MAX_NDIM> n = {0, 0};

    PrintTiles(resultTilesAndInfo);

    for (auto &op : cubeOperations) {
        int64_t shapeM = op->GetIOperands()[0]->shape[0];
        int64_t shapeK = op->GetIOperands()[0]->shape[1];
        int64_t shapeN = (op->GetIOperands()[0]->shape[1] == op->GetIOperands()[1]->shape[0]) ? op->GetIOperands()[1]->shape[1] : op->GetIOperands()[1]->shape[0];

        DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
        DataType outputType = (IsFloat(op->GetOOperands()[0])) ? DataType::DT_FP32 : DataType::DT_INT32;

        curShapeAndType = {{shapeM, shapeK, shapeN}, {inputType, outputType}};

        m[0] = resultTilesAndInfo[curShapeAndType][M_DIM];
        k[0] = resultTilesAndInfo[curShapeAndType][K_DIM];
        n[0] = resultTilesAndInfo[curShapeAndType][N_DIM];

        // The algorithm calculates tiles for L0, let's assume that tiles for L1 are the same
        m[1] = m[0];
        k[1] = k[0];
        k[MAX_KDIM - 1] = k[0];
        n[1] = n[0];

        // Set new tiles for each operation (M = m, N = n, K = k, setL1Tile = true)
        op->GetTileShapeForSetting().SetCubeTile(m, k, n, true);
    }
}
} // namespace npu::tile_fwk