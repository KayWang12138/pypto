
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
#include <queue>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/derivation_tile_shape.h"
#include "passes/pass_log/pass_log.h"
#include "set_heuristic_tile_shapes.h"

#include <fstream>

#define MODULE_NAME "TileInference"
 
using namespace npu::tile_fwk;
 
namespace npu::tile_fwk {
Status SetHeuristicTileShapes::RunOnFunction(Function &function) {
    SetHeuristicTileShapesFunc(function);
    return SUCCESS;
}
 
const std::unordered_map<DataType, int64_t> Latency{
    {DataType::DT_FP16, 200},
    {DataType::DT_FP32, 200},
    {DataType::DT_INT32, 0},
    {DataType::DT_INT16, 0}
};
 
const std::unordered_map<DataType, int64_t> Parallelism{
    {DataType::DT_FP16, 64}, // 128B/cycle
    {DataType::DT_FP32, 32},
    {DataType::DT_INT32, 32},
    {DataType::DT_INT16, 64}
};

const std::set<Opcode> cubeMMOps = {
    Opcode::OP_A_MUL_B,
    Opcode::OP_A_MUL_BT,
    Opcode::OP_AT_MUL_B,
    Opcode::OP_AT_MUL_BT,
    Opcode::OP_A_MULACC_B,
    Opcode::OP_A_MULACC_BT
};

const std::set<Opcode> stopOps = {
    // Unary ops
    //reduce operations
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWSUM,
    Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWSUMLINE,
    Opcode::OP_ROWMAXLINE,
    Opcode::OP_ROWMINLINE,
    // topk operations
    Opcode::OP_TOPK,
    Opcode::OP_TILEDMRGSORT,
    Opcode::OP_BITSORT,
    Opcode::OP_MRGSORT,
    Opcode::OP_ARGSORT,
    Opcode::OP_TOPK_SORT,
    Opcode::OP_TOPK_MERGE,
    Opcode::OP_TOPK_EXTRACT,

    // Binary ops
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWSUM_SINGLE,

    Opcode::OP_A_MUL_B,
    Opcode::OP_A_MUL_BT,
    Opcode::OP_AT_MUL_B,
    Opcode::OP_AT_MUL_BT,
    Opcode::OP_A_MULACC_B,
    Opcode::OP_A_MULACC_BT,

    Opcode::OP_INDEX_OUTCAST,
    Opcode::OP_INDEX_PUT, //TODO: check on test if it should be here
    //Opcode::OP_INDEX_ADD, //TODO: check on test if it should be here
    Opcode::OP_SCATTER_ELEMENT,
    Opcode::OP_SCATTER,
    //Opcode::OP_SCATTER_UPDATE, Need to check
    //Opcode::OP_SCATTER_SCALAR, Need to check
};

const std::set<Opcode> wholeLastDimOps = { // Ops with Tile[lastDim] = Shape[lastDim]
    // Unary ops
    Opcode::OP_TRANSPOSE_MOVEIN,
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_VNCHWCONV,
};

const std::set<Opcode> reduceOps = {
    // Unary ops
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWSUM,

    Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWSUMLINE,
    Opcode::OP_ROWMAXLINE,
    Opcode::OP_ROWMINLINE,
    // Binary ops
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWSUM_SINGLE
};

const std::set<Opcode> transposeOps = {
    Opcode::OP_TRANSPOSE_MOVEIN,
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_VNCHWCONV
};

const std::set<Opcode> topkOps = {
    Opcode::OP_TOPK,
    Opcode::OP_TILEDMRGSORT,
    Opcode::OP_BITSORT,
    Opcode::OP_MRGSORT,
    Opcode::OP_ARGSORT,
    Opcode::OP_TOPK_SORT,
    Opcode::OP_TOPK_MERGE,
    Opcode::OP_TOPK_EXTRACT
};

const std::set<Opcode> scatterOps = {
    Opcode::OP_INDEX_OUTCAST,
    Opcode::OP_INDEX_PUT, //TODO: check on test if it should be here
    Opcode::OP_SCATTER_ELEMENT,
    Opcode::OP_SCATTER 
};

const std::set<Opcode> gatherVectorOps = {
    Opcode::OP_GATHER_ELEMENT,
    Opcode::OP_GATHER,
    Opcode::OP_GATHER_FROM_UB
};

const std::set<Opcode> gatherMoveOps = {
    Opcode::OP_GATHER_IN_L1,
    Opcode::OP_GATHER_IN_UB
};

uint64_t GetLatency(DataType dtype) {
    auto iterDtype = Latency.find(dtype);
    if (iterDtype == Latency.end()) {
        return DEFAULT_LATENCY;
    }
    return iterDtype->second;
}
 
uint64_t GetParallelism(DataType dtype) {
    auto iterDtype = Parallelism.find(dtype);
    if (iterDtype == Parallelism.end()) {
        return DEFAULT_MAX_PARALLELISM;
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
    if (vectorRatio.empty()) {
        return 0.0;
    }
    double product = 1.0;
    for (double num : vectorRatio) {
        product *= num;
    }
    return std::pow(product, 1.0 / vectorRatio.size());
}
 
std::map<int, int> FordBellman(const std::vector<std::pair<int, int>> &edges, Function &function) {
    std::map<int, int> subgrDepthMap;
    for (auto &op : function.Operations()) {
        subgrDepthMap[op.GetOpMagic()] = INT_MAX;
    }
    subgrDepthMap[-1] = 0;
    
    bool any = true;
    while (any == true) {
        any = false;
        for (auto elem: edges) {
            if (subgrDepthMap[elem.second] > subgrDepthMap[elem.first] - 1) {
                subgrDepthMap[elem.second] = subgrDepthMap[elem.first] - 1;
                any = true;
            }
        }
    }
    return subgrDepthMap;
}

void FindCubeTilesCombinations(std::map<std::vector<int64_t>, double>& setOfCubeTiles, int64_t m, int64_t k, int64_t n, int64_t inputTypeSize, int64_t outputTypeSize) {
    // Platform params
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    std::vector<int64_t> tmpTile = {0, 0, 0}; // m,k,n
    if (((m * k * inputTypeSize) <= (L0A_MAX_SIZE / DOUBLE_BUFFER)) && ((k * n * inputTypeSize) <= (L0B_MAX_SIZE / DOUBLE_BUFFER)) && ((m * n * outputTypeSize) <= (L0C_MAX_SIZE / DOUBLE_BUFFER))) {
        tmpTile[M_DIM] = m;
        tmpTile[K_DIM] = k;
        tmpTile[N_DIM] = n;
        setOfCubeTiles[tmpTile] = 0.f; // set initial score = 0
    }
}

void FindScoreForCubeTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                           int64_t cubeL1ReuseMode, int64_t cubeNBuffer, int64_t numOfMatmuls) {
    // Platform params
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    const int64_t CUBE_CORES = Platform::Instance().GetSoc().GetAICoreNum();

    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
 
    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
 
    uint64_t inputMKN = std::max(M, MIN_TILE) * std::max(K, MIN_TILE) * std::max(N, MIN_TILE);
    std::vector<double> vectorRatio = {0.f, 0.f, 0.f};
    for (auto & [tile, score] : setOfCubeTiles) {
        uint64_t mkn = tile[M_DIM] * tile[K_DIM] * tile[N_DIM];
 
        // If the tiling size = shape size -> the preferred option
        score = (tile[M_DIM] == std::max(M, MIN_TILE)) ? (score + WHOLE_M_SCORE) : score;
        score = (tile[K_DIM] == std::max(K, MIN_TILE)) ? (score + WHOLE_K_SCORE) : score;
        score = (tile[N_DIM] == std::max(N, MIN_TILE)) ? (score + WHOLE_N_SCORE) : score;
 
        // The more filled L0A, L0B, L0C is better
        double utilizationL0A = static_cast<double>((tile[M_DIM] * tile[K_DIM] * inputTypeSize)) / (L0A_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0B = static_cast<double>((tile[K_DIM] * tile[N_DIM] * inputTypeSize)) / (L0B_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0C = static_cast<double>((tile[M_DIM] * tile[N_DIM] * outputTypeSize)) / (L0C_MAX_SIZE / DOUBLE_BUFFER);
        vectorRatio = {utilizationL0A, utilizationL0B, utilizationL0C};
        double geomeanUtilizationL0 = CalculateGeometricMean(vectorRatio);
        score += WEIGHT_L0 * geomeanUtilizationL0;
 
        // The closer the tasksRatio is to 1, the better
        double tasks = numOfMatmuls * (std::max(M, MIN_TILE) / static_cast<double>(tile[M_DIM])) * (std::max(N, MIN_TILE) / static_cast<double>(tile[N_DIM])) / (cubeL1ReuseMode * cubeNBuffer);
        double tasksRatioLess = (tasks < CUBE_CORES) ? (CUBE_CORES / tasks  - 1) : 0;
        double tasksRatioMore = (tasks > 2 * CUBE_CORES) ? (tasks / (2 * CUBE_CORES) - 1) : 0;
 
        // Penalty for tasks < CUBE_CORES & tasks > 2 * CUBE_CORES
        score -= TASKS_CUBE_WEIGHT * (tasksRatioLess + tasksRatioMore);
 
        // The more residualTasks the better
        int64_t residualTasks = (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES) == 0) ? static_cast<int64_t>(CUBE_CORES) : (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES));
        score += RESIDUAL_CUBE_TASKS_WEIGHT * residualTasks;
 
        //The closer the ratio's is to 1, the better
        double ratioMK = (tile[M_DIM] > tile[K_DIM]) ? static_cast<double>((tile[M_DIM] / tile[K_DIM])) : static_cast<double>((tile[K_DIM] / tile[M_DIM]));
        double ratioKN = (tile[K_DIM] > tile[N_DIM]) ? static_cast<double>((tile[K_DIM] / tile[N_DIM])) : static_cast<double>((tile[N_DIM] / tile[K_DIM]));
        double ratioMN = (tile[M_DIM] > tile[N_DIM]) ? static_cast<double>((tile[M_DIM] / tile[N_DIM])) : static_cast<double>((tile[N_DIM] / tile[M_DIM]));
        vectorRatio = {ratioMK, ratioKN, ratioMN};
        double ratioMKN = CalculateGeometricMean(vectorRatio);
 
        // Penalty for bad balance
        score -= BALANCE_WEIGHT * ratioMKN;
 
        // Consider num of L1CopyIn cycles
        uint64_t numL1CopyInL1A = inputMKN / (mkn * cubeL1ReuseMode * cubeNBuffer); // Num of L1CopyIn instructions for A
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
        double cyclesLog = std::log2(cyclesL1A + cyclesL1B);
 
        // Penalty for large num of cycles
        score -= CYCLES_WEIGHT * cyclesLog;
    }
}
 
void SetPossibleCubeTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles) {
    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
 
    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
 
    int64_t newM = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(M, MIN_TILE))))));
    int64_t newK = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(K, MIN_TILE))))));
    int64_t newN = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(N, MIN_TILE))))));
 
    for (int64_t m = MIN_TILE; m <= newM; m *= FACTOR) {
        m = m > std::max(M, MIN_TILE) ? std::max(M, MIN_TILE) : m;
        for (int64_t k = MIN_TILE; k <= newK; k *= FACTOR) {
            k = k > std::max(K, MIN_TILE) ? std::max(K, MIN_TILE) : k;
            for (int64_t n = MIN_TILE; n <= newN; n *= FACTOR) {
                n = n > std::max(N, MIN_TILE) ? std::max(N, MIN_TILE) : n;
                FindCubeTilesCombinations(setOfCubeTiles, m, k, n, inputTypeSize, outputTypeSize);
            }
        }
    }
}

std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> FindL1Tiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::vector<int64_t> resultL0Tiles) {
    const int64_t L1_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1);

    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
    
    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    
    // Will consider kL1 4 times larger than kL0 if kL0 less than MIN_KL1_TILE
    int64_t kL1 = (resultL0Tiles[K_DIM] <= MIN_KL1_TILE) ? resultL0Tiles[K_DIM] * KL1_FACTOR : resultL0Tiles[K_DIM];
    kL1 = std::min (kL1, K);
    int64_t kLA1 = kL1;
    int64_t kLB1 = kL1;
    
    // Check if A or B can fit to L1 (At least 1 piece of the A matrix and 1 piece of the B matrix must be in the L1)
    if ((M <= resultL0Tiles[M_DIM]) && (K <= MAX_KL1)) {
        int64_t occupiedL1Memory = (resultL0Tiles[M_DIM] * K + kLB1 * resultL0Tiles[N_DIM]) * inputTypeSize;
        kLA1 = (occupiedL1Memory <= L1_MAX_SIZE) ? K : kL1;
    }
    if ((N <= resultL0Tiles[N_DIM]) && (K <= MAX_KL1)) {
        int64_t occupiedL1Memory = (resultL0Tiles[M_DIM] * kLA1 + K * resultL0Tiles[N_DIM]) * inputTypeSize;       
        kLB1 = (occupiedL1Memory <= L1_MAX_SIZE) ? K : kL1;
    }
    std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> resultCubeTiles = {{resultL0Tiles[M_DIM], resultL0Tiles[M_DIM]}, {resultL0Tiles[K_DIM], kLA1, kLB1}, {resultL0Tiles[N_DIM], resultL0Tiles[N_DIM]}};
    
    return resultCubeTiles;
}

std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> FindAndSetCubeTileShapes(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, int64_t numOfMatmuls, int64_t cubeL1ReuseMode, int64_t cubeNBuffer) {
    // Define set of possible cube tiles
    std::map<std::vector<int64_t>, double> setOfCubeTiles;
    SetPossibleCubeTiles(shapeAndTypeInfo, setOfCubeTiles);
 
    // Find score for each set of cube tiles
    FindScoreForCubeTiles(shapeAndTypeInfo, setOfCubeTiles, cubeL1ReuseMode, cubeNBuffer, numOfMatmuls);
    
    // Find set of tiles with max Score
    double maxScore = -std::numeric_limits<double>::max();
    int64_t mFinal = 0;
    int64_t kFinal = 0;
    int64_t nFinal = 0;
    for (auto & [tile, score] : setOfCubeTiles) {
        if (maxScore < score) {
            mFinal = tile[M_DIM];
            kFinal = tile[K_DIM];
            nFinal = tile[N_DIM];
            maxScore = score;
        }
    }
    
    std::vector<int64_t> resultL0Tiles;
    resultL0Tiles.push_back(mFinal);
    resultL0Tiles.push_back(kFinal);
    resultL0Tiles.push_back(nFinal);
    
    // Calc L1 tile shapes for the best found L0 tile shapes
    std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> resultCubeTiles;
    #ifdef L1_TILES_SETTING
    resultCubeTiles = FindL1Tiles(shapeAndTypeInfo, resultL0Tiles);
    #else
    resultCubeTiles = std::make_tuple(
        std::array<int64_t, MAX_MDIM>{mFinal, mFinal},
        std::array<int64_t, MAX_KDIM>{kFinal, kFinal},
        std::array<int64_t, MAX_NDIM>{nFinal, nFinal}
    );
    #endif
    return resultCubeTiles;
}

size_t DimsCalculation(Operation *op, size_t tensorsNum, bool isInput) {
    size_t tensorDims = 0;
    for (size_t tensor = 0; tensor < tensorsNum; tensor++) {
        if (isInput) {
            tensorDims = std::max(tensorDims, op->GetIOperands()[tensor]->shape.size());
        } else {
            tensorDims = std::max(tensorDims, op->GetOOperands()[tensor]->shape.size());
        }
    }
    return tensorDims;
}

std::vector<int64_t> MaxInputShapeCalculation(Operation *op, size_t inputsNum, size_t inputDims) {
    // Find the maximum values of the shape dimensions among the inputs, to find the maximum boundary of the tile values
    std::vector<int64_t> maxInputShape(inputDims, LLONG_MIN);
    for (size_t input = 0; input < inputsNum; input++) {
        for (size_t inputDim = 0; inputDim < op->GetIOperands()[input]->shape.size(); inputDim++) {
            maxInputShape[inputDim] = std::max(maxInputShape[inputDim], op->GetIOperands()[input]->shape[inputDim]);
        }
    }
    return maxInputShape;
}

void PrintOpInfo(const Operation *op) {
    std::cout << "!Op : " << op->GetOpcodeStr() << " Magic : " << op->GetOpMagic() << " Tiles : " << op->GetTileShape().ToString(TileType::VEC) << std::endl;
    for (size_t it = 0; it < op->GetIOperands().size(); it++) {
        std::cout << "Input " << it << " Magic = " << op->GetIOperands()[it]->magic << " shape = ["; 
        auto InputShape = op->GetIOperands()[it]->shape;
        for (size_t j = 0; j < InputShape.size(); j++) {
            std::cout << InputShape[j] << " ";
        }
        std::cout << "]  ";
    }
    for (size_t it = 0; it < op->GetOOperands().size(); it++) {
        std::cout << "Output " << it << " Magic = " << op->GetOOperands()[it]->magic << " shape = ["; 
        auto OutputShape = op->GetOOperands()[it]->shape;
        for (size_t j = 0; j < OutputShape.size(); j++) {
            std::cout << OutputShape[j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}

void AdjustTilesToReshape(Operation * op, Shape opBaseInputShape, Shape opBaseOutputShape, Shape& vectorTilesOld, Shape& outTileShape) {
    // Try to find dimensions that were not changed in reshape
    std::queue<size_t> unstableDims;
    size_t inProd = 1, outProd = 1;
    std::cout<<"Unstable :  ";
    for (size_t inPos=0, outPos=0; (inPos < opBaseInputShape.size()) && (outPos < opBaseOutputShape.size()); ) {
        if ((opBaseInputShape[inPos] == -1)  || (opBaseOutputShape[outPos] == -1)) {
            unstableDims.push(inPos);
            for(; (inPos < opBaseInputShape.size()); inPos++) {
                unstableDims.push(inPos);
                std::cout<<inPos<<"  ";
            }
            break;
        }
        if (inProd < outProd) {
            inProd *= opBaseInputShape[inPos];
            unstableDims.push(inPos);
            std::cout<<inPos<<"  ";
            inPos++;
        } else if (inProd > outProd) {
            outProd *= opBaseOutputShape[outPos];
            outPos++;
        } else {
            inProd = 1; outProd = 1;
            if (opBaseInputShape[inPos] == opBaseOutputShape[outPos]) {
                inPos++; outPos++;
                continue;
            }
            inProd *= opBaseInputShape[inPos];
            outProd *= opBaseOutputShape[outPos];
            unstableDims.push(inPos);
            std::cout<<inPos<<"  ";
            if (opBaseInputShape[inPos] < opBaseOutputShape[outPos]) {
                inPos++;
            } else {
                outPos++;
            }
        }
    }
    DerivationTileShape derivationTileShapePass;
    int32_t curDiTile; //vectorTilesOld.size() - 1;
    Shape tmpVectorTilesOld(vectorTilesOld);
    Status curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
    while (curStatus != SUCCESS) {
        std::cout<<"Derivation failed! \n";
        if(unstableDims.size() == 0) {
            //APASS_LOG_WARN_F(Elements::Operation, "DerivationReshapeTileShape failed. %s", GetFormatBacktrace(*opBase).c_str());
            //return;
            std::cout<<"Come out "<<curDiTile<<"  "<<vectorTilesOld.size()<<"  "<<opBaseInputShape.size()<<"\n";
            break;
        }
        std::cout<<"size "<<outTileShape.size()<<"\n";
        curDiTile = unstableDims.front();
        vectorTilesOld[curDiTile] = 1;
        unstableDims.pop();
        std::cout<<"vecOld "<<vectorTilesOld.size()<<"  "<<unstableDims.size()<<"\n";
        for(auto s: vectorTilesOld) {
            std::cout<<s<<" , ";
        }
        std::cout<<"\n";
        if(unstableDims.size() == 0) {
            vectorTilesOld[curDiTile] = std::min(opBaseInputShape[curDiTile], opBaseOutputShape[opBaseOutputShape.size()-(opBaseInputShape.size() - curDiTile)]);
        }
        curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
    }

    std::cout<<"size333 "<<outTileShape.size()<<"\n";
    for(auto s: outTileShape) {
        std::cout<<s<<" , ";
    }
    std::cout<<"\n";

    if(curStatus != SUCCESS) {
        for (size_t di=0; di< opBaseInputShape.size(); di++) {
            vectorTilesOld[di] = (opBaseInputShape[di] != -1) ? opBaseInputShape[di] : 1;
            std::cout<<" "<<vectorTilesOld[di]<<" , ";
        }
        std::cout<<"\n";
        curStatus = derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
        if(curStatus != SUCCESS) {
            std::cout<<"Not success again\n";
            outTileShape.resize(opBaseOutputShape.size());
            for (size_t di=0; di< opBaseOutputShape.size(); di++) {
                outTileShape[di] = (opBaseOutputShape[di] != -1) ? opBaseOutputShape[di] : 1;
            }
            APASS_LOG_WARN_F(Elements::Operation, "DerivationReshapeTileShape failed. %s", GetFormatBacktrace(*op).c_str());
        }
        return;
    } 
    auto outTileProd = std::accumulate(outTileShape.begin(), outTileShape.end(), 1, std::multiplies<int64_t>());
    auto inTileProd = std::accumulate(vectorTilesOld.begin(), vectorTilesOld.end(), 1, std::multiplies<int64_t>());
    if (inTileProd > outTileProd) {
        for (size_t di = 0; di < vectorTilesOld.size(); di++) {
            if (vectorTilesOld[di] > 1) {
                vectorTilesOld[di] = outTileProd / inTileProd * vectorTilesOld[di];
                std::cout<<"CORRECT!!!!!!!!!!!!!!!!!!!!!!\n";
            }
        }
    }
    if(derivationTileShapePass.DerivationReshapeTileShape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape) != SUCCESS) {
        std::cout<<"Something wrong happens!\n";
        return;
    }
}

void TileThroughReshape(Operation *op, std::vector<int64_t>& vectorTilesOld, bool isForward) {
    std::vector<int64_t> opBaseInputShape; 
    std::vector<int64_t> opBaseOutputShape;
    std::cout << "Forward " << isForward << "\n";

    if (isForward) {
        opBaseInputShape = op->GetIOperands()[0]->shape;
        opBaseOutputShape = op->GetOOperands()[0]->shape;
    } else {
        opBaseInputShape = op->GetOOperands()[0]->shape;
        opBaseOutputShape = op->GetIOperands()[0]->shape;
    }
    Shape outTileShape(opBaseOutputShape.size());
        
    AdjustTilesToReshape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);
    
    vectorTilesOld.resize(outTileShape.size());
    vectorTilesOld = outTileShape;

    std::cout << "From derivation ";
    for (auto s: outTileShape) {
        std::cout<<s<<"  ";
    }
    std::cout<<"\n";
}

void AdjustTileToUB(uint32_t argsCount, uint32_t maxTypeSize, uint32_t inputTypeSize, Shape& vectorTilesNew) {
    std::cout << "argsCount = " << argsCount << " sizes = " << maxTypeSize << " " << inputTypeSize <<std::endl;
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    int64_t maxTile = (UB_MAX_SIZE / maxTypeSize / argsCount);
    uint32_t inputDims = vectorTilesNew.size();
    std::cout<<"Cur maxTile 1 : "<< maxTile << " vectorTilesNew[inputDims - 1] : "<< vectorTilesNew[inputDims - 1] <<"\n";
    maxTile /= (((vectorTilesNew[inputDims - 1] + (BLOCK_SIZE / inputTypeSize) - 1) / (BLOCK_SIZE / inputTypeSize)) * (BLOCK_SIZE / inputTypeSize));
    std::cout<<"Cur maxTile 2 : "<<maxTile<<" vectorTilesNew.size() : "<<vectorTilesNew.size()<<"\n";
    if (inputDims == 1) {return; }
    for (int64_t di=inputDims - 2 ; di >= 0; di--) {
        std::cout<<" 1 "<<vectorTilesNew[di]<<'\n';
        vectorTilesNew[di] = std::min(vectorTilesNew[di], maxTile);
        std::cout<<" 2 "<<vectorTilesNew[di]<<'\n';
        vectorTilesNew[di] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesNew[di]))));
        std::cout<<" 3 "<<vectorTilesNew[di]<<'\n';
        maxTile = maxTile / vectorTilesNew[di];
        std::cout<<"Cur maxTile 3 : "<<maxTile<<" vectorTilesNew[di] : "<<vectorTilesNew[di]<<"\n";
   }
}

void TileThroughGather(Operation *producerOp, Operation *op, std::vector<int64_t>& vectorTilesOld) {
    std::cout << "!TileThroughGather BACKWARD!" << std::endl;
    std::cout << "Operation " << op->GetOpcodeStr() << std::endl;
    std::cout << "Producer " << producerOp->GetOpcodeStr() << std::endl;
    PrintOpInfo(op);
    DataType inputType = producerOp->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = producerOp->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    size_t inputsNum = op->GetIOperands().size(); 
    size_t outputsProducerNum = producerOp->GetOOperands().size(); 
    size_t outputsOpNum = op->GetOOperands().size(); 
    size_t inputDims = DimsCalculation(op, inputsNum, true); 
    size_t outputProducerDims = DimsCalculation(producerOp, outputsProducerNum, false);
    size_t outputOpDims = DimsCalculation(op, outputsOpNum, false); 
    uint32_t argsCount = inputsNum; // Output possibly can be placed in one of the arguments
    std::cout << "inputDims = " << inputDims << " inputsNum = " << inputsNum << std::endl;
    std::cout << "outputProducerDims = " << outputProducerDims << " outputOpDims = " << outputOpDims << std::endl;
    std::vector<int64_t> vectorTilesOldTmp = vectorTilesOld;

    if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
        int magicFirst = op->GetIOperands()[0]->magic;
        int magicSecond = op->GetIOperands()[1]->magic;
        int magicProducer = producerOp->GetOOperands()[0]->magic;
        
        std::cout << "magicFirst = " << magicFirst << std::endl;
        std::cout << "magicSecond = " << magicSecond << std::endl;
        std::cout << "magicProducer = " << magicProducer << std::endl;
        
        if (magicFirst == magicProducer) {
            std::cout << "magicFirst == magicProducer" << std::endl;
            if (outputProducerDims == 2) {
                vectorTilesOld.resize(outputProducerDims);
                vectorTilesOld[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(producerOp->GetOOperands()[0]->shape[0]))));
                vectorTilesOld[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[outputOpDims - 1]))));
            } else {
                ALOG_ERROR_F("Gather first input should be 2D, need to check this case");
            }
        }
        
        if (magicSecond == magicProducer) {
            std::cout << "magicSecond == magicProducer" << std::endl;
            if (outputProducerDims == 1) {
                vectorTilesOld.resize(outputProducerDims);
                vectorTilesOld[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[0]))));
    
            } else if (outputProducerDims == 2) {
                vectorTilesOld.resize(outputProducerDims);
                vectorTilesOld[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[0]))));
                vectorTilesOld[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[1]))));
            } else {
                ALOG_ERROR_F("Gather second input should be 1D or 2D, need to check this case");
            }
        }
    } else {
        int magicFirst = op->GetIOperands()[0]->magic;
        int magicSecond = op->GetIOperands()[1]->magic;
        int magicThird = op->GetIOperands()[2]->magic;
        int magicProducer = producerOp->GetOOperands()[0]->magic;
        
        std::cout << "magicFirst = " << magicFirst << std::endl;
        std::cout << "magicSecond = " << magicSecond << std::endl;
        std::cout << "magicThird = " << magicThird << std::endl;
        std::cout << "magicProducer = " << magicProducer << std::endl;
        
        if (magicFirst == magicProducer) {
            vectorTilesOld[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[0]))));
            vectorTilesOld[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(vectorTilesOldTmp[1]))));
        }
        
        if ((magicSecond == magicProducer) || (magicThird == magicProducer)) { // For 2nd and 3rd inputs it might be better to set a tile rather than based on the input shape, check the perf to find out
            vectorTilesOld[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(producerOp->GetOOperands()[0]->shape[0]))));
            vectorTilesOld[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(producerOp->GetOOperands()[0]->shape[1]))));
        }
    }
    
    AdjustTileToUB(argsCount, maxTypeSize, inputTypeSize, vectorTilesOld);
    
    for (size_t i = 0; i < vectorTilesOld.size(); i++) {
        std::cout << "!vectorTilesOld end[" << i << "] = " << vectorTilesOld[i] << std::endl;
    }
}

void TileThroughTranspose(Operation* op, std::vector<int64_t>& tile) {
    auto perm = op->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    std::swap(tile[perm[0]], tile[perm[1]]);
}

std::vector<uint32_t> FindChangedDims(Shape inShape, Shape outShape) {
    std::cout << "FindChangedDims" << std::endl;
    std::vector<uint32_t> changedDims;
    std::cout << "inShape: ";
    for (size_t p = 0; p < inShape.size(); p++) {
        std::cout << inShape[p] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "outShape: ";
    for (size_t p = 0; p < outShape.size(); p++) {
        std::cout << outShape[p] << " ";
    }
    std::cout << std::endl;
    
    for (size_t p = 0; p < inShape.size(); p++) {
        // assume that -1 is any big value
        if ((inShape[p] != outShape[p]) || ((inShape[p] == -1) || (outShape[p] == -1))) {
            changedDims.insert(changedDims.begin(), p);
            std::cout<<"split dim "<<p<<"\n";
        }
    }
    return changedDims;
}

void TileThroughViewAssemble(Operation* op, std::vector<int64_t>& tile) {             
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    ASSERT(inShape.size() == outShape.size()) <<"VIEW have different number of dimension in input and output "<<inShape.size()<<" vs "<<outShape.size()<<"\n";
    auto changedDims = FindChangedDims(inShape, outShape);
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    std::cout << "After FindChangedDims" << std::endl;
    for (auto di : changedDims) {
        std::cout << "di = " << di << std::endl;
        auto prevVal = tile[di];
        std::cout << "prevVal = " << prevVal << std::endl;
        tile[di] = op->GetOOperands()[0]->shape[di] != -1 ? op->GetOOperands()[0]->shape[di] : tile[di]; 
        auto wholeSize = std::accumulate(tile.begin(), tile.end(), 1, std::multiplies<int64_t>());
        if (wholeSize > ((int64_t)UB_MAX_SIZE / maxTypeSize)) {
            tile[di] = prevVal;
        }
    }
    std::cout << "TileThroughViewAssemble end" << std::endl;
}

void AdjustTileToUB(Operation* op, Shape& vectorTilesNew) {
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    Shape vectorTilesOut(vectorTilesNew);

    std::cout<<"sss ";
    for(auto v: vectorTilesOut){
        std::cout<<v<<" , ";
    }
    std::cout<<"\n";

    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        TileThroughReshape(op, vectorTilesOut, true);
    } else if (transposeOps.find(op->GetOpcode()) != transposeOps.end()) {
        TileThroughTranspose(op, vectorTilesOut);
    };

    std::cout<<"sss333333 ";
    for(auto v: vectorTilesOut){
        std::cout<<v<<" , ";
    }
    std::cout<<"\n";

    size_t memUsed = 0;
    for(auto inp: op->GetIOperands()) {
        Shape realTile(vectorTilesNew);
        size_t memUsedTmp = 1;
        for(size_t di=0; di < std::min(vectorTilesNew.size(), inp->GetShape().size()); di++) {
            realTile[di] = (inp->GetShape()[di] != -1) ? std::min(vectorTilesNew[di], inp->GetShape()[di]) : vectorTilesNew[di];
            if (di == vectorTilesNew.size() - 1) {
                realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType())) - 1) / (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType())) * (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType()));
            }
            memUsedTmp *= realTile[di];
        }
        memUsedTmp *= BytesOf(inp->tensor->GetDataType());
        memUsed += memUsedTmp;
        std::cout<<"Inp "<<memUsedTmp<<"  "<<memUsed<<"\n";
    }
    for(auto out: op->GetOOperands()) {
        Shape realTile(vectorTilesOut);
        size_t memUsedTmp = 1;
        for(size_t di=0; di < std::min(vectorTilesOut.size(), out->GetShape().size()); di++) {
            realTile[di] =  (out->GetShape()[di] != -1) ? std::min(vectorTilesOut[di], out->GetShape()[di]) : vectorTilesOut[di];
            if (di == vectorTilesOut.size() - 1) {
                realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) - 1) / (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) * (BLOCK_SIZE / BytesOf(out->tensor->GetDataType()));
            }
            memUsedTmp *= realTile[di];
            std::cout<<realTile[di]<<"  ";
        }
        std::cout<<"\n";
        memUsedTmp *= BytesOf(out->tensor->GetDataType());
        memUsed += memUsedTmp;
        std::cout<<"Out "<<memUsedTmp<<"  "<<memUsed<<"\n";
    }
    std::cout<<"Memused "<<memUsed<<"\n";

    while (memUsed > UB_MAX_SIZE) {
        size_t d = 0;
        while(vectorTilesNew[d] == 1) {
            d++;
        }
        vectorTilesNew[d] /= 2;

        std::cout<<"changed d "<<d<<"  "<<vectorTilesNew[d];

        vectorTilesOut = vectorTilesNew;

        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            TileThroughReshape(op, vectorTilesOut, true);
        } else if (transposeOps.find(op->GetOpcode()) != transposeOps.end()) {
            TileThroughTranspose(op, vectorTilesOut);
        }

        std::cout<<"out ";
        for(auto v : vectorTilesOut) {
            std::cout<<v<<" , ";
        }
        std::cout<<"\n";

        memUsed = 0;
        for(auto inp: op->GetIOperands()) {
            Shape realTile(vectorTilesNew);
            size_t memUsedTmp = 1;
            for(size_t di=0; di < std::min(vectorTilesNew.size(), inp->GetShape().size()); di++) {
                realTile[di] = (inp->GetShape()[di] != -1) ? std::min(vectorTilesNew[di], inp->GetShape()[di]) : vectorTilesNew[di];
                if (di == vectorTilesNew.size() - 1) {
                    realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType())) - 1) / (BLOCK_SIZE /BytesOf(inp->tensor->GetDataType())) * (BLOCK_SIZE / BytesOf(inp->tensor->GetDataType()));
                }
                memUsedTmp *= realTile[di];
            }
            memUsedTmp *= BytesOf(inp->tensor->GetDataType());
            memUsed += memUsedTmp;
        }
        for(auto out: op->GetOOperands()) {
            Shape realTile(vectorTilesOut);
            size_t memUsedTmp = 1;
            for(size_t di=0; di < std::min(vectorTilesOut.size(), out->GetShape().size()); di++) {
                realTile[di] = (out->GetShape()[di] != -1) ? std::min(vectorTilesOut[di], out->GetShape()[di]) : vectorTilesOut[di];
                if (di == vectorTilesOut.size() - 1) {
                    realTile[di] = (realTile[di] + (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) - 1) / (BLOCK_SIZE / BytesOf(out->tensor->GetDataType())) * (BLOCK_SIZE / BytesOf(out->tensor->GetDataType()));
                }
                memUsedTmp *= realTile[di];
            }
            memUsedTmp *=  BytesOf(out->tensor->GetDataType());
            memUsed += memUsedTmp;
        }
        std::cout<<"memUsed "<<memUsed<<"\n";
    }
}

void ReshapeTileSetting(Operation *op, std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    std::vector<int64_t> opBaseInputShape; 
    std::vector<int64_t> opBaseOutputShape;
    opBaseInputShape = op->GetIOperands()[0]->shape;
    opBaseOutputShape = op->GetOOperands()[0]->shape;

    Shape outTileShape;
    AdjustTilesToReshape(op, opBaseInputShape, opBaseOutputShape, vectorTilesOld, outTileShape);

    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;

    std::cout<<"From derivationSET ";
    for (auto s: vectorTilesNew) {
        std::cout<<s<<"  ";
    }
    std::cout<<"\n";

    AdjustTileToUB(op, vectorTilesNew);
}

void TransposeTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    ASSERT(op->GetIOperands().size() == 1); //transpose have 1 input
    ASSERT(op->GetOOperands().size() == 1); //transpose have 1 output
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    ASSERT(inputType == outputType);
    uint8_t typeSize = BytesOf(inputType);
    std::vector<int64_t> inputShape = op->GetIOperands()[0]->GetShape();
    size_t inputDims = inputShape.size();
    std::cout<<"transpose as consumer process";
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = (op->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV) ? 3 : 1; //at least input and output shuld be placed in UB

    int64_t maxTile = (UB_MAX_SIZE / typeSize / argsCount);
    ASSERT(inputShape[inputDims - 1] != -1);

    uint8_t blockSizeForType = BLOCK_SIZE / typeSize;
    auto lastDimAligned = ((inputShape[inputDims - 1] + blockSizeForType - 1) / blockSizeForType) * blockSizeForType;
    std::cout<<"cur tile "<<lastDimAligned<<"\n";
    if (lastDimAligned > maxTile) {
        ALOG_ERROR_F("Transpose have row that more than UB, doesn't support");
    }

    vectorTilesNew.resize(vectorTilesOld.size());
    std::cout<<"old size "<<vectorTilesNew.size()<<"  "<<inputDims<<"\n";
    vectorTilesNew = vectorTilesOld;
    std::cout<<"old size "<<vectorTilesOld.size()<<"  "<<inputDims<<"\n";
    vectorTilesNew[inputDims - 1] = inputShape[inputDims - 1];
    std::cout<<"vectile last "<<vectorTilesNew[inputDims - 1]<<"\n";
    maxTile /= lastDimAligned;
    
    std::cout<<"old one "<<vectorTilesNew[inputDims - 2]<<"  "<<(vectorTilesNew[inputDims - 2] + VNCHWCONV_POINTERS - 1)<<"  "<<(vectorTilesNew[inputDims - 2] + VNCHWCONV_POINTERS - 1) / VNCHWCONV_POINTERS * VNCHWCONV_POINTERS<<"\n";
    vectorTilesNew[inputDims - 2] = (vectorTilesNew[inputDims - 2] != 1) ? (vectorTilesNew[inputDims - 2] + VNCHWCONV_POINTERS - 1) / VNCHWCONV_POINTERS * VNCHWCONV_POINTERS : 1;
    std::cout<<" "<<vectorTilesNew[inputDims - 2]<<"\n";
    vectorTilesNew[inputDims - 2] = std::min(maxTile / VNCHWCONV_POINTERS * VNCHWCONV_POINTERS, vectorTilesNew[inputDims - 2]); // UB overflow condition
    maxTile /= vectorTilesNew[inputDims - 2];

    for (int64_t di = inputDims - 3; di >= 0; di--) {
        if (maxTile < vectorTilesNew[di]) {
            uint32_t coeff = (inputShape[di] + maxTile - 1) / maxTile;
            vectorTilesNew[di] = inputShape[di] / coeff;
        }
        maxTile /= vectorTilesNew[di];
    }
}

void ViewTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    std::cout<<"View process\n";
    auto allViews = op->GetIOperands()[0]->GetConsumers();
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    PrintOpInfo(op);
    ASSERT(inShape.size() == outShape.size()) <<"VIEW have different number of dimension in input and output "<<inShape.size()<<" vs "<<outShape.size()<<"\n";
    auto changedDims = FindChangedDims(inShape, outShape);

    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;

    for (auto sp : changedDims) {
        auto tile = inShape[sp];
        std::cout<<"Initial tile "<<tile<<"\n";
        for(auto v : allViews) {
            if(v->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            tile = (tile != -1) ? std::gcd(tile, v->GetOOperands()[0]->shape[sp]) : v->GetOOperands()[0]->shape[sp];
            std::cout<<"Updated tile "<<tile<<"\n";
        }
        ASSERT((size_t)sp < vectorTilesNew.size())<<"Split dim > size "<<sp<<" vs "<<vectorTilesNew.size()<<"\n";
        vectorTilesNew[sp] = (tile != -1) ? tile : vectorTilesNew[sp];
        std::cout<<"Tile "<<sp<<"  "<<vectorTilesNew[sp]<<"\n";
    }
}

void AssembleTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    auto allViews = op->GetOOperands()[0]->GetProducers();
    auto inShape = op->GetIOperands()[0]->shape;
    auto outShape = op->GetOOperands()[0]->shape;
    ASSERT(inShape.size() == outShape.size()) <<"VIEW have different number of dimension in input and output "<<inShape.size()<<" vs "<<outShape.size()<<"\n";
    auto changedDims = FindChangedDims(inShape, outShape);
    PrintOpInfo(op);
    for (size_t p=0; p<inShape.size(); p++) {
        if ((inShape[p] != outShape[p]) || ((inShape[p] == -1) || (outShape[p] == -1))) {
            changedDims.push_back(p);
            std::cout<<"split dim "<<p<<"\n";
        }
    }
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    
    for (auto sp : changedDims) {
        auto tile = outShape[sp];
        for(auto v : allViews) {
            if (v->GetOpcode() == Opcode::OP_ASSEMBLE) {
                tile = (tile != -1) ? std::gcd(tile, v->GetIOperands()[0]->shape[sp]) : v->GetIOperands()[0]->shape[sp];
            }
        }
        ASSERT((size_t)sp < vectorTilesNew.size())<<"Split dim > size "<<sp<<" vs "<<vectorTilesNew.size()<<"\n";
        vectorTilesNew[sp] = (tile != -1) ? std::gcd(tile, vectorTilesNew[sp]) : vectorTilesNew[sp];
    }
}

void OpWithSeveralInputsTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {  

    for (size_t i = 0; i < vectorTilesOld.size(); i++) {
        std::cout << "vectorTilesOld OpWithSeveralInputsTileSetting [" << i << "] = " << vectorTilesOld[i] << std::endl;
    }
    
    for (size_t i = 0; i < vectorTilesNew.size(); i++) {
        std::cout << "vectorTilesNew OpWithSeveralInputsTileSetting [" << i << "] = " << vectorTilesNew[i] << std::endl;
    }
    
    std::cout << "Operation OpWithSeveralInputsTileSetting: " << op->GetOpcodeStr() << std::endl;
    
    size_t inputsNum = op->GetIOperands().size(); 
    size_t inputDims = DimsCalculation(op, inputsNum, true); 
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;
    
    std::vector<Operation*> toUpdate;
    std::vector<size_t> inputInTile;

    for(size_t inp = 0; inp < op->GetIOperands().size(); inp++) {
        std::cout << "op->GetIOperands().size() = " << op->GetIOperands().size() << " inp = " << inp << std::endl;
        ASSERT(op->GetIOperands()[inp]->GetProducers().size() >= 1);
        auto prodOp = *(op->GetIOperands()[inp]->GetProducers().begin());
        auto consOp = *(op->GetOOperands()[0]->GetConsumers().begin());
        std::cout << "Producer: " << prodOp->GetOpcodeStr() << " magic : " << prodOp->GetOpMagic() << " Op: " << op->GetOpcodeStr() << " magic : " << op->GetOpMagic() << std::endl;
        std::vector<int64_t> inpTile(inputDims, 1);
        if (((prodOp->GetIOperands().size() == 0) || (prodOp->GetIOperands()[0]->GetProducers().size() == 0))) {
            //std::cout<<"No inputs from prod, ignore its tiling\n";
            if ((consOp->GetTileShape().GetVecTile().tile.size() != 1) || (consOp->GetTileShape().GetVecTile().tile[0] != -1)) {
                inpTile = consOp->GetTileShape().GetVecTile().tile;
            } else {
                inpTile = vectorTilesNew;
            }
            if (inpTile.back() == 1) {
                inpTile.back() = op->GetIOperands()[inp]->GetShape()[inputDims - 1];
            }
            toUpdate.push_back(prodOp);
            //continue;
        } else {
            inpTile = prodOp->GetTileShape().GetVecTile().tile;
        }

        std::cout << "PrintOpInfo Producer begin" << std::endl;
        PrintOpInfo(prodOp);
        std::cout << "PrintOpInfo Producer end" << std::endl;
        
        //tile not set yet
        if((inpTile.size() == 1) && (inpTile[0] == -1)) {
            vectorTilesNew.resize(1);
            vectorTilesNew[0] = -1;
            std::cout<<"tile not set yet for : \n";
            PrintOpInfo(prodOp);
            //continue;
            return;
        }

        //in all other cases recalculate input tile to output tile of operation
        if (prodOp->GetOpcode() == Opcode::OP_RESHAPE) {
            TileThroughReshape(prodOp, inpTile, true);
        } else if (transposeOps.find(prodOp->GetOpcode()) != transposeOps.end()) {
            TileThroughTranspose(prodOp, inpTile);
        }

        //evaluate new tile according to all existing tiles
        for (size_t di=0; di < inputDims; di++) {
            std::cout << "1 vectorTilesNew [" << di << "]= " << vectorTilesNew[di] << " inpTile [" << di << "]= " << inpTile[di] << std::endl;
            inpTile[di] = std::min(inpTile[di], op->GetIOperands()[inp]->GetShape()[di]);
            std::cout << "2 vectorTilesNew [" << di << "]= " << vectorTilesNew[di] << " inpTile [" << di << "]= " << inpTile[di] << std::endl;
            vectorTilesNew[di] = (inpTile[di] != -1) ? std::lcm(vectorTilesNew[di], inpTile[di]) : vectorTilesNew[di];
            std::cout << "3 vectorTilesNew [" << di << "]= " << vectorTilesNew[di] << " inpTile [" << di << "]= " << inpTile[di] << std::endl;
            //vectorTilesNew[di] = (inpTile[di] != -1) ? std::gcd(vectorTilesNew[di], inpTile[di]) : vectorTilesNew[di];
            std::cout<<"updated vec tile "<<vectorTilesNew[di]<<" for input "<<inp<<"\n";
        }
        inputInTile.push_back(inp);
    }
   
    //adjust final tiling to fit to UB

    for (size_t i = 0; i < vectorTilesNew.size(); i++) {
        std::cout << "vectorTilesNew AdjustTileToUB before [" << i << "] = " << vectorTilesNew[i] << std::endl;
    }
    
    AdjustTileToUB(op, vectorTilesNew);

    for (size_t i = 0; i < vectorTilesNew.size(); i++) {
        std::cout << "vectorTilesNew AdjustTileToUB after [" << i << "] = " << vectorTilesNew[i] << std::endl;
    }
    
    for(auto& opupd: toUpdate) {
        std::cout<<"Update Vector op\n\n";
        PrintOpInfo(opupd);
        opupd->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
        PrintOpInfo(opupd);
    }
}

void GatherTileSetting(Operation *op, std::vector<int64_t>& vectorTilesNew) { 
    std::cout << "!GatherTileSetting"  << std::endl;
    std::cout << "Operation " << op->GetOpcodeStr() << std::endl;
    PrintOpInfo(op);
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    size_t inputsNum = op->GetIOperands().size(); 
    size_t outputsNum = op->GetOOperands().size(); 
    size_t inputDims = DimsCalculation(op, inputsNum, true); 
    size_t outputDims = DimsCalculation(op, outputsNum, false); 
    uint32_t argsCount = inputsNum + outputsNum;
    argsCount = (op->GetOpcode() == Opcode::OP_GATHER_ELEMENT) ? argsCount + 1 : argsCount;
    std::cout << "inputDims = " << inputDims << " inputsNum = " << inputsNum << std::endl;
    std::cout << "outputDims = " << outputDims << " outputsNum = " << outputsNum << std::endl;
    vectorTilesNew.clear();
    vectorTilesNew.resize(outputDims, -1);
    
    std::vector<Operation*> toUpdate;
    std::vector<size_t> inputInTile;

    bool firstInputHasTile = false;
    bool secondInputHasTile = false;
    
    // Fill vectorTilesNew based on input tiles
    for (size_t inp = 0; inp < op->GetIOperands().size(); inp++) {
        auto prodOp = *(op->GetIOperands()[inp]->GetProducers().begin());
        std::cout << "Operation " << prodOp->GetOpcodeStr() << " " << prodOp->GetOpcodeStr() << std::endl;
        PrintOpInfo(prodOp);
    
        ASSERT(op->GetIOperands()[inp]->GetProducers().size() == 1);
        std::cout<<prodOp->GetOpMagic()<<"  "<<op->GetOpMagic()<<"\n";
        
        if ((prodOp->GetIOperands().size() == 0) || (prodOp->GetIOperands()[0]->GetProducers().size() == 0)) {
            std::cout<<"No inputs from prod, ignore its tiling\n";
            toUpdate.push_back(prodOp);
            continue;
        }

        std::vector<int64_t> inpTile(inputDims, 1);
        inpTile = prodOp->GetTileShape().GetVecTile().tile;
        PrintOpInfo(prodOp);

        // Tile not set yet
        if ((inpTile.size() == 1) && (inpTile[0] == -1)) {
            continue;
        }

        // In all other cases recalculate input tile to output tile of operation
        if (prodOp->GetOpcode() == Opcode::OP_RESHAPE) {
            TileThroughReshape(prodOp, inpTile, true);
        } else if (transposeOps.find(prodOp->GetOpcode()) != transposeOps.end()) {
            TileThroughTranspose(prodOp, inpTile);
        }

        // Evaluate new tile according to all existing tiles
        if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
            std::cout << "!gatherVectorOps branch" << std::endl;
            if (inp == 0) { // Broadcast tiles from first input
                if (outputDims == 2) {
                    vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[1]))));
                } else if (outputDims == 3) {
                    vectorTilesNew[2] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[1]))));
                } else {
                    ALOG_ERROR_F("outputDims != 2 or 3, need to check this case");
                }
                firstInputHasTile = true;
            } else if (inp == 1) { // Broadcast tiles from second input
                if (outputDims == 2) {
                    vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[0]))));
                } else if (outputDims == 3) {
                    vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[0]))));
                    vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[1]))));
                } else {
                    ALOG_ERROR_F("outputDims != 2 or 3, need to check this case");
                }
                secondInputHasTile = true;
            }
        } else {
            std::cout << "!gatherMoveOps branch" << std::endl;
            if (inp == 0) { // Broadcast tiles from 1st input if they are 
                vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[0]))));
                vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[1]))));
                firstInputHasTile = true;
            } else if (inp == 1){ // Otherwise broadcast tiles from 2nd or 3rd input if they are set
                vectorTilesNew[0] = (vectorTilesNew[0] == -1) ? static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[0])))) : vectorTilesNew[0];
                vectorTilesNew[1] = (vectorTilesNew[1] == -1) ? static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(inpTile[1])))) : vectorTilesNew[1];
                secondInputHasTile = true;
            }
            // inp == 2 -> blockTable: is not sliced in ExpandFunction, it is always processed as a whole
        }
        for (size_t i = 0; i < inpTile.size(); i++) {
            std::cout << "inpTile [" << i << "] = " << inpTile[i] << std::endl;
        }
        
        for (size_t i = 0; i < vectorTilesNew.size(); i++) {
            std::cout << "vectorTilesNew [" << i << "] = " << vectorTilesNew[i] << std::endl;
        }
        
        inputInTile.push_back(inp);
    }
    std::cout << "!firstInputHasTile = " << firstInputHasTile << " secondInputHasTile = " << secondInputHasTile << std::endl;
    
    for (size_t i = 0; i < op->GetIOperands()[0]->shape.size(); i++) {
        std::cout << "firstInputShape [" << i << "] = " << op->GetIOperands()[0]->shape[i] << std::endl;
    }
    for (size_t i = 0; i < op->GetIOperands()[1]->shape.size(); i++) {
        std::cout << "secondInputShape [" << i << "] = " << op->GetIOperands()[1]->shape[i] << std::endl;
    }
    
    if (gatherVectorOps.find(op->GetOpcode()) != gatherVectorOps.end()) {
        if (!firstInputHasTile) {
            std::cout << "firstInputHasTile condition" << std::endl;
            auto firstInputShape = op->GetIOperands()[0]->shape;
            if (outputDims == 2) {
                vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(firstInputShape[1]))));
            } else if (outputDims == 3) {
                vectorTilesNew[2] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(firstInputShape[1]))));
            } else {
                ALOG_ERROR_F("outputDims != 2 or 3, need to check this case");
            }
        }
        if (!secondInputHasTile) {
            std::cout << "SecondInputHasTile condition" << std::endl;
            auto secondInputShape = op->GetIOperands()[1]->shape;
            if (outputDims == 2) {
                vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(secondInputShape[0]))));
            } else if (outputDims == 3) {
                vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(secondInputShape[0]))));
                vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(secondInputShape[1]))));
            } else {
                ALOG_ERROR_F("outputDims != 2 or 3, need to check this case");
            }
        }
    } else {
        if (!firstInputHasTile && !secondInputHasTile) { // If the tiles were not broadcasted, set tiles based on the first input
            auto firstInputShape = op->GetIOperands()[0]->shape;
            vectorTilesNew[0] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(firstInputShape[0]))));
            vectorTilesNew[1] = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(firstInputShape[1]))));
        }
    }
    std::cout << "AdjustTileToUB" << std::endl;
    AdjustTileToUB(argsCount, maxTypeSize, inputTypeSize, vectorTilesNew);
    
    /*for(auto& opupd: toUpdate) {
        std::cout<<"Update Vector op\n\n";
        PrintOpInfo(opupd);
        opupd->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
        PrintOpInfo(opupd);
    } */ // May be no need
    
    std::cout << "!FINAL TILES!!!" << std::endl;
    for (size_t i = 0; i < vectorTilesNew.size(); i++) {
        std::cout << "vectorTilesNew final [" << i << "] = " << vectorTilesNew[i] << std::endl;
    }
    PrintOpInfo(op);
}

void DefaultTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = vectorTilesOld;

    std::cout<<"Vec default set1 : ";
    for(auto v: vectorTilesNew) {
        std::cout<<v<<"  ";
    }
    std::cout<<"\n";

    uint32_t typedBlock  = BLOCK_SIZE / inputTypeSize;
    if (vectorTilesNew[vectorTilesNew.size() - 1]  %  typedBlock != 0) {
        vectorTilesNew[vectorTilesNew.size() - 1] = (vectorTilesNew[vectorTilesNew.size() - 1] + typedBlock - 1) / typedBlock * typedBlock;
    }

    std::cout<<"Vec default set2 : ";
    for(auto v: vectorTilesNew) {
        std::cout<<v<<"  ";
    }
    std::cout<<"\n";

    AdjustTileToUB(op, vectorTilesNew);

    std::cout<<"Vec default set3 : ";
    for(auto v: vectorTilesNew) {
        std::cout<<v<<"  ";
    }
    std::cout<<"\n";
    std::cout<<"same tiles as old \n";
}

void ExpandTileSetting(Operation *op, const std::vector<int64_t>& vectorTilesOld, std::vector<int64_t>& vectorTilesNew) {
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    vectorTilesNew.resize(vectorTilesOld.size());
    vectorTilesNew = (vectorTilesOld[vectorTilesOld.size() - 1] == 1) ? op->GetOOperands()[0]->shape : vectorTilesOld;
    uint32_t typedBlock  = BLOCK_SIZE / inputTypeSize;
    if (vectorTilesNew[vectorTilesNew.size() - 1]  %  typedBlock != 0) {
        vectorTilesNew[vectorTilesNew.size() - 1] = (vectorTilesNew[vectorTilesNew.size() - 1] + typedBlock - 1) / typedBlock * typedBlock;
    }
    AdjustTileToUB(op, vectorTilesNew);
}

void ReduceTileSetting(Operation *op, std::vector<int64_t>& vectorTilesNew) {
    size_t inputsNum = op->GetIOperands().size();
    size_t outputsNum = op->GetOOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    size_t outputDims = DimsCalculation(op, outputsNum, false);                   
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);

    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = 4; //need to place output and temporal buffer up to input size for small shapes

    int64_t tileSize = UB_MAX_SIZE / maxTypeSize / argsCount;
    
    // Find the maximum values of the shape dimensions among the inputs, to find the maximum boundary of the tile values
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    ASSERT(outputsNum == 1) << "Reduce have several outputs\n";
    ASSERT(inputDims == outputDims) << "reduce have different shape size for input and output\n";
    int32_t reducedDim = -1;
    for(uint32_t i=0; i<inputDims; i++) {
        if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (op->GetOOperands()[0]->shape[i] == 1) && (reducedDim == -1)) {
            reducedDim = i;
        } else if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (op->GetOOperands()[0]->shape[i] == 1) && (reducedDim != -1)) {
            std::cout<<"Several reduced dimensions need check";
            return;
        }
    }
    ASSERT(reducedDim >= 0) << "not found reduced dims\n";
   
    // Last Dim processing
    int64_t curTile = (reducedDim == (int32_t)(inputDims -1)) ? std::max(maxInputShape[reducedDim], BLOCK_SIZE / inputTypeSize) : maxInputShape[reducedDim];
    curTile = (curTile >= (UINT8MAX * BLOCK_SIZE / inputTypeSize)) ? std::min(static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(UINT8MAX * BLOCK_SIZE / inputTypeSize)))), curTile) : curTile; // Consider additional restriction for REDUCE ops
    vectorTilesNew[reducedDim] = curTile;
    tileSize /= curTile;
    std::cout<<"reduced dim "<<reducedDim<<"  "<<curTile<<"\n";
    
    // Other Dims processing
    bool setBlock = false;
    for (int64_t dim = inputDims - 1; dim >= 0; dim--) {
        if (dim == reducedDim) {
            continue;
        }
        curTile = ((reducedDim == (int32_t)(inputDims - 1))) ? (((maxInputShape[dim] != 1) && (!setBlock) /*&& (vectorTilesReduce[reducedDim] == maxInputShape[reducedDim])*/) ? (BLOCK_SIZE / inputTypeSize) : 1) : std::min(static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(tileSize)))), static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(maxInputShape[dim])))));
        if (curTile == BLOCK_SIZE / inputTypeSize) {
            setBlock = true;
        }
        vectorTilesNew[dim] = curTile;
        tileSize /= curTile;
        std::cout<<"dim "<<dim<<"  "<<curTile<<"\n";
    }
}

void IndexInOutTileSetting(Operation *op, std::vector<int64_t>& vectorTilesNew) {
    size_t inputsNum = op->GetIOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);            
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    int64_t curTile = std::max(maxInputShape[inputDims - 1], BLOCK_SIZE / inputTypeSize);
    curTile = (maxInputShape[inputDims - 1] >= (UINT8MAX * BLOCK_SIZE / inputTypeSize)) ? std::min(static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(UINT8MAX * BLOCK_SIZE / inputTypeSize)))), curTile) : curTile; // Consider additional restriction for REDUCE ops
    //temporal solution, check if it s is good
    vectorTilesNew[inputDims - 1] = curTile;
    auto curProd = curTile;
    for (size_t di = 0; di < inputDims - 1; di++) {
        if (op->GetIOperands()[0]->shape[di] != 1) {
            if (curProd > MIN_TILE_SIZE) {
                vectorTilesNew[di] = 1;
            } else {
                vectorTilesNew[di] = std::max(((MIN_TILE_SIZE/curProd) / (BLOCK_SIZE / inputTypeSize)) * (BLOCK_SIZE / inputTypeSize), (long int)1);
                vectorTilesNew[di] = std::min(vectorTilesNew[di], op->GetIOperands()[0]->shape[di]);
                curProd *= vectorTilesNew[di];
            }
        }
    }
}

void TopkTileSetting(Operation *op, std::vector<int64_t>& vectorTilesNew) {
    size_t inputsNum = op->GetIOperands().size();
    size_t outputsNum = op->GetOOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    size_t outputDims = DimsCalculation(op, outputsNum, false);            
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = op->GetOOperands()[0]->tensor->GetDataType();
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint32_t argsCount = 3 + 4; //input + 2 outputs + temporal buffer = 4 * input
    int64_t tileSize = (UB_MAX_SIZE / maxTypeSize / argsCount);

    ASSERT(op->GetIOperands().size() == 1);
    ASSERT(op->GetOOperands().size() >= 1);
    ASSERT(inputDims == outputDims) << "reduce have different shape size for input and output\n";

    auto maxInputShape = op->GetIOperands()[0]->GetShape();
    int32_t reducedDim = -1;
    for (uint32_t i = 0; i < inputDims; i++) {
        if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (reducedDim == -1)) {
            reducedDim = i;
        } else if ((maxInputShape[i] != op->GetOOperands()[0]->shape[i]) && (reducedDim != -1)) {
            std::cout<<"Several reduced dimensions need check";
            return;
        }
    }
    if (reducedDim == -1) {
        reducedDim = inputDims - 1; //assume topk will only sort and will not miss anything
    }
    
    // Last Dim processing
    /*int64_t curTile = std::max(maxInputShape[reducedDim], BLOCK_SIZE / inputTypeSize) * 2;
    const uint64_t UB_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    uint64_t wholeInpSize = UB_MAX_SIZE / inputTypeSize;
    uint64_t wholeOutSize = 0;
    uint64_t tmpSize = 4 * wholeInpSize;
    
    
    while (wholeInpSize + wholeOutSize + tmpSize > (size_t)UB_MAX_SIZE / inputTypeSize) {
        wholeInpSize = 0;
        wholeOutSize = 0;
        tmpSize = 0;
        curTile = curTile / 2;

        for (auto intens : op->GetIOperands()) {
            for(size_t di=0; di < intens->GetShape().size(); di++) {
                if (di == (size_t)reducedDim) {
                    wholeInpSize += std::min(intens->GetShape()[di], curTile);
                } else {
                    wholeInpSize += std::min(intens->GetShape()[di], (int64_t)1);
                }
            }
        }
        
        for (auto intens : op->GetOOperands()) {
            for(size_t di=0; di < intens->GetShape().size(); di++) {
                if (di == (size_t)reducedDim) {
                    wholeOutSize += std::min(intens->GetShape()[di], curTile);
                } else {
                    wholeOutSize += std::min(intens->GetShape()[di], (int64_t)1);
                }
            }
        }
        tmpSize = 4 * wholeInpSize;
    } */        
    vectorTilesNew[reducedDim] = std::min(2 * op->GetOOperands()[0]->GetShape()[reducedDim], maxInputShape[reducedDim]);
    vectorTilesNew[reducedDim] = std::min(tileSize, vectorTilesNew[reducedDim]);
    //tileSize /= vectorTilesNew[reducedDim];
    
    // Other Dims processing
    //bool setBlock = false;
    for (int64_t dim = inputDims - 1; dim >= 0; dim--) {
        if (dim == reducedDim) {
            continue;
        }
        // curTile = ((reducedDim == (int32_t)(inputDims -1))) ? (((maxInputShape[dim] != 1) && (!setBlock) /*&& (vectorTilesReduce[reducedDim] == maxInputShape[reducedDim])*/) ? (BLOCK_SIZE / inputTypeSize) : 1) : std::min(static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(tileSize)))), static_cast<int64_t>(std::pow(2, static_cast<int64_t>(std::log2(maxInputShape[dim])))));
        //if (curTile == BLOCK_SIZE / inputTypeSize) {
        //    setBlock = true;
        //}
        vectorTilesNew[dim] = 1;//curTile;
        //tileSize /= curTile;
        //std::cout<<"dim "<<dim<<"  "<<curTile<<"\n";
    }
    /*if (inputDims != 1) {
        vectorTilesReduce[0] *= tileSize;
    }*/
    //std::reverse(vectorTilesReduce.begin(), vectorTilesReduce.end());
}

void AdaptTileToRealOutputShape(Operation* op, int opBaseMagic, std::vector<int64_t>& vectorTiles) {
    size_t outputsNum = op->GetOOperands().size();
    size_t outputDims = DimsCalculation(op, outputsNum, false);
    std::vector<int64_t> maxOutShape(outputDims, -1);
    for (auto outOp : op->GetOOperands()) {
        bool outToOpbase = false;
        for(auto c: outOp->GetConsumers()) {
            if (c->GetOpMagic() == opBaseMagic) {
                outToOpbase = true;
                break;
            }
        }
        if (!outToOpbase) {
            continue;
        }
        for (size_t di = 0; di < outOp->shape.size(); di++) {
            maxOutShape[di] = (outOp->shape[di] != -1) ? std::max(maxOutShape[di], outOp->shape[di]) : maxOutShape[di];
        }
    }
    std::cout << "Vectiles old before: ";
    for(auto vo: vectorTiles) {
        std::cout<<vo<<"  ,  ";
    }
    std::cout<<"\n";
    for(size_t s = vectorTiles.size(); s < maxOutShape.size(); s++) {
        vectorTiles.insert(vectorTiles.begin(), 1);
    }
        
    vectorTiles.resize(maxOutShape.size()); //resize is required for case if vectorTiles.size > shape.size

    for (size_t di=0; di<vectorTiles.size(); di++) {
        vectorTiles[di] = (maxOutShape[di] != -1) ? std::min(maxOutShape[di], vectorTiles[di]) : vectorTiles[di];
    }
    
    std::cout<<"Vectiles old after: ";
    for(auto vo: vectorTiles) {
        std::cout<<vo<<"  ,  ";
    }
    std::cout<<"\n";
}

void AdaptTileToRealInputShape(Operation* op, std::vector<int64_t>& vectorTiles) {
    if ((vectorTiles.size() == 1) && (vectorTiles[0] == -1)) {
        return;
    }
    size_t inputsNum = op->GetIOperands().size();
    size_t inputDims = DimsCalculation(op, inputsNum, true);
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
    std::cout<<"Max input shape "<<maxInputShape[0]<<"   "<<maxInputShape[1]<<"\n";
    
    for (size_t di = 0; di < vectorTiles.size(); di++) {
        vectorTiles[di] = (maxInputShape[di] != -1) ? std::min(vectorTiles[di], maxInputShape[di]) : vectorTiles[di];
    }
    std::cout<<"Vectiles old \n";
    for(auto vo: vectorTiles) {
        std::cout<<vo<<"  ,  ";
    }
    std::cout<<"\n";
}

bool IsBroadcastedShapes(Operation *op) {
    auto inputsNum = op->GetIOperands().size();
    auto outputsNum = op->GetOOperands().size();
    if ((inputsNum <= 1) || (outputsNum == 0)) {
        return false;
    }

    Shape outShape = op->GetOOperands()[0]->GetShape();
    for (auto out: op->GetOOperands()) {
        if (outShape.size() != out->GetShape().size()) {
            return false;
        }
        for (size_t di = 0; di < outShape.size(); di++) {
            if (outShape[di] != out->GetShape()[di]) {
                return false;
            }
        }
    }

    for (auto inp:  op->GetIOperands()) {
        auto inpShape = inp->GetShape();
        if (inpShape.size() != outShape.size()) {
            return false;
        }
        for (size_t di = 0; di < outShape.size(); di++) {
            if (outShape[di] % inpShape[di] != 0) {
                return false;
            }
        }
    }
    return true;
}

void BackPropagate(Operation *producerOp, Operation *op) {
    size_t inputsNum = producerOp->GetIOperands().size();
    size_t inputDims = DimsCalculation(producerOp, inputsNum, true);

    std::cout << "!Propagate producerOp type " << producerOp->GetCoreTypeStr() << "\n";
    std::cout << "!Propagate op " << op->GetOpcodeStr() << "\n";
    std::cout << "!Propagate producerOp " << producerOp->GetOpcodeStr() << "\n";

    std::vector<int64_t> vectorTilesNew(inputDims, -1);
    std::vector<int64_t> vectorTilesOld = op->GetTileShape().GetVecTile().tile;

    // Correct vectorTileOld to real shape of output
    if (gatherVectorOps.find(op->GetOpcode()) == gatherVectorOps.end() && gatherMoveOps.find(op->GetOpcode()) == gatherMoveOps.end()) { // No need to adapt tiles for Gather OPs 
        AdaptTileToRealOutputShape(producerOp, op->GetOpMagic(), vectorTilesOld);
    } else {
        TileThroughGather(producerOp, op, vectorTilesOld);
    }

    // Recalculate tile from tile of output to tile of input tensor
    if (transposeOps.find(producerOp->GetOpcode()) != transposeOps.end()) {
        TileThroughTranspose(producerOp, vectorTilesOld);
    } else if (producerOp->GetOpcode() == Opcode::OP_RESHAPE) {
        TileThroughReshape(producerOp, vectorTilesOld, false);
    }
    
    if (producerOp->GetOpcode() == Opcode::OP_VIEW) {
        std::cout << "ViewTileSetting";
        ViewTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    } else if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        AssembleTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    } else if (wholeLastDimOps.find(producerOp->GetOpcode()) != wholeLastDimOps.end()) {
        std::cout << "TransposeTileSetting";
        TransposeTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    } else if (producerOp->GetOpcode() == Opcode::OP_EXPAND) {
        std::cout << "ExpandTileSetting" << std::endl;
        ExpandTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    } else {
        DefaultTileSetting(producerOp, vectorTilesOld, vectorTilesNew);
    }

    //if (gatherVectorOps.find(producerOp->GetOpcode()) == gatherVectorOps.end() &&
        //producerOp->GetOpcode() != Opcode::OP_EXPAND) { // No need to adapt tiles for Gather OPs and Expand
        //AdaptTileToRealInputShape(producerOp, vectorTilesNew);
    //}
    
    producerOp->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    PrintOpInfo(producerOp);
}

void Propagate(Operation *consumerOp, Operation *op) {
    size_t inputsNum = consumerOp->GetIOperands().size();
    size_t inputDims = DimsCalculation(consumerOp, inputsNum, true);

    std::cout << "!Propagate consumerOp type " << consumerOp->GetCoreTypeStr() << "\n";
    std::cout << "!Propagate op " << op->GetOpcodeStr() << "\n";
    std::cout << "!Propagate consumerOp " << consumerOp->GetOpcodeStr() << "\n";
                        
    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(consumerOp, inputsNum, inputDims);
    std::vector<int64_t> vectorTilesNew(inputDims, -1);
    std::vector<int64_t> vectorTilesOld = op->GetTileShape().GetVecTile().tile;

    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        std::cout << "TileThroughReshape" << std::endl;
        TileThroughReshape(op, vectorTilesOld, true);
    } else if (transposeOps.find(op->GetOpcode()) != transposeOps.end()) {
        std::cout << "TileThroughTranspose" << std::endl;
        TileThroughTranspose(op, vectorTilesOld);
    } else if ((op->GetOpcode() == Opcode::OP_VIEW) || (op->GetOpcode() == Opcode::OP_ASSEMBLE)) {
        std::cout << "TileThroughViewAssemble" << std::endl;
        TileThroughViewAssemble(op, vectorTilesOld);
    }

    std::cout << " -----------------> BEFORE AdaptTileToRealOutputShape operation : " << consumerOp->GetOpcodeStr() << " Set old vector tile : ";
    for (size_t i = 0; i < vectorTilesOld.size(); i++) {
        std::cout << vectorTilesOld[i] << "  ";
    }
    std::cout<<"\n";
    
    AdaptTileToRealOutputShape(op, consumerOp->GetOpMagic(), vectorTilesOld);

    std::cout << " -----------------> AFTER AdaptTileToRealOutputShape operation : " << consumerOp->GetOpcodeStr() << " Set old vector tile : ";
    for (size_t i = 0; i < vectorTilesOld.size(); i++) {
        std::cout << vectorTilesOld[i] << "  ";
    }
    std::cout<<"\n";
    
    if (consumerOp->GetOpcode() == Opcode::OP_VIEW) {
        std::cout << "ViewTileSetting" << std::endl;
        ViewTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        std::cout << "AssembleTileSetting" << std::endl;
        AssembleTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (wholeLastDimOps.find(consumerOp->GetOpcode()) != wholeLastDimOps.end()) {
        std::cout << "TransposeTileSetting" << std::endl;
        TransposeTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (gatherVectorOps.find(consumerOp->GetOpcode()) != gatherVectorOps.end() || gatherMoveOps.find(consumerOp->GetOpcode()) != gatherMoveOps.end()) {
        std::cout << "GatherTileSetting" << std::endl;
        GatherTileSetting(consumerOp, vectorTilesNew);
    } else if (IsBroadcastedShapes(consumerOp)) {
        std::cout << "OpWithSeveralInputsTileSetting" << std::endl;
        OpWithSeveralInputsTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
        std::cout << "ReshapeTileSetting" << std::endl;
        ReshapeTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else if (consumerOp->GetOpcode() == Opcode::OP_EXPAND) {
        std::cout << "ExpandTileSetting" << std::endl;
        ExpandTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    } else {
        std::cout << "DefaultTileSetting" << std::endl;
        DefaultTileSetting(consumerOp, vectorTilesOld, vectorTilesNew);
    }

    std::cout << " -----------------> BEFORE ADAPT Operation : " << consumerOp->GetOpcodeStr() << " Set new vector tile : ";
    for (size_t i = 0; i < vectorTilesNew.size(); i++){
        std::cout << vectorTilesNew[i] << "  ";
    }
    std::cout<<"\n";
    
    if (gatherVectorOps.find(consumerOp->GetOpcode()) == gatherVectorOps.end() &&
        consumerOp->GetOpcode() != Opcode::OP_EXPAND) { // No need to adapt tiles for Gather OPs and Expand
        AdaptTileToRealInputShape(consumerOp, vectorTilesNew);
    }

    std::cout << " -----------------> AFTER ADAPT Operation : " << consumerOp->GetOpcodeStr() << " Set new vector tile : ";
    for (size_t i = 0; i < vectorTilesNew.size(); i++){
        std::cout << vectorTilesNew[i] << "  ";
    }
    std::cout<<"\n";
    
    consumerOp->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    PrintOpInfo(consumerOp);
}

std::pair<std::vector<int64_t>, std::vector<DataType>> ShapeAndTypeSetting(Operation *op, int64_t &shapeM, int64_t &shapeK, int64_t &shapeN) {
    bool isTransA = op->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A);
    bool isTransB = op->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B);
    
    if (!isTransA && !isTransB) { 
        // [m, k] * [k, n]
        shapeM = op->GetIOperands()[0]->shape[0];
        shapeK = op->GetIOperands()[0]->shape[1];
        shapeN = op->GetIOperands()[1]->shape[1];
    }
    if (!isTransA && isTransB) {
        // [m, k] * [n, k]
        shapeM = op->GetIOperands()[0]->shape[0];
        shapeK = op->GetIOperands()[0]->shape[1];
        shapeN = op->GetIOperands()[1]->shape[0];
    }
    if (isTransA && !isTransB) {
        // [k, m] * [k, n]
        shapeM = op->GetIOperands()[0]->shape[1];
        shapeK = op->GetIOperands()[0]->shape[0];
        shapeN = op->GetIOperands()[1]->shape[1];
    }
    if (isTransA && isTransB) {
        // [k, m] * [n, k]
        shapeM = op->GetIOperands()[0]->shape[1];
        shapeK = op->GetIOperands()[0]->shape[0];
        shapeN = op->GetIOperands()[1]->shape[0];
    }
    
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = (IsFloat(op->GetOOperands()[0])) ? DataType::DT_FP32 : DataType::DT_INT32;
    return {{shapeM, shapeK, shapeN}, {inputType, outputType}};
}

void UniqueTilesFilling(Function &function, std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, int64_t> &uniqueTiles,
                        std::pair<std::vector<int64_t>, std::vector<DataType>> &curShapeAndType, int64_t &shapeM, int64_t &shapeK, int64_t &shapeN) {
    for (auto &op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC") {
            // Calculate ShapeAndType for each operation
            curShapeAndType = ShapeAndTypeSetting(&op, shapeM, shapeN, shapeK);

            // Find set of tiles by key
            auto it = uniqueTiles.find(curShapeAndType);
            if (it != uniqueTiles.end()) {
                uniqueTiles[curShapeAndType]++;
            } else {
                uniqueTiles[curShapeAndType] = 1;
            }
        }
    }
}

void SetHeuristicCubeTiles(Function &function, std::unordered_set<Operation *> cubeOperations) {
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, int64_t> uniqueTiles;
    std::pair<std::vector<int64_t>, std::vector<DataType>> curShapeAndType = {{0, 0, 0}, {DataType::DT_FP16, DataType::DT_FP16}}; // shapeM, shapeK, shapeN, InputType, OutputType
    
    int64_t cubeL1ReuseMode = (function.paramConfigs_.cubeL1ReuseSetting.size() == 1 && function.paramConfigs_.cubeL1ReuseSetting.begin()->first == -1) ? function.paramConfigs_.cubeL1ReuseSetting.begin()->second : 1;
    int64_t cubeNBuffer = (function.paramConfigs_.cubeNBufferSetting.size() == 1 && function.paramConfigs_.cubeNBufferSetting.begin()->first == -1) ? function.paramConfigs_.cubeNBufferSetting.begin()->second : 1;
    int64_t shapeM = 0, shapeK = 0, shapeN = 0;
    UniqueTilesFilling(function, uniqueTiles, curShapeAndType, shapeM, shapeK, shapeN);
 
    // Find and set heuristic cube tile shapes
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>>> resultCubeTilesAndInfo;
    for (auto & [shapeAndTypeInfo, numOfMatmuls] : uniqueTiles) {
        std::tuple<std::array<int64_t, MAX_MDIM>, std::array<int64_t, MAX_KDIM>, std::array<int64_t, MAX_NDIM>> resultCubeTiles = FindAndSetCubeTileShapes(shapeAndTypeInfo, numOfMatmuls, cubeL1ReuseMode, cubeNBuffer);
        resultCubeTilesAndInfo[shapeAndTypeInfo] = resultCubeTiles;
    }
 
    std::array<int64_t, MAX_MDIM> m = {0, 0};
    std::array<int64_t, MAX_KDIM> k = {0, 0, 0};
    std::array<int64_t, MAX_NDIM> n = {0, 0};
 
    for (auto &op : cubeOperations) {
        // Calculate ShapeAndType for each operation
        curShapeAndType = ShapeAndTypeSetting(op, shapeM, shapeN, shapeK);
 
        m = std::get<0>(resultCubeTilesAndInfo[curShapeAndType]);
        k = std::get<1>(resultCubeTilesAndInfo[curShapeAndType]);
        n = std::get<2>(resultCubeTilesAndInfo[curShapeAndType]);
        
        // Set new tiles for each operation (M = m, N = n, K = k, enableSplitK = false)
        op->GetTileShapeForSetting().SetCubeTile(m, k, n, false);  
        std::cout << op->GetOpcodeStr() << "  " << op->GetOpMagic() << "  " << op->GetTileShape().ToString() << "\n";
    }
}

std::vector<Operation *> FordBellman(Function &function, std::unordered_set<Operation *> cubeOperations, std::map<int, int> &subgrDepthMap) {
    // Create edges
    std::vector<std::pair<int, int>> edges;
    for (auto &op : function.Operations()) {
        for (auto consumerOp : op.ConsumerOps()) {
            edges.push_back({consumerOp->GetOpMagic(), op.GetOpMagic()});
        }
    }
    
    // Create lastVertices
    std::vector<int> lastVertices;
    for (auto &op : function.Operations()) {
        if (op.ConsumerOps().size() == 0 && op.ProducerOps().size() != 0) {
            lastVertices.push_back(op.GetOpMagic());
        }
    }
    
    // Define depth for each node using FordBellman algorithm
    for (size_t idx = 0; idx < lastVertices.size(); idx++) {
        edges.push_back({-1, lastVertices[idx]});
    }
    auto d = FordBellman(edges, function);
    for (auto& [magic, depth] : d) {
        subgrDepthMap[magic] = std::max(subgrDepthMap[magic], -depth - 1);
    }
    subgrDepthMap.erase(-1);
    edges.clear();
    lastVertices.clear();
    
    // Sort cube operations by depth
    std::vector<std::pair<Operation *, int>> cubeTmpOperations;
    for (auto cubeOp : cubeOperations) {
        cubeTmpOperations.push_back(std::make_pair(cubeOp, subgrDepthMap[cubeOp->GetOpMagic()]));
    }
    std::sort(cubeTmpOperations.begin(), cubeTmpOperations.end(), [](const std::pair<Operation *, int> &x, const std::pair<Operation *, int> &y) {return x.second < y.second;});
    
    std::vector<Operation *> cubeOrderedOperations;
    for (auto op : cubeTmpOperations) {
        cubeOrderedOperations.push_back(op.first);
    }
    return cubeOrderedOperations;
}

bool DuplicateTileSetting(Operation *opInit, Operation *opNew, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    std::vector<int64_t> vectorTilesNew;
    if (opNew->GetIOperands().size() == 0) {
        vectorTilesNew = opInit->GetTileShape().GetVecTile().tile;
        opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
        if (!visitedBFS[opNew->GetOpMagic()]) {      
            queueBFS.push(opNew);
        }  
        visitedBFS[opNew->GetOpMagic()] = true;
        return true;
    }
    return false;
}

void CubeInDepsProcessing(Operation *cubeOp, Operation *opBase) {
    std::vector<int64_t> vectorTilesCube;
    auto &cubeTile = cubeOp->GetTileShape().GetCubeTile();
    int magicA = cubeOp->GetIOperands()[0]->magic;
    int magicB = cubeOp->GetIOperands()[1]->magic;
    
    std::vector<int64_t> vectorTilesA = {cubeTile.m[0], cubeTile.k[0]};
    std::vector<int64_t> vectorTilesB = {cubeTile.k[0], cubeTile.n[0]};

    bool isTransA = cubeOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A);
    bool isTransB = cubeOp->GetBoolAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B);

    ASSERT((opBase->GetOOperands()[0]->magic == magicA) || (opBase->GetOOperands()[0]->magic == magicB));
    if ((isTransA && !isTransB) || (isTransA && isTransB)) {
        std::reverse(vectorTilesA.begin(), vectorTilesA.end());
    }
    if ((!isTransA && isTransB) || (isTransA && isTransB)) {
        std::reverse(vectorTilesB.begin(), vectorTilesB.end());
    }

    vectorTilesCube = (opBase->GetOOperands()[0]->magic == magicA) ? vectorTilesA : vectorTilesB; 
    
    // Adjust final tiling to fit to UB - should be moved from here to real operations to adapt for it
    cubeOp->GetTileShapeForSetting().SetVecTile(vectorTilesCube); // Set vector tiles for Matmuls
    std::cout<<opBase->GetOOperands()[0]->magic<<"  vs  A "<<magicA<<"  B  "<<magicB<<" Set new vector tile for matmul: "<<vectorTilesCube[0]<<"  "<<vectorTilesCube[1]<<"\n";
}

void CubeOutDepsProcessing(Operation *cubeOp, Operation *opBase) {
    auto &cubeTile = cubeOp->GetTileShape().GetCubeTile();
    int magicC = cubeOp->GetOOperands()[0]->magic;   
    std::vector<int64_t> vectorTilesC = {cubeTile.m[0], cubeTile.n[0]};
    bool haveInp = false;
    for(auto inp: opBase->GetIOperands()) {
        if (inp->magic == magicC) {
            haveInp = true;
            break;
        }
    }
    ASSERT(haveInp == true);
    cubeOp->GetTileShapeForSetting().SetVecTile(vectorTilesC); 
}

void UpdateBFS(Operation *op, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    std::cout<<"check "<<op->GetOpMagic()<<"  "<<visitedBFS[op->GetOpMagic()]<<"\n";
    if (!visitedBFS[op->GetOpMagic()]) {      
        queueBFS.push(op);
        std::cout << "added " << op->GetOpMagic() << "\n";
    }  
    visitedBFS[op->GetOpMagic()] = true;
    std::cout<<"check fin "<<op->GetOpMagic()<<"  "<<visitedBFS[op->GetOpMagic()]<<"\n";
}

void BackwardPropagation(std::vector<Operation *> orderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto cubeOp : orderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            std::cout << "!Operation : " << op->GetOpcodeStr() << " Magic : " << op->GetOpMagic() << " Tiles : " << op->GetTileShape().ToString(TileType::CUBE) << " " << op->GetTileShape().ToString(TileType::VEC) << std::endl;
            for (auto producerOp : op->ProducerOps()) {
                if (stopOps.find(producerOp->GetOpcode()) != stopOps.end()) {
                    continue;
                }
                std::cout << "!1. Producer: " << producerOp->GetOpcodeStr() << " Magic : " << producerOp->GetOpMagic() << " Tiles : " << producerOp->GetTileShape().ToString(TileType::VEC) << std::endl;
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedBFS);
                if (isContinue) {
                    continue;
                }

                if (cubeMMOps.find(op->GetOpcode()) != cubeMMOps.end()) {
                    CubeInDepsProcessing(op, producerOp);
                }              
                
                BackPropagate(producerOp, op);

                std::cout << "\n----------------------------------------------------------------------------------\n" << std::endl;
                // Update queueBFS and visitedBFS
                UpdateBFS(producerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void ForwardPropagation(std::vector<Operation *> orderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto cubeOp : orderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            std::cout << "!Operation : " << op->GetOpcodeStr() << " Magic : " << op->GetOpMagic() << " Tiles : " << op->GetTileShape().ToString(TileType::CUBE) << " " << op->GetTileShape().ToString(TileType::VEC) << std::endl;
            for (auto consumerOp : op->ConsumerOps()) {
                if (stopOps.find(consumerOp->GetOpcode()) != stopOps.end()) {
                    continue;
                }

                if (cubeMMOps.find(op->GetOpcode()) != cubeMMOps.end()) {
                    CubeOutDepsProcessing(op, consumerOp);
                }
                std::cout << "!2. Consumer: " << consumerOp->GetOpcodeStr() << " Magic : " << consumerOp->GetOpMagic() << " Tiles : " << consumerOp->GetTileShape().ToString(TileType::VEC) << std::endl;   
                std::cout << "!2. op: " << op->GetOpcodeStr() << " Magic : " << op->GetOpMagic() << " Tiles : " << op->GetTileShape().ToString(TileType::VEC) << std::endl;                  
                auto prevTile = consumerOp->GetTileShape().GetVecTile();
                // Find the maximum values of the shape dimensions among the outputs, to find the maximum boundary of the tile values
                Propagate(consumerOp, op);

                std::cout << "\n----------------------------------------------------------------------------------"<<visitedBFS[consumerOp->GetOpMagic()]<<"\n" << std::endl;
                if ((consumerOp->GetTileShape().GetVecTile().size() == 1) && (consumerOp->GetTileShape().GetVecTile()[0] == -1)) {
                    continue;
                }

                if (prevTile.size() != consumerOp->GetTileShape().GetVecTile().size()) {
                    visitedBFS[consumerOp->GetOpMagic()] = false;
                    std::cout<<"different sizes "<<prevTile.size()<<"  "<<consumerOp->GetTileShape().GetVecTile().size()<<"\n";
                } 
                size_t di = 0;
                while ((di < prevTile.size()) && (prevTile[di] == consumerOp->GetTileShape().GetVecTile()[di])) {
                    di++;
                }
                if (di < prevTile.size()) {
                    visitedBFS[consumerOp->GetOpMagic()] = false;
                    std::cout<<"different values "<<prevTile.size()<<"  "<<di<<"  "<<prevTile[di]<<"  "<<consumerOp->GetTileShape().GetVecTile()[di]<<"\n";
                }

                // Update queueBFS and visitedBFS
                UpdateBFS(consumerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void SetReduceTiles(std::vector<Operation *> reduceOrderedOperations) {
    for (auto op : reduceOrderedOperations) {
        PrintOpInfo(op);
        size_t inputsNum = op->GetIOperands().size();
        size_t inputDims = DimsCalculation(op, inputsNum, true);

        std::vector<int64_t> vectorTilesReduce(inputDims, 1);
        
        // Find the maximum values of the shape dimensions among the inputs, to find the maximum boundary of the tile values
        std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims);
        std::cout<<"reduce op "<<op->GetOpcodeStr()<<"  "<<inputDims<<"\n";
        if (reduceOps.find(op->GetOpcode()) != reduceOps.end())  {
            ReduceTileSetting(op, vectorTilesReduce);
        } else if (scatterOps.find(op->GetOpcode()) != scatterOps.end()) {
            IndexInOutTileSetting(op, vectorTilesReduce);
        } else {
            TopkTileSetting(op, vectorTilesReduce);
        }
        op->GetTileShapeForSetting().SetVecTile(vectorTilesReduce);
        PrintOpInfo(op);
    }
}

std::vector<Operation *> FillNoConsumersOperations(Function &function) {
    std::vector<Operation *> noConsumersOperations;
    std::vector<int64_t> vectorTilesNew;
    for (auto &op : function.Operations()) {
        if (op.ConsumerOps().size() == 0) {
            vectorTilesNew.clear();
            size_t inputDims = op.GetIOperands()[0]->shape.size();
            int64_t inputTypeSize = BytesOf(op.GetIOperands()[0]->tensor->GetDataType());
            int64_t defaultTileSize = DEFAULT_TILE_SIZE;
            ASSERT(inputTypeSize != 0);
            if (op.GetTileShape().GetVecTile()[0] == -1) {
                int64_t curTile = std::min(defaultTileSize, static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(std::max(op.GetIOperands()[0]->shape[inputDims - 1], BLOCK_SIZE / inputTypeSize))))));
                if(curTile == 0) {
                    std::cout<<"some wrong \n";
                    curTile = 2048;
                }
                ASSERT(curTile != 0);
                vectorTilesNew.push_back(curTile);
                defaultTileSize /= curTile;
                for (size_t j = 1; j < inputDims; j++) {
                    curTile = std::min(defaultTileSize,  static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(op.GetIOperands()[0]->shape[inputDims - 1 - j])))));
                    if (curTile == 0) {
                        std::cout<<"some wrong \n";
                        curTile = 2048;
                    }
                    ASSERT(curTile != 0);
                    vectorTilesNew.push_back(curTile);
                    defaultTileSize /= curTile;
                }
                std::reverse(vectorTilesNew.begin(), vectorTilesNew.end());
                op.GetTileShapeForSetting().SetVecTile(vectorTilesNew);
            }
            noConsumersOperations.push_back(&op);
        }
    }
    return noConsumersOperations;
}

void BackwardNoConsumersPropagation(std::vector<Operation *> noConsumersOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto noConsumerOp : noConsumersOperations) {
        queueBFS.push(noConsumerOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                bool isVisitedNode = (producerOp->GetTileShape().GetVecTile()[0] != -1);
                if (isVisitedNode) {
                    // Update queueBFS and visitedBFS
                    UpdateBFS(producerOp, queueBFS, visitedBFS);
                    continue;
                }
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedBFS);
                if (isContinue) {
                    continue;
                }
                
                // Call propagation
                BackPropagate(producerOp, op);

                // Update queueBFS and visitedBFS
                UpdateBFS(producerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void SetHeuristicVectorTiles(Function &function, std::unordered_set<Operation *> cubeOperations) {
    // Define cube operations ordered by depth
    std::map<int, int> subgrDepthMap;
    std::map<uint8_t, std::vector<Operation *>> priorOps;
    //for(auto c: cubeOperations) {
    //    priorOps[1].push_back(c);//FordBellman(function, cubeOperations, subgrDepthMap);
    //}
    priorOps[1] = FordBellman(function, cubeOperations, subgrDepthMap);
       
    for (auto op : priorOps[1]) {
        std::cout << "Cube op = " << op->GetOpMagic() << " depth = " << subgrDepthMap[op->GetOpMagic()] << std::endl;
    }
    
    // Define auxiliary data structures
    std::queue<Operation *> queueBFS;
    std::map<int, bool> visitedBFS;

    // Sort reduce operations by depth
    std::vector<std::pair<Operation *, int>> reduceTmpOperations;
    for (auto &op : function.Operations()) {
        if ((reduceOps.find(op.GetOpcode()) != reduceOps.end()) ||
            (scatterOps.find(op.GetOpcode()) != scatterOps.end()) ||
            (topkOps.find(op.GetOpcode()) != topkOps.end())) {
            //priorOps[2].push_back(&op);
            reduceTmpOperations.push_back(std::make_pair(&op, subgrDepthMap[op.GetOpMagic()]));
        }
    }  
    std::sort(reduceTmpOperations.begin(), reduceTmpOperations.end(), [](const std::pair<Operation *, int> &x, const std::pair<Operation *, int> &y) {return x.second < y.second;});
    
    for (auto op: reduceTmpOperations) {
        priorOps[2].push_back(op.first);
    }
    reduceTmpOperations.clear();

    // Need to set initial vector tiles for ReduceOps
    SetReduceTiles(priorOps[2]);
    std::cout<<"reduce operations "<<priorOps[2].size()<<"\n";

    std::cout<<"Cube operations number: "<<priorOps[1].size()<<"\n";
    for(auto pOps: priorOps) {
        // Backward propagation
        auto start = std::chrono::high_resolution_clock::now();
        std::cout<<"Backward kkk\n";
        BackwardPropagation(pOps.second, queueBFS, visitedBFS);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ALOG_ERROR_F("              backward of pass is %ld us.", duration.count());

         // Forward propagation
        std::cout<<"Forward kkk\n";
        start = std::chrono::high_resolution_clock::now();
        ForwardPropagation(pOps.second, queueBFS, visitedBFS);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ALOG_ERROR_F("              forward of pass is %ld us.", duration.count());
    }

    std::vector<int64_t> vectorTilesNew;
    std::cout << "\n\n!3. BACKWARD PROPAGATION \n\n" << std::endl;
    // 3. Backward propagation Matmul (need to set tiles for rest -1 tiles), Start from nodes without consumers
    std::vector<Operation *> noConsumersOperations = FillNoConsumersOperations(function);

    BackwardNoConsumersPropagation(noConsumersOperations, queueBFS, visitedBFS);
}

void SetHeuristicTileShapes::SetHeuristicTileShapesFunc(Function &function) const {
    (void)function;

    // Find all cube operations from all operations
    std::unordered_set<Operation *> cubeOperations;
    for (auto &op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC") {
            cubeOperations.insert(&op);
            std::cout << "Cube op111 = " << op.GetOpMagic() << "  " << op.GetOpcodeStr() << std::endl;
        }
    }

    #ifdef CUBE_TILES 
    int opIdx = 0;
    std::ofstream mainFile("main_tiles.txt", std::ofstream::app);
    if (mainFile.is_open()) {
        mainFile << "\n\n---------------------------------------BEGIN---------------------------------------\n\n" << std::endl;
        for (auto &op : function.Operations()) {
            if (op.GetCoreTypeStr() == "AIC") {
                mainFile << "!Cube Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                mainFile << op.GetTileShape().ToString(TileType::CUBE) << std::endl;
            } else if (op.GetCoreTypeStr() == "AIV") {
                mainFile << "!Vector Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                mainFile << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            } else {
                mainFile << "!Other Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                mainFile << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            }
            for (size_t it = 0; it < op.GetIOperands().size(); it++) {
                mainFile << "Input " << it << " Magic = " << op.GetIOperands()[it]->magic << " DataType = " << BytesOf(op.GetIOperands()[it]->tensor->GetDataType()) << " Shape = ["; 
                auto InputShape = op.GetIOperands()[it]->shape;
                for (size_t j = 0; j < InputShape.size(); j++) {
                    mainFile << InputShape[j] << " ";
                }
                mainFile<< "]  ";
            }
            for (size_t it = 0; it < op.GetOOperands().size(); it++) {
                mainFile << "Output " << it << " Magic = " << op.GetOOperands()[it]->magic << " DataType = " << BytesOf(op.GetOOperands()[it]->tensor->GetDataType()) << " Shape = ["; 
                auto OutputShape = op.GetOOperands()[it]->shape;
                for (size_t j = 0; j < OutputShape.size(); j++) {
                    mainFile << OutputShape[j] << " ";
                }
                mainFile << "]" << std::endl;
            } 
            mainFile << "\n----------------------------------------------------------------------------------" << std::endl;
            opIdx++;
        }
        mainFile << "\n\n---------------------------------------END---------------------------------------\n\n" << std::endl;
        mainFile.close();
    }
    SetHeuristicCubeTiles(function, cubeOperations);
    #endif

    #ifdef VECTOR_TILES
    // Define -1 tile shapes for non-cubes operations
    // Also check if there are any operations where vector tiles are required
    std::vector<int64_t> defTile = {-1};
    for (auto &op : function.Operations()) {
        if(op.GetTileShape().GetVecTile().size() != 0) {
            std::cout<<"OPeration22 "<<op.GetOpcodeStr()<<" "<<op.GetOpMagic()<<"  "<<op.GetTileShape().ToString()<<"\n";
        }
        op.GetTileShapeForSetting().SetVecTile(defTile);
    }

    // Set heuristic vector tiles
    SetHeuristicVectorTiles(function, cubeOperations);
  
    // Check that all tiles was defined by algorithm
    for (auto &op : function.Operations()) {
        ASSERT(op.GetTileShape().GetVecTile()[0] != -1) << "Not all tiles was set";
    }
    
    opIdx = 0;
    std::ofstream customFile("custom_tiles.txt", std::ofstream::app);
    if (customFile.is_open()) {
        customFile << "\n\n---------------------------------------BEGIN---------------------------------------\n\n"  << std::endl;
        for (auto &op : function.Operations()) {
            if (op.GetCoreTypeStr() == "AIC") {
                customFile << "!Cube Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                customFile << op.GetTileShape().ToString(TileType::CUBE) << std::endl;
            } else if (op.GetCoreTypeStr() == "AIV") {
                customFile << "!Vector Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                customFile << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            } else {
                customFile << "!Other Operation " << opIdx << " : " << op.GetOpcodeStr() << " magic : " << op.GetOpMagic() << std::endl;
                customFile << op.GetTileShape().ToString(TileType::VEC) << std::endl;
            }
            for (size_t it = 0; it < op.GetIOperands().size(); it++) {
                customFile << "Input " << it << " Magic = " << op.GetIOperands()[it]->magic << " DataType = " << BytesOf(op.GetIOperands()[it]->tensor->GetDataType()) << " Shape = ["; 
                auto InputShape = op.GetIOperands()[it]->shape;
                for (size_t j = 0; j < InputShape.size(); j++) {
                    customFile << InputShape[j] << " ";
                }
                customFile << "]  ";
            }
            for (size_t it = 0; it < op.GetOOperands().size(); it++) { 
                customFile << "Output " << it << " Magic = " << op.GetOOperands()[it]->magic << " DataType = " << BytesOf(op.GetOOperands()[it]->tensor->GetDataType()) << " Shape = ["; 
                auto OutputShape = op.GetOOperands()[it]->shape;
                for (size_t j = 0; j < OutputShape.size(); j++) {
                    customFile << OutputShape[j] << " ";
                }
                customFile << "]" << std::endl;
            } 
            customFile << "\n----------------------------------------------------------------------------------" << std::endl;
            opIdx++;
        }
        customFile << "\n\n---------------------------------------END---------------------------------------\n\n" << std::endl;
        customFile.close();
    }
    #endif
}
} // namespace npu::tile_fwk