/**

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file page_attention.cpp
\brief
*/

#include "interface/function/function.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

const int NUM1024 = 1024;

namespace npu::tile_fwk {
    

struct KvSlcTileShapeConfig {
    std::array<int, TILE_VEC_DIMS> v0TileShape;
}; 


void KvSlcCompute(Tensor &topK_indcies, Tensor &topK_tensor_shape, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, int front, int near, int topk, int l_prime,
                  int n2, Tensor &blockTable, int blockSize, Tensor &k_slcOut, Tensor &v_slcOut, Tensor &kvSlcActSeqs, KvSlcTileShapeConfig &tileConfig, bool debug = false) {
    auto v0Tile = tileConfig.v0TileShape;
    SymbolicScalar b = topK_indcies->shape[0];
    SymbolicScalar s = topK_indcies->shape[1];
    SymbolicScalar kv_lora_rank = kvNopeCache.GetShape(1) / n2;
    SymbolicScalar rope_dim = kRopeCache.GetShape(1) / n2;
    LOOP("LOOP_L0_batchIdx", FunctionType::DYNAMIC_LOOP, batchIdx, LoopRange(0, b, 1), {}, true) {
        SymbolicScalar curActSeq = GetTensorData(kvActSeqs, {batchIdx});
        LOOP("LOOP_L1_slcIdx", FunctionType::DYNAMIC_LOOP, slcIdx, LoopRange(0, s, 1)) {
            LOOP("LOOP_L2_kvSlcIdx", FunctionType::DYNAMIC_LOOP, nkvIdx, LoopRange(0, n2, 1)) {
                TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                SymbolicScalar s_slc = GetTensorData(topK_tensor_shape, {batchIdx, slcIdx});
                SymbolicScalar positions = 0;
                SymbolicScalar prime_value = l_prime;
                SymbolicScalar slcSeqLen = 0;
                for (int topKIdx = 0; topKIdx < topk; topKIdx++) {
                    // 获取到topk的position
                    // 头部的front个
                    if (topKIdx < front) {
                        positions = topKIdx * l_prime;
                    } else if (topKIdx > (topk - near - front)) {
                    // 尾部的near个
                        positions = (s_slc - near + (topKIdx - (topk - front - near)) - 1) * l_prime;
                    } else {
                        // 中间的topk-front-near个
                        SymbolicScalar topk_index;
                        if (debug) {
                            TileShape::Current().SetVecTile(1, 1, NUM16);
                            topk_index = GetTensorData(topK_indcies, {batchIdx, slcIdx, topKIdx - front});
                        } else {
                            topk_index = GetTensorData(topK_indcies, {batchIdx, slcIdx, topKIdx - front});
                        }
                        positions = topk_index * prime_value;
                    }
                    slcSeqLen = slcSeqLen + prime_value;
                    SymbolicScalar blockIdxInBatch;
                    SymbolicScalar tail;
                    if (blockSize != 0) {
                        blockIdxInBatch = positions / blockSize;
                        tail = positions % blockSize;
                    }
                    SymbolicScalar slcBlockIdx = GetTensorData(blockTable, {batchIdx, blockIdxInBatch});
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    auto kv_slcBlock = View(kvNopeCache, {l_prime, kv_lora_rank}, {slcBlockIdx * blockSize + tail, nkvIdx * kv_lora_rank});
                    auto kRope_slcBlock = View(kRopeCache, {l_prime, rope_dim}, {slcBlockIdx * blockSize + tail, nkvIdx * rope_dim});
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    auto kv_slcBlock_fp32 = Cast(kv_slcBlock, DataType::DT_FP32);
                    auto kRope_slcBlock_fp32 = Cast(kRope_slcBlock, DataType::DT_FP32);
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    auto kv_slcBlock_tiled = MulS(kv_slcBlock_fp32, Element(kv_slcBlock_fp32->Datatype(), float(1)));
                    auto kRope_slcBlock_tiled = MulS(kRope_slcBlock_fp32, Element(kRope_slcBlock_fp32->Datatype(), float(1)));
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    auto kv_slcBlock_fp16 = Cast(kv_slcBlock_tiled, k_slcOut->Datatype());
                    auto kRope_slcBlock_fp16 = Cast(kRope_slcBlock_tiled, v_slcOut->Datatype());
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    SymbolicScalar output_axis1_value =
                    batchIdx * s * n2 * topk * l_prime + slcIdx * n2 * topk * l_prime + nkvIdx * topk * l_prime + topKIdx * l_prime;
                    Assemble(kv_slcBlock_fp16, {output_axis1_value, 0}, k_slcOut);
                    Assemble(kRope_slcBlock_fp16, {output_axis1_value, kv_lora_rank}, k_slcOut);
                    Assemble(kv_slcBlock_fp16, {output_axis1_value, 0}, v_slcOut);
                }
                SetTensorData(slcSeqLen, {batchIdx, slcIdx}, kvSlcActSeqs);
            }
        }
    }
}

}

int main() { // b_n_s_s2_h_q_lora_rank
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    KvSlcTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.v0TileShape = {nTile, 32};

    std::vector<int> input_param = {1, 1, 1, 64, 64, 1, 2, 4, 16, 32};
    int b = input_param[0];
    int s = input_param[1];
    int n2 = input_param[2];
    int kv_lora_rank = input_param[3];
    int rope_dim = input_param[4];
    int front = input_param[5];
    int near = input_param[6];
    int topK = input_param[7];
    int l_prime = input_param[8];
    int blockSize = input_param[9];

    int blockNum = 0;
    std::vector<int> seq(NUM1024, 1);
    for (auto s_item : seq) {
        blockNum += CeilDiv(s_item, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor topk_tensor(DT_INT32, {b, s, topK - front - near}, "topk_tensor");
    Tensor topk_tensor_shape(DT_INT32, {b, s}, "topk_tensor_shape");
    Tensor kvNopeCache(DT_FP16, {int(blockNum * blockSize), n2 * kv_lora_rank}, "kNopeCache");
    Tensor kRopeCache(DT_FP16, {int(blockNum * blockSize), n2 * rope_dim}, "vNopeCache");
    Tensor kvActSeqs(DT_INT32, {b}, "kvActSeqs");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor k_slcOut(DT_FP16, {b * s * n2 * topK * l_prime, rope_dim + kv_lora_rank}, "k_slcOut");
    Tensor v_slcOut(DT_FP16, {b * s * n2 * topK * l_prime, kv_lora_rank}, "v_slcOut");
    Tensor kvSlcActSeqs(DT_INT32, {b, s}, "kvSlcActSeqs");
    FUNCTION("main_slc",
        {topk_tensor, topk_tensor_shape, kvNopeCache, kRopeCache, kvActSeqs, blockTable}, {k_slcOut, v_slcOut, kvSlcActSeqs}) {
        KvSlcCompute(topk_tensor, topk_tensor_shape, kvNopeCache, kRopeCache, kvActSeqs, front, near, topK, l_prime, n2, blockTable, blockSize, k_slcOut, v_slcOut, kvSlcActSeqs, tileConfig);
    }
}

