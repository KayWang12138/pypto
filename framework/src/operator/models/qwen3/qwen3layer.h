#pragma once
#ifndef QWEN3_ATTENTION
#define QWEN3_ATTENTION

#include "tilefwk/tensor.h"
#include "interface/inner/tilefwk.h"
// #include "operator/models/qwen3/qwen3_def.h"

namespace npu::tile_fwk {

struct Qwen3AttentionDims {
    int b;
    int n;    // Query Heads
    int n_kv; // KV Heads
    int s;
    int d;
    int singleM;
    int singleN;
};

struct Qwen3MlpDims {
    int b;
    int s;
    int h;
    int inter;
};

struct Qwen3AttenTileShapeConfig {
    std::array<int, TILE_VEC_FOUR_DIMS> vecTileShape; // tileshape for transpose/softmax/context
    float scale;                                      // custom scale, <=0 means auto 1/sqrt(d)
};

struct Qwen3LayerDims {
    Qwen3AttentionDims attn;
    Qwen3MlpDims mlp;
};

// Qwen3PagedAttentionProlog:
//   - 输入 hiddenStates(2D) + QKV权重/偏置 + q/k norm权重 + (cos,sin) + cache_position
//   - 计算 q/k/v projection、q/k rmsnorm、q/k rope
//   - 将 roped K 与 V 写入 paged KV cache（2D: [blockNum*blockSize, nKv*d]）
//   - 输出 query(2D: [B*S*nQ, d]) 供 Qwen3PagedAttention 使用
//   - 注意：key/value cache 的 out 参数为 **in-place 输出**（out 与输入 cache 共享内存，只更新指定 row）
template <typename T>
void Qwen3PagedAttentionPrologCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight,
    Tensor &kBias, Tensor &vWeight, Tensor &vBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &cos, Tensor &sin,
    Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, int nQ, int nKv, int blockSize, Tensor &queryOut,
    Tensor &keyCacheOut, Tensor &valueCacheOut);

template <typename T>
void Qwen3PagedAttentionProlog(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &cos, Tensor &sin,
    Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, int nQ, int nKv, int blockSize, Tensor &queryOut,
    Tensor &keyCacheOut, Tensor &valueCacheOut);

// simple attention
template <typename T>
void Qwen3AttentionCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &attentionOut, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &dims);

template <typename T>
void Qwen3Attention(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &attentionOut, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &dims);

// decode阶段 支持GQA
template <typename T>
void Qwen3PagedAttentionCompute(Tensor &query, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, int nQ, int nKv, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false, int windowSize = 0);

template <typename T>
void Qwen3PagedAttention(Tensor &query, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable, Tensor &actSeqs,
    int nQ, int nKv, int blockSize, float softmaxScale, Tensor &attentionOut, PaTileShapeConfig &tileConfig,
    int maxUnrollTimes = 1, bool isNzFormat = false, int windowSize = 0);

template <typename T>
void Qwen3MLPCompute(Tensor &hiddenStates, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &mlpOut,
    const Qwen3MlpDims &dims);

template <typename T>
void Qwen3MLP(Tensor &hiddenStates, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &mlpOut,
    const Qwen3MlpDims &dims);

template <typename T>
void Qwen3LayerCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &layerOut, Qwen3AttenTileShapeConfig &tileConfig,
    const Qwen3AttentionDims &attnDims, const Qwen3MlpDims &mlpDims);

template <typename T>
void Qwen3Layer(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias, Tensor &vWeight,
    Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &gateWeight,
    Tensor &upWeight, Tensor &downWeight, Tensor &layerOut, Qwen3AttenTileShapeConfig &tileConfig,
    const Qwen3AttentionDims &attnDims, const Qwen3MlpDims &mlpDims);

/**
 * Qwen3LayerNew（decode / paged-attn 版本，Pre-Norm）:
 *   x -> RMSNorm -> (PagedAttentionProlog: QKV+RoPE+写KV cache) -> (PagedAttention: context)
 *     -> OProj -> +residual -> RMSNorm -> MLP -> +residual
 *
 * 约定：
 * - hiddenStates: 2D [B*S, hidden]，hidden = nQ * d
 * - key/value cache: 2D [blockNum*blockSize, nKv*d]，in-place 更新
 * - layerOut: 2D [B*S, hidden]
 */
template <typename T>
void Qwen3LayerNewCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &cos, Tensor &sin, Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, int nQ, int nKv, int blockSize,
    Tensor &layerOut, Tensor &keyCacheOut, Tensor &valueCacheOut, PaTileShapeConfig &paTileConfig,
    const Qwen3MlpDims &mlpDims, float softmaxScale = 0.0f, int maxUnrollTimes = 1, bool isNzFormat = false,
    int windowSize = 0);

template <typename T>
void Qwen3LayerNew(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &cos, Tensor &sin, Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, int nQ, int nKv, int blockSize,
    Tensor &layerOut, Tensor &keyCacheOut, Tensor &valueCacheOut, PaTileShapeConfig &paTileConfig,
    const Qwen3MlpDims &mlpDims, float softmaxScale = 0.0f, int maxUnrollTimes = 1, bool isNzFormat = false,
    int windowSize = 0);

} // namespace npu::tile_fwk

#endif // QWEN3_ATTENTION
