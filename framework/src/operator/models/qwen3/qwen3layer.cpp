#include <cmath>
#include <iostream>
#include <type_traits>
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/qwen3/qwen3layer.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

namespace {
template <typename T>
constexpr DataType GetDataType() {
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    }
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    }
    return DT_FP32;
}
} // namespace

/************************************* Qwen3Attention *************************************/
template <typename T>
void Qwen3AttentionCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &attentionOut, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &dims) {
    const DataType dt = GetDataType<T>();
    auto vecTile = tileConfig.vecTileShape;
    if (vecTile[0] <= 0 || vecTile[1] <= 0 || vecTile[2] <= 0 || vecTile[3] <= 0) {
        vecTile = {1, 1, 1, 1};
    }

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        // 全局设定一个安全的 vec tile，保证后续 Cast/基本算子合法（最后轴对齐到 head_dim）
        TileShape::Current().SetVecTile({1, 1, 1, dims.d});

        // 将输入转为目标精度，再用 FP32 计算保证精度
        auto x = hiddenStates;
        auto x2d = Reshape(x, {dims.b * dims.s, dims.n * dims.d});

        // QKV 线性
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
        auto q = Matrix::Matmul(dt, x2d, qWeight, false, false);
        q = Cast(Add(Cast(q, DT_FP32), Cast(qBias, DT_FP32)), dt);

        auto k = Matrix::Matmul(dt, x2d, kWeight, false, false);
        k = Cast(Add(Cast(k, DT_FP32), Cast(kBias, DT_FP32)), dt);

        auto v = Matrix::Matmul(dt, x2d, vWeight, false, false);
        v = Cast(Add(Cast(v, DT_FP32), Cast(vBias, DT_FP32)), dt);

        // Q/K RMSNorm 并乘以权重
        q = Reshape(q, {dims.b * dims.s * dims.n, dims.d});
        TileShape::Current().SetVecTile({16, 16});
        q = RmsNorm(q); // 与q=Cast(RmsNorm(Cast(q, DT_FP32)), dt)等价

        q = Mul(q, qNormWeight); // 与q = Cast(Mul(Cast(q, DT_FP32), Cast(qNormWeight, DT_FP32)), dt);等价
        q = Reshape(q, {dims.b, dims.s, dims.n, dims.d});

        k = Reshape(k, {dims.b * dims.s * dims.n_kv, dims.d});
        TileShape::Current().SetVecTile({16, 16});
        k = RmsNorm(k);
        k = Mul(k, kNormWeight);
        k = Reshape(k, {dims.b, dims.s, dims.n_kv, dims.d});

        // [B, S, N, D] -> [B, N, S, D]
        TileShape::Current().SetVecTile({1, 16, dims.n, dims.d});
        q = Transpose(q, {1, 2});
        TileShape::Current().SetVecTile({1, 16, dims.n_kv, dims.d});
        k = Transpose(k, {1, 2});
        TileShape::Current().SetVecTile({1, 16, dims.n_kv, dims.d});
        v = Reshape(v, {dims.b, dims.s, dims.n_kv, dims.d});
        v = Transpose(v, {1, 2});

        // 注意力分数
        // TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
        TileShape::Current().SetCubeTile({16, 16}, {16, 16}, {16, 16});
        auto score = Matrix::BatchMatmul(dt, q, k, false, true);

        float scaleVal = tileConfig.scale;
        if (scaleVal <= 0.0f) {
            scaleVal = 1.0f / std::sqrt(static_cast<float>(dims.d));
        }
        score = Mul(score, Element(dt, scaleVal)); // 这里Mul的输入是FP32或者BF16都可以

        // softmax + context
        // 设置 softmax vec tile，避免全 1 情况触发底层断言
        const int nTileSoftmax = std::max(1, std::min(dims.n, 2));
        const int sTileSoftmax = std::max(1, std::min(dims.s, 8));
        TileShape::Current().SetVecTile({1, nTileSoftmax, sTileSoftmax, sTileSoftmax});
        auto probs = SoftmaxNew(score);

        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
        auto context = Matrix::BatchMatmul(dt, probs, v, false, false);

        // [B, N, S, D] -> [B, S, N, D]
        TileShape::Current().SetVecTile({1, 16, dims.n, dims.d});
        context = Transpose(context, {1, 2});

        // 展平后输出投影
        auto context2d = Reshape(context, {dims.b * dims.s, dims.n * dims.d});

        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
        auto out = Matrix::Matmul(dt, context2d, oWeight, false, false);
        out = Add(Cast(out, DT_FP32), Cast(oBias, DT_FP32));

        attentionOut = Cast(out, dt);
    }
}

template <typename T>
void Qwen3Attention(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &attentionOut, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &dims) {
    FUNCTION("Qwen3Attention",
        {hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias, qNormWeight, kNormWeight},
        {attentionOut}) {
        Qwen3AttentionCompute<T>(hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias,
            qNormWeight, kNormWeight, attentionOut, tileConfig, dims);
    }
}

/************************************* Qwen3PagedAttentionProlog *************************************/
template <typename T>
void Qwen3PagedAttentionPrologCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight,
    Tensor &kBias, Tensor &vWeight, Tensor &vBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &cos, Tensor &sin,
    Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, int nQ, int nKv, int blockSize, Tensor &queryOut,
    Tensor &keyCacheOut, Tensor &valueCacheOut) {
    const DataType dt = GetDataType<T>();
    ASSERT(nQ % nKv == 0) << "GQA require nQ % nKv == 0";

    // hiddenStates: [B*S, H], H = nQ*d
    ASSERT(hiddenStates.GetShape().size() == SHAPE_DIM2);
    ASSERT(qWeight.GetShape().size() == SHAPE_DIM2);
    ASSERT(kWeight.GetShape().size() == SHAPE_DIM2);
    ASSERT(vWeight.GetShape().size() == SHAPE_DIM2);
    ASSERT(qNormWeight.GetShape().size() == SHAPE_DIM1);
    ASSERT(kNormWeight.GetShape().size() == SHAPE_DIM1);

    const int d = qNormWeight.GetShape()[0];
    ASSERT(d > 0) << "head dim d must be > 0";
    ASSERT(d % 2 == 0) << "RoPE requires even head dim, got d=" << d;

    const int b = cacheIndex.GetShape()[0];
    const int s = cacheIndex.GetShape()[1];
    const int h = nQ * d;
    const int kvh = nKv * d;

    int tileB = b; // 当前不切分 batch，后续可调整为 std::min(b, 8) 来进一步优化
    int tileS = s;
    int tileBS = tileB * tileS;
    SymbolicScalar bLoop = b / tileB;

    // MatMul tiling helper
    auto alignUp = [](int val, int base) -> int {
        if (val <= 0) {
            return base;
        }
        return ((val + base - 1) / base) * base;
    };
    auto setMatmulTile = [&](int m, int k, int n) {
        const int mTile = alignUp(std::min(std::max(m, 1), 64), 16);
        int nTile = alignUp(std::min(std::max(n, 1), 512), 16);
        int kTile = alignUp(std::min(std::max(k, 1), 256), 16);

        int bytesPerElem = 2;
        if (dt == DataType::DT_FP32) {
            bytesPerElem = 4;
        }
        const int64_t l0bBytes = 64 * 1024;
        while (static_cast<int64_t>(kTile) * static_cast<int64_t>(nTile) * bytesPerElem > l0bBytes) {
            if (kTile > 16) {
                kTile = std::max(16, kTile / 2);
            } else if (nTile > 16) {
                nTile = std::max(16, nTile / 2);
            } else {
                break;
            }
            nTile = alignUp(nTile, 16);
            kTile = alignUp(kTile, 16);
        }
        TileShape::Current().SetCubeTile({mTile, mTile}, {kTile, kTile}, {nTile, nTile});
        TileShape::Current().SetMatrixSize({m, k, n});
    };

    LOOP("LOOP_L0_Qwen3PAProlog_B", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        SymbolicScalar bOffset = bIdx * tileB;

        // View 切片输入
        auto hiddenView = View(hiddenStates, {tileBS, h}, {bOffset * tileS, 0});
        auto cosView = View(cos, {tileB, tileS, d}, {bOffset, 0, 0});
        auto sinView = View(sin, {tileB, tileS, d}, {bOffset, 0, 0});
        auto cacheIndexView = View(cacheIndex, {tileB, tileS}, {bOffset, 0});

        // QKV projection
        TileShape::Current().SetVecTile({std::min(32, tileBS), std::min(64, d)});

        config::SetSemanticLabel("qwen3_pa_prolog_q_proj");
        setMatmulTile(tileBS, h, h);
        auto q = Matrix::Matmul(dt, hiddenView, qWeight, false, false);
        q = Cast(Add(Cast(q, DT_FP32), Cast(qBias, DT_FP32)), dt); // [tileBS, nQ*d]

        config::SetSemanticLabel("qwen3_pa_prolog_k_proj");
        setMatmulTile(tileBS, h, kvh);
        auto k = Matrix::Matmul(dt, hiddenView, kWeight, false, false);
        k = Cast(Add(Cast(k, DT_FP32), Cast(kBias, DT_FP32)), dt); // [tileBS, nKv*d]

        config::SetSemanticLabel("qwen3_pa_prolog_v_proj");
        setMatmulTile(tileBS, h, kvh);
        auto v = Matrix::Matmul(dt, hiddenView, vWeight, false, false);
        v = Cast(Add(Cast(v, DT_FP32), Cast(vBias, DT_FP32)), dt); // [tileBS, nKv*d]

        // Q/K RMSNorm per head
        TileShape::Current().SetVecTile({std::min(16, tileBS * nQ), d});
        config::SetSemanticLabel("qwen3_pa_prolog_q_rmsnorm");
        q = Reshape(q, {tileBS * nQ, d});
        q = RmsNorm(q);
        q = Mul(q, qNormWeight);

        TileShape::Current().SetVecTile({std::min(16, tileBS * nKv), d});
        config::SetSemanticLabel("qwen3_pa_prolog_k_rmsnorm");
        k = Reshape(k, {tileBS * nKv, d});
        k = RmsNorm(k);
        k = Mul(k, kNormWeight);

        // reshape to 4D for RoPE
        config::SetSemanticLabel("qwen3_pa_prolog_reshape4d");
        auto q4d = Reshape(q, {tileB, tileS, nQ, d}, true);
        auto k4d = Reshape(k, {tileB, tileS, nKv, d}, true);
        auto v4d = Reshape(v, {tileB, tileS, nKv, d}, true);

        // RoPE 使用较小的 tile
        ASSERT(cos.GetShape().size() == SHAPE_DIM3);
        ASSERT(sin.GetShape().size() == SHAPE_DIM3);

        Tensor qEmbed(dt, {tileB, tileS, nQ, d}, "qEmbed");
        Tensor kEmbed(dt, {tileB, tileS, nKv, d}, "kEmbed");

        // RoPE 前设置 VecTile
        const int dTile = std::min(d, 64); // DeepSeek 使用 qkRopeHeadDim=64
        TileShape::Current().SetVecTile({std::min(32, tileB), 1, dTile});

        RoPETileShapeConfigNew ropeCfg{
            {tileB, tileS, dTile}, // threeDimsTileShape
            {tileB, tileS, 1, dTile}, // fourDimsTileShapeQ
            {tileB, tileS, 1, dTile}, // fourDimsTileShapeK
            {tileB, tileS, 1, dTile / 2, 2}, // fiveDimsTileShape
        };
        config::SetSemanticLabel("qwen3_pa_prolog_rope");
        ApplyRotaryPosEmbV2(q4d, k4d, cosView, sinView, qEmbed, kEmbed, 2, ropeCfg);

        // queryOut: 2D - 使用 Assemble 组装输出
        config::SetSemanticLabel("qwen3_pa_prolog_query_out");
        auto queryOutView = Reshape(qEmbed, {tileBS * nQ, d}, true);
        std::vector<SymbolicScalar> queryOffset = {bOffset * tileS * nQ, 0};
        Assemble(queryOutView, queryOffset, queryOut);

        // KV cache update
        config::SetSemanticLabel("qwen3_pa_prolog_cache_rows");
        auto kRows = Reshape(kEmbed, {tileBS, kvh}, true);
        auto vRows = Reshape(v4d, {tileBS, kvh}, true);
        config::SetSemanticLabel("qwen3_pa_prolog_vrows_materialize");
        auto vRowsMat = Assign(vRows);

        // ScatterUpdate - 保持原有 tile 设置
        TileShape::Current().SetVecTile({1, kvh});
        config::SetSemanticLabel("qwen3_pa_prolog_k_cache_scatter");
        keyCacheOut = ScatterUpdate(keyCache, cacheIndexView, kRows, 0, "PA_BSND", blockSize);
        config::SetSemanticLabel("qwen3_pa_prolog_v_cache_scatter");
        valueCacheOut = ScatterUpdate(valueCache, cacheIndexView, vRowsMat, 0, "PA_BSND", blockSize);
    }
}

template <typename T>
void Qwen3PagedAttentionProlog(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &cos, Tensor &sin,
    Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, int nQ, int nKv, int blockSize, Tensor &queryOut,
    Tensor &keyCacheOut, Tensor &valueCacheOut) {
    FUNCTION("Qwen3PagedAttentionProlog",
        {
            hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, qNormWeight, kNormWeight, cos, sin,
            cacheIndex, keyCache, valueCache
    },
        {queryOut}, {{keyCacheOut, keyCache}, {valueCacheOut, valueCache}}) {
        Qwen3PagedAttentionPrologCompute<T>(hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, qNormWeight,
            kNormWeight, cos, sin, cacheIndex, keyCache, valueCache, nQ, nKv, blockSize, queryOut, keyCacheOut,
            valueCacheOut);
    }
}

/************************************* Qwen3PagedAttention *************************************/
/**
 * @brief Qwen3 PagedAttention 计算核（decode阶段）：使用 paged KV cache，支持 GQA 与可选 sliding window。
 *
 * 约定：
 * - query/out 为 2D 展平：shape = [B * S_q * nQ, d]（等价于 [B, S_q, nQ, d]，
 *   其中 S_q(s1) = query.shape[0] / (B * nQ)）
 * - key/value cache 为 2D：shape = [blockNum * blockSize, nKv * d]
 *   - 行索引 row = globalBlockId * blockSize + offsetInBlock
 *   - 列按 kv head 拼接：headIdx 切片为 [headIdx*d, (headIdx+1)*d)
 *
 * @tparam T 数据类型（如 fp16/bf16）。
 * @param[in] query 2D Tensor，shape = [B * S_q * nQ, d]。
 * @param[in] keyCache 2D Tensor，shape = [blockNum * blockSize, nKv * d]。
 * @param[in] valueCache 2D Tensor，shape = [blockNum * blockSize, nKv * d]。
 * @param[in] blockTable 2D Tensor，shape = [B, maxBlockNumPerBatch]，dtype=int32。
 *                        blockTable[b, bn] 为第 b 条序列第 bn 个 block 的全局 block id（用于索引 cache 行）。
 * @param[in] actSeqs 1D Tensor，shape = [B]，dtype=int32。actSeqs[b] 为第 b 条序列当前 KV 的有效长度（token 数）。
 * @param[in] nQ Query head 数。
 * @param[in] nKv KV head 数（要求 nQ % nKv == 0，group = nQ / nKv）。
 * @param[in] blockSize 每个 cache block 的 token 数。
 * @param[in] softmaxScale 注意力缩放系数（通常为 1/sqrt(d)）。
 * @param[out] attentionOut 2D Tensor，shape = [B * S_q * nQ, d]。
 * @param[in] tileConfig 内部 tiling 配置（headNumQTile、c1/v1/c2/v2 tile shape 等）。
 * @param[in] maxUnrollTimes block 循环的最大 unroll hint（内部按 2 的幂做 unroll）。
 * @param[in] isNzFormat 兼容参数：cache Tensor 自身携带物理 format，本函数仅按 View 切片使用。
 * @param[in] windowSize sliding window 大小（token 数）；<=0 表示全量 attention，>0 仅关注最后 windowSize 个 token。
 */
template <typename T>
void Qwen3PagedAttentionCompute(Tensor &query, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, int nQ, int nKv, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes, bool isNzFormat, int windowSize) {
    (void)isNzFormat; // cache tensor本身携带format，算子内部只按view切片使用
    auto dtype = query.GetStorage()->Datatype();

    ASSERT(nQ % nKv == 0) << "GQA require nQ % nKv == 0, got nQ=" << nQ << ", nKv=" << nKv;
    const int group = nQ / nKv; // nKv等于group数，这里的group意思是每个group内的q头数，实际上是nQ_in_group

    // query: [B*s1*nQ, d]
    const int d = query.GetShape()[1];

    const int gTile = tileConfig.headNumQTile;
    ASSERT(group >= gTile) << "headNumQTile must be <= group(nQ/nKv). group=" << group << ", gTile=" << gTile;
    ASSERT(group % gTile == 0) << "Currently requires (nQ/nKv) % headNumQTile == 0 to avoid tail heads. group=" << group
                               << ", gTile=" << gTile;

    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    // 迭代变量
    SymbolicScalar batchSize = blockTable.GetShape()[0];
    // 入参是B*S*N合轴（2D），这里推回s1
    SymbolicScalar s1Size = query.GetShape()[0] / batchSize / nQ; // s1

    // 优化：合并 L2 (nKv) 和 L3 (gLoop) 循环为单层循环
    // 直接按 nQ/nTile 切分，减少循环嵌套深度
    const int nTile = gTile;           // 每次处理 nTile 个 Q heads
    SymbolicScalar nLoop = nQ / nTile; // nQ=32, nTile=4 -> nLoop=8

    // 注意：这里只是Compute，外层FUNCTION由调用者包装（测试里会包一层更方便打印）
    LOOP("LOOP_L0_b_Qwen3PA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1), {}, true) {
        SymbolicScalar curActSeq = GetTensorData(actSeqs, {bIdx});
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_Qwen3PA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Size, 1)) {
            // 支持s1>1(MTP)的轻量因果长度：pos = actSeq - s1 + 1 + s1Idx
            SymbolicScalar curSeq = std::max(curActSeq - s1Size + 1 + s1Idx, 0);
            curSeq.AsIntermediateVariable();
            SymbolicScalar bnPerBatch = (curSeq + blockSize - 1) / blockSize;
            bnPerBatch.AsIntermediateVariable();
            // sliding window: only attend last windowSize tokens (windowSize<=0 means full)
            SymbolicScalar winActual = (windowSize > 0) ? std::min(curSeq, windowSize) : curSeq;
            SymbolicScalar startPos = curSeq - winActual;
            SymbolicScalar blockStartIndex = startPos / blockSize;
            SymbolicScalar blockStartOffset = startPos % blockSize;
            SymbolicScalar blockEndIndex = bnPerBatch - 1;
            SymbolicScalar tableLoop = blockEndIndex - blockStartIndex + 1;
            tableLoop.AsIntermediateVariable();

            // 优化后：合并 L2+L3 为单层循环，直接按 nQ/nTile 切分
            LOOP("LOOP_L2_nQ_Qwen3PA", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                // 计算当前 nTile 对应的 KV head index (GQA: 多个 Q heads 共享一个 KV head)
                // kvIdx = (nIdx * nTile) / group，当 nTile=group 时，kvIdx = nIdx
                SymbolicScalar kvIdx = (nIdx * nTile) / group;
                kvIdx.AsIntermediateVariable();

                const int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {curNTile, d}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {curNTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {curNTile, 1}, "miUpdate");

                // query 2D layout: [b, s1, nQ, d] -> flatten(b*s1*nQ, d)
                SymbolicScalar curOffset = bIdx * s1Size * nQ + s1Idx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0};

                LOOP("LOOP_L3_bn_Qwen3PA", FunctionType::DYNAMIC_LOOP, bnIdx, LoopRange(0, tableLoop, 1),
                    PowersOf2(maxUnrollTimes)) {
                    SymbolicScalar bn = blockStartIndex + bnIdx;
                    const int curS2Tile = blockSize;

                    config::SetSemanticLabel("Qwen3PA");
                    auto qi = View(query, {curNTile, d}, {curOffset, 0});

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();

                    // key/value cache 2D layout: [blockNum*blockSize, nKv*d]
                    SymbolicScalar validS2Full = std::min(curSeq - bn * blockSize, blockSize);
                    SymbolicScalar curStartOffset = 0;
                    IF(IsLoopBegin(bnIdx, 0)) {
                        curStartOffset = blockStartOffset;
                    }
                    SymbolicScalar validS2 = validS2Full - curStartOffset;
                    auto kj = View(
                        keyCache, {curS2Tile, d}, {validS2, d}, {curBlockIdx * blockSize + curStartOffset, kvIdx * d});
                    auto vj = View(valueCache, {curS2Tile, d}, {validS2, d},
                        {curBlockIdx * blockSize + curStartOffset, kvIdx * d});

                    // C1: (q @ kT)
                    config::SetSemanticLabel("Qwen3PA_QKMM");
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false, true);

                    // softmax streaming
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Mul");
                    auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (nTile, s2)
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Amax");
                    auto tildaMij = Amax(sijScale, -1, true); // (nTile, 1)
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Sub");
                    auto tsub = Sub(sijScale, tildaMij);
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Exp");
                    auto tildaPij = Exp(tsub);
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Cast");
                    auto tildaPijCast = Cast(tildaPij, dtype);
                    config::SetSemanticLabel("Qwen3PA_SoftMax_Sum");
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTile, 1)

                    IF(IsLoopBegin(bnIdx, 0)) {
                        config::SetSemanticLabel("Qwen3PA_PVMM");
                        // C2
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        TileShape::Current().SetMatrixSize(
                            {tildaPijCast.GetShape()[0], tildaPijCast.GetShape()[1], vj.GetShape()[1]});
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijCast, vj, false, false);

                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF(IsLoopEnd(bnIdx, tableLoop)) {
                            config::SetSemanticLabel("Qwen3PA_Assemble");
                            auto oiFinal = Div(oiTmp, tildaLij);
                            Assemble(oiFinal, oiOffset, attentionOut);
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                            liUpdate = tildaLij;
                            miUpdate = tildaMij;
                        }
                    }
                    ELSE {
                        // online update
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        auto miNew = Maximum(mi, tildaMij); // (nTile, 1)
                        auto t1 = Sub(mi, miNew);
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew);
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij);
                        auto t6 = Mul(t2, li);
                        auto liNew = Add(t6, t5);

                        auto q3 = Mul(oi, t2); // (nTile, d)
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        config::SetSemanticLabel("Qwen3PA_UpdateMM2");
                        TileShape::Current().SetMatrixSize(
                            {tildaPijCast.GetShape()[0], tildaPijCast.GetShape()[1], vj.GetShape()[1]});
                        auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijCast, vj, false, false); // (nTile, d)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);
                        auto oiTmp = Add(q3, q2);

                        IF(IsLoopEnd(bnIdx, tableLoop)) {
                            auto oiFinal = Div(oiTmp, liNew);
                            Assemble(oiFinal, oiOffset, attentionOut);
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                            liUpdate = liNew;
                            miUpdate = miNew;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void Qwen3PagedAttention(Tensor &query, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable, Tensor &actSeqs,
    int nQ, int nKv, int blockSize, float softmaxScale, Tensor &attentionOut, PaTileShapeConfig &tileConfig,
    int maxUnrollTimes, bool isNzFormat, int windowSize) {
    FUNCTION("Qwen3PagedAttention", {query, keyCache, valueCache, blockTable, actSeqs}, {attentionOut}) {
        Qwen3PagedAttentionCompute<T>(query, keyCache, valueCache, blockTable, actSeqs, nQ, nKv, blockSize,
            softmaxScale, attentionOut, tileConfig, maxUnrollTimes, isNzFormat, windowSize);
    }
}

/************************************* Qwen3MLP *************************************/
template <typename T>
void Qwen3MLPCompute(Tensor &hiddenStates, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &mlpOut,
    const Qwen3MlpDims &dims) {
    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        const DataType dt = GetDataType<T>();
        auto x = hiddenStates;

        // NOTE:
        // - MLP 包含 3 个大 MatMul（gate/up/down），对大模型维度 (H=4096, INTER=12288) 若 cube tile 过小，
        //   会导致 MatMul 被切得过碎，任务数暴涨，进而拖慢 codegen/compile 与运行速度。
        // - 这里按 (M,K,N) 动态选择较大的 cube tile（上限 256），显著降低 task 颗粒度；小维度仍保持合理。
        auto alignUp = [](int val, int base) -> int {
            if (val <= 0) {
                return base;
            }
            return ((val + base - 1) / base) * base;
        };
        auto setMatmulTile = [&](int m, int k, int n) {
            // MatMul tile: (m, k, n) = (row, reduce, col)
            // M 维一般不大（B*S），上限取 64；K/N 上限取 256，避免过大导致片上压力。
            const int mTile = alignUp(std::min(std::max(m, 1), 64), 16);
            int kTile = alignUp(std::min(std::max(k, 1), 256), 16);
            int nTile = alignUp(std::min(std::max(n, 1), 256), 16);

            // L0B 容量约束：B 矩阵 tile 通常以 (kTile x nTile) 搬运到 MEM_L0B。
            // 例如 BF16 下 256x256 = 131072B，会超过 MEM_L0B=65536B
            int bytesPerElem = 2; // FP16/BF16
            if (dt == DataType::DT_FP32) {
                bytesPerElem = 4;
            }
            const int64_t l0bBytes = 64 * 1024;
            while (static_cast<int64_t>(kTile) * static_cast<int64_t>(nTile) * bytesPerElem > l0bBytes) {
                // 优先缩小更大的那一维，直到满足 L0B 容量
                if (nTile >= kTile) {
                    if (nTile <= 16) {
                        break;
                    }
                    nTile = std::max(16, (nTile / 2));
                } else {
                    if (kTile <= 16) {
                        break;
                    }
                    kTile = std::max(16, (kTile / 2));
                }
                // 保持 16 对齐
                nTile = alignUp(nTile, 16);
                kTile = alignUp(kTile, 16);
            }

            TileShape::Current().SetCubeTile({mTile, mTile}, {kTile, kTile}, {nTile, nTile});
            TileShape::Current().SetMatrixSize({m, k, n});
        };

        // elementwise vec tile（2D），默认按 hidden 维对齐；后续会根据 matmul 的 matrix_size/cube_tile 做主要优化
        TileShape::Current().SetVecTile({1, dims.h});

        const int m = dims.b * dims.s;

        config::SetSemanticLabel("mlp_gate_proj");
        setMatmulTile(m, dims.h, dims.inter);
        auto gate = Matrix::Matmul(dt, x, gateWeight, false, false);
        config::SetSemanticLabel("mlp_up_proj");
        setMatmulTile(m, dims.h, dims.inter);
        auto up = Matrix::Matmul(dt, x, upWeight, false, false);

        // Sigmoid 函数会强制 in-place 修改输入 tensor 的类型，建议用独立 tensor 承接其输入，避免影响后续算子。
        config::SetSemanticLabel("mlp_gate_act");
        // 与 python golden 对齐：gate_act = gate * sigmoid(gate.to(fp32)).to(dtype)
        auto gateFp32 = Cast(gate, DataType::DT_FP32);
        auto gateSigFp32 = Sigmoid(gateFp32);
        auto gateSigDt = Cast(gateSigFp32, dt);
        auto gateAct = Mul(Cast(gate, dt), gateSigDt); // SiLU: x * sigmoid(x)

        // inter = gate_act * up（按 bf16/fp16 逐元素乘，再转回 dt）
        config::SetSemanticLabel("mlp_inter");
        auto interFp32 = Mul(Cast(gateAct, DataType::DT_FP32), Cast(up, DataType::DT_FP32));
        auto inter = Cast(interFp32, dt);

        config::SetSemanticLabel("mlp_down_proj");
        setMatmulTile(m, dims.inter, dims.h);
        auto out = Matrix::Matmul(dt, inter, downWeight, false, false);
        mlpOut = out;
    }
}

template <typename T>
void Qwen3MLP(Tensor &hiddenStates, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &mlpOut,
    const Qwen3MlpDims &dims) {
    FUNCTION("Qwen3MLP", {hiddenStates, gateWeight, upWeight, downWeight}, {mlpOut}) {
        Qwen3MLPCompute<T>(hiddenStates, gateWeight, upWeight, downWeight, mlpOut, dims);
    }
}

/************************************* Qwen3Layer *************************************/
template <typename T>
void Qwen3LayerCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, Tensor &layerOut, Qwen3AttenTileShapeConfig &tileConfig,
    const Qwen3AttentionDims &attnDims, const Qwen3MlpDims &mlpDims) {
    const DataType dt = GetDataType<T>();
    // Align with golden python(Qwen3LayerTest.core):
    //   x -> RMSNorm -> Attention -> +residual -> RMSNorm -> MLP -> +residual
    Tensor norm1(dt, {attnDims.b * attnDims.s, attnDims.n * attnDims.d}, "LayerNorm1");
    Tensor attnOut(dt, {attnDims.b * attnDims.s, attnDims.n * attnDims.d}, "LayerAttnOut");
    Tensor h1(dt, {attnDims.b * attnDims.s, attnDims.n * attnDims.d}, "LayerH1");
    Tensor norm2(dt, {attnDims.b * attnDims.s, attnDims.n * attnDims.d}, "LayerNorm2");
    Tensor mlpOut(dt, {mlpDims.b * mlpDims.s, mlpDims.h}, "LayerMlpOut");

    // RMSNorm 1
    LOOP("LOOP_L0_RMSNorm1", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        TileShape::Current().SetVecTile({16, 16});
        norm1 = RmsNorm(hiddenStates);
    }

    // Attention(norm1)
    Qwen3AttentionCompute<T>(norm1, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias, qNormWeight,
        kNormWeight, attnOut, tileConfig, attnDims);

    // residual 1: hiddenStates + attnOut
    LOOP("LOOP_L0_ResidualAdd1", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        TileShape::Current().SetVecTile({16, 16});
        h1 = Cast(Add(Cast(hiddenStates, DT_FP32), Cast(attnOut, DT_FP32)), dt);
    }

    // RMSNorm 2
    LOOP("LOOP_L0_RMSNorm2", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        TileShape::Current().SetVecTile({16, 16});
        norm2 = RmsNorm(h1);
    }

    // MLP(norm2)
    Qwen3MLPCompute<T>(norm2, gateWeight, upWeight, downWeight, mlpOut, mlpDims);

    // residual 2: h1 + mlpOut
    LOOP("LOOP_L0_ResidualAdd2", FunctionType::DYNAMIC_LOOP, xxxx, LoopRange(1)) {
        (void)xxxx;
        TileShape::Current().SetVecTile({16, 16});
        layerOut = Cast(Add(Cast(h1, DT_FP32), Cast(mlpOut, DT_FP32)), dt);
    }
}

template <typename T>
void Qwen3Layer(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias, Tensor &vWeight,
    Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight, Tensor &gateWeight,
    Tensor &upWeight, Tensor &downWeight, Tensor &layerOut, Qwen3AttenTileShapeConfig &tileConfig,
    const Qwen3AttentionDims &attnDims, const Qwen3MlpDims &mlpDims) {
    FUNCTION("Qwen3Layer",
        {hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias, qNormWeight, kNormWeight,
            gateWeight, upWeight, downWeight},
        {layerOut}) {
        Qwen3LayerCompute<T>(hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias, qNormWeight,
            kNormWeight, gateWeight, upWeight, downWeight, layerOut, tileConfig, attnDims, mlpDims);
    }
}

/************************************* Qwen3LayerNew *************************************/
// Qwen3LayerNew (decode / paged-attn, Pre-Norm):
//   hiddenStates -> RMSNorm -> Prolog (QKV + RoPE + write KV cache)
//     -> PagedAttention -> OProj -> +residual
//     -> RMSNorm -> MLP -> +residual -> layerOut
//
// 注意：此函数没有外层 FUNCTION 包装。它调用多个独立的子 FUNCTION：
// - Qwen3PagedAttentionProlog (独立 FUNCTION)
// - Qwen3PagedAttention (独立 FUNCTION)
// - Qwen3MLP (独立 FUNCTION)
// 以及内联的小 FUNCTION (RMSNorm, OProj+Residual 等)
//
// 调用方需要在外层用 FUNCTION 包装来注册所有输入/输出 tensor。
template <typename T>
void Qwen3LayerNewCompute(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &cos, Tensor &sin, Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, int nQ, int nKv, int blockSize,
    Tensor &layerOut, Tensor &keyCacheOut, Tensor &valueCacheOut, PaTileShapeConfig &paTileConfig,
    const Qwen3MlpDims &mlpDims, float softmaxScale, int maxUnrollTimes, bool isNzFormat, int windowSize) {
    const DataType dt = GetDataType<T>();

    // hiddenStates: [B*S, hidden], hidden = nQ * d
    const int64_t bs = hiddenStates.GetShape()[0];
    const int64_t hidden = hiddenStates.GetShape()[1];
    const int64_t d = qNormWeight.GetShape()[0];
    ASSERT(d > 0);
    ASSERT(hidden == static_cast<int64_t>(nQ) * d)
        << "hiddenStates.shape[1] must be nQ*d, got hidden=" << hidden << ", nQ=" << nQ << ", d=" << d;

    // ========== LOOP 1: RMSNorm 1 ==========
    Tensor norm1(dt, {bs, hidden}, "LayerNewNorm1");
    LOOP("LOOP_L0_LayerNew_RMSNorm1", FunctionType::DYNAMIC_LOOP, xxxx1, LoopRange(1)) {
        (void)xxxx1;
        TileShape::Current().SetVecTile({16, 16});
        config::SetSemanticLabel("qwen3_layernew_rmsnorm1");
        norm1 = RmsNorm(hiddenStates);
    }

    // ========== LOOP 2: Prolog ==========
    Tensor queryOut(dt, {bs * nQ, d}, "LayerNewQueryOut");
    Qwen3PagedAttentionPrologCompute<T>(norm1, qWeight, qBias, kWeight, kBias, vWeight, vBias, qNormWeight, kNormWeight,
        cos, sin, cacheIndex, keyCache, valueCache, nQ, nKv, blockSize, queryOut, keyCacheOut, valueCacheOut);

    // ========== LOOP 3: PagedAttention ==========
    Tensor paOut(DT_FP32, {bs * nQ, d}, "LayerNewPAOut");
    float scaleVal = softmaxScale;
    if (scaleVal <= 0.0f) {
        scaleVal = 1.0f / std::sqrt(static_cast<float>(d));
    }
    Qwen3PagedAttentionCompute<T>(queryOut, keyCacheOut, valueCacheOut, blockTable, actSeqs, nQ, nKv, blockSize,
        scaleVal, paOut, paTileConfig, maxUnrollTimes, isNzFormat, windowSize);

    // ========== LOOP 4: OProj + Residual1 ==========
    Tensor h1(dt, {bs, hidden}, "LayerNewH1");
    LOOP("LOOP_L0_LayerNew_OProjRes1", FunctionType::DYNAMIC_LOOP, xxxx4, LoopRange(1)) {
        (void)xxxx4;
        config::SetSemanticLabel("qwen3_layernew_oproj_res1");
        auto context2d = Reshape(paOut, {bs, hidden});
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
        auto tmp = Matrix::Matmul(DT_FP32, context2d, Cast(oWeight, DT_FP32), false, false);
        tmp = Add(tmp, Cast(oBias, DT_FP32));
        auto attnOut = Cast(tmp, dt);
        TileShape::Current().SetVecTile({16, 16});
        h1 = Cast(Add(Cast(hiddenStates, DT_FP32), Cast(attnOut, DT_FP32)), dt);
    }

    // ========== LOOP 5: RMSNorm 2 ==========
    Tensor norm2(dt, {bs, hidden}, "LayerNewNorm2");
    LOOP("LOOP_L0_LayerNew_RMSNorm2", FunctionType::DYNAMIC_LOOP, xxxx5, LoopRange(1)) {
        (void)xxxx5;
        TileShape::Current().SetVecTile({16, 16});
        config::SetSemanticLabel("qwen3_layernew_rmsnorm2");
        norm2 = RmsNorm(h1);
    }

    // ========== LOOP 6: MLP ==========
    Tensor mlpOut(dt, {bs, hidden}, "LayerNewMlpOut");
    Qwen3MLPCompute<T>(norm2, gateWeight, upWeight, downWeight, mlpOut, mlpDims);

    // ========== LOOP 7: Residual 2 ==========
    LOOP("LOOP_L0_LayerNew_Res2", FunctionType::DYNAMIC_LOOP, xxxx7, LoopRange(1)) {
        (void)xxxx7;
        TileShape::Current().SetVecTile({16, 16});
        config::SetSemanticLabel("qwen3_layernew_residual2");
        layerOut = Cast(Add(Cast(h1, DT_FP32), Cast(mlpOut, DT_FP32)), dt);
    }
}

// Qwen3LayerNew 带 FUNCTION 包装版本
// 如需使用，调用方应自行用 FUNCTION 包装 Qwen3LayerNewCompute
template <typename T>
void Qwen3LayerNew(Tensor &hiddenStates, Tensor &qWeight, Tensor &qBias, Tensor &kWeight, Tensor &kBias,
    Tensor &vWeight, Tensor &vBias, Tensor &oWeight, Tensor &oBias, Tensor &qNormWeight, Tensor &kNormWeight,
    Tensor &cos, Tensor &sin, Tensor &cacheIndex, Tensor &keyCache, Tensor &valueCache, Tensor &blockTable,
    Tensor &actSeqs, Tensor &gateWeight, Tensor &upWeight, Tensor &downWeight, int nQ, int nKv, int blockSize,
    Tensor &layerOut, Tensor &keyCacheOut, Tensor &valueCacheOut, PaTileShapeConfig &paTileConfig,
    const Qwen3MlpDims &mlpDims, float softmaxScale, int maxUnrollTimes, bool isNzFormat, int windowSize) {
    // 直接调用 Compute 版本，调用方需要自己处理 FUNCTION 包装
    Qwen3LayerNewCompute<T>(hiddenStates, qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias, qNormWeight,
        kNormWeight, cos, sin, cacheIndex, keyCache, valueCache, blockTable, actSeqs, gateWeight, upWeight, downWeight,
        nQ, nKv, blockSize, layerOut, keyCacheOut, valueCacheOut, paTileConfig, mlpDims, softmaxScale, maxUnrollTimes,
        isNzFormat, windowSize);
}

} // namespace npu::tile_fwk

// 显式实例化
template void npu::tile_fwk::Qwen3AttentionCompute<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Qwen3AttenTileShapeConfig &,
    const Qwen3AttentionDims &);
template void npu::tile_fwk::Qwen3Attention<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Qwen3AttenTileShapeConfig &,
    const Qwen3AttentionDims &);

template void npu::tile_fwk::Qwen3PagedAttentionCompute<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, int, int, int, float, Tensor &, PaTileShapeConfig &, int, bool, int);
template void npu::tile_fwk::Qwen3PagedAttention<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, int, int, int, float, Tensor &, PaTileShapeConfig &, int, bool, int);

template void npu::tile_fwk::Qwen3PagedAttentionPrologCompute<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int,
    int, int, Tensor &, Tensor &, Tensor &);
template void npu::tile_fwk::Qwen3PagedAttentionProlog<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int, int, int,
    Tensor &, Tensor &, Tensor &);

template void npu::tile_fwk::Qwen3MLPCompute<npu::tile_fwk::bfloat16>(
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, const Qwen3MlpDims &);
template void npu::tile_fwk::Qwen3MLP<npu::tile_fwk::bfloat16>(
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, const Qwen3MlpDims &);

template void npu::tile_fwk::Qwen3LayerCompute<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Qwen3AttenTileShapeConfig &, const Qwen3AttentionDims &, const Qwen3MlpDims &);
template void npu::tile_fwk::Qwen3Layer<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Qwen3AttenTileShapeConfig &, const Qwen3AttentionDims &, const Qwen3MlpDims &);

template void npu::tile_fwk::Qwen3LayerNewCompute<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int, int, int, Tensor &, Tensor &, Tensor &,
    PaTileShapeConfig &, const Qwen3MlpDims &, float, int, bool, int);
template void npu::tile_fwk::Qwen3LayerNew<npu::tile_fwk::bfloat16>(Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
    Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int, int, int, Tensor &, Tensor &, Tensor &, PaTileShapeConfig &,
    const Qwen3MlpDims &, float, int, bool, int);
