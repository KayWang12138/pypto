#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

// 常量定义
constexpr float SQRT_2_PI = 0.7978845608f;      // sqrt(2/π)
constexpr float GELU_COEF = 0.044715f;          // GELU tanh 近似系数
constexpr float SQRT_2_INV = 0.7071067812f;     // 1/sqrt(2)

template <typename T>
class SumLstmKernel {
public:
    __aicore__ inline SumLstmKernel() {}

    __aicore__ inline void Init(GM_ADDR states4d, GM_ADDR z4_4d, GM_ADDR prevCell,
                                GM_ADDR wCell, GM_ADDR bCell, GM_ADDR wState, GM_ADDR bState,
                                GM_ADDR outState, GM_ADDR outCell, GM_ADDR workspace,
                                const SumLstmTilingData* tilingData) {
        // 获取 Tiling 参数
        totalSamples = tilingData->totalSamples;
        hiddenDim = tilingData->hiddenDim;
        gatedDim = tilingData->gatedDim;
        coreNum = tilingData->coreNum;
        samplesPerCore = tilingData->samplesPerCore;
        remainSamples = tilingData->remainSamples;
        tileNumPerCore = tilingData->tileNumPerCore;
        tileSamples = tilingData->tileSamples;
        lastTileSamples = tilingData->lastTileSamples;
        hiddenDimAligned = tilingData->hiddenDimAligned;
        gatedDimAligned = tilingData->gatedDimAligned;
        alpha = tilingData->alpha;
        epsCell = tilingData->epsCell;
        epsState = tilingData->epsState;
        useFastGelu = tilingData->useFastGelu;
        hasWCell = tilingData->hasWCell;
        hasBCell = tilingData->hasBCell;
        hasWState = tilingData->hasWState;
        hasBState = tilingData->hasBState;

        // 计算当前核心处理范围
        uint32_t blockIdx = GetBlockIdx();
        uint32_t startSample = blockIdx * samplesPerCore;
        if (blockIdx < remainSamples) {
            startSample += blockIdx;
            coreSamples = samplesPerCore + 1;
        } else {
            startSample += remainSamples;
            coreSamples = samplesPerCore;
        }

        // 设置 Global Memory 地址
        gmStates4d.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(states4d) + startSample * gatedDim, coreSamples * gatedDim);
        gmZ4_4d.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z4_4d) + startSample * gatedDim, coreSamples * gatedDim);
        gmPrevCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(prevCell) + startSample * hiddenDim, coreSamples * hiddenDim);
        gmOutState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outState) + startSample * hiddenDim, coreSamples * hiddenDim);
        gmOutCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outCell) + startSample * hiddenDim, coreSamples * hiddenDim);

        // 可选输入
        if (hasWCell) {
            gmWCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wCell), hiddenDim);
        }
        if (hasBCell) {
            gmBCell.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bCell), hiddenDim);
        }
        if (hasWState) {
            gmWState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wState), hiddenDim);
        }
        if (hasBState) {
            gmBState.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bState), hiddenDim);
        }

        // 初始化 Pipe Buffer
        uint32_t bufferSize = tileSamples * hiddenDimAligned * sizeof(T);
        uint32_t bufferSizeGated = tileSamples * gatedDimAligned * sizeof(T);

        // 输入队列
        pipe.InitBuffer(inQueueStates, BUFFER_NUM, bufferSizeGated);
        pipe.InitBuffer(inQueueZ4, BUFFER_NUM, bufferSizeGated);
        pipe.InitBuffer(inQueuePrevCell, BUFFER_NUM, bufferSize);

        // 输出队列
        pipe.InitBuffer(outQueueState, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueCell, BUFFER_NUM, bufferSize);

        // 中间计算 buffer (门控和激活)
        pipe.InitBuffer(tempBuffer1, bufferSize);
        pipe.InitBuffer(tempBuffer2, bufferSize);
        pipe.InitBuffer(tempBuffer3, bufferSize);
        pipe.InitBuffer(tempBuffer4, bufferSize);

        // 可选输入 buffer
        if (hasWCell || hasBCell || hasWState || hasBState) {
            pipe.InitBuffer(optBuffer, hiddenDimAligned * sizeof(T));
        }

        // RMSNorm 归约工作 buffer (float 类型，用于高精度累加)
        pipe.InitBuffer(reduceWorkBuf, hiddenDimAligned * sizeof(float));

        // 计算 tile 数量
        actualTileNum = (coreSamples + tileSamples - 1) / tileSamples;
    }

    __aicore__ inline void Process() {
        for (uint32_t tileIdx = 0; tileIdx < actualTileNum; ++tileIdx) {
            uint32_t currentTileSamples = tileSamples;
            if (tileIdx == actualTileNum - 1) {
                currentTileSamples = coreSamples - tileIdx * tileSamples;
            }
            currentTileSize = currentTileSamples;
            tileOffset = tileIdx * tileSamples;

            CopyIn(tileIdx, currentTileSamples);
            Compute(currentTileSamples);
            CopyOut(tileIdx, currentTileSamples);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t samples) {
        // 搬运 states_4d (GM -> Local)
        LocalTensor<T> statesLocal = inQueueStates.AllocTensor<T>();
        uint32_t gatedBytes = gatedDim * sizeof(T);
        uint8_t gatedRpad = (gatedDimAligned - gatedDim);
        AscendC::DataCopyExtParams copyParamsGated = {(uint16_t)samples, gatedBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParamsGated = {false, 0, gatedRpad, (T)0};
        AscendC::DataCopyPad<T>(statesLocal, gmStates4d[tileOffset * gatedDim], copyParamsGated, padParamsGated);
        inQueueStates.EnQue(statesLocal);

        // 搬运 z4_4d
        LocalTensor<T> z4Local = inQueueZ4.AllocTensor<T>();
        AscendC::DataCopyPad<T>(z4Local, gmZ4_4d[tileOffset * gatedDim], copyParamsGated, padParamsGated);
        inQueueZ4.EnQue(z4Local);

        // 搬运 prev_cell
        LocalTensor<T> prevCellLocal = inQueuePrevCell.AllocTensor<T>();
        uint32_t hiddenBytes = hiddenDim * sizeof(T);
        uint8_t hiddenRpad = (hiddenDimAligned - hiddenDim);
        AscendC::DataCopyExtParams copyParamsHidden = {(uint16_t)samples, hiddenBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParamsHidden = {false, 0, hiddenRpad, (T)0};
        AscendC::DataCopyPad<T>(prevCellLocal, gmPrevCell[tileOffset * hiddenDim], copyParamsHidden, padParamsHidden);
        inQueuePrevCell.EnQue(prevCellLocal);
    }

    __aicore__ inline void Compute(uint32_t samples) {
        // 获取输入数据
        LocalTensor<T> statesLocal = inQueueStates.DeQue<T>();
        LocalTensor<T> z4Local = inQueueZ4.DeQue<T>();
        LocalTensor<T> prevCellLocal = inQueuePrevCell.DeQue<T>();

        // 获取输出 buffer
        LocalTensor<T> outStateLocal = outQueueState.AllocTensor<T>();
        LocalTensor<T> outCellLocal = outQueueCell.AllocTensor<T>();

        // 获取临时 buffer
        LocalTensor<T> temp1 = tempBuffer1.Get<T>();
        LocalTensor<T> temp2 = tempBuffer2.Get<T>();
        LocalTensor<T> temp3 = tempBuffer3.Get<T>();
        LocalTensor<T> temp4 = tempBuffer4.Get<T>();

        uint32_t computeLen = samples * hiddenDim;
        uint32_t computeLenAligned = samples * hiddenDimAligned;

        // 处理每个样本
        for (uint32_t s = 0; s < samples; ++s) {
            // 使用对齐后的维度进行索引 (DataCopyPad 后 Local tensor 的行间距是对齐后的)
            uint32_t statesOffsetAligned = s * gatedDimAligned;
            uint32_t hiddenOffsetAligned = s * hiddenDimAligned;

            // Step 1: 计算门控预激活值
            T alphaT = static_cast<T>(alpha);

            // Forget gate: pre_f = s0 + alpha * z0
            LocalTensor<T> preF = temp1[hiddenOffsetAligned];
            Muls(preF, z4Local[statesOffsetAligned], alphaT, hiddenDim);
            Add(preF, statesLocal[statesOffsetAligned], preF, hiddenDim);

            // Input gate: pre_i = s1 + alpha * z1
            LocalTensor<T> preI = temp2[hiddenOffsetAligned];
            Muls(preI, z4Local[statesOffsetAligned + hiddenDim], alphaT, hiddenDim);
            Add(preI, statesLocal[statesOffsetAligned + hiddenDim], preI, hiddenDim);

            // Output gate: pre_o = s2 + alpha * z2
            LocalTensor<T> preO = temp3[hiddenOffsetAligned];
            Muls(preO, z4Local[statesOffsetAligned + 2 * hiddenDim], alphaT, hiddenDim);
            Add(preO, statesLocal[statesOffsetAligned + 2 * hiddenDim], preO, hiddenDim);

            // Cell pre-activation: cpre = s3 + alpha * z3
            LocalTensor<T> cpre = temp4[hiddenOffsetAligned];
            Muls(cpre, z4Local[statesOffsetAligned + 3 * hiddenDim], alphaT, hiddenDim);
            Add(cpre, statesLocal[statesOffsetAligned + 3 * hiddenDim], cpre, hiddenDim);

            // Step 2: 计算门控值 (sigmoid)
            Sigmoid(preF, preF, hiddenDim);  // f
            Sigmoid(preI, preI, hiddenDim);  // i
            Sigmoid(preO, preO, hiddenDim);  // o

            // Step 3: 计算新的 Cell 状态
            // cpre_norm = RMSNorm(cpre, eps_cell)
            LocalTensor<T> cpreNorm = outCellLocal[hiddenOffsetAligned];
            ComputeRMSNorm(cpreNorm, cpre, hiddenDim, epsCell);

            // cpre_norm = cpre_norm * w_cell + b_cell (如果有)
            if (hasWCell) {
                LocalTensor<T> wCellLocal = optBuffer.Get<T>();
                uint32_t wBytes = hiddenDim * sizeof(T);
                uint8_t wRpad = (hiddenDimAligned - hiddenDim);
                AscendC::DataCopyExtParams cpParams = {1, wBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams = {false, 0, wRpad, (T)0};
                AscendC::DataCopyPad<T>(wCellLocal, gmWCell[0], cpParams, padParams);
                pipe_barrier(PIPE_ALL);
                Mul(cpreNorm, cpreNorm, wCellLocal, hiddenDim);
            }
            if (hasBCell) {
                LocalTensor<T> bCellLocal = optBuffer.Get<T>();
                uint32_t bBytes = hiddenDim * sizeof(T);
                uint8_t bRpad = (hiddenDimAligned - hiddenDim);
                AscendC::DataCopyExtParams cpParams = {1, bBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams = {false, 0, bRpad, (T)0};
                AscendC::DataCopyPad<T>(bCellLocal, gmBCell[0], cpParams, padParams);
                pipe_barrier(PIPE_ALL);
                Add(cpreNorm, cpreNorm, bCellLocal, hiddenDim);
            }

            // cact = GELU(cpre_norm)
            LocalTensor<T> cact = cpre;  // 复用 cpre buffer
            ComputeGelu(cact, cpreNorm, hiddenDim);

            // out_cell = prev_cell * f + cact * i
            LocalTensor<T> outCellSample = outCellLocal[hiddenOffsetAligned];
            Mul(outCellSample, prevCellLocal[hiddenOffsetAligned], preF, hiddenDim);
            Mul(cact, cact, preI, hiddenDim);
            Add(outCellSample, outCellSample, cact, hiddenDim);

            // Step 4: 计算输出隐藏状态
            // cnew_norm = RMSNorm(out_cell, eps_state)
            LocalTensor<T> cnewNorm = outStateLocal[hiddenOffsetAligned];
            ComputeRMSNorm(cnewNorm, outCellSample, hiddenDim, epsState);

            // cnew_norm = cnew_norm * w_state + b_state (如果有)
            if (hasWState) {
                LocalTensor<T> wStateLocal = optBuffer.Get<T>();
                uint32_t wBytes = hiddenDim * sizeof(T);
                uint8_t wRpad = (hiddenDimAligned - hiddenDim);
                AscendC::DataCopyExtParams cpParams = {1, wBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams = {false, 0, wRpad, (T)0};
                AscendC::DataCopyPad<T>(wStateLocal, gmWState[0], cpParams, padParams);
                pipe_barrier(PIPE_ALL);
                Mul(cnewNorm, cnewNorm, wStateLocal, hiddenDim);
            }
            if (hasBState) {
                LocalTensor<T> bStateLocal = optBuffer.Get<T>();
                uint32_t bBytes = hiddenDim * sizeof(T);
                uint8_t bRpad = (hiddenDimAligned - hiddenDim);
                AscendC::DataCopyExtParams cpParams = {1, bBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams = {false, 0, bRpad, (T)0};
                AscendC::DataCopyPad<T>(bStateLocal, gmBState[0], cpParams, padParams);
                pipe_barrier(PIPE_ALL);
                Add(cnewNorm, cnewNorm, bStateLocal, hiddenDim);
            }

            // sact = GELU(cnew_norm)
            LocalTensor<T> sact = cact;  // 复用 buffer
            ComputeGelu(sact, cnewNorm, hiddenDim);

            // out_state = sact * o
            LocalTensor<T> outStateSample = outStateLocal[hiddenOffsetAligned];
            Mul(outStateSample, sact, preO, hiddenDim);
        }

        // 释放输入 buffer
        inQueueStates.FreeTensor(statesLocal);
        inQueueZ4.FreeTensor(z4Local);
        inQueuePrevCell.FreeTensor(prevCellLocal);

        // 入队输出
        outQueueState.EnQue(outStateLocal);
        outQueueCell.EnQue(outCellLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t samples) {
        // 搬运 out_state (Local -> GM)
        LocalTensor<T> outStateLocal = outQueueState.DeQue<T>();
        uint32_t hiddenBytes = hiddenDim * sizeof(T);
        // srcStride: Local tensor 每行有 hiddenDimAligned 元素，需要跳过 padding
        uint32_t srcStrideBytes = (hiddenDimAligned - hiddenDim) * sizeof(T);
        AscendC::DataCopyExtParams copyParams = {(uint16_t)samples, hiddenBytes, srcStrideBytes, 0, 0};
        AscendC::DataCopyPad<T>(gmOutState[tileOffset * hiddenDim], outStateLocal, copyParams);
        outQueueState.FreeTensor(outStateLocal);

        // 搬运 out_cell
        LocalTensor<T> outCellLocal = outQueueCell.DeQue<T>();
        AscendC::DataCopyPad<T>(gmOutCell[tileOffset * hiddenDim], outCellLocal, copyParams);
        outQueueCell.FreeTensor(outCellLocal);
    }

    // RMSNorm 实现: x / sqrt(mean(x²) + ε) - 向量归约优化版
    __aicore__ inline void ComputeRMSNorm(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t len, float eps) {
        LocalTensor<float> workBuf = reduceWorkBuf.Get<float>();

        // Step 1: 计算 x²
        Mul(dst, src, src, len);

        // Step 2: 类型转换 half -> float (提高归约精度)
        Cast(workBuf, dst, RoundMode::CAST_NONE, len);

        // Step 3: 向量归约求和
        // ReduceSum(dst, src, workLocal, reduceLen) - 结果存入 dst[0]
        ReduceSum(workBuf, workBuf, workBuf, len);

        pipe_barrier(PIPE_ALL);
        float sum = workBuf.GetValue(0);

        // Step 4: 计算 rsqrt
        float mean = sum / static_cast<float>(static_cast<int32_t>(len));
        float rsqrtVal = 1.0f / sqrt(mean + eps);
        T rsqrtT = static_cast<T>(rsqrtVal);

        // Step 5: dst = src * rsqrt
        Muls(dst, src, rsqrtT, len);
    }

    // GELU 实现
    __aicore__ inline void ComputeGelu(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t len) {
        if (useFastGelu) {
            // Fast GELU (tanh 近似): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            // temp = x³
            Mul(dst, src, src, len);
            Mul(dst, dst, src, len);

            // temp = 0.044715 * x³
            Muls(dst, dst, static_cast<T>(GELU_COEF), len);

            // temp = x + 0.044715 * x³
            Add(dst, src, dst, len);

            // temp = sqrt(2/π) * (x + 0.044715 * x³)
            Muls(dst, dst, static_cast<T>(SQRT_2_PI), len);

            // temp = tanh(...)
            Tanh(dst, dst, len);

            // temp = 1 + tanh(...)
            Adds(dst, dst, static_cast<T>(1.0f), len);

            // temp = x * (1 + tanh(...))
            Mul(dst, src, dst, len);

            // dst = 0.5 * x * (1 + tanh(...))
            Muls(dst, dst, static_cast<T>(0.5f), len);
        } else {
            // Precise GELU (erf): 0.5 * x * (1 + erf(x / sqrt(2)))
            // temp = x / sqrt(2)
            Muls(dst, src, static_cast<T>(SQRT_2_INV), len);

            // temp = erf(x / sqrt(2))
            Erf(dst, dst, len);

            // temp = 1 + erf(...)
            Adds(dst, dst, static_cast<T>(1.0f), len);

            // temp = x * (1 + erf(...))
            Mul(dst, src, dst, len);

            // dst = 0.5 * x * (1 + erf(...))
            Muls(dst, dst, static_cast<T>(0.5f), len);
        }
    }

private:
    // Tiling 参数
    uint32_t totalSamples;
    uint32_t hiddenDim;
    uint32_t gatedDim;
    uint32_t coreNum;
    uint32_t samplesPerCore;
    uint32_t remainSamples;
    uint32_t tileNumPerCore;
    uint32_t tileSamples;
    uint32_t lastTileSamples;
    uint32_t hiddenDimAligned;
    uint32_t gatedDimAligned;
    float alpha;
    float epsCell;
    float epsState;
    uint32_t useFastGelu;
    uint32_t hasWCell;
    uint32_t hasBCell;
    uint32_t hasWState;
    uint32_t hasBState;

    // 运行时参数
    uint32_t coreSamples;
    uint32_t actualTileNum;
    uint32_t currentTileSize;
    uint32_t tileOffset;

    // Pipe
    TPipe pipe;

    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueStates;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ4;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePrevCell;

    // 输出队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueState;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCell;

    // 临时 buffer
    TBuf<QuePosition::VECCALC> tempBuffer1;
    TBuf<QuePosition::VECCALC> tempBuffer2;
    TBuf<QuePosition::VECCALC> tempBuffer3;
    TBuf<QuePosition::VECCALC> tempBuffer4;
    TBuf<QuePosition::VECCALC> optBuffer;
    TBuf<QuePosition::VECCALC> reduceWorkBuf;  // RMSNorm 归约工作 buffer

    // Global Memory
    GlobalTensor<T> gmStates4d;
    GlobalTensor<T> gmZ4_4d;
    GlobalTensor<T> gmPrevCell;
    GlobalTensor<T> gmWCell;
    GlobalTensor<T> gmBCell;
    GlobalTensor<T> gmWState;
    GlobalTensor<T> gmBState;
    GlobalTensor<T> gmOutState;
    GlobalTensor<T> gmOutCell;
};

extern "C" __global__ __aicore__ void sum_lstm(GM_ADDR states4d, GM_ADDR z4_4d, GM_ADDR prevCell,
                                                GM_ADDR wCell, GM_ADDR bCell, GM_ADDR wState, GM_ADDR bState,
                                                GM_ADDR outState, GM_ADDR outCell,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    // 获取数据类型大小来判断使用哪个模板
    uint32_t dataTypeSize = tilingData.dataTypeSize;

    if (TILING_KEY_IS(0)) {
        // 标准处理路径
        if (dataTypeSize == 2) {
            // float16 或 bfloat16
            SumLstmKernel<half> op;
            op.Init(states4d, z4_4d, prevCell, wCell, bCell, wState, bState,
                    outState, outCell, workspace, &tilingData);
            op.Process();
        }
    }
}
