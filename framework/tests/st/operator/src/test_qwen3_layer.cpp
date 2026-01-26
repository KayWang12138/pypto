#include <gtest/gtest.h>
#include <cmath>
#include "inner/config.h"
#include "tilefwk/config.h"
#include "tilefwk/data_type.h"
#include "interface/program/program.h"
#include "machine/cache_manager/cache_manager.h"
#include "test_suite_stest_ops.h"
#include "operator/models/qwen3/qwen3layer.h"
#include "test_dev_func_runner.h"
#include "test_data_loader.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestQwen3Atten : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_4 = 4;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;

namespace {
template <typename T>
static void DumpRow(const char *tag, const uint8_t *bufBytes, int row, int cols, int maxCols = 8) {
    auto *buf = reinterpret_cast<const T *>(bufBytes);
    std::cout << tag << " row=" << row << " [";
    int show = cols < maxCols ? cols : maxCols;
    for (int c = 0; c < show; ++c) {
        float v = static_cast<float>(buf[row * cols + c]);
        std::cout << v;
        if (c + 1 < show) {
            std::cout << ", ";
        }
    }
    if (cols > show) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}
} // namespace

/************************************* Qwen3Attention *************************************/

template <typename T>
void Qwen3AttentionCompute(
    TestDataLoader &data, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &dims) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int n = std::get<int>(data.Param("n"));
    int n_kv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    npu::tile_fwk::Qwen3AttentionCompute<T>(data.InputTensorCheck("hidden_states", dType, {b * s, n * d}),
        data.InputTensorCheck("attn_q_w", dType, {n * d, n * d}), data.InputTensorCheck("attn_q_b", dType, {n * d}),
        data.InputTensorCheck("attn_k_w", dType, {n * d, n_kv * d}),
        data.InputTensorCheck("attn_k_b", dType, {n_kv * d}),
        data.InputTensorCheck("attn_v_w", dType, {n * d, n_kv * d}),
        data.InputTensorCheck("attn_v_b", dType, {n_kv * d}), data.InputTensorCheck("attn_o_w", dType, {n * d, n * d}),
        data.InputTensorCheck("attn_o_b", dType, {n * d}), data.InputTensorCheck("attn_q_norm_w", dType, {d}),
        data.InputTensorCheck("attn_k_norm_w", dType, {d}), data.OutputTensor("attention_out"), tileConfig, dims);
}

template <typename T>
void qwen3Atten(TestDataLoader &data, Qwen3AttenTileShapeConfig &tileConfig) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int n = std::get<int>(data.Param("n"));
    int d = std::get<int>(data.Param("d"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    // DataType dType = CostModel::ToDataType(dTypeStr);
    Qwen3AttentionDims dimsCfg = {b, n, std::get<int>(data.Param("n_kv")), s, d, d, d};
    data.Dump();

    FUNCTION("Qwen3Attention", data.GetInputTensorList(), data.GetOutputTensorList()) {
        Qwen3AttentionCompute<T>(data, tileConfig, dimsCfg);
        std::cout << "[Qwen3Atten] Qwen3AttentionCompute done" << std::endl;
    }
    std::cout << "[Qwen3Atten] FUNCTION done" << std::endl;

    // auto goldenData = data.GoldenDataCheck("attention_out", dType, {b * s, n * d});
    auto goldenData = data.GoldenData("attention_out");
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("attention_out")];
    std::cout << "[Qwen3Atten] goldenData done" << std::endl;
    std::cout << "[Qwen3Atten] outputData done" << std::endl;
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    std::cout << "[Qwen3Atten] DevFuncRunner::Run done" << std::endl;
    EXPECT_TRUE(resultCmp<T>((T *)goldenData->data(), (T *)outputData->data(), goldenData->GetSize(), 0.001f));
    std::cout << "[Qwen3Atten] resultCmp done" << std::endl;
#endif
}

/************************************* Qwen3PagedAttentionProlog *************************************/
template <typename T>
void Qwen3PagedAttentionPrologCompute(TestDataLoader &data) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int nQ = std::get<int>(data.Param("n"));
    int nKv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int blockSize = std::get<int>(data.Param("block_size"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    auto hiddenSize = nQ * d;
    auto kvHidden = nKv * d;

    auto &hiddenStates = data.InputTensorCheck("hidden_states", dType, {b * s, hiddenSize});
    auto &qW = data.InputTensorCheck("attn_q_w", dType, {hiddenSize, hiddenSize});
    auto &qB = data.InputTensorCheck("attn_q_b", dType, {hiddenSize});
    auto &kW = data.InputTensorCheck("attn_k_w", dType, {hiddenSize, kvHidden});
    auto &kB = data.InputTensorCheck("attn_k_b", dType, {kvHidden});
    auto &vW = data.InputTensorCheck("attn_v_w", dType, {hiddenSize, kvHidden});
    auto &vB = data.InputTensorCheck("attn_v_b", dType, {kvHidden});
    auto &qNormW = data.InputTensorCheck("attn_q_norm_w", dType, {d});
    auto &kNormW = data.InputTensorCheck("attn_k_norm_w", dType, {d});

    auto &cos = data.InputTensorCheck("cos", dType, {b, s, d});
    auto &sin = data.InputTensorCheck("sin", dType, {b, s, d});
    auto &cacheIndex = data.InputTensorCheck("cache_index", DT_INT32, {b, s});

    auto &kCacheRaw = data.InputTensor("key_cache");
    auto kRows = kCacheRaw.GetShape()[0];
    auto kFmt = kCacheRaw.GetStorage()->Format();
    auto &keyCache = data.InputTensorCheck("key_cache", dType, {kRows, kvHidden}, kFmt);

    auto &vCacheRaw = data.InputTensor("value_cache");
    auto vRows = vCacheRaw.GetShape()[0];
    auto vFmt = vCacheRaw.GetStorage()->Format();
    auto &valueCache = data.InputTensorCheck("value_cache", dType, {vRows, kvHidden}, vFmt);

    auto &queryOut = data.OutputTensorCheck("query_out", dType, {b * s * nQ, d});
    auto &keyCacheOut = data.OutputTensorCheck("key_cache_out", dType, {kRows, kvHidden}, kFmt);
    auto &valueCacheOut = data.OutputTensorCheck("value_cache_out", dType, {vRows, kvHidden}, vFmt);

    npu::tile_fwk::Qwen3PagedAttentionPrologCompute<T>(hiddenStates, qW, qB, kW, kB, vW, vB, qNormW, kNormW, cos, sin,
        cacheIndex, keyCache, valueCache, nQ, nKv, blockSize, queryOut, keyCacheOut, valueCacheOut);
}

template <typename T>
void qwen3PagedAttnProlog(TestDataLoader &data) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    // config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    // config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    // // 可选优化：打开二进制缓存（~/.ast_data/<soc>/...），第二次跑同一个 case 会快很多。
    // config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, true);
    // (void)CacheManager::Instance().Initialize();

    // Build tensors explicitly here so we can declare inplace outputs (cache update).
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int nQ = std::get<int>(data.Param("n"));
    int nKv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int blockSize = std::get<int>(data.Param("block_size"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    auto hiddenSize = nQ * d;
    auto kvHidden = nKv * d;

    auto &hiddenStates = data.InputTensorCheck("hidden_states", dType, {b * s, hiddenSize});
    auto &qW = data.InputTensorCheck("attn_q_w", dType, {hiddenSize, hiddenSize});
    auto &qB = data.InputTensorCheck("attn_q_b", dType, {hiddenSize});
    auto &kW = data.InputTensorCheck("attn_k_w", dType, {hiddenSize, kvHidden});
    auto &kB = data.InputTensorCheck("attn_k_b", dType, {kvHidden});
    auto &vW = data.InputTensorCheck("attn_v_w", dType, {hiddenSize, kvHidden});
    auto &vB = data.InputTensorCheck("attn_v_b", dType, {kvHidden});
    auto &qNormW = data.InputTensorCheck("attn_q_norm_w", dType, {d});
    auto &kNormW = data.InputTensorCheck("attn_k_norm_w", dType, {d});

    auto &cos = data.InputTensorCheck("cos", dType, {b, s, d});
    auto &sin = data.InputTensorCheck("sin", dType, {b, s, d});
    auto &cacheIndex = data.InputTensorCheck("cache_index", DT_INT32, {b, s});

    auto &kCacheRaw = data.InputTensor("key_cache");
    auto kRows = kCacheRaw.GetShape()[0];
    auto kFmt = kCacheRaw.GetStorage()->Format();
    auto &keyCache = data.InputTensorCheck("key_cache", dType, {kRows, kvHidden}, kFmt);

    auto &vCacheRaw = data.InputTensor("value_cache");
    auto vRows = vCacheRaw.GetShape()[0];
    auto vFmt = vCacheRaw.GetStorage()->Format();
    auto &valueCache = data.InputTensorCheck("value_cache", dType, {vRows, kvHidden}, vFmt);

    // DEBUG
    auto &queryOut = data.OutputTensorCheck("query_out", dType, {b * s * nQ, d});
    // auto &queryOut = data.OutputTensorCheck("query_out", dType, {b * s * nKv, d});
    // auto &queryOut = data.OutputTensorCheck("query_out", DT_INT32, {b, s});
    // key/value cache are inplace outputs: out shares memory with input cache.
    auto &keyCacheOut = data.OutputTensorCheck("key_cache_out", dType, {kRows, kvHidden}, kFmt);
    auto &valueCacheOut = data.OutputTensorCheck("value_cache_out", dType, {vRows, kvHidden}, vFmt);

    data.Dump();

    FUNCTION("Qwen3PagedAttentionProlog", data.GetInputTensorList(),
        {
            queryOut
    },
        {{keyCacheOut, keyCache}, {valueCacheOut, valueCache}}) {
        npu::tile_fwk::Qwen3PagedAttentionPrologCompute<T>(hiddenStates, qW, qB, kW, kB, vW, vB, qNormW, kNormW, cos,
            sin, cacheIndex, keyCache, valueCache, nQ, nKv, blockSize, queryOut, keyCacheOut, valueCacheOut);
        std::cout << "[Qwen3PagedAttentionProlog] Qwen3PagedAttentionPrologCompute done" << std::endl;
    }
    std::cout << "[Qwen3PagedAttentionProlog] FUNCTION done" << std::endl;

    auto goldenQuery = data.GoldenData("query_out");
    auto goldenK = data.GoldenData("key_cache_out");
    auto goldenV = data.GoldenData("value_cache_out");
    auto outQuery = data.GetOutputDataList()[data.GetOutputNameToIdx("query_out")];
    // inplace outputs: compare golden cache_out with updated input cache buffers
    auto outKInplace = data.GetInputDataList()[data.GetInputNameToIdx("key_cache")];
    auto outVInplace = data.GetInputDataList()[data.GetInputNameToIdx("value_cache")];
    // Save pre-run caches for debug
    std::vector<uint8_t> keyCacheBefore(outKInplace->begin(), outKInplace->end());
    std::vector<uint8_t> valueCacheBefore(outVInplace->begin(), outVInplace->end());

#ifdef BUILD_WITH_CANN
    auto *func = Program::GetInstance().GetLastFunction();
    std::cout << "[Qwen3PagedAttentionProlog] dyndevAttr=" << (func ? func->GetDyndevAttribute() : nullptr)
              << std::endl;
    if (func && func->GetDyndevAttribute()) {
        std::cout << "[Qwen3PagedAttentionProlog] devProgBinary.size="
                  << func->GetDyndevAttribute()->devProgBinary.size() << std::endl;
    }
    std::cout << "[Qwen3PagedAttentionProlog] DevFuncRunner::Run begin" << std::endl;
    // Only query_out is a real output buffer. Cache outputs are inplace (written back to input caches).
    std::vector<RawTensorDataPtr> outList = {outQuery};
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), outList);
    /**************************** */
    // 默认 DevFuncRunner 会同时跑 RunModel(software test-mode) + RunOnBoard；前者会启动 aicpu 线程模拟执行，
    // 对于大 GEMM 会非常慢。此处仅做 correctness：直接跳过 RunModel，只跑 on-board kernel launch。
    // DeviceLauncherConfig runCfg;
    // runCfg.onBoard = true;
    // runCfg.runModel = false;
    // DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), outList, runCfg);
    /**************************** */
    std::cout << "[Qwen3PagedAttentionProlog] DevFuncRunner::Run done" << std::endl;
    std::cout << "[Qwen3PagedAttentionProlog] resultCmp begin" << std::endl;
    EXPECT_TRUE(resultCmp<T>((T *)goldenQuery->data(), (T *)outQuery->data(), goldenQuery->GetSize(), 0.001f));
    // EXPECT_TRUE(resultCmp<int32_t>((int32_t *)goldenQuery->data(), (int32_t *)outQuery->data(),
    // goldenQuery->GetSize(), 0.0f));
    EXPECT_TRUE(resultCmp<T>((T *)goldenK->data(), (T *)outKInplace->data(), goldenK->GetSize(), 0.001f));
    bool vOk = resultCmp<T>((T *)goldenV->data(), (T *)outVInplace->data(), goldenV->GetSize(), 0.001f);
    if (!vOk) {
        // 只打印被更新的两行：row = cache_index[b,0]，看它是“没更新(=before)”还是“写错(≈goldenK)”
        auto cacheIdx = data.GetInputDataList()[data.GetInputNameToIdx("cache_index")];
        auto *idxPtr = reinterpret_cast<int32_t *>(cacheIdx->data());
        int row0 = idxPtr[0];
        int row1 = idxPtr[1];
        int cols = kvHidden; // nKv*d
        std::cout << "[DEBUG] cache_index rows = [" << row0 << ", " << row1 << "], cols=" << cols << std::endl;

        DumpRow<T>("[DEBUG] valueCacheBefore", valueCacheBefore.data(), row0, cols);
        DumpRow<T>("[DEBUG] valueCacheAfter ", outVInplace->data(), row0, cols);
        DumpRow<T>("[DEBUG] goldenV        ", goldenV->data(), row0, cols);
        DumpRow<T>("[DEBUG] goldenK        ", goldenK->data(), row0, cols);

        DumpRow<T>("[DEBUG] valueCacheBefore", valueCacheBefore.data(), row1, cols);
        DumpRow<T>("[DEBUG] valueCacheAfter ", outVInplace->data(), row1, cols);
        DumpRow<T>("[DEBUG] goldenV        ", goldenV->data(), row1, cols);
        DumpRow<T>("[DEBUG] goldenK        ", goldenK->data(), row1, cols);
    }
    EXPECT_TRUE(vOk);
    std::cout << "[Qwen3PagedAttentionProlog] resultCmp done" << std::endl;
#endif
}

/************************************* Qwen3PagedAttention *************************************/
template <typename T>
void Qwen3PagedAttentionCompute(
    TestDataLoader &data, PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int nQ = std::get<int>(data.Param("n"));
    int nKv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int blockSize = std::get<int>(data.Param("block_size"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    // 输入输出都是2D
    auto &query = data.InputTensorCheck("query", dType, {b * s * nQ, d});

    auto &kCacheRaw = data.InputTensor("key_cache");
    auto kRows = kCacheRaw.GetShape()[0];
    auto kFmt = kCacheRaw.GetStorage()->Format();
    auto &keyCache = data.InputTensorCheck("key_cache", dType, {kRows, nKv * d}, kFmt);

    auto &vCacheRaw = data.InputTensor("value_cache");
    auto vRows = vCacheRaw.GetShape()[0];
    auto vFmt = vCacheRaw.GetStorage()->Format();
    auto &valueCache = data.InputTensorCheck("value_cache", dType, {vRows, nKv * d}, vFmt);

    auto &blkRaw = data.InputTensor("block_table");
    auto blkCols = blkRaw.GetShape()[1];
    auto &blockTable = data.InputTensorCheck("block_table", DT_INT32, {b, blkCols}, blkRaw.GetStorage()->Format());

    auto &actSeqs = data.InputTensorCheck("act_seqs", DT_INT32, {b});

    auto &out = data.OutputTensorCheck("paged_attention_out", DT_FP32, {b * s * nQ, d});

    float softmaxScale = static_cast<float>(1.0 / sqrtf(static_cast<float>(d)));
    npu::tile_fwk::Qwen3PagedAttentionCompute<T>(query, keyCache, valueCache, blockTable, actSeqs, nQ, nKv, blockSize,
        softmaxScale, out, tileConfig, maxUnrollTimes, isNzFormat);
}

template <typename T>
void qwen3PagedAttn(
    TestDataLoader &data, PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    data.Dump();

    FUNCTION("Qwen3PagedAttention", data.GetInputTensorList(), data.GetOutputTensorList()) {
        Qwen3PagedAttentionCompute<T>(data, tileConfig, maxUnrollTimes, isNzFormat);
        std::cout << "[Qwen3PagedAttention] Qwen3PagedAttentionCompute done" << std::endl;
    }
    std::cout << "[Qwen3PagedAttention] FUNCTION done" << std::endl;

    auto goldenData = data.GoldenData("paged_attention_out");
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("paged_attention_out")];
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    EXPECT_TRUE(
        resultCmp<float>((float *)goldenData->data(), (float *)outputData->data(), goldenData->GetSize(), 0.001f));
#endif
}

template <typename T>
void Qwen3PagedAttentionWindowCompute(
    TestDataLoader &data, PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int nQ = std::get<int>(data.Param("n"));
    int nKv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int blockSize = std::get<int>(data.Param("block_size"));
    int windowSize = std::get<int>(data.Param("window_size"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    auto &query = data.InputTensorCheck("query", dType, {b * s * nQ, d});

    auto &kCacheRaw = data.InputTensor("key_cache");
    auto kRows = kCacheRaw.GetShape()[0];
    auto kFmt = kCacheRaw.GetStorage()->Format();
    auto &keyCache = data.InputTensorCheck("key_cache", dType, {kRows, nKv * d}, kFmt);

    auto &vCacheRaw = data.InputTensor("value_cache");
    auto vRows = vCacheRaw.GetShape()[0];
    auto vFmt = vCacheRaw.GetStorage()->Format();
    auto &valueCache = data.InputTensorCheck("value_cache", dType, {vRows, nKv * d}, vFmt);

    auto &blkRaw = data.InputTensor("block_table");
    auto blkCols = blkRaw.GetShape()[1];
    auto &blockTable = data.InputTensorCheck("block_table", DT_INT32, {b, blkCols}, blkRaw.GetStorage()->Format());
    auto &actSeqs = data.InputTensorCheck("act_seqs", DT_INT32, {b});

    auto &out = data.OutputTensorCheck("paged_attention_out", DT_FP32, {b * s * nQ, d});

    float softmaxScale = static_cast<float>(1.0 / sqrtf(static_cast<float>(d)));
    npu::tile_fwk::Qwen3PagedAttentionCompute<T>(query, keyCache, valueCache, blockTable, actSeqs, nQ, nKv, blockSize,
        softmaxScale, out, tileConfig, maxUnrollTimes, isNzFormat, windowSize);
}

template <typename T>
void qwen3PagedAttnWindow(
    TestDataLoader &data, PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    data.Dump();

    FUNCTION("Qwen3PagedAttentionWindow", data.GetInputTensorList(), data.GetOutputTensorList()) {
        Qwen3PagedAttentionWindowCompute<T>(data, tileConfig, maxUnrollTimes, isNzFormat);
        std::cout << "[Qwen3PagedAttentionWindow] Qwen3PagedAttentionWindowCompute done" << std::endl;
    }
    std::cout << "[Qwen3PagedAttentionWindow] FUNCTION done" << std::endl;

    auto goldenData = data.GoldenData("paged_attention_out");
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("paged_attention_out")];
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    EXPECT_TRUE(
        resultCmp<float>((float *)goldenData->data(), (float *)outputData->data(), goldenData->GetSize(), 0.001f));
#endif
}

/************************************* Qwen3MLP *************************************/
template <typename T>
void Qwen3MLPCompute(TestDataLoader &data, const Qwen3MlpDims &dims) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int h = std::get<int>(data.Param("h"));
    int inter = std::get<int>(data.Param("inter"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    npu::tile_fwk::Qwen3MLPCompute<T>(data.InputTensorCheck("hidden_states", dType, {b * s, h}),
        data.InputTensorCheck("gate_w", dType, {h, inter}), data.InputTensorCheck("up_w", dType, {h, inter}),
        data.InputTensorCheck("down_w", dType, {inter, h}), data.OutputTensor("mlp_out"), dims);
}

template <typename T>
void qwen3Mlp(TestDataLoader &data) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int h = std::get<int>(data.Param("h"));
    int inter = std::get<int>(data.Param("inter"));
    Qwen3MlpDims dimsCfg = {b, s, h, inter};

    data.Dump();

    FUNCTION("Qwen3MLP", data.GetInputTensorList(), data.GetOutputTensorList()) {
        Qwen3MLPCompute<T>(data, dimsCfg);
        std::cout << "[Qwen3MLP] Qwen3MLPCompute done" << std::endl;
    }
    std::cout << "[Qwen3MLP] FUNCTION done" << std::endl;

    auto goldenData = data.GoldenData("mlp_out");
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("mlp_out")];
    std::cout << "[Qwen3MLP] goldenData done" << std::endl;
    std::cout << "[Qwen3MLP] outputData done" << std::endl;
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    std::cout << "[Qwen3MLP] DevFuncRunner::Run done" << std::endl;
    EXPECT_TRUE(resultCmp<T>((T *)goldenData->data(), (T *)outputData->data(), goldenData->GetSize(), 0.001f));
    std::cout << "[Qwen3MLP] resultCmp done" << std::endl;
#endif
}

/************************************* Qwen3Layer *************************************/
template <typename T>
void Qwen3LayerCompute(TestDataLoader &data, Qwen3AttenTileShapeConfig &tileConfig, const Qwen3AttentionDims &attnDims,
    const Qwen3MlpDims &mlpDims) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int h = std::get<int>(data.Param("h"));
    int n = std::get<int>(data.Param("n"));
    int n_kv = std::get<int>(data.Param("n_kv"));
    // Layer 用例的 golden 配置里没有显式参数 d，按定义 d = h / n 推导
    ASSERT_GT(n, 0);
    ASSERT_EQ(h % n, 0);
    int d = h / n;
    int inter = std::get<int>(data.Param("inter"));
    (void)b;
    (void)s;
    (void)h;
    (void)n;
    (void)n_kv;
    (void)d;
    (void)inter; // params already in dims
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    npu::tile_fwk::Qwen3LayerCompute<T>(data.InputTensorCheck("hidden_states", dType, {b * s, h}),
        data.InputTensorCheck("attn_q_w", dType, {h, h}), data.InputTensorCheck("attn_q_b", dType, {h}),
        data.InputTensorCheck("attn_k_w", dType, {h, n_kv * d}), data.InputTensorCheck("attn_k_b", dType, {n_kv * d}),
        data.InputTensorCheck("attn_v_w", dType, {h, n_kv * d}), data.InputTensorCheck("attn_v_b", dType, {n_kv * d}),
        data.InputTensorCheck("attn_o_w", dType, {h, h}), data.InputTensorCheck("attn_o_b", dType, {h}),
        data.InputTensorCheck("attn_q_norm_w", dType, {d}), data.InputTensorCheck("attn_k_norm_w", dType, {d}),
        data.InputTensorCheck("gate_w", dType, {h, inter}), data.InputTensorCheck("up_w", dType, {h, inter}),
        data.InputTensorCheck("down_w", dType, {inter, h}), data.OutputTensor("layer_out"), tileConfig, attnDims,
        mlpDims);
}

template <typename T>
void qwen3Layer(TestDataLoader &data, Qwen3AttenTileShapeConfig &tileConfig) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int h = std::get<int>(data.Param("h"));
    int n = std::get<int>(data.Param("n"));
    int n_kv = std::get<int>(data.Param("n_kv"));
    // Layer 用例的 golden 配置里没有显式参数 d，按定义 d = h / n 推导
    ASSERT_GT(n, 0);
    ASSERT_EQ(h % n, 0);
    int d = h / n;
    int inter = std::get<int>(data.Param("inter"));
    std::string dtypeStr = std::get<string>(data.Param("dtype"));
    // DataType dType = CostModel::ToDataType(dtypeStr);
    Qwen3AttentionDims attnDims = {b, n, n_kv, s, d, d, d};
    Qwen3MlpDims mlpDims = {b, s, h, inter};

    data.Dump();

    FUNCTION("Qwen3Layer", data.GetInputTensorList(), data.GetOutputTensorList()) {
        Qwen3LayerCompute<T>(data, tileConfig, attnDims, mlpDims);
        std::cout << "[Qwen3Layer] Qwen3LayerCompute done" << std::endl;
    }

    auto goldenData = data.GoldenData("layer_out");
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("layer_out")];
    std::cout << "[Qwen3Layer] goldenData done" << std::endl;
    std::cout << "[Qwen3Layer] outputData done" << std::endl;
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    std::cout << "[Qwen3Layer] DevFuncRunner::Run done" << std::endl;
    EXPECT_TRUE(resultCmp<T>((T *)goldenData->data(), (T *)outputData->data(), goldenData->GetSize(), 0.001f));
    std::cout << "[Qwen3Layer] resultCmp done" << std::endl;
#endif
}

TEST_F(TestQwen3Atten, TestQwen3Atten_B_1_S_16_N_2_D_16_BF16) {
    Qwen3AttenTileShapeConfig tileConfig;
    tileConfig.vecTileShape = {NUM_1, NUM_16, NUM_2, NUM_16};
    tileConfig.scale = 0.0f;

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Atten<npu::tile_fwk::bfloat16>(data, tileConfig);
}

TEST_F(TestQwen3Atten, TestQwen3PagedAttentionProlog_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnProlog<npu::tile_fwk::bfloat16>(data);
}

// qwen3-8b aligned: nQ=32, nKv=8, d=128, blockSize=16 (prolog: QKV + RoPE + write KV cache)
// time: 979.98us  AI Core: 47.01%  Total Core:52   Total Task Count: 193
// NOTE: 虽然最后结果比较理想，但是运行这个test用时会比较长（500s），一直卡在FUNCTION解析这里（已通过View + LOOP +
// Assemble 模式解决）
TEST_F(TestQwen3Atten, TestQwen3PagedAttentionProlog_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnProlog<npu::tile_fwk::bfloat16>(data);
}

// NOTE: 运行了65分钟，一直卡在FUNCTION （已通过View + LOOP + Assemble 模式解决）
// time: 3348.74us  AI Core: 45.31%  Total Core:52   Total Task Count: 221
TEST_F(TestQwen3Atten, TestQwen3PagedAttentionProlog_B_8_S_1_N_32_KV_8_D_128_BLK_16_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnProlog<npu::tile_fwk::bfloat16>(data);
}

// Qwen3-8B aligned: B=32, nQ=32, nKv=8, d=128, blockSize=4096 (高吞吐配置)
// time: 214.94us  AI Core: 43.41%  Total Core:52   Total Task Count: 561
TEST_F(TestQwen3Atten, TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_4096_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnProlog<npu::tile_fwk::bfloat16>(data);
}

// Qwen3-8B aligned: B=32, nQ=32, nKv=8, d=128, blockSize=2048
TEST_F(TestQwen3Atten, TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_2048_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnProlog<npu::tile_fwk::bfloat16>(data);
}

TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_2; // group = nQ/nKv = 2，tile内不跨kv组
    const int blockSize = NUM_16;
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, blockSize};
    tileConfig.c1TileShape = {gTile, gTile, blockSize, blockSize, blockSize, blockSize};
    tileConfig.v1TileShape = {gTile, blockSize};
    tileConfig.c2TileShape = {gTile, gTile, blockSize, blockSize, blockSize, blockSize};
    tileConfig.v2TileShape = {gTile, blockSize};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig);
}

// Qwen3-8B aligned: nQ=32, nKv=8, d=128, blockSize=16
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4; // group = nQ/nKv = 4, choose gTile=4 to reduce g-loop overhead
    const int blockSize = NUM_16;
    const int dTile = 64;         // d=128, use 64 for k/n/vec tiling
    const int maxUnrollTimes = 1; // unroll bn-loop (powers-of-2 hint) to reduce overhead
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};
    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v1TileShape = {gTile, blockSize};
    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, blockSize, blockSize, dTile, dTile};
    tileConfig.v2TileShape = {gTile, dTile};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// Time: 45.94us  AI Core: 27.72%  Total Core: 24  Total Task Count: 92
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_4_S_1_N_32_KV_1_D_128_BLK_128_SKV_256_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = 32;             // group = nQ/nKv = 32, tile=32
    const int blockSize = 128;        // blockSize=128
    const int dTile = 64;             // d=128, tile by 64
    const int maxUnrollTimes = NUM_2; // tableLoop=skv/blockSize=2, unroll up to 2
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};
    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v1TileShape = {gTile, dTile};
    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v2TileShape = {gTile, dTile};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// time: 240.22us  AI Core: 15.31%  Total Core:38  Total Task Count:800
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_4_S_1_N_32_KV_8_D_128_BLK_128_SKV_256_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4;          // group = nQ/nKv = 4, tile=4
    const int blockSize = 128;        // blockSize=128
    const int dTile = 64;             // d=128, tile by 64
    const int maxUnrollTimes = NUM_2; // tableLoop=skv/blockSize=2, unroll up to 2
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v1TileShape = {gTile, dTile};
    tileConfig.c2TileShape = {gTile, gTile, blockSize, blockSize, dTile, dTile};
    tileConfig.v2TileShape = {gTile, dTile};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// Large batch + long context, MQA (nKv=1)
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_512_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = 32;             // group = nQ/nKv = 32, tile=32
    const int s2Tile = 256;           // tile along s2(validS2<=512) to avoid over-large cube tile
    const int dTile = 64;             // d=128, tile by 64
    const int maxUnrollTimes = NUM_4; // bn=skv/blockSize=4096/512=8
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};
    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {gTile, dTile};
    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {gTile, dTile};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// High throughput config aligned with DynamicPA: B=32, N=32, blockSize=4096, skv=4096 (single block)
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_4096_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = 32;        // group = nQ/nKv = 32, tile=32
    const int dTile = 64;        // d=128, tile by 64
    const int s2Tile = 128;      // tile along s2(validS2<=4096) for matmul/softmax
    const int maxUnrollTimes = 1; // nKv=1, unroll not required

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile}; // unused in Qwen3PagedAttentionCompute

    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, s2Tile, s2Tile};
    tileConfig.v1TileShape = {16, 256};

    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 128};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// High throughput config aligned with DynamicPA: B=32, N=128, blockSize=4096, skv=4096 (single block)
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_128_BLK_4096_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = 128;        // group = nQ/nKv = 128, tile=128
    const int dTile = 64;         // d=128, tile by 64
    const int s2Tile = 128;       // tile along s2(validS2<=4096) for matmul/softmax
    const int maxUnrollTimes = 4; // bn=skv/blockSize=4096/4096=1

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile}; // unused in Qwen3PagedAttentionCompute

    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, s2Tile, s2Tile};
    tileConfig.v1TileShape = {16, 256};

    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 128};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// High throughput config aligned with DynamicPA (dn=512): B=32, N=128, blockSize=4096, skv=4096 (single block)
// 与PagedAttention的测试dynamic_pa_high_throughput_dview_large对齐。
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_512_BLK_4096_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = 128;        // group = nQ/nKv = 128, tile=128
    const int dTile = 64;         // d=512, tile by 64
    const int s2Tile = 128;       // tile along s2(validS2<=4096) for matmul/softmax
    const int maxUnrollTimes = 1; // bn=skv/blockSize=4096/4096=1

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile}; // unused in Qwen3PagedAttentionCompute

    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, s2Tile, s2Tile};
    tileConfig.v1TileShape = {16, 256};

    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256}; // dim=512，这里设置为256，分为两个tile

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// NOTE: real qwen3-8B
// qwen3-8b GQA config with huge blockSize (single block): B=32, nQ=32, nKv=8, d=128, blockSize=4096, skv=4096
// 优化前: Time: 3375.64us  AI Core: 28.82%  Total Core: 47  Total Task Count: 10240
// 优化: 增大 s2Tile (256->512), maxUnrollTimes (1->4), v2TileShape (128->256)
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_4096_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4;      // group = nQ/nKv = 4, tile=4 (max, cannot increase due to group constraint)
    const int dTile = 64;         // d=128, tile by 64
    const int s2Tile = 512;       // tile along s2(validS2<=4096) for softmax, increased from 256
    const int maxUnrollTimes = 4; // enable loop unroll optimization

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile}; // unused in Qwen3PagedAttentionCompute

    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, 128, 128};
    tileConfig.v1TileShape = {gTile, s2Tile};

    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {gTile, 256}; // increased from 128 for better vector parallelism

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// NOTE: real qwen3-8B
// qwen3-8b GQA config with huge blockSize (single block): B=32, nQ=32, nKv=8, d=128, blockSize=2048, skv=2048
TEST_F(TestQwen3Atten, TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_2048_SKV_2048_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4;      // group = nQ/nKv = 4, tile=4 (max, cannot increase due to group constraint)
    const int dTile = 64;         // d=128, tile by 64
    const int s2Tile = 512;       // tile along s2(validS2<=2048) for softmax
    const int maxUnrollTimes = 4; // enable loop unroll optimization

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile}; // unused in Qwen3PagedAttentionCompute

    // QK: [gTile, d] x [s2, d]^T => [gTile, s2]
    tileConfig.c1TileShape = {gTile, gTile, dTile, dTile, 128, 128};
    tileConfig.v1TileShape = {gTile, s2Tile};

    // PV: [gTile, s2] x [s2, d] => [gTile, d]
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {gTile, 256};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttn<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

TEST_F(TestQwen3Atten, TestQwen3PagedAttentionWindow_B_2_S_1_N_4_KV_2_D_16_BLK_16_WIN_16_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_2; // group = nQ/nKv = 2
    const int blockSize = NUM_16;
    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, blockSize};
    tileConfig.c1TileShape = {gTile, gTile, blockSize, blockSize, blockSize, blockSize};
    tileConfig.v1TileShape = {gTile, blockSize};
    tileConfig.c2TileShape = {gTile, gTile, blockSize, blockSize, blockSize, blockSize};
    tileConfig.v2TileShape = {gTile, blockSize};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3PagedAttnWindow<npu::tile_fwk::bfloat16>(data, tileConfig);
}

TEST_F(TestQwen3Atten, TestQwen3MLP_B_1_S_16_H_32_INTER_64_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

// medium size 可以通过
TEST_F(TestQwen3Atten, TestQwen3MLP_B_1_S_1_H_1024_INTER_1024_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

// medium size 可以通过
TEST_F(TestQwen3Atten, TestQwen3MLP_B_1_S_1_H_1024_INTER_6144_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

TEST_F(TestQwen3Atten, TestQwen3MLP_B_1_S_1_H_2048_INTER_12288_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

// 通过
TEST_F(TestQwen3Atten, TestQwen3MLP_B_1_S_1_H_4096_INTER_12288_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

// qwen3-8B MLP dims: H=4096, INTER=12288 (B=32, S=1)
// time: 603.54us  AI Core: 30.41%  Total Core: 60  Total Task Count: 374
TEST_F(TestQwen3Atten, TestQwen3MLP_B_32_S_1_H_4096_INTER_12288_BF16) {
    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Mlp<npu::tile_fwk::bfloat16>(data);
}

TEST_F(TestQwen3Atten, TestQwen3Layer_B_1_S_16_H_32_N_2_KV_2_INTER_64_BF16) {
    Qwen3AttenTileShapeConfig tileConfig;
    tileConfig.vecTileShape = {NUM_1, NUM_16, NUM_2, NUM_16};
    tileConfig.scale = 0.0f;

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3Layer<npu::tile_fwk::bfloat16>(data, tileConfig);
}

/************************************* Qwen3LayerNew (decode paged-attn) *************************************/

template <typename T>
void Qwen3LayerNewCompute(TestDataLoader &data, PaTileShapeConfig &paTileConfig, const Qwen3MlpDims &mlpDims,
    float softmaxScale, int maxUnrollTimes, bool isNzFormat = false, int windowSize = 0) {
    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int n = std::get<int>(data.Param("n"));
    int n_kv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int block_size = std::get<int>(data.Param("block_size"));
    int skv = std::get<int>(data.Param("skv"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    int h = n * d;
    int kv_hidden = n_kv * d;
    int block_num = (skv + block_size - 1) / block_size * b;
    int max_block_num_per_batch = (skv + block_size - 1) / block_size;

    npu::tile_fwk::Qwen3LayerNewCompute<T>(data.InputTensorCheck("hidden_states", dType, {b * s, h}),
        data.InputTensorCheck("attn_q_w", dType, {h, h}), data.InputTensorCheck("attn_q_b", dType, {h}),
        data.InputTensorCheck("attn_k_w", dType, {h, kv_hidden}), data.InputTensorCheck("attn_k_b", dType, {kv_hidden}),
        data.InputTensorCheck("attn_v_w", dType, {h, kv_hidden}), data.InputTensorCheck("attn_v_b", dType, {kv_hidden}),
        data.InputTensorCheck("attn_o_w", dType, {h, h}), data.InputTensorCheck("attn_o_b", dType, {h}),
        data.InputTensorCheck("attn_q_norm_w", dType, {d}), data.InputTensorCheck("attn_k_norm_w", dType, {d}),
        data.InputTensorCheck("cos", dType, {b, s, d}), data.InputTensorCheck("sin", dType, {b, s, d}),
        data.InputTensorCheck("cache_index", DT_INT32, {b, s}),
        data.InputTensorCheck("key_cache", dType, {block_num * block_size, kv_hidden}),
        data.InputTensorCheck("value_cache", dType, {block_num * block_size, kv_hidden}),
        data.InputTensorCheck("block_table", DT_INT32, {b, max_block_num_per_batch}),
        data.InputTensorCheck("act_seqs", DT_INT32, {b}), data.InputTensorCheck("gate_w", dType, {h, mlpDims.inter}),
        data.InputTensorCheck("up_w", dType, {h, mlpDims.inter}),
        data.InputTensorCheck("down_w", dType, {mlpDims.inter, h}), n, n_kv, block_size, data.OutputTensor("layer_out"),
        data.OutputTensor("key_cache_out"), data.OutputTensor("value_cache_out"), paTileConfig, mlpDims, softmaxScale,
        maxUnrollTimes, isNzFormat, windowSize);
}

template <typename T>
void qwen3LayerNew(TestDataLoader &data, PaTileShapeConfig &paTileConfig, int maxUnrollTimes = 1) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    // STITCH参数调优
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 50);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 200);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 30);
    // config::SetRuntimeOption(STITCH_FUNCTION_NUM_STEP, 30);
    // config::SetRuntimeOption(STITCH_FUNCTION_SIZE, 20000);

    // 控制流缓存配置
    // config::SetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM, 100);
    // config::SetRuntimeOption<int64_t>(CFGCACHE_ROOT_TASK_NUM, 1000);
    // config::SetRuntimeOption<int64_t>(CFGCACHE_LEAF_TASK_NUM, 10000);

    int b = std::get<int>(data.Param("b"));
    int s = std::get<int>(data.Param("s"));
    int n = std::get<int>(data.Param("n"));
    int n_kv = std::get<int>(data.Param("n_kv"));
    int d = std::get<int>(data.Param("d"));
    int inter = std::get<int>(data.Param("inter"));
    int block_size = std::get<int>(data.Param("block_size"));
    int skv = std::get<int>(data.Param("skv"));
    std::string dtypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dtypeStr);

    int h = n * d;
    int kv_hidden = n_kv * d;
    int block_num = (skv + block_size - 1) / block_size * b;
    int max_block_num_per_batch = (skv + block_size - 1) / block_size;

    Qwen3MlpDims mlpDims = {b, s, h, inter};
    float softmaxScale = 1.0f / std::sqrt(static_cast<float>(d));

    // 显式构建 tensor 以便使用 in-place 语法
    auto &hiddenStates = data.InputTensorCheck("hidden_states", dType, {b * s, h});
    auto &attnQw = data.InputTensorCheck("attn_q_w", dType, {h, h});
    auto &attnQb = data.InputTensorCheck("attn_q_b", dType, {h});
    auto &attnKw = data.InputTensorCheck("attn_k_w", dType, {h, kv_hidden});
    auto &attnKb = data.InputTensorCheck("attn_k_b", dType, {kv_hidden});
    auto &attnVw = data.InputTensorCheck("attn_v_w", dType, {h, kv_hidden});
    auto &attnVb = data.InputTensorCheck("attn_v_b", dType, {kv_hidden});
    auto &attnOw = data.InputTensorCheck("attn_o_w", dType, {h, h});
    auto &attnOb = data.InputTensorCheck("attn_o_b", dType, {h});
    auto &attnQnormW = data.InputTensorCheck("attn_q_norm_w", dType, {d});
    auto &attnKnormW = data.InputTensorCheck("attn_k_norm_w", dType, {d});
    auto &cos = data.InputTensorCheck("cos", dType, {b, s, d});
    auto &sin = data.InputTensorCheck("sin", dType, {b, s, d});
    auto &cacheIndex = data.InputTensorCheck("cache_index", DT_INT32, {b, s});
    auto &keyCache = data.InputTensorCheck("key_cache", dType, {block_num * block_size, kv_hidden});
    auto &valueCache = data.InputTensorCheck("value_cache", dType, {block_num * block_size, kv_hidden});
    auto &blockTable = data.InputTensorCheck("block_table", DT_INT32, {b, max_block_num_per_batch});
    auto &actSeqs = data.InputTensorCheck("act_seqs", DT_INT32, {b});
    auto &gateW = data.InputTensorCheck("gate_w", dType, {h, inter});
    auto &upW = data.InputTensorCheck("up_w", dType, {h, inter});
    auto &downW = data.InputTensorCheck("down_w", dType, {inter, h});

    auto &layerOut = data.OutputTensorCheck("layer_out", dType, {b * s, h});
    auto &keyCacheOut = data.OutputTensorCheck("key_cache_out", dType, {block_num * block_size, kv_hidden});
    auto &valueCacheOut = data.OutputTensorCheck("value_cache_out", dType, {block_num * block_size, kv_hidden});

    data.Dump();

    // 外层 FUNCTION 包装，使用 in-place 语法处理 KV cache
    FUNCTION("Qwen3LayerNew", data.GetInputTensorList(),
        {
            layerOut
    },                                            // 普通输出
        {{keyCacheOut, keyCache}, {valueCacheOut, valueCache}}) { // in-place 输出
        npu::tile_fwk::Qwen3LayerNewCompute<T>(hiddenStates, attnQw, attnQb, attnKw, attnKb, attnVw, attnVb, attnOw,
            attnOb, attnQnormW, attnKnormW, cos, sin, cacheIndex, keyCache, valueCache, blockTable, actSeqs, gateW, upW,
            downW, n, n_kv, block_size, layerOut, keyCacheOut, valueCacheOut, paTileConfig, mlpDims, softmaxScale,
            maxUnrollTimes, false, 0);
        std::cout << "[Qwen3LayerNew] Qwen3LayerNewCompute done" << std::endl;
    }
    std::cout << "[Qwen3LayerNew] FUNCTION done" << std::endl;

    auto goldenLayerOut = data.GoldenData("layer_out");
    auto outputLayerOut = data.GetOutputDataList()[data.GetOutputNameToIdx("layer_out")];
    std::cout << "[Qwen3LayerNew] goldenData done" << std::endl;
    std::cout << "[Qwen3LayerNew] outputData done" << std::endl;

#ifdef BUILD_WITH_CANN
    // 只传递普通输出 (layer_out) 到 DevFuncRunner，in-place 输出会写回到输入 buffer
    std::vector<RawTensorDataPtr> outList = {outputLayerOut};
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), outList);
    std::cout << "[Qwen3LayerNew] DevFuncRunner::Run done" << std::endl;
    // 对于多步骤 BF16 操作 (RMSNorm + Prolog + PagedAttention + OProj + MLP + Residuals)，0.1 是合理的误差阈值
    EXPECT_TRUE(
        resultCmp<T>((T *)goldenLayerOut->data(), (T *)outputLayerOut->data(), goldenLayerOut->GetSize(), 0.1f));
    std::cout << "[Qwen3LayerNew] resultCmp done" << std::endl;
#endif
}

// Qwen3-8B Layer (decode): B=32, S=1, nQ=32, nKv=8, D=128, blockSize=4096, inter=12288, skv=4096
// 恢复原始配置以验证精度
TEST_F(TestQwen3Atten, TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_4096_INTER_12288_SKV_4096_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4; // group = nQ/nKv = 4
    const int dTile = 64;    // d=128, tile by 64
    const int s2Tile = 4096; // 优化: 增大 softmax tile 到整个 s2 维度
    const int maxUnrollTimes = 4;

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};

    // QK: [gTile=4, d=128] x [s2=4096, d=128]^T => [4, 4096]
    // 优化: c1 nTile=256 可行，kTile=128 可行
    tileConfig.c1TileShape = {gTile, gTile, 128, 128, 256, 256};
    tileConfig.v1TileShape = {gTile, s2Tile};

    // PV: [gTile=4, s2=4096] x [s2=4096, d=128] => [4, 128]
    // 优化: c2 kTile=128 可行
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128};
    tileConfig.v2TileShape = {gTile, 512}; // 增大 v2 tile

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3LayerNew<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// Qwen3-8B Layer (decode): B=32, S=1, nQ=32, nKv=8, D=128, blockSize=1024, inter=12288, skv=1024
TEST_F(TestQwen3Atten, TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_1024_INTER_12288_SKV_1024_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4;  // group = nQ/nKv = 4
    const int dTile = 128;    // d=128, tile by 128
    const int s2Tile = 1024;  // 与 blockSize 匹配
    const int maxUnrollTimes = 8;

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};

    // QK: [gTile=4, d=128] x [s2=1024, d=128]^T => [4, 1024]
    tileConfig.c1TileShape = {gTile, gTile, 128, 128, 256, 256};
    tileConfig.v1TileShape = {gTile, s2Tile};

    // PV: [gTile=4, s2=1024] x [s2=1024, d=128] => [4, 128]
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128};
    tileConfig.v2TileShape = {gTile, 512};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3LayerNew<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}

// Qwen3-8B Layer (decode): B=32, S=1, nQ=32, nKv=8, D=128, blockSize=2048, inter=12288, skv=2048
TEST_F(TestQwen3Atten, TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_2048_INTER_12288_SKV_2048_BF16) {
    PaTileShapeConfig tileConfig;
    const int gTile = NUM_4;  // group = nQ/nKv = 4
    const int dTile = 128;    // d=128, tile by 128
    const int s2Tile = 2048;  // 与 blockSize 匹配
    const int maxUnrollTimes = 8;

    tileConfig.headNumQTile = gTile;
    tileConfig.v0TileShape = {gTile, dTile};

    // QK: [gTile=4, d=128] x [s2=2048, d=128]^T => [4, 2048]
    tileConfig.c1TileShape = {gTile, gTile, 128, 128, 256, 256};
    tileConfig.v1TileShape = {gTile, s2Tile};

    // PV: [gTile=4, s2=2048] x [s2=2048, d=128] => [4, 128]
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128};
    tileConfig.v2TileShape = {gTile, 512};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    qwen3LayerNew<npu::tile_fwk::bfloat16>(data, tileConfig, maxUnrollTimes);
}
