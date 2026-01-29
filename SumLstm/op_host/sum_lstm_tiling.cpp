#include "sum_lstm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include <cmath>
#include <algorithm>

namespace optiling {

// 计算对齐值 (对齐到 32 字节边界)
static inline uint32_t AlignUp32B(uint32_t size, uint32_t elemSize) {
    uint32_t alignElems = 32 / elemSize;  // 32B 对应的元素数
    return (size + alignElems - 1) / alignElems * alignElems;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    SumLstmTilingData tiling;

    // 获取输入 tensor 信息
    const gert::StorageShape* states4dShape = context->GetInputShape(0);  // states_4d
    const gert::StorageShape* prevCellShape = context->GetInputShape(2);  // prev_cell

    if (states4dShape == nullptr || prevCellShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取数据类型
    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dataTypeSize = (dtype == ge::DT_FLOAT16) ? 2 : 4;
    tiling.set_dataTypeSize(dataTypeSize);

    // 解析形状参数
    const gert::Shape& statesShape = states4dShape->GetStorageShape();
    int32_t dimCount = statesShape.GetDimNum();

    // 最后一维是 4D (gatedDim)
    uint32_t gatedDim = static_cast<uint32_t>(statesShape.GetDim(dimCount - 1));
    uint32_t hiddenDim = gatedDim / 4;

    // 计算总样本数 (除最后一维外的所有维度乘积)
    uint32_t totalSamples = 1;
    for (int32_t i = 0; i < dimCount - 1; ++i) {
        totalSamples *= static_cast<uint32_t>(statesShape.GetDim(i));
    }

    tiling.set_totalSamples(totalSamples);
    tiling.set_hiddenDim(hiddenDim);
    tiling.set_gatedDim(gatedDim);

    // 计算对齐后的维度
    uint32_t hiddenDimAligned = AlignUp32B(hiddenDim, dataTypeSize);
    uint32_t gatedDimAligned = AlignUp32B(gatedDim, dataTypeSize);
    tiling.set_hiddenDimAligned(hiddenDimAligned);
    tiling.set_gatedDimAligned(gatedDimAligned);

    // 获取硬件平台信息
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    // 获取 UB 大小
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 计算每个样本需要的 UB 内存
    // 输入: states_4d(4D), z4_4d(4D), prev_cell(D) = 9D
    // 中间: pre_f/i/o(D), cpre(D), f/i/o(D), cpre_norm(D), cact(D), out_cell(D), cnew_norm(D), sact(D) = 12D
    // 输出: out_state(D), out_cell(D) = 2D
    // 可选: w_cell(D), b_cell(D), w_state(D), b_state(D) = 4D
    // 总计约 27D (保守估计)
    uint32_t elementsPerSample = 30 * hiddenDimAligned;
    uint32_t bytesPerSample = elementsPerSample * dataTypeSize;

    // 双缓冲
    uint32_t doubleBufferFactor = 2;
    uint32_t bytesPerSampleWithBuffer = bytesPerSample * doubleBufferFactor;

    // 计算每个核心可以处理的样本数 (预留一些空间)
    uint64_t availableUb = ubSize * 8 / 10;  // 使用 80% 的 UB
    uint32_t maxSamplesPerTile = static_cast<uint32_t>(availableUb / bytesPerSampleWithBuffer);
    if (maxSamplesPerTile == 0) {
        maxSamplesPerTile = 1;
    }

    // 分核策略: 均匀分配样本到各核心
    uint32_t usedCoreNum = std::min(coreNum, totalSamples);
    uint32_t samplesPerCore = totalSamples / usedCoreNum;
    uint32_t remainSamples = totalSamples % usedCoreNum;

    tiling.set_coreNum(usedCoreNum);
    tiling.set_samplesPerCore(samplesPerCore);
    tiling.set_remainSamples(remainSamples);

    // 计算每个核心的 tile 数量
    uint32_t tileSamples = std::min(maxSamplesPerTile, samplesPerCore + 1);
    uint32_t tileNumPerCore = (samplesPerCore + tileSamples - 1) / tileSamples;
    if (tileNumPerCore == 0) {
        tileNumPerCore = 1;
    }
    uint32_t lastTileSamples = samplesPerCore - (tileNumPerCore - 1) * tileSamples;
    if (lastTileSamples == 0) {
        lastTileSamples = tileSamples;
    }

    tiling.set_tileNumPerCore(tileNumPerCore);
    tiling.set_tileSamples(tileSamples);
    tiling.set_lastTileSamples(lastTileSamples);
    tiling.set_ubBufferSize(tileSamples * hiddenDimAligned * dataTypeSize);

    // 获取算子属性
    const auto* attrs = context->GetAttrs();
    float alpha = 1.0f;
    float epsCell = 1e-6f;
    float epsState = 1e-6f;
    bool useFastGelu = true;

    if (attrs != nullptr) {
        const float* alphaPtr = attrs->GetAttrPointer<float>(0);
        const float* epsCellPtr = attrs->GetAttrPointer<float>(1);
        const float* epsStatePtr = attrs->GetAttrPointer<float>(2);
        const bool* useFastGeluPtr = attrs->GetAttrPointer<bool>(3);

        if (alphaPtr) alpha = *alphaPtr;
        if (epsCellPtr) epsCell = *epsCellPtr;
        if (epsStatePtr) epsState = *epsStatePtr;
        if (useFastGeluPtr) useFastGelu = *useFastGeluPtr;
    }

    tiling.set_alpha(alpha);
    tiling.set_epsCell(epsCell);
    tiling.set_epsState(epsState);
    tiling.set_useFastGelu(useFastGelu ? 1 : 0);

    // 检查可选输入
    uint32_t hasWCell = (context->GetInputTensor(3) != nullptr) ? 1 : 0;
    uint32_t hasBCell = (context->GetInputTensor(4) != nullptr) ? 1 : 0;
    uint32_t hasWState = (context->GetInputTensor(5) != nullptr) ? 1 : 0;
    uint32_t hasBState = (context->GetInputTensor(6) != nullptr) ? 1 : 0;

    tiling.set_hasWCell(hasWCell);
    tiling.set_hasBCell(hasBCell);
    tiling.set_hasWState(hasWState);
    tiling.set_hasBState(hasBState);

    // Debug 日志
    printf("[SumLstm Tiling] totalSamples=%u, hiddenDim=%u, gatedDim=%u, coreNum=%u, samplesPerCore=%u, remainSamples=%u, tileSamples=%u, tileNumPerCore=%u, alpha=%f, epsCell=%f, epsState=%f, useFastGelu=%u, hasWCell=%u, hasBCell=%u, hasWState=%u, hasBState=%u, dataTypeSize=%u, ubSize=%lu\n", totalSamples, hiddenDim, gatedDim, usedCoreNum, samplesPerCore, remainSamples, tileSamples, tileNumPerCore, alpha, epsCell, epsState, useFastGelu ? 1 : 0, hasWCell, hasBCell, hasWState, hasBState, dataTypeSize, ubSize);

    // 设置 Tiling 数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 设置 Block 维度 (核心数)
    context->SetBlockDim(usedCoreNum);

    // 设置 Tiling Key (0: 标准路径)
    context->SetTilingKey(0);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

// InferShape 实现
static graphStatus InferShape(gert::InferShapeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 获取 prev_cell 的形状作为输出形状参考 (..., D)
    const gert::Shape* prevCellShape = context->GetInputShape(2);
    if (prevCellShape == nullptr) {
        return GRAPH_FAILED;
    }

    // out_state 和 out_cell 的形状与 prev_cell 相同
    gert::Shape* outStateShape = context->GetOutputShape(0);
    gert::Shape* outCellShape = context->GetOutputShape(1);

    if (outStateShape == nullptr || outCellShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outStateShape = *prevCellShape;
    *outCellShape = *prevCellShape;

    printf("[SumLstm InferShape] output shape dims=%d\n", static_cast<int>(prevCellShape->GetDimNum()));

    return GRAPH_SUCCESS;
}

// InferDataType 实现
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 输出数据类型与输入 states_4d 相同
    DataType inputDtype = context->GetInputDataType(0);

    context->SetOutputDataType(0, inputDtype);  // out_state
    context->SetOutputDataType(1, inputDtype);  // out_cell

    printf("[SumLstm InferDataType] dtype=%d\n", static_cast<int>(inputDtype));

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class SumLstm : public OpDef {
public:
    explicit SumLstm(const char* name) : OpDef(name) {
        // 必选输入
        this->Input("states_4d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("z4_4d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("prev_cell")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 可选输入
        this->Input("w_cell")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b_cell")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_state")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b_state")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 输出
        this->Output("out_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out_cell")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 属性
        this->Attr("alpha")
            .AttrType(OPTIONAL)
            .Float(1.0);

        this->Attr("eps_cell")
            .AttrType(OPTIONAL)
            .Float(1e-6);

        this->Attr("eps_state")
            .AttrType(OPTIONAL)
            .Float(1e-6);

        this->Attr("use_fast_gelu")
            .AttrType(OPTIONAL)
            .Bool(true);

        // 注册 InferShape 和 InferDataType
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // 注册 AICore Tiling
        this->AICore()
            .SetTiling(optiling::TilingFunc);

        // 添加硬件配置
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

// 注册算子
OP_ADD(SumLstm);

}  // namespace ops
