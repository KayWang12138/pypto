#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <functional>
#include "codegen/codegen_common.h"
#include "codegen_op_cloudnpu.h"
#include "interface/utils/log.h"
#include "securec.h"
#include "interface/operation/distributed/distributed_common.h"

namespace npu::tile_fwk {

using AtomicType = Distributed::AtomicType;
using DistributedOpAttr = Distributed::DistributedOpAttr;
constexpr int32_t GM2UB_SHMEMDATA_INDEX = 2;

void CheckInRange(int64_t value)
{
    if (value < std::numeric_limits<uint32_t>::min() || value > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Invalid value: " + std::to_string(value));
    }
}

// ---------------------------------------------------------
// 1. 策略接口
// ---------------------------------------------------------
class DistributedOpStrategy {
public:
    virtual ~DistributedOpStrategy() = default;
    
    virtual std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const = 0;
    
    virtual std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const {
        return "";
    }
    
    virtual std::unordered_set<int32_t> GetSkipOperands() const {
        return {};
    }
};

// ---------------------------------------------------------
// 2. 具体策略实现
// ---------------------------------------------------------

// --- ShmemPut/Get 相关策略 ---
class ShmemPutGetStrategy : public DistributedOpStrategy {
private:
    bool isPut_;
    
    std::pair<int32_t, int32_t> getDataIndices(Opcode opCode) const {
        static const std::unordered_map<Opcode, std::array<int32_t, 2>> map = {
            {Opcode::OP_SHMEM_PUT, {3, 4}},
            {Opcode::OP_SHMEM_GET, {0, 3}},
            {Opcode::OP_SHMEM_PUT_UB2GM, {1, GM2UB_SHMEMDATA_INDEX}},
            {Opcode::OP_SHMEM_GET_GM2UB, {0, 3}}
        };
        const auto& arr = map.at(opCode);
        return {arr[0], arr[1]};
    }

public:
    ShmemPutGetStrategy(bool isPut) : isPut_(isPut) {}

    std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        auto [nonShmemDataIndex, shmemDataIndex] = getDataIndices(host.opCode);
        
        const auto& tileShape = host.originShape[shmemDataIndex];
        int64_t tileRowShape = tileShape[tileShape.size() - 2];
        int64_t tileColShape = tileShape[tileShape.size() - 1];

        auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        int64_t bufferRowShape = distributedOpAttr.copyBufferShape[0];
        int64_t bufferColShape = distributedOpAttr.copyBufferShape[1];

        const auto& shmemTensorRawShape = host.rawShape[shmemDataIndex];
        const auto& nonShmemTensorRawShape = host.rawShape[nonShmemDataIndex];
        int64_t srcStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
        int64_t dstStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
        
        if ((host.opCode == Opcode::OP_SHMEM_GET) || (host.opCode == Opcode::OP_SHMEM_GET_GM2UB)) {
            srcStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
            dstStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
        }

        CheckInRange(tileRowShape);
        CheckInRange(tileColShape);
        CheckInRange(bufferRowShape);
        CheckInRange(bufferColShape);
        CheckInRange(srcStride);
        CheckInRange(dstStride);

        oss << "<" << DataType2CCEStr(host.operandDtype[nonShmemDataIndex]) << ", " 
            << DataType2CCEStr(host.operandDtype[shmemDataIndex]) << ", " 
            << tileRowShape << ", " << tileColShape << ", " << bufferRowShape 
            << ", " << bufferColShape << ", " << srcStride << ", " << dstStride << ", "
            << Distributed::AtomicTypeToString(distributedOpAttr.atomicType) << ">";
        return oss.str();
    }

    std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        auto [nonShmemDataIndex, shmemDataIndex] = getDataIndices(host.opCode);
        
        if (host.opCode == Opcode::OP_SHMEM_PUT) {
            int32_t nonShmemDataDim = host.originShape[nonShmemDataIndex].size();
            int32_t shmemDataDim = 4;
            std::string viewOffsetStr = host.dynamicValidShape[shmemDataIndex][2].Dump();
            size_t firstComma = viewOffsetStr.find(",");
            size_t lastComma = viewOffsetStr.rfind(",");
            std::string viewOffset = viewOffsetStr.substr(firstComma + 1, lastComma - firstComma - 1);
            
            oss << ", " << host.GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim) 
                << ", " << host.GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
            if (viewOffset.find("RUNTIME_GetTensorDataInt32Dim2") != std::string::npos) {
                oss << ", " << viewOffset;
            } else {
                oss << ", " << -1;
            }
        } 
        else if (host.opCode == Opcode::OP_SHMEM_GET || host.opCode == Opcode::OP_SHMEM_GET_GM2UB) {
            int32_t nonShmemDataDim = host.originShape[nonShmemDataIndex].size();
            int32_t shmemDataDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim) 
                << ", " << host.GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
        }
        else {
             // OP_SHMEM_PUT_UB2GM
             int32_t nonShmemDataDim = 2;
             int32_t shmemDataDim = 4;
             oss << ", " << host.GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim)
                 << ", " << host.GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
        }
        return oss.str();
    }

    std::unordered_set<int32_t> GetSkipOperands() const override {
        if (isPut_) return {0, 2};
        return {2};
    }
};

// --- Signal 相关策略 ---
class ShmemSignalStrategy : public DistributedOpStrategy {
public:
    std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        oss << "<" << std::to_string(distributedOpAttr.signalValue) << ", "
            << std::to_string(distributedOpAttr.signalStride) << ", "
            << std::to_string(distributedOpAttr.tileRowShape) << ", "
            << std::to_string(distributedOpAttr.tileColShape) << ", "
            << Distributed::AtomicTypeToString(distributedOpAttr.atomicType) << ">";
        return oss.str();
    }

    std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        int32_t shmemSignalIndex = 3;
        int32_t shmemSignalDim = 5;
        oss << ", " << host.GenOffsetsAndRawShapes(shmemSignalIndex, shmemSignalDim)
            << ", " << host.GenShapes(shmemSignalIndex, shmemSignalDim);
        return oss.str();
    }

    std::unordered_set<int32_t> GetSkipOperands() const override { return {0, 2}; }
};

// --- Set 相关策略 ---
class ShmemSetStrategy : public DistributedOpStrategy {
public:
    std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        int32_t shmemTensorIndex = 3;
        auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        int64_t bufferEleNum = distributedOpAttr.setBufferShape[0];
        int64_t rawShapeRow;
        int64_t rawShapeCol;

        if (distributedOpAttr.setType == 0) {
            rawShapeRow = host.originShape[shmemTensorIndex][2];
            rawShapeCol = host.originShape[shmemTensorIndex][3];
        } else {
            rawShapeRow = Distributed::MAX_TILE_NUM;
            rawShapeCol = Distributed::SHMEM_SIGNAL_STRIDE;
        }
        oss << "<" << host.GetTemplateDType() << ", " << host.originShape[shmemTensorIndex][1] << ", "
            << rawShapeRow << ", " << rawShapeCol << ", " << bufferEleNum << ">";
        return oss.str();
    }

    std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        int32_t shmemTensorIndex = 3;
        if (distributedOpAttr.setType == 0) {
            int32_t shmemTensorDim = 4;
            oss << ", " << host.GenOffsets(shmemTensorIndex, shmemTensorDim);
        } else {
            int32_t shmemTensorDim = 5;
            oss << ", " << host.GenOffsetsAndRawShapes(shmemTensorIndex, shmemTensorDim) 
                << ", " << host.GenShapes(shmemTensorIndex, shmemTensorDim);
        }
        return oss.str();
    }

    std::unordered_set<int32_t> GetSkipOperands() const override { return {0, 2}; }
};

// --- Moe Distributed Combine 策略 ---
class MoeDistributedCombineStrategy : public DistributedOpStrategy {
    bool isSend_;
public:
    MoeDistributedCombineStrategy(bool isSend) : isSend_(isSend) {}

    std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const override {
        int32_t expandXIndex = 4;
        int32_t outIndex = 0;
        int32_t index = isSend_ ? expandXIndex : outIndex;
        
        std::ostringstream oss;
        auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        
        int64_t secondToLastIndex = 2;
        int64_t rowShape = host.originShape[index][host.originShape[index].size() - secondToLastIndex];
        if (distributedOpAttr.rowShape != -1) {
            rowShape = distributedOpAttr.rowShape;
        }
        int64_t colShape = host.originShape[index][host.originShape[index].size() - 1];
        
        oss << "<" << host.GetTemplateDType() << ", " << distributedOpAttr.topK << ", " << rowShape << ", " 
            << colShape << ", " << distributedOpAttr.paddedColShape << ">";
        return oss.str();
    }

    std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        int32_t expandXIndex = 4;
        int32_t expandXDim = 2;
        int32_t shmemDataIndex = 6;
        int32_t shmemDataDim = 4;

        if (isSend_) {
            oss << ", " << host.GenOffsets(expandXIndex, expandXDim);
        } else {
            auto distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
            oss << ", " << host.GenOffsets(shmemDataIndex, shmemDataDim) << ", " << distributedOpAttr.rowOffset;
        }
        return oss.str();
    }

    std::unordered_set<int32_t> GetSkipOperands() const override {
        return isSend_ ? std::unordered_set<int32_t>{0} : std::unordered_set<int32_t>{4};
    }
};

// --- MoeDistributedDispatch 策略 ---
// 处理 OP_SEND_TO_ROUTING_EXPERT, OP_FFN_SCHED 等算子
class MoeDistributedDispatch : public DistributedOpStrategy {
public:
    std::string GenTemplateParams(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        DistributedOpAttr distributedOpAttr;
        if (host.opAttrs.count(OpAttributeKey::distributedOpAttr) != 0) {
            distributedOpAttr = AnyCast<DistributedOpAttr>(host.opAttrs.at(OpAttributeKey::distributedOpAttr));
        }
        if (distributedOpAttr.extraTemplateParam.empty()) {
            oss << "<" << host.GetTemplateDType() << ">";
        } else {
            oss << "<" << host.GetTemplateDType() << ", " << distributedOpAttr.extraTemplateParam << ">";
        }
        return oss.str();
    }

    std::string GenExtraParamsStr(const CodeGenOpCloudNPU& host) const override {
        std::ostringstream oss;
        
        if (host.opCode == Opcode::OP_SEND_TO_ROUTING_EXPERT) {
            int32_t expertTableIndex = 6;
            int32_t expertTableDim = 2;
            int32_t shmemDataIndex = 5;
            int32_t shmemDataDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(expertTableIndex, expertTableDim) 
                << ", " << host.GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
        } 
        else if (host.opCode == Opcode::OP_SEND_TO_SHARED_EXPERT) {
            int32_t tokenIndex = 2;
            int32_t tokenDim = 2;
            int32_t shmemDataIndex = 3;
            int32_t shmemDataDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(tokenIndex, tokenDim) 
                << ", " << host.GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
        }
        else if (host.opCode == Opcode::OP_COPY_TO_LOCAL_EXPERT) {
            int32_t tokenIndex = 3;
            int32_t tokenDim = 2;
            oss << ", " << host.GenOffsetsAndRawShapes(tokenIndex, tokenDim);
        }
        else if (host.opCode == Opcode::OP_DISPATCH_SET_FLAG) {
            int32_t shmemFlagIndex = 5;
            int32_t shmemFlagDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(shmemFlagIndex, shmemFlagDim);
        }
        else if (host.opCode == Opcode::OP_FFN_SCHED || 
                 host.opCode == Opcode::OP_FFN_BATCHING || 
                 host.opCode == Opcode::OP_FFN_VALIDCNT) {
            int32_t shmemIndex = 3;
            int32_t shmemDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(shmemIndex, shmemDim);
        }
        else if (host.opCode == Opcode::OP_FFN_COMBINEINFO) {
            int32_t shmemIndex = 2;
            int32_t shmemDim = 4;
            oss << ", " << host.GenOffsetsAndRawShapes(shmemIndex, shmemDim);
        }
        return oss.str();
    }
};

// ---------------------------------------------------------
// 3. 工厂
// ---------------------------------------------------------
class DistributedOpStrategyFactory {
public:
    static std::unique_ptr<DistributedOpStrategy> CreateStrategy(Opcode opCode) {
        switch (opCode) {
            case Opcode::OP_SHMEM_PUT:
            case Opcode::OP_SHMEM_PUT_UB2GM:
                return std::make_unique<ShmemPutGetStrategy>(true);
            
            case Opcode::OP_SHMEM_GET:
            case Opcode::OP_SHMEM_GET_GM2UB:
                return std::make_unique<ShmemPutGetStrategy>(false);
            
            case Opcode::OP_SHMEM_SIGNAL:
                return std::make_unique<ShmemSignalStrategy>();
                
            case Opcode::OP_SHMEM_SET:
                return std::make_unique<ShmemSetStrategy>();

            case Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND:
                return std::make_unique<MoeDistributedCombineStrategy>(true);

            case Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE:
                return std::make_unique<MoeDistributedCombineStrategy>(false);
            
            // 使用 MoeDistributedDispatch 处理剩余算子
            case Opcode::OP_SEND_TO_ROUTING_EXPERT:
            case Opcode::OP_SEND_TO_SHARED_EXPERT:
            case Opcode::OP_COPY_TO_LOCAL_EXPERT:
            case Opcode::OP_DISPATCH_SET_FLAG:
            case Opcode::OP_FFN_SCHED:
            case Opcode::OP_FFN_BATCHING:
            case Opcode::OP_FFN_VALIDCNT:
            case Opcode::OP_FFN_COMBINEINFO:
                return std::make_unique<MoeDistributedDispatch>();
        }
    }
};

// ---------------------------------------------------------
// 4. 修改后的上下文类
// ---------------------------------------------------------

std::string CodeGenOpCloudNPU::GetTemplateDType() const
{
    static const std::unordered_map<Opcode, int32_t> dTypeOperandIndexMap = {
        {Opcode::OP_FFN_BATCHING, 0},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, 0},
        {Opcode::OP_COPY_TO_LOCAL_EXPERT, 0},
        {Opcode::OP_SEND_TO_ROUTING_EXPERT, 1},
        {Opcode::OP_SEND_TO_SHARED_EXPERT, 1},
        {Opcode::OP_FFN_SCHED, 1},
        {Opcode::OP_FFN_VALIDCNT, 1},
        {Opcode::OP_SHMEM_PUT, 1},
        {Opcode::OP_SHMEM_PUT_UB2GM, 1},
        {Opcode::OP_SHMEM_SIGNAL, 1},
        {Opcode::OP_SHMEM_WAIT_UNTIL, 1},
        {Opcode::OP_SHMEM_GET, 1},
        {Opcode::OP_SHMEM_GET_GM2UB, 1},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, 1},
        {Opcode::OP_FFN_COMBINEINFO, 2},
        {Opcode::OP_SHMEM_SET, 3},
        {Opcode::OP_DISPATCH_SET_FLAG, 4},
    };
    auto it = dTypeOperandIndexMap.find(opCode);
    ASSERT(it != dTypeOperandIndexMap.end()) << "Opcode is out of range";
    int32_t operandIndex = it->second;
    return DataType2CCEStr(operandDtype[operandIndex]);
}

// 基础工具函数
std::string CodeGenOpCloudNPU::GenOffsets(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_OFFSET)[0];
}

std::string CodeGenOpCloudNPU::GenShapes(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, "SHAPE")[0];
}

std::string CodeGenOpCloudNPU::GenRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_RAW_SHAPE)[0];
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenOffsets(operandIndex, dim) + ", " + GenRawShapes(operandIndex, dim);
}

// 核心入口
std::string CodeGenOpCloudNPU::GenDistOp() const
{
    auto strategy = DistributedOpStrategyFactory::CreateStrategy(opCode);
    
    std::ostringstream oss;
    oss << tileOpName 
        << strategy->GenTemplateParams(*this) 
        << "(" 
        << GenParamsStr(strategy->GetSkipOperands()) 
        << strategy->GenExtraParamsStr(*this) 
        << ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk
