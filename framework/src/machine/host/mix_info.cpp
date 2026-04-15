#include <fstream>
#include <nlohmann/json.hpp>
#include "mix_info.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
using json = nlohmann::json;
namespace npu {
namespace tile_fwk {

void GetExecuteFunc(Function* func, std::map<int, std::set<Function*>>& leafFunctions)
{
    auto funcType = func->GetGraphType();
    if (func->IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH},
            GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            GetExecuteFunc(callFunc, leafFunctions);
        }
        return;
    } else if (funcType == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto wrapId = callopAttr->wrapId;
            if (wrapId == -1) {
                continue;
            }
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            leafFunctions[wrapId].insert(callFunc);
        }
        return;
    } else if (funcType == GraphType::TILE_GRAPH) {
        GetExecuteFunc(func->GetRootFunction(), leafFunctions);
    }
    return;
}

struct SyncInfo {
    bool isSet;
    int eventID;
};

struct CoreTask {
    uint64_t hashValue;
    std::vector<SyncInfo> syncMsg;
};

struct WrapInfo {
    int wrapID;
    std::vector<CoreTask> coreTask;
};

struct MixInfo {
    uint64_t mixId;
    std::vector<WrapInfo> wrapInfos;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SyncInfo, isSet, eventID)

// 绑定CoreTask
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CoreTask, hashValue, syncMsg)

// 绑定WrapInfo
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WrapInfo, wrapID, coreTask)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MixInfo, mixId, wrapInfos)

void DumpMixInfo(const std::map<uint64_t, std::map<int, WrapInfo>>& wrapInfos) {
    std::vector<MixInfo> wrapinfoList;
    for (auto& [mixId, rootWrapinfo] : wrapInfos) {
        MixInfo mixInfo;
        mixInfo.mixId = mixId;
        for (auto& [wrapId, wrapInfo] : rootWrapinfo) {
            mixInfo.wrapInfos.push_back(wrapInfo);
        }
        wrapinfoList.push_back(mixInfo);
    }
    json j = wrapinfoList;
    std::string path = npu::tile_fwk::config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::ofstream of(path);
    if (of.is_open()) {
        of << j.dump(4);
        of.close();
    }
}

int GetMixInfoMain(Function* topFunc)
{
    std::map<int, std::set<Function*>> leafFunctions;
    GetExecuteFunc(topFunc, leafFunctions);
    std::map<uint64_t, std::map<int, WrapInfo>> wrapInfos;
    for (auto& [wrapID, leafFuncs] : leafFunctions) {
        for (auto& leafFunc : leafFuncs) {
            auto leafAttr = leafFunc->GetLeafFuncAttribute();
            if (leafAttr == nullptr) {
                continue;
            }
            auto mixId = leafAttr->mixId;
            if (wrapInfos.find(mixId) == wrapInfos.end() ||
                wrapInfos[mixId].find(wrapID) == wrapInfos[mixId].end()) {
                WrapInfo info;
                info.wrapID = wrapID;
                wrapInfos[mixId][wrapID] = info;
            }
            CoreTask leafFuncSyncInfo;
            leafFuncSyncInfo.hashValue = leafFunc->GetFunctionHash().GetHash();
            for (auto& op : leafFunc->Operations(false).DuplicatedOpList()) {
                SyncInfo syncInfo;
                if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
                    syncInfo.isSet = true;
                } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
                    syncInfo.isSet = false;
                } else {
                    continue;
                }
                syncInfo.eventID = op->GetSyncQueue().eventId_;
                leafFuncSyncInfo.syncMsg.push_back(syncInfo);
            }
            wrapInfos[mixId][wrapID].coreTask.push_back(leafFuncSyncInfo);
        }
    }
    DumpMixInfo(wrapInfos);
    return 0;
}
} // namespace tile_fwk
} // namespace npu