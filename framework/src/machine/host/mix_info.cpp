#include <nlohmann/json.hpp>
#include "mix_info.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
using json = nlohmann::json;
namespace npu {
namespace tile_fwk {

void GetExecuteFunc(Function* func, std::map<int, std::set<Function*>>& leafFunctions, int wrapid = -1)
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
            GetExecuteFunc(callFunc, leafFunctions, callopAttr->wrapId);
        }
        return;
    } else if (funcType == GraphType::EXECUTE_GRAPH) {
        leafFunctions[wrapid].insert(func);
        return;
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
    int warpID;
    std::vector<CoreTask> coreTask;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SyncInfo, isSet, eventID)

// 绑定CoreTask
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CoreTask, hashValue, syncMsg)

// 绑定WrapInfo
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WrapInfo, warpID, coreTask)

std::string GetMixInfoMain(Function* topFunc)
{
    std::map<int, std::set<Function*>> leafFunctions;
    GetExecuteFunc(topFunc, leafFunctions);
    std::map<int, WrapInfo> wrapInfos;
    for (auto& [warpID, leafFuncs] : leafFunctions) {
        for (auto& leafFunc : leafFuncs) {
            auto leafAttr = leafFunc->GetLeafFuncAttribute();
            if (leafAttr == nullptr) {
                continue;
            }
            if (wrapInfos.find(warpID) == wrapInfos.end()) {
                WrapInfo info;
                wrapInfos[warpID] = info;
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
            wrapInfos[warpID].coreTask.push_back(leafFuncSyncInfo);
        }
    }
    std::vector<WrapInfo> wrapinfoList;
    for (auto& [wrapid, wrapinfo] : wrapInfos) {
        wrapinfoList.push_back(wrapinfo);
    }
    json j = wrapinfoList;
    return j.dump(4);
}
} // namespace tile_fwk
} // namespace npu