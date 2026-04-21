/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_host_topo.cpp
 * \brief Implementation of host-side static topology dumps. All CSV layout,
 */
#include "dump_host_topo.h"

#include <fstream>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/tensor_slot.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_operation.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dump {
namespace {

/// True iff runtime debug-mode CSV dumps should be produced.
bool DumpEnabled()
{
    return config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL;
}

/// CSV-quote a string per RFC4180 (wrap with ", double inner ").
std::string CsvQuote(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

/// Append `tok` to `acc` (a `sep`-separated string), preserving order and
/// dropping duplicates. No-op when `tok` is empty.
void AppendUniqueToken(std::string& acc, const std::string& tok, char sep = ';')
{
    if (tok.empty()) {
        return;
    }
    size_t pos = 0;
    while (pos <= acc.size()) {
        size_t s = acc.find(sep, pos);
        size_t end = (s == std::string::npos) ? acc.size() : s;
        if (acc.compare(pos, end - pos, tok) == 0) {
            return;
        }
        if (s == std::string::npos) {
            break;
        }
        pos = s + 1;
    }
    if (!acc.empty()) {
        acc.push_back(sep);
    }
    acc.append(tok);
}

/// Open `<LogTopFolder>/<file>` and write the header iff `enabled`. Returns
/// an ofstream that is_open() iff the file was actually opened.
std::ofstream OpenCsv(const std::string& fileName,
                      std::initializer_list<const char*> header,
                      std::string& outPath, bool enabled)
{
    std::ofstream ofs;
    if (!enabled) {
        return ofs;
    }
    outPath = config::LogTopFolder() + "/" + fileName;
    ofs.open(outPath);
    if (!ofs.is_open()) {
        return ofs;
    }
    bool first = true;
    for (const char* col : header) {
        if (!first) {
            ofs << ',';
        }
        ofs << col;
        first = false;
    }
    ofs << '\n';
    return ofs;
}

/// Write the trailing 5 columns shared by partial / full-cover rows of
/// slot_cell_table.csv: dim, "[cellShape]", "[strideShape]", cellCount.
/// Caller emits the leading slot/policy/root/funcKey columns and the
/// trailing outcastCount.
void WriteCellMatchDesc(std::ostream& os, const dynamic::DevCellMatchTableDesc& desc)
{
    int dim = desc.GetDimensionSize();
    os << dim << ",\"[";
    for (int d = 0; d < dim; ++d) {
        if (d != 0) {
            os << ',';
        }
        os << desc.GetCellShape(d);
    }
    os << "]\",\"[";
    for (int d = 0; d < dim; ++d) {
        if (d != 0) {
            os << ',';
        }
        os << desc.GetStrideShape(d);
    }
    os << "]\"," << desc.GetStride(0);
}

} // namespace

// -----------------------------------------------------------------------------
// DumpSlotMapping: one-shot dump of slot_mapping.csv.
// -----------------------------------------------------------------------------
void DumpSlotMapping(const TensorSlotManager& slotManager,
                     const std::unordered_map<int, int>& slotIdxMapping,
                     const IncastOutcastLink& inoutLink)
{
    std::string path;
    std::ofstream ofs = OpenCsv(
        "slot_mapping.csv",
        {"frontendSlotIdx", "runtimeSlotIdx", "slotRole", "tensorName", "funcRawName"},
        path, DumpEnabled());
    if (!ofs.is_open()) {
        return;
    }

    std::set<int> inputRuntimeSlots(
        inoutLink.inputSlotIndexList.begin(), inoutLink.inputSlotIndexList.end());
    std::set<int> outputRuntimeSlots(
        inoutLink.outputSlotIndexList.begin(), inoutLink.outputSlotIndexList.end());

    // slotIndexDict maps TensorSlot -> frontendSlotIdx. Multiple TensorSlots
    // can share the same frontend index (alias / inplace), so we aggregate
    // their tensor names and function raw names into ';'-joined columns.
    std::unordered_map<int, std::string> feSlotToTensorName;
    std::unordered_map<int, std::string> feSlotToFuncName;
    for (const auto& kv : slotManager.slotIndexDict) {
        int feIdx = kv.second;
        auto nameIt = slotManager.slotNameDict.find(kv.first);
        if (nameIt != slotManager.slotNameDict.end() && !nameIt->second.empty()) {
            AppendUniqueToken(feSlotToTensorName[feIdx], nameIt->second);
        }
        auto funcIt = slotManager.slotFuncNameDict.find(kv.first);
        if (funcIt == slotManager.slotFuncNameDict.end() || funcIt->second.empty()) {
            continue;
        }
        // slotFuncNameDict already stores a ';'-joined token list; re-split
        // so we can dedup against names already collected for this feIdx.
        const std::string& joined = funcIt->second;
        size_t start = 0;
        while (start <= joined.size()) {
            size_t sep = joined.find(';', start);
            size_t end = (sep == std::string::npos) ? joined.size() : sep;
            if (end > start) {
                AppendUniqueToken(feSlotToFuncName[feIdx], joined.substr(start, end - start));
            }
            if (sep == std::string::npos) {
                break;
            }
            start = sep + 1;
        }
    }

    for (const auto& slot : slotIdxMapping) {
        const char* role = "INTERNAL";
        bool isInput = inputRuntimeSlots.count(slot.second) != 0;
        bool isOutput = outputRuntimeSlots.count(slot.second) != 0;
        if (isInput && isOutput) {
            role = "INOUT";
        } else if (isInput) {
            role = "INPUT";
        } else if (isOutput) {
            role = "OUTPUT";
        }
        ofs << slot.first << ',' << slot.second << ',' << role << ','
            << CsvQuote(feSlotToTensorName[slot.first]) << ','
            << CsvQuote(feSlotToFuncName[slot.first]) << '\n';
    }
    ofs.close();
    MACHINE_LOGD("SlotMapping dumped to %s, total %zu entries",
                 path.c_str(), slotIdxMapping.size());
}

// -----------------------------------------------------------------------------
// StaticTopoCsvWriter
// -----------------------------------------------------------------------------
struct StaticTopoCsvWriter::Impl {
    std::ofstream ofs;
    std::string path;
};

StaticTopoCsvWriter::StaticTopoCsvWriter() : impl_(std::make_unique<Impl>())
{
    impl_->ofs = OpenCsv(
        "static_topo.csv",
        {"funcKey", "rootHash", "rawName", "opIdx", "opmagic", "leafHash",
         "coreType", "psgId", "incastSlots", "outcastSlots", "staticSuccessors"},
        impl_->path, DumpEnabled());
}

StaticTopoCsvWriter::~StaticTopoCsvWriter()
{
    if (impl_->ofs.is_open()) {
        impl_->ofs.close();
        MACHINE_LOGD("StaticTopo dumped to %s", impl_->path.c_str());
    }
}

bool StaticTopoCsvWriter::Enabled() const
{
    return impl_->ofs.is_open();
}

void StaticTopoCsvWriter::WriteFunction(int devRootKey, dynamic::DevAscendFunction& funcBin,
                                        const std::vector<CceCodeInfo>& cceCodeInfoList)
{
    if (!impl_->ofs.is_open()) {
        return;
    }
    auto& os = impl_->ofs;
    for (size_t opIdx = 0; opIdx < funcBin.GetOperationSize(); opIdx++) {
        int cceIndex = funcBin.GetOperationAttrCalleeIndex(opIdx);
        bool cceValid = (cceIndex >= 0 && cceIndex < static_cast<int>(cceCodeInfoList.size()));
        uint64_t leafHash = cceValid ? cceCodeInfoList[cceIndex].funcHash : 0;
        uint32_t coreType = cceValid ? cceCodeInfoList[cceIndex].coreType : 0;
        uint32_t psgId = cceValid ? cceCodeInfoList[cceIndex].psgId : 0;

        os << devRootKey << ',' << funcBin.rootHash << ','
           << funcBin.GetRawName() << ',' << opIdx << ','
           << funcBin.GetOperationDebugOpmagic(opIdx) << ','
           << leafHash << ',' << coreType << ',' << psgId << ',';

        // incastSlots: nested list "[g1_a/b;g2_c/d]" across all incast groups.
        os << '[';
        for (size_t i = 0; i < funcBin.GetIncastSize(); i++) {
            auto& incast = funcBin.GetIncast(i);
            if (i > 0) {
                os << ';';
            }
            for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
                if (j > 0) {
                    os << '/';
                }
                os << funcBin.At(incast.fromSlotList, j);
            }
        }
        os << "],[";

        // outcastSlots: same nested layout as incast.
        for (size_t i = 0; i < funcBin.GetOutcastSize(); i++) {
            auto& outcast = funcBin.GetOutcast(i);
            if (i > 0) {
                os << ';';
            }
            for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
                if (j > 0) {
                    os << '/';
                }
                os << funcBin.At(outcast.toSlotList, j);
            }
        }
        os << ']';

        // staticSuccessors: trailing variable-length opIdx list, one per column.
        auto& succList = funcBin.GetOperationDepGraphSuccList(opIdx);
        for (size_t j = 0; j < succList.size(); j++) {
            os << ',' << funcBin.At(succList, j);
        }
        os << '\n';
    }
}

// -----------------------------------------------------------------------------
// SlotCellTableCsvWriter
// -----------------------------------------------------------------------------
struct SlotCellTableCsvWriter::Impl {
    std::ofstream ofs;
    std::string path;
};

SlotCellTableCsvWriter::SlotCellTableCsvWriter(bool fillContent)
    : impl_(std::make_unique<Impl>())
{
    impl_->ofs = OpenCsv(
        "slot_cell_table.csv",
        {"slotIdx", "stitchPolicy", "rootHash", "funcKey",
         "dim", "cellShape", "strideShape", "cellCount", "outcastCount"},
        impl_->path, fillContent && DumpEnabled());
}

SlotCellTableCsvWriter::~SlotCellTableCsvWriter()
{
    if (impl_->ofs.is_open()) {
        impl_->ofs.flush();
        impl_->ofs.close();
    }
}

bool SlotCellTableCsvWriter::Enabled() const
{
    return impl_->ofs.is_open();
}

void SlotCellTableCsvWriter::WritePartial(int slotIdx,
                                          const dynamic::DevCellMatchTableDesc& desc,
                                          size_t outcastCount)
{
    if (!impl_->ofs.is_open()) {
        return;
    }
    // rootHash=0, funcKey=-1 are sentinels meaning "aggregated across roots".
    impl_->ofs << slotIdx << ",partial,0,-1,";
    WriteCellMatchDesc(impl_->ofs, desc);
    impl_->ofs << ',' << outcastCount << '\n';
}

void SlotCellTableCsvWriter::WriteFullCover(int slotIdx, uint64_t rootHash, int funcKey,
                                            const dynamic::DevCellMatchTableDesc& desc)
{
    if (!impl_->ofs.is_open()) {
        return;
    }
    impl_->ofs << slotIdx << ",fullcover," << rootHash << ',' << funcKey << ',';
    WriteCellMatchDesc(impl_->ofs, desc);
    impl_->ofs << ",1\n";
}

} // namespace npu::tile_fwk::dump
