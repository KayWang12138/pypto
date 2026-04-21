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
 * \file dump_host_topo.h
 * \brief Host-side static topology dumps used by the runtime dependency
 *        verification framework.
 */

#ifndef DUMP_HOST_TOPO_H
#define DUMP_HOST_TOPO_H

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace npu::tile_fwk {

struct TensorSlotManager;
struct IncastOutcastLink;
struct CceCodeInfo;

namespace dynamic {
struct DevAscendFunction;
struct DevCellMatchTableDesc;
} // namespace dynamic

namespace dump {

/// Dump slot_mapping.csv once per program. The mapping rows are derived from
/// `slotManager` (frontend slot indices, tensor names, function raw names)
/// and `slotIdxMapping` / `inoutLink` (runtime slot index, role).
void DumpSlotMapping(const TensorSlotManager& slotManager,
                     const std::unordered_map<int, int>& slotIdxMapping,
                     const IncastOutcastLink& inoutLink);

/// RAII writer for static_topo.csv. Opens the file (and emits the header) on
/// construction iff dump is enabled, closes on destruction. WriteFunction()
/// is called once per devRoot's encoded funcBin.
class StaticTopoCsvWriter {
public:
    StaticTopoCsvWriter();
    ~StaticTopoCsvWriter();

    StaticTopoCsvWriter(const StaticTopoCsvWriter&) = delete;
    StaticTopoCsvWriter& operator=(const StaticTopoCsvWriter&) = delete;

    bool Enabled() const;
    void WriteFunction(int devRootKey, dynamic::DevAscendFunction& funcBin,
                       const std::vector<CceCodeInfo>& cceCodeInfoList);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// RAII writer for slot_cell_table.csv. Opens the file iff dump is enabled
/// AND `fillContent` is true (real-encode pass, not the size-probing pass).
/// Caller emits one WritePartial() per partial-update slot and one
/// WriteFullCover() per (slot, root) for full-cover slots.
class SlotCellTableCsvWriter {
public:
    explicit SlotCellTableCsvWriter(bool fillContent);
    ~SlotCellTableCsvWriter();

    SlotCellTableCsvWriter(const SlotCellTableCsvWriter&) = delete;
    SlotCellTableCsvWriter& operator=(const SlotCellTableCsvWriter&) = delete;

    bool Enabled() const;
    void WritePartial(int slotIdx,
                      const dynamic::DevCellMatchTableDesc& desc,
                      size_t outcastCount);
    void WriteFullCover(int slotIdx, uint64_t rootHash, int funcKey,
                        const dynamic::DevCellMatchTableDesc& desc);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dump
} // namespace npu::tile_fwk

#endif // DUMP_HOST_TOPO_H
