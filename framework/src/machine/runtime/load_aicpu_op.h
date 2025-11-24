/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file load_aicpu_op.h
 * \brief
 */

 
#include <string>
#include <vector>
#include <memory>
#include "runtime/mem.h"
#include "machine/utils/machine_ws_intf.h"
#include "rts/rts_kernel.h"
#include <unordered_map>

struct BatchLoadOpFromBufArgs {
  uint32_t soNum;
  uint64_t args;
} __attribute__((packed));

struct CustAicpuSoBuf {
  uint64_t kernelSoBuf;
  uint32_t kernelSoBufLen;
  uint64_t kernelSoName;
  uint32_t kernelSoNameLen;
} __attribute__((packed));

struct OpKernelBin
{
    std::string name_;
    std::vector<char> data_;
    OpKernelBin(const std::string &name, std::vector<char> &data) : name_(name),
                 data_(data) {}
    const std::string &GetName() const { return name_; }
    const uint8_t *GetBinData() const { return reinterpret_cast<const uint8_t*>(data_.data()); }
    size_t GetBinDataSize() const { return data_.size(); }
};

using customKernelBinPtr = std::shared_ptr<OpKernelBin>;
namespace npu::tile_fwk {
class LoadAicpuOp
{
private:
    void SetAiCpuKernel();  
    void LoadCustomAicpuSo(const void *args, rtStream_t stream);
    customKernelBinPtr customKerBin_;
    rtFuncHandle funcHandle_;
    void *customBinHandle_ = nullptr;
    std::string builtInOpJsonPath_;
    std::unordered_map<std::string, rtFuncHandle> builtInFuncMap_;
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp() {};
    int LaunchBuiltInOp(rtStream_t stream, AstKernelArgs *kArgs, const int &aicpuNum, const std::string &funcName);
    int GetBuiltInOpBinHandle();
    int LaunchCustomOp(rtStream_t stream, AstKernelArgs *kArgs, std::string &OpType);
    void CustomAiCpuSoLoad();
    void GenBuiltInOpInfo(const std::string &jsonPath);
    static LoadAicpuOp &GetInstance() {
      static LoadAicpuOp loadCustomAicpuOp;
      return loadCustomAicpuOp;
    }
};

} // namespace
