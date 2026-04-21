#!/usr/bin/env bash
# Self-contained patcher for a local Ascend CANN install whose headers
# predate the "sinkable op" API. Run it ONCE on a machine whose CANN
# install lacks these headers, before trying to build pypto-exported
# custom ops via examples/export/load_and_compile.py.
#
# Three files must be present for the generated cpp to compile:
#   exe_graph/runtime/sinkable_op_execution_context.h
#   exe_graph/runtime/op_compile_context.h
#   graph/custom_op.h  (must contain SinkableExecuteOp)
#
# Behavior:
#   * For the first two: written if missing, left alone if already present.
#   * For custom_op.h: if the existing file already references
#     SinkableExecuteOp it's left alone; otherwise backed up to
#     <file>.pypto_pre_patch_bak and replaced with the new-API version.
#
# No args, no external files. Detects the Ascend install via
# $ASCEND_HOME_PATH, $ASCEND_INSTALL_PATH, or /usr/local/Ascend/cann*.
# Needs write access under <install>/<arch>/include. Re-run with sudo if
# the install is root-owned.
set -euo pipefail

log() { printf '%s\n' "$*" >&2; }

# ---------- detect Ascend install ------------------------------------------

detect_ascend_root() {
    if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
        printf '%s' "${ASCEND_HOME_PATH}"; return
    fi
    if [[ -n "${ASCEND_INSTALL_PATH:-}" && -d "${ASCEND_INSTALL_PATH}" ]]; then
        printf '%s' "${ASCEND_INSTALL_PATH}"; return
    fi
    if [[ -d /usr/local/Ascend/cann ]]; then
        printf '%s' /usr/local/Ascend/cann; return
    fi
    local cand
    cand="$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -n "${cand}" && -d "${cand}" ]]; then
        printf '%s' "${cand}"; return
    fi
    log "[FAIL] Could not auto-detect Ascend install."
    log "       Set \$ASCEND_HOME_PATH or install CANN under /usr/local/Ascend/."
    exit 1
}

resolve_arch_subdir() {
    local root="$1"
    local arch
    # Prefer a subdir matching the host arch, then fall back to any that exists.
    local host="$(uname -m 2>/dev/null || echo unknown)"
    case "${host}" in
        aarch64) arch_order=("aarch64-linux" "x86_64-linux") ;;
        x86_64)  arch_order=("x86_64-linux" "aarch64-linux") ;;
        *)       arch_order=("aarch64-linux" "x86_64-linux") ;;
    esac
    for arch in "${arch_order[@]}"; do
        if [[ -d "${root}/${arch}/include" ]]; then
            printf '%s' "${root}/${arch}"; return
        fi
    done
    if [[ -d "${root}/include" ]]; then
        printf '%s' "${root}"; return
    fi
    log "[FAIL] No include/ subdir under ${root}"
    exit 1
}

ASCEND_ROOT="$(detect_ascend_root)"
ASCEND_HOME="$(resolve_arch_subdir "${ASCEND_ROOT}")"
INC="${ASCEND_HOME}/include"
log "[INFO] Ascend install: ${ASCEND_ROOT}"
log "[INFO] Arch subtree:   ${ASCEND_HOME}"

if [[ ! -w "${INC}" ]]; then
    log "[WARN] ${INC} is not writable as current user;"
    log "       re-run this script with sudo if any write step fails."
fi

# ---------- embedded header contents ---------------------------------------
#
# Three headers, inlined verbatim. Do not edit below this line — the
# contents are what gets written into the Ascend install.

SINKABLE_OP_EXECUTION_CONTEXT_H=$(cat <<'__SINKABLE_OP_EXECUTION_CONTEXT_H__'
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SINKABLE_OP_EXECUTION_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SINKABLE_OP_EXECUTION_CONTEXT_H_

#include <type_traits>
#include <memory>
#include "tensor.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
using rtStream = void *;
enum class SinkableOpIo : uint32_t {
  kInput = 0,
  kOutput,
};
class SinkableOpExecutionContext : public ExtendedKernelContext {
 public:
  /**
   * 获取所属的执行流
   * @return
   */
  rtStream GetStream() const;

  /**
   * 根据输入index，获取输入shape指针
   * @param index 输入index
   * @return 输入shape指针，index非法时，返回空指针
   */
  const Shape *GetInputShape(const size_t index) const {
    if (index >= GetComputeNodeInputNum()) {
      return nullptr;
    }
    return GetInputPointer<Shape>(index);
  }
  /**
   * 根据输入index，获取输出tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const Tensor *GetInputTensor(const size_t index) const {
    if (index >= GetComputeNodeInputNum()) {
      return nullptr;
    }
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor *GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Shape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape指针，index或relative_index非法时，返回空指针
   */
  const Shape *GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return Tensor指针，index非法时，返回空指针
   */
  const Tensor *GetRequiredInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Shape *GetRequiredInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 根据输出index，获取输出shape指针
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const Shape *GetOutputShape(const size_t index) {
    const size_t output_num = GetComputeNodeOutputNum();
    if (index >= output_num) {
      return nullptr;
    }
    const size_t input_num = GetComputeNodeInputNum();
    return GetInputPointer<Shape>(input_num + index);
  }

  /**
   * 获取index指定的输出Tensor指针
   * @param index 输出索引
   * @return 输出Tensor指针，异常时返回空指针
   */
  const Tensor *GetOutputTensor(size_t index) const {
    const size_t output_num = GetComputeNodeOutputNum();
    if (index >= output_num) {
      return nullptr;
    }
    const size_t input_num = GetComputeNodeInputNum();
    return GetInputPointer<Tensor>(input_num + index);
  }

  /**
   * 分配workspace内存，placement为device
   * @param size 内存大小，单位为字节
   * @return 地址指针，异常时返回空指针
   * 生命周期：内存由context构造方管理，接口调用者不需要主动释放
   */
  void *MallocWorkSpace(size_t size);

  void *RegisterBin(const void *data, size_t size) {
    // todo
    (void)data;
    (void)size;
    return nullptr;
  }

  ge::graphStatus SpecifyIoOffset(SinkableOpIo type, size_t offset[], size_t offset_num);
  void *HostArgsToDevice(const void* host_args, size_t arg_size);

  ge::graphStatus SetBlockDim(uint32_t block_dim);
  uint32_t GetBlockDim() const;

  /**
   * 注册kernel bin地址与大小
   * 注意：context不持有该内存，调用方需保证data在context使用期间有效
   */
  ge::graphStatus SetKernelBin(std::shared_ptr<const void> data, uint32_t size);
  const uint8_t *GetKernelBinAddr() const;
  uint32_t GetKernelBinSize() const;
  std::vector<uint64_t> GetIoAddrs() const;

 private:
  enum class AdditionalInputIndex : uint32_t {
    kDeviceAllocator = 0,
    kStream,
    // add new extend output here
    kNum
  };

  enum class AdditionalOutputIndex : uint32_t {
    kWorkSpace, // can be delete
    KKernelArgs,
    // add new extend output here
    kNum
  };
};

static_assert(std::is_standard_layout<SinkableOpExecutionContext>::value,
              "The class SinkableOpExecutionContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SINKABLE_OP_EXECUTION_CONTEXT_H_
__SINKABLE_OP_EXECUTION_CONTEXT_H__
)

OP_COMPILE_CONTEXT_H=$(cat <<'__OP_COMPILE_CONTEXT_H__'
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_OP_COMPILE_CONTEXT_H
#define CANN_GRAPH_ENGINE_OP_COMPILE_CONTEXT_H

#include "exe_graph/runtime/extended_kernel_context.h"
#include "platform/platform_infos_def.h"

namespace gert {

class OpCompileContext : public ExtendedKernelContext {
 public:
  ge::graphStatus GetOption(const ge::AscendString &option_key, ge::AscendString &option) const;
  ge::graphStatus GetPlatformInfos(fe::PlatFormInfos &platform_info, fe::OptionalInfos& optional_infos) const;
};

}  // namespace gert
#endif  // CANN_GRAPH_ENGINE_OP_COMPILE_CONTEXT_H
__OP_COMPILE_CONTEXT_H__
)

CUSTOM_OP_H=$(cat <<'__CUSTOM_OP_H__'
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
#define METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/sinkable_op_execution_context.h"
#include "exe_graph/runtime/op_compile_context.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "graph/ge_error_codes.h"

namespace ge {
class BaseCustomOp {
 public:
  virtual graphStatus PrepareExecute(gert::SinkableOpExecutionContext *ctx) = 0;
  virtual ~BaseCustomOp() = default;
};
class CompilableOp : virtual public BaseCustomOp {
  public:
  ~CompilableOp() override = default;
  /**
   * CompilableOp类自定义OP的算子编译函数
   * @param ctx 算子编译上下文
   * @return 状态码
   */
  virtual graphStatus Compile(gert::OpCompileContext *ctx) = 0;
};

class EagerExecuteOp : virtual public BaseCustomOp {
 public:
  /**
   * Eager类自定义OP的执行函数
   * @param ctx 执行时上下文，可通过上下文获取input tensor，分配输出内存，分配workspace等
   * @return 状态码
   */
  virtual graphStatus Execute(gert::EagerOpExecutionContext *ctx) = 0;
};

class SinkableExecuteOp : virtual public BaseCustomOp {
 public:
  /**
   * sinkable类自定义OP的执行函数
   * @param ctx。执行时上下文，可通过上下文获取input tensor，分配输出内存，分配workspace等
   * @return 状态码
   */
  virtual graphStatus PrepareExecute(gert::SinkableOpExecutionContext *ctx) = 0;
};

using BaseOpCreator = std::function<std::unique_ptr<BaseCustomOp>()>;

class CustomOpCreatorRegister {
public:
  CustomOpCreatorRegister(const AscendString &operator_type, const BaseOpCreator &op_creator);
  ~CustomOpCreatorRegister() = default;
};
}  // namespace ge

#define REG_JOIN(g_register, y) g_register##y
#define REG_AUTO_MAPPING_OP(custom_op_class) REG_AUTO_MAPPING_OP_UNIQ(__COUNTER__, custom_op_class)
#define REG_AUTO_MAPPING_OP_UNIQ(ctr, custom_op_class)             \
  static const ge::CustomOpCreatorRegister REG_JOIN(custom_op_register, ctr)( \
      #custom_op_class, []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<custom_op_class>(); })

#endif  // METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
__CUSTOM_OP_H__
)

# ---------- patch helpers --------------------------------------------------

write_if_missing() {
    # $1 = relative path under include/
    # $2 = content variable name
    local rel="$1" var="$2" dst="${INC}/${1}"
    if [[ -f "${dst}" ]]; then
        log "[SKIP] ${rel} already exists"
        return
    fi
    mkdir -p "$(dirname "${dst}")"
    printf '%s\n' "${!var}" > "${dst}"
    log "[OK  ] wrote ${dst}"
}

write_if_marker_missing() {
    # $1 = relative path
    # $2 = marker (grep regex) indicating the file is already up-to-date
    # $3 = content variable name
    local rel="$1" marker="$2" var="$3" dst="${INC}/${1}"
    if [[ -f "${dst}" ]] && grep -q -- "${marker}" "${dst}"; then
        log "[SKIP] ${rel} already has new API (grep: ${marker})"
        return
    fi
    mkdir -p "$(dirname "${dst}")"
    if [[ -f "${dst}" ]]; then
        cp -p "${dst}" "${dst}.pypto_pre_patch_bak"
        log "[INFO] backed up existing ${rel} to ${rel}.pypto_pre_patch_bak"
    fi
    printf '%s\n' "${!var}" > "${dst}"
    log "[OK  ] wrote ${dst}"
}

# ---------- apply ----------------------------------------------------------

write_if_missing "exe_graph/runtime/sinkable_op_execution_context.h" SINKABLE_OP_EXECUTION_CONTEXT_H
write_if_missing "exe_graph/runtime/op_compile_context.h"            OP_COMPILE_CONTEXT_H
write_if_marker_missing "graph/custom_op.h" "SinkableExecuteOp"      CUSTOM_OP_H

log "[DONE] Ascend headers patched under ${INC}."
