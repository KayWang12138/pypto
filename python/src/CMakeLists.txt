# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

pybind11_add_module(pypto_impl
        MODULE
            pybind11.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/common/vk_status.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/ir/gpu_vk_tensor_ir.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/ir/gpu_vk_dispatch_ir.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/ir/gpu_vk_shader_ir.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/gpu_vk_pass_manager.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/constant_fold.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/layout_normalize.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/broadcast_legalize.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/op_fusion.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/tile_lowering.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/dispatch_lowering.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/fusion_planner.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/passes/shared_memory_planner.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_instance.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_device.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_buffer.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_pipeline.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_launcher.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_allocator.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_descriptor_cache.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_pipeline_cache.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_executable.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_runner.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/vk_tensor_storage.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/runtime/torch_vk_interop.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/cache/shape_cache.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/cache/artifact_cache.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/codegen/shader_meta.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/codegen/glsl_codegen.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/codegen/spirv_compiler.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/codegen/gpu_vk_artifact.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/profile/vk_profiler.cpp
            ${PROJECT_SOURCE_DIR}/framework/src/gpu_vk/profile/workgroup_tuner.cpp
            bindings/torch_tensor_converter.cpp
            bindings/enum.cpp
            bindings/element.cpp
            bindings/tensor.cpp
            bindings/symbolic_scalar.cpp
            bindings/controller.cpp
            bindings/operation.cpp
            bindings/runtime.cpp
            bindings/gpu_vk_runtime.cpp
            bindings/gpu_vk_ir.cpp
            bindings/gpu_vk_codegen.cpp
            bindings/torch_vk_interop.cpp
            bindings/gpu_vk_profile.cpp
            bindings/cost_model.cpp
            bindings/pass_config.cpp
            bindings/function.cpp
            bindings/distributed.cpp
            bindings/platform.cpp
)
target_include_directories(pypto_impl
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            ${PROJECT_SOURCE_DIR}/framework/src
)
target_compile_definitions(pypto_impl
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_VULKAN}>:BUILD_WITH_VULKAN>>
            $<BUILD_INTERFACE:$<$<BOOL:${GPU_VK_ENABLE_VALIDATION}>:GPU_VK_ENABLE_VALIDATION>>
            $<BUILD_INTERFACE:$<$<BOOL:${GPU_VK_ENABLE_SHADER_CACHE}>:GPU_VK_ENABLE_SHADER_CACHE>>
)

# 所依赖的本项目 so 均在 pypto 包 __init__.py 中加载, 此处不再 link
target_link_libraries(pypto_impl
        PRIVATE
            json
            c_sec
            intf_pub_cxx17
            tile_fwk_intf_pub
)
if (TARGET c_sec_shared)
    add_dependencies(pypto_impl c_sec_shared)
endif ()
