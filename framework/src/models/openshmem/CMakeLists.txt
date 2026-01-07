# OpenSHMEM + NPU 构建配置

set(COMMON_COMPILE_OPTIONS
    -fPIC
    -std=c++17
    -Wall
    -Werror
    -O2
)

# OpenSHMEM核心库
add_library(openshmem_npu STATIC
    openshmem_npu.h
    openshmem_npu.cpp
    openshmem_pto_integration.h
    openshmem_pto_integration.cpp
    openshmem_network_topology.cpp
)

target_include_directories(openshmem_npu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/framework/src/interface
    ${CMAKE_SOURCE_DIR}/framework/src/cost_model
    ${CMAKE_SOURCE_DIR}/framework/src/models/comet
    /usr/local/Ascend/ascend-toolkit/latest/include
)

target_compile_options(openshmem_npu PRIVATE ${COMMON_COMPILE_OPTIONS})

target_link_libraries(openshmem_npu PUBLIC
    tile_fwk_simulation
    tile_fwk_simulation_ca
    comet_model
)

# OpenSHMEM演示程序
add_executable(openshmem_npu_demo
    ../openshmem_npu_demo.cpp
)

target_include_directories(openshmem_npu_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/framework/src/interface
    ${CMAKE_SOURCE_DIR}/framework/src/models/comet
)

target_compile_options(openshmem_npu_demo PRIVATE ${COMMON_COMPILE_OPTIONS})

target_link_libraries(openshmem_npu_demo PRIVATE
    openshmem_npu
    comet_model
)

# 安装配置
install(TARGETS openshmem_npu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS openshmem_npu_demo
    RUNTIME DESTINATION bin
)

install(FILES
    openshmem_npu.h
    openshmem_pto_integration.h
    DESTINATION include/openshmem
)
