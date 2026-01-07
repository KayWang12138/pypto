#
# NPUCOMM Examples CMakeLists.txt
#

# Simple standalone test (no external dependencies)
add_executable(simple_test simple_test.cpp)
target_compile_features(simple_test PRIVATE cxx_std_17)

# NVSHMEM-compatible example
add_executable(nvshmem_example nvshmem_example.cpp)
target_link_libraries(nvshmem_example PRIVATE npucomm)
target_compile_features(nvshmem_example PRIVATE cxx_std_17)

# Basic communication example (requires full NPUCOMM library)
if(TARGET npucomm)
    add_executable(basic_comm_example basic_comm_example.cpp)
    target_link_libraries(basic_comm_example PRIVATE npucomm)
    target_compile_features(basic_comm_example PRIVATE cxx_std_17)

    # Fine-grain communication example
    add_executable(fine_grain_example fine_grain_example.cpp)
    target_link_libraries(fine_grain_example PRIVATE npucomm)
    target_compile_features(fine_grain_example PRIVATE cxx_std_17)

    # PGAS memory model example
    add_executable(pgas_memory_example pgas_memory_example.cpp)
    target_link_libraries(pgas_memory_example PRIVATE npucomm)
    target_compile_features(pgas_memory_example PRIVATE cxx_std_17)

    # Performance benchmark example
    add_executable(perf_benchmark perf_benchmark.cpp)
    target_link_libraries(perf_benchmark PRIVATE npucomm)
    target_compile_features(perf_benchmark PRIVATE cxx_std_17)
endif()

# Install examples
install(TARGETS simple_test nvshmem_example
    RUNTIME DESTINATION bin/examples
)

if(TARGET npucomm)
    install(TARGETS
        basic_comm_example
        fine_grain_example
        pgas_memory_example
        perf_benchmark
        RUNTIME DESTINATION bin/examples
    )
endif()