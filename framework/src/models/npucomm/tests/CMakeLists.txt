#
# NPUCOMM Tests CMakeLists.txt
#

# Basic functionality tests
add_executable(npucomm_basic_test npucomm_basic_test.cpp)
target_link_libraries(npucomm_basic_test PRIVATE npucomm)
target_compile_features(npucomm_basic_test PRIVATE cxx_std_17)

# Memory management tests
add_executable(npucomm_memory_test npucomm_memory_test.cpp)
target_link_libraries(npucomm_memory_test PRIVATE npucomm)
target_compile_features(npucomm_memory_test PRIVATE cxx_std_17)

# Communication pattern tests
add_executable(npucomm_comm_test npucomm_comm_test.cpp)
target_link_libraries(npucomm_comm_test PRIVATE npucomm)
target_compile_features(npucomm_comm_test PRIVATE cxx_std_17)

# Synchronization tests
add_executable(npucomm_sync_test npucomm_sync_test.cpp)
target_link_libraries(npucomm_sync_test PRIVATE npucomm)
target_compile_features(npucomm_sync_test PRIVATE cxx_std_17)

# Atomic operations tests
add_executable(npucomm_atomic_test npucomm_atomic_test.cpp)
target_link_libraries(npucomm_atomic_test PRIVATE npucomm)
target_compile_features(npucomm_atomic_test PRIVATE cxx_std_17)

# Performance regression tests
add_executable(npucomm_perf_test npucomm_perf_test.cpp)
target_link_libraries(npucomm_perf_test PRIVATE npucomm)
target_compile_features(npucomm_perf_test PRIVATE cxx_std_17)

# Install tests
install(TARGETS
    npucomm_basic_test
    npucomm_memory_test
    npucomm_comm_test
    npucomm_sync_test
    npucomm_atomic_test
    npucomm_perf_test
    RUNTIME DESTINATION bin/tests
)

# Add test target
add_test(NAME npucomm_basic_test COMMAND npucomm_basic_test)
add_test(NAME npucomm_memory_test COMMAND npucomm_memory_test)
add_test(NAME npucomm_comm_test COMMAND npucomm_comm_test)
add_test(NAME npucomm_sync_test COMMAND npucomm_sync_test)
add_test(NAME npucomm_atomic_test COMMAND npucomm_atomic_test)
add_test(NAME npucomm_perf_test COMMAND npucomm_perf_test)
