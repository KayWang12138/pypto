# cmake/third_party/tracr/tracr.cmake

function(tracr_enable target)
    message(STATUS "Enabling TraCR for target: ${target}")

    if (NOT TARGET ${target})
        message(FATAL_ERROR "Target '${target}' does not exist.")
    endif()

    # Make sure json is available first
    include(${CMAKE_SOURCE_DIR}/cmake/third_party/nlohmann_json/nlohmann_json.cmake)

    # Link tracr with json
    target_link_libraries(${target} PRIVATE json)

    # Create the TraCR include directory path
    set(TRACR_INCLUDE_DIR
        ${CMAKE_SOURCE_DIR}/cmake/third_party/tracr/include
    )

    # Check if it even exists
    if (NOT EXISTS "${TRACR_INCLUDE_DIR}/tracr/tracr.hpp")
        message(FATAL_ERROR
            "tracr.hpp not found at ${TRACR_INCLUDE_DIR}/tracr/tracr.hpp"
        )
    endif()

    # --- include the directories ---
    target_include_directories(${target} PRIVATE
        ${TRACR_INCLUDE_DIR}
    )

    # --- compiler flags of TraCR ---
    if (BUILD_TRACR)
        # Flag to enable/disable TraCR calls at compile time
        target_compile_definitions(${target} PRIVATE ENABLE_TRACR)

        # TraCR threads capacity (default is 1<<20 ~= 1 million traces per thread = ~17MB per thread buffer size)
        # if (TRACR_CAPACITY)
        #     target_compile_definitions(${target} PRIVATE TRACR_CAPACITY="(1<<20)")
        # endif()

        # TraCR full size buffer modes:
        # default (none):            Abort if buffer is full
        # TRACR_POLICY_PERIODIC:     If buffer is full, overwrite from the beginning
        # TRACR_POLICY_STOP_IF_FULL: If buffer is full, ignore incoming traces
        # if (TRACR_POLICY)
        #     target_compile_definitions(${target} PRIVATE TRACR_POLICY_PERIODIC)
        # endif()

        # Flag to enable TraCR debugging prints (Not yet working!)
        # if (TRACR_DEBUG)
        #     target_compile_definitions(${target} PRIVATE ENABLE_TRACR_DEBUG)
        # endif()
    endif()
endfunction()