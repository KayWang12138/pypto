# framework/src/machine/tracr/tracr.cmake

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
        ${CMAKE_SOURCE_DIR}/framework/src/machine/tracr/include
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
        set(TRACR_CAPACITY "" CACHE STRING "Optional TraCR buffer capacity (empty = use internal default)")

        if(NOT TRACR_CAPACITY STREQUAL "")
            message(WARNING "TraCR adding capacity: ${TRACR_CAPACITY}")

            if(NOT TRACR_CAPACITY MATCHES "^[0-9]+$")
                message(FATAL_ERROR "TRACR_CAPACITY must be a positive integer")
            endif()

            target_compile_definitions(${target} PRIVATE
                TRACR_CAPACITY=${TRACR_CAPACITY}
            )
        endif()

        # TraCR full size buffer modes:
        # default (none):            Abort if buffer is full
        # TRACR_POLICY_PERIODIC:     If buffer is full, overwrite from the beginning
        # TRACR_POLICY_STOP_IF_FULL: If buffer is full, ignore incoming traces
        # if (TRACR_POLICY)
        #     target_compile_definitions(${target} PRIVATE TRACR_POLICY_PERIODIC)
        # endif()
        set(TRACR_POLICY "" CACHE STRING "TraCR policy (empty = use C++ default)")

        set_property(CACHE TRACR_POLICY PROPERTY STRINGS
            ""  # allow empty (use C++ default)
            TRACR_POLICY_PERIODIC
            TRACR_POLICY_STOP_IF_FULL
        )

        if(NOT TRACR_POLICY STREQUAL "")
            if(TRACR_POLICY STREQUAL "TRACR_POLICY_PERIODIC")
                message(WARNING "TraCR adding policy: 'TRACR_POLICY_PERIODIC'")
                target_compile_definitions(${target} PRIVATE TRACR_POLICY_PERIODIC)
            elseif(TRACR_POLICY STREQUAL "TRACR_POLICY_STOP_IF_FULL")
                message(WARNING "TraCR adding policy: 'TRACR_POLICY_STOP_IF_FULL'")
                target_compile_definitions(${target} PRIVATE TRACR_POLICY_STOP_IF_FULL)
            else()
                message(FATAL_ERROR "Unknown TRACR_POLICY: ${TRACR_POLICY}")
            endif()
        else()
            message(STATUS "No TraCR policy given: using C++ default")
        endif()

        # Flag to enable TraCR debugging prints (TODO: Not yet working!)
        # if (TRACR_DEBUG)
        #     target_compile_definitions(${target} PRIVATE ENABLE_TRACR_DEBUG)
        # endif()
    endif()
endfunction()