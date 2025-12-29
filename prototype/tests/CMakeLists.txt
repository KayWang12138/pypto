if (NOT PTO_ENABLE_TESTS)

    add_executable(main_test
        main_test.cpp
    )

    target_link_libraries(main_test
        PRIVATE ir_proto
    )

endif()

if (PTO_ENABLE_TESTS)

    find_package(GTest QUIET)
    message(STATUS "GTest found: ${GTest_FOUND}")

    if (GTest_FOUND)
        message(STATUS "Using system-installed GoogleTest: ${GTest_VERSION}")
    else()
        message(STATUS "System GoogleTest not found, fetching from GitHub")

        include(FetchContent)

        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
            GIT_TAG        v1.14.0
        )

        # Windows CRT 兼容（Linux 下无影响）
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)
    endif()

    # -----------------------
    # 测试目标
    # -----------------------
    add_executable(gtest_test
        gtest_ir_builder.cpp
    )
    target_link_libraries(gtest_test
        PRIVATE
            ir_proto
            GTest::gtest
            GTest::gtest_main
    )

    # -----------------------
    # CTest / gtest 集成
    # -----------------------
    include(GoogleTest)
    gtest_discover_tests(gtest_test)

endif()