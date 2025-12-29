add_library(pass_proto)

target_compile_options(pass_proto PRIVATE -fPIC)

file(GLOB_RECURSE SRC_CPP_FILES
     CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

target_sources(pass_proto
    PRIVATE
        ${SRC_CPP_FILES}
)

target_include_directories(pass_proto 
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Link with ir_proto since pass depends on IR structures
target_link_libraries(pass_proto
    PUBLIC
        ir_proto
)

