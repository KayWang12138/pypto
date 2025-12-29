add_library(ir_proto)

target_compile_options(ir_proto PRIVATE -fPIC)

file(GLOB_RECURSE SRC_CPP_FILES
     CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

target_sources(ir_proto
    PRIVATE
        ${SRC_CPP_FILES}
)

target_include_directories(ir_proto 
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)