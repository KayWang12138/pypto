add_library(ir_proto SHARED)

target_compile_options(ir_proto PRIVATE -fPIC)
target_compile_definitions(ir_proto 
    PUBLIC
        _GLIBCXX_USE_CXX11_ABI=0
)

file(GLOB_RECURSE IR_FILES
     CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

target_sources(ir_proto
    PRIVATE
        ${IR_FILES}
)

target_include_directories(ir_proto 
    PUBLIC
        ${PTO_FWK_SRC_ROOT}/framework/include
)
