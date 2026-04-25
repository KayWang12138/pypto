// Add demo for op_kernel_lib.cpython-*.so
// (produced by load_and_compile.py against the add .onnx that
// ../export_demo.py writes out).
//
// Exercises the generated custom_op_def TU by invoking the InferShape /
// InferDataType callbacks it registers into ops::OpDefFactory.
// See ../../../op_probe_core.hpp for the probe-pattern walkthrough.
// This file supplies the op name and the per-op test vectors.
#include "op_probe_core.hpp"

namespace {
constexpr const char *opType = "AddPyptoCustomOp";
}

int main(int argc, char **argv) {
    return op_probe::run_custom_op_def(argc, argv, opType, [](ops::OpDef &op) {
        // Add: 2 inputs → 1 output. SHAPE = (32, 32, 1, 64) per export_demo.py.
        op_probe::InvokeInferShape(op, {{32, 32, 1, 64}, {32, 32, 1, 64}}, 1);
        op_probe::InvokeInferShape(op, {{1, 4, 1, 64}, {1, 4, 1, 64}}, 1);

        op_probe::InvokeInferDataType(op, {ge::DT_FLOAT16, ge::DT_FLOAT16}, 1);
        op_probe::InvokeInferDataType(op, {ge::DT_FLOAT, ge::DT_FLOAT}, 1);
    });
}
