// Softmax demo for op_kernel_lib.cpython-*.so
// (produced by load_and_compile.py against the softmax .onnx that
// ../export_demo.py writes out).
//
// Exercises the generated custom_op_def TU by invoking the InferShape /
// InferDataType callbacks it registers into ops::OpDefFactory.
// See ../../../op_probe_core.hpp for the probe-pattern walkthrough.
// This file supplies the op name and the per-op test vectors.
#include "op_probe_core.hpp"

namespace {
constexpr const char *opType = "SoftmaxPyptoCustomOp";
}

int main(int argc, char **argv) {
    return op_probe::run_custom_op_def(argc, argv, opType, [](ops::OpDef &op) {
        // Softmax: 1 input → 1 output.
        op_probe::InvokeInferShape(op, {{32, 32, 64}}, 1);
        op_probe::InvokeInferShape(op, {{1, 4, 64}}, 1);
        op_probe::InvokeInferShape(op, {{8}}, 1);
        op_probe::InvokeInferShape(op, {{2, 3, 4, 5}}, 1);

        op_probe::InvokeInferDataType(op, {ge::DT_FLOAT}, 1);
        op_probe::InvokeInferDataType(op, {ge::DT_FLOAT16}, 1);
        op_probe::InvokeInferDataType(op, {ge::DT_BF16}, 1);
    });
}
