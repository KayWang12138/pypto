// Attention demo for op_kernel_lib.cpython-*.so
// (produced by load_and_compile.py against the attention .onnx that
// ../export_demo.py writes out).
//
// Exercises the generated custom_op_def TU by invoking the InferShape /
// InferDataType callbacks it registers into ops::OpDefFactory. The op
// takes 3 inputs (Q, K, V) and produces 1 output.
// See ../../../op_probe_core.hpp for the probe-pattern walkthrough.
// This file supplies the op name and the per-op test vectors.
#include "op_probe_core.hpp"

namespace {
constexpr const char *opType = "AttentionPyptoCustomOp";
}

int main(int argc, char **argv) {
    return op_probe::run_custom_op_def(argc, argv, opType, [](ops::OpDef &op) {
        // Attention: 3 inputs (Q, K, V) → 1 output.
        // ATTN_SHAPE = (BATCH=2, HEADS=8, SEQ=16, HEAD_DIM=64), all BF16.
        op_probe::InvokeInferShape(op, {{2, 8, 16, 64}, {2, 8, 16, 64}, {2, 8, 16, 64}}, 1);

        op_probe::InvokeInferDataType(op, {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}, 1);
    });
}
