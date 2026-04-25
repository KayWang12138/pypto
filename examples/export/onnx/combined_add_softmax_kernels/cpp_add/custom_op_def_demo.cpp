// Add-side demo for the combined Add+Softmax model.
//
// ../export_demo.py writes ONE .onnx containing both AddPyptoCustomOp and
// SoftmaxPyptoCustomOp. This dir targets AddPyptoCustomOp; ../cpp_softmax/
// targets SoftmaxPyptoCustomOp. Each dir invokes load_and_compile.py with
// its own --op-type, producing a separate .so per op (load_and_compile.py
// extracts one custom op at a time).
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
