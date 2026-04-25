// Shared helpers for per-op smoke probes that exercise a pypto-exported
// op_kernel_lib.cpython-*.so via ops::OpDefFactory.
//
// Each per-example custom_op_def_demo.cpp includes this header and
// delegates to ``op_probe::run_custom_op_def(argc, argv, opType, probe_fn)``.
// The probe_fn is a lambda that issues op-specific ``InvokeInferShape`` /
// ``InvokeInferDataType`` calls against the freshly-created OpDef.
//
// The name reflects the narrow scope of this driver: it only exercises
// the generated ``custom_op_def`` TU (OpDefFactory + InferShape /
// InferDataType). Other cpp-codegen surfaces (PrepareExecute / executor,
// onnx_plugin) are NOT driven here and would need their own runners.
//
// ``op_probe::run_custom_op_def`` does:
//   1) Embed libpython — the .so's InferShape/calcWorkspace implementations
//      delegate to embedded pybind11 snippets, so the interpreter must exist.
//   2) dlopen the .so with RTLD_LAZY — static ctors run (OP_ADD + REG_AUTO_MAPPING_OP),
//      but PrepareExecute-related symbols that the current CANN may not ship
//      remain unresolved (this probe never calls them).
//   3) Enumerate ops::OpDefFactory, confirm the op is registered.
//   4) ops::OpDefFactory::OpDefCreate(opType) — replays the generated
//      class ctor which wires InferShape / InferDataType callbacks.
//   5) Hand the OpDef to the caller's probe_fn.
//   6) Print a [DONE] banner.
//
// OpDefFactory::OpDefCreate / GetAllOp are declared *private* in the
// public header but exported from libregister.so as regular symbols (C++
// access control is compile-time only). We reopen them with the classic
// ``#define private public`` trick for a single TU — it is contained
// entirely within this header so consumers don't have to repeat it.
#pragma once

#include <dlfcn.h>
#include <Python.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// Pull in stdlib pieces the Ascend headers need BEFORE opening access, so
// the ``private`` rewrite only affects Ascend code.
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define private public
#define protected public
#include "register/op_def_registry.h"
#undef private
#undef protected

#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/shape.h"
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"

namespace op_probe {

// ---------- Minimal KernelRunContext plumbing --------------------------------
//
// gert::InferShapeContext / gert::InferDataTypeContext are zero-method wrappers
// over a POD KernelRunContext laid out as:
//   { input_size; output_size; compute_node_info; kernel_extend_info;
//     output_start; values[N]; }
// Each values[i] points to an AsyncAnyValue. For Shape (larger than a pointer)
// the AsyncAnyValue holds a pointer to the Shape; for ge::DataType (fits
// inline) the value is written into the inplace union directly.

struct CtxBuf {
    std::vector<uint8_t> storage;
    ::KernelRunContext *ctx;
};

inline CtxBuf MakeCtx(std::vector<AsyncAnyValue *> &values, size_t n_in, size_t n_out) {
    const size_t n = n_in + n_out;
    const size_t base = sizeof(::KernelRunContext);
    const size_t extra = (n > 1 ? (n - 1) * sizeof(AsyncAnyValue *) : 0);
    CtxBuf out;
    out.storage.assign(base + extra, 0);
    out.ctx = reinterpret_cast<::KernelRunContext *>(out.storage.data());
    out.ctx->input_size = n_in;
    out.ctx->output_size = n_out;
    out.ctx->compute_node_info = nullptr;
    out.ctx->kernel_extend_info = nullptr;
    out.ctx->output_start = nullptr;
    for (size_t i = 0; i < n; ++i) {
        out.ctx->values[i] = values[i];
    }
    return out;
}

inline std::string ShapeToStr(const gert::Shape &s) {
    std::string out = "(";
    for (size_t i = 0; i < s.GetDimNum(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    if (s.GetDimNum() == 1) out += ",";
    out += ")";
    return out;
}

inline const char *DataTypeName(ge::DataType dt) {
    switch (dt) {
        case ge::DT_FLOAT:     return "DT_FLOAT";
        case ge::DT_FLOAT16:   return "DT_FLOAT16";
        case ge::DT_BF16:      return "DT_BF16";
        case ge::DT_INT32:     return "DT_INT32";
        case ge::DT_INT64:     return "DT_INT64";
        case ge::DT_INT8:      return "DT_INT8";
        case ge::DT_BOOL:      return "DT_BOOL";
        case ge::DT_UNDEFINED: return "DT_UNDEFINED";
        default:               return "DT_<other>";
    }
}

// ---------- Generic N-input / M-output invokers -----------------------------

inline int InvokeInferShape(
    ops::OpDef &op,
    const std::vector<std::vector<int64_t>> &input_dims,
    size_t n_out) {
    auto infer = op.GetInferShape();
    if (infer == nullptr) {
        std::fprintf(stderr, "[FAIL] ops::OpDef::GetInferShape() returned nullptr\n");
        return -1;
    }
    const size_t n_in = input_dims.size();

    std::vector<gert::Shape> in_shapes(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        for (auto d : input_dims[i]) in_shapes[i].AppendDim(d);
    }
    std::vector<gert::Shape> out_shapes(n_out);

    const size_t total = n_in + n_out;
    std::vector<AsyncAnyValue> av(total);
    std::vector<AsyncAnyValue *> vals;
    vals.reserve(total);
    for (size_t i = 0; i < n_in; ++i) {
        av[i].data.pointer = &in_shapes[i];
        av[i].deleter = nullptr;
        vals.push_back(&av[i]);
    }
    for (size_t j = 0; j < n_out; ++j) {
        av[n_in + j].data.pointer = &out_shapes[j];
        av[n_in + j].deleter = nullptr;
        vals.push_back(&av[n_in + j]);
    }
    CtxBuf ctx_buf = MakeCtx(vals, n_in, n_out);
    auto status = infer(reinterpret_cast<gert::InferShapeContext *>(ctx_buf.ctx));

    std::string in_str, out_str;
    for (size_t i = 0; i < n_in; ++i) {
        if (i) in_str += ", ";
        in_str += ShapeToStr(in_shapes[i]);
    }
    for (size_t j = 0; j < n_out; ++j) {
        if (j) out_str += ", ";
        out_str += ShapeToStr(out_shapes[j]);
    }
    std::fprintf(stdout,
                 "[%s] InferShape(in=[%s]) -> status=%u out=[%s]\n",
                 status == ge::GRAPH_SUCCESS ? " OK " : "FAIL",
                 in_str.c_str(), status, out_str.c_str());
    return status == ge::GRAPH_SUCCESS ? 0 : -1;
}

inline int InvokeInferDataType(
    ops::OpDef &op,
    const std::vector<ge::DataType> &in_dts,
    size_t n_out) {
    auto infer = op.GetInferDataType();
    if (infer == nullptr) {
        std::fprintf(stderr, "[FAIL] ops::OpDef::GetInferDataType() returned nullptr\n");
        return -1;
    }
    const size_t n_in = in_dts.size();
    const size_t total = n_in + n_out;

    std::vector<AsyncAnyValue> av(total);
    std::vector<AsyncAnyValue *> vals;
    vals.reserve(total);
    for (size_t i = 0; i < n_in; ++i) {
        *reinterpret_cast<ge::DataType *>(av[i].data.inplace) = in_dts[i];
        av[i].deleter = nullptr;
        vals.push_back(&av[i]);
    }
    for (size_t j = 0; j < n_out; ++j) {
        *reinterpret_cast<ge::DataType *>(av[n_in + j].data.inplace) = ge::DT_UNDEFINED;
        av[n_in + j].deleter = nullptr;
        vals.push_back(&av[n_in + j]);
    }
    CtxBuf ctx_buf = MakeCtx(vals, n_in, n_out);
    auto status = infer(reinterpret_cast<gert::InferDataTypeContext *>(ctx_buf.ctx));

    std::string in_str, out_str;
    for (size_t i = 0; i < n_in; ++i) {
        if (i) in_str += ", ";
        in_str += DataTypeName(in_dts[i]);
    }
    for (size_t j = 0; j < n_out; ++j) {
        if (j) out_str += ", ";
        out_str += DataTypeName(*reinterpret_cast<ge::DataType *>(av[n_in + j].data.inplace));
    }
    std::fprintf(stdout,
                 "[%s] InferDataType(in=[%s]) -> status=%u out=[%s]\n",
                 status == ge::GRAPH_SUCCESS ? " OK " : "FAIL",
                 in_str.c_str(), status, out_str.c_str());
    return status == ge::GRAPH_SUCCESS ? 0 : -1;
}

inline void DumpRegisteredOps() {
    auto &all = ops::OpDefFactory::GetAllOp();
    std::fprintf(stdout,
                 "[INFO] ops::OpDefFactory::GetAllOp() -> %zu op(s) registered\n",
                 all.size());
    for (auto &n : all) {
        std::fprintf(stdout, "         - %s\n",
                     n.GetString() ? n.GetString() : "<null>");
    }
}

// ---------- Top-level custom_op_def probe driver ----------------------------

using ProbeFn = std::function<void(ops::OpDef &op)>;

inline int run_custom_op_def(int argc, char **argv, const char *op_type, const ProbeFn &probe_fn) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s <path-to-op_kernel_lib.cpython-*.so>\n",
                     argv[0]);
        return 2;
    }
    const char *so_path = argv[1];
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Py_Initialize();
    std::fprintf(stdout, "[ OK ] Py_Initialize (%s)\n", Py_GetVersion());

    void *handle = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        std::fprintf(stderr, "[FAIL] dlopen(%s) -> %s\n", so_path, dlerror());
        return 1;
    }
    std::fprintf(stdout, "[ OK ] dlopen(%s)\n", so_path);

    DumpRegisteredOps();
    bool present = false;
    for (auto &n : ops::OpDefFactory::GetAllOp()) {
        if (n.GetString() && std::strcmp(n.GetString(), op_type) == 0) {
            present = true;
            break;
        }
    }
    if (!present) {
        std::fprintf(stderr,
                     "[FAIL] %s missing from ops::OpDefFactory — OP_ADD didn't run\n",
                     op_type);
        return 1;
    }
    std::fprintf(stdout, "[ OK ] %s found in ops::OpDefFactory\n", op_type);

    ops::OpDef op = ops::OpDefFactory::OpDefCreate(op_type);
    std::fprintf(stdout,
                 "[ OK ] ops::OpDefFactory::OpDefCreate(\"%s\") returned an OpDef\n",
                 op_type);

    probe_fn(op);

    std::fprintf(stdout, "[DONE] %s .so exercised via ops::OpDefFactory\n", op_type);
    return 0;
}

}  // namespace op_probe
