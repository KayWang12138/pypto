#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_operation(py::module &m) {
    m.def(
        "add", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Add(left, right); }, "Tensor add.");
    m.def("sub", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Sub(left, right); }, "Tensor sub.");
    m.def("mul", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Mul(left, right); }, "Tensor mul.");
    m.def("div", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Div(left, right); }, "Tensor div.");
    m.def(
        "view",
        [](const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets) {
            return npu::tile_fwk::View(operand, shapes, offsets);
        },
        "Tensor view.");
    m.def("exp", [](const Tensor &operand) { return npu::tile_fwk::Exp(operand); }, "Tensor exp.");

    m.def(
        "transpose",
        [](const Tensor &operand, const std::vector<int> &transposeShape) {
            return npu::tile_fwk::Transpose(operand, transposeShape);
        },
        "Tensor transpose.");
    m.def("abs", [](const Tensor &operand) { return npu::tile_fwk::Abs(operand); }, "Tensor abs.");
    m.def("reciprocal", [](const Tensor &operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("sqrt", [](const Tensor &operand) { return npu::tile_fwk::Sqrt(operand); }, "Tensor sqrt.");

    m.def(
        "cast",
        [](const Tensor &operand, DataType new_data_type, CastMode mode) {
            return npu::tile_fwk::Cast(operand, new_data_type, mode);
        },
        py::arg("operand"), py::arg("new_data_type"), py::arg("mode") = CAST_NONE, "Tensor cast.");

    m.def(
        "add_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::AddS(left, right); },
        "Tensor add scalar.");
    m.def(
        "sub_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::SubS(left, right); },
        "Tensor sub scalar.");
    m.def(
        "mul_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::MulS(left, right); },
        "Tensor mul scalar.");
    m.def(
        "div_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::DivS(left, right); },
        "Tensor div scalar.");

    m.def(
        "row_max_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::RowMaxSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row max single.");

    m.def(
        "row_sum_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::RowSumSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row sum single.");
    m.def(
        "maximum", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Maximum(left, right); },
        py::arg("left"), py::arg("right"), "Tensor maximum.");
    m.def(
        "rms_norm", [](const Tensor &operand) { return npu::tile_fwk::RmsNorm(operand); }, py::arg("operand"),
        "Tensor rms norm.");

    m.def(
        "matmul",
        [](DataType out_type, const Tensor &a, const Tensor &b, bool a_trans, bool b_trans) {
            if (a_trans) {
                if (b_trans) {
                    return npu::tile_fwk::Matrix::Matmul<true, true>(out_type, a, b);
                } else {
                    return npu::tile_fwk::Matrix::Matmul<true, false>(out_type, a, b);
                }
            } else {
                if (b_trans) {
                    return npu::tile_fwk::Matrix::Matmul<false, true>(out_type, a, b);
                } else {
                    return npu::tile_fwk::Matrix::Matmul<false, false>(out_type, a, b);
                }
            }
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), py::arg("a_trans") = false, py::arg("b_trans") = false,
        "Matrix multiply.");

    m.def(
        "assemble",
        [](const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensor_int_pairs) {
            return npu::tile_fwk::Assemble(tensor_int_pairs);
        },
        "Tensor::Assemble");
}
} // namespace pypto