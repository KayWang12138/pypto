#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_tensor(py::module_ &m){
    py::class_<Tensor>(m, "tensor")
        .def(py::init<DataType, std::vector<int>, std::string>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "Unknown")
        .def(
            "__add__", [](Tensor &self, Tensor tensor) { return npu::tile_fwk::Add(self, tensor); }, "Tensor add.")
        .def("get_dtype", &Tensor::GetDataType)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("get_shape", py::overload_cast<>(&Tensor::GetShape, py::const_),
            py::return_value_policy::reference_internal)
        .def("assign",
            py::overload_cast<const Tensor&>(&Tensor::operator=),
            "Assigns from another tensor by copying its content.",
            py::return_value_policy::reference_internal
        )
        .def("move",
            [](Tensor &self, Tensor &other) -> Tensor& {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("other"),
            py::return_value_policy::reference_internal
        );

    py::class_<Element>(m, "element")
        .def(py::init<DataType, int64_t>(), py::arg("type"), py::arg("sData"))
        .def(py::init<DataType, uint64_t>(), py::arg("type"), py::arg("uData"))
        .def(py::init<DataType, double>(), py::arg("type"), py::arg("fData"))
        .def("get_data_type", &Element::GetDataType)
        .def("get_signed_data", &Element::GetSignedData)
        .def("get_unsigned_data", &Element::GetUnsignedData)
        .def("get_float_data", &Element::GetFloatData);
}
}