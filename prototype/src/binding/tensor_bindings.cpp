// PTO-IR prototype: Tensor bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/type.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to bind IR Tensor class directly to Python.
static void BindTensor(py::module_ &m) {
    py::class_<Tensor, Value, std::shared_ptr<Tensor>>(m, "Tensor")
        // Constructor with static integer shape: Tensor(dtype, shape, name, format)
        .def(py::init([](DataType dtype,
                         const std::vector<long long> &shape,
                         const std::string &name,
                         TileOpFormat format) {
            std::vector<size_t> staticShape(shape.begin(), shape.end());
            return std::make_shared<Tensor>(dtype, staticShape, name, format);
        }),
             py::arg("dtype"),
             py::arg("shape"),
             py::arg("name") = "",
             py::arg("format") = TileOpFormat::TILEOP_ND,
             "Construct tensor with static integer shape")
        // Constructor with symbolic shape: Tensor(dtype, shape, name, format)
        .def(py::init([](DataType dtype,
                         const std::vector<Scalar> &shape,
                         const std::string &name,
                         TileOpFormat format) {
            return std::make_shared<Tensor>(shape, dtype, name, format);
        }),
             py::arg("dtype"),
             py::arg("shape"),
             py::arg("name") = "",
             py::arg("format") = TileOpFormat::TILEOP_ND,
             "Construct tensor with symbolic shape")
        // Properties from Value base class
        .def("GetDataType", &Tensor::GetDataType,
             "Get the data type of the tensor")
        .def("GetName", &Tensor::GetName,
             py::return_value_policy::reference_internal,
             "Get the name of the tensor")
        .def("SetName", &Tensor::SetName,
             py::arg("name"),
             "Set the name of the tensor")
        // Tensor-specific properties
        .def("GetShape", &Tensor::GetShape,
             py::return_value_policy::reference_internal,
             "Get the shape of the tensor as vector of Scalars")
        .def("Dim", [](const Tensor &t) { 
            return static_cast<int>(t.GetShape().size()); 
        },
             "Get the number of dimensions")
        .def("GetFormat", &Tensor::GetFormat,
             "Get the tile operation format (ND, NZ, etc.)")
        .def("SetFormat", &Tensor::SetFormat,
             py::arg("format"),
             "Set the tile operation format")
        .def("Format", [](const Tensor &t) {
            return static_cast<int>(t.GetFormat());
        },
             "Get format as integer (for compatibility)")
        // Python runtime ID for debugging
        .def("Id", [](const Tensor &t) -> int {
            return t.GetID();
        },
             "Get runtime ID for debugging")
        // Move semantics: copy shared_ptr
        .def("Move", [](std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            // In Python, we just reassign the shared_ptr (handled by pybind11)
            // This is for compatibility with existing Python code
            *self = *other;
        },
             py::arg("other"),
             "Move/copy tensor data from another tensor")
        // Print function
        .def("Print", [](const Tensor &t) {
            std::ostringstream os;
            t.Print(os, 0);
            return os.str();
        },
             "Print tensor representation")
        .def("__repr__", [](const Tensor &t) {
            std::ostringstream os;
            os << "Tensor(shape=[";
            const auto& shape = t.GetShape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) os << ", ";
                shape[i].Print(os, 0);
            }
            os << "], dtype=" << DataTypeToString(t.GetDataType());
            os << ", name=\"" << t.GetName() << "\")";
            return os.str();
        });
}

// Stubbed helpers for scalar/tensor interaction used by python/pypto/tensor.py.
static void SetTensorData(const Scalar &/*value*/,
                          const std::vector<long long> &/*indices*/,
                          std::shared_ptr<Tensor> /*tensor*/) {
    // TODO: Implement IR operation to set tensor element.
}

static Scalar GetTensorData(const std::shared_ptr<Tensor> &tensor,
                            const std::vector<long long> &indices) {
    (void)indices;
    if (!tensor) return Scalar(int64_t{0});
    const auto &shape = tensor->GetShape();
    if (shape.empty()) {
        return Scalar(int64_t{0});
    }
    // TODO: Implement proper IR read; currently return a dummy scalar.
    return Scalar(int64_t{0});
}

static Scalar GetInputShape(const std::shared_ptr<Tensor> &tensor, int dim) {
    if (!tensor) throw std::runtime_error("GetInputShape: tensor is null");
    const auto &shape = tensor->GetShape();
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size()) {
        throw std::out_of_range("dimension index out of range in GetInputShape");
    }
    // shape is now std::vector<Scalar>, so return the Scalar directly
    return shape[static_cast<size_t>(dim)];
}

// Main tensor bindings function
void BindTensorBindings(py::module_ &m) {
    // Tensor class used by python/pypto/tensor.py as the backing implementation.
    BindTensor(m);

    // Scalar/tensor interaction helpers expected by python/pypto/tensor.py.
    m.def("SetTensorData", &SetTensorData,
          py::arg("value"), py::arg("indices"), py::arg("tensor"));
    m.def("GetTensorData", &GetTensorData,
          py::arg("tensor"), py::arg("indices"));
    m.def("GetInputShape", &GetInputShape,
          py::arg("tensor"), py::arg("dim"));
}

}  // namespace pto

