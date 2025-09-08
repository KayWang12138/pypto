#include "pybind_common.h"
#include "bindings/bindings.h"
using namespace npu::tile_fwk;

namespace pypto {
PYBIND11_MODULE(pto, m) {
    m.doc() = "PyPTO";
    bind_enum(m);
    bind_tensor(m);
    bind_symbolic_scalar(m);
    bind_controller(m);
    bind_operation(m);
};
} // namespace pypto
