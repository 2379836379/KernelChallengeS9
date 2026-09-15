#include <torch/extension.h>
#include "function.h"

TORCH_LIBRARY(myops, m) {
    m.def("concat_custom(Tensor[] inputs, int dim) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("concat", &concat_custom, "ConcatCustom Ascend C operator");
}
