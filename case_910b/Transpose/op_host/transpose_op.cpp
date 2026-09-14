#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"

namespace transpose_kernel {
at::Tensor launch(const at::Tensor& input, const at::IntArrayRef& dims,
                  const at::IntArrayRef& result_shape);
}

at::Tensor my_op_impl_npu(const at::Tensor& input, const at::IntArrayRef& dims,
                          const at::IntArrayRef& result_shape) {
    TORCH_CHECK(input.defined(), "input tensor must be defined");
    TORCH_CHECK(dims.size() == static_cast<size_t>(input.dim()), "dims must be a permutation of all input dimensions");
    TORCH_CHECK(result_shape.size() == dims.size(), "result_shape rank must match dims");
    std::vector<int64_t> normalized_dims(dims.begin(), dims.end());
    std::vector<bool> seen(input.dim(), false);
    for (auto& value : normalized_dims) {
        if (value < 0) value += input.dim();
        TORCH_CHECK(value >= 0 && value < input.dim(), "dimension is out of range");
        TORCH_CHECK(!seen[value], "dims must not contain duplicates");
        seen[value] = true;
    }
    for (size_t axis = 0; axis < normalized_dims.size(); ++axis)
        TORCH_CHECK(result_shape[axis] == input.size(normalized_dims[axis]), "result_shape does not match dims");
    return transpose_kernel::launch(input, normalized_dims, result_shape);
}

TORCH_LIBRARY(myops, m) { m.def("my_op(Tensor input, int[] dims, int[] result_shape) -> Tensor"); }
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) { m.impl("my_op", &my_op_impl_npu); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_op", &my_op_impl_npu, "torch.permute"); }
