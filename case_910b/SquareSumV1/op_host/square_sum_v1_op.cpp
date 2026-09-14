#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"

namespace square_sum_v1_kernel {
at::Tensor launch(const at::Tensor& input, const at::IntArrayRef& axis,
                  bool keep_dims, const at::IntArrayRef& result_shape);
}

at::Tensor my_op_impl_npu(const at::Tensor& input, const at::IntArrayRef& axis,
                          bool keep_dims, const at::IntArrayRef& result_shape) {
    TORCH_CHECK(input.defined(), "input tensor must be defined");
    TORCH_CHECK(axis.size() <= static_cast<size_t>(input.dim()), "axis contains too many dimensions");
    for (const auto value : axis)
        TORCH_CHECK(value >= -input.dim() && value < input.dim(), "axis is out of range");
    const auto expected_rank = keep_dims ? input.dim() : input.dim() - axis.size();
    TORCH_CHECK(result_shape.size() == static_cast<size_t>(expected_rank), "result_shape rank does not match reduction settings");
    return square_sum_v1_kernel::launch(input, axis, keep_dims, result_shape);
}

TORCH_LIBRARY(myops, m) { m.def("my_op(Tensor input, int[] axis, bool keep_dims, int[] result_shape) -> Tensor"); }
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) { m.impl("my_op", &my_op_impl_npu); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_op", &my_op_impl_npu, "torch.sum(torch.square)"); }
