/**
 * Host-side validation, output shape inference and PyTorch registration for
 * the basic concat operator.
 */
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>

#include "../common/pytorch_npu_helper.hpp"

namespace concat_kernel {
at::Tensor launch(const std::vector<at::Tensor>& inputs, int64_t dim,
                  const at::IntArrayRef& output_shape);
}  // namespace concat_kernel

using tensor_list = std::vector<at::Tensor>;

at::Tensor my_op_impl_npu(const tensor_list& inputs, int64_t dim,
                          const at::IntArrayRef& output_shape) {
    TORCH_CHECK(!inputs.empty(), "custom_op expects at least one input tensor");

    const auto& first = inputs.front();
    TORCH_CHECK(first.defined(), "input tensor must be defined");
    const auto ndim = first.dim();
    TORCH_CHECK(ndim > 0, "custom_op does not support scalar tensors");
    TORCH_CHECK(output_shape.size() == static_cast<size_t>(ndim),
                "output_shape must have the same rank as the input tensors");

    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "concat dimension is out of range");

    auto expected_shape = first.sizes().vec();
    int64_t concat_size = 0;
    for (const auto& input : inputs) {
        TORCH_CHECK(input.defined(), "input tensor must be defined");
        TORCH_CHECK(input.dim() == ndim,
                    "all input tensors must have the same rank");
        TORCH_CHECK(input.device() == first.device(),
                    "all input tensors must be on the same device");
        TORCH_CHECK(input.scalar_type() == first.scalar_type(),
                    "all input tensors must have the same dtype");
        for (int64_t axis = 0; axis < ndim; ++axis) {
            if (axis != dim) {
                TORCH_CHECK(input.size(axis) == first.size(axis),
                            "input sizes must match except on concat dimension");
            }
        }
        concat_size += input.size(dim);
    }

    expected_shape[dim] = concat_size;
    for (int64_t axis = 0; axis < ndim; ++axis) {
        TORCH_CHECK(output_shape[axis] == expected_shape[axis],
                    "output_shape does not match the inputs");
    }
    return concat_kernel::launch(inputs, dim, output_shape);
}

TORCH_LIBRARY(myops, m) {
    m.def("my_op(Tensor[] inputs, int dim, int[] output_shape) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &my_op_impl_npu, "torch.cat");
}
