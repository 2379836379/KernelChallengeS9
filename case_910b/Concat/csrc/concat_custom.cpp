#include <torch/library.h>
#include <torch/extension.h>

#include "pytorch_npu_helper.hpp"

using tensor_list = std::vector<at::Tensor>;

at::Tensor concat_custom_impl_npu(const tensor_list& inputs, int64_t dim) {
    TORCH_CHECK(!inputs.empty(), "concat_custom expects at least one input");
    const auto& first = inputs.front();
    TORCH_CHECK(first.defined() && first.dim() > 0,
                "input must be a defined non-scalar tensor");
    TORCH_CHECK(first.is_contiguous(), "inputs must be contiguous");
    const auto rank = first.dim();
    if (dim < 0) dim += rank;
    TORCH_CHECK(dim >= 0 && dim < rank, "concat dimension is out of range");

    auto output_shape = first.sizes().vec();
    output_shape[dim] = 0;
    for (const auto& input : inputs) {
        TORCH_CHECK(input.defined() && input.dim() == rank,
                    "all inputs must have the same rank");
        TORCH_CHECK(input.device() == first.device(),
                    "all inputs must be on the same device");
        TORCH_CHECK(input.scalar_type() == first.scalar_type(),
                    "all inputs must have the same dtype");
        TORCH_CHECK(input.is_contiguous(), "inputs must be contiguous");
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (axis == dim) output_shape[axis] += input.size(axis);
            else TORCH_CHECK(input.size(axis) == first.size(axis),
                             "input sizes must match except on concat dimension");
        }
    }
    auto result = at::empty(output_shape, first.options());
    EXEC_NPU_CMD(aclnnConcatCustom, at::TensorList(inputs), dim, result);
    return result;
}

at::Tensor concat_custom_impl_meta(const tensor_list& inputs, int64_t dim) {
    TORCH_CHECK(!inputs.empty(), "concat_custom expects at least one input");
    auto output_shape = inputs.front().sizes().vec();
    if (dim < 0) dim += inputs.front().dim();
    output_shape[dim] = 0;
    for (const auto& input : inputs) output_shape[dim] += input.size(dim);
    return at::empty(output_shape, inputs.front().options());
}

at::Tensor concat_custom(const tensor_list& inputs, int64_t dim) {
    return concat_custom_impl_npu(inputs, dim);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("concat_custom", &concat_custom_impl_npu);
}
TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("concat_custom", &concat_custom_impl_meta);
}
