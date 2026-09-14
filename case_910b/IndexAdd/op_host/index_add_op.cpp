#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"

namespace index_add_kernel {
at::Tensor launch(const at::Tensor& input, const at::Tensor& index,
                  const at::Tensor& source, int64_t dim);
}

at::Tensor my_op_impl_npu(const at::Tensor& input, const at::Tensor& index,
                          const at::Tensor& source, int64_t dim) {
    TORCH_CHECK(input.defined() && index.defined() && source.defined(), "custom_op expects three defined input tensors");
    TORCH_CHECK(input.device() == index.device() && input.device() == source.device(), "input, index and source must be on the same device");
    TORCH_CHECK(input.dim() > 0 && index.dim() == 1 && source.dim() == input.dim(), "invalid input ranks");
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dimension is out of range");
    TORCH_CHECK(source.size(dim) == index.numel(), "source size on dim must equal index length");
    for (int64_t axis = 0; axis < input.dim(); ++axis) if (axis != dim)
        TORCH_CHECK(source.size(axis) == input.size(axis), "source and input sizes must match except on dim");
    return index_add_kernel::launch(input, index, source, dim);
}

TORCH_LIBRARY(myops, m) { m.def("my_op(Tensor input, Tensor index, Tensor source, int dim) -> Tensor"); }
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) { m.impl("my_op", &my_op_impl_npu); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_op", &my_op_impl_npu, "torch.index_add"); }
