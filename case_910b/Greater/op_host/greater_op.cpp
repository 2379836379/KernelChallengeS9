#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"

namespace greater_kernel {
at::Tensor launch(const at::Tensor& x1, const at::Tensor& x2,
                  const at::IntArrayRef& output_shape);
}

at::Tensor my_op_impl_npu(const at::Tensor& x1, const at::Tensor& x2) {
    TORCH_CHECK(x1.defined() && x2.defined(), "custom_op expects two defined input tensors");
    TORCH_CHECK(x1.device() == x2.device(), "input tensors must be on the same device");
    const auto output_shape = at::infer_size(x1.sizes(), x2.sizes());
    return greater_kernel::launch(x1, x2, output_shape);
}

TORCH_LIBRARY(myops, m) { m.def("my_op(Tensor input, Tensor other) -> Tensor"); }
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) { m.impl("my_op", &my_op_impl_npu); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_op", &my_op_impl_npu, "torch.gt"); }
