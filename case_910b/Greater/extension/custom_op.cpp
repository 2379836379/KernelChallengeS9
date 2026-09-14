/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor my_op_impl_npu(const at::Tensor& x1, const at::Tensor& x2) {
    TORCH_CHECK(x1.defined() && x2.defined(),
                "custom_op expects two defined input tensors");
    TORCH_CHECK(x1.device() == x2.device(),
                "input tensors must be on the same device");

    // aclnnGreater follows PyTorch broadcasting semantics. infer_size also
    // validates incompatible input shapes before the output is allocated.
    const auto output_size = at::infer_size(x1.sizes(), x2.sizes());
    auto result = at::empty(output_size, x1.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnGreater, x1, x2, result);
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, Tensor other) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.gt");
}
