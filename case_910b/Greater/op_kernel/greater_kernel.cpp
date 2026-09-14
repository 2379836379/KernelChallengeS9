#include "../common/pytorch_npu_helper.hpp"

namespace greater_kernel {
at::Tensor launch(const at::Tensor& x1, const at::Tensor& x2,
                  const at::IntArrayRef& output_shape) {
    auto result = at::empty(output_shape, x1.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnGreater, x1, x2, result);
    return result;
}
}  // namespace greater_kernel
