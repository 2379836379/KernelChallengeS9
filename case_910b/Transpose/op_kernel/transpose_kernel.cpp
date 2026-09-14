#include "../common/pytorch_npu_helper.hpp"

namespace transpose_kernel {
at::Tensor launch(const at::Tensor& input, const at::IntArrayRef& dims,
                  const at::IntArrayRef& result_shape) {
    auto result = at::empty(result_shape, input.options());
    EXEC_NPU_CMD(aclnnTranspose, input, dims, result);
    return result;
}
}  // namespace transpose_kernel
