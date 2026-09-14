#include "../common/pytorch_npu_helper.hpp"

namespace square_sum_v1_kernel {
at::Tensor launch(const at::Tensor& input, const at::IntArrayRef& axis,
                  bool keep_dims, const at::IntArrayRef& result_shape) {
    auto result = at::empty(result_shape, input.options());
    EXEC_NPU_CMD(aclnnSquareSumV1, input, axis, keep_dims, result);
    return result;
}
}  // namespace square_sum_v1_kernel
