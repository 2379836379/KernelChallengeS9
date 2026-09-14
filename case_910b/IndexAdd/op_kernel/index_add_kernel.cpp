#include "../common/pytorch_npu_helper.hpp"

namespace index_add_kernel {
at::Tensor launch(const at::Tensor& input, const at::Tensor& index,
                  const at::Tensor& source, int64_t dim) {
    auto result = at::empty_like(input);
    EXEC_NPU_CMD(aclnnIndexAdd, input, index, source, dim, result);
    return result;
}
}  // namespace index_add_kernel
