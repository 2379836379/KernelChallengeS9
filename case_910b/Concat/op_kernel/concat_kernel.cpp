/**
 * Device-side launch helper for the basic concat implementation.
 *
 * The actual kernel is supplied by the Ascend aclnnConcat operator. Keeping
 * the launch in this layer makes the host registration independent of the
 * device API details and leaves room for an Ascend C kernel later.
 */
#include "../common/pytorch_npu_helper.hpp"

namespace concat_kernel {

at::Tensor launch(const std::vector<at::Tensor>& inputs, int64_t dim,
                  const at::IntArrayRef& output_shape) {
    auto result = at::empty(output_shape, inputs.front().options());
    EXEC_NPU_CMD(aclnnConcat, at::TensorList(inputs), dim, result);
    return result;
}

}  // namespace concat_kernel
