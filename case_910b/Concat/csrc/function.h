#ifndef CONCAT_FUNCTION_H
#define CONCAT_FUNCTION_H

#include <ATen/ATen.h>
#include <vector>

at::Tensor concat_custom(const std::vector<at::Tensor>& inputs, int64_t dim);

#endif  // CONCAT_FUNCTION_H
