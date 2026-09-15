#include <cstdint>

struct ConcatTilingData {
    int64_t dim;
    uint32_t input_count;
    uint32_t output_elements;
};

// Shape inference for the basic Ascend C kernel. The kernel consumes flattened
// contiguous tensors, so the supported form concatenates along the last axis.
extern "C" bool ConcatInferShape(const int64_t* const* input_shapes,
                                  const uint32_t* input_ranks,
                                  uint32_t input_count, int64_t dim,
                                  int64_t* output_shape, uint32_t* output_rank) {
    if (input_count == 0 || input_shapes == nullptr || input_ranks == nullptr) return false;
    const uint32_t rank = input_ranks[0];
    if (rank == 0) return false;
    if (dim < 0) dim += rank;
    if (dim != static_cast<int64_t>(rank - 1)) return false;
    for (uint32_t i = 0; i < input_count; ++i) {
        if (input_ranks[i] != rank) return false;
        for (uint32_t axis = 0; axis + 1 < rank; ++axis) {
            if (input_shapes[i][axis] != input_shapes[0][axis]) return false;
        }
    }
    for (uint32_t axis = 0; axis < rank; ++axis) output_shape[axis] = input_shapes[0][axis];
    output_shape[rank - 1] = 0;
    for (uint32_t i = 0; i < input_count; ++i) output_shape[rank - 1] += input_shapes[i][rank - 1];
    *output_rank = rank;
    return true;
}

extern "C" ConcatTilingData ConcatMakeTiling(int64_t dim, uint32_t input_count,
                                              uint32_t output_elements) {
    return ConcatTilingData{dim, input_count, output_elements};
}
