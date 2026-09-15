#include <cstdint>

struct GreaterTilingData {
    uint32_t total_length;
};

extern "C" bool GreaterInferShape(const int64_t* x1_shape, uint32_t x1_rank,
                                   const int64_t* x2_shape, uint32_t x2_rank,
                                   int64_t* output_shape, uint32_t* output_rank) {
    const uint32_t rank = x1_rank > x2_rank ? x1_rank : x2_rank;
    for (uint32_t i = 0; i < rank; ++i) {
        const int64_t d1 = i < rank - x1_rank ? 1 : x1_shape[i - rank + x1_rank];
        const int64_t d2 = i < rank - x2_rank ? 1 : x2_shape[i - rank + x2_rank];
        if (d1 != d2 && d1 != 1 && d2 != 1) return false;
        output_shape[i] = d1 > d2 ? d1 : d2;
    }
    *output_rank = rank;
    return true;
}

extern "C" GreaterTilingData GreaterMakeTiling(uint32_t element_count) {
    return GreaterTilingData{element_count};
}
