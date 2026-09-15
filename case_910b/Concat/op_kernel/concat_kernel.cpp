#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE_LENGTH = 4096;

class KernelConcat {
public:
    __aicore__ inline void Init(GM_ADDR input_table, GM_ADDR output,
                                GM_ADDR lengths, uint32_t input_count) {
        input_table_.SetGlobalBuffer((__gm__ uint64_t*)input_table, input_count);
        lengths_.SetGlobalBuffer((__gm__ uint32_t*)lengths, input_count);
        output_.SetGlobalBuffer((__gm__ half*)output, 0);
        input_count_ = input_count;
    }

    __aicore__ inline void Process() {
        uint32_t output_offset = 0;
        for (uint32_t input_index = 0; input_index < input_count_; ++input_index) {
            const uint32_t length = lengths_.GetValue(input_index);
            const auto address = input_table_.GetValue(input_index);
            GlobalTensor<half> input;
            input.SetGlobalBuffer((__gm__ half*)address, length);
            for (uint32_t offset = 0; offset < length; offset += TILE_LENGTH) {
                const uint32_t count = (offset + TILE_LENGTH <= length)
                                           ? TILE_LENGTH : length - offset;
                LocalTensor<half> local;
                DataCopy(local, input[offset], count);
                DataCopy(output_[output_offset + offset], local, count);
            }
            output_offset += length;
        }
    }

private:
    GlobalTensor<uint64_t> input_table_;
    GlobalTensor<uint32_t> lengths_;
    GlobalTensor<half> output_;
    uint32_t input_count_{0};
};
}  // namespace

extern "C" __global__ __aicore__ void concat_custom(
    GM_ADDR input_table, GM_ADDR output, GM_ADDR lengths, uint32_t input_count) {
    KernelConcat op;
    op.Init(input_table, output, lengths, input_count);
    op.Process();
}
