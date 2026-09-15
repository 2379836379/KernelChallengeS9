#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 4096;

class KernelGreater {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t total_length) {
        total_length_ = total_length;
        x1_gm_.SetGlobalBuffer((__gm__ half*)x1, total_length_);
        x2_gm_.SetGlobalBuffer((__gm__ half*)x2, total_length_);
        y_gm_.SetGlobalBuffer((__gm__ uint8_t*)y, total_length_);
        pipe_.InitBuffer(in_x1_, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe_.InitBuffer(in_x2_, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe_.InitBuffer(out_y_, BUFFER_NUM, TILE_LENGTH * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        for (uint32_t offset = 0; offset < total_length_; offset += TILE_LENGTH) {
            const uint32_t length = (offset + TILE_LENGTH <= total_length_)
                                        ? TILE_LENGTH : total_length_ - offset;
            auto x1_local = in_x1_.AllocTensor<half>();
            auto x2_local = in_x2_.AllocTensor<half>();
            auto y_local = out_y_.AllocTensor<uint8_t>();
            DataCopy(x1_local, x1_gm_[offset], length);
            DataCopy(x2_local, x2_gm_[offset], length);
            Compare(y_local, x1_local, x2_local, CMPMODE::GT, length);
            DataCopy(y_gm_[offset], y_local, length);
            in_x1_.FreeTensor(x1_local);
            in_x2_.FreeTensor(x2_local);
            out_y_.FreeTensor(y_local);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> in_x1_;
    TQue<QuePosition::VECIN, BUFFER_NUM> in_x2_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> out_y_;
    GlobalTensor<half> x1_gm_;
    GlobalTensor<half> x2_gm_;
    GlobalTensor<uint8_t> y_gm_;
    uint32_t total_length_{0};
};
}  // namespace

extern "C" __global__ __aicore__ void greater_custom(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t total_length) {
    KernelGreater op;
    op.Init(x1, x2, y, total_length);
    op.Process();
}
