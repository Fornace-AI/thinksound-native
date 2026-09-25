// gemm_mpp.metal — GEMM via MetalPerformancePrimitives matmul2d (Metal 4 tensor ops).
// Y[M,N] = X[M,K] · W[N,K]^T (+bias). Same GemmP contract as gemm_tiled.metal.
// Tensor extents are (fastest, slowest): X (K,M), W (K,N) read transposed, Y (N,M).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct GemmP {
    uint M, N, K;
    uint ldx, ldw, ldy;
    uint bias_mode;   // 0 none, 1 = add f32 bias[n]
    uint pad0;
};

// threadgroup output tile: TM rows of M x TN cols of N, 4 simdgroups cooperating
template <typename WT, int TM, int TN, bool RELAXED>
static inline void gemm_mpp_body(const device WT * W, const device float * X, device float * Y,
                                 const device float * bias, constant GemmP & p, uint2 tg) {
    using TXt = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    using TWt = tensor<device WT, dextents<int32_t, 2>, tensor_inline>;
    TXt tX((device float *) X, dextents<int32_t, 2>(p.K, p.M), array<int32_t, 2>{1, (int32_t) p.ldx});
    TWt tW((device WT *) W, dextents<int32_t, 2>(p.K, p.N), array<int32_t, 2>{1, (int32_t) p.ldw});
    TXt tY(Y, dextents<int32_t, 2>(p.N, p.M), array<int32_t, 2>{1, (int32_t) p.ldy});

    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                              false, true, RELAXED);
    matmul2d<desc, execution_simdgroups<4>> op;
    const int m0 = (int) tg.y * TM, n0 = (int) tg.x * TN;
    auto mA = tX.slice(0, m0);
    auto mB = tW.slice(0, n0);
    auto mC = tY.slice(n0, m0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    op.run(mA, mB, cT);
    if (p.bias_mode) {
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            const int n = n0 + cT.get_multidimensional_index(i)[0];
            if (n < (int) p.N) cT[i] += bias[n];
        }
    }
    cT.store(mC);
}

#define MPP_KERNEL(NAME, WT, TM, TN, RELAXED)                                                    \
kernel void NAME(const device WT * W [[buffer(0)]], const device float * X [[buffer(1)]],       \
                 device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],        \
                 constant GemmP & p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {  \
    gemm_mpp_body<WT, TM, TN, RELAXED>(W, X, Y, bias, p, tg.xy);                                 \
}
MPP_KERNEL(gemm_mpp_h, half, 64, 32, false)
MPP_KERNEL(gemm_mpp_f, float, 64, 32, false)
MPP_KERNEL(gemm_mpp_h_rel, half, 64, 32, true)
MPP_KERNEL(gemm_mpp_h_32x64, half, 32, 64, false)
MPP_KERNEL(gemm_mpp_h_64x64, half, 64, 64, false)
