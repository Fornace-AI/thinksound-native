// gemm_tiled.metal — simdgroup-matrix GEMM: Y[M,N] = X[M,K] · W[N,K]^T (+bias).
// X, Y f32 row-major (strides ldx, ldy); W half or f32 row-major [N,K] (stride ldw).
// Both operands are staged into threadgroup memory as f32 (W widened on load), so
// activations never pass through half (T5 activations exceed f16 range).
#include <metal_stdlib>
using namespace metal;

struct GemmP {
    uint M, N, K;
    uint ldx, ldw, ldy;
    uint bias_mode;   // 0 none, 1 = add f32 bias[n]
    uint pad0;
};

// threadgroup tile BM x BN, K strip BK; 4 simdgroups in a 2x2 grid, each owning
// a (BM/2) x (BN/2) sub-tile as (BM/16) x (BN/16) 8x8 accumulators.
constant constexpr uint BK = 32, NT = 128;

template <typename WT, uint BM, uint BN>
static inline void gemm_tiled_body(const device WT * W, const device float * X, device float * Y,
                                   const device float * bias, constant GemmP & p,
                                   threadgroup float * sX, threadgroup float * sW,
                                   uint2 tg, uint tid, uint sg) {
    constexpr uint SM = BM / 2, SN = BN / 2, FM = SM / 8, FN = SN / 8;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint sm = (sg >> 1) * SM, sn = (sg & 1) * SN;
    simdgroup_float8x8 acc[FM][FN];
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += NT) {
            const uint r = e / BK, c = e % BK, m = m0 + r, k = k0 + c;
            sX[e] = (m < p.M && k < p.K) ? X[(size_t) m * p.ldx + k] : 0.0f;
        }
        for (uint e = tid; e < BN * BK; e += NT) {
            const uint r = e / BK, c = e % BK, n = n0 + r, k = k0 + c;
            sW[e] = (n < p.N && k < p.K) ? (float) W[(size_t) n * p.ldw + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[FM], b[FN];
            for (uint i = 0; i < FM; ++i) simdgroup_load(a[i], sX + (sm + 8 * i) * BK + kk, BK);
            // W tile rows are n: a transposed load gives the K x N block of W^T
            for (uint j = 0; j < FN; ++j)
                simdgroup_load(b[j], sW + (sn + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
            for (uint i = 0; i < FM; ++i)
                for (uint j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // epilogue via threadgroup memory: bounds-checked store + bias
    threadgroup float * sY = sW;   // sized max(BN*BK, BM*BN) by the caller
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) simdgroup_store(acc[i][j], sY + (sm + 8 * i) * BN + sn + 8 * j, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += NT) {
        const uint r = e / BN, c = e % BN, m = m0 + r, n = n0 + c;
        if (m < p.M && n < p.N) Y[(size_t) m * p.ldy + n] = sY[e] + (p.bias_mode ? bias[n] : 0.0f);
    }
}

#define TILED_KERNEL(NAME, WT, BM, BN)                                                          \
kernel void NAME(const device WT * W [[buffer(0)]], const device float * X [[buffer(1)]],       \
                 device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],        \
                 constant GemmP & p [[buffer(4)]],                                               \
                 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                   \
    threadgroup float sX[BM * BK], sW[(BN * BK > BM * BN) ? BN * BK : BM * BN];                  \
    gemm_tiled_body<WT, BM, BN>(W, X, Y, bias, p, sX, sW, tg.xy, tid, sg);                       \
}
TILED_KERNEL(gemm_t_h, half, 32, 64)
TILED_KERNEL(gemm_t_f, float, 32, 64)
TILED_KERNEL(gemm_t64_h, half, 64, 64)
