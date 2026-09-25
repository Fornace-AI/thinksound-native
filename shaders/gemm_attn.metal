// gemm_attn.metal — scalar GEMM/GEMV (half/f32 weights), batched-head attention.
#include <metal_stdlib>
using namespace metal;

struct GemmP {
    uint M, N, K;
    uint ldx, ldw, ldy;
    uint bias_mode;   // 0 none, 1 = add f32 bias[n]
    uint pad0;
};

// ---------------------------------------------------------------------------
// GEMM: Y[M,N] = X[M,K] · W[N,K]^T (+bias), row-major activations.
// Scalar one-thread-per-output formulation. Runtime::gemm dispatches the MPP
// kernels in gemm_mpp.metal; these stay as the ts-gemm-bench reference.
// ---------------------------------------------------------------------------

template <typename WT>
static inline void gemm_body(
    const device WT * W, const device float * X, device float * Y,
    const device float * bias, constant GemmP & p,
    uint3 tgid, uint3 tid)
{
    uint m = tgid.y * 8 + tid.x / 32;    // 256 threads = 8 rows × 32 cols
    uint n = tgid.x * 32 + tid.x % 32;
    if (m >= p.M || n >= p.N) return;
    const device WT * w = W + (size_t) n * p.ldw;
    const device float * x = X + (size_t) m * p.ldx;
    float acc = p.bias_mode ? bias[n] : 0.0f;
    for (uint k = 0; k < p.K; ++k) acc += (float) w[k] * x[k];
    Y[(size_t) m * p.ldy + n] = acc;
}

kernel void gemm_h(const device half * W [[buffer(0)]], const device float * X [[buffer(1)]],
                   device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],
                   constant GemmP & p [[buffer(4)]],
                   uint3 tgid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    gemm_body<half>(W, X, Y, bias, p, tgid, tid);
}
kernel void gemm_f(const device float * W [[buffer(0)]], const device float * X [[buffer(1)]],
                   device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],
                   constant GemmP & p [[buffer(4)]],
                   uint3 tgid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    gemm_body<float>(W, X, Y, bias, p, tgid, tid);
}

// gemv: y[n] = bias[n] + Σ_k W[n,k]·x[k]   (M = 1)
template <typename WT>
static inline float gemv_body(const device WT * W, const device float * X,
                              const device float * bias, uint n, uint K, uint ldw, uint bias_mode) {
    // weights stored numpy row-major [N,K]: element (n,k) at n·K + k
    const device WT * w = W + (size_t) n * ldw;
    float acc = bias_mode ? bias[n] : 0.0f;
    for (uint k = 0; k < K; ++k) acc += (float) w[k] * X[k];
    return acc;
}
kernel void gemv_h(const device half * W [[buffer(0)]], const device float * X [[buffer(1)]],
                   device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],
                   constant GemmP & p [[buffer(4)]],
                   uint3 tgid [[threadgroup_position_in_grid]],
                   uint3 tid [[thread_position_in_threadgroup]]) {
    uint n = tgid.x * 64 + tid.x;
    if (n >= p.N) return;
    Y[n] = gemv_body<half>(W, X, bias, n, p.K, p.ldw, p.bias_mode);
}
kernel void gemv_f(const device float * W [[buffer(0)]], const device float * X [[buffer(1)]],
                   device float * Y [[buffer(2)]], const device float * bias [[buffer(3)]],
                   constant GemmP & p [[buffer(4)]],
                   uint3 tgid [[threadgroup_position_in_grid]],
                   uint3 tid [[thread_position_in_threadgroup]]) {
    uint n = tgid.x * 64 + tid.x;
    if (n >= p.N) return;
    Y[n] = gemv_body<float>(W, X, bias, n, p.K, p.ldw, p.bias_mode);
}

// ---------------------------------------------------------------------------
// Attention (one head per threadgroup row-tile):
//   scores P[q,k] = scale · Q[q]·K[k] (+ optional T5 bias / CLIP causal mask)
//   then row softmax, then pav: O[q,d] = Σ_k P[q,k] · V[k,d]
// Q/K/V are column slices of a [S,3H] qkv buffer (ld = 3H), head h at column
// h·hd. One kernel launch covers ALL heads via grid.y.
// ---------------------------------------------------------------------------

struct AttnP {
    uint pad_on;      // 1: mask keys where pad flag == 0 (t5)
    uint Sq, Sk, hd, nh;
    uint ld;
    uint ldo;
    uint q_off, k_off, v_off, o_off;
    uint bias_mode;   // 0 none, 1 t5 bucket table, 2 clip causal+pad
    uint eot;
    float scale;
    uint p_off;       // probs scratch offset (elements), layout [nh][Sq][Sk]
};

kernel void attn_qk(const device float * QKV [[buffer(0)]],
                    device float * P [[buffer(1)]],
                    const device int * bkt [[buffer(2)]],     // [Sq*Sk] bucket ids (t5)
                    const device float * rb [[buffer(3)]],    // rel bias weights [32*nh]
                    constant AttnP & p [[buffer(4)]],
                    uint3 tgid [[threadgroup_position_in_grid]],
                    uint3 tid [[thread_position_in_threadgroup]])
{
    constexpr uint QT = 16, KT = 16, HD = 128;   // HD covers up to 128 dims
    threadgroup float Qs[QT][HD];
    threadgroup float Ks[KT][HD];
    const uint lid = tid.x;
    const uint h = tgid.y;
    const uint q0 = tgid.x * QT;
    const uint hd = p.hd;

    // stage Q tile
    for (uint i = lid; i < QT * hd; i += 256) {
        uint r = i / hd, c = i % hd;
        uint g = q0 + r;
        Qs[r][c] = (g < p.Sq) ? QKV[(size_t) g * p.ld + p.q_off + h * hd + c] : 0.0f;
    }
    for (uint k0 = 0; k0 < p.Sk; k0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < KT * hd; i += 256) {
            uint r = i / hd, c = i % hd;
            uint g = k0 + r;
            Ks[r][c] = (g < p.Sk) ? QKV[(size_t) g * p.ld + p.k_off + h * hd + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 256 threads: 16x16 (q_local, k_local)
        uint ql = lid / KT, kl = lid % KT;
        uint q = q0 + ql, k = k0 + kl;
        if (q < p.Sq && k < p.Sk) {
            float s = 0.0f;
            for (uint d = 0; d < hd; ++d) s += Qs[ql][d] * Ks[kl][d];
            s *= p.scale;
            if (p.bias_mode == 1) {
                const device int * pad = bkt + (size_t) p.Sq * p.Sk;   // pad flags packed after buckets
                if (p.pad_on && pad[k] == 0) s = -INFINITY;
                else s += rb[(size_t) bkt[q * p.Sk + k] * p.nh + h];
            } else if (p.bias_mode == 2) {
                if (!(k <= q && k <= p.eot)) s = -INFINITY;
            }
            P[(size_t) p.p_off + ((size_t) h * p.Sq + q) * p.Sk + k] = s;
        }
    }
}

kernel void softmax_rows(device float * P [[buffer(0)]], constant AttnP & p [[buffer(1)]],
                         uint3 tgid [[threadgroup_position_in_grid]])
{
    uint h = tgid.y, q = tgid.x;
    if (q >= p.Sq) return;
    device float * row = P + (size_t) p.p_off + ((size_t) h * p.Sq + q) * p.Sk;
    float mx = -INFINITY;
    for (uint k = 0; k < p.Sk; ++k) mx = max(mx, row[k]);
    float sum = 0.0f;
    for (uint k = 0; k < p.Sk; ++k) { row[k] = exp(row[k] - mx); sum += row[k]; }
    float inv = 1.0f / sum;
    for (uint k = 0; k < p.Sk; ++k) row[k] *= inv;
}

// O[q, d] = Σ_k P[h,q,k] · V[k,d];  256 threads = 16 q × 16 d lanes unrolled
kernel void attn_pav(const device float * P [[buffer(0)]], const device float * QKV [[buffer(1)]],
                     device float * O [[buffer(2)]], constant AttnP & p [[buffer(3)]],
                     uint3 tgid [[threadgroup_position_in_grid]],
                     uint3 tid [[thread_position_in_threadgroup]])
{
    constexpr uint QT = 16, KT = 32;
    threadgroup float Ps[QT][KT];
    threadgroup float Vs[KT][128];
    const uint lid = tid.x;
    const uint h = tgid.y;
    const uint q0 = tgid.x * QT;
    const uint hd = p.hd;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // 8 d-lanes per thread
    uint ql = lid / 16, dl = lid % 16;         // dl covers hd/8 iterations of 16 lanes
    for (uint k0 = 0; k0 < p.Sk; k0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < QT * KT; i += 256) {
            uint r = i / KT, c = i % KT;
            uint q = q0 + r, k = k0 + c;
            Ps[r][c] = (q < p.Sq && k < p.Sk) ? P[(size_t) p.p_off + ((size_t) h * p.Sq + q) * p.Sk + k] : 0.0f;
        }
        for (uint i = lid; i < KT * hd; i += 256) {
            uint r = i / hd, c = i % hd;
            uint k = k0 + r;
            Vs[r][c] = (k < p.Sk) ? QKV[(size_t) k * p.ld + p.v_off + h * hd + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < 8; ++d) {
            uint dc = dl + d * 16;
            if (dc < hd) {
                float s = 0.0f;
                for (uint k = 0; k < KT; ++k) s += Ps[ql][k] * Vs[k][dc];
                acc[d] += s;
            }
        }
    }
    uint q = q0 + ql;
    for (uint d = 0; d < 8; ++d) {
        uint dc = dl + d * 16;
        if (q < p.Sq && dc < hd)
            O[(size_t) q * p.ldo + p.o_off + h * hd + dc] = acc[d];
    }
}
