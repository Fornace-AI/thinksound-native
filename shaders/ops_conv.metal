// ops_conv.metal — norms, adaLN modulation, RoPE, activations, Vocos convs, misc.
// Convention: tensor args in buffers 0..k, ONE constant param struct last.
#include <metal_stdlib>
using namespace metal;

struct N1 { uint n; };

// ---- activations / elementwise ----
kernel void silu_(device float * x [[buffer(0)]], constant N1 & p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v = x[gid];
    x[gid] = v / (1.0f + exp(-v));
}
kernel void gelu_(device float * x [[buffer(0)]], constant N1 & p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v = x[gid];
    // standard tanh approximation of GELU
    x[gid] = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}
kernel void gelu_quick_(device float * x [[buffer(0)]], constant N1 & p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v = x[gid];
    x[gid] = v / (1.0f + exp(-1.702f * v));
}
kernel void copy_(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                 constant N1 & p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    y[gid] = x[gid];
}

kernel void swiglu_(device const float * a [[buffer(0)]], device const float * b [[buffer(1)]],
                    device float * y [[buffer(2)]], constant N1 & p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float va = a[gid], vb = b[gid];
    y[gid] = (va / (1.0f + exp(-va))) * vb;
}
kernel void add_(device const float * a [[buffer(0)]], device const float * b [[buffer(1)]],
                 device float * y [[buffer(2)]], constant N1 & p [[buffer(3)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    y[gid] = a[gid] + b[gid];
}

struct CH { uint H; uint n; };
kernel void add_bcast_col(device const float * x [[buffer(0)]], device const float * v [[buffer(1)]],
                          device float * y [[buffer(2)]], constant CH & p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint c = gid % p.H;
    y[gid] = x[gid] + v[c];
}

// ---- norms ----
struct QS3 { uint H; uint row_off; };

// torch qkv output is element-interleaved (col 3n+j: j=0 q, 1 k, 2 v).
// rewrite a stream region [row_off, row_off+rows) x 3H into block q|k|v layout.
kernel void qkv_split3(device const float * src [[buffer(0)]], device float * dst [[buffer(1)]],
                        constant QS3 & p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    // src is already offset to the stream's first row; dst rows are written at row_off.
    uint c = gid.x;                       // dim within one of q/k/v
    if (c >= p.H) return;
    size_t ld = (size_t) 3 * p.H;
    size_t srow = (size_t) gid.y * ld;
    size_t drow = (size_t) (p.row_off + gid.y) * ld;
    for (uint j = 0; j < 3; ++j)
        dst[drow + (size_t) j * p.H + c] = src[srow + (size_t) 3 * c + j];
}

struct HE { uint H; float eps; };
kernel void ln_plain(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                     constant HE & p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    device const float * r = x + (size_t) gid * p.H;
    device float * o = y + (size_t) gid * p.H;
    float mean = 0.0f;
    for (uint i = 0; i < p.H; ++i) mean += r[i];
    mean /= (float) p.H;
    float var = 0.0f;
    for (uint i = 0; i < p.H; ++i) { float d = r[i] - mean; var += d * d; }
    var /= (float) p.H;
    float inv = 1.0f / sqrt(var + p.eps);
    for (uint i = 0; i < p.H; ++i) o[i] = (r[i] - mean) * inv;
}
kernel void ln_wb(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                  device const float * w [[buffer(2)]], device const float * b [[buffer(3)]],
                  constant HE & p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    device const float * r = x + (size_t) gid * p.H;
    device float * o = y + (size_t) gid * p.H;
    float mean = 0.0f;
    for (uint i = 0; i < p.H; ++i) mean += r[i];
    mean /= (float) p.H;
    float var = 0.0f;
    for (uint i = 0; i < p.H; ++i) { float d = r[i] - mean; var += d * d; }
    var /= (float) p.H;
    float inv = 1.0f / sqrt(var + p.eps);
    for (uint i = 0; i < p.H; ++i) o[i] = (r[i] - mean) * inv * w[i] + b[i];
}
kernel void rms_w(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                  device const float * w [[buffer(2)]], constant HE & p [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    device const float * r = x + (size_t) gid * p.H;
    device float * o = y + (size_t) gid * p.H;
    float ss = 0.0f;
    for (uint i = 0; i < p.H; ++i) ss += r[i] * r[i];
    float inv = 1.0f / sqrt(ss / (float) p.H + p.eps);
    for (uint i = 0; i < p.H; ++i) o[i] = r[i] * inv * w[i];
}

// per-head RMSNorm on a column slice of a [S, ld] buffer (DiT q/k norm).
// grid (rows, nh); slice base = row_off·ld + col_off.
struct RH { uint ld, hd, col_off, row_off; float eps; };
kernel void rms_head(device float * x [[buffer(0)]], device const float * w [[buffer(1)]],
                     constant RH & p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    device float * r = x + (size_t) (p.row_off + gid.x) * p.ld + p.col_off + gid.y * p.hd;
    float ss = 0.0f;
    for (uint i = 0; i < p.hd; ++i) ss += r[i] * r[i];
    float inv = 1.0f / sqrt(ss / (float) p.hd + p.eps);
    for (uint i = 0; i < p.hd; ++i) r[i] = r[i] * inv * w[i];
}

// ---- adaLN ----
struct AD { uint H, per_row, chunks, base; float eps; };
// modulate(layernorm(x), shift=m[base], scale=m[base+H])
kernel void adaln_ln(device const float * x [[buffer(0)]], device const float * mod [[buffer(1)]],
                     device float * y [[buffer(2)]], constant AD & p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    device const float * r = x + (size_t) gid * p.H;
    device const float * m = mod + (p.per_row ? (size_t) gid * p.chunks * p.H : 0) + p.base;
    device float * o = y + (size_t) gid * p.H;
    float mean = 0.0f, var = 0.0f;
    for (uint i = 0; i < p.H; ++i) mean += r[i];
    mean /= (float) p.H;
    for (uint i = 0; i < p.H; ++i) { float d = r[i] - mean; var += d * d; }
    var /= (float) p.H;
    float inv = 1.0f / sqrt(var + p.eps);
    for (uint i = 0; i < p.H; ++i)
        o[i] = (r[i] - mean) * inv * (1.0f + m[p.H + i]) + m[i];
}

// y = x + f · gate ; gate = mod[chunks·H slice gate_idx]
struct GA { uint H, per_row, chunks, gate_idx; };
kernel void gate_add(device const float * x [[buffer(0)]], device const float * f [[buffer(1)]],
                     device const float * mod [[buffer(2)]], device float * y [[buffer(3)]],
                     constant GA & p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint s = gid / p.H, c = gid % p.H;
    device const float * m = mod + (p.per_row ? (size_t) s * p.chunks * p.H : 0);
    y[gid] = x[gid] + f[gid] * m[p.gate_idx * p.H + c];
}

// ---- RoPE, interleaved pairs, per head ----
struct RP { uint ld, hd, col_off, row_off; float fscale, base; };
kernel void rope_pairs(device float * qkv [[buffer(0)]], device const int * pos [[buffer(1)]],
                       constant RP & p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint npair = p.hd / 2;
    uint head = gid.y / npair, j = gid.y % npair;
    float inv = pow(p.base, -2.0f * (float) j / (float) p.hd);
    float ang = (float) (pos ? pos[gid.x] : (int) gid.x) * p.fscale * inv;
    float cs = cos(ang), sn = sin(ang);
    device float * r = qkv + (size_t) (p.row_off + gid.x) * p.ld + p.col_off + head * p.hd;
    float x0 = r[2 * j], x1 = r[2 * j + 1];
    r[2 * j] = x0 * cs - x1 * sn;
    r[2 * j + 1] = x0 * sn + x1 * cs;
}

// ---- column mean over S ----
struct SH { uint S, H; };
kernel void mean_cols(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                      constant SH & p [[buffer(2)]],
                      uint3 tgid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    const uint lid = tid.x;
    threadgroup float partial[8][32];
    uint c = tgid.x * 8 + lid / 32;
    uint lane = lid % 32;
    float acc = 0.0f;
    if (c < p.H) {
        for (uint s = lane; s < p.S; s += 32) acc += x[(size_t) s * p.H + c];
    }
    partial[lid / 32][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8 && tgid.x * 8 + lid < p.H) {
        float sum = 0.0f;
        for (uint k = 0; k < 32; ++k) sum += partial[lid][k];
        y[tgid.x * 8 + lid] = sum / (float) p.S;
    }
}

// ---- sampler ----
struct CD { float cfg, dt; uint n; };
kernel void cfg_euler(device const float * fc [[buffer(0)]], device const float * fu [[buffer(1)]],
                      device float * x [[buffer(2)]], constant CD & p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v = fu[gid] + (fc[gid] - fu[gid]) * p.cfg;
    x[gid] += p.dt * v;
}

// ---- Vocos ----
kernel void snake_(device const float * x [[buffer(0)]], device float * y [[buffer(1)]],
                   device const float * ab [[buffer(2)]], constant SH & p [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.S * p.H) return;
    uint c = gid % p.H;   // H = channels here
    float v = x[gid];
    float ea = exp(ab[c]), eb = exp(-ab[p.H + c]);
    float s = sin(v * ea);
    y[gid] = v + s * s * eb;
}

struct ConvP {
    uint T, OL, IC, OC, K, stride, pad, dil;
    uint ldx, ldy;
    uint convt;
    uint use_bias;
};
template <typename WT>
static inline void conv_body(device const float * x, device const WT * w,
                             device const float * bias, device float * y,
                             constant ConvP & p,
                             uint3 tgid, uint lid) {
    const uint tl = lid % 32, oc_l = lid / 32;             // 8 oc per TG
    const uint t = tgid.x * 32 + tl;
    const uint oc = tgid.y * 8 + oc_l;
    if (t >= p.OL || oc >= p.OC) return;
    float acc = (p.use_bias && bias) ? bias[oc] : 0.0f;
    for (uint ic = 0; ic < p.IC; ++ic) {
        for (uint k = 0; k < p.K; ++k) {
            int ti;
            if (p.convt) {
                int ui = (int) t - (int) p.pad + (int) k;
                if (ui < 0 || ui % (int) p.stride != 0) continue;
                ti = ui / (int) p.stride;
            } else {
                ti = (int) (t * p.stride) - (int) p.pad + (int) (k * p.dil);
            }
            if (ti < 0 || ti >= (int) p.T) continue;
            acc += x[(size_t) ti * p.ldx + ic] * (float) w[((size_t) oc * p.IC + ic) * p.K + k];
        }
    }
    y[(size_t) t * p.ldy + oc] = acc;
}
kernel void conv1d_fwd(device const float * x [[buffer(0)]], device const half * w [[buffer(1)]],
                       device const float * bias [[buffer(2)]], device float * y [[buffer(3)]],
                       constant ConvP & p [[buffer(4)]],
                       uint3 tgid [[threadgroup_position_in_grid]],
                       uint3 tid [[thread_position_in_threadgroup]]) {
    conv_body<half>(x, w, bias, y, p, tgid, tid.x);
}
kernel void conv1d_fwd_f(device const float * x [[buffer(0)]], device const float * w [[buffer(1)]],
                         device const float * bias [[buffer(2)]], device float * y [[buffer(3)]],
                         constant ConvP & p [[buffer(4)]],
                         uint3 tgid [[threadgroup_position_in_grid]],
                         uint3 tid [[thread_position_in_threadgroup]]) {
    conv_body<float>(x, w, bias, y, p, tgid, tid.x);
}
