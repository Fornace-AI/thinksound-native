// net.mm — runtime + weight store implementation.
#include "net.h"
#include <cstring>

namespace tn {

static uint32_t grid(size_t n, uint32_t tile) {
    return (uint32_t) ((n + tile - 1) / tile);
}

void Runtime::gemm(bool wf16, const void * W, const void * X, void * Y, const void * bias,
                   const GemmParams & p) {
    gpu::GemmP cb{p.M, p.N, p.K, p.ldx ? p.ldx : p.K, p.ldw ? p.ldw : p.K,
                  p.ldy ? p.ldy : p.N, 0, 0};
    dev.dispatch(wf16 ? "gemm_h" : "gemm_f",
                 grid(p.N, 32), grid(p.M, 8), 1, 256, 1, 1,
                 {W, X, Y, bias ? bias : W}, &cb, sizeof(cb));
}

void Runtime::gemv(const void * W, const void * X, void * Y, const void * bias,
                   uint32_t N, uint32_t K, uint32_t ldw, bool wf16) {
    gpu::GemmP cb{1, N, K, K, ldw ? ldw : K, N, 0, 0};
    dev.dispatch(wf16 ? "gemv_h" : "gemv_f",
                 grid(N, 64), 1, 1, 64, 1, 1, {W, X, Y, bias ? bias : W}, &cb, sizeof(cb));
}

void Runtime::silu(void * x, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("silu_", grid(n, 256), 1, 1, 256, 1, 1, {x}, cb, 4);
}
void Runtime::copy(const void * x, void * y, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("copy_", grid(n, 256), 1, 1, 256, 1, 1, {x, y}, cb, 4);
}
void Runtime::gelu(void * x, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("gelu_", grid(n, 256), 1, 1, 256, 1, 1, {x}, cb, 4);
}
void Runtime::gelu_quick(void * x, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("gelu_quick_", grid(n, 256), 1, 1, 256, 1, 1, {x}, cb, 4);
}

void Runtime::swiglu(const void * a, const void * b, void * y, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("swiglu_", grid(n, 256), 1, 1, 256, 1, 1, {a, b, y}, cb, 4);
}
void Runtime::add_(const void * a, const void * b, void * y, size_t n) {
    uint32_t cb[1] = {(uint32_t) n};
    dev.dispatch("add_", grid(n, 256), 1, 1, 256, 1, 1, {a, b, y}, cb, 4);
}
void Runtime::add_bcast_col(const void * x, const void * v, void * y, uint32_t S, uint32_t H) {
    uint32_t cb[2] = {H, S * H};
    dev.dispatch("add_bcast_col", grid((size_t) S * H, 256), 1, 1, 256, 1, 1, {x, v, y}, cb, 8);
}
void Runtime::ln_plain(const void * x, void * y, uint32_t S, uint32_t H, float eps) {
    struct { uint32_t H; float eps; } cb{H, eps};
    dev.dispatch("ln_plain", S, 1, 1, 1, 1, 1, {x, y}, &cb, sizeof(cb));
}
void Runtime::ln_wb(const void * x, void * y, const void * w, const void * b, uint32_t S, uint32_t H, float eps) {
    struct { uint32_t H; float eps; } cb{H, eps};
    dev.dispatch("ln_wb", S, 1, 1, 1, 1, 1, {x, y, w, b}, &cb, sizeof(cb));
}
void Runtime::rms_w(const void * x, void * y, const void * w, uint32_t S, uint32_t H, float eps) {
    struct { uint32_t H; float eps; } cb{H, eps};
    dev.dispatch("rms_w", S, 1, 1, 1, 1, 1, {x, y, w}, &cb, sizeof(cb));
}
void Runtime::cfg_euler(const void * fc, const void * fu, void * x, size_t n, float cfg, float dt) {
    struct { float cfg; float dt; uint32_t n; } cb{cfg, dt, (uint32_t) n};
    dev.dispatch("cfg_euler", grid(n, 256), 1, 1, 256, 1, 1, {fc, fu, x}, &cb, sizeof(cb));
}
void Runtime::mean_cols(const void * x, void * y, uint32_t S, uint32_t H) {
    uint32_t cb[2] = {S, H};
    dev.dispatch("mean_cols", grid(H, 8), 1, 1, 256, 1, 1, {x, y}, cb, 8);
}

// ---- WeightSet ----

const WeightSet::Ent & WeightSet::need(const std::string & name) const {
    auto it = w.find(name);
    if (it == w.end()) {
        fprintf(stderr, "weights: missing tensor %s\n", name.c_str());
        static Ent empty{};
        return empty;
    }
    return it->second;
}

std::vector<float> WeightSet::host(const std::string & name) const {
    const GgufTensor * t = file.find(name);
    if (!t) { fprintf(stderr, "weights: missing tensor %s (host)\n", name.c_str()); return {}; }
    return to_f32(*t);
}

bool WeightSet::load(Runtime & rt, const std::string & path, bool keep_f32) {
    if (!file.open(path)) return false;
    // materialize every tensor as a device buffer
    for (const auto & [name, t] : file.tensors_map()) {
        bool f32 = (keep_f32 && t.type == ML_F32) || (t.type == ML_F32 && t.n_dims <= 1);
        Ent e;
        e.f16 = !f32;
        e.n = t.n_dims >= 2 ? t.ne[1] : 1;
        e.k = t.n_dims >= 1 ? t.ne[0] : 1;
        if (f32) {
            std::vector<float> v = to_f32(t);
            e.buf = rt.dev.alloc(v.size() * 4);
            if (!e.buf.ok()) return false;
            memcpy(e.buf.p, v.data(), v.size() * 4);
        } else {
            // convert to f16 (bf16/f16/q8_0 -> f32 -> f16)
            std::vector<float> v = to_f32(t);
            std::vector<uint16_t> h(v.size());
            auto f2h = [](float f) -> uint16_t {
                uint32_t x; memcpy(&x, &f, 4);
                uint32_t sign = (x >> 16) & 0x8000;
                int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
                uint32_t man = x & 0x7FFFFF;
                if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0));
                if (((x >> 23) & 0xFF) == 0) return (uint16_t) sign;   // flush zero/denormal
                if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);     // overflow -> inf
                if (exp <= 0) {                                        // underflow
                    if (exp < -10) return (uint16_t) sign;
                    man |= 0x800000;
                    uint32_t shift = (uint32_t)(1 - exp);
                    uint32_t dm = man >> (shift + 13);
                    return (uint16_t)(sign | dm);
                }
                // round to nearest even on the 13 dropped bits
                uint16_t base = (uint16_t)(sign | ((uint32_t) exp << 10) | (man >> 13));
                uint32_t rem = man & 0x1FFF;
                if (rem > 0x1000 || (rem == 0x1000 && (base & 1))) base += 1;
                return base;
            };
            for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
            e.buf = rt.dev.alloc(h.size() * 2);
            if (!e.buf.ok()) return false;
            memcpy(e.buf.p, h.data(), h.size() * 2);
        }
        w[name] = e;
    }
    return true;
}

} // namespace tn
