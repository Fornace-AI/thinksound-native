// net.h — runtime helpers over gpu::Device: weight store, dispatch wrappers.
#pragma once
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "gguf.h"
#include "gpu.h"

namespace tn {

struct GemmParams {          // host mirror of GemmP
    uint32_t M, N, K;
    uint32_t ldx, ldw, ldy;
};

class Runtime {
public:
    gpu::Device dev;

    bool init(const std::string & metallib) { return dev.init(metallib); }

    // Y[M,N] = X[M,K] · W[N,K]^T (+bias). wf16 selects half/f32 weights.
    void gemm(bool wf16, const void * W, const void * X, void * Y, const void * bias,
              const GemmParams & p);
    void gemv(const void * W, const void * X, void * Y, const void * bias,
              uint32_t N, uint32_t K, uint32_t ldw, bool wf16 = true);
    // conv1d as GEMM: im2col(x [T,IC]) -> [OL, IC*K], times W [OC, IC*K] (gguf [K,IC,OC]).
    // y [OL,OC] (+bias). convt: zero-stuffed input, stride-1 conv with the converted kernel.
    void conv(bool wf16, const void * W, const void * bias, const void * x, void * y,
              uint32_t T, uint32_t OL, uint32_t IC, uint32_t OC, uint32_t K,
              uint32_t stride, uint32_t pad, uint32_t dil, bool convt = false);
    gpu::Buf col;                          // im2col scratch (grown on demand, reused)
    uint32_t simd_w_h = 0, simd_w_f = 0;   // cached threadExecutionWidth of the gemm kernels

    // one activation row buffer [S,H] (elementwise kernels)
    void silu(void * x, size_t n);
    void copy(const void * x, void * y, size_t n);
    void geglu(const void * a, const void * b, void * y, size_t n);   // gelu_tanh(a)·b
    void gelu_quick(void * x, size_t n);
    void swiglu(const void * a, const void * b, void * y, size_t n);
    void add_(const void * a, const void * b, void * y, size_t n);
    void add_bcast_col(const void * x, const void * v, void * y, uint32_t S, uint32_t H);
    void ln_plain(const void * x, void * y, uint32_t S, uint32_t H, float eps);
    void ln_wb(const void * x, void * y, const void * w, const void * b, uint32_t S, uint32_t H, float eps);
    void rms_w(const void * x, void * y, const void * w, uint32_t S, uint32_t H, float eps);
    // cfg combine + euler: x += dt·(fu + cfg·(fc−fu))
    void cfg_euler(const void * fc, const void * fu, void * x, size_t n, float cfg, float dt);
    void mean_cols(const void * x, void * y, uint32_t S, uint32_t H);
};

// Loaded weight set from one gguf: device buffers, f16 or f32 (round-trip safe).
struct WeightSet {
    GgufFile file;                        // keeps mapping alive (mmap)
    struct Ent {
        gpu::Buf buf;
        bool f16 = true;
        uint64_t n = 0, k = 0;            // rows (out), cols (in) — row-major
    };
    std::map<std::string, Ent> w;
    const std::map<std::string, Ent> & ents() const { return w; }

    // Load: name → [N,K] row-major device buffer. f32 tensors are kept f32 when
    // keep_f32 (encoders); converted to f16 otherwise (DiT/VAE come bf16 → f16).
    bool load(Runtime & rt, const std::string & path, bool keep_f32);
    const Ent * find(const std::string & name) const {
        auto it = w.find(name);
        return it == w.end() ? nullptr : &it->second;
    }
    const Ent & need(const std::string & name) const;
    bool is_f16(const std::string & name) const {
        const Ent * e = find(name); return e && e->f16;
    }
    // host f32 copy of one tensor (for constants: embeddings, empties)
    std::vector<float> host(const std::string & name) const;
    int64_t ki64(const std::string & key, int64_t def) const { return file.get_i64(key, def); }
    double kf64(const std::string & key, double def) const { return file.get_f64(key, def); }
};

} // namespace tn
