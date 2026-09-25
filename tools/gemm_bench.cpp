// tools/gemm_bench.cpp — GEMM candidates on the real model shapes: correctness vs a CPU
// double reference (sampled rows) and GPU time per call.
//   ts-gemm-bench [metallib] [candidate,...]
// Candidates: scalar (gemm_h/gemm_f), tiled (gemm_t_*), mpp (gemm_mpp_*), mpp_rel,
// mpp_32x64, mpp_64x64, gemv (M=1 only).
#include "gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using tn::gpu::Buf;
using tn::gpu::Device;

struct Shape { uint32_t M, N, K; bool f16; bool bias; uint32_t xpad, ypad, yoff; float xscale; const char * tag; };
struct Cand { const char * name; const char * kh; const char * kf; uint32_t tm, tn, sg; };   // tile M, tile N, simdgroups (0 = 256-thread scalar)

static const Cand CANDS[] = {
    {"scalar", "gemm_h", "gemm_f", 8, 32, 0},
    {"tiled", "gemm_t_h", "gemm_t_f", 32, 64, 4},
    {"tiled64", "gemm_t64_h", nullptr, 64, 64, 4},
    {"mpp", "gemm_mpp_h", "gemm_mpp_f", 64, 32, 4},
    {"mpp_rel", "gemm_mpp_h_rel", nullptr, 64, 32, 4},
    {"mpp_32x64", "gemm_mpp_h_32x64", nullptr, 32, 64, 4},
    {"mpp_64x64", "gemm_mpp_h_64x64", nullptr, 64, 64, 4},
    {"gemv", "gemv_h", "gemv_f", 1, 64, 0},
};

static const Shape SHAPES[] = {
    {1024, 1024, 1024, true, true, 0, 0, 0, 1.0f, "square 1024^3"},
    {172, 6144, 1024, true, true, 0, 0, 0, 1.0f, "DiT adaLN latent"},
    {172, 3072, 1024, true, true, 0, 0, 0, 1.0f, "DiT qkv latent"},
    {154, 2816, 1024, true, false, 0, 0, 0, 1.0f, "DiT text ffn w1"},
    {154, 1024, 2816, true, false, 0, 0, 0, 1.0f, "DiT text ffn w2"},
    {43, 1024, 3072, true, true, 0, 0, 0, 1.0f, "brief 43x1024x3072"},
    {390, 1152, 1152, true, true, 0, 0, 0, 1.0f, "brief 390x1152^2"},
    {77, 2048, 2048, true, false, 0, 4096, 2048, 3e4f, "T5 v (ldy 3H, off 2H, |x|~1e5)"},
    {77, 5120, 2048, true, false, 0, 0, 0, 3e4f, "T5 wi (|x|~1e5)"},
    {77, 1024, 1024, false, true, 0, 2048, 1024, 1.0f, "CLIP k f32 (ldy 3H)"},
    {77, 4096, 1024, false, true, 0, 0, 0, 1.0f, "CLIP fc1 f32"},
    {16, 1024, 1024, true, true, 64, 0, 0, 1.0f, "clip_S=16 (ldx pad)"},
    {1, 6144, 1024, true, true, 0, 0, 0, 1.0f, "M=1 adaLN clip/text"},
    {1, 1024, 1024, true, true, 0, 0, 0, 1.0f, "M=1 t_embed"},
};

struct GemmP { uint32_t M, N, K, ldx, ldw, ldy, bias_mode, pad0; };

int main(int argc, char ** argv) {
    const std::string lib = argc > 1 ? argv[1] : "build/thinksound.metallib";
    std::string only = argc > 2 ? argv[2] : "";
    Device dev;
    if (!dev.init(lib)) return 1;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    printf("%-34s %-10s %9s %8s %10s\n", "shape (MxNxK)", "cand", "ms/call", "TFLOP/s", "rel-err");
    for (const Shape & s : SHAPES) {
        const uint32_t ldx = s.K + s.xpad, ldy = s.N + s.ypad;
        std::vector<float> x((size_t) s.M * ldx), w((size_t) s.N * s.K), b(s.N);
        for (auto & v : x) v = nd(rng) * s.xscale;
        for (auto & v : w) v = nd(rng) * 0.03f;
        for (auto & v : b) v = nd(rng);
        std::vector<_Float16> wh(w.size());
        for (size_t i = 0; i < w.size(); ++i) { wh[i] = (_Float16) w[i]; if (s.f16) w[i] = (float) wh[i]; }
        Buf X = dev.alloc(x.size() * 4), Wb = dev.alloc(w.size() * (s.f16 ? 2 : 4)), B = dev.alloc(b.size() * 4);
        Buf Y = dev.alloc(((size_t) s.M * ldy + s.yoff) * 4);
        memcpy(X.p, x.data(), x.size() * 4);
        memcpy(B.p, b.data(), b.size() * 4);
        if (s.f16) memcpy(Wb.p, wh.data(), wh.size() * 2); else memcpy(Wb.p, w.data(), w.size() * 4);
        float * y = (float *) Y.p + s.yoff;
        // CPU reference on sampled rows
        std::vector<uint32_t> rows = {0, s.M / 2, s.M - 1};
        std::vector<double> ref(rows.size() * s.N);
        for (size_t ri = 0; ri < rows.size(); ++ri)
            for (uint32_t n = 0; n < s.N; ++n) {
                double a = s.bias ? b[n] : 0.0;
                for (uint32_t k = 0; k < s.K; ++k) a += (double) x[(size_t) rows[ri] * ldx + k] * w[(size_t) n * s.K + k];
                ref[ri * s.N + n] = a;
            }
        const double flops = 2.0 * s.M * s.N * s.K;
        GemmP p{s.M, s.N, s.K, ldx, s.K, ldy, s.bias ? 1u : 0u, 0};
        char tag[64];
        snprintf(tag, sizeof tag, "%s %ux%ux%u", s.tag, s.M, s.N, s.K);
        for (const Cand & c : CANDS) {
            if (!only.empty() && only.find(c.name) == std::string::npos) continue;
            const char * k = s.f16 ? c.kh : c.kf;
            if (!k) continue;
            const bool gemv = !strcmp(c.name, "gemv");
            if (gemv && s.M != 1) continue;
            uint32_t tx = c.sg ? dev.exec_width(k) * c.sg : (gemv ? 64 : 256);
            uint32_t gx = (s.N + c.tn - 1) / c.tn, gy = gemv ? 1 : (s.M + c.tm - 1) / c.tm;
            auto run = [&]() { dev.dispatch(k, gx, gy, 1, tx, 1, 1, {Wb.p, X.p, (void *) y, B.p}, &p, sizeof(p)); };
            memset(Y.p, 0xff, Y.bytes);   // NaN fill: unwritten outputs show up as errors
            run(); dev.commit();
            double dd = 0, rr = 0;
            for (size_t ri = 0; ri < rows.size(); ++ri)
                for (uint32_t n = 0; n < s.N; ++n) {
                    double d = y[(size_t) rows[ri] * ldy + n] - ref[ri * s.N + n];
                    dd += std::isnan(d) ? 1e30 : d * d; rr += ref[ri * s.N + n] * ref[ri * s.N + n];
                }
            // guard the strided output: columns beyond N / before yoff must be untouched (NaN)
            bool clobber = false;
            if (s.ypad && !std::isnan(y[(size_t) 0 * ldy + s.N])) clobber = true;
            if (s.yoff && !std::isnan(((float *) Y.p)[s.yoff - 1])) clobber = true;
            const int reps = (int) std::max(3.0, std::min(200.0, 4e10 / flops));
            run(); dev.commit();
            double best = 1e30;
            for (int t = 0; t < 3; ++t) {
                for (int r = 0; r < reps; ++r) run();
                dev.commit();
                best = std::min(best, dev.last_gpu_ms() / reps);
            }
            printf("%-34s %-10s %9.4f %8.3f %10.3g%s\n", tag, c.name, best, flops / (best * 1e9),
                   std::sqrt(dd / rr), clobber ? "  CLOBBERED-PADDING" : "");
            fflush(stdout);
        }
    }
    return 0;
}
