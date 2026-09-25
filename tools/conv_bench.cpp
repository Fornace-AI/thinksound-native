// tools/conv_bench.cpp — conv1d candidates on the real DiT/VAE shapes: scalar conv1d_fwd(_f)
// vs Runtime::conv (im2col + MPP GEMM). Correctness vs CPU double on sampled rows, GPU ms/call.
//   ts-conv-bench [metallib]
#include "net.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct Case { uint32_t T, OL, IC, OC, K, stride, pad, dil; bool convt, f16, bias; const char * tag; };

static const Case CASES[] = {
    {172, 172, 1024, 2816, 3, 1, 1, 1, false, true, false, "DiT latent ffn w1/w3 k3"},
    {172, 172, 2816, 1024, 3, 1, 1, 1, false, true, false, "DiT latent ffn w2 k3"},
    {172, 172, 1024, 1024, 3, 1, 1, 1, false, true, true, "DiT latent linear1 k3"},
    {64, 64, 1024, 2816, 3, 1, 1, 1, false, true, false, "DiT clip ffn w1 k3"},
    {172, 172, 64, 1024, 3, 1, 1, 1, false, true, true, "DiT audio_input_proj.0 k3"},
    {172, 172, 1024, 64, 7, 1, 3, 1, false, true, true, "DiT final conv k7"},
    {172, 1376, 2048, 1024, 16, 8, 11, 1, true, false, true, "VAE convT stage1 x8"},
    {1376, 1376, 1024, 1024, 7, 1, 27, 9, false, false, true, "VAE stage1 k7 dil9"},
    {352256, 352256, 128, 128, 7, 1, 9, 3, false, false, true, "VAE stage5 k7 dil3"},
    {352256, 352256, 128, 128, 1, 1, 0, 1, false, false, true, "VAE stage5 k1"},
    {352256, 352256, 128, 2, 7, 1, 3, 1, false, false, false, "VAE final k7 ->2"},
};

int main(int argc, char ** argv) {
    tn::Runtime rt;
    if (!rt.init(argc > 1 ? argv[1] : "build/thinksound.metallib")) return 1;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    printf("%-44s %-7s %9s %8s %10s\n", "case (OLxOCxIC*K)", "cand", "ms/call", "TFLOP/s", "rel-err");
    for (const Case & c : CASES) {
        const size_t nw = (size_t) c.OC * c.IC * c.K;
        std::vector<float> x((size_t) c.T * c.IC), w(nw), b(c.OC);
        for (auto & v : x) v = nd(rng);
        for (auto & v : w) v = nd(rng) * 0.03f;
        for (auto & v : b) v = nd(rng);
        std::vector<_Float16> wh(nw);
        for (size_t i = 0; i < nw; ++i) { wh[i] = (_Float16) w[i]; if (c.f16) w[i] = (float) wh[i]; }
        tn::gpu::Buf X = rt.dev.alloc(x.size() * 4), W = rt.dev.alloc(nw * (c.f16 ? 2 : 4));
        tn::gpu::Buf B = rt.dev.alloc(b.size() * 4), Y = rt.dev.alloc((size_t) c.OL * c.OC * 4);
        memcpy(X.p, x.data(), x.size() * 4);
        memcpy(B.p, b.data(), b.size() * 4);
        if (c.f16) memcpy(W.p, wh.data(), nw * 2); else memcpy(W.p, w.data(), nw * 4);
        // CPU reference, 3 sampled output rows
        const uint32_t rows[3] = {0, c.OL / 2 + 1, c.OL - 1};
        std::vector<double> ref(3 * c.OC);
        for (int ri = 0; ri < 3; ++ri)
            for (uint32_t oc = 0; oc < c.OC; ++oc) {
                double a = c.bias ? b[oc] : 0.0;
                for (uint32_t ic = 0; ic < c.IC; ++ic)
                    for (uint32_t k = 0; k < c.K; ++k) {
                        int ti;
                        if (c.convt) {
                            int ui = (int) rows[ri] - (int) c.pad + (int) k;
                            if (ui < 0 || ui % (int) c.stride) continue;
                            ti = ui / (int) c.stride;
                        } else ti = (int) (rows[ri] * c.stride) - (int) c.pad + (int) (k * c.dil);
                        if (ti < 0 || ti >= (int) c.T) continue;
                        a += (double) x[(size_t) ti * c.IC + ic] * w[((size_t) oc * c.IC + ic) * c.K + k];
                    }
                ref[ri * c.OC + oc] = a;
            }
        const double flops = 2.0 * c.OL * c.OC * c.IC * c.K / (c.convt ? c.stride : 1);
        const char * cname[3] = {"scalar", "im2col", "i2c32x64"};
        for (int cand = 0; cand < (c.f16 ? 3 : 2); ++cand) {
            auto run = [&]() {
                if (cand == 0) {
                    tn::gpu::ConvP p{c.T, c.OL, c.IC, c.OC, c.K, c.stride, c.pad, c.dil, c.IC, c.OC,
                                     c.convt ? 1u : 0u, c.bias ? 1u : 0u};
                    rt.dev.dispatch(c.f16 ? "conv1d_fwd" : "conv1d_fwd_f", (c.OL + 31) / 32, (c.OC + 7) / 8, 1,
                                    256, 1, 1, {X.p, W.p, B.p, Y.p}, &p, sizeof(p));
                } else if (cand == 1) {
                    rt.conv(c.f16, W.p, c.bias ? B.p : nullptr, X.p, Y.p, c.T, c.OL, c.IC, c.OC, c.K,
                            c.stride, c.pad, c.dil, c.convt);
                } else {   // same im2col, MPP 32x64 tile GEMM (single chunk: DiT shapes only)
                    const uint32_t KC = c.IC * c.K;
                    if (rt.col.bytes < (size_t) c.OL * KC * 4) rt.col = rt.dev.alloc((size_t) c.OL * KC * 4);
                    struct { uint32_t T, rows, IC, K, stride, pad, dil, ldx, convt, t0; } ib{
                        c.T, c.OL, c.IC, c.K, c.stride, c.pad, c.dil, c.IC, 0, 0};
                    rt.dev.dispatch("im2col", (uint32_t) (((size_t) c.OL * KC + 255) / 256), 1, 1, 256, 1, 1,
                                    {X.p, rt.col.p}, &ib, sizeof(ib));
                    tn::gpu::GemmP gp{c.OL, c.OC, KC, KC, KC, c.OC, c.bias ? 1u : 0u, 0};
                    rt.dev.dispatch("gemm_mpp_h_32x64", (c.OC + 63) / 64, (c.OL + 31) / 32, 1,
                                    4 * rt.dev.exec_width("gemm_mpp_h_32x64"), 1, 1,
                                    {W.p, rt.col.p, Y.p, c.bias ? B.p : W.p}, &gp, sizeof(gp));
                }
            };
            memset(Y.p, 0xff, Y.bytes);
            run(); rt.dev.commit();
            double dd = 0, rr = 0;
            for (int ri = 0; ri < 3; ++ri)
                for (uint32_t oc = 0; oc < c.OC; ++oc) {
                    double d = ((float *) Y.p)[(size_t) rows[ri] * c.OC + oc] - ref[ri * c.OC + oc];
                    dd += std::isnan(d) ? 1e30 : d * d; rr += ref[ri * c.OC + oc] * ref[ri * c.OC + oc];
                }
            const int reps = (int) std::max(2.0, std::min(100.0, 2e10 / flops));
            double best = 1e30;
            for (int t = 0; t < 3; ++t) {
                for (int r = 0; r < reps; ++r) run();
                rt.dev.commit();
                best = std::min(best, rt.dev.last_gpu_ms() / reps);
            }
            char tag[96];
            snprintf(tag, sizeof tag, "%s %ux%ux%u", c.tag, c.OL, c.OC, c.IC * c.K);
            printf("%-44s %-8s %9.3f %8.3f %10.3g\n", tag, cname[cand], best,
                   flops / (best * 1e9), std::sqrt(dd / rr));
            fflush(stdout);
        }
    }
    return 0;
}
