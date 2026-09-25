// tools/conv_test.cpp — isolate conv1d_fwd correctness on the real audio_input_proj.0 weight.
#include "net.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    tn::Runtime rt;
    if (!rt.init("/tmp/ts-weights/thinksound.metallib")) { printf("metallib fail\n"); return 1; }
    tn::WeightSet W;
    if (!W.load(rt, "/tmp/ts-weights/dit-bf16.gguf", false)) return 1;

    const uint32_t T = 43, IC = 64, OC = 1024, K = 3, pad = 1;
    // deterministic input: x[t,ic] = sin(t + ic)
    std::vector<float> xh((size_t) T * IC);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t ic = 0; ic < IC; ++ic)
            xh[(size_t) t * IC + ic] = sinf((float) t + (float) ic * 0.37f);
    tn::gpu::Buf x = rt.dev.alloc(xh.size() * 4);
    memcpy(x.p, xh.data(), xh.size() * 4);
    tn::gpu::Buf y = rt.dev.alloc((size_t) T * OC * 4);

    const tn::WeightSet::Ent & we = W.need("audio_input_proj.0.weight");
    const tn::WeightSet::Ent & be = W.need("audio_input_proj.0.bias");
    printf("weight ent: n=%u k=%u f16=%d\n", we.n, we.k, (int) we.f16);

    tn::gpu::ConvP p{T, T, IC, OC, K, 1, pad, 1, IC, OC, 0};
    rt.dev.dispatch("conv1d_fwd", (T + 31) / 32, (OC + 7) / 8, 1, 256, 1, 1,
                    {x.p, we.buf.p, be.buf.p, y.p}, &p, sizeof(p));
    rt.dev.commit();

    // print row 0, first 8 channels
    float * yp = (float *) y.p;
    printf("y[0][0..7]:");
    for (int i = 0; i < 8; ++i) printf(" %.4f", yp[i]);
    printf("\ny[5][0..7]:");
    for (int i = 0; i < 8; ++i) printf(" %.4f", yp[(size_t) 5 * OC + i]);
    printf("\n");

    // host reference from the same loaded weights
    std::vector<float> w = W.host("audio_input_proj.0.weight");   // gguf flat: k + K*ic + K*IC*oc
    std::vector<float> b = W.host("audio_input_proj.0.bias");
    printf("host ref y[0][0..7]:");
    for (int i = 0; i < 8; ++i) {
        float acc = b[i];
        for (uint32_t ic = 0; ic < IC; ++ic)
            for (uint32_t k = 0; k < K; ++k) {
                int ti = (int) 0 - (int) pad + (int) k;
                if (ti < 0 || ti >= (int) T) continue;
                acc += xh[(size_t) ti * IC + ic] * w[((size_t) i * IC + ic) * K + k];
            }
        printf(" %.4f", acc);
    }
    printf("\nhost ref y[5][0..7]:");
    for (int i = 0; i < 8; ++i) {
        float acc = b[i];
        for (uint32_t ic = 0; ic < IC; ++ic)
            for (uint32_t k = 0; k < K; ++k) {
                int ti = (int) 5 - (int) pad + (int) k;
                if (ti < 0 || ti >= (int) T) continue;
                acc += xh[(size_t) ti * IC + ic] * w[((size_t) i * IC + ic) * K + k];
            }
        printf(" %.4f", acc);
    }
    printf("\n");
    return 0;
}
