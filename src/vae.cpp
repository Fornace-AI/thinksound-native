// vae.cpp — Vocos conv-decoder on Metal (f32 weights via conv1d_fwd_f).
// Activations are time-major [T, C] (channel fastest). Every kernel reads one
// buffer and writes a different one: x (stream), h/g (scratch).
#include "models.h"
#include <algorithm>
#include <cstring>

namespace tn {

bool VaeModel::load(Runtime & rt, const std::string & path) {
    W = std::make_unique<WeightSet>();
    // VAE weights stay f32 (decoder fidelity; weight traffic is small vs activations)
    if (!W->load(rt, path, true)) return false;
    return true;
}

std::vector<float> VaeModel::decode(Runtime & rt, const float * latent, uint32_t T, uint32_t & T_audio) {
    auto need = [&](const std::string & n) { return W->need(n).buf.p; };
    auto tens = [&](const std::string & n) { return W->file.find(n); };   // ne = [K, IC, OC]
    const uint32_t strides[5] = {8, 8, 4, 4, 2};
    const uint32_t dils[3] = {1, 3, 9};

    // scratch sized for the largest stage: T·hop samples × widest post-upsample width
    uint32_t Ta = T;
    size_t max_elems = (size_t) T * tens("decoder.layers.0.weight")->ne[2];
    for (int n = 0; n < 5; ++n) {
        Ta *= strides[n];
        const GgufTensor * t = tens("decoder.layers." + std::to_string(n + 1) + ".layers.1.weight");
        max_elems = std::max(max_elems, (size_t) Ta * (size_t) t->ne[2]);
    }
    gpu::Buf x = rt.dev.alloc(max_elems * 4), h = rt.dev.alloc(max_elems * 4),
             g = rt.dev.alloc(max_elems * 4);
    gpu::Buf in = rt.dev.alloc((size_t) T * LATENT_DIM * 4);
    memcpy(in.p, latent, (size_t) T * LATENT_DIM * 4);

    auto conv = [&](const std::string & wn, bool bias, const void * src, void * dst,
                    uint32_t Tin, uint32_t OL, uint32_t stride, uint32_t pad, uint32_t dil, bool convt) {
        const GgufTensor * t = tens(wn + ".weight");
        const uint32_t K = (uint32_t) t->ne[0], IC = (uint32_t) t->ne[1], OC = (uint32_t) t->ne[2];
        const void * b = bias ? need(wn + ".bias") : nullptr;
        gpu::ConvP p{Tin, OL, IC, OC, K, stride, pad, dil, IC, OC, convt ? 1u : 0u, b ? 1u : 0u};
        rt.dev.dispatch("conv1d_fwd_f", (OL + 31) / 32, (OC + 7) / 8, 1, 256, 1, 1,
                        {src, need(wn + ".weight"), b ? b : src, dst}, &p, sizeof(p));
        return OC;
    };
    std::vector<gpu::Buf> abs;   // alpha|beta tables, alive until commit
    auto snake = [&](const std::string & pre, const void * src, void * dst, uint32_t S, uint32_t C) {
        std::vector<float> al = W->host(pre + ".alpha"), be = W->host(pre + ".beta");
        gpu::Buf ab = rt.dev.alloc((size_t) C * 2 * 4);
        memcpy(ab.p, al.data(), (size_t) C * 4);
        memcpy((float *) ab.p + C, be.data(), (size_t) C * 4);
        abs.push_back(ab);
        uint32_t cb[2] = {S, C};
        rt.dev.dispatch("snake_", (uint32_t) (((size_t) S * C + 255) / 256), 1, 1, 256, 1, 1,
                        {src, dst, ab.p}, cb, 8);
    };

    // conv1: [T,64] -> [T,2048] k7 pad3
    uint32_t cur_t = T;
    uint32_t cur_c = conv("decoder.layers.0", true, in.p, x.p, T, T, 1, 3, 1, false);

    for (int n = 0; n < 5; ++n) {
        const std::string bp = "decoder.layers." + std::to_string(n + 1);
        const uint32_t st = strides[n];
        // snake -> transposed conv (zero-stuff + stride-1 conv with the converted kernel)
        snake(bp + ".layers.0", x.p, h.p, cur_t, cur_c);
        const uint32_t K = (uint32_t) tens(bp + ".layers.1.weight")->ne[0];
        cur_c = conv(bp + ".layers.1", true, h.p, x.p, cur_t, cur_t * st, st, K - 1 - st / 2, 1, true);
        cur_t *= st;
        // 3 residual units: x += conv1(snake(conv7_dil(snake(x))))
        for (int r = 0; r < 3; ++r) {
            const std::string rp = bp + ".layers." + std::to_string(r + 2);
            snake(rp + ".layers.0", x.p, h.p, cur_t, cur_c);
            conv(rp + ".layers.1", true, h.p, g.p, cur_t, cur_t, 1, 3 * dils[r], dils[r], false);
            snake(rp + ".layers.2", g.p, h.p, cur_t, cur_c);
            conv(rp + ".layers.3", true, h.p, g.p, cur_t, cur_t, 1, 0, 1, false);
            rt.add_(x.p, g.p, x.p, (size_t) cur_t * cur_c);
        }
    }
    // final snake + conv k7 (no bias) -> [T_audio, 2]
    snake("decoder.layers.6", x.p, h.p, cur_t, cur_c);
    conv("decoder.layers.7", false, h.p, g.p, cur_t, cur_t, 1, 3, 1, false);
    rt.dev.commit();
    T_audio = cur_t;
    std::vector<float> out((size_t) T_audio * 2);
    memcpy(out.data(), g.p, out.size() * 4);
    return out;   // interleaved [t, ch]
}

} // namespace tn
