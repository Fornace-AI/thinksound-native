// vae.cpp — Vocos conv-decoder on Metal (f32 weights via conv1d_fwd_f).
#include "models.h"
#include <cstring>

namespace tn {

bool VaeModel::load(Runtime & rt, const std::string & path) {
    W = std::make_unique<WeightSet>();
    // VAE weights stay f32 (decoder fidelity; weight traffic is small vs activations)
    if (!W->load(rt, path, true)) return false;
    return true;
}

namespace {
inline void conv(Runtime & rt, const void * w, const void * b, const void * x, void * y,
                 uint32_t Tin, uint32_t OL, uint32_t IC, uint32_t OC, uint32_t K,
                 uint32_t stride, uint32_t pad, uint32_t dil, bool convt) {
    gpu::ConvP p{Tin, OL, IC, OC, K, stride, pad, dil, IC, OC, convt ? 1u : 0u};
    rt.dev.dispatch("conv1d_fwd_f", (OL + 31) / 32, (OC + 7) / 8, 1, 256, 1, 1,
                    {x, w, b ? b : x, y}, &p, sizeof(p));
}
} // namespace

std::vector<float> VaeModel::decode(Runtime & rt, const float * latent, uint32_t T, uint32_t & T_audio) {
    auto need = [&](const std::string & n) { return W->need(n).buf.p; };
    // stage buffers: ping-pong, sized per stage from actual weight dims
    const uint32_t strides[5] = {8, 8, 4, 4, 2};
    const uint32_t dils[3] = {1, 3, 9};
    std::vector<gpu::Buf> bufs(2);

    auto dim = [&](const std::string & n) -> std::pair<uint64_t, uint64_t> {
        const GgufTensor * t = W->file.find(n);
        if (!t) return {0, 0};
        // conv weights stored [K, IC, OC] in ggml ne order
        return {t->ne[1], t->ne[2]};   // IC, OC
    };

    uint64_t cur_c = LATENT_DIM;
    uint64_t cur_t = T;
    size_t max_elems = (size_t) T * 4096;   // upper bound for staging
    // allocate ping-pong lazily per stage below; compute sizes as we go
    auto next_buf = [&](size_t elems) -> gpu::Buf { return rt.dev.alloc(elems * 4); };

    // input upload
    gpu::Buf a = rt.dev.alloc((size_t) T * LATENT_DIM * 4);
    memcpy(a.p, latent, (size_t) T * LATENT_DIM * 4);
    gpu::Buf b = next_buf(max_elems);

    // conv1: [T,64] -> [T, C1] k7 pad3
    {
        auto [ic, oc] = dim("decoder.layers.0.weight");
        conv(rt, need("decoder.layers.0.weight"), need("decoder.layers.0.bias"),
             a.p, b.p, (uint32_t) cur_t, (uint32_t) cur_t, (uint32_t) ic, (uint32_t) oc, 7, 1, 3, 1, false);
        cur_c = oc;
    }
    std::swap(a, b);

    // 5 decoder blocks
    for (int n = 0; n < 5; ++n) {
        const std::string bp = "decoder.layers." + std::to_string(n + 1);
        const uint32_t st = strides[n];
        // snake (in a → out b)
        {
            gpu::Buf ab = rt.dev.alloc((size_t) cur_c * 2 * 4);
            memcpy(ab.p, need(bp + ".layers.0.alpha"), 0);   // placeholder, replaced below
            // alpha/beta are separate device tensors; snake_ needs one buffer
            std::vector<float> al = W->host(bp + ".layers.0.alpha");
            std::vector<float> be = W->host(bp + ".layers.0.beta");
            std::vector<float> ab2(al.size() * 2);
            memcpy(ab2.data(), al.data(), al.size() * 4);
            memcpy(ab2.data() + al.size(), be.data(), be.size() * 4);
            memcpy(ab.p, ab2.data(), ab2.size() * 4);
            uint32_t cb[2] = {(uint32_t) cur_t, (uint32_t) cur_c};
            rt.dev.dispatch("snake_", (uint32_t) (((size_t) cur_t * cur_c + 255) / 256), 1, 1,
                            256, 1, 1, {a.p, b.p, ab.p}, cb, 8);
        }
        std::swap(a, b);
        // convT: [T,C] -> [T*st, Cout]
        uint64_t oc;
        {
            auto [ic, oc2] = dim(bp + ".layers.1.weight");
            oc = oc2;
            const uint32_t K = (uint32_t) W->file.find(bp + ".layers.1.weight")->ne[0];
            const uint32_t pad = K - 1 - st / 2;
            b = next_buf((size_t) cur_t * st * oc);
            conv(rt, need(bp + ".layers.1.weight"), need(bp + ".layers.1.bias"),
                 a.p, b.p, (uint32_t) cur_t, (uint32_t)(cur_t * st), (uint32_t) cur_c,
                 (uint32_t) oc, K, st, pad, 1, true);
            cur_t *= st;
            cur_c = oc;
        }
        std::swap(a, b);
        // 3 residual units: snake+conv dil k7, snake+conv k1, add
        for (int r = 0; r < 3; ++r) {
            const std::string rp = bp + ".layers." + std::to_string(r + 2);
            gpu::Buf res = rt.dev.alloc((size_t) cur_t * cur_c * 4);
            memcpy(res.p, a.p, (size_t) cur_t * cur_c * 4);
            // snake 1
            {
                std::vector<float> al = W->host(rp + ".layers.0.alpha");
                std::vector<float> be = W->host(rp + ".layers.0.beta");
                std::vector<float> ab2(al.size() * 2);
                memcpy(ab2.data(), al.data(), al.size() * 4);
                memcpy(ab2.data() + al.size(), be.data(), be.size() * 4);
                gpu::Buf ab = rt.dev.alloc(ab2.size() * 4);
                memcpy(ab.p, ab2.data(), ab2.size() * 4);
                uint32_t cb[2] = {(uint32_t) cur_t, (uint32_t) cur_c};
                rt.dev.dispatch("snake_", (uint32_t) (((size_t) cur_t * cur_c + 255) / 256), 1, 1,
                                256, 1, 1, {a.p, b.p, ab.p}, cb, 8);
            }
            std::swap(a, b);
            auto [ic1, oc1] = dim(rp + ".layers.1.weight");
            b = next_buf((size_t) cur_t * oc1);
            conv(rt, need(rp + ".layers.1.weight"), need(rp + ".layers.1.bias"),
                 a.p, b.p, (uint32_t) cur_t, (uint32_t) cur_t, (uint32_t) ic1, (uint32_t) oc1,
                 7, 1, 3 * dils[r], dils[r], false);
            std::swap(a, b);
            // snake 2 + conv k1
            {
                std::vector<float> al = W->host(rp + ".layers.2.alpha");
                std::vector<float> be = W->host(rp + ".layers.2.beta");
                std::vector<float> ab2(al.size() * 2);
                memcpy(ab2.data(), al.data(), al.size() * 4);
                memcpy(ab2.data() + al.size(), be.data(), be.size() * 4);
                gpu::Buf ab = rt.dev.alloc(ab2.size() * 4);
                memcpy(ab.p, ab2.data(), ab2.size() * 4);
                uint32_t cb[2] = {(uint32_t) cur_t, (uint32_t) cur_c};
                rt.dev.dispatch("snake_", (uint32_t) (((size_t) cur_t * cur_c + 255) / 256), 1, 1,
                                256, 1, 1, {a.p, b.p, ab.p}, cb, 8);
            }
            std::swap(a, b);
            b = next_buf((size_t) cur_t * oc1);
            conv(rt, need(rp + ".layers.3.weight"), need(rp + ".layers.3.bias"),
                 a.p, b.p, (uint32_t) cur_t, (uint32_t) cur_t, (uint32_t) oc1, (uint32_t) oc1,
                 1, 1, 0, 1, false);
            std::swap(a, b);
            // residual add
            uint32_t nn = (uint32_t) ((size_t) cur_t * cur_c);
            rt.add_(a.p, res.p, a.p, nn);
        }
    }
    // final snake + conv2 k7 no bias -> [T_audio, 2]
    {
        std::vector<float> al = W->host("decoder.layers.6.alpha");
        std::vector<float> be = W->host("decoder.layers.6.beta");
        std::vector<float> ab2(al.size() * 2);
        memcpy(ab2.data(), al.data(), al.size() * 4);
        memcpy(ab2.data() + al.size(), be.data(), be.size() * 4);
        gpu::Buf ab = rt.dev.alloc(ab2.size() * 4);
        memcpy(ab.p, ab2.data(), ab2.size() * 4);
        uint32_t cb[2] = {(uint32_t) cur_t, (uint32_t) cur_c};
        rt.dev.dispatch("snake_", (uint32_t) (((size_t) cur_t * cur_c + 255) / 256), 1, 1,
                        256, 1, 1, {a.p, b.p, ab.p}, cb, 8);
        std::swap(a, b);
        b = next_buf((size_t) cur_t * 2);
        conv(rt, need("decoder.layers.7.weight"), nullptr, a.p, b.p, (uint32_t) cur_t,
             (uint32_t) cur_t, (uint32_t) cur_c, 2, 7, 1, 3, 1, false);
        std::swap(a, b);
    }
    rt.dev.commit();
    T_audio = (uint32_t) cur_t;
    std::vector<float> out((size_t) T_audio * 2);
    memcpy(out.data(), a.p, out.size() * 4);
    return out;   // interleaved [t, ch]
}

} // namespace tn
