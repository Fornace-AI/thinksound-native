// pipeline.cpp — top-level text->audio generator (orchestrates models + writes WAV).
#include "models.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace tn {

namespace {
bool write_wav_file(const std::string & path, const float * inter, uint32_t T, int sr) {
    // planar conversion for peak normalize
    std::vector<float> L(T), R(T);
    float peak = 0.0f;
    for (uint32_t i = 0; i < T; ++i) {
        L[i] = inter[2 * i + 0]; R[i] = inter[2 * i + 1];
        peak = std::max(peak, std::max(std::abs(L[i]), std::abs(R[i])));
    }
    float scale = peak > 0.0f ? 1.0f / peak : 1.0f;
    const uint32_t data_bytes = T * 4;   // 2ch * 16bit = 4 bytes/sample
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(2);
    w32((uint32_t) sr); w32((uint32_t)(sr * 4)); w16(4); w16(16);
    fwrite("data", 1, 4, f); w32(data_bytes);
    for (uint32_t i = 0; i < T; ++i) {
        float l = std::max(-1.0f, std::min(1.0f, L[i] * scale));
        float r = std::max(-1.0f, std::min(1.0f, R[i] * scale));
        w16((uint16_t)(int16_t) std::lround(l * 32767.0f));
        w16((uint16_t)(int16_t) std::lround(r * 32767.0f));
    }
    fclose(f);
    return true;
}
} // namespace

int generate_wav(const GenOpts & o, const std::string & out_path, std::string & err) {
    Runtime rt;
    std::string metallib = o.dir + "/thinksound.metallib";
    if (!rt.init(metallib)) { err = "cannot load " + metallib; return -1; }

    const std::string dir = o.dir;
    ClipTokenizer clip_tok;
    if (!clip_tok.load(dir + "/clip-tokenizer.gguf")) { err = "clip-tokenizer load"; return -1; }
    T5Tokenizer t5_tok;
    if (!t5_tok.load(dir + "/t5-tokenizer.gguf")) { err = "t5-tokenizer load"; return -1; }

    auto clip_ids = clip_tok.encode(o.caption, CLIP_TOKEN_CTX);
    auto t5_ids = t5_tok.encode(o.cot, T5_TOKEN_CTX, true);

    ClipModel clip;
    if (!clip.load(rt, dir + "/metaclip-text-f32.gguf")) { err = "clip load"; return -1; }
    gpu::Buf clip_toks = rt.dev.alloc((size_t) CLIP_TOKEN_CTX * 1024 * 4);
    gpu::Buf clip_glob = rt.dev.alloc(1024 * 4);
    clip.encode(rt, clip_ids, clip_toks, clip_glob);

    T5Model t5;
    if (!t5.load(rt, dir + "/t5-q8.gguf")) { err = "t5 load"; return -1; }
    gpu::Buf t5_toks = rt.dev.alloc((size_t) T5_TOKEN_CTX * 2048 * 4);
    t5.encode(rt, t5_ids, t5_toks);
    rt.dev.commit();
    if (getenv("TN_TRACE")) {
        auto stat = [](const char * n, const gpu::Buf & b, size_t cnt) {
            double ss = 0; size_t nan = 0; const float * p = (const float *) b.p;
            for (size_t i = 0; i < cnt; ++i) { if (std::isnan(p[i])) ++nan; else ss += (double) p[i] * p[i]; }
            fprintf(stderr, "cond %-12s n=%zu nan=%zu |v|=%.3f\n", n, cnt, nan, std::sqrt(ss));
        };
        stat("clip_toks", clip_toks, (size_t) CLIP_TOKEN_CTX * 1024);
        stat("clip_glob", clip_glob, 1024);
        stat("t5_toks", t5_toks, (size_t) T5_TOKEN_CTX * 2048);
    }

    DitModel dit;
    if (!dit.load(rt, dir + "/dit-bf16.gguf")) { err = "dit load"; return -1; }

    const int dur = std::max(1, std::min(30, o.dur));
    const uint32_t T = (uint32_t) latent_T(dur);
    const uint32_t clip_S = (uint32_t) CLIP_FPS * dur;
    const uint32_t sync_S = (uint32_t) SYNC_FPS * dur;

    // empty conditioning (replicated empty_clip_feat and empty_sync_feat)
    std::vector<float> ec_h = dit.W->host("empty_clip_feat");   // [1024]
    std::vector<float> es_h = dit.W->host("empty_sync_feat");   // [768]
    gpu::Buf clip_f = rt.dev.alloc((size_t) clip_S * 1024 * 4);
    gpu::Buf sync_f = rt.dev.alloc((size_t) sync_S * 768 * 4);
    for (uint32_t s = 0; s < clip_S; ++s) memcpy((char *) clip_f.p + s * 1024 * 4, ec_h.data(), 1024 * 4);
    for (uint32_t s = 0; s < sync_S; ++s) memcpy((char *) sync_f.p + s * 768 * 4, es_h.data(), 768 * 4);

    if (!dit.prep_conditioning(rt, clip_f, sync_f, T, clip_S, sync_S)) { err = "prep_cond"; return -1; }

    // text cat cond: [pad(clip_toks, 1024, 0) | t5_toks] -> [154, 2048]
    gpu::Buf text_cat_cond = rt.dev.alloc((size_t) 154 * 2048 * 4);
    memset(text_cat_cond.p, 0, (size_t) 154 * 2048 * 4);
    for (uint32_t s = 0; s < CLIP_TOKEN_CTX; ++s)
        memcpy((char *) text_cat_cond.p + (size_t) s * 2048 * 4,
               (char *) clip_toks.p + (size_t) s * 1024 * 4, 1024 * 4);
    memcpy((char *) text_cat_cond.p + (size_t) CLIP_TOKEN_CTX * 2048 * 4,
           t5_toks.p, (size_t) T5_TOKEN_CTX * 2048 * 4);

    // text cat uncond: [empty_string_feat (1024->pad 2048) | empty_t5_feat (2048)]
    std::vector<float> es_str = dit.W->host("empty_string_feat");   // [77, 1024]
    std::vector<float> et_t5  = dit.W->host("empty_t5_feat");       // [77, 2048]
    gpu::Buf text_cat_unc = rt.dev.alloc((size_t) 154 * 2048 * 4);
    memset(text_cat_unc.p, 0, (size_t) 154 * 2048 * 4);
    for (uint32_t s = 0; s < CLIP_TOKEN_CTX; ++s)
        memcpy((char *) text_cat_unc.p + (size_t) s * 2048 * 4,
               (char *) es_str.data() + (size_t) s * 1024 * 4, 1024 * 4);
    memcpy((char *) text_cat_unc.p + (size_t) CLIP_TOKEN_CTX * 2048 * 4,
           et_t5.data(), (size_t) T5_TOKEN_CTX * 2048 * 4);

    gpu::Buf gtext_unc = rt.dev.alloc(1024 * 4);
    memset(gtext_unc.p, 0, 1024 * 4);

    gpu::Buf text_pp_cond = rt.dev.alloc((size_t) 154 * dit.H * 4);
    gpu::Buf gcmlp_cond = rt.dev.alloc(dit.H * 4);
    gpu::Buf text_pp_unc = rt.dev.alloc((size_t) 154 * dit.H * 4);
    gpu::Buf gcmlp_unc = rt.dev.alloc(dit.H * 4);

    dit.prep_text_arm(rt, text_cat_cond, clip_glob, text_pp_cond, gcmlp_cond);
    dit.prep_text_arm(rt, text_cat_unc, gtext_unc, text_pp_unc, gcmlp_unc);

    // initial noise: mt19937 seeded randn, drawn linearly in the reference's
    // channel-major order (draw i -> c = i / T, t = i % T), stored time-major [T,64]
    std::mt19937 rng(o.seed);
    std::normal_distribution<float> nd(0, 1);
    gpu::Buf x = rt.dev.alloc((size_t) T * LATENT_DIM * 4);
    for (uint32_t c = 0; c < LATENT_DIM; ++c)
        for (uint32_t t = 0; t < T; ++t) ((float *) x.p)[(size_t) t * LATENT_DIM + c] = nd(rng);
    if (const char * p0 = getenv("TN_DUMP_X0")) {
        FILE * g = fopen(p0, "wb");
        if (g) { fwrite(x.p, 4, (size_t) T * LATENT_DIM, g); fclose(g); }
    }

    gpu::Buf fc = rt.dev.alloc((size_t) T * LATENT_DIM * 4);
    gpu::Buf fu = rt.dev.alloc((size_t) T * LATENT_DIM * 4);

    // flow-matching Euler steps
    for (int s = 0; s < o.steps; ++s) {
        float t_curr = 1.0f - (float) s / o.steps;
        float t_prev = 1.0f - (float) (s + 1) / o.steps;
        float dt = t_prev - t_curr;
        dit.forward(rt, x, t_curr, text_pp_cond, gcmlp_cond, fc);
        dit.forward(rt, x, t_curr, text_pp_unc, gcmlp_unc, fu);
        rt.cfg_euler(fc.p, fu.p, x.p, (size_t) T * LATENT_DIM, o.cfg, dt);
        if (getenv("TN_TRACE")) {
            rt.dev.commit();
            double ss = 0, mx = 0; float * xp = (float *) x.p;
            for (size_t i = 0; i < (size_t) T * LATENT_DIM; ++i) { ss += (double) xp[i] * xp[i]; mx = std::max(mx, (double) std::abs(xp[i])); }
            fprintf(stderr, "step %2d  t=%.4f  |x|=%.3f  max|x|=%.3f\n", s, t_curr, std::sqrt(ss), mx);
        }
    }
    rt.dev.commit();

    // VAE decode
    VaeModel vae;
    if (!vae.load(rt, dir + "/vae-f32.gguf")) { err = "vae load"; return -1; }
    uint32_t T_audio = 0;
    std::vector<float> audio = vae.decode(rt, (const float *) x.p, T, T_audio);
    if (audio.empty() || T_audio == 0) { err = "vae decode"; return -1; }

    if (!write_wav_file(out_path, audio.data(), T_audio, SAMPLE_RATE)) {
        err = "write wav failed";
        return -1;
    }
    return (int) T_audio;
}

} // namespace tn
