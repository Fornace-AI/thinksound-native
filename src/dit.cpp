// dit.cpp — ThinkSound MMDiT setup + hoisted conditioning preprocess.
#include "models.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tn {

static void conv(Runtime & rt, const void * w, const void * b, const void * x, void * y,
                 uint32_t T, uint32_t IC, uint32_t OC, uint32_t K, uint32_t pad, uint32_t dil) {
    gpu::ConvP p{T, T, IC, OC, K, 1, pad, dil, IC, OC, 0, b ? 1u : 0u};
    rt.dev.dispatch("conv1d_fwd", (T + 31) / 32, (OC + 7) / 8, 1, 256, 1, 1,
                    {x, w, b ? b : x, y}, &p, sizeof(p));
}
static void convmlp(Runtime & rt, WeightSet & W, const std::string & pre,
                    const void * x, void * h1, void * h3, void * y, uint32_t T, uint32_t C, uint32_t H_ffn) {
    conv(rt, W.need(pre + ".w1.weight").buf.p, nullptr, x, h1, T, C, H_ffn, 3, 1, 1);
    conv(rt, W.need(pre + ".w3.weight").buf.p, nullptr, x, h3, T, C, H_ffn, 3, 1, 1);
    rt.swiglu(h1, h3, h1, (size_t) T * H_ffn);
    conv(rt, W.need(pre + ".w2.weight").buf.p, nullptr, h1, y, T, H_ffn, C, 3, 1, 1);
}
static void mlp(Runtime & rt, WeightSet & W, const std::string & pre,
                const void * x, void * h1, void * h3, void * y, uint32_t rows, uint32_t C, uint32_t H_ffn) {
    rt.gemm(true, W.need(pre + ".w1.weight").buf.p, x, h1, nullptr, {rows, H_ffn, C});
    rt.gemm(true, W.need(pre + ".w3.weight").buf.p, x, h3, nullptr, {rows, H_ffn, C});
    rt.swiglu(h1, h3, h1, (size_t) rows * H_ffn);
    rt.gemm(true, W.need(pre + ".w2.weight").buf.p, h1, y, nullptr, {rows, C, H_ffn});
}

bool DitModel::load(Runtime & rt, const std::string & path) {
    W = std::make_unique<WeightSet>();
    if (!W->load(rt, path, false)) return false;
    H = (uint32_t) W->ki64("dit.hidden_dim", 1024);
    NH = (uint32_t) W->ki64("dit.num_heads", 16);
    hd = H / NH;
    depth = (uint32_t) W->ki64("dit.depth", 21);
    fused_depth = (uint32_t) W->ki64("dit.fused_depth", 14);
    latent_dim = (uint32_t) W->ki64("dit.latent_dim", 64);
    rope_theta = (float) W->kf64("dit.rope_theta", 10000.0);
    t_max_period = (float) W->kf64("dit.t_embed_max_period", 1.0);
    n_joint = depth - fused_depth;

    const GgufTensor * w1 = W->file.find("joint_blocks.0.latent_block.ffn.w1.weight");
    H_ffn = (w1 && w1->n_dims >= 3) ? (uint32_t) w1->ne[2] : 2816;

    loaded = true;
    return true;
}

static void rms_dbg(const char * n, const void * p, size_t cnt) {
    const float * f = (const float *) p;
    double ss = 0;
    for (size_t i = 0; i < cnt; ++i) ss += (double) f[i] * f[i];
    fprintf(stderr, "[prep] %-14s rms=%.6g\n", n, sqrt(ss / (double) cnt));
}
bool DitModel::prep_conditioning(Runtime & rt, gpu::Buf & clip_f, gpu::Buf & sync_f,
                                  uint32_t T_, uint32_t clip_S_, uint32_t sync_S_) {
    T = T_; clip_S = clip_S_; sync_S = sync_S_;
    auto rms = [](const char * n, const void * p, size_t cnt) { rms_dbg(n, p, cnt); };
    const size_t S = (size_t) T + clip_S + text_S;
    const size_t SM = S * H * 4;
    const size_t SM_ffn = S * H_ffn * 4;

    clip_pp = rt.dev.alloc((size_t) clip_S * H * 4);
    sync_gather = rt.dev.alloc((size_t) T * H * 4);
    extc = rt.dev.alloc((size_t) T * H * 4);
    extc_silu = rt.dev.alloc((size_t) T * H * 4);
    x_in = rt.dev.alloc((size_t) T * H * 4);
    jointq = rt.dev.alloc(S * 3 * H * 4);
    sc_qkv = rt.dev.alloc(S * 3 * H * 4);   // de-interleaved (block q|k|v) copy
    joint_att = rt.dev.alloc(SM);
    mod_lat = rt.dev.alloc((size_t) T * 6 * H * 4);
    mod_ct = rt.dev.alloc(6 * H * 4);
    mod_fin = rt.dev.alloc((size_t) T * 2 * H * 4);
    te_out = rt.dev.alloc(H * 4);
    gc_pre = rt.dev.alloc(H * 4);
    global_c = rt.dev.alloc(H * 4);
    probs = rt.dev.alloc((size_t) NH * S * S * 4);
    sc_l1 = rt.dev.alloc(SM);
    sc_r = rt.dev.alloc(SM);
    sc_f1 = rt.dev.alloc(SM_ffn);
    sc_f3 = rt.dev.alloc(SM_ffn);

    size_t max_prep = (size_t) std::max({sync_S, clip_S, text_S, (uint32_t) 1});
    h1 = rt.dev.alloc(max_prep * H_ffn * 4);
    h3 = rt.dev.alloc(max_prep * H_ffn * 4);
    if (!probs.ok() || !jointq.ok() || !sc_qkv.ok() || !sc_f1.ok()) return false;

    // sync: + sync_pos_emb per 8-frame segment, conv, silu, convmlp, gather to T
    {
        std::vector<float> spe = W->host("sync_pos_emb");   // [8,768]
        std::vector<float> sbuf((size_t) sync_S * 768);
        memcpy(sbuf.data(), sync_f.p, sbuf.size() * 4);
        for (uint32_t s = 0; s < sync_S; ++s) {
            const float * sp = spe.data() + (size_t)(s % 8) * 768;
            for (int c = 0; c < 768; ++c) sbuf[(size_t) s * 768 + c] += sp[c];
        }
        gpu::Buf staged = rt.dev.alloc(sbuf.size() * 4);
        memcpy(staged.p, sbuf.data(), sbuf.size() * 4);
        gpu::Buf smid = rt.dev.alloc((size_t) sync_S * H * 4);
        conv(rt, W->need("sync_input_proj.0.weight").buf.p, W->need("sync_input_proj.0.bias").buf.p,
             staged.p, smid.p, sync_S, 768, H, 7, 3, 1);
        rt.silu(smid.p, (size_t) sync_S * H);
        convmlp(rt, *W, "sync_input_proj.2", smid.p, h1.p, h3.p, smid.p, sync_S, H, H_ffn);
        rt.dev.commit();
        if (getenv("TN_DEBUG")) rms("smid", smid.p, (size_t) sync_S * H);
        std::vector<float> srows((size_t) sync_S * H);
        memcpy(srows.data(), smid.p, srows.size() * 4);
        for (uint32_t i = 0; i < T; ++i) {
            int32_t si = (int32_t) floorf(((float) i + 0.5f) * (float) sync_S / (float) T);
            memcpy((char *) sync_gather.p + (size_t) i * H * 4,
                   (char *) srows.data() + (size_t) si * H * 4, H * 4);
        }
    }

    // clip: lin + silu + convmlp -> clip_pp [clip_S,H]
    rt.gemm(true, W->need("clip_input_proj.0.weight").buf.p, clip_f.p, clip_pp.p,
            W->need("clip_input_proj.0.bias").buf.p, {clip_S, H, 1024});
    rt.silu(clip_pp.p, (size_t) clip_S * H);
    convmlp(rt, *W, "clip_input_proj.2", clip_pp.p, h1.p, h3.p, clip_pp.p, clip_S, H, H_ffn);
    rt.dev.commit();
    if (getenv("TN_DEBUG")) rms("clip_pp", clip_pp.p, (size_t) clip_S * H);

    // clip_fc = lin(clip_cond_proj, mean_S(clip_pp))
    gpu::Buf cmean = rt.dev.alloc(H * 4);
    rt.mean_cols(clip_pp.p, cmean.p, clip_S, H);
    gpu::Buf clip_fc = rt.dev.alloc(H * 4);
    rt.gemv(W->need("clip_cond_proj.weight").buf.p, cmean.p, clip_fc.p,
            W->need("clip_cond_proj.bias").buf.p, H, 1024, 1024);
    rt.dev.commit();
    memcpy(clip_fc_host.data(), clip_fc.p, H * 4);
    return true;
}

void DitModel::prep_text_arm(Runtime & rt, gpu::Buf & text_cat, gpu::Buf & gtext,
                             gpu::Buf & text_pp_out, gpu::Buf & gcmlp_out) {
    rt.gemm(true, W->need("text_input_proj.0.weight").buf.p, text_cat.p, text_pp_out.p,
            W->need("text_input_proj.0.bias").buf.p, {text_S, H, 2048});
    if (getenv("TN_DEBUG")) {
        rt.dev.commit();
        rms_dbg("text_gemm_raw", text_pp_out.p, (size_t) text_S * H);
        // CPU recompute of row 0 straight from the same unified-memory buffers
        const uint16_t * wh16 = (const uint16_t *) W->need("text_input_proj.0.weight").buf.p;
        const float * bh = (const float *) W->need("text_input_proj.0.bias").buf.p;
        const float * xin = (const float *) text_cat.p;
        const float * yout = (const float *) text_pp_out.p;
        double ss = 0;
        for (uint32_t n = 0; n < H; ++n) {
            float acc = bh[n];
            for (uint32_t k = 0; k < 2048; ++k) {
                uint16_t h = wh16[(size_t) n * 2048 + k];
                uint32_t sign = (uint32_t)(h & 0x8000) << 16;
                uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
                float wv;
                if (exp == 0 && man == 0) wv = 0.0f;
                else if (exp == 0x1f) wv = std::numeric_limits<float>::quiet_NaN();
                else { uint32_t u = sign | ((exp + 112u) << 23) | (man << 13); memcpy(&wv, &u, 4); }
                acc += wv * xin[k];
            }
            ss += (double) (acc - yout[n]) * (acc - yout[n]);
        }
        fprintf(stderr, "[prep] row0 cpu-vs-gpu mse=%.6g\n", ss / H);
    }
    rt.silu(text_pp_out.p, (size_t) text_S * H);
    if (getenv("TN_DEBUG")) { rt.dev.commit(); rms_dbg("text_lin", text_pp_out.p, (size_t) text_S * H); }
    rt.gemm(true, W->need("text_input_proj.2.w1.weight").buf.p, text_pp_out.p, sc_f1.p, nullptr, {text_S, H_ffn, H});
    rt.gemm(true, W->need("text_input_proj.2.w3.weight").buf.p, text_pp_out.p, sc_f3.p, nullptr, {text_S, H_ffn, H});
    if (getenv("TN_DEBUG")) { rt.dev.commit(); rms_dbg("text_w1", sc_f1.p, (size_t) text_S * H_ffn); rms_dbg("text_w3", sc_f3.p, (size_t) text_S * H_ffn); }
    rt.swiglu(sc_f1.p, sc_f3.p, sc_f1.p, (size_t) text_S * H_ffn);
    rt.gemm(true, W->need("text_input_proj.2.w2.weight").buf.p, sc_f1.p, text_pp_out.p, nullptr, {text_S, H, H_ffn});
    rt.dev.commit();
    if (getenv("TN_DEBUG")) {
        const float * f = (const float *) text_pp_out.p;
        double ss = 0;
        for (size_t i = 0; i < (size_t) text_S * H; ++i) ss += (double) f[i] * f[i];
        fprintf(stderr, "[prep] %-14s rms=%.6g\n", "text_pp", sqrt(ss / (double) (text_S * H)));
    }
    // text_fc = lin(text_cond_proj, gtext); gcmlp = mlp(clip_fc + text_fc)
    gpu::Buf text_fc = rt.dev.alloc(H * 4);
    rt.gemv(W->need("text_cond_proj.weight").buf.p, gtext.p, text_fc.p,
            W->need("text_cond_proj.bias").buf.p, H, 1024, 1024);
    rt.dev.commit();
    std::vector<float> tfc(H);
    memcpy(tfc.data(), text_fc.p, H * 4);
    for (uint32_t i = 0; i < H; ++i) tfc[i] += clip_fc_host[i];
    gpu::Buf sum = rt.dev.alloc(H * 4);
    memcpy(sum.p, tfc.data(), H * 4);
    mlp(rt, *W, "global_cond_mlp", sum.p, sc_f1.p, sc_f3.p, gcmlp_out.p, 1, H, H_ffn);
    rt.dev.commit();
}

} // namespace tn
