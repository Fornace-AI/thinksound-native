// dit_fwd.cpp — per-step MMDiT forward (joint + fused blocks, final layer).
// Streams update in place; scratch buffers are dedicated (no aliasing).
#include "models.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace tn {

namespace {
inline void conv(Runtime & rt, const void * w, const void * b, const void * x, void * y,
                 uint32_t T, uint32_t IC, uint32_t OC, uint32_t K, uint32_t pad, uint32_t dil) {
    gpu::ConvP p{T, T, IC, OC, K, 1, pad, dil, IC, OC, 0, b ? 1u : 0u};
    rt.dev.dispatch("conv1d_fwd", (T + 31) / 32, (OC + 7) / 8, 1, 256, 1, 1,
                    {x, w, b ? b : x, y}, &p, sizeof(p));
}
struct AD { uint32_t H, per_row, chunks, base; float eps; };
struct GA { uint32_t H, per_row, chunks, gate_idx; };
struct RH { uint32_t ld, hd, col_off, row_off; float eps; };
struct RP { uint32_t ld, hd, col_off, row_off; float fscale, base; };
} // namespace

struct QS3 { uint32_t H, row_off; };

static void pre_attn(Runtime & rt, DitModel & m, const std::string & bp,
                     const void * x, const void * mod, bool per_row, uint32_t rows,
                     uint32_t row_off, void * qkv_dst,
                     const int * pos, float fscale, bool is_text, uint32_t chunks = 6) {
    const uint32_t H = m.H;
    AD ad{H, per_row ? 1u : 0u, chunks, 0u, 1e-5f};
    rt.dev.dispatch("adaln_ln", rows, 1, 1, 1, 1, 1, {x, mod, m.sc_f1.p}, &ad, sizeof(ad));
    rt.gemm(true, m.W->need(bp + ".attn.qkv.weight").buf.p, m.sc_f1.p, qkv_dst,
            m.W->need(bp + ".attn.qkv.bias").buf.p, {rows, 3 * H, H, H, H, 3 * H});
    // de-interleave q|k|v into block layout in sc_qkv; downstream reads sc_qkv
    QS3 qs{H, row_off};
    rt.dev.dispatch("qkv_split3", (H + 63) / 64 * 64, rows, 1, 64, 1, 1,
                    {qkv_dst, m.sc_qkv.p}, &qs, sizeof(qs));
    RH rq{3 * H, m.hd, 0, row_off, 1e-6f};
    rt.dev.dispatch("rms_head", rows, m.NH, 1, 1, 1, 1,
                    {m.sc_qkv.p, m.W->need(bp + ".attn.q_norm.weight").buf.p}, &rq, sizeof(rq));
    RH rk{3 * H, m.hd, H, row_off, 1e-6f};
    rt.dev.dispatch("rms_head", rows, m.NH, 1, 1, 1, 1,
                    {m.sc_qkv.p, m.W->need(bp + ".attn.k_norm.weight").buf.p}, &rk, sizeof(rk));
    if (!is_text) {
        RP pq{3 * H, m.hd, 0, row_off, fscale, m.rope_theta};
        rt.dev.dispatch("rope_pairs", rows, H / 2, 1, 1, 1, 1, {m.sc_qkv.p, pos}, &pq, sizeof(pq));
        RP pk{3 * H, m.hd, H, row_off, fscale, m.rope_theta};
        rt.dev.dispatch("rope_pairs", rows, H / 2, 1, 1, 1, 1, {m.sc_qkv.p, pos}, &pk, sizeof(pk));
    }
}

static void post_attn(Runtime & rt, DitModel & m, const std::string & bp,
                      void * x, const void * attn_out, const void * mod, bool per_row,
                      uint32_t rows, bool is_text) {
    const uint32_t H = m.H;
    const uint32_t H_ffn = m.H_ffn;
    if (is_text)
        rt.gemm(true, m.W->need(bp + ".linear1.weight").buf.p, attn_out, m.sc_l1.p,
                m.W->need(bp + ".linear1.bias").buf.p, {rows, H, H});
    else
        conv(rt, m.W->need(bp + ".linear1.weight").buf.p, m.W->need(bp + ".linear1.bias").buf.p,
             attn_out, m.sc_l1.p, rows, H, H, 3, 1, 1);
    GA ga{H, per_row ? 1u : 0u, 6u, 2u};
    rt.dev.dispatch("gate_add", (uint32_t) (((size_t) rows * H + 255) / 256), 1, 1, 256, 1, 1,
                    {x, m.sc_l1.p, mod, x}, &ga, sizeof(ga));
    AD ad{H, per_row ? 1u : 0u, 6u, 3u * H, 1e-5f};
    rt.dev.dispatch("adaln_ln", rows, 1, 1, 1, 1, 1, {x, mod, m.sc_r.p}, &ad, sizeof(ad));
    if (is_text) {
        rt.gemm(true, m.W->need(bp + ".ffn.w1.weight").buf.p, m.sc_r.p, m.sc_f1.p, nullptr, {rows, H_ffn, H});
        rt.gemm(true, m.W->need(bp + ".ffn.w3.weight").buf.p, m.sc_r.p, m.sc_f3.p, nullptr, {rows, H_ffn, H});
        rt.swiglu(m.sc_f1.p, m.sc_f3.p, m.sc_f1.p, (size_t) rows * H_ffn);
        rt.gemm(true, m.W->need(bp + ".ffn.w2.weight").buf.p, m.sc_f1.p, m.sc_f3.p, nullptr, {rows, H, H_ffn});
    } else {
        conv(rt, m.W->need(bp + ".ffn.w1.weight").buf.p, nullptr, m.sc_r.p, m.sc_f1.p, rows, H, H_ffn, 3, 1, 1);
        conv(rt, m.W->need(bp + ".ffn.w3.weight").buf.p, nullptr, m.sc_r.p, m.sc_f3.p, rows, H, H_ffn, 3, 1, 1);
        rt.swiglu(m.sc_f1.p, m.sc_f3.p, m.sc_f1.p, (size_t) rows * H_ffn);
        conv(rt, m.W->need(bp + ".ffn.w2.weight").buf.p, nullptr, m.sc_f1.p, m.sc_f3.p, rows, H_ffn, H, 3, 1, 1);
    }
    GA gf{H, per_row ? 1u : 0u, 6u, 5u};
    rt.dev.dispatch("gate_add", (uint32_t) (((size_t) rows * H + 255) / 256), 1, 1, 256, 1, 1,
                    {x, m.sc_f3.p, mod, x}, &gf, sizeof(gf));
}

static void attention(Runtime & rt, DitModel & m, uint32_t S) {
    gpu::AttnP p{};
    p.Sq = p.Sk = S;
    p.hd = m.hd;
    p.nh = m.NH;
    p.ld = 3 * m.H;
    p.ldo = m.H;
    p.bias_mode = 0;
    p.scale = 1.0f / sqrtf((float) m.hd);
    p.q_off = 0;
    p.k_off = m.H;
    p.v_off = 2 * m.H;
    p.o_off = 0;
    p.p_off = 0;
    rt.dev.dispatch("attn_qk", (S + 15) / 16, m.NH, 1, 256, 1, 1,
                    {m.sc_qkv.p, m.probs.p, m.probs.p, m.probs.p}, &p, sizeof(p));
    rt.dev.dispatch("softmax_rows", S, m.NH, 1, 1, 1, 1, {m.probs.p}, &p, sizeof(p));
    rt.dev.dispatch("attn_pav", (S + 15) / 16, m.NH, 1, 256, 1, 1,
                    {m.probs.p, m.sc_qkv.p, m.joint_att.p}, &p, sizeof(p));
}

void DitModel::forward(Runtime & rt, gpu::Buf & x, float t,
                       gpu::Buf & text_pp, gpu::Buf & gcmlp, gpu::Buf & flow) {
    const uint32_t T_ = T;
    const uint32_t C = clip_S;
    const uint32_t TS = text_S;
    const uint32_t H_ = H;

    // ---- timestep embedding ----
    std::vector<float> temb(H_);
    int half = H_ / 2;
    float fscale_t = 10000.0f / t_max_period;
    for (int kk = 0; kk < half; ++kk) {
        float fr = fscale_t * powf(10000.0f, -2.0f * kk / (float) H_);
        temb[kk]        = cosf(t * fr);
        temb[half + kk] = sinf(t * fr);
    }
    gpu::Buf in_temb = rt.dev.alloc(H_ * 4);
    memcpy(in_temb.p, temb.data(), H_ * 4);
    rt.gemv(W->need("t_embed.mlp.0.weight").buf.p, in_temb.p, te_out.p,
            W->need("t_embed.mlp.0.bias").buf.p, H_, H_, H_);
    rt.silu(te_out.p, H_);
    // second layer via scratch: gemv cannot run in place
    gpu::Buf te_scr = rt.dev.alloc(H_ * 4);
    rt.gemv(W->need("t_embed.mlp.2.weight").buf.p, te_out.p, te_scr.p,
            W->need("t_embed.mlp.2.bias").buf.p, H_, H_, H_);

    // global_c = gcmlp + te   (te_scr holds t_emb; keep te_out mirror for dumps)
    rt.add_(gcmlp.p, te_scr.p, global_c.p, H_);
    rt.copy(te_scr.p, te_out.p, H_);

    // extended_c = bcast(global_c) + sync_gather  [T_,H]
    rt.add_bcast_col(sync_gather.p, global_c.p, extc.p, T_, H_);
    rt.copy(extc.p, extc_silu.p, (size_t) T_ * H_);
    rt.silu(extc_silu.p, (size_t) T_ * H_);
    // keep the pre-silu global_c for dumps; clip/text adaLN take silu(global_c)
    rt.copy(global_c.p, gc_pre.p, H_);
    rt.silu(global_c.p, H_);

    // audio_input_proj: conv + silu + convmlp -> x_in [T_,H]
    conv(rt, W->need("audio_input_proj.0.weight").buf.p, W->need("audio_input_proj.0.bias").buf.p,
         x.p, x_in.p, T_, latent_dim, H_, 3, 1, 1);
    rt.silu(x_in.p, (size_t) T_ * H_);
    conv(rt, W->need("audio_input_proj.2.w1.weight").buf.p, nullptr, x_in.p, sc_f1.p, T_, H_, H_ffn, 3, 1, 1);
    conv(rt, W->need("audio_input_proj.2.w3.weight").buf.p, nullptr, x_in.p, sc_f3.p, T_, H_, H_ffn, 3, 1, 1);
    rt.swiglu(sc_f1.p, sc_f3.p, sc_f1.p, (size_t) T_ * H_ffn);
    conv(rt, W->need("audio_input_proj.2.w2.weight").buf.p, nullptr, sc_f1.p, x_in.p, T_, H_ffn, H_, 3, 1, 1);

    if (const char * dd = getenv("TN_DUMP_EARLY")) {
        auto wr = [&](const char * n, const void * p, size_t cnt) {
            std::string path = std::string(dd) + "/nat_" + n + ".bin";
            FILE * f = fopen(path.c_str(), "wb");
            fwrite(p, 4, cnt, f); fclose(f);
        };
        rt.dev.commit();
        wr("t_emb", te_out.p, H_);
        wr("global_c", gc_pre.p, H_);
        wr("extended_c", extc.p, (size_t) T_ * H_);
        wr("x_in", x_in.p, (size_t) T_ * H_);
    }

    // ---- joint blocks ----
    // clip/text streams evolve per forward; start each forward from the prep outputs
    rt.copy(clip_pp.p, clip_s.p, (size_t) C * H_);
    rt.copy(text_pp.p, text_s.p, (size_t) TS * H_);
    void * lat = x_in.p;
    const uint32_t S = T_ + C + TS;
    const float clip_fscale = (float) T_ / (float) C;
    if (!pos_lat.ok()) {
        std::vector<int32_t> pl(T_), pc(C);
        for (uint32_t i = 0; i < T_; ++i) pl[i] = (int32_t) i;
        for (uint32_t i = 0; i < C; ++i) pc[i] = (int32_t) i;
        pos_lat = rt.dev.alloc(T_ * 4);
        memcpy(pos_lat.p, pl.data(), T_ * 4);
        pos_clip = rt.dev.alloc(C * 4);
        memcpy(pos_clip.p, pc.data(), C * 4);
    }
    for (uint32_t i = 0; i < n_joint; ++i) {
        const bool pre_only = (i == n_joint - 1);
        char jb[80];
        snprintf(jb, sizeof jb, "joint_blocks.%u", i);

        // Latent modulation: 6*H
        rt.gemm(true, W->need(std::string(jb) + ".latent_block.adaLN_modulation.1.weight").buf.p,
                extc_silu.p, mod_lat.p,
                W->need(std::string(jb) + ".latent_block.adaLN_modulation.1.bias").buf.p,
                {T_, 6 * H_, H_});
        pre_attn(rt, *this, std::string(jb) + ".latent_block", lat, mod_lat.p, true, T_, 0,
                 jointq.p, (const int *) pos_lat.p, 1.0f, false, 6);

        // Clip modulation: 2*H in block 6, 6*H in blocks 0..5
        uint32_t ct_chunks = pre_only ? 2u : 6u;
        rt.gemv(W->need(std::string(jb) + ".clip_block.adaLN_modulation.1.weight").buf.p,
                global_c.p, mod_clip.p,
                W->need(std::string(jb) + ".clip_block.adaLN_modulation.1.bias").buf.p,
                ct_chunks * H_, H_, H_);
        pre_attn(rt, *this, std::string(jb) + ".clip_block", clip_s.p, mod_clip.p, false, C, T_,
                 (char *) jointq.p + (size_t) T_ * 3 * H_ * 4, (const int *) pos_clip.p,
                 clip_fscale, false, ct_chunks);

        // Text modulation: 2*H in block 6, 6*H in blocks 0..5
        rt.gemv(W->need(std::string(jb) + ".text_block.adaLN_modulation.1.weight").buf.p,
                global_c.p, mod_text.p,
                W->need(std::string(jb) + ".text_block.adaLN_modulation.1.bias").buf.p,
                ct_chunks * H_, H_, H_);
        pre_attn(rt, *this, std::string(jb) + ".text_block", text_s.p, mod_text.p, false, TS, T_ + C,
                 (char *) jointq.p + (size_t)(T_ + C) * 3 * H_ * 4, nullptr, 1.0f, true, ct_chunks);

        if (i == 0 && getenv("TN_DUMP_EARLY")) {
            const char * dd = getenv("TN_DUMP_EARLY");
            rt.dev.commit();
            auto wr0 = [&](const char * n, const void * p, size_t cnt) {
                FILE * f = fopen((std::string(dd) + "/nat_" + n + ".bin").c_str(), "wb");
                fwrite(p, 4, cnt, f); fclose(f);
            };
            wr0("text_xn", sc_f1.p, (size_t) TS * H_);
            // a0_q/k/v: extract column slices of jointq [S,3H] -> [S,H] each
            {
                std::vector<float> jq((size_t) S * 3 * H_);
                memcpy(jq.data(), sc_qkv.p, jq.size() * 4);
                std::vector<float> q((size_t) S * H_), k((size_t) S * H_), v((size_t) S * H_);
                for (uint32_t s = 0; s < S; ++s) {
                    memcpy(q.data() + (size_t) s * H_, jq.data() + (size_t) s * 3 * H_, H_ * 4);
                    memcpy(k.data() + (size_t) s * H_, jq.data() + (size_t) s * 3 * H_ + H_, H_ * 4);
                    memcpy(v.data() + (size_t) s * H_, jq.data() + (size_t) s * 3 * H_ + 2 * H_, H_ * 4);
                }
                wr0("a0_q", q.data(), q.size());
                wr0("a0_k", k.data(), k.size());
                wr0("a0_v", v.data(), v.size());
            }
        }

        attention(rt, *this, S);

        post_attn(rt, *this, std::string(jb) + ".latent_block", lat, joint_att.p, mod_lat.p,
                  true, T_, false);
        if (!pre_only) {
            post_attn(rt, *this, std::string(jb) + ".clip_block", clip_s.p,
                      (char *) joint_att.p + (size_t) T_ * H_ * 4, mod_clip.p, false, C, false);
            post_attn(rt, *this, std::string(jb) + ".text_block", text_s.p,
                      (char *) joint_att.p + (size_t)(T_ + C) * H_ * 4, mod_text.p, false, TS, true);
        }
        if (i == 0 && getenv("TN_DUMP_EARLY")) {
            const char * dd = getenv("TN_DUMP_EARLY");
            rt.dev.commit();
            auto wr0 = [&](const char * n, const void * p, size_t cnt) {
                FILE * f = fopen((std::string(dd) + "/nat_" + n + ".bin").c_str(), "wb");
                fwrite(p, 4, cnt, f); fclose(f);
            };
            wr0("a0_out", joint_att.p, (size_t) S * H_);
            wr0("jb0_lat", lat, (size_t) T_ * H_);
            wr0("jb0_clip", clip_s.p, (size_t) C * H_);
            wr0("jb0_text", text_s.p, (size_t) TS * H_);
        }
    }

    // after_joint snapshot (before fused blocks modify lat in place)
    if (getenv("TN_DUMP_EARLY")) {
        rt.dev.commit();
        const char * dd = getenv("TN_DUMP_EARLY");
        FILE * f = fopen((std::string(dd) + "/nat_after_joint.bin").c_str(), "wb");
        fwrite(lat, 4, (size_t) T_ * H_, f); fclose(f);
    }

    // ---- fused blocks ----
    for (uint32_t i = 0; i < fused_depth; ++i) {
        char fb[80];
        snprintf(fb, sizeof fb, "fused_blocks.%u", i);
        rt.gemm(true, W->need(std::string(fb) + ".adaLN_modulation.1.weight").buf.p,
                extc_silu.p, mod_lat.p,
                W->need(std::string(fb) + ".adaLN_modulation.1.bias").buf.p,
                {T_, 6 * H_, H_});
        pre_attn(rt, *this, fb, lat, mod_lat.p, true, T_, 0, jointq.p, (const int *) pos_lat.p,
                 1.0f, false, 6);
        attention(rt, *this, T_);
        post_attn(rt, *this, fb, lat, joint_att.p, mod_lat.p, true, T_, false);
    }
    if (getenv("TN_DUMP_EARLY")) {
        rt.dev.commit();
        const char * dd = getenv("TN_DUMP_EARLY");
        FILE * f = fopen((std::string(dd) + "/nat_after_fused.bin").c_str(), "wb");
        fwrite(lat, 4, (size_t) T_ * H_, f); fclose(f);
    }

    // ---- final layer ----
    rt.gemm(true, W->need("final_layer.adaLN_modulation.1.weight").buf.p,
            extc_silu.p, mod_fin.p,
            W->need("final_layer.adaLN_modulation.1.bias").buf.p,
            {T_, 2 * H_, H_});
    AD af{H_, 1u, 2u, 0u, 1e-5f};
    rt.dev.dispatch("adaln_ln", T_, 1, 1, 1, 1, 1, {lat, mod_fin.p, sc_l1.p}, &af, sizeof(af));
    if (getenv("TN_DUMP_EARLY")) {
        rt.dev.commit();
        const char * dd = getenv("TN_DUMP_EARLY");
        FILE * f = fopen((std::string(dd) + "/nat_final_fx.bin").c_str(), "wb");
        fwrite(sc_l1.p, 4, (size_t) T_ * H_, f); fclose(f);
        f = fopen((std::string(dd) + "/nat_final_mod.bin").c_str(), "wb");
        fwrite(mod_fin.p, 4, (size_t) T_ * 2 * H_, f); fclose(f);
    }
    conv(rt, W->need("final_layer.conv.weight").buf.p, W->need("final_layer.conv.bias").buf.p,
         sc_l1.p, flow.p, T_, H_, latent_dim, 7, 3, 1);
}

} // namespace tn
