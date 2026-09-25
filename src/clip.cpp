// clip.cpp — MetaCLIP text encoder on GPU (f32 weights).
#include "models.h"

namespace tn {

bool ClipModel::load(Runtime & rt, const std::string & path) {
    W = std::make_unique<WeightSet>();
    if (!W->load(rt, path, true)) return false;
    H = (uint32_t) W->ki64("clip.hidden", 1024);
    NL = (uint32_t) W->ki64("clip.layers", 24);
    NH = (uint32_t) W->ki64("clip.heads", 16);
    hd = H / NH;
    ctx = (uint32_t) W->ki64("clip.ctx", 77);
    // scratch
    xb = rt.dev.alloc((size_t) ctx * H * 4);            // hidden
    rb = rt.dev.alloc((size_t) ctx * H * 4);            // normed
    qkv = rt.dev.alloc((size_t) ctx * 3 * H * 4);       // [ctx, 3H]
    att = rt.dev.alloc((size_t) ctx * H * 4);           // attention out rows
    probs = rt.dev.alloc((size_t) NH * ctx * ctx * 4);  // attn scratch
    xb2 = rt.dev.alloc((size_t) ctx * 4 * H * 4);       // mlp hidden
    pos = rt.dev.alloc(ctx * 4);
    std::vector<int32_t> pv(ctx);
    for (uint32_t i = 0; i < ctx; ++i) pv[i] = (int32_t) i;
    memcpy(pos.p, pv.data(), ctx * 4);
    // embeddings host-side gather source
    tok_host = W->host("tok_emb");
    pos_host = W->host("pos_emb");
    return xb.ok() && qkv.ok() && probs.ok() && tok_host.size() == (size_t) 49408 * H;
}

void ClipModel::encode(Runtime & rt, const std::vector<int32_t> & ids,
                       gpu::Buf & per_token, gpu::Buf & global) {
    const uint32_t n = ctx;   // always 77 (eos padded)
    // x = tok_emb[ids] + pos_emb
    {
        float * dst = (float *) xb.p;
        for (uint32_t s = 0; s < n; ++s) {
            const float * te = tok_host.data() + (size_t) ids[s] * H;
            const float * pe = pos_host.data() + (size_t) s * H;
            for (uint32_t c = 0; c < H; ++c) dst[(size_t) s * H + c] = te[c] + pe[c];
        }
    }
    // eos position for mask + pooling
    uint32_t eos_pos = 0; int32_t mx = ids[0];
    for (uint32_t i = 1; i < n; ++i) if (ids[i] > mx) { mx = ids[i]; eos_pos = i; }

    // TN_DUMP_CLIP=<dir>: layer-0 stage buffers (full rows) for offline parity
    const char * dump_dir = getenv("TN_DUMP_CLIP");
    auto dump = [&](uint32_t l, const char * nm, const void * p, size_t cnt) {
        if (!dump_dir || l != 0) return;
        rt.dev.commit();
        FILE * g = fopen((std::string(dump_dir) + "/clipl0_" + nm + ".bin").c_str(), "wb");
        if (g) { fwrite(p, 4, cnt, g); fclose(g); }
    };
    for (uint32_t l = 0; l < NL; ++l) {
        dump(l, "xb", xb.p, (size_t) n * H);
        char b[64];
        snprintf(b, sizeof b, "blk.%u", l);
        rt.ln_wb(xb.p, rb.p, W->need(std::string(b) + ".ln1.w").buf.p,
                 W->need(std::string(b) + ".ln1.b").buf.p, n, H, 1e-5f);
        // q,k,v into qkv [n, 3H] slices
        const void * qw = W->need(std::string(b) + ".q.w").buf.p;
        rt.gemm(false, qw, rb.p, qkv.p, W->need(std::string(b) + ".q.b").buf.p,
                {n, H, H, H, H, 3 * H});
        rt.gemm(false, W->need(std::string(b) + ".k.w").buf.p, rb.p,
                (char *) qkv.p + (size_t) H * 4, W->need(std::string(b) + ".k.b").buf.p,
                {n, H, H, H, H, 3 * H});
        rt.gemm(false, W->need(std::string(b) + ".v.w").buf.p, rb.p,
                (char *) qkv.p + (size_t) 2 * H * 4, W->need(std::string(b) + ".v.b").buf.p,
                {n, H, H, H, H, 3 * H});
        dump(l, "rb", rb.p, (size_t) n * H);
        dump(l, "qkv", qkv.p, (size_t) n * 3 * H);
        attn_call(rt, n, 2 /*clip mask*/, eos_pos, nullptr, nullptr);
        dump(l, "probs", probs.p, (size_t) NH * n * n);
        dump(l, "att", att.p, (size_t) n * H);
        // out proj + residual
        rt.gemm(false, W->need(std::string(b) + ".out.w").buf.p, att.p, xb2.p,
                W->need(std::string(b) + ".out.b").buf.p, {n, H, H});
        rt.add_(xb.p, xb2.p, xb.p, (size_t) n * H);
        dump(l, "xattn", xb.p, (size_t) n * H);
        // mlp
        rt.ln_wb(xb.p, rb.p, W->need(std::string(b) + ".ln2.w").buf.p,
                 W->need(std::string(b) + ".ln2.b").buf.p, n, H, 1e-5f);
        rt.gemm(false, W->need(std::string(b) + ".fc1.w").buf.p, rb.p, xb2.p,
                W->need(std::string(b) + ".fc1.b").buf.p, {n, 4 * H, H});
        rt.gelu_quick(xb2.p, (size_t) n * 4 * H);
        dump(l, "act", xb2.p, (size_t) n * 4 * H);
        rt.gemm(false, W->need(std::string(b) + ".fc2.w").buf.p, xb2.p, rb.p,
                W->need(std::string(b) + ".fc2.b").buf.p, {n, H, 4 * H});
        rt.add_(xb.p, rb.p, xb.p, (size_t) n * H);
        dump(l, "xout", xb.p, (size_t) n * H);
    }
    rt.ln_wb(xb.p, per_token.p, W->need("out_ln.w").buf.p, W->need("out_ln.b").buf.p, n, H, 1e-5f);
    // pooled: per_token row eos_pos → text_proj gemv
    rt.gemv(W->need("text_proj").buf.p, (char *) per_token.p + (size_t) eos_pos * H * 4,
            global.p, nullptr, H, H, H, false);
}

void ClipModel::attn_call(Runtime & rt, uint32_t n, uint32_t bias_mode, uint32_t eot,
                          const int * bkt, const float * rbias) {
    gpu::AttnP p{};
    p.Sq = p.Sk = n;
    p.hd = hd;
    p.nh = NH;
    p.ld = 3 * H;             // q/k/v row stride in qkv
    p.ldo = H;                // output row stride in att
    p.q_off = 0;
    p.k_off = H;
    p.v_off = 2 * H;
    p.o_off = 0;
    p.bias_mode = bias_mode;
    p.eot = eot;
    p.scale = 1.0f / sqrtf((float) hd);
    p.p_off = 0;
    rt.dev.dispatch("attn_qk", (n + 15) / 16, NH, 1, 256, 1, 1,
                    {qkv.p, probs.p, bkt ? (const void *) bkt : probs.p,
                     rbias ? (const void *) rbias : probs.p}, &p, sizeof(p));
    rt.dev.dispatch("softmax_rows", n, NH, 1, 1, 1, 1, {probs.p}, &p, sizeof(p));
    rt.dev.dispatch("attn_pav", (n + 15) / 16, NH, 1, 256, 1, 1,
                    {probs.p, qkv.p, att.p}, &p, sizeof(p));
}

} // namespace tn
