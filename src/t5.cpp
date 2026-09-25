// t5.cpp — T5 v1.1 encoder on GPU (f32 weights, q8 dequantized at load).
#include "models.h"
#include <cmath>

namespace tn {

static int t5_bucket(int rel, int num_buckets, int max_dist) {
    int ret = 0; num_buckets /= 2;
    if (rel > 0) ret += num_buckets;
    int n = std::abs(rel);
    int max_exact = num_buckets / 2;
    if (n < max_exact) ret += n;
    else {
        int v = max_exact + (int) (std::log((double) n / max_exact) /
                std::log((double) max_dist / max_exact) * (num_buckets - max_exact));
        ret += std::min(v, num_buckets - 1);
    }
    return ret;
}

bool T5Model::load(Runtime & rt, const std::string & path) {
    W = std::make_unique<WeightSet>();
    if (!W->load(rt, path, true)) return false;
    d_model = (uint32_t) W->ki64("t5.d_model", 2048);
    d_ff = (uint32_t) W->ki64("t5.d_ff", 5120);
    NL = (uint32_t) W->ki64("t5.n_layers", 24);
    NH = (uint32_t) W->ki64("t5.n_heads", 32);
    hd = (uint32_t) W->ki64("t5.d_kv", 64);
    n_bucket = (uint32_t) W->ki64("t5.rel_buckets", 32);
    max_dist = (uint32_t) W->ki64("t5.rel_max_dist", 128);
    eps = (float) W->kf64("t5.eps", 1e-6);
    hd = d_model / NH;
    const uint32_t n = T5_TOKEN_CTX;
    xb = rt.dev.alloc((size_t) n * d_model * 4);
    rb = rt.dev.alloc((size_t) n * d_model * 4);
    qkv = rt.dev.alloc((size_t) n * 3 * d_model * 4);
    att = rt.dev.alloc((size_t) n * d_model * 4);
    probs = rt.dev.alloc((size_t) NH * n * n * 4);
    ff1 = rt.dev.alloc((size_t) n * d_ff * 4);
    ff2 = rt.dev.alloc((size_t) n * d_ff * 4);
    bktb = rt.dev.alloc((size_t) n * n * 4 + n * 4);   // buckets [n*n] + pad flags [n]
    // bucket table [n*n] i32
    {
        std::vector<int32_t> bk((size_t) n * n);
        for (int q = 0; q < n; ++q)
            for (int k = 0; k < n; ++k)
                bk[(size_t) q * n + k] = t5_bucket(k - q, n_bucket, max_dist);
        memcpy(bktb.p, bk.data(), bk.size() * 4);
    }
    tok_host = W->host("tok_emb");
    return xb.ok() && qkv.ok() && probs.ok() && tok_host.size() == 32128ull * d_model;
}

void T5Model::attn_call(Runtime & rt, uint32_t n) {
    gpu::AttnP p{};
    p.Sq = p.Sk = n;
    p.hd = hd;
    p.nh = NH;
    p.ld = 3 * d_model;
    p.ldo = d_model;
    p.q_off = 0;
    p.k_off = d_model;
    p.v_off = 2 * d_model;
    p.o_off = 0;
    p.bias_mode = 1;
    p.pad_on = 1;
    p.scale = 1.0f;   // T5 folds 1/sqrt(d_kv) into the q weights (reference: plain soft_max)
    p.p_off = 0;
    const void * rbias = W->need("rel_bias").buf.p;
    rt.dev.dispatch("attn_qk", (n + 15) / 16, NH, 1, 256, 1, 1,
                    {qkv.p, probs.p, bktb.p, rbias}, &p, sizeof(p));
    rt.dev.dispatch("softmax_rows", n, NH, 1, 1, 1, 1, {probs.p}, &p, sizeof(p));
    rt.dev.dispatch("attn_pav", (n + 15) / 16, NH, 1, 256, 1, 1,
                    {probs.p, qkv.p, att.p}, &p, sizeof(p));
}

void T5Model::encode(Runtime & rt, const std::vector<int32_t> & ids, gpu::Buf & out) {
    const uint32_t n = T5_TOKEN_CTX;
    // pad flags: key masked -inf where id == 0 (matches reference pd table)
    {
        int32_t * pf = (int32_t *) bktb.p + (size_t) n * n;
        for (uint32_t s = 0; s < n; ++s) pf[s] = ids[s] != 0;
    }
    // embed gather (host)
    {
        float * dst = (float *) xb.p;
        for (uint32_t s = 0; s < n; ++s) {
            const float * te = tok_host.data() + (size_t) ids[s] * d_model;
            for (uint32_t c = 0; c < d_model; ++c) dst[(size_t) s * d_model + c] = te[c];
        }
    }
    // TN_DUMP_T5=<dir>: layer-0 stage buffers (full [n, ...] rows) for offline parity
    const char * dump_dir = getenv("TN_DUMP_T5");
    auto dump = [&](uint32_t l, const char * nm, const void * p, size_t cnt) {
        if (!dump_dir || l != 0) return;
        rt.dev.commit();
        FILE * g = fopen((std::string(dump_dir) + "/t5l0_" + nm + ".bin").c_str(), "wb");
        if (g) { fwrite(p, 4, cnt, g); fclose(g); }
    };
    for (uint32_t l = 0; l < NL; ++l) {
        char b[64];
        snprintf(b, sizeof b, "blk.%u", l);
        std::string bp = b;
        dump(l, "xb", xb.p, (size_t) n * d_model);
        rt.rms_w(xb.p, rb.p, W->need(bp + ".attn_norm").buf.p, n, d_model, eps);
        rt.gemm(W->is_f16(bp + ".q"), W->need(bp + ".q").buf.p, rb.p, qkv.p, nullptr, {n, d_model, d_model, d_model, d_model, 3 * d_model});
        rt.gemm(W->is_f16(bp + ".k"), W->need(bp + ".k").buf.p, rb.p, (char *) qkv.p + (size_t) d_model * 4,
                nullptr, {n, d_model, d_model, d_model, d_model, 3 * d_model});
        rt.gemm(W->is_f16(bp + ".v"), W->need(bp + ".v").buf.p, rb.p, (char *) qkv.p + (size_t) 2 * d_model * 4,
                nullptr, {n, d_model, d_model, d_model, d_model, 3 * d_model});
        dump(l, "rb", rb.p, (size_t) n * d_model);
        dump(l, "qkv", qkv.p, (size_t) n * 3 * d_model);
        attn_call(rt, n);
        dump(l, "probs", probs.p, (size_t) NH * n * n);
        dump(l, "att", att.p, (size_t) n * d_model);
        rt.gemm(W->is_f16(bp + ".o"), W->need(bp + ".o").buf.p, att.p, ff1.p, nullptr, {n, d_model, d_model});
        rt.add_(xb.p, ff1.p, xb.p, (size_t) n * d_model);
        dump(l, "xattn", xb.p, (size_t) n * d_model);
        rt.rms_w(xb.p, rb.p, W->need(bp + ".ffn_norm").buf.p, n, d_model, eps);
        rt.gemm(W->is_f16(bp + ".wi_0"), W->need(bp + ".wi_0").buf.p, rb.p, ff1.p, nullptr, {n, d_ff, d_model});
        rt.gemm(W->is_f16(bp + ".wi_1"), W->need(bp + ".wi_1").buf.p, rb.p, ff2.p, nullptr, {n, d_ff, d_model});
        rt.geglu(ff1.p, ff2.p, ff1.p, (size_t) n * d_ff);
        dump(l, "gated", ff1.p, (size_t) n * d_ff);
        rt.gemm(W->is_f16(bp + ".wo"), W->need(bp + ".wo").buf.p, ff1.p, rb.p, nullptr, {n, d_model, d_ff});
        rt.add_(xb.p, rb.p, xb.p, (size_t) n * d_model);
        dump(l, "xout", xb.p, (size_t) n * d_model);
    }
    rt.rms_w(xb.p, out.p, W->need("out_norm").buf.p, n, d_model, eps);
}

} // namespace tn
