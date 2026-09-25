// tools/parity_dump.cpp — run one native DiT forward on the reference parity inputs,
// dump matching boundary stages to <dir>/nat_*.bin for stage-by-stage comparison.
#include "models.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}
// host [rows, C] row-major -> ggml [feature, seq] layout (c fastest)
static std::vector<float> to_feat_layout(const float * p, uint32_t rows, uint32_t C) {
    std::vector<float> out((size_t) rows * C);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < C; ++c) out[(size_t) c * rows + r] = p[(size_t) r * C + c];
    return out;
}
static void dump(const std::string & dir, const std::string & name, const std::vector<float> & v) {
    std::string p = dir + "/nat_" + name + ".bin";
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", p.c_str()); return; }
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %s (%zu)\n", p.c_str(), v.size());
}

int main(int argc, char ** argv) {
    const char * dir = argc > 1 ? argv[1] : "/tmp/ts-weights";
    const char * pdir = argc > 2 ? argv[2] : "/tmp/ts-parity";
    int dur = argc > 3 ? atoi(argv[3]) : 2;
    float t = argc > 4 ? (float) atof(argv[4]) : 1.0f;

    const uint32_t T = (uint32_t) roundf(44100.0f / 64.0f / 32.0f * 0.46875f * 0 + 0);  // unused
    (void) T;
    // dims: T = round(21.53*dur), clip_S = 8*dur, sync_S = 24*dur
    const uint32_t Tt = (uint32_t) roundf(21.533203125f * dur);
    const uint32_t clip_S = (uint32_t) (8 * dur);
    const uint32_t sync_S = (uint32_t) (24 * dur);
    printf("T=%u clip_S=%u sync_S=%u\n", Tt, clip_S, sync_S);

    std::vector<float> noise = read_bin(std::string(pdir) + "/noise.bin");

    tn::Runtime rt;
    if (!rt.init(std::string(dir) + "/thinksound.metallib")) { fprintf(stderr, "metallib\n"); return 1; }

    tn::DitModel dit;
    if (!dit.load(rt, std::string(dir) + "/dit-bf16.gguf")) return 1;

    // text arm inputs built FIRST: text_cat is the corruption canary
    const uint32_t TS = 154;
    std::vector<float> t5feat = read_bin(std::string(pdir) + "/t5feat.bin");
    std::vector<float> mt = read_bin(std::string(pdir) + "/mt.bin");
    std::vector<float> mg = read_bin(std::string(pdir) + "/mg.bin");
    tn::gpu::Buf text_cat = rt.dev.alloc((size_t) TS * 2048 * 4);
    memset(text_cat.p, 0, (size_t) TS * 2048 * 4);
    for (uint32_t s = 0; s < 77; ++s)
        memcpy((char *) text_cat.p + (size_t) s * 2048 * 4, (char *) mt.data() + (size_t) s * 1024 * 4, 1024 * 4);
    memcpy((char *) text_cat.p + (size_t) 77 * 2048 * 4, t5feat.data(), (size_t) 77 * 2048 * 4);
    tn::gpu::Buf gtext = rt.dev.alloc(1024 * 4);
    memcpy(gtext.p, mg.data(), 1024 * 4);
    auto canary = [&](const char * tag) {
        const float * f = (const float *) text_cat.p;
        double ss = 0;
        for (size_t i = 0; i < (size_t) TS * 2048; ++i) ss += (double) f[i] * f[i];
        fprintf(stderr, "[canary %s] text_cat rms=%.6g\n", tag, sqrt(ss / (double) (TS * 2048)));
    };
    canary("built");
    // weight canary: FNV hash of every weight buffer, rechecked after prep
    auto whash = [&](const char * tag) -> void {
        uint64_t h = 1469598103934665603ull;
        for (const auto & kv : dit.W->ents()) {
            const tn::gpu::Buf & b = kv.second.buf;
            if (!b.ok()) continue;
            const uint8_t * p = (const uint8_t *) b.p;
            for (size_t i = 0; i < b.bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        }
        fprintf(stderr, "[whash %s] %llu\n", tag, (unsigned long long) h);
    };
    whash("loaded");

    // conditioning: empty clip/sync expanded
    std::vector<float> ec_h = dit.W->host("empty_clip_feat");
    std::vector<float> es_h = dit.W->host("empty_sync_feat");
    tn::gpu::Buf clip_f = rt.dev.alloc((size_t) clip_S * 1024 * 4);
    tn::gpu::Buf sync_f = rt.dev.alloc((size_t) sync_S * 768 * 4);
    for (uint32_t s = 0; s < clip_S; ++s) memcpy((char *) clip_f.p + s * 1024 * 4, ec_h.data(), 1024 * 4);
    for (uint32_t s = 0; s < sync_S; ++s) memcpy((char *) sync_f.p + s * 768 * 4, es_h.data(), 768 * 4);
    if (!dit.prep_conditioning(rt, clip_f, sync_f, Tt, clip_S, sync_S)) return 1;

    canary("after-prep");
    whash("after-prep");
    tn::gpu::Buf text_pp = rt.dev.alloc((size_t) TS * dit.H * 4);
    tn::gpu::Buf gcmlp = rt.dev.alloc(dit.H * 4);
    dit.prep_text_arm(rt, text_cat, gtext, text_pp, gcmlp);
    canary("after-textarm");

    // dump conditioning stages BEFORE forward (blocks update these in place)
    auto dumpRC = [&](const char * name, const void * p, uint32_t rows, uint32_t C) {
        std::vector<float> tmp((size_t) rows * C);
        memcpy(tmp.data(), p, (size_t) rows * C * 4);
        dump(pdir, name, tmp);   // ggml ne0-fastest == row-major here
    };
    dumpRC("clip_pp", dit.clip_pp.p, clip_S, dit.H);
    dumpRC("text_pp", text_pp.p, TS, dit.H);
    dumpRC("gcmlp", gcmlp.p, 1, dit.H);

    // reference noise is time-major flat: x[t,c] = noise[c*Tt + t]; we are row-major [t,c]
    tn::gpu::Buf x = rt.dev.alloc(noise.size() * 4);
    for (uint32_t t = 0; t < Tt; ++t)
        for (uint32_t c = 0; c < 64; ++c)
            ((float *) x.p)[(size_t) t * 64 + c] = noise[(size_t) c * Tt + t];
    tn::gpu::Buf flow = rt.dev.alloc(noise.size() * 4);
    setenv("TN_DUMP_EARLY", pdir, 1);
    dit.forward(rt, x, t, text_pp, gcmlp, flow);
    rt.dev.commit();
    // early dumps (t_emb/global_c/extended_c/x_in) are raw row-major; convert
    {
        std::vector<float> tmp((size_t) Tt * dit.H);
        FILE * f = fopen((std::string(pdir) + "/nat_x_in.bin").c_str(), "rb");
        if (f) { fread(tmp.data(), 4, tmp.size(), f); fclose(f);
                 dump(pdir, "x_in", tmp); }
        f = fopen((std::string(pdir) + "/nat_extended_c.bin").c_str(), "rb");
        if (f) { fread(tmp.data(), 4, tmp.size(), f); fclose(f);
                 dump(pdir, "extended_c", tmp); }
    }
    {   // flow: reference layout is [T,64] c-fast, same as the GPU buffer
        std::vector<float> tmp((size_t) Tt * 64);
        memcpy(tmp.data(), flow.p, tmp.size() * 4);
        dump(pdir, "flow", tmp);
    }
    return 0;
}
