// enc_check.cpp — verify live CLIP + T5 encoders against golden features.
#include "models.h"
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); exit(1); }
    fclose(f);
    return v;
}

static void cmp(const char * name, const float * a, const float * b, size_t n) {
    double da = 0, db = 0, dd = 0;
    for (size_t i = 0; i < n; ++i) {
        da += (double) a[i] * a[i]; db += (double) b[i] * b[i];
        double d = (double) a[i] - b[i]; dd += d * d;
    }
    printf("%-14s rel-L2 = %.6g   (|a|=%.3f |b|=%.3f)\n", name,
           std::sqrt(dd) / (std::sqrt(db) + 1e-30), std::sqrt(da), std::sqrt(db));
}

int main(int argc, char ** argv) {
    const char * dir = argc > 1 ? argv[1] : "/tmp/ts-weights";
    const char * pdir = argc > 2 ? argv[2] : "/tmp/ts-parity";

    tn::Runtime rt;
    if (!rt.init(std::string(dir) + "/thinksound.metallib")) return 1;

    // golden token ids
    FILE * f = fopen((std::string(pdir) + "/t5_ids.bin").c_str(), "rb");
    std::vector<int32_t> t5_ids(tn::T5_TOKEN_CTX);
    if (fread(t5_ids.data(), 4, t5_ids.size(), f) != t5_ids.size()) return 1;
    fclose(f);
    f = fopen((std::string(pdir) + "/clip_ids.bin").c_str(), "rb");
    std::vector<int32_t> clip_ids(tn::CLIP_TOKEN_CTX);
    if (fread(clip_ids.data(), 4, clip_ids.size(), f) != clip_ids.size()) return 1;
    fclose(f);

    // T5
    tn::T5Model t5;
    if (!t5.load(rt, std::string(dir) + "/t5-q8.gguf")) return 1;
    tn::gpu::Buf t5_out = rt.dev.alloc((size_t) tn::T5_TOKEN_CTX * 2048 * 4);
    t5.encode(rt, t5_ids, t5_out);
    rt.dev.commit();
    std::vector<float> gold_t5 = read_bin(std::string(pdir) + "/t5feat.bin");
    cmp("t5 features", (const float *) t5_out.p, gold_t5.data(), gold_t5.size());

    // CLIP
    tn::ClipModel clip;
    if (!clip.load(rt, std::string(dir) + "/metaclip-text-f32.gguf")) return 1;
    tn::gpu::Buf clip_toks = rt.dev.alloc((size_t) tn::CLIP_TOKEN_CTX * 1024 * 4);
    tn::gpu::Buf clip_glob = rt.dev.alloc(1024 * 4);
    clip.encode(rt, clip_ids, clip_toks, clip_glob);
    rt.dev.commit();
    std::vector<float> gold_mt = read_bin(std::string(pdir) + "/mt.bin");
    cmp("clip tokens", (const float *) clip_toks.p, gold_mt.data(), gold_mt.size());
    std::vector<float> gold_mg = read_bin(std::string(pdir) + "/mg.bin");
    cmp("clip global", (const float *) clip_glob.p, gold_mg.data(), gold_mg.size());
    return 0;
}
