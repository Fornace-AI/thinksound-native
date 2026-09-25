// models.h — ThinkSound native model orchestration.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "net.h"
#include "tokenizers.h"

namespace tn {

constexpr int CLIP_TOKEN_CTX = 77;
constexpr int T5_TOKEN_CTX = 77;
constexpr int LATENT_DIM = 64;
constexpr int SAMPLE_RATE = 44100;
constexpr int VAE_HOP = 2048;
constexpr int CLIP_FPS = 8;
constexpr int SYNC_FPS = 24;

inline int64_t latent_T(int dur) {
    return (int64_t) llround((double) SAMPLE_RATE / VAE_HOP * dur);
}

// ---- MetaCLIP text encoder ----
class ClipModel {
public:
    std::unique_ptr<WeightSet> W;
    uint32_t H = 1024, NL = 24, NH = 16, hd = 64, ctx = 77;
    gpu::Buf xb, rb, qkv, att, probs, xb2, pos;
    std::vector<float> tok_host, pos_host;

    bool load(Runtime & rt, const std::string & path);
    // per_token [ctx,H], global [H]
    void encode(Runtime & rt, const std::vector<int32_t> & ids,
                gpu::Buf & per_token, gpu::Buf & global);
    void attn_call(Runtime & rt, uint32_t n, uint32_t bias_mode, uint32_t eot,
                   const int * bkt, const float * rbias);
};

// ---- T5 v1.1 encoder ----
class T5Model {
public:
    std::unique_ptr<WeightSet> W;
    uint32_t d_model = 2048, d_ff = 5120, NL = 24, NH = 32, hd = 64, n_bucket = 32, max_dist = 128;
    float eps = 1e-6f;
    gpu::Buf xb, rb, qkv, att, probs, ff1, ff2, bktb, rbiasb, padb;
    std::vector<float> tok_host;

    bool load(Runtime & rt, const std::string & path);
    void encode(Runtime & rt, const std::vector<int32_t> & ids, gpu::Buf & out);
    void attn_call(Runtime & rt, uint32_t n);
};

// ---- MMDiT (ThinkSound) ----
class DitModel {
public:
    std::unique_ptr<WeightSet> W;
    uint32_t H = 1024, NH = 16, hd = 64, depth = 21, fused_depth = 14;
    uint32_t latent_dim = 64, H_ffn = 2816;
    float rope_theta = 10000.0f, t_max_period = 1.0f;

    // geometry
    uint32_t T = 0, clip_S = 0, sync_S = 0, text_S = 154, n_joint = 7;

    // buffers
    gpu::Buf clip_pp, sync_gather, text_pp_cond, text_pp_unc, gcmlp_cond, gcmlp_unc;
    gpu::Buf extc, extc_silu, x_in, jointq, sc_qkv, joint_att, mod_lat, mod_ct, mod_fin,
             probs, te_out, global_c, gc_pre, pos_lat, pos_clip;
    gpu::Buf sc_l1, sc_r, sc_f1, sc_f3;   // per-block scratch (no aliasing)
    gpu::Buf h1, h3;                      // prep-time scratch (convmlp temporaries)
    std::vector<float> clip_fc_host{std::vector<float>(1024)};
    bool loaded = false;

    bool load(Runtime & rt, const std::string & path);
    // shared conditioning (clip/sync paths + clip_fc). Call once per generation.
    bool prep_conditioning(Runtime & rt, gpu::Buf & clip_f, gpu::Buf & sync_f,
                            uint32_t T_, uint32_t clip_S_, uint32_t sync_S_);
    // per-arm text conditioning (text_pp + gcmlp). Call once per arm.
    void prep_text_arm(Runtime & rt, gpu::Buf & text_cat, gpu::Buf & gtext,
                       gpu::Buf & text_pp_out, gpu::Buf & gcmlp_out);
    // one forward at flow time t for one arm; writes flow [T,64]
    void forward(Runtime & rt, gpu::Buf & x, float t, gpu::Buf & text_pp, gpu::Buf & gcmlp,
                 gpu::Buf & flow);
};

// ---- Vocos VAE decoder ----
class VaeModel {
public:
    std::unique_ptr<WeightSet> W;
    bool load(Runtime & rt, const std::string & path);
    // latent [T,64] f32 host → audio interleaved stereo f32 host
    std::vector<float> decode(Runtime & rt, const float * latent, uint32_t T, uint32_t & T_audio);
    std::vector<gpu::Buf> stage;   // ping-pong buffers per block stage
};

// ---- pipeline ----
struct GenOpts {
    std::string caption, cot, dir;
    int dur = 9, steps = 24;
    float cfg = 5.0f;
    unsigned seed = 4242;
};
int generate_wav(const GenOpts & o, const std::string & out_path, std::string & err);

} // namespace tn
