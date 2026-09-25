// gpu.h — thin Metal harness. Unified-memory shared buffers (Apple Silicon),
// command-buffer batching, kernel dispatch with buffer args.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tn::gpu {

struct Impl; // Objective-C++ side

// Shared-memory buffer: host pointer == device pointer (unified memory).
struct Buf {
    void * p = nullptr;      // host-visible base
    size_t bytes = 0;
    bool ok() const { return p != nullptr; }
};

class Device {
public:
    ~Device();
    // metallib_path: precompiled library. If empty, compiles kernels.metal from
    // the directory of the executable.
    bool init(const std::string & metallib_path);
    Buf alloc(size_t bytes);
    // Wrap existing host memory as a device buffer (no copy; must stay alive).
    Buf wrap(const void * p, size_t bytes);

    // --- command batching ---
    void begin();
    // kernel dispatch. buffers: alternating (const void* ptr, size_t offset) pairs.
    // scalars: copied into a constant buffer slot (max 64 bytes) passed as last arg.
    void dispatch(const std::string & kernel,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t tx, uint32_t ty, uint32_t tz,
                  const std::vector<const void *> & buffers,
                  const void * scalars = nullptr, size_t scalar_bytes = 0);
    void commit();   // submit + wait
    double last_gpu_ms() const { return last_gpu_ms_; }

    Impl * impl() const { return impl_; }

private:
    Impl * impl_ = nullptr;
    double last_gpu_ms_ = 0.0;
};

// Shared constant-block struct definitions mirror kernels.metal.

struct GemmP {
    uint32_t M, N, K;
    uint32_t ldx, ldw, ldy;   // element strides between rows
    uint32_t x_rows2;         // 1 = single X; 2 = X holds two stacked arms [2M,K]
    uint32_t pad0;
};
static_assert(sizeof(GemmP) <= 64, "constant block budget");

struct AttnP {
    uint32_t Sq, Sk, hd, nh;
    uint32_t ld;              // row stride of q/k/v buffer
    uint32_t ldo;             // row stride of output buffer
    uint32_t q_off, k_off, v_off, o_off; // column offsets (elements)
    uint32_t bias_mode;       // 0 none, 1 t5 rel bias, 2 clip causal+pad
    uint32_t eot;             // clip: eos position
    float scale;
    uint32_t p_off;           // scratch offset (elements) for probs
};
static_assert(sizeof(AttnP) <= 64, "constant block budget");

struct ConvP {
    uint32_t T, OL, IC, OC, K, stride, pad, dil;
    uint32_t ldx, ldy;        // row strides (elements)
    uint32_t flip;            // convT gather test (maps to metal convt)
    uint32_t use_bias;        // 1 when a real bias is bound
};
static_assert(sizeof(ConvP) <= 64, "constant block budget");

} // namespace tn::gpu
