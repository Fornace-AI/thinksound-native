// gguf.cpp — minimal mmap GGUF reader implementation.
#include "gguf.h"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tn {

namespace {
const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"

struct Rd {
    const uint8_t *& p;
    const uint8_t * end;
    bool bad = false;
    void raw(void * dst, size_t n) {
        if (bad || p + n > end) { bad = true; return; }
        memcpy(dst, p, n);
        p += n;
    }
    uint32_t u32() { uint32_t v; raw(&v, 4); return v; }
    int32_t i32() { return (int32_t) u32(); }
    uint64_t u64() { uint64_t v; raw(&v, 8); return v; }
    int64_t i64() { return (int64_t) u64(); }
    float f32() { float v; raw(&v, 4); return v; }
    double f64() { double v; raw(&v, 8); return v; }
    uint8_t u8() { uint8_t v; raw(&v, 1); return v; }
    bool boolean() { return u8() != 0; }
    std::string str() {
        uint64_t n = u64();
        if (bad || n > (uint64_t)(end - p)) { bad = true; return {}; }
        std::string s((const char *) p, (size_t) n);
        p += n;
        return s;
    }
};
} // namespace

uint64_t GgufTensor::type_size(uint32_t t) {
    switch (t) {
        case ML_F32: return 4;
        case ML_F16: return 2;
        case ML_BF16: return 2;
        case ML_Q8_0: return 2 + 16;   // f16 scale + 16 * i8
        case ML_Q4_0: return 2 + 8;
        case ML_Q4_1: return 2 + 2 + 8;
        default: return 0;
    }
}

uint64_t GgufTensor::block_elems(uint32_t t) {
    switch (t) {
        case ML_Q8_0: return 16;
        case ML_Q4_0:
        case ML_Q4_1: return 16;
        default: return 1;
    }
}

uint64_t GgufTensor::nelem() const {
    uint64_t n = 1;
    for (uint32_t d = 0; d < n_dims && d < 4; ++d) n *= ne[d];
    return n;
}

GgufFile::~GgufFile() {
    if (map_) munmap(map_, map_len_);
    if (fd_ >= 0) close(fd_);
}

bool GgufFile::parse_header(const uint8_t *& p, const uint8_t * end) {
    Rd r{p, end};
    uint32_t magic = r.u32();
    if (magic != GGUF_MAGIC) { fprintf(stderr, "gguf: bad magic %08x\n", magic); return false; }
    uint32_t version = r.u32();
    if (version < 2 || version > 3) { fprintf(stderr, "gguf: version %u unsupported\n", version); return false; }
    uint64_t n_tensors = r.u64();
    uint64_t n_kv = r.u64();
    if (r.bad) return false;

    // key-value pairs
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str();
        uint32_t ty = r.u32();
        if (r.bad) return false;
        detail::KvValue v;
        v.type = ty;
        switch (ty) {
            case GGUF_U8: case GGUF_I8: v.i = r.u8(); break;
            case GGUF_BOOL: v.b = r.boolean(); v.i = v.b; break;
            case GGUF_U16: { uint16_t x; r.raw(&x, 2); v.i = x; break; }
            case GGUF_I16: { int16_t x; r.raw(&x, 2); v.i = x; break; }
            case GGUF_U32: { uint32_t x; r.raw(&x, 4); v.i = x; break; }
            case GGUF_I32: v.i = r.i32(); break;
            case GGUF_U64: case GGUF_I64: v.i = (int64_t) r.u64(); break;
            case GGUF_F32: v.f = r.f32(); break;
            case GGUF_F64: v.f = r.f64(); break;
            case GGUF_STRING: v.s = r.str(); break;
            case GGUF_ARRAY: {
                v.arr_elem_type = r.u32();
                uint64_t n = r.u64();
                if (r.bad) return false;
                if (v.arr_elem_type == GGUF_STRING) {
                    for (uint64_t k = 0; k < n; ++k) v.arr_s.push_back(r.str());
                } else if (v.arr_elem_type == GGUF_F32) {
                    for (uint64_t k = 0; k < n; ++k) v.arr_f.push_back(r.f32());
                } else if (v.arr_elem_type == GGUF_F64) {
                    for (uint64_t k = 0; k < n; ++k) v.arr_f.push_back((float) r.f64());
                } else if (v.arr_elem_type == GGUF_U32 || v.arr_elem_type == GGUF_I32) {
                    for (uint64_t k = 0; k < n; ++k) v.arr_i.push_back(r.i32());
                } else if (v.arr_elem_type == GGUF_U64 || v.arr_elem_type == GGUF_I64) {
                    for (uint64_t k = 0; k < n; ++k) v.arr_i.push_back((int64_t) r.u64());
                } else {
                    for (uint64_t k = 0; k < n; ++k) r.raw(nullptr, 1);
                }
                break;
            }
            default: return false;
        }
        if (r.bad) return false;
        kv_[key] = std::move(v);  // requires kv_ member type change
    }

    // tensor infos
    uint64_t align = 32;
    if (kv_.count("general.alignment")) align = (uint64_t) kv_["general.alignment"].i;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.str();
        t.n_dims = r.u32();
        if (t.n_dims > 4) return false;
        for (uint32_t d = 0; d < t.n_dims; ++d) t.ne[d] = r.u64();
        t.type = r.u32();
        t.offset = r.u64();
        if (r.bad) return false;
        uint64_t blocks = 1;
        for (uint32_t d = 0; d < t.n_dims; ++d)
            blocks *= (t.ne[d] + GgufTensor::block_elems(t.type) - 1) / GgufTensor::block_elems(t.type);
        t.nbytes = blocks * GgufTensor::type_size(t.type);
        tensors_[t.name] = t;
    }
    // data section begins after header, aligned
    uint64_t hdr_end = (uint64_t)(r.p - (const uint8_t *) map_);
    data_start_ = (hdr_end + align - 1) & ~(align - 1);
    for (auto & [name, t] : tensors_) t.data = (const uint8_t *) map_ + data_start_ + t.offset;
    p = r.p;
    return true;
}

bool GgufFile::open(const std::string & path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { perror(("open " + path).c_str()); return false; }
    struct stat st{};
    if (fstat(fd_, &st) != 0) return false;
    map_len_ = (uint64_t) st.st_size;
    map_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) { map_ = nullptr; return false; }
    const uint8_t * p = (const uint8_t *) map_;
    if (!parse_header(p, p + map_len_)) return false;
    data_size_ = map_len_ - data_start_;
    return true;
}

int64_t GgufFile::get_i64(const std::string & k, int64_t def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second.i;
}
double GgufFile::get_f64(const std::string & k, double def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    return it->second.type == GGUF_F32 || it->second.type == GGUF_F64 ? it->second.f : (double) it->second.i;
}
std::string GgufFile::get_str(const std::string & k, const std::string & def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second.s;
}
std::vector<std::string> GgufFile::get_arr_str(const std::string & k) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? std::vector<std::string>{} : it->second.arr_s;
}
std::vector<float> GgufFile::get_arr_f32(const std::string & k) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? std::vector<float>{} : it->second.arr_f;
}
const GgufTensor * GgufFile::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

std::vector<float> to_f32(const GgufTensor & t) {
    std::vector<float> out((size_t) t.nelem());
    const uint8_t * d = t.data;
    if (t.type == ML_F32) {
        memcpy(out.data(), d, out.size() * 4);
    } else if (t.type == ML_F16 || t.type == ML_BF16) {
        bool bf = t.type == ML_BF16;
        const uint16_t * h = (const uint16_t *) d;
        for (size_t i = 0; i < out.size(); ++i) {
            uint32_t u = h[i];
            if (bf) {
                uint32_t f = (u << 16);
                memcpy(&out[i], &f, 4);
            } else {
                // f16 -> f32
                uint32_t sign = (u >> 15) & 1, exp = (u >> 10) & 0x1F, man = u & 0x3FF;
                uint32_t f;
                if (exp == 0) f = sign << 31;
                else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
                else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
                memcpy(&out[i], &f, 4);
            }
        }
    } else if (t.type == ML_Q8_0) {
        size_t blocks = out.size() / 16;
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t * blk = d + b * 18;
            uint16_t su; memcpy(&su, blk, 2);
            // f16 scale
            uint32_t exp = (su >> 10) & 0x1F, man = su & 0x3FF, sign = (su >> 15) & 1, sf;
            if (exp == 0) sf = sign << 31;
            else sf = (sign << 31) | ((exp + 112) << 23) | (man << 13);
            float s; memcpy(&s, &sf, 4);
            const int8_t * q = (const int8_t *) (blk + 2);
            for (int k = 0; k < 16; ++k) out[b * 16 + k] = s * (float) q[k];
        }
    } else {
        fprintf(stderr, "to_f32: unsupported type %u for %s\n", t.type, t.name.c_str());
        return {};
    }
    return out;
}

} // namespace tn
