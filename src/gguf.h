// gguf.h — minimal mmap GGUF reader (no ggml). ThinkSound native port.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace tn {

namespace detail {
struct KvValue {
    uint32_t type = 0;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<std::string> arr_s;
    std::vector<float> arr_f;
    std::vector<int64_t> arr_i;
    uint32_t arr_elem_type = 0;
};
}

enum GgufType : uint32_t {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32,
    GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64,
};

enum MlType : uint32_t {
    ML_F32 = 0, ML_F16 = 1, ML_Q4_0 = 2, ML_Q4_1 = 3, ML_Q8_0 = 8, ML_BF16 = 30,
};

// One tensor: logical shape + dtype + byte view into the mapping.
struct GgufTensor {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {0, 0, 0, 0};   // ne[0] fastest (input channels); ne[1] rows (out)
    uint32_t type = ML_F32;
    uint64_t offset = 0;             // bytes from data section start
    const uint8_t * data = nullptr;  // absolute pointer into mapping
    uint64_t nbytes = 0;

    // Element count of the logical shape.
    uint64_t nelem() const;
    // Bytes per stored element (quant blocks included).
    static uint64_t type_size(uint32_t t);
    // Elements covered per quant block (1 for scalar types).
    static uint64_t block_elems(uint32_t t);
};

class GgufFile {
public:
    ~GgufFile();
    bool open(const std::string & path);

    // KV metadata
    bool has(const std::string & key) const { return kv_.count(key) != 0; }
    int64_t get_i64(const std::string & key, int64_t def) const;
    double get_f64(const std::string & key, double def) const;
    std::string get_str(const std::string & key, const std::string & def) const;
    std::vector<std::string> get_arr_str(const std::string & key) const;
    std::vector<float> get_arr_f32(const std::string & key) const;

    // Tensors
    size_t n_tensors() const { return tensors_.size(); }
    const GgufTensor * find(const std::string & name) const;
    const std::map<std::string, GgufTensor> & tensors_map() const { return tensors_; }

    uint64_t data_size() const { return data_size_; }

private:
    int fd_ = -1;
    void * map_ = nullptr;
    uint64_t map_len_ = 0;
    uint64_t data_start_ = 0;   // file offset of tensor data section
    uint64_t data_size_ = 0;
    std::map<std::string, detail::KvValue> kv_;
    std::map<std::string, GgufTensor> tensors_;

    bool parse_header(const uint8_t *& p, const uint8_t * end);
};

// Host dequantize to f32 (q8_0, bf16, f16, f32 passthrough).
std::vector<float> to_f32(const GgufTensor & t);

} // namespace tn
