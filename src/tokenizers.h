// tokenizers.h — CLIP BPE + T5 sentencepiece Viterbi (port of thinksound.cpp tokenizer).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gguf.h"

namespace tn {

struct ClipTokenizer {
    std::map<std::string, int32_t> vocab;
    std::map<std::string, int32_t> merge_rank;
    std::map<int, std::string> byte2u;
    int32_t bos_id = 49406, eos_id = 49407;

    bool load(const std::string & gguf_path);
    // Lowercase, whitespace-split, byte BPE. Always length ctx (eos-padded).
    std::vector<int32_t> encode(const std::string & text, int ctx) const;
};

struct T5Tokenizer {
    std::vector<std::string> pieces;
    std::vector<float> scores;
    std::map<std::string, int32_t> piece2id;
    int32_t unk_id = 2, eos_id = 1, pad_id = 0;

    bool load(const std::string & gguf_path);
    // Sentencepiece unigram Viterbi. pad=true pads with pad_id to ctx.
    std::vector<int32_t> encode(const std::string & text, int ctx, bool pad) const;
};

} // namespace tn
