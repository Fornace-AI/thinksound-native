// tokenizers.cpp — direct port of upstream tokenizer.cpp (license: upstream thinksound.cpp,
// MIT). Logic kept identical for token parity.
#include "tokenizers.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>

namespace tn {
namespace {
std::string utf8(int cp) {
    std::string s;
    if (cp < 0x80) s += (char) cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}
std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}
} // namespace

bool ClipTokenizer::load(const std::string & path) {
    GgufFile m;
    if (!m.open(path)) return false;
    bos_id = (int32_t) m.get_i64("clip.bos_id", 49406);
    eos_id = (int32_t) m.get_i64("clip.eos_id", 49407);
    auto v = m.get_arr_str("clip.vocab");
    for (size_t i = 0; i < v.size(); ++i) vocab[v[i]] = (int32_t) i;
    auto mg = m.get_arr_str("clip.merges");
    for (size_t i = 0; i < mg.size(); ++i) merge_rank[mg[i]] = (int32_t) i;
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b)
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    for (size_t i = 0; i < bs.size(); ++i) byte2u[bs[i]] = utf8(cs[i]);
    return !vocab.empty();
}

std::vector<int32_t> ClipTokenizer::encode(const std::string & text, int ctx) const {
    std::string t;
    for (char ch : text) t += (char) std::tolower((unsigned char) ch);
    std::vector<std::string> words;
    { std::string w; for (char ch : t) { if (std::isspace((unsigned char) ch)) { if (!w.empty()) { words.push_back(w); w.clear(); } } else w += ch; } if (!w.empty()) words.push_back(w); }

    std::vector<int32_t> ids; ids.push_back(bos_id);
    for (const auto & w : words) {
        std::vector<std::string> syms;
        for (unsigned char b : w) syms.push_back(byte2u.at(b));
        if (syms.empty()) continue;
        syms.back() += "</w>";
        for (;;) {
            int best_rank = INT32_MAX, best_i = -1;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = merge_rank.find(syms[i] + " " + syms[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_i = (int) i; }
            }
            if (best_i < 0) break;
            syms[best_i] += syms[best_i + 1];
            syms.erase(syms.begin() + best_i + 1);
        }
        for (const auto & s : syms) {
            auto it = vocab.find(s);
            if (it != vocab.end()) ids.push_back(it->second);
        }
    }
    ids.push_back(eos_id);
    if ((int) ids.size() > ctx) { ids.resize(ctx); ids[ctx - 1] = eos_id; }
    while ((int) ids.size() < ctx) ids.push_back(eos_id);
    return ids;
}

bool T5Tokenizer::load(const std::string & path) {
    GgufFile m;
    if (!m.open(path)) return false;
    unk_id = (int32_t) m.get_i64("t5.unk_id", 2);
    eos_id = (int32_t) m.get_i64("t5.eos_id", 1);
    pad_id = (int32_t) m.get_i64("t5.pad_id", 0);
    pieces = m.get_arr_str("t5.pieces");
    auto sc = m.get_arr_f32("t5.scores");
    scores.resize(pieces.size());
    for (size_t i = 0; i < pieces.size() && i < sc.size(); ++i) scores[i] = sc[i];
    for (size_t i = 0; i < pieces.size(); ++i) piece2id[pieces[i]] = (int32_t) i;
    return !pieces.empty();
}

std::vector<int32_t> T5Tokenizer::encode(const std::string & text, int ctx, bool pad) const {
    const std::string UND = "\xE2\x96\x81";
    std::string t;
    {
        std::string s = text;
        size_t a = s.find_first_not_of(" \t\n"), b = s.find_last_not_of(" \t\n");
        if (a != std::string::npos) s = s.substr(a, b - a + 1); else s.clear();
        t = UND;
        for (char ch : s) t += (ch == ' ') ? UND : std::string(1, ch);
    }
    std::vector<std::string> chars = utf8_chars(t);
    int L = (int) chars.size();
    double min_score = 0.0;
    for (float s : scores) if ((double) s < min_score) min_score = (double) s;
    const double unk_pen = min_score - 10.0;
    std::vector<double> best(L + 1, -1e18);
    std::vector<int> bp(L + 1, -1), bpid(L + 1, -1);
    best[0] = 0;
    for (int i = 0; i < L; ++i) {
        if (best[i] <= -1e17) continue;
        std::string piece;
        for (int j = i; j < L; ++j) {
            piece += chars[j];
            auto it = piece2id.find(piece);
            if (it != piece2id.end()) {
                double sc = best[i] + scores[it->second];
                if (sc > best[j + 1]) { best[j + 1] = sc; bp[j + 1] = i; bpid[j + 1] = it->second; }
            }
            if ((int) piece.size() > 64) break;
        }
        double su = best[i] + unk_pen;
        if (su > best[i + 1]) { best[i + 1] = su; bp[i + 1] = i; bpid[i + 1] = unk_id; }
    }
    std::vector<int32_t> ids;
    for (int i = L; i > 0; i = bp[i]) { ids.push_back(bpid[i]); if (bp[i] < 0) break; }
    std::reverse(ids.begin(), ids.end());
    ids.push_back(eos_id);
    if ((int) ids.size() > ctx) { ids.resize(ctx); ids[ctx - 1] = eos_id; }
    if (pad) while ((int) ids.size() < ctx) ids.push_back(pad_id);
    return ids;
}

} // namespace tn
