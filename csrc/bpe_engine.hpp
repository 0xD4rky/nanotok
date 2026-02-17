#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class BPEEngine {
public:
  struct Config {
    bool add_prefix_space = false;
    bool trim_offsets = true;
  };

  struct AddedToken {
    std::string content;
    int id;
    bool special;
  };

  // Pre-tokenizer modes
  enum PreTokenizerMode : int {
    PT_DEFAULT    = 0,  // Qwen / generic ByteLevel (hand-written approx)
    PT_CL100K     = 1,  // cl100k_base (GPT-4)
    PT_O200K      = 2,  // o200k_base  (GPT-4o)
    PT_GPT2       = 3,  // GPT-2 / r50k / p50k
    PT_METASPACE  = 4,  // SentencePiece Metaspace (Mistral, etc.)
    PT_SPLIT_SPC  = 5,  // Split on space (Gemma, etc.)
  };

  explicit BPEEngine(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f) throw std::runtime_error("Cannot open tokenizer.json at " + tokenizer_json_path);
    json j;
    f >> j;
    parse_and_build(j);
  }

  static BPEEngine from_json_string(const std::string& json_str) {
    json j = json::parse(json_str);
    BPEEngine engine;
    engine.parse_and_build(j);
    return engine;
  }

  void set_pretokenizer_mode(int mode) { pt_mode_ = static_cast<PreTokenizerMode>(mode); }
  int get_pretokenizer_mode() const { return static_cast<int>(pt_mode_); }

  std::vector<int> encode(const std::string& text,
                          const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<int> out;
    out.reserve(text.size() / 3);
    encode_into(text, allowed_special, out);
    return out;
  }

  std::vector<std::vector<int>> batch_encode(
      const std::vector<std::string>& texts,
      const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<std::vector<int>> results(texts.size());
    for (size_t i = 0; i < texts.size(); i++) {
      results[i].reserve(texts[i].size() / 3);
      encode_into(texts[i], allowed_special, results[i]);
    }
    return results;
  }

  // Encode pre-tokenized chunks (bypasses built-in pre-tokenizer).
  std::vector<int> encode_chunks(const std::vector<std::string>& chunks) const {
    std::vector<int> out;
    for (const auto& chunk : chunks) {
      if (!chunk.empty())
        encode_chunk(chunk.data(), chunk.size(), out);
    }
    return out;
  }

  // ---- Decode: precomputed contiguous pool, two-pass exact-alloc ----

  std::string decode(const std::vector<int>& ids) const {
    const int n = static_cast<int>(ids.size());
    const unsigned piece_count = static_cast<unsigned>(decode_pieces_.size());
    const char* pool = decode_pool_.data();

    size_t total = 0;
    for (int i = 0; i < n; i++) {
      unsigned id = static_cast<unsigned>(ids[i]);
      if (__builtin_expect(id < piece_count, 1))
        total += decode_pieces_[id].len;
    }

    std::string out(total, '\0');
    char* dst = out.data();
    for (int i = 0; i < n; i++) {
      unsigned id = static_cast<unsigned>(ids[i]);
      if (__builtin_expect(id < piece_count, 1)) {
        auto& p = decode_pieces_[id];
        memcpy(dst, pool + p.offset, p.len);
        dst += p.len;
      }
    }

    return out;
  }

  std::vector<std::string> batch_decode(
      const std::vector<std::vector<int>>& batch_ids) const {
    std::vector<std::string> results(batch_ids.size());
    for (size_t i = 0; i < batch_ids.size(); i++)
      results[i] = decode(batch_ids[i]);
    return results;
  }

  inline int token_to_id(const std::string& t) const {
    auto it = token_to_id_.find(t);
    return (it == token_to_id_.end()) ? -1 : it->second;
  }

  int vocab_size() const {
    return static_cast<int>(id_to_token_vec_.size());
  }

  std::string id_to_token(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_vec_.size()))
      return "";
    return id_to_token_vec_[id];
  }

  const std::vector<AddedToken>& get_added_tokens() const {
    return added_tokens_;
  }

  std::unordered_set<int> get_all_special_ids() const {
    std::unordered_set<int> ids;
    for (const auto& t : added_tokens_) {
      if (t.special) ids.insert(t.id);
    }
    return ids;
  }

  const Config& config() const { return config_; }

  // Debug: classify a Unicode codepoint
  int debug_classify(int codepoint) const {
    if (is_unicode_letter(static_cast<uint32_t>(codepoint))) return 1;
    if (is_unicode_digit(static_cast<uint32_t>(codepoint))) return 2;
    return 0;
  }
  int debug_letter_ranges_count() const { return static_cast<int>(unicode_letter_ranges_count); }

  // Debug: return pre-tokenizer chunk boundaries for a text
  std::vector<std::string> debug_chunks(const std::string& text) const {
    std::vector<std::string> chunks;
    size_t pos = 0;
    const size_t len = text.size();
    const char* data = text.data();
    while (pos < len) {
      size_t chunk_end;
      switch (pt_mode_) {
        case PT_CL100K: case PT_O200K:
          chunk_end = find_next_chunk_cl100k(data, len, pos); break;
        case PT_GPT2:
          chunk_end = find_next_chunk_gpt2(data, len, pos); break;
        default:
          chunk_end = find_next_chunk(data, len, pos); break;
      }
      if (chunk_end > pos) {
        chunks.push_back(text.substr(pos, chunk_end - pos));
        pos = chunk_end;
      } else {
        chunks.push_back(text.substr(pos, 1));
        pos++;
      }
    }
    return chunks;
  }

private:
  BPEEngine() = default;

  Config config_;
  PreTokenizerMode pt_mode_ = PT_DEFAULT;
  bool byte_fallback_ = false;  // SentencePiece-style <0xHH> byte tokens
  bool ignore_merges_ = false;  // Skip BPE if full chunk is already in vocab
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_vec_;
  std::vector<std::pair<int,int>> merges_;
  std::vector<AddedToken> added_tokens_;

  int byte_initial_id_[256];

  enum CClass : uint8_t { CC_OTHER=0, CC_LETTER=1, CC_DIGIT=2, CC_SPACE=3, CC_NL=4, CC_APOS=5, CC_HIGH=6 };
  uint8_t cclass_[256];

  // Unicode category tables (generated from UCD)
  #include "unicode_tables.inc"

  // Binary search in sorted range table: returns true if cp is in any [lo, hi] range
  static bool in_ranges(uint32_t cp, const uint32_t ranges[][2], size_t count) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (cp > ranges[mid][1]) lo = mid + 1;
      else if (cp < ranges[mid][0]) hi = mid;
      else return true;
    }
    return false;
  }

  static bool is_unicode_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return in_ranges(cp, unicode_letter_ranges, unicode_letter_ranges_count);
  }

  static bool is_unicode_digit(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(cp, unicode_number_ranges, unicode_number_ranges_count);
  }

  // Decode one UTF-8 character and return its codepoint + byte length
  static inline size_t utf8_decode(const char* s, size_t len, size_t pos, uint32_t& cp) {
    uint8_t c0 = static_cast<uint8_t>(s[pos]);
    if (c0 < 0x80) { cp = c0; return 1; }
    if ((c0 >> 5) == 0x6 && pos + 1 < len) {
      cp = ((c0 & 0x1F) << 6) | (static_cast<uint8_t>(s[pos+1]) & 0x3F);
      return 2;
    }
    if ((c0 >> 4) == 0xE && pos + 2 < len) {
      cp = ((c0 & 0x0F) << 12) | ((static_cast<uint8_t>(s[pos+1]) & 0x3F) << 6) | (static_cast<uint8_t>(s[pos+2]) & 0x3F);
      return 3;
    }
    if ((c0 >> 3) == 0x1E && pos + 3 < len) {
      cp = ((c0 & 0x07) << 18) | ((static_cast<uint8_t>(s[pos+1]) & 0x3F) << 12) | ((static_cast<uint8_t>(s[pos+2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[pos+3]) & 0x3F);
      return 4;
    }
    cp = c0; return 1; // fallback: treat as single byte
  }

  // Classify a position in text as CClass, consuming the full UTF-8 char.
  // Returns the CClass and advances `bytes` to the number of bytes consumed.
  CClass classify_utf8(const char* text, size_t len, size_t pos, size_t& bytes) const {
    uint8_t c0 = static_cast<uint8_t>(text[pos]);
    if (c0 < 0x80) {
      bytes = 1;
      return static_cast<CClass>(cclass_[c0]);
    }
    // Multi-byte UTF-8: decode codepoint and classify
    uint32_t cp;
    bytes = utf8_decode(text, len, pos, cp);
    if (is_unicode_letter(cp)) return CC_LETTER;
    if (is_unicode_digit(cp)) return CC_DIGIT;
    return CC_OTHER;
  }

  struct MergeInfo { int rank; int merged_id; };

  struct MergeMap {
    static constexpr uint64_t EMPTY = ~0ULL;
    struct Slot { uint64_t key; MergeInfo val; };
    std::vector<Slot> slots_;
    uint32_t mask_ = 0;

    static uint64_t mix(uint64_t h) {
      h ^= h >> 33;
      h *= 0xff51afd7ed558ccdULL;
      h ^= h >> 33;
      return h;
    }
    void build(size_t count) {
      uint32_t cap = 16;
      while (cap < count * 2) cap <<= 1;
      slots_.assign(cap, {EMPTY, {}});
      mask_ = cap - 1;
    }
    void insert(uint64_t key, MergeInfo val) {
      uint32_t idx = static_cast<uint32_t>(mix(key)) & mask_;
      while (slots_[idx].key != EMPTY) {
        if (slots_[idx].key == key) { slots_[idx].val = val; return; }
        idx = (idx + 1) & mask_;
      }
      slots_[idx] = {key, val};
    }
    inline const MergeInfo* find(uint64_t key) const {
      uint32_t idx = static_cast<uint32_t>(mix(key)) & mask_;
      while (true) {
        if (slots_[idx].key == key) return &slots_[idx].val;
        if (slots_[idx].key == EMPTY) return nullptr;
        idx = (idx + 1) & mask_;
      }
    }
  } merge_map_;

  std::unordered_map<char, std::vector<std::pair<std::string,int>>> special_by_prefix_;

  struct CacheSlot {
    uint64_t hash;
    uint16_t data_len;
    uint16_t token_count;
    int tokens[13];
  };
  static constexpr size_t CACHE_SLOTS = 1 << 14;
  static constexpr size_t CACHE_MASK = CACHE_SLOTS - 1;
  static constexpr int MAX_CACHED_TOKENS = 13;
  mutable std::vector<CacheSlot> slot_cache_;

  struct DecodePiece { uint32_t offset; uint16_t len; };
  std::vector<DecodePiece> decode_pieces_;
  std::vector<char> decode_pool_;

  uint8_t decode_byte_[512];
  bool decode_valid_[512];

  static inline uint64_t pack_pair(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
  }

  static inline uint64_t chunk_hash(const char* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
      h ^= static_cast<uint8_t>(data[i]);
      h *= 0x100000001b3ULL;
    }
    return h ^ (h >> 32);
  }

  // ---- Encode core ----

  void encode_into(const std::string& text,
                   const std::unordered_set<std::string>& allowed_special,
                   std::vector<int>& out) const {
    // SentencePiece-style: replace spaces with ▁, encode as single chunk
    if (pt_mode_ == PT_METASPACE) {
      encode_spiece(text, allowed_special, out, /*prepend_underscore=*/true);
      return;
    }
    if (pt_mode_ == PT_SPLIT_SPC) {
      encode_spiece(text, allowed_special, out, /*prepend_underscore=*/false);
      return;
    }

    size_t pos = 0;
    const size_t len = text.size();
    const char* data = text.data();

    while (pos < len) {
      if (__builtin_expect(!special_by_prefix_.empty(), 0)) {
        auto it = special_by_prefix_.find(data[pos]);
        if (it != special_by_prefix_.end()) {
          size_t matched = 0;
          int matched_id = -1;
          for (const auto& [tok, id] : it->second) {
            if (!allowed_special.empty() && !allowed_special.count(tok)) continue;
            if (tok.size() <= len - pos &&
                memcmp(tok.data(), data + pos, tok.size()) == 0) {
              if (tok.size() > matched) { matched = tok.size(); matched_id = id; }
            }
          }
          if (matched) {
            out.push_back(matched_id);
            pos += matched;
            continue;
          }
        }
      }

      size_t chunk_end;
      switch (pt_mode_) {
        case PT_CL100K:
        case PT_O200K:
          chunk_end = find_next_chunk_cl100k(data, len, pos);
          break;
        case PT_GPT2:
          chunk_end = find_next_chunk_gpt2(data, len, pos);
          break;
        default:
          chunk_end = find_next_chunk(data, len, pos);
          break;
      }

      if (__builtin_expect(chunk_end > pos, 1)) {
        if (byte_fallback_)
          encode_chunk_sp(data + pos, chunk_end - pos, out);
        else
          encode_chunk(data + pos, chunk_end - pos, out);
        pos = chunk_end;
      } else {
        pos++;
      }
    }
  }

  // SentencePiece-style encode: replace spaces with ▁, encode as one chunk.
  // Used for both Metaspace (prepend ▁) and Split-on-space (no prepend).
  void encode_spiece(const std::string& text,
                     const std::unordered_set<std::string>& allowed_special,
                     std::vector<int>& out,
                     bool prepend_underscore) const {
    // ▁ = U+2581 = UTF-8 bytes E2 96 81
    static const std::string SP = "\xe2\x96\x81";

    std::string transformed;
    transformed.reserve(text.size() + text.size() / 4 + 3);
    for (char c : text) {
      if (c == ' ')
        transformed += SP;
      else
        transformed.push_back(c);
    }
    // Prepend ▁ only if requested AND text doesn't already start with space/▁
    if (prepend_underscore && !text.empty() && text[0] != ' ')
      transformed = SP + transformed;

    // Encode the entire transformed text as one BPE chunk
    encode_chunk_sp(transformed.data(), transformed.size(), out);
  }

  // Convert raw bytes to GPT-2 byte-level string
  std::string bytes_to_bytelevel(const char* data, size_t len) const {
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; i++) {
      uint8_t b = static_cast<uint8_t>(data[i]);
      int id = byte_initial_id_[b];
      if (id >= 0 && id < static_cast<int>(id_to_token_vec_.size()))
        result += id_to_token_vec_[id];
    }
    return result;
  }

  inline void encode_chunk(const char* data, size_t len, std::vector<int>& out) const {
    if (__builtin_expect(!len, 0)) return;

    uint64_t h = chunk_hash(data, len);
    size_t slot_idx = static_cast<size_t>(h) & CACHE_MASK;
    auto& slot = slot_cache_[slot_idx];

    if (__builtin_expect(slot.hash == h && slot.data_len == static_cast<uint16_t>(len), 1)) {
      out.insert(out.end(), slot.tokens, slot.tokens + slot.token_count);
      return;
    }

    if (len == 1) {
      int id = byte_initial_id_[static_cast<uint8_t>(data[0])];
      if (id >= 0) {
        out.push_back(id);
        slot = {h, 1, 1, {id}};
      }
      return;
    }

    // ignore_merges: try full-word lookup before BPE
    if (__builtin_expect(ignore_merges_, 0)) {
      std::string bl = bytes_to_bytelevel(data, len);
      auto it = token_to_id_.find(bl);
      if (it != token_to_id_.end()) {
        out.push_back(it->second);
        slot = {h, static_cast<uint16_t>(len), 1, {it->second}};
        return;
      }
    }

    size_t before = out.size();
    bpe_encode(data, len, out);
    size_t produced = out.size() - before;

    if (produced <= MAX_CACHED_TOKENS && len <= 0xFFFF) {
      slot.hash = h;
      slot.data_len = static_cast<uint16_t>(len);
      slot.token_count = static_cast<uint16_t>(produced);
      memcpy(slot.tokens, &out[before], produced * sizeof(int));
    }
  }

  // SentencePiece BPE: character-level initial tokenization, byte fallback for unknowns
  void encode_chunk_sp(const char* data, size_t len, std::vector<int>& out) const {
    if (__builtin_expect(!len, 0)) return;

    uint64_t h = chunk_hash(data, len);
    size_t slot_idx = static_cast<size_t>(h) & CACHE_MASK;
    auto& slot = slot_cache_[slot_idx];

    if (__builtin_expect(slot.hash == h && slot.data_len == static_cast<uint16_t>(len), 1)) {
      out.insert(out.end(), slot.tokens, slot.tokens + slot.token_count);
      return;
    }

    // ignore_merges: try full-word lookup before BPE
    if (__builtin_expect(ignore_merges_, 0)) {
      std::string chunk(data, len);
      auto it = token_to_id_.find(chunk);
      if (it != token_to_id_.end()) {
        out.push_back(it->second);
        slot = {h, static_cast<uint16_t>(len), 1, {it->second}};
        return;
      }
    }

    size_t before = out.size();

    // Step 1: Split into UTF-8 characters and map to initial token IDs
    std::vector<int> init_ids;
    init_ids.reserve(len);

    size_t i = 0;
    while (i < len) {
      uint32_t cp;
      size_t adv = utf8_decode(data, len, i, cp);
      std::string ch(data + i, adv);

      auto it = token_to_id_.find(ch);
      if (it != token_to_id_.end()) {
        init_ids.push_back(it->second);
      } else {
        // Byte fallback: emit <0xHH> for each byte of the character
        for (size_t b = 0; b < adv; b++) {
          int bid = byte_initial_id_[static_cast<uint8_t>(data[i + b])];
          if (bid >= 0) init_ids.push_back(bid);
        }
      }
      i += adv;
    }

    // Step 2: Apply BPE merges (same algorithm as bpe_encode but on token IDs directly)
    int n = static_cast<int>(init_ids.size());
    if (n <= 1) {
      for (int j = 0; j < n; j++) out.push_back(init_ids[j]);
    } else {
      std::vector<int> ranks_vec(n, INT_MAX);
      for (int j = 0; j < n - 1; j++) {
        auto m = merge_map_.find(pack_pair(init_ids[j], init_ids[j + 1]));
        ranks_vec[j] = m ? m->rank : INT_MAX;
      }

      while (n > 1) {
        int min_rank = INT_MAX, min_idx = -1;
        for (int j = 0; j < n - 1; j++) {
          if (ranks_vec[j] < min_rank) {
            min_rank = ranks_vec[j];
            min_idx = j;
          }
        }
        if (min_idx < 0) break;

        auto m = merge_map_.find(pack_pair(init_ids[min_idx], init_ids[min_idx + 1]));
        init_ids[min_idx] = m->merged_id;

        int move_count = n - min_idx - 2;
        if (move_count > 0) {
          memmove(&init_ids[min_idx + 1], &init_ids[min_idx + 2], move_count * sizeof(int));
          memmove(&ranks_vec[min_idx + 1], &ranks_vec[min_idx + 2], move_count * sizeof(int));
        }
        n--;

        if (min_idx < n - 1) {
          auto r = merge_map_.find(pack_pair(init_ids[min_idx], init_ids[min_idx + 1]));
          ranks_vec[min_idx] = r ? r->rank : INT_MAX;
        } else {
          ranks_vec[min_idx] = INT_MAX;
        }
        if (min_idx > 0) {
          auto r = merge_map_.find(pack_pair(init_ids[min_idx - 1], init_ids[min_idx]));
          ranks_vec[min_idx - 1] = r ? r->rank : INT_MAX;
        }
      }

      for (int j = 0; j < n; j++)
        out.push_back(init_ids[j]);
    }

    size_t produced = out.size() - before;
    if (produced <= MAX_CACHED_TOKENS && len <= 0xFFFF) {
      slot.hash = h;
      slot.data_len = static_cast<uint16_t>(len);
      slot.token_count = static_cast<uint16_t>(produced);
      memcpy(slot.tokens, &out[before], produced * sizeof(int));
    }
  }

  void bpe_encode(const char* data, size_t len, std::vector<int>& out) const {
    constexpr int STACK_MAX = 128;
    int stack_ids[STACK_MAX], stack_ranks[STACK_MAX];
    std::vector<int> heap_ids, heap_ranks;
    int *ids, *ranks;

    if (__builtin_expect(len <= static_cast<size_t>(STACK_MAX), 1)) {
      ids = stack_ids;
      ranks = stack_ranks;
    } else {
      heap_ids.resize(len);
      heap_ranks.resize(len);
      ids = heap_ids.data();
      ranks = heap_ranks.data();
    }

    int n = static_cast<int>(len);
    for (int i = 0; i < n; i++)
      ids[i] = byte_initial_id_[static_cast<uint8_t>(data[i])];

    for (int i = 0; i < n - 1; i++) {
      auto m = merge_map_.find(pack_pair(ids[i], ids[i + 1]));
      ranks[i] = m ? m->rank : INT_MAX;
    }
    ranks[n - 1] = INT_MAX;

    while (n > 1) {
      int min_rank = INT_MAX, min_idx = -1;
      for (int i = 0; i < n - 1; i++) {
        if (ranks[i] < min_rank) {
          min_rank = ranks[i];
          min_idx = i;
        }
      }
      if (__builtin_expect(min_idx < 0, 0)) break;

      auto m = merge_map_.find(pack_pair(ids[min_idx], ids[min_idx + 1]));
      ids[min_idx] = m->merged_id;

      int move_count = n - min_idx - 2;
      if (move_count > 0) {
        memmove(&ids[min_idx + 1], &ids[min_idx + 2], move_count * sizeof(int));
        memmove(&ranks[min_idx + 1], &ranks[min_idx + 2], move_count * sizeof(int));
      }
      n--;

      if (min_idx < n - 1) {
        auto r = merge_map_.find(pack_pair(ids[min_idx], ids[min_idx + 1]));
        ranks[min_idx] = r ? r->rank : INT_MAX;
      } else {
        ranks[min_idx] = INT_MAX;
      }
      if (min_idx > 0) {
        auto r = merge_map_.find(pack_pair(ids[min_idx - 1], ids[min_idx]));
        ranks[min_idx - 1] = r ? r->rank : INT_MAX;
      }
    }

    for (int i = 0; i < n; i++)
      if (ids[i] >= 0) out.push_back(ids[i]);
  }

  // ---- Pre-tokenizer: Default (Qwen / generic ByteLevel) ----

  size_t find_next_chunk(const char* text, size_t len, size_t start) const {
    if (start >= len) return start;
    size_t pos = start;
    const uint8_t cc_first = cclass_[static_cast<uint8_t>(text[pos])];

    if (cc_first == CC_APOS && pos + 1 < len) {
      const char next = text[pos + 1];
      if (next == 's' || next == 'S' || next == 't' || next == 'T' ||
          next == 'm' || next == 'M' || next == 'd' || next == 'D')
        return pos + 2;
      if (pos + 2 < len) {
        if ((next == 'l' || next == 'L') && (text[pos+2] == 'l' || text[pos+2] == 'L')) return pos + 3;
        if ((next == 'r' || next == 'R') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
        if ((next == 'v' || next == 'V') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
      }
    }

    bool has_prefix = false;
    if (cc_first != CC_LETTER && cc_first != CC_HIGH && cc_first != CC_DIGIT && cc_first != CC_NL) {
      has_prefix = true;
      pos++;
      if (pos >= len) return pos;
    }

    uint8_t cc = cclass_[static_cast<uint8_t>(text[pos])];

    if (cc == CC_LETTER || cc == CC_HIGH) {
      while (pos < len) {
        uint8_t c = cclass_[static_cast<uint8_t>(text[pos])];
        if (c != CC_LETTER && c != CC_HIGH) break;
        pos++;
      }
      return pos;
    }
    if (cc == CC_DIGIT) {
      size_t count = 0;
      while (pos < len && cclass_[static_cast<uint8_t>(text[pos])] == CC_DIGIT && count < 3) { pos++; count++; }
      return pos;
    }
    if (!has_prefix) {
      if (cc_first == CC_NL) {
        while (pos < len) {
          uint8_t c = cclass_[static_cast<uint8_t>(text[pos])];
          if (c != CC_SPACE && c != CC_NL) break;
          pos++;
        }
        return pos;
      }
      if (cc_first == CC_SPACE) {
        while (pos < len && text[pos] == ' ') pos++;
        return pos;
      }
    }
    return pos > start ? pos : start + 1;
  }

  // ---- Pre-tokenizer: cl100k_base / o200k_base (GPT-4/4o) ----
  // Matches: (?i:'s|'t|'re|'ve|'m|'ll|'d)
  //        | [^\r\n\p{L}\p{N}]?\p{L}+
  //        | \p{N}{1,3}
  //        |  ?[^\s\p{L}\p{N}]+[\r\n]*
  //        | \s*[\r\n]+
  //        | \s+(?!\S)
  //        | \s+

  size_t find_next_chunk_cl100k(const char* text, size_t len, size_t start) const {
    if (start >= len) return start;
    size_t pos = start;
    size_t blen;
    CClass cc0 = classify_utf8(text, len, pos, blen);

    // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (text[pos] == '\'' && pos + 1 < len) {
      char c1 = text[pos + 1] | 0x20;
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
        return pos + 2;
      if (pos + 2 < len) {
        char c2 = text[pos + 2] | 0x20;
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e'))
          return pos + 3;
      }
    }

    // Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
    {
      size_t p = pos;
      if (cc0 != CC_NL && cc0 != CC_LETTER && cc0 != CC_DIGIT)
        p += blen;  // consume optional prefix (may be multi-byte)
      if (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_LETTER) {
          while (p < len) {
            cc = classify_utf8(text, len, p, cb);
            if (cc != CC_LETTER) break;
            p += cb;
          }
          return p;
        }
      }
    }

    // Alt 3: \p{N}{1,3}
    if (cc0 == CC_DIGIT) {
      size_t p = pos;
      int count = 0;
      while (p < len && count < 3) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc != CC_DIGIT) break;
        p += cb; count++;
      }
      return p;
    }

    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
      size_t p = pos;
      if (p < len && text[p] == ' ') p++;
      size_t pstart = p;
      while (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_OTHER || cc == CC_APOS) p += cb;
        else break;
      }
      if (p > pstart) {
        while (p < len && (text[p] == '\r' || text[p] == '\n')) p++;
        return p;
      }
    }

    // Alt 5: \s*[\r\n]+
    if (cc0 == CC_SPACE || cc0 == CC_NL) {
      size_t ws_end = pos;
      while (ws_end < len && (cclass_[static_cast<uint8_t>(text[ws_end])] == CC_SPACE ||
                              cclass_[static_cast<uint8_t>(text[ws_end])] == CC_NL))
        ws_end++;

      size_t last_nl = 0;
      bool has_nl = false;
      for (size_t p = pos; p < ws_end; p++) {
        if (cclass_[static_cast<uint8_t>(text[p])] == CC_NL) {
          last_nl = p;
          has_nl = true;
        }
      }
      if (has_nl) return last_nl + 1;

      // Alt 6: \s+(?!\S)
      if (ws_end >= len) return ws_end;
      if (ws_end - pos >= 2) return ws_end - 1;

      // Alt 7: \s+
      return ws_end;
    }

    // Fallback: consume one UTF-8 char
    return pos + blen;
  }

  // ---- Pre-tokenizer: GPT-2 / r50k / p50k ----
  // Matches: '(?:[sdmt]|ll|ve|re)
  //        |  ?\p{L}+
  //        |  ?\p{N}+
  //        |  ?[^\s\p{L}\p{N}]+
  //        | \s+(?!\S)
  //        | \s+

  size_t find_next_chunk_gpt2(const char* text, size_t len, size_t start) const {
    if (start >= len) return start;
    size_t pos = start;
    size_t blen;
    CClass cc0 = classify_utf8(text, len, pos, blen);

    // Alt 1: Contractions (case sensitive in GPT-2)
    if (cc0 == CC_APOS && pos + 1 < len) {
      const char next = text[pos + 1];
      if (next == 's' || next == 't' || next == 'm' || next == 'd')
        return pos + 2;
      if (pos + 2 < len) {
        if (next == 'l' && text[pos+2] == 'l') return pos + 3;
        if (next == 'r' && text[pos+2] == 'e') return pos + 3;
        if (next == 'v' && text[pos+2] == 'e') return pos + 3;
      }
    }

    // Alt 2:  ?\p{L}+
    {
      size_t p = pos;
      if (p < len && text[p] == ' ') p++;
      if (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_LETTER) {
          while (p < len) {
            cc = classify_utf8(text, len, p, cb);
            if (cc != CC_LETTER) break;
            p += cb;
          }
          return p;
        }
      }
    }

    // Alt 3:  ?\p{N}+
    {
      size_t p = pos;
      if (p < len && text[p] == ' ') p++;
      if (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_DIGIT) {
          while (p < len) {
            cc = classify_utf8(text, len, p, cb);
            if (cc != CC_DIGIT) break;
            p += cb;
          }
          return p;
        }
      }
    }

    // Alt 4:  ?[^\s\p{L}\p{N}]+
    {
      size_t p = pos;
      if (p < len && text[p] == ' ') p++;
      size_t pstart = p;
      while (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_OTHER || cc == CC_APOS) p += cb;
        else break;
      }
      if (p > pstart) return p;
    }

    // Alt 5: \s+(?!\S)
    if (cc0 == CC_SPACE || cc0 == CC_NL) {
      size_t ws_end = pos;
      while (ws_end < len && (cclass_[static_cast<uint8_t>(text[ws_end])] == CC_SPACE ||
                              cclass_[static_cast<uint8_t>(text[ws_end])] == CC_NL))
        ws_end++;

      if (ws_end >= len) return ws_end;
      if (ws_end - pos >= 2) return ws_end - 1;

      // Alt 6: \s+
      return ws_end;
    }

    return pos + blen;
  }

  // ---- Construction ----

  void parse_and_build(const json& j) {
    const auto& model = j.at("model");
    if (model.contains("type") && model["type"].is_string() &&
        model["type"].get<std::string>() != "BPE")
      throw std::runtime_error("Only BPE model is supported");

    const auto& vocab = model.at("vocab");
    token_to_id_.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it)
      token_to_id_.emplace(it.key(), it.value().get<int>());

    if (model.contains("byte_fallback") && model["byte_fallback"].is_boolean())
      byte_fallback_ = model["byte_fallback"].get<bool>();
    if (model.contains("ignore_merges") && model["ignore_merges"].is_boolean())
      ignore_merges_ = model["ignore_merges"].get<bool>();

    if (!model.contains("merges"))
      throw std::runtime_error("BPE merges missing");
    const auto& merges_json = model.at("merges");
    merges_.reserve(merges_json.size());
    for (const auto& m : merges_json) {
      std::string left, right;
      if (m.is_string()) {
        // Format 1: "token_a token_b"
        const std::string s = m.get<std::string>();
        auto sp = s.find(' ');
        if (sp == std::string::npos) continue;
        left = s.substr(0, sp);
        right = s.substr(sp + 1);
      } else if (m.is_array() && m.size() == 2) {
        // Format 2: ["token_a", "token_b"]
        left = m[0].get<std::string>();
        right = m[1].get<std::string>();
      } else {
        continue;
      }
      auto it_l = token_to_id_.find(left);
      auto it_r = token_to_id_.find(right);
      if (it_l != token_to_id_.end() && it_r != token_to_id_.end())
        merges_.emplace_back(it_l->second, it_r->second);
    }

    if (j.contains("pre_tokenizer")) {
      detect_pretokenizer_mode(j["pre_tokenizer"]);
    }

    if (j.contains("added_tokens")) {
      for (const auto& t : j["added_tokens"]) {
        std::string content = t.at("content").get<std::string>();
        int id = t.at("id").get<int>();
        bool is_special = t.value("special", false);

        added_tokens_.push_back({content, id, is_special});

        if (!token_to_id_.count(content))
          token_to_id_.emplace(content, id);

        if (!is_special) continue;
        special_by_prefix_[content.empty() ? '\0' : content[0]].emplace_back(content, id);
      }
      for (auto& [_, vec] : special_by_prefix_)
        std::sort(vec.begin(), vec.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    build_tables();
  }

  // Safe string getter: returns empty string if field missing or wrong type
  static std::string json_str(const json& obj, const std::string& key) {
    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) return "";
    return obj[key].get<std::string>();
  }

  // Auto-detect pre-tokenizer mode from tokenizer.json pre_tokenizer config
  void detect_pretokenizer_mode(const json& pt) {
    bool found_regex = false;  // track if we already matched a regex pattern

    auto check_node = [&](const json& node) {
      if (!node.is_object()) return;
      std::string type = json_str(node, "type");

      // ByteLevel pre-tokenizer
      if (type == "ByteLevel") {
        if (node.contains("add_prefix_space") && node["add_prefix_space"].is_boolean())
          config_.add_prefix_space = node["add_prefix_space"].get<bool>();
        if (node.contains("trim_offsets") && node["trim_offsets"].is_boolean())
          config_.trim_offsets = node["trim_offsets"].get<bool>();
        // ByteLevel with use_regex (default true) applies GPT-2 regex
        bool use_regex = true;
        if (node.contains("use_regex") && node["use_regex"].is_boolean())
          use_regex = node["use_regex"].get<bool>();
        if (use_regex && !found_regex)
          pt_mode_ = PT_GPT2;
      }

      // Metaspace pre-tokenizer (SentencePiece style)
      if (type == "Metaspace") {
        pt_mode_ = PT_METASPACE;
      }

      // Split pre-tokenizer
      if (type == "Split" && node.contains("pattern")) {
        const auto& pattern = node["pattern"];

        // Check for Split on literal string (e.g. Gemma splits on " ")
        if (pattern.is_object() && pattern.contains("String")) {
          std::string split_str = json_str(pattern, "String");
          if (split_str == " ") {
            pt_mode_ = PT_SPLIT_SPC;
            return;
          }
        }

        // Check for Split with Regex pattern
        std::string regex_str;
        if (pattern.is_object() && pattern.contains("Regex"))
          regex_str = json_str(pattern, "Regex");

        if (!regex_str.empty()) {
          found_regex = true;
          bool has_broad_prefix = regex_str.find("[^\\r\\n\\p{L}\\p{N}]") != std::string::npos;
          bool has_contractions_ci = regex_str.find("(?i:") != std::string::npos ||
                                    regex_str.find("(?i:'") != std::string::npos;

          if (has_broad_prefix) {
            if (has_contractions_ci)
              pt_mode_ = PT_CL100K;
            else
              pt_mode_ = PT_O200K;
          } else if (regex_str.find("\\p{L}") != std::string::npos) {
            pt_mode_ = PT_GPT2;
          }
        }
      }
    };

    if (pt.is_object()) {
      check_node(pt);
      if (json_str(pt, "type") == "Sequence" && pt.contains("pretokenizers") &&
          pt["pretokenizers"].is_array()) {
        for (const auto& sub : pt["pretokenizers"])
          check_node(sub);
      }
    }
    if (pt.is_array()) {
      for (const auto& n : pt) check_node(n);
    }
  }

  void build_tables() {
    memset(cclass_, CC_OTHER, sizeof(cclass_));
    for (int i = 'A'; i <= 'Z'; i++) cclass_[i] = CC_LETTER;
    for (int i = 'a'; i <= 'z'; i++) cclass_[i] = CC_LETTER;
    for (int i = '0'; i <= '9'; i++) cclass_[i] = CC_DIGIT;
    for (int i = 0x80; i <= 0xFF; i++) cclass_[i] = CC_HIGH;
    cclass_[static_cast<uint8_t>(' ')] = CC_SPACE;
    cclass_[static_cast<uint8_t>('\t')] = CC_SPACE;
    cclass_[static_cast<uint8_t>('\n')] = CC_NL;
    cclass_[static_cast<uint8_t>('\r')] = CC_NL;
    cclass_[static_cast<uint8_t>('\'')] = CC_APOS;

    int max_id = 0;
    for (auto& [tok, id] : token_to_id_)
      max_id = std::max(max_id, id);
    id_to_token_vec_.resize(max_id + 1);
    for (auto& [tok, id] : token_to_id_)
      id_to_token_vec_[id] = tok;

    memset(decode_valid_, 0, sizeof(decode_valid_));

    if (byte_fallback_) {
      // SentencePiece style: byte tokens are <0x00> through <0xFF>
      for (int b = 0; b < 256; b++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "<0x%02X>", b);
        auto it = token_to_id_.find(std::string(hex));
        byte_initial_id_[b] = (it != token_to_id_.end()) ? it->second : -1;
      }
    } else {
      // GPT-2 byte-level: bytes mapped via bytes_to_unicode()
      uint32_t byte2unicode[256];
      std::unordered_set<int> visible;
      for (int i = 0; i < 256; i++)
        if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
          visible.insert(i);

      int n = 0;
      for (int b = 0; b < 256; b++) {
        uint32_t cp = visible.count(b) ? b : 256 + n++;
        byte2unicode[b] = cp;
        if (cp < 512) {
          decode_byte_[cp] = static_cast<uint8_t>(b);
          decode_valid_[cp] = true;
        }
      }

      for (int b = 0; b < 256; b++) {
        std::string utf8 = codepoint_to_utf8(byte2unicode[b]);
        auto it = token_to_id_.find(utf8);
        byte_initial_id_[b] = (it != token_to_id_.end()) ? it->second : -1;
      }
    }

    merge_map_.build(merges_.size());
    for (size_t i = 0; i < merges_.size(); i++) {
      auto [a, b] = merges_[i];
      if (a >= 0 && a <= max_id && b >= 0 && b <= max_id) {
        std::string merged = id_to_token_vec_[a] + id_to_token_vec_[b];
        auto it = token_to_id_.find(merged);
        if (it != token_to_id_.end())
          merge_map_.insert(pack_pair(a, b), {static_cast<int>(i), it->second});
      }
    }

    slot_cache_.resize(CACHE_SLOTS);
    memset(slot_cache_.data(), 0, CACHE_SLOTS * sizeof(CacheSlot));

    build_decode_pool();
  }

  // Try to parse <0xHH> byte token, returns -1 if not a byte token
  static int parse_byte_token(const std::string& tok) {
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
      int hi = 0, lo = 0;
      char c3 = tok[3], c4 = tok[4];
      if (c3 >= '0' && c3 <= '9') hi = c3 - '0';
      else if (c3 >= 'A' && c3 <= 'F') hi = c3 - 'A' + 10;
      else return -1;
      if (c4 >= '0' && c4 <= '9') lo = c4 - '0';
      else if (c4 >= 'A' && c4 <= 'F') lo = c4 - 'A' + 10;
      else return -1;
      return hi * 16 + lo;
    }
    return -1;
  }

  void build_decode_pool() {
    size_t vs = id_to_token_vec_.size();
    decode_pieces_.resize(vs, {0, 0});

    size_t pool_estimate = 0;
    for (size_t id = 0; id < vs; id++)
      pool_estimate += id_to_token_vec_[id].size();
    decode_pool_.reserve(pool_estimate);

    for (size_t id = 0; id < vs; id++) {
      const std::string& tok = id_to_token_vec_[id];
      uint32_t start = static_cast<uint32_t>(decode_pool_.size());

      if (byte_fallback_) {
        // SentencePiece style: token strings are raw UTF-8,
        // except <0xHH> tokens which decode to a single byte.
        int byte_val = parse_byte_token(tok);
        if (byte_val >= 0) {
          decode_pool_.push_back(static_cast<char>(byte_val));
        } else {
          // Token string IS the decoded output (raw UTF-8)
          decode_pool_.insert(decode_pool_.end(), tok.begin(), tok.end());
        }
      } else {
        // GPT-2 byte-level: decode Unicode codepoints through byte mapping
        size_t i = 0;
        while (i < tok.size()) {
          const unsigned char c0 = tok[i];
          uint32_t cp;
          size_t adv;
          if (c0 < 0x80) { cp = c0; adv = 1; }
          else if ((c0 >> 5) == 0x6) { cp = ((c0 & 0x1F) << 6) | (tok[i+1] & 0x3F); adv = 2; }
          else if ((c0 >> 4) == 0xE) { cp = ((c0 & 0x0F) << 12) | ((tok[i+1] & 0x3F) << 6) | (tok[i+2] & 0x3F); adv = 3; }
          else { cp = ((c0 & 0x07) << 18) | ((tok[i+1] & 0x3F) << 12) | ((tok[i+2] & 0x3F) << 6) | (tok[i+3] & 0x3F); adv = 4; }
          i += adv;

          if (cp < 512 && decode_valid_[cp])
            decode_pool_.push_back(static_cast<char>(decode_byte_[cp]));
        }
      }

      uint16_t len = static_cast<uint16_t>(decode_pool_.size() - start);
      decode_pieces_[id] = {start, len};
    }
  }

  static inline std::string codepoint_to_utf8(uint32_t cp) {
    if (cp <= 0x7F) return std::string(1, static_cast<char>(cp));
    if (cp <= 0x7FF) return std::string{
      static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
    if (cp <= 0xFFFF) return std::string{
      static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)),
      static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
    return std::string{
      static_cast<char>(0xF0 | ((cp >> 18) & 0x07)),
      static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
      static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
  }
};
