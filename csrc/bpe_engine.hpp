#pragma once
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
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
    PT_SPLIT_SPC  = 5,  // Split on space, merge ▁ with next (SentencePiece style)
    PT_REGEX      = 6,  // Compiled regex pre-tokenizer
    PT_SPLIT_MERGE_PREV = 7,  // Split on space, merge space with previous chunk (Gemma)
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
  bool get_metaspace_add_prefix() const { return metaspace_add_prefix_; }
  void set_cache_enabled(bool enabled) { cache_enabled_ = enabled; }
  bool cache_enabled() const { return cache_enabled_; }
  void clear_cache() const {
    if (!slot_cache_.empty())
      memset(slot_cache_.data(), 0, CACHE_SLOTS * sizeof(CacheSlot));
  }

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

  // ---- Decode: two-pass exact-alloc with prefetch ----

  std::string decode(const std::vector<int>& ids) const {
    const int n = static_cast<int>(ids.size());
    if (__builtin_expect(!n, 0)) return {};

    const unsigned pc = static_cast<unsigned>(decode_pieces_.size());
    const DecodePiece* __restrict__ pieces = decode_pieces_.data();
    const char* __restrict__ pool = decode_pool_.data();
    const int* __restrict__ src = ids.data();

    size_t total = 0;
    for (int i = 0; i < n; i++) {
      unsigned id = static_cast<unsigned>(src[i]);
      if (__builtin_expect(id < pc, 1))
        total += pieces[id].len;
    }

    std::string out;
#if defined(_LIBCPP_VERSION)
    out.__resize_default_init(total);
#else
    out.resize(total);
#endif
    char* __restrict__ dst = out.data();

    if (__builtin_expect(n >= 16, 0)) {
      for (int i = 0; i < n; i++) {
        unsigned id = static_cast<unsigned>(src[i]);
        if (__builtin_expect(id < pc, 1)) {
          if (i + 4 < n) {
            unsigned pfid = static_cast<unsigned>(src[i + 4]);
            if (pfid < pc)
              __builtin_prefetch(pool + pieces[pfid].offset, 0, 1);
          }
          uint32_t off = pieces[id].offset;
          uint16_t plen = pieces[id].len;
          memcpy(dst, pool + off, plen);
          dst += plen;
        }
      }
    } else {
      for (int i = 0; i < n; i++) {
        unsigned id = static_cast<unsigned>(src[i]);
        if (__builtin_expect(id < pc, 1)) {
          uint32_t off = pieces[id].offset;
          uint16_t plen = pieces[id].len;
          memcpy(dst, pool + off, plen);
          dst += plen;
        }
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
    if (pt_mode_ == PT_REGEX) {
      std::vector<SplitSpan> splits;
      split_with_regex(data, len, 0, len, splits);
      for (const auto& s : splits) {
        if (s.remove) continue;
        if (s.end <= s.start) continue;
        chunks.push_back(text.substr(s.start, s.end - s.start));
      }
      return chunks;
    }
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
  std::string metaspace_replacement_ = "\xe2\x96\x81";
  bool metaspace_add_prefix_ = true;
  bool metaspace_split_ = true;
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_vec_;
  std::vector<std::pair<int,int>> merges_;
  std::vector<AddedToken> added_tokens_;

  int byte_initial_id_[256];

  enum CClass : uint8_t { CC_OTHER=0, CC_LETTER=1, CC_DIGIT=2, CC_SPACE=3, CC_NL=4, CC_APOS=5, CC_HIGH=6 };
  uint8_t cclass_[256];

  // Unicode category tables (generated from UCD)
  #include "unicode_tables.inc"

  // Unicode White_Space and Separator (Z) ranges (compact list)
  static constexpr uint32_t unicode_whitespace_ranges[][2] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0},
    {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
    {0x205F, 0x205F}, {0x3000, 0x3000},
  };
  static constexpr size_t unicode_whitespace_ranges_count =
      sizeof(unicode_whitespace_ranges) / sizeof(unicode_whitespace_ranges[0]);

  static constexpr uint32_t unicode_separator_ranges[][2] = {
    {0x0020, 0x0020}, {0x00A0, 0x00A0}, {0x1680, 0x1680}, {0x2000, 0x200A},
    {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
  };
  static constexpr size_t unicode_separator_ranges_count =
      sizeof(unicode_separator_ranges) / sizeof(unicode_separator_ranges[0]);

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

  static bool is_unicode_mark(uint32_t cp) {
    if (cp < 0x0300) return false;
    if (cp <= 0x036F) return true;  // Combining Diacritical Marks
    if (cp >= 0x0591 && cp <= 0x05BD) return true;  // Hebrew marks
    if (cp >= 0x0610 && cp <= 0x061A) return true;  // Arabic marks
    if (cp >= 0x064B && cp <= 0x065F) return true;
    if (cp >= 0x0900 && cp <= 0x0903) return true;  // Devanagari marks
    if (cp >= 0x093A && cp <= 0x094F) return true;
    if (cp >= 0x0980 && cp <= 0x0983) return true;
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;  // Combining marks for symbols
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;
    return false;
  }

  static bool is_unicode_whitespace(uint32_t cp) {
    if (cp < 0x80) return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
    return in_ranges(cp, unicode_whitespace_ranges, unicode_whitespace_ranges_count);
  }

  static bool is_unicode_separator(uint32_t cp) {
    if (cp < 0x80) return cp == ' ';
    return in_ranges(cp, unicode_separator_ranges, unicode_separator_ranges_count);
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

  enum class SplitBehavior : uint8_t {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
    Unknown,
  };

  // ---- Regex compiler (subset) ----
  struct RegexProgram {
    enum NodeType : uint8_t { LITERAL, CLASS, CONCAT, ALT, REPEAT, ANCHOR_START, ANCHOR_END, EMPTY };

    struct Range { uint32_t lo; uint32_t hi; };
    struct CharClass {
      std::vector<Range> ranges;
      bool negated = false;
      bool cat_letter = false;
      bool cat_number = false;
      bool cat_space = false;
      bool cat_sep = false;
      bool cat_mark = false;

      void add_literal(uint32_t cp) { ranges.push_back({cp, cp}); }

      void merge_from(const CharClass& other) {
        ranges.insert(ranges.end(), other.ranges.begin(), other.ranges.end());
        cat_letter = cat_letter || other.cat_letter;
        cat_number = cat_number || other.cat_number;
        cat_space = cat_space || other.cat_space;
        cat_sep = cat_sep || other.cat_sep;
        cat_mark = cat_mark || other.cat_mark;
        if (other.negated) negated = true;
      }

      bool matches(uint32_t cp, bool ci) const {
        bool in = false;
        if (cat_letter && is_unicode_letter(cp)) in = true;
        if (cat_number && is_unicode_digit(cp)) in = true;
        if (cat_space && is_unicode_whitespace(cp)) in = true;
        if (cat_sep && is_unicode_separator(cp)) in = true;
        if (cat_mark && is_unicode_mark(cp)) in = true;
        for (const auto& r : ranges) {
          if (cp >= r.lo && cp <= r.hi) { in = true; break; }
        }
        if (negated) in = !in;
        if (!ci) return in;
        if (cp < 0x80) {
          uint32_t lo = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(cp)));
          uint32_t hi = static_cast<uint32_t>(std::toupper(static_cast<unsigned char>(cp)));
          bool in2 = false;
          if (cat_letter && (is_unicode_letter(lo) || is_unicode_letter(hi))) in2 = true;
          if (cat_number && (is_unicode_digit(lo) || is_unicode_digit(hi))) in2 = true;
          if (cat_space && (is_unicode_whitespace(lo) || is_unicode_whitespace(hi))) in2 = true;
          if (cat_sep && (is_unicode_separator(lo) || is_unicode_separator(hi))) in2 = true;
          if (cat_mark && (is_unicode_mark(lo) || is_unicode_mark(hi))) in2 = true;
          for (const auto& r : ranges) {
            if ((lo >= r.lo && lo <= r.hi) || (hi >= r.lo && hi <= r.hi)) { in2 = true; break; }
          }
          if (negated) in2 = !in2;
          return in || in2;
        }
        return in;
      }
    };

    struct Node {
      NodeType type = LITERAL;
      bool ci = false;
      uint32_t literal = 0;
      CharClass cls;
      std::vector<int> children;  // CONCAT/ALT
      int child = -1;             // REPEAT
      int min = 0;
      int max = 0;                // -1 = infinity
    };

    struct MatchSpan { size_t start; size_t end; bool is_match; };

    enum Op : uint8_t { OP_CHAR, OP_CLASS, OP_SPLIT, OP_JMP, OP_MATCH, OP_ASSERT_START, OP_ASSERT_END, OP_ANY };
    struct Inst {
      Op op = OP_MATCH;
      int out1 = -1;
      int out2 = -1;
      uint32_t c = 0;
      CharClass cls;
      bool ci = false;
    };

    std::vector<Node> nodes;
    std::vector<Inst> insts;
    int root = -1;
    int start = -1;
    std::string error;
    bool can_match_empty = false;
    bool start_any = false;
    bool start_class_valid = false;
    CharClass start_class;

    mutable std::vector<int> nfa_curr_;
    mutable std::vector<int> nfa_next_;
    mutable std::vector<int> nfa_seen_;
    mutable std::vector<int> nfa_seen2_;
    mutable std::vector<MatchSpan> nfa_matches_;

    int new_node(NodeType t) {
      nodes.push_back(Node{});
      nodes.back().type = t;
      return static_cast<int>(nodes.size() - 1);
    }

    int add_inst(Op op) {
      insts.push_back(Inst{});
      insts.back().op = op;
      return static_cast<int>(insts.size() - 1);
    }

    struct Patch { int state; bool out2; };
    struct Frag { int start; std::vector<Patch> out; };

    static bool is_hex(char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }

    static int hex_val(char c) {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
      if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
      return -1;
    }

    struct Parser {
      const std::string& s;
      size_t pos = 0;
      RegexProgram& prog;
      bool ci = false;

      Parser(const std::string& str, RegexProgram& p) : s(str), prog(p) {}

      bool eof() const { return pos >= s.size(); }
      char peek() const { return eof() ? '\0' : s[pos]; }
      char get() { return eof() ? '\0' : s[pos++]; }

      bool consume(char c) {
        if (peek() == c) { pos++; return true; }
        return false;
      }

      int parse_regex() { return parse_alt(); }

      int parse_alt() {
        std::vector<int> alts;
        alts.push_back(parse_concat());
        while (consume('|')) {
          alts.push_back(parse_concat());
        }
        if (alts.size() == 1) return alts[0];
        int node = prog.new_node(ALT);
        prog.nodes[node].children = std::move(alts);
        return node;
      }

      int parse_concat() {
        std::vector<int> seq;
        while (!eof()) {
          char c = peek();
          if (c == '|' || c == ')') break;
          int atom = parse_repeat();
          if (atom < 0) return atom;
          seq.push_back(atom);
        }
        if (seq.empty()) {
          int node = prog.new_node(EMPTY);
          return node;
        }
        if (seq.size() == 1) return seq[0];
        int node = prog.new_node(CONCAT);
        prog.nodes[node].children = std::move(seq);
        return node;
      }

      int parse_repeat() {
        int atom = parse_atom();
        if (atom < 0) return atom;
        if (eof()) return atom;

        int min = -1, max = -1;
        if (consume('*')) { min = 0; max = -1; }
        else if (consume('+')) { min = 1; max = -1; }
        else if (consume('?')) { min = 0; max = 1; }
        else if (consume('{')) {
          int m = parse_number();
          if (m < 0) { prog.error = "invalid quantifier"; return -1; }
          int n = m;
          if (consume(',')) {
            if (peek() == '}') n = -1;
            else {
              n = parse_number();
              if (n < 0) { prog.error = "invalid quantifier"; return -1; }
            }
          }
          if (!consume('}')) { prog.error = "unterminated quantifier"; return -1; }
          min = m; max = n;
        } else {
          return atom;
        }

        if (!eof() && peek() == '?') {
          prog.error = "lazy quantifier not supported";
          return -1;
        }

        int node = prog.new_node(REPEAT);
        prog.nodes[node].child = atom;
        prog.nodes[node].min = min;
        prog.nodes[node].max = max;
        return node;
      }

      int parse_atom() {
        if (eof()) { prog.error = "unexpected end"; return -1; }
        char c = get();
        if (c == '(') {
          bool prev_ci = ci;
          if (consume('?')) {
            if (consume(':')) {
            } else if (consume('i')) {
              if (!consume(':')) { prog.error = "unsupported group flag"; return -1; }
              ci = true;
            } else {
              prog.error = "unsupported group construct";
              return -1;
            }
          }
          int sub = parse_alt();
          if (sub < 0) return sub;
          if (!consume(')')) { prog.error = "unterminated group"; return -1; }
          ci = prev_ci;
          return sub;
        }
        if (c == '[') return parse_class();
        if (c == '^') return prog.new_node(ANCHOR_START);
        if (c == '$') return prog.new_node(ANCHOR_END);
        if (c == '.') { prog.error = "unsupported '.'"; return -1; }
        if (c == '\\') return parse_escape_atom();
        if (c == '*' || c == '+' || c == '?' || c == '{' || c == '}' || c == '|') {
          prog.error = "unexpected quantifier or alternation";
          return -1;
        }

        int node = prog.new_node(LITERAL);
        prog.nodes[node].literal = static_cast<uint8_t>(c);
        prog.nodes[node].ci = ci;
        return node;
      }

      int parse_escape_atom() {
        if (eof()) { prog.error = "dangling escape"; return -1; }
        char c = get();
        if (c == 'p') {
          if (!consume('{')) { prog.error = "invalid \\p"; return -1; }
          std::string cat;
          while (!eof() && peek() != '}') cat.push_back(get());
          if (!consume('}')) { prog.error = "unterminated \\p"; return -1; }
          int node = prog.new_node(CLASS);
          prog.nodes[node].ci = ci;
          if (cat == "L" || cat == "Lu" || cat == "Lt" || cat == "Lm" || cat == "Lo") prog.nodes[node].cls.cat_letter = true;
          else if (cat == "N") prog.nodes[node].cls.cat_number = true;
          else if (cat == "Z") prog.nodes[node].cls.cat_sep = true;
          else if (cat == "M") prog.nodes[node].cls.cat_mark = true;
          else { prog.error = "unsupported \\p category"; return -1; }
          return node;
        }
        if (c == 's') {
          int node = prog.new_node(CLASS);
          prog.nodes[node].ci = ci;
          prog.nodes[node].cls.cat_space = true;
          return node;
        }
        if (c == 'S') {
          int node = prog.new_node(CLASS);
          prog.nodes[node].ci = ci;
          prog.nodes[node].cls.cat_space = true;
          prog.nodes[node].cls.negated = true;
          return node;
        }
        uint32_t cp = 0;
        if (c == 'n') cp = '\n';
        else if (c == 't') cp = '\t';
        else if (c == 'r') cp = '\r';
        else if (c == '\\') cp = '\\';
        else if (c == 'x') {
          if (pos + 1 >= s.size() || !is_hex(s[pos]) || !is_hex(s[pos+1])) {
            prog.error = "invalid \\x escape";
            return -1;
          }
          cp = (hex_val(s[pos]) << 4) | hex_val(s[pos+1]);
          pos += 2;
        } else if (c == 'u') {
          if (pos + 3 >= s.size()) { prog.error = "invalid \\u escape"; return -1; }
          cp = 0;
          for (int i = 0; i < 4; i++) {
            if (!is_hex(s[pos+i])) { prog.error = "invalid \\u escape"; return -1; }
            cp = (cp << 4) | hex_val(s[pos+i]);
          }
          pos += 4;
        } else {
          if (std::isalnum(static_cast<unsigned char>(c))) {
            prog.error = "unsupported escape";
            return -1;
          }
          cp = static_cast<uint8_t>(c);
        }
        int node = prog.new_node(LITERAL);
        prog.nodes[node].literal = cp;
        prog.nodes[node].ci = ci;
        return node;
      }

      int parse_class() {
        int node = prog.new_node(CLASS);
        prog.nodes[node].ci = ci;
        CharClass& cls = prog.nodes[node].cls;
        if (consume('^')) cls.negated = true;
        bool have_last = false;
        uint32_t last_cp = 0;
        bool any_content = false;
        while (!eof()) {
          if (peek() == ']') {
            get();
            if (have_last) cls.ranges.push_back({last_cp, last_cp});
            if (!any_content && !have_last && cls.ranges.empty() &&
                !cls.cat_letter && !cls.cat_number && !cls.cat_space && !cls.cat_sep && !cls.cat_mark) {
              prog.error = "empty char class";
              return -1;
            }
            return node;
          }
          uint32_t cp = 0;
          bool is_cat = false;
          bool is_space = false;
          bool is_sep = false;
          bool is_letter = false;
          bool is_number = false;
          bool is_mark = false;
          char c = get();
          if (c == '\\') {
            if (eof()) { prog.error = "dangling escape"; return -1; }
            char ec = get();
            if (ec == 'p') {
              if (!consume('{')) { prog.error = "invalid \\p"; return -1; }
              std::string cat;
              while (!eof() && peek() != '}') cat.push_back(get());
              if (!consume('}')) { prog.error = "unterminated \\p"; return -1; }
              if (cat == "L" || cat == "Lu" || cat == "Lt" || cat == "Lm" || cat == "Lo") is_letter = true;
              else if (cat == "N") is_number = true;
              else if (cat == "Z") is_sep = true;
              else if (cat == "M") is_mark = true;
              else { prog.error = "unsupported \\p category"; return -1; }
              is_cat = true;
            } else if (ec == 's') {
              is_space = true;
              is_cat = true;
            } else if (ec == 'n') cp = '\n';
            else if (ec == 't') cp = '\t';
            else if (ec == 'r') cp = '\r';
            else if (ec == 'x') {
              if (pos + 1 >= s.size() || !is_hex(s[pos]) || !is_hex(s[pos+1])) {
                prog.error = "invalid \\x escape";
                return -1;
              }
              cp = (hex_val(s[pos]) << 4) | hex_val(s[pos+1]);
              pos += 2;
            } else if (ec == 'u') {
              if (pos + 3 >= s.size()) { prog.error = "invalid \\u escape"; return -1; }
              cp = 0;
              for (int i = 0; i < 4; i++) {
                if (!is_hex(s[pos+i])) { prog.error = "invalid \\u escape"; return -1; }
                cp = (cp << 4) | hex_val(s[pos+i]);
              }
              pos += 4;
            } else {
              if (std::isalnum(static_cast<unsigned char>(ec))) {
                prog.error = "unsupported escape";
                return -1;
              }
              cp = static_cast<uint8_t>(ec);
            }
          } else {
            cp = static_cast<uint8_t>(c);
          }

          if (is_cat) {
            if (have_last) { cls.ranges.push_back({last_cp, last_cp}); have_last = false; }
            if (is_letter) cls.cat_letter = true;
            if (is_number) cls.cat_number = true;
            if (is_space) cls.cat_space = true;
            if (is_sep) cls.cat_sep = true;
            if (is_mark) cls.cat_mark = true;
            any_content = true;
            continue;
          }
          any_content = true;

          if (peek() == '-' && pos + 1 < s.size() && s[pos + 1] != ']') {
            get();
            uint32_t cp2 = 0;
            char c2 = get();
            if (c2 == '\\') {
              if (eof()) { prog.error = "dangling escape"; return -1; }
              char ec2 = get();
              if (ec2 == 'n') cp2 = '\n';
              else if (ec2 == 't') cp2 = '\t';
              else if (ec2 == 'r') cp2 = '\r';
              else if (ec2 == 'x') {
                if (pos + 1 >= s.size() || !is_hex(s[pos]) || !is_hex(s[pos+1])) {
                  prog.error = "invalid \\x escape";
                  return -1;
                }
                cp2 = (hex_val(s[pos]) << 4) | hex_val(s[pos+1]);
                pos += 2;
              } else if (ec2 == 'u') {
                if (pos + 3 >= s.size()) { prog.error = "invalid \\u escape"; return -1; }
                cp2 = 0;
                for (int i = 0; i < 4; i++) {
                  if (!is_hex(s[pos+i])) { prog.error = "invalid \\u escape"; return -1; }
                  cp2 = (cp2 << 4) | hex_val(s[pos+i]);
                }
                pos += 4;
              } else {
                if (std::isalnum(static_cast<unsigned char>(ec2))) {
                  prog.error = "unsupported escape";
                  return -1;
                }
                cp2 = static_cast<uint8_t>(ec2);
              }
            } else {
              cp2 = static_cast<uint8_t>(c2);
            }
            if (have_last) {
              cls.ranges.push_back({last_cp, last_cp});
              have_last = false;
            }
            if (cp2 < cp) std::swap(cp, cp2);
            cls.ranges.push_back({cp, cp2});
          } else {
            if (have_last) cls.ranges.push_back({last_cp, last_cp});
            last_cp = cp;
            have_last = true;
          }
        }
        prog.error = "unterminated char class";
        return -1;
      }

      int parse_number() {
        int v = 0;
        bool any = false;
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
          any = true;
          v = v * 10 + (get() - '0');
        }
        return any ? v : -1;
      }
    };

    static void patch(std::vector<Inst>& insts, const std::vector<Patch>& out, int target) {
      for (const auto& p : out) {
        if (p.out2) insts[p.state].out2 = target;
        else insts[p.state].out1 = target;
      }
    }

    static std::vector<Patch> append(const std::vector<Patch>& a, const std::vector<Patch>& b) {
      std::vector<Patch> out;
      out.reserve(a.size() + b.size());
      out.insert(out.end(), a.begin(), a.end());
      out.insert(out.end(), b.begin(), b.end());
      return out;
    }

    Frag empty_frag() {
      int j = add_inst(OP_JMP);
      return Frag{j, {Patch{j, false}}};
    }

    Frag literal_frag(uint32_t cp, bool ci) {
      int s = add_inst(OP_CHAR);
      insts[s].c = cp;
      insts[s].ci = ci;
      return Frag{s, {Patch{s, false}}};
    }

    Frag class_frag(const CharClass& cls, bool ci) {
      int s = add_inst(OP_CLASS);
      insts[s].cls = cls;
      insts[s].ci = ci;
      return Frag{s, {Patch{s, false}}};
    }

    Frag assert_frag(Op op) {
      int s = add_inst(op);
      return Frag{s, {Patch{s, false}}};
    }

    Frag concat_frag(const Frag& a, const Frag& b) {
      patch(insts, a.out, b.start);
      return Frag{a.start, b.out};
    }

    Frag alt_frag(const Frag& a, const Frag& b) {
      int s = add_inst(OP_SPLIT);
      insts[s].out1 = a.start;
      insts[s].out2 = b.start;
      return Frag{s, append(a.out, b.out)};
    }

    Frag repeat_frag(int child_idx, int min, int max) {
      Frag frag = empty_frag();
      if (min > 0) {
        frag = compile_node(child_idx);
        for (int i = 1; i < min; i++) {
          frag = concat_frag(frag, compile_node(child_idx));
        }
      }
      if (max == min) return frag;

      if (max < 0) {
        Frag rep = compile_node(child_idx);
        int s = add_inst(OP_SPLIT);
        insts[s].out1 = rep.start;
        patch(insts, rep.out, s);
        std::vector<Patch> out = {Patch{s, true}};
        Frag loop{ s, out };
        return (min > 0) ? concat_frag(frag, loop) : loop;
      }

      for (int i = 0; i < max - min; i++) {
        Frag opt = compile_node(child_idx);
        int s = add_inst(OP_SPLIT);
        insts[s].out1 = opt.start;
        std::vector<Patch> out = append(opt.out, {Patch{s, true}});
        Frag optfrag{s, out};
        if (min > 0 || frag.start != -1) frag = concat_frag(frag, optfrag);
        else frag = optfrag;
      }
      return frag;
    }

    Frag compile_node(int idx) {
      const Node& n = nodes[idx];
      switch (n.type) {
        case EMPTY:
          return empty_frag();
        case LITERAL:
          return literal_frag(n.literal, n.ci);
        case CLASS:
          return class_frag(n.cls, n.ci);
        case ANCHOR_START:
          return assert_frag(OP_ASSERT_START);
        case ANCHOR_END:
          return assert_frag(OP_ASSERT_END);
        case CONCAT: {
          Frag frag = compile_node(n.children[0]);
          for (size_t i = 1; i < n.children.size(); i++)
            frag = concat_frag(frag, compile_node(n.children[i]));
          return frag;
        }
        case ALT: {
          Frag frag = compile_node(n.children[0]);
          for (size_t i = 1; i < n.children.size(); i++)
            frag = alt_frag(frag, compile_node(n.children[i]));
          return frag;
        }
        case REPEAT:
          return repeat_frag(n.child, n.min, n.max);
      }
      return empty_frag();
    }

    bool nullable(int idx) const {
      const Node& n = nodes[idx];
      switch (n.type) {
        case EMPTY: return true;
        case LITERAL:
        case CLASS:
        case ANCHOR_START:
        case ANCHOR_END:
          return n.type == ANCHOR_START || n.type == ANCHOR_END;
        case CONCAT: {
          for (int child : n.children) if (!nullable(child)) return false;
          return true;
        }
        case ALT:
          for (int child : n.children) if (nullable(child)) return true;
          return false;
        case REPEAT:
          return (n.min == 0) || nullable(n.child);
      }
      return false;
    }

    void firstset(int idx, CharClass& out, bool& any) const {
      if (any) return;
      const Node& n = nodes[idx];
      switch (n.type) {
        case EMPTY:
        case ANCHOR_START:
        case ANCHOR_END:
          return;
        case LITERAL: {
          if (n.ci && n.literal >= 0x80) { any = true; return; }
          uint32_t cp = n.literal;
          if (n.ci && cp < 0x80) {
            out.add_literal(static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(cp))));
            out.add_literal(static_cast<uint32_t>(std::toupper(static_cast<unsigned char>(cp))));
          } else {
            out.add_literal(cp);
          }
          return;
        }
        case CLASS:
          if (n.ci || n.cls.negated) { any = true; return; }
          out.merge_from(n.cls);
          return;
        case ALT:
          for (int child : n.children) firstset(child, out, any);
          return;
        case CONCAT: {
          for (int child : n.children) {
            firstset(child, out, any);
            if (!nullable(child)) break;
          }
          return;
        }
        case REPEAT:
          firstset(n.child, out, any);
          return;
      }
    }

    bool compile(const std::string& pattern) {
      nodes.clear();
      insts.clear();
      error.clear();
      root = -1;
      start = -1;
      can_match_empty = false;
      start_any = false;
      start_class_valid = false;
      start_class = CharClass{};

      Parser p(pattern, *this);
      int root_idx = p.parse_regex();
      if (root_idx < 0 || !error.empty()) return false;
      if (!p.eof()) { error = "trailing pattern"; return false; }
      root = root_idx;

      can_match_empty = nullable(root);
      bool any = false;
      CharClass fs;
      firstset(root, fs, any);
      start_any = any;
      start_class_valid = !any;
      if (!any) start_class = fs;

      Frag frag = compile_node(root);
      int m = add_inst(OP_MATCH);
      patch(insts, frag.out, m);
      start = frag.start;

      size_t n = insts.size();
      nfa_curr_.reserve(n);
      nfa_next_.reserve(n);
      nfa_seen_.resize(n, 0);
      nfa_seen2_.resize(n, 0);

      return true;
    }

    void add_state(std::vector<int>& list, std::vector<int>& seen, int state, size_t pos, size_t full_len) const {
      if (state < 0) return;
      if (seen[state]) return;
      seen[state] = 1;
      const Inst& inst = insts[state];
      switch (inst.op) {
        case OP_SPLIT:
          add_state(list, seen, inst.out1, pos, full_len);
          add_state(list, seen, inst.out2, pos, full_len);
          return;
        case OP_JMP:
          add_state(list, seen, inst.out1, pos, full_len);
          return;
        case OP_ASSERT_START:
          if (pos == 0) add_state(list, seen, inst.out1, pos, full_len);
          return;
        case OP_ASSERT_END:
          if (pos == full_len) add_state(list, seen, inst.out1, pos, full_len);
          return;
        default:
          list.push_back(state);
          return;
      }
    }

    bool match_at(const char* text, size_t full_len, size_t limit_end, size_t pos, size_t& end) const {
      if (start < 0) return false;
      const size_t n = insts.size();

      nfa_curr_.clear();
      nfa_next_.clear();
      memset(nfa_seen_.data(), 0, n * sizeof(int));
      memset(nfa_seen2_.data(), 0, n * sizeof(int));

      add_state(nfa_curr_, nfa_seen_, start, pos, full_len);
      size_t best_end = static_cast<size_t>(-1);
      size_t p = pos;
      while (true) {
        for (int s : nfa_curr_) {
          if (insts[s].op == OP_MATCH) best_end = p;
        }
        if (p >= limit_end) break;
        uint32_t cp;
        size_t adv = utf8_decode(text, full_len, p, cp);
        if (adv == 0) break;
        nfa_next_.clear();
        memset(nfa_seen2_.data(), 0, n * sizeof(int));
        for (int s : nfa_curr_) {
          const Inst& inst = insts[s];
          switch (inst.op) {
            case OP_CHAR: {
              uint32_t lit = inst.c;
              uint32_t cpc = cp;
              if (inst.ci && cpc < 0x80) {
                cpc = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(cpc)));
                lit = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(lit)));
              }
              if (cpc == lit) add_state(nfa_next_, nfa_seen2_, inst.out1, p + adv, full_len);
              break;
            }
            case OP_CLASS:
              if (inst.cls.matches(cp, inst.ci)) add_state(nfa_next_, nfa_seen2_, inst.out1, p + adv, full_len);
              break;
            case OP_ANY:
              add_state(nfa_next_, nfa_seen2_, inst.out1, p + adv, full_len);
              break;
            default:
              break;
          }
        }
        nfa_curr_.swap(nfa_next_);
        p += adv;
        if (nfa_curr_.empty()) break;
      }
      if (best_end != static_cast<size_t>(-1)) {
        end = best_end;
        return true;
      }
      return false;
    }

    bool can_start_at(const char* text, size_t full_len, size_t limit_end, size_t pos) const {
      if (start_any) return true;
      if (!start_class_valid) return true;
      if (pos >= limit_end) return can_match_empty;
      uint32_t cp;
      size_t adv = utf8_decode(text, full_len, pos, cp);
      if (adv == 0) return true;
      return start_class.matches(cp, false);
    }

    bool find_next(const char* text, size_t full_len, size_t limit_end, size_t start_pos,
                   size_t& mstart, size_t& mend) const {
      size_t pos = start_pos;
      while (pos <= limit_end) {
        if (can_start_at(text, full_len, limit_end, pos) || can_match_empty) {
          size_t end;
          if (match_at(text, full_len, limit_end, pos, end)) {
            mstart = pos;
            mend = end;
            return true;
          }
        }
        if (pos >= limit_end) break;
        uint32_t cp;
        size_t adv = utf8_decode(text, full_len, pos, cp);
        pos += (adv ? adv : 1);
      }
      return false;
    }

    void find_matches(const char* text, size_t full_len, size_t start_pos,
                      size_t limit_end, std::vector<MatchSpan>& out) const {
      out.clear();
      size_t pos = start_pos;
      while (pos <= limit_end) {
        size_t mstart = 0, mend = 0;
        if (!find_next(text, full_len, limit_end, pos, mstart, mend)) {
          out.push_back({pos, limit_end, false});
          break;
        }
        out.push_back({pos, mstart, false});
        out.push_back({mstart, mend, true});
        if (mend == mstart) {
          if (mstart >= limit_end) {
            out.push_back({limit_end, limit_end, false});
            break;
          } else {
            uint32_t cp;
            size_t adv = utf8_decode(text, full_len, mstart, cp);
            pos = mstart + (adv ? adv : 1);
          }
        } else {
          pos = mend;
        }
      }
    }
  };

  struct CompiledSplit {
    mutable RegexProgram prog;
    SplitBehavior behavior = SplitBehavior::Unknown;
    bool invert = false;
  };
  mutable std::vector<CompiledSplit> regex_splits_;
  bool regex_enabled_ = false;
  bool pretok_strict_ = true;
  bool regex_compile_failed_ = false;
  std::string regex_error_;

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
    uint64_t hash_lo;
    uint64_t hash_hi;
    uint16_t data_len;
    uint16_t token_count;
    int tokens[11];
  };
  static_assert(sizeof(CacheSlot) == 64, "CacheSlot must be 64 bytes");
  static constexpr size_t CACHE_SLOTS = 1 << 14;
  static constexpr size_t CACHE_MASK = CACHE_SLOTS - 1;
  static constexpr int MAX_CACHED_TOKENS = 11;
  static constexpr int BPE_HEAP_THRESHOLD = 64;
  mutable std::vector<CacheSlot> slot_cache_;
  bool cache_enabled_ = true;

  struct DecodePiece { uint32_t offset; uint16_t len; };
  std::vector<DecodePiece> decode_pieces_;
  std::vector<char> decode_pool_;

  uint8_t decode_byte_[512];
  bool decode_valid_[512];

  static inline uint64_t pack_pair(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
  }

  struct Hash128 { uint64_t lo; uint64_t hi; };
  static inline Hash128 chunk_hash(const char* data, size_t len) {
    uint64_t h1 = 0xcbf29ce484222325ULL;
    uint64_t h2 = 0x84222325cbf29ce4ULL;
    for (size_t i = 0; i < len; i++) {
      uint64_t v = static_cast<uint8_t>(data[i]);
      h1 = (h1 ^ v) * 0x100000001b3ULL;
      h2 = (h2 ^ v) * 0x9E3779B97F4A7C15ULL;
    }
    h1 ^= (h1 >> 33);
    h2 ^= (h2 >> 29);
    return {h1, h2};
  }

  // ---- Encode core ----

  void encode_into(const std::string& text,
                   const std::unordered_set<std::string>& allowed_special,
                   std::vector<int>& out) const {
    // SentencePiece-style: replace spaces with ▁, encode as single chunk
    if (pt_mode_ == PT_METASPACE) {
      encode_spiece(text, allowed_special, out, /*prepend_underscore=*/metaspace_add_prefix_,
                    /*split=*/metaspace_split_);
      return;
    }
    if (pt_mode_ == PT_SPLIT_SPC) {
      encode_spiece(text, allowed_special, out, /*prepend_underscore=*/false,
                    /*split=*/true);
      return;
    }
    if (pt_mode_ == PT_SPLIT_MERGE_PREV) {
      encode_split_merge_prev(text, out);
      return;
    }
    if (pt_mode_ == PT_REGEX) {
      encode_with_regex_pretok(text, allowed_special, out);
      return;
    }

    size_t pos = 0;
    const size_t len = text.size();
    const char* data = text.data();

    while (pos < len) {
      if (__builtin_expect(!special_by_prefix_.empty() && !allowed_special.empty(), 0)) {
        auto it = special_by_prefix_.find(data[pos]);
        if (it != special_by_prefix_.end()) {
          size_t matched = 0;
          int matched_id = -1;
          for (const auto& [tok, id] : it->second) {
            if (!allowed_special.count(tok)) continue;
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
                     bool prepend_underscore,
                     bool split) const {
    (void)allowed_special;
    const std::string& SP = metaspace_replacement_;
    auto is_ws = [](char c) { return c == ' '; };

    if (!split) {
      std::string transformed;
      transformed.reserve(text.size() + text.size() / 4 + 3);
      for (char c : text) {
        if (is_ws(c))
          transformed += SP;
        else
          transformed.push_back(c);
      }
      if (prepend_underscore && !text.empty() && !is_ws(text[0]))
        transformed = SP + transformed;

      encode_chunk_sp(transformed.data(), transformed.size(), out);
      return;
    }

    int pending = 0;
    if (prepend_underscore && !text.empty() && !is_ws(text[0]))
      pending = 1;

    std::string chunk;
    chunk.reserve(64);
    bool building = false;

    for (char c : text) {
      if (is_ws(c)) {
        if (building) {
          encode_chunk_sp(chunk.data(), chunk.size(), out);
          chunk.clear();
          building = false;
        }
        pending++;
      } else {
        if (!building) {
          if (pending > 1) {
            for (int i = 0; i < pending - 1; i++)
              encode_chunk_sp(SP.data(), SP.size(), out);
          }
          if (pending > 0)
            chunk.assign(SP);
          else
            chunk.clear();
          pending = 0;
          building = true;
        }
        chunk.push_back(c);
      }
    }

    if (building) {
      encode_chunk_sp(chunk.data(), chunk.size(), out);
    } else if (pending > 0) {
      for (int i = 0; i < pending; i++)
        encode_chunk_sp(SP.data(), SP.size(), out);
    }
  }

  // Split on space, merge space with PREVIOUS chunk (Gemma-style)
  // "a b c" -> chunks ["a ", "b ", "c"], with spaces replaced by ▁
  void encode_split_merge_prev(const std::string& text, std::vector<int>& out) const {
    const size_t len = text.size();
    if (!len) return;
    const char* data = text.data();
    const std::string& SP = metaspace_replacement_;

    auto emit_chunk = [&](size_t start, size_t end) {
      if (end <= start) return;
      if (byte_fallback_) {
        std::string chunk;
        chunk.reserve(end - start + SP.size());
        for (size_t j = start; j < end; j++) {
          if (data[j] == ' ')
            chunk += SP;
          else
            chunk.push_back(data[j]);
        }
        encode_chunk_sp(chunk.data(), chunk.size(), out);
      } else {
        encode_chunk(data + start, end - start, out);
      }
    };

    size_t chunk_start = 0;
    size_t i = 0;
    while (i < len) {
      if (data[i] == ' ') {
        size_t chunk_end = i + 1;
        emit_chunk(chunk_start, chunk_end);
        chunk_start = chunk_end;
        i = chunk_end;
      } else {
        i++;
      }
    }
    emit_chunk(chunk_start, len);
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

    if (__builtin_expect(len == 1, 0)) {
      int id = byte_initial_id_[static_cast<uint8_t>(data[0])];
      if (id >= 0) out.push_back(id);
      return;
    }

    if (__builtin_expect(!cache_enabled_ || len > 0xFFFF, 0)) {
      if (__builtin_expect(ignore_merges_, 0)) {
        std::string bl = bytes_to_bytelevel(data, len);
        auto it = token_to_id_.find(bl);
        if (it != token_to_id_.end()) { out.push_back(it->second); return; }
      }
      bpe_encode(data, len, out);
      return;
    }

    Hash128 h = chunk_hash(data, len);
    size_t slot_idx = static_cast<size_t>(h.lo) & CACHE_MASK;
    CacheSlot& slot = slot_cache_[slot_idx];

    if (__builtin_expect(slot.hash_lo == h.lo &&
                         slot.hash_hi == h.hi &&
                         slot.data_len == static_cast<uint16_t>(len), 1)) {
      const int tc = slot.token_count;
      const size_t old_sz = out.size();
      out.resize(old_sz + tc);
      memcpy(&out[old_sz], slot.tokens, tc * sizeof(int));
      return;
    }

    if (__builtin_expect(ignore_merges_, 0)) {
      std::string bl = bytes_to_bytelevel(data, len);
      auto it = token_to_id_.find(bl);
      if (it != token_to_id_.end()) {
        out.push_back(it->second);
        slot = {h.lo, h.hi, static_cast<uint16_t>(len), 1, {it->second}};
        return;
      }
    }

    size_t before = out.size();
    bpe_encode(data, len, out);
    size_t produced = out.size() - before;

    if (produced <= MAX_CACHED_TOKENS) {
      slot.hash_lo = h.lo;
      slot.hash_hi = h.hi;
      slot.data_len = static_cast<uint16_t>(len);
      slot.token_count = static_cast<uint16_t>(produced);
      memcpy(slot.tokens, &out[before], produced * sizeof(int));
    }
  }

  // SentencePiece BPE: character-level initial tokenization, byte fallback for unknowns
  void encode_chunk_sp(const char* data, size_t len, std::vector<int>& out) const {
    if (__builtin_expect(!len, 0)) return;

    CacheSlot* slot = nullptr;
    Hash128 h{0, 0};
    const bool cache_ok = cache_enabled_ && len <= 0xFFFF;
    if (cache_ok) {
      h = chunk_hash(data, len);
      size_t slot_idx = static_cast<size_t>(h.lo) & CACHE_MASK;
      slot = &slot_cache_[slot_idx];
      if (__builtin_expect(slot->hash_lo == h.lo &&
                           slot->hash_hi == h.hi &&
                           slot->data_len == static_cast<uint16_t>(len), 1)) {
        out.insert(out.end(), slot->tokens, slot->tokens + slot->token_count);
        return;
      }
    }

    // ignore_merges: try full-word lookup before BPE
    if (__builtin_expect(ignore_merges_, 0)) {
      std::string chunk(data, len);
      auto it = token_to_id_.find(chunk);
      if (it != token_to_id_.end()) {
        out.push_back(it->second);
        if (cache_ok) {
          slot->hash_lo = h.lo;
          slot->hash_hi = h.hi;
          slot->data_len = static_cast<uint16_t>(len);
          slot->token_count = 1;
          slot->tokens[0] = it->second;
        }
        return;
      }
    }

    size_t before = out.size();

    constexpr int SP_STACK = 128;
    int stack_init[SP_STACK], stack_ranks[SP_STACK];
    std::vector<int> heap_init;
    int* init_ids;

    if (__builtin_expect(len <= SP_STACK, 1)) {
      init_ids = stack_init;
    } else {
      heap_init.resize(len);
      init_ids = heap_init.data();
    }

    int nids = 0;
    size_t i = 0;
    while (i < len) {
      uint32_t cp;
      size_t adv = utf8_decode(data, len, i, cp);
      auto it = token_to_id_.find(std::string(data + i, adv));
      if (it != token_to_id_.end()) {
        init_ids[nids++] = it->second;
      } else {
        for (size_t b = 0; b < adv; b++) {
          int bid = byte_initial_id_[static_cast<uint8_t>(data[i + b])];
          if (bid >= 0) init_ids[nids++] = bid;
        }
      }
      i += adv;
    }

    if (nids <= 1) {
      if (nids == 1) out.push_back(init_ids[0]);
    } else if (nids <= BPE_HEAP_THRESHOLD) {
      int sp_ranks[SP_STACK];
      bpe_merge_small(init_ids, nids, sp_ranks, out);
    } else {
      bpe_merge_heap(init_ids, nids, out);
    }

    size_t produced = out.size() - before;
    if (cache_ok && produced <= MAX_CACHED_TOKENS) {
      slot->hash_lo = h.lo;
      slot->hash_hi = h.hi;
      slot->data_len = static_cast<uint16_t>(len);
      slot->token_count = static_cast<uint16_t>(produced);
      memcpy(slot->tokens, &out[before], produced * sizeof(int));
    }
  }

  void bpe_encode(const char* data, size_t len, std::vector<int>& out) const {
    constexpr int STACK_MAX = 128;
    int stack_ids[STACK_MAX], stack_ranks[STACK_MAX];

    int n = static_cast<int>(len);
    if (__builtin_expect(n <= 1, 0)) {
      if (n == 1) {
        int id = byte_initial_id_[static_cast<uint8_t>(data[0])];
        if (id >= 0) out.push_back(id);
      }
      return;
    }

    int* ids;
    std::vector<int> heap_ids;
    if (__builtin_expect(n <= STACK_MAX, 1)) {
      ids = stack_ids;
    } else {
      heap_ids.resize(n);
      ids = heap_ids.data();
    }

    for (int i = 0; i < n; i++)
      ids[i] = byte_initial_id_[static_cast<uint8_t>(data[i])];

    if (n <= BPE_HEAP_THRESHOLD) {
      bpe_merge_small(ids, n, stack_ranks, out);
    } else {
      bpe_merge_heap(ids, n, out);
    }
  }

  __attribute__((always_inline))
  void bpe_merge_small(int* ids, int n, int* ranks, std::vector<int>& out) const {
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

  void bpe_merge_heap(int* ids, int n, std::vector<int>& out) const {
    struct HeapItem { int rank; int left; int right; };
    struct HeapCmp {
      bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.left > b.left;
      }
    };

    std::vector<int> prev(n), next(n);
    std::vector<uint8_t> alive(n, 1);
    for (int i = 0; i < n; i++) {
      prev[i] = i - 1;
      next[i] = (i + 1 < n) ? (i + 1) : -1;
    }

    std::vector<HeapItem> heap;
    heap.reserve(static_cast<size_t>(n));
    HeapCmp cmp;

    auto push_pair = [&](int left) {
      if (left < 0) return;
      int right = next[left];
      if (right < 0) return;
      auto m = merge_map_.find(pack_pair(ids[left], ids[right]));
      if (m) {
        heap.push_back({m->rank, left, right});
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
    };

    for (int i = 0; i < n - 1; i++)
      push_pair(i);

    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      HeapItem item = heap.back();
      heap.pop_back();
      int left = item.left;
      int right = item.right;
      if (left < 0 || right < 0) continue;
      if (!alive[left] || !alive[right]) continue;
      if (next[left] != right) continue;

      auto m = merge_map_.find(pack_pair(ids[left], ids[right]));
      if (!m || m->rank != item.rank) continue;

      ids[left] = m->merged_id;
      alive[right] = 0;

      int right_next = next[right];
      next[left] = right_next;
      if (right_next >= 0)
        prev[right_next] = left;

      int left_prev = prev[left];
      if (left_prev >= 0) push_pair(left_prev);
      push_pair(left);
    }

    for (int i = 0; i != -1; i = next[i]) {
      if (alive[i] && ids[i] >= 0)
        out.push_back(ids[i]);
    }
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

  struct SplitSpan { size_t start; size_t end; bool remove; };
  mutable std::vector<SplitSpan> regex_splits_buf_;
  mutable std::vector<RegexProgram::MatchSpan> regex_matches_buf_;

  void split_with_regex_single(const CompiledSplit& cs, const char* text, size_t full_len,
                               size_t start_pos, size_t end_pos,
                               std::vector<SplitSpan>& out) const {
    cs.prog.find_matches(text, full_len, start_pos, end_pos, regex_matches_buf_);
    if (cs.invert) {
      for (auto& m : regex_matches_buf_) m.is_match = !m.is_match;
    }
    const auto& matches = regex_matches_buf_;
    out.clear();
    out.reserve(matches.size());
    switch (cs.behavior) {
      case SplitBehavior::Removed: {
        for (const auto& m : matches)
          out.push_back({m.start, m.end, m.is_match});
        break;
      }
      case SplitBehavior::Isolated: {
        for (const auto& m : matches)
          out.push_back({m.start, m.end, false});
        break;
      }
      case SplitBehavior::Contiguous: {
        bool prev_match = false;
        for (const auto& m : matches) {
          if (out.empty() || m.is_match != prev_match) {
            out.push_back({m.start, m.end, false});
          } else {
            out.back().end = m.end;
          }
          prev_match = m.is_match;
        }
        break;
      }
      case SplitBehavior::MergedWithPrevious: {
        bool prev_match = false;
        for (const auto& m : matches) {
          if (!out.empty() && m.is_match && !prev_match) {
            out.back().end = m.end;
          } else {
            out.push_back({m.start, m.end, false});
          }
          prev_match = m.is_match;
        }
        break;
      }
      case SplitBehavior::MergedWithNext: {
        bool prev_match = false;
        for (const auto& m : matches) {
          if (!out.empty() && prev_match && !m.is_match) {
            out.back().end = m.end;
          } else {
            out.push_back({m.start, m.end, false});
          }
          prev_match = m.is_match;
        }
        break;
      }
      case SplitBehavior::Unknown:
        throw std::runtime_error("Unsupported split behavior");
    }
  }

  void split_with_regex(const char* text, size_t full_len, size_t start_pos, size_t end_pos,
                        std::vector<SplitSpan>& out) const {
    if (!regex_enabled_ || regex_splits_.empty())
      throw std::runtime_error("Regex pre-tokenizer not initialized");
    if (regex_splits_.size() == 1) {
      split_with_regex_single(regex_splits_[0], text, full_len, start_pos, end_pos, out);
      return;
    }
    split_with_regex_single(regex_splits_[0], text, full_len, start_pos, end_pos, out);
    std::vector<SplitSpan> prev, next_splits;
    for (size_t si = 1; si < regex_splits_.size(); si++) {
      prev.swap(out);
      out.clear();
      for (const auto& span : prev) {
        if (span.remove || span.end <= span.start) {
          out.push_back(span);
          continue;
        }
        split_with_regex_single(regex_splits_[si], text, full_len, span.start, span.end, next_splits);
        out.insert(out.end(), next_splits.begin(), next_splits.end());
      }
    }
  }

  void encode_regex_span(const char* text, size_t full_len, size_t start_pos, size_t end_pos,
                         std::vector<int>& out) const {
    split_with_regex(text, full_len, start_pos, end_pos, regex_splits_buf_);
    for (const auto& s : regex_splits_buf_) {
      if (s.remove) continue;
      if (s.end <= s.start) continue;
      if (byte_fallback_)
        encode_chunk_sp(text + s.start, s.end - s.start, out);
      else
        encode_chunk(text + s.start, s.end - s.start, out);
    }
  }

  void encode_with_regex_pretok(const std::string& text,
                                const std::unordered_set<std::string>& allowed_special,
                                std::vector<int>& out) const {
    const size_t len = text.size();
    const char* data = text.data();
    const bool has_specials = !special_by_prefix_.empty() && !allowed_special.empty();

    if (__builtin_expect(!has_specials, 1)) {
      encode_regex_span(data, len, 0, len, out);
      return;
    }

    size_t pos = 0;
    while (pos < len) {
      auto it = special_by_prefix_.find(data[pos]);
      if (it != special_by_prefix_.end()) {
        size_t matched = 0;
        int matched_id = -1;
        for (const auto& [tok, id] : it->second) {
          if (!allowed_special.count(tok)) continue;
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

      size_t span_end = len;
      for (size_t p = pos + 1; p < len; p++) {
        auto sit = special_by_prefix_.find(data[p]);
        if (sit == special_by_prefix_.end()) continue;
        for (const auto& [tok, id] : sit->second) {
          if (!allowed_special.count(tok)) continue;
          if (tok.size() <= len - p &&
              memcmp(tok.data(), data + p, tok.size()) == 0) {
            span_end = p;
            goto found_special;
          }
        }
      }
      found_special:

      if (span_end > pos)
        encode_regex_span(data, len, pos, span_end, out);
      pos = span_end;
    }
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

    if (pt_mode_ == PT_CL100K && text[pos] == '\'' && pos + 1 < len) {
      char c1 = text[pos + 1] | 0x20;
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
        return pos + 2;
      if (pos + 2 < len) {
        char c2 = text[pos + 2] | 0x20;
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e'))
          return pos + 3;
      }
    }

    // Alt 1/2: [^\r\n\p{L}\p{N}]?\p{L}+  (with contraction suffix for o200k)
    {
      size_t p = pos;
      if (cc0 != CC_NL && cc0 != CC_LETTER && cc0 != CC_DIGIT)
        p += blen;
      if (p < len) {
        size_t cb;
        CClass cc = classify_utf8(text, len, p, cb);
        if (cc == CC_LETTER) {
          while (p < len) {
            cc = classify_utf8(text, len, p, cb);
            if (cc != CC_LETTER) break;
            p += cb;
          }
          if (pt_mode_ == PT_O200K && p < len && text[p] == '\'' && p + 1 < len) {
            char c1 = text[p + 1] | 0x20;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
              return p + 2;
            if (p + 2 < len) {
              char c2 = text[p + 2] | 0x20;
              if ((c1 == 'l' && c2 == 'l') || (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e'))
                return p + 3;
            }
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
        if (pt_mode_ == PT_O200K)
          while (p < len && (text[p] == '\r' || text[p] == '\n' || text[p] == '/')) p++;
        else
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
    const char* fallback_env = std::getenv("NANOTOK_PRETOKENIZER_FALLBACK");
    if (fallback_env && fallback_env[0] != '\0') pretok_strict_ = false;
    regex_enabled_ = false;
    regex_compile_failed_ = false;
    regex_error_.clear();
    regex_splits_.clear();
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

    bool has_pretokenizer = j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null();
    if (has_pretokenizer) {
      if (!try_build_compiled_pretokenizer(j["pre_tokenizer"])) {
        detect_pretokenizer_mode(j["pre_tokenizer"]);
      }
      if (pretok_strict_ && regex_compile_failed_) {
        throw std::runtime_error("Unsupported regex pre-tokenizer: " + regex_error_);
      }
    }

    if (byte_fallback_ && pt_mode_ != PT_METASPACE) {
      bool normalizer_replaces_space = false;
      bool normalizer_prepends = false;
      if (j.contains("normalizer") && !j["normalizer"].is_null()) {
        auto check_normalizer = [&](const json& n) {
          std::string type = json_str(n, "type");
          if (type == "Replace" && n.contains("pattern")) {
            const auto& pat = n["pattern"];
            if (pat.is_object() && json_str(pat, "String") == " " &&
                json_str(n, "content") == "\xe2\x96\x81")
              normalizer_replaces_space = true;
          }
          if (type == "Prepend" && json_str(n, "prepend") == "\xe2\x96\x81")
            normalizer_prepends = true;
        };
        const auto& norm = j["normalizer"];
        if (json_str(norm, "type") == "Sequence" && norm.contains("normalizers")) {
          for (const auto& sub : norm["normalizers"])
            check_normalizer(sub);
        } else {
          check_normalizer(norm);
        }
      }
      if (normalizer_replaces_space || normalizer_prepends) {
        pt_mode_ = PT_METASPACE;
        metaspace_add_prefix_ = normalizer_prepends;
        metaspace_split_ = false;
      } else if (!has_pretokenizer && pt_mode_ == PT_DEFAULT) {
        static const std::string SP = "\xe2\x96\x81";
        if (token_to_id_.count(SP) || token_to_id_.count(SP + "a")) {
          pt_mode_ = PT_METASPACE;
          metaspace_add_prefix_ = true;
          metaspace_split_ = false;
        }
      }
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

  static SplitBehavior parse_split_behavior(const json& node) {
    std::string behavior = json_str(node, "behavior");
    if (behavior == "Removed") return SplitBehavior::Removed;
    if (behavior == "Isolated") return SplitBehavior::Isolated;
    if (behavior == "MergedWithPrevious") return SplitBehavior::MergedWithPrevious;
    if (behavior == "MergedWithNext") return SplitBehavior::MergedWithNext;
    if (behavior == "Contiguous") return SplitBehavior::Contiguous;
    return SplitBehavior::Unknown;
  }

  static bool parse_bool(const json& obj, const std::string& key, bool def) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_boolean())
      return obj[key].get<bool>();
    return def;
  }

  static std::string normalize_regex_pattern(const std::string& pattern) {
    std::string out = pattern;
    const std::string needle = "\\s+(?!\\S)";
    const std::string repl = "\\s+$";
    size_t pos = 0;
    while ((pos = out.find(needle, pos)) != std::string::npos) {
      out.replace(pos, needle.size(), repl);
      pos += repl.size();
    }
    return out;
  }

  static std::string escape_regex_literal(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char c : s) {
      switch (c) {
        case '.': case '^': case '$': case '|': case '?': case '*': case '+':
        case '(': case ')': case '[': case ']': case '{': case '}': case '\\':
          out.push_back('\\');
          break;
        default:
          break;
      }
      out.push_back(c);
    }
    return out;
  }

  bool try_compile_one_split(const json& split_node, const std::string& regex_str) {
    SplitBehavior behavior = parse_split_behavior(split_node);
    if (behavior == SplitBehavior::Unknown) {
      regex_compile_failed_ = true;
      regex_error_ = "unsupported split behavior";
      return false;
    }
    bool invert = parse_bool(split_node, "invert", false);
    std::string pattern = normalize_regex_pattern(regex_str);

    CompiledSplit cs;
    if (!cs.prog.compile(pattern)) {
      regex_compile_failed_ = true;
      regex_error_ = cs.prog.error.empty() ? "compile failed" : cs.prog.error;
      return false;
    }
    cs.behavior = behavior;
    cs.invert = invert;
    regex_splits_.push_back(std::move(cs));
    return true;
  }

  bool try_handle_split_node(const json& node) {
    if (!node.is_object()) return false;
    std::string type = json_str(node, "type");
    if (type != "Split") return false;
    if (!node.contains("pattern")) return false;
    const auto& pattern = node["pattern"];
    if (pattern.is_object()) {
      if (pattern.contains("Regex") && pattern["Regex"].is_string())
        return try_compile_one_split(node, pattern["Regex"].get<std::string>());
      if (pattern.contains("String") && pattern["String"].is_string())
        return try_compile_one_split(node, escape_regex_literal(pattern["String"].get<std::string>()));
    }
    return false;
  }

  bool try_build_compiled_pretokenizer(const json& pt) {
    regex_splits_.clear();

    auto process_json_array = [&](const json& items) -> bool {
      bool has_bytelevel = false;
      for (const auto& sub : items) {
        std::string type = json_str(sub, "type");
        if (type == "ByteLevel") {
          has_bytelevel = true;
          if (sub.contains("add_prefix_space") && sub["add_prefix_space"].is_boolean())
            config_.add_prefix_space = sub.at("add_prefix_space").get<bool>();
          if (sub.contains("trim_offsets") && sub["trim_offsets"].is_boolean())
            config_.trim_offsets = sub.at("trim_offsets").get<bool>();
        }
      }
      if (has_bytelevel) return false;
      for (const auto& sub : items) {
        std::string type = json_str(sub, "type");
        if (type == "Split") {
          if (!try_handle_split_node(sub)) return false;
        }
      }
      return !regex_splits_.empty();
    };

    if (pt.is_object()) {
      if (json_str(pt, "type") == "Sequence" && pt.contains("pretokenizers") &&
          pt["pretokenizers"].is_array()) {
        if (process_json_array(pt["pretokenizers"])) {
          regex_enabled_ = true;
          pt_mode_ = PT_REGEX;
          return true;
        }
      } else {
        if (!byte_fallback_ && try_handle_split_node(pt) && !regex_splits_.empty()) {
          regex_enabled_ = true;
          pt_mode_ = PT_REGEX;
          return true;
        }
      }
    }
    if (pt.is_array()) {
      if (process_json_array(pt)) {
        regex_enabled_ = true;
        pt_mode_ = PT_REGEX;
        return true;
      }
    }
    return false;
  }

  // Auto-detect pre-tokenizer mode from tokenizer.json pre_tokenizer config
  void detect_pretokenizer_mode(const json& pt) {
    if (regex_enabled_) return;
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
        std::string repl = json_str(node, "replacement");
        if (!repl.empty())
          metaspace_replacement_ = repl;
        if (node.contains("add_prefix_space") && node["add_prefix_space"].is_boolean())
          metaspace_add_prefix_ = node["add_prefix_space"].get<bool>();
        std::string prepend_scheme = json_str(node, "prepend_scheme");
        if (!prepend_scheme.empty()) {
          if (prepend_scheme == "always" || prepend_scheme == "first")
            metaspace_add_prefix_ = true;
          else if (prepend_scheme == "never")
            metaspace_add_prefix_ = false;
        }
        if (node.contains("split") && node["split"].is_boolean())
          metaspace_split_ = node["split"].get<bool>();
      }

      // Split pre-tokenizer
      if (type == "Split" && node.contains("pattern")) {
        const auto& pattern = node["pattern"];

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
        int byte_val = parse_byte_token(tok);
        if (byte_val >= 0) {
          decode_pool_.push_back(static_cast<char>(byte_val));
        } else if (pt_mode_ == PT_METASPACE || pt_mode_ == PT_SPLIT_SPC) {
          const std::string& sp = metaspace_replacement_;
          size_t i = 0;
          while (i < tok.size()) {
            if (i + sp.size() <= tok.size() &&
                memcmp(tok.data() + i, sp.data(), sp.size()) == 0) {
              decode_pool_.push_back(' ');
              i += sp.size();
            } else {
              decode_pool_.push_back(tok[i]);
              i++;
            }
          }
        } else {
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
