"""
Benchmark nanotok vs tiktoken, HuggingFace tokenizers, and transformers.

Usage:
  python benchmarks/bench.py
  python benchmarks/bench.py --model meta-llama/Llama-3.2-1B
  python benchmarks/bench.py --encoding o200k_base
"""

import argparse
import gc
import os
import platform
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Corpus ───────────────────────────────────────────────────────────────────

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

CODE = """\
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Protocol

class TokenStream(Protocol):
    async def __aiter__(self) -> AsyncIterator[str]: ...

@dataclass
class ConversationTurn:
    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)

class ConversationManager:
    def __init__(self, max_history: int = 100):
        self._history: list[ConversationTurn] = []
        self._max_history = max_history
        self._token_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def add_turn(self, role: str, content: str, **kwargs) -> ConversationTurn:
        async with self._lock:
            turn = ConversationTurn(role=role, content=content, **kwargs)
            self._history.append(turn)
            self._token_counts[role] += len(content.split())
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            return turn

    async def stream_response(self, stream: TokenStream) -> str:
        chunks = []
        async for token in stream:
            chunks.append(token)
        full = "".join(chunks)
        await self.add_turn("assistant", full)
        return full

    def get_context_window(self, max_tokens: int = 4096) -> list[ConversationTurn]:
        result, total = [], 0
        for turn in reversed(self._history):
            est = len(turn.content.split())
            if total + est > max_tokens:
                break
            result.append(turn)
            total += est
        return list(reversed(result))

    def export_json(self, path: Path) -> None:
        data = [{"role": t.role, "content": t.content, "tool_calls": t.tool_calls} for t in self._history]
        path.write_text(json.dumps(data, indent=2))
"""

MATH = (
    "The Navier-Stokes equations: ρ(∂v/∂t + v·∇v) = −∇p + μ∇²v + f. "
    "Schrödinger equation: iℏ ∂Ψ/∂t = ĤΨ. "
    "Riemann zeta: ζ(s) = Σ(n=1→∞) 1/nˢ for Re(s) > 1. "
    "Euler product: ζ(s) = Π_p (1 − p⁻ˢ)⁻¹ over all primes. "
    "Cauchy integral: f(a) = (1/2πi) ∮ f(z)/(z−a) dz. "
    "Maxwell: ∇·E = ρ/ε₀, ∇·B = 0, ∇×E = −∂B/∂t, ∇×B = μ₀(J + ε₀∂E/∂t). "
)

MULTILINGUAL = (
    "Hello World! Bonjour le monde! ¡Hola Mundo! Hallo Welt! "
    "こんにちは世界。自然言語処理は人工知能の重要な分野です。 "
    "你好世界。深度学习模型需要大量训练数据和计算资源。 "
    "안녕하세요 세계! 한국어는 교착어로서 형태소 분석이 중요합니다. "
    "Привет мир! Машинное обучение — подмножество ИИ. "
    "مرحبا بالعالم! معالجة اللغة الطبيعية مجال مهم. "
    "नमस्ते दुनिया! प्राकृतिक भाषा प्रसंस्करण महत्वपूर्ण है। "
)


def _fetch_prose() -> str:
    cache = os.path.join(os.path.dirname(__file__), ".corpus_cache.txt")
    if os.path.exists(cache):
        with open(cache) as f:
            return f.read()

    print("  Downloading prose corpus (Project Gutenberg)...", end=" ", flush=True)
    try:
        req = urllib.request.Request(GUTENBERG_URL, headers={"User-Agent": "nanotok-bench/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
        lines = raw.split("\n")
        start = next((i for i, l in enumerate(lines) if "Chapter 1" in l), 0)
        end = next((i for i, l in enumerate(lines) if "*** END OF THE PROJECT" in l), len(lines))
        text = "\n".join(lines[start:end]).strip()
        with open(cache, "w") as f:
            f.write(text)
        print(f"cached ({len(text):,} bytes)")
        return text
    except Exception as e:
        print(f"failed ({e}), using fallback")
        return ("It is a truth universally acknowledged, that a single man in "
                "possession of a good fortune, must be in want of a wife. ") * 300


def build_corpus() -> dict[str, str]:
    prose = _fetch_prose()
    return {
        "prose (short)":   prose[:500],
        "prose (1KB)":     prose[:1_000],
        "code":            CODE,
        "math/science":    MATH,
        "multilingual":    MULTILINGUAL,
        "prose (10KB)":    prose[:10_000],
        "prose (100KB)":   prose[:100_000],
        "prose (full)":    prose,
    }


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    n_tokens: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    mb_s: float
    tok_s: float


@dataclass
class Backend:
    name: str
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    encode_batch: Callable[[list[str]], list[list[int]]] | None = None


# ── Measurement ──────────────────────────────────────────────────────────────

def measure(fn, arg, *, warmup: int = 10, runs: int = 200) -> list[float]:
    for _ in range(warmup):
        fn(arg)

    gc.disable()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    gc.enable()
    return times


def calc_stats(times: list[float], text_bytes: int, n_tokens: int) -> Stats:
    s = sorted(times)
    med = statistics.median(s)
    return Stats(
        n_tokens=n_tokens,
        median_ms=med * 1000,
        p95_ms=s[int(len(s) * 0.95)] * 1000,
        p99_ms=s[int(len(s) * 0.99)] * 1000,
        mb_s=text_bytes / med / 1e6 if med > 0 else float("inf"),
        tok_s=n_tokens / med if med > 0 else float("inf"),
    )


def adaptive_params(text_bytes: int) -> tuple[int, int]:
    if text_bytes < 500:     return 20, 500
    if text_bytes < 5_000:   return 15, 300
    if text_bytes < 50_000:  return 10, 100
    if text_bytes < 200_000: return 5, 30
    return 3, 10


# ── Backend Loading ──────────────────────────────────────────────────────────

def load_hf_backends(model: str) -> list[Backend]:
    backends = []

    try:
        from nanotok import Tokenizer
        tok = Tokenizer.from_pretrained(model)
        backends.append(Backend("nanotok", tok.encode, tok.decode, tok.encode_batch))
    except Exception as e:
        print(f"  ✗ nanotok: {e}")

    try:
        from tokenizers import Tokenizer as HFTokenizer
        tok = HFTokenizer.from_pretrained(model)
        backends.append(Backend(
            name="tokenizers (Rust)",
            encode=lambda t, _t=tok: _t.encode(t, add_special_tokens=False).ids,
            decode=lambda ids, _t=tok: _t.decode(ids),
            encode_batch=lambda ts, _t=tok: [e.ids for e in _t.encode_batch(ts, add_special_tokens=False)],
        ))
    except Exception as e:
        print(f"  ✗ tokenizers: {e}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        backends.append(Backend(
            name="transformers",
            encode=lambda t, _t=tok: _t.encode(t, add_special_tokens=False),
            decode=lambda ids, _t=tok: _t.decode(ids),
            encode_batch=lambda ts, _t=tok: _t(ts, add_special_tokens=False)["input_ids"],
        ))
    except Exception as e:
        print(f"  ✗ transformers: {e}")

    return backends


def load_tiktoken_backends(encoding: str) -> list[Backend]:
    backends = []

    try:
        from nanotok import Tokenizer
        tok = Tokenizer.from_tiktoken(encoding)
        backends.append(Backend("nanotok", tok.encode, tok.decode, tok.encode_batch))
    except Exception as e:
        print(f"  ✗ nanotok: {e}")

    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        backends.append(Backend("tiktoken (Rust)", enc.encode, enc.decode, enc.encode_batch))
    except Exception as e:
        print(f"  ✗ tiktoken: {e}")

    return backends


# ── Token Parity ─────────────────────────────────────────────────────────────

def verify_parity(backends: list[Backend], corpus: dict[str, str]) -> bool:
    ok = True
    for label, text in corpus.items():
        ref_ids, ref_name = None, None
        for b in backends:
            ids = b.encode(text)
            if ref_ids is None:
                ref_ids, ref_name = ids, b.name
            elif ids != ref_ids:
                ok = False
                print(f"\n    MISMATCH '{label}': {b.name} ({len(ids)} tok) vs {ref_name} ({len(ref_ids)} tok)")
                for i, (a, r) in enumerate(zip(ids, ref_ids)):
                    if a != r:
                        print(f"      first diff at index {i}: {a} vs {r}")
                        break
    return ok


# ── Output Formatting ────────────────────────────────────────────────────────

HEADER = f"{'Backend':<24} {'Tokens':>7} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'MB/s':>10} {'tok/s':>12} {'vs best':>8}"
RULE   = "─" * len(HEADER)


def _row(name: str, s: Stats, vs: str = "") -> str:
    return (f"{name:<24} {s.n_tokens:>7,} {s.median_ms:>10.3f} {s.p95_ms:>10.3f} "
            f"{s.p99_ms:>10.3f} {s.mb_s:>10.1f} {s.tok_s:>12,.0f} {vs:>8}")


def _print_results(results: list[tuple[str, Stats]]):
    best = min(s.median_ms for _, s in results)
    for name, s in results:
        vs = "base" if abs(s.median_ms - best) < best * 0.005 else f"{s.median_ms / best:.2f}x"
        print(f"    {_row(name, s, vs)}")


# ── Benchmarks ───────────────────────────────────────────────────────────────

def _run_bench(label: str, backends: list[Backend], fn_key: str, corpus: dict[str, str],
               *, skip_long: bool = False) -> dict[str, dict[str, Stats]]:
    all_results: dict[str, dict[str, Stats]] = {}
    print(f"\n  ┌─ {label} {'─' * (len(HEADER) - len(label) + 1)}┐")

    for name, text in corpus.items():
        nbytes = len(text.encode())
        if skip_long and nbytes > 50_000:
            continue
        warmup, runs = adaptive_params(nbytes)

        print(f"\n    {name} ({nbytes:,} bytes) — {warmup} warmup, {runs} runs")
        print(f"    {HEADER}")
        print(f"    {RULE}")

        results, per_backend = [], {}
        for b in backends:
            if fn_key == "encode":
                n_tok = len(b.encode(text))
                times = measure(b.encode, text, warmup=warmup, runs=runs)
            else:
                ids = b.encode(text)
                n_tok = len(ids)
                times = measure(b.decode, ids, warmup=warmup, runs=runs)

            stats = calc_stats(times, nbytes, n_tok)
            results.append((b.name, stats))
            per_backend[b.name] = stats

        _print_results(results)
        all_results[name] = per_backend

    return all_results


def bench_batch(backends: list[Backend], corpus: dict[str, str]):
    text = next((v for k, v in corpus.items() if "1KB" in k), next(iter(corpus.values())))
    print(f"\n  ┌─ BATCH ENCODE {'─' * (len(HEADER) - 13)}┐")

    for batch_size in [100, 1_000]:
        batch = [text] * batch_size
        total_bytes = len(text.encode()) * batch_size
        warmup, runs = 3, max(5, 20 // max(1, batch_size // 100))

        print(f"\n    {batch_size:,} texts × {len(text.encode()):,} bytes — {warmup} warmup, {runs} runs")
        print(f"    {HEADER}")
        print(f"    {RULE}")

        results = []
        for b in backends:
            if b.encode_batch is None:
                continue
            total_tok = sum(len(ids) for ids in b.encode_batch(batch))
            times = measure(b.encode_batch, batch, warmup=warmup, runs=runs)
            results.append((b.name, calc_stats(times, total_bytes, total_tok)))

        _print_results(results)


def print_summary(encode_results: dict[str, dict[str, Stats]], names: list[str]):
    print(f"\n  ┌─ SUMMARY (encode p50 ms) {'─' * 52}┐")
    header = f"    {'Text':<20}" + "".join(f"{n:>18}" for n in names) + f"{'fastest':>12}"
    print(f"\n{header}")
    print(f"    {'─' * (len(header) - 4)}")

    for label, per_backend in encode_results.items():
        medians = {n: per_backend[n].median_ms for n in names if n in per_backend}
        if not medians:
            continue
        best = min(medians.values())
        best_name = next(n for n, v in medians.items() if v == best)
        row = f"    {label:<20}"
        for n in names:
            ms = medians.get(n, float("nan"))
            row += f"{ms:>16.3f}{' *' if ms == best else '  '}"
        print(f"{row}{best_name:>12}")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_group(title: str, backends: list[Backend], corpus: dict[str, str]):
    if len(backends) < 2:
        print(f"\n  Skipping '{title}': need >= 2 backends (got: {', '.join(b.name for b in backends) or 'none'})\n")
        return

    print(f"\n{'=' * 76}")
    print(f" {title:^74} ")
    print(f"{'=' * 76}")

    print(f"\n  Backends: {', '.join(b.name for b in backends)}")
    print(f"  Verifying token parity across {len(corpus)} texts...", end=" ")
    ok = verify_parity(backends, corpus)
    print("✓ all match" if ok else "\n  ✗ MISMATCH (benchmarks still run for speed comparison)")

    encode_results = _run_bench("ENCODE", backends, "encode", corpus)
    _run_bench("DECODE", backends, "decode", corpus, skip_long=True)
    bench_batch(backends, corpus)
    print_summary(encode_results, [b.name for b in backends])
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark nanotok against other tokenizer backends")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--encoding", default="cl100k_base")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} · {platform.system()} {platform.machine()}")
    print(f"Model: {args.model} | Encoding: {args.encoding}\n")

    corpus = build_corpus()
    print(f"Corpus: {len(corpus)} texts, {sum(len(v.encode()) for v in corpus.values()):,} total bytes")
    for label, text in corpus.items():
        print(f"  {label:<20} {len(text.encode()):>10,} bytes")

    print(f"\nLoading backends...")
    hf_backends = load_hf_backends(args.model)
    tt_backends = load_tiktoken_backends(args.encoding)

    run_group(f"HuggingFace Model: {args.model}", hf_backends, corpus)
    run_group(f"tiktoken Encoding: {args.encoding}", tt_backends, corpus)


if __name__ == "__main__":
    main()
