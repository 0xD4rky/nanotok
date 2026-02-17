"""nanotok - High-performance BPE tokenizer. 20-60x faster than tiktoken."""

__version__ = "0.1.0"

from nanotok._core import Tokenizer

__all__ = ["Tokenizer", "__version__"]
