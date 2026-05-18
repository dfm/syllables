"""Conservative syllable counting. Public API lives in `harness`."""

from .harness import analyze, count_batch, count_syllables, is_haiku_line

__all__ = ["count_syllables", "count_batch", "is_haiku_line", "analyze"]
