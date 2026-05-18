"""Public syllable-counting API: deterministic heuristics + the neural
model, composed.

This is the layer the haiku project (and anything else) imports. It owns
the *integration* and the *policy*; it touches neither `heuristics.py` nor
`model.py` (the other session's files) -- it composes them:

  * `heuristics.analyze()` is deterministic-only (no model tier exists
    there anymore); it resolves a token or marks it OOV in its trace.
  * those OOV tokens, batched across the whole `count_batch` call, go to
    `model.Model.logits` in ONE padded fixed-size forward (`ceil(N/B)`
    compiles-once calls) -- the design justified from firehose data.
  * a single `fuzziness` knob is the only abstention policy (v1: the model
    softmax gate). `fuzziness=0` = strict/conservative; `1` = best-guess.

Contract: `count_batch(lines) -> list[int | None]`. `None` means "not a
confidently-known count (uncertain, or > max_syllables)" -- the caller
discards it. `int` is always a confident exact count within max_syllables.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import heuristics

CKPT = None  # default: weights shipped in the package (syllables/weights/)
MODEL_BATCH = 64  # fixed JAX batch (one XLA compile); from firehose sizing
# fuzziness=0 endpoints; the gate linearly relaxes to (0, 0) at fuzziness=1.
_TOP0, _MARGIN0 = 0.85, 0.35


# --- neural model: lazy, optional, batched, fixed-size ------------------- #
_MDL = None
_MDL_TRIED = False


def _model():
    """The serving model, or None if jax/checkpoint unavailable (then OOV
    tokens simply abstain -- the deterministic pipeline still works)."""
    global _MDL, _MDL_TRIED
    if not _MDL_TRIED:
        _MDL_TRIED = True
        try:
            from . import model

            _MDL = model.Model.load(CKPT)
        except Exception:
            _MDL = None
    return _MDL


def _probs_fixed(mdl, words: list[str]) -> np.ndarray:
    """probs for `words`, every forward padded to exactly MODEL_BATCH so
    JAX compiles once. Dummy pad rows ('a') are sliced off."""
    out = []
    for i in range(0, len(words), MODEL_BATCH):
        chunk = words[i : i + MODEL_BATCH]
        pad = MODEL_BATCH - len(chunk)
        p = mdl.probs(list(chunk) + ["a"] * pad)
        out.append(np.asarray(p)[: len(chunk)])
    return np.concatenate(out, 0) if out else np.zeros((0, 1))


def _gate(p: np.ndarray, fuzziness: float) -> int | None:
    """Softmax row -> count, or None. fuzziness in [0,1] relaxes the
    confidence/margin requirement linearly from strict to argmax-always."""
    f = min(max(fuzziness, 0.0), 1.0)
    order = p.argsort()[::-1]
    top = float(p[order[0]])
    second = float(p[order[1]])
    if top >= (1 - f) * _TOP0 and (top - second) >= (1 - f) * _MARGIN0:
        return int(order[0]) + 1
    return None


def _is_full_url(raw: str) -> bool:
    """True for a real posted URL (scheme/path/query) -> the line is
    discarded. False for a bare domain (`ebay.com`, `www.x.org`), which
    stays spoken-counted: that's a fine thing to read aloud."""
    b = raw
    low = b.lower()
    for pre in ("https://", "http://"):
        if low.startswith(pre):
            return True  # explicit scheme = a real URL
    i = b.find("/")
    return (i != -1 and bool(b[i + 1 :].strip("/ "))) or "?" in b


def _looks_junk(w: str) -> bool:
    """An OOV token that is not a plausible spoken word: a long ID/macro
    mash (`hhFDWkdAxDjkbJfXdOwv`), random interior caps, or a long
    consonant run. The model has no business guessing these and a line
    containing one should be discarded, not labelled."""
    if len(w) >= 16:
        return True
    s = w[1:]
    if (
        sum(
            a.isupper() != b.isupper()
            for a, b in zip(s, s[1:], strict=False)  # adjacent pairs
            if a.isalpha() and b.isalpha()
        )
        >= 4
    ):
        return True
    return len(w) >= 7 and not (set(w.lower()) & set("aeiouy"))


# --- token classification from the deterministic trace ------------------- #
def _classify(c) -> tuple[str, object]:
    """A Counted -> ('det', count) resolved | ('oov', word) model-fillable
    | ('hard', None) unrecoverable abstain (roman / ambiguous-initialism /
    junk / an OOV sub-word of a number: the whole line must abstain)."""
    src = getattr(c, "source", "")
    note = getattr(c, "note", "") or ""
    if "spoken url" in note:  # heuristics url/email
        if _is_full_url(c.raw) or src == "abstain":
            # a posted URL, or a host/label we couldn't count (an OOV
            # sub-word silently counted 0) -> abstain, never a guess
            return "hard", None
        return "det", c.count  # bare domain, resolved -> spoken
    if src not in ("abstain", "model"):
        return "det", c.count  # incl. silent (count 0)
    tail = note.split(";")[-1].strip()
    if (
        "[OOV:" not in note
        and "initialism" not in note
        and "roman" not in note
        and tail == "OOV"
        and any(ch.isalpha() for ch in c.raw)
    ):
        if _looks_junk(c.raw):
            return "hard", None  # never model an ID/macro mash
        # hand the model the normalized form the deterministic layer
        # tried (collapse elongation, fold diacritics) -- it was trained
        # on clean a-z words, not raw emphasis-spam ("blarghhhh").
        norm = heuristics._strip_diacritics(
            heuristics.collapse_elongation(c.raw.lower())[0]
        )
        return "oov", norm or c.raw.lower()
    return "hard", None


@lru_cache(maxsize=4096)
def _resolve_line(text: str):
    """Deterministic pass (cached). Returns (det_sum, oov_words tuple,
    hard) where det_sum already counts everything heuristics resolved and
    oov_words are the tokens needing the model (or None if `hard`)."""
    det, oov, hard = 0, [], False
    for c in heuristics.analyze(text).tokens:
        kind, val = _classify(c)
        if kind == "hard":
            hard = True
            break
        if kind == "oov":
            oov.append(val)
        else:
            det += val
    return det, tuple(oov), hard


def count_batch(
    lines, *, max_syllables: int | None = None, fuzziness: float = 0.0
) -> list[int | None]:
    """Conservative syllable counts for many lines. `None` = discard
    (uncertain, hard-abstain, or > max_syllables). One batched, fixed-size
    model forward for every OOV token across the whole call."""
    n = len(lines)
    det = [0] * n
    line_oov: list[list[int]] = [[] for _ in range(n)]  # idx into `flat`
    result: list[int | None] = [None] * n
    flat: list[str] = []

    for i, line in enumerate(lines):
        d, oov, hard = _resolve_line(line)
        if hard:
            continue  # -> None
        # early-stop: OOV floors at 1 syllable; if the minimum already
        # blows the budget, no point asking the model.
        if max_syllables is not None and d + len(oov) > max_syllables:
            continue  # -> None
        det[i] = d
        if not oov:
            result[i] = d  # pure-heuristic line
            continue
        for w in oov:
            line_oov[i].append(len(flat))
            flat.append(w)

    need = [i for i in range(n) if line_oov[i]]
    if need:
        mdl = _model()
        if mdl is None:
            return result  # OOV lines stay None
        probs = _probs_fixed(mdl, flat)
        counts = [_gate(probs[j], fuzziness) for j in range(len(flat))]
        for i in need:
            cs = [counts[j] for j in line_oov[i]]
            if any(c is None for c in cs):
                continue  # any OOV abstains
            tot = det[i] + sum(cs)
            if max_syllables is None or tot <= max_syllables:
                result[i] = tot
    return result


def count_syllables(line: str, **kwargs) -> int | None:
    """Single line; `None` = discard. Exactly `count_batch` of one line --
    forwards `**kwargs` so the two can never drift (max_syllables,
    fuzziness)."""
    return count_batch([line], **kwargs)[0]


def is_haiku_line(line: str, *, fuzziness: float = 0.0) -> int | None:
    """5 or 7 if the line is *confidently* exactly that many syllables,
    else None. Drop-in conservative replacement for the haiku project's
    `count_syllables(t) in {5,7}` check."""
    n = count_syllables(line, max_syllables=7, fuzziness=fuzziness)
    return n if n in (5, 7) else None


# Rich, auditable per-token trace (deterministic; no model involved).
analyze = heuristics.analyze
