"""Training data: source parsing, reconciliation, splits.

The input contract (alphabet, `encode`, MAX_LEN, NUM_CLASSES, PAD) lives
in `model.py` so serving never imports this module. Words are stripped to
lowercase a-z (the pinned alphabet) before keying, so `it's`/`its` and
`well-being`/`wellbeing` collapse to one key -- collided entries are
merged by unioning their per-source counts.

`NUM_CLASSES` and `load()['vocab_size']` mirror the model contract for
training code; `split='all'` trains on every word (no holdout).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import model

_DATA = Path(__file__).resolve().parent.parent / "data"
_CMUDICT_PATH = _DATA / "cmudict.dict"
_WIKIPRON_PATH = _DATA / "wikipron_us.tsv"
_KAIKKI_PATH = _DATA / "kaikki_counts.tsv"

MAX_COUNT = model.MAX_COUNT
NUM_CLASSES = model.NUM_CLASSES


def _az(w: str) -> str:
    return "".join(c for c in w.lower() if "a" <= c <= "z")


# --------------------------------------------------------------------------- #
# per-source parsing
# --------------------------------------------------------------------------- #
_CMU_VARIANT = re.compile(r"^(.+?)\((\d+)\)$")


def parse_cmudict(path=_CMUDICT_PATH) -> dict[str, dict]:
    """word -> {"primary": int, "counts": tuple}. Gold, phonological:
    count = phonemes carrying a stress digit."""
    primary: dict[str, int] = {}
    counts: dict[str, set[int]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line:
            continue
        head, *phones = line.split(" ")
        m = _CMU_VARIANT.match(head)
        word = m.group(1) if m else head
        n = sum(1 for p in phones if p and p[-1].isdigit())
        counts.setdefault(word, set()).add(n)
        if m is None and word not in primary:
            primary[word] = n
    for word, cs in counts.items():
        primary.setdefault(word, min(cs))
    return {
        w: {"primary": primary[w], "counts": tuple(sorted(counts[w]))} for w in counts
    }


_WP_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝ")
_WP_SYLLABIC = "̩"


def _wp_count(ipa: str) -> int:
    n, prev = 0, False
    for ph in ipa.split():
        cur = any(c in _WP_VOWELS for c in ph) or _WP_SYLLABIC in ph
        if cur and not prev:
            n += 1
        prev = cur
    return n


def parse_wikipron(path=_WIKIPRON_PATH) -> dict[str, dict]:
    """word -> {"counts": tuple}. Broad IPA; breadth/slang, ~92% vs CMU
    (systematic -1 hiatus undercount)."""
    counts: dict[str, set[int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word, _, ipa = line.strip().partition("\t")
        if not word or not ipa:
            continue
        c = _wp_count(ipa)
        if c >= 1:
            counts.setdefault(word.lower(), set()).add(c)
    return {w: {"counts": tuple(sorted(cs))} for w, cs in counts.items()}


def parse_kaikki(path=_KAIKKI_PATH) -> dict[str, dict]:
    """word -> {"ipa": tuple, "hyph": tuple} from the compact TSV that
    scripts/fetch_kaikki.py produced. Only `hyph` is used downstream."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        w, ip, hy = (line.split("\t") + ["", ""])[:3]
        out[w] = {
            "ipa": tuple(int(x) for x in ip.split(",") if x),
            "hyph": tuple(int(x) for x in hy.split(",") if x),
        }
    return out


# --------------------------------------------------------------------------- #
# reconciliation (a-z keyed; collisions merged by count union)
# --------------------------------------------------------------------------- #
def build_sources() -> dict[str, dict]:
    """Reconcile the three sources into one a-z-keyed labelled table.

    Reliable = CMUdict (gold) and kaikki *hyphenation* (orthogonal errors,
    fixes WikiPron's -1 hiatus undercount). kaikki IPA-dots dropped (too
    noisy). WikiPron is breadth/slang only; its lone disagreement is the
    hiatus artifact, so it never widens valid or triggers ambiguity vs a
    reliable source.

      primary    CMU primary, else kaikki-hyph, else WikiPron
      valid      union of reliable sources; if none, the WikiPron set
      ambiguous  a reliable source lists >1, or CMU vs kaikki-hyph disagree
      source     provenance of `primary`
    """
    cm: dict[str, dict] = {}
    for w, e in parse_cmudict().items():
        k = _az(w)
        if not k:
            continue
        d = cm.setdefault(k, {"primary": e["primary"], "counts": set()})
        d["counts"].update(e["counts"])
        d["primary"] = min(d["primary"], e["primary"])

    wp: dict[str, set] = {}
    for w, e in parse_wikipron().items():
        k = _az(w)
        if k:
            wp.setdefault(k, set()).update(e["counts"])

    kk: dict[str, set] = {}
    for w, e in parse_kaikki().items():
        k = _az(w)
        if k and e["hyph"]:
            kk.setdefault(k, set()).update(e["hyph"])

    out: dict[str, dict] = {}
    for w in set(cm) | set(wp) | set(kk):
        cmu_c = set(cm[w]["counts"]) if w in cm else None
        wp_c = set(wp[w]) if w in wp else None
        kk_c = set(kk[w]) if w in kk else None
        if cmu_c is None and wp_c is None and kk_c is None:
            continue

        reliable = [s for s in (cmu_c, kk_c) if s]
        valid = set().union(*reliable) if reliable else set(wp_c)

        if cmu_c:
            primary, src = cm[w]["primary"], "cmu"
        elif kk_c:
            primary, src = min(kk_c), "kaikki"
        else:
            primary, src = min(wp_c), "wikipron"

        ambiguous = (
            (cmu_c is not None and len(cmu_c) > 1)
            or (kk_c is not None and len(kk_c) > 1)
            or (cmu_c is not None and kk_c is not None and cmu_c != kk_c)
        )
        out[w] = {
            "primary": primary,
            "valid": frozenset(valid),
            "ambiguous": ambiguous,
            "source": src,
        }
    return out


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
_SUFFIXES = (
    "iness",
    "ation",
    "ments",
    "ment",
    "ness",
    "iest",
    "ies",
    "ied",
    "ing",
    "edly",
    "ed",
    "es",
    "er",
    "est",
    "ly",
    "al",
    "ic",
    "s",
    "y",
)


def stem(w: str) -> str:
    """Crude inflectional stem -- a leakage key, not real morphology. Keeps
    run/runs/running in one split partition for the 'hard' split."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def load(
    seed: int = 0,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    split: str = "random",
):
    table = build_sources()
    words = sorted(w for w, e in table.items() if w and 1 <= e["primary"] <= MAX_COUNT)

    X = np.stack([model.encode(w) for w in words])
    y = np.array([table[w]["primary"] for w in words], dtype=np.int32)
    valid = [
        frozenset(c for c in table[w]["valid"] if 1 <= c <= MAX_COUNT) for w in words
    ]
    amb = np.array([table[w]["ambiguous"] for w in words], dtype=bool)

    Y = np.zeros((len(words), NUM_CLASSES), dtype=np.float32)
    for i, vs in enumerate(valid):
        for c in vs:
            Y[i, c - 1] = 1.0 / len(vs)

    rng = np.random.default_rng(seed)
    n = len(words)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    if split == "random":
        perm = rng.permutation(n)
        test_idx, val_idx = perm[:n_test], perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]
    elif split == "hard":
        groups: dict[str, list[int]] = {}
        for i, w in enumerate(words):
            groups.setdefault(stem(w), []).append(i)
        keys = list(groups)
        rng.shuffle(keys)
        perm = np.array([i for k in keys for i in groups[k]], dtype=int)
        test_idx, val_idx = perm[:n_test], perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]
    elif split == "source_oov":
        test_idx = np.array(
            [i for i, w in enumerate(words) if table[w]["source"] == "wikipron"],
            dtype=int,
        )
        rest = rng.permutation(
            np.array(
                [i for i, w in enumerate(words) if table[w]["source"] != "wikipron"],
                dtype=int,
            )
        )
        val_idx, train_idx = rest[:n_val], rest[n_val:]
    elif split == "all":
        # Ship model: every word trains, no holdout (don't split out the
        # hard examples). train.py disables early-stop when val is empty.
        train_idx = np.arange(n)
        val_idx = test_idx = np.array([], dtype=int)
    else:
        raise ValueError(f"unknown split {split!r}")

    def take(idx):
        return {
            "X": X[idx],
            "y": y[idx],
            "Y": Y[idx],
            "valid": [valid[i] for i in idx],
            "amb": amb[idx],
            "words": [words[i] for i in idx],
        }

    return {
        "vocab_size": model.VOCAB_SIZE,  # pinned constant (27)
        "train": take(train_idx),
        "val": take(val_idx),
        "test": take(test_idx),
    }
