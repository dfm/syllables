"""Quick qualitative probe of the shipped model + the public counter.

    uv run python scripts/infer.py [WEIGHTS.npz]   # default: packaged

Per word: the conservative public count (`None` = abstain) and the raw
model softmax top-3 -- a sanity check on what the net believes vs. what
the system commits to. A probe, not an evaluation.
"""

from __future__ import annotations

import sys

from syllables import count_syllables, model

PROBE = {
    "gen-z slang": [
        "yeet",
        "rizz",
        "bussin",
        "skibidi",
        "gyat",
        "delulu",
        "sigma",
        "goated",
        "mogging",
        "simp",
    ],
    "eye-dialect": [
        "gonna",
        "wanna",
        "lemme",
        "finna",
        "tryna",
        "prolly",
        "yall",
        "aint",
        "cuz",
        "tho",
    ],
    "tech / brand": [
        "tiktok",
        "selfie",
        "crypto",
        "doomscroll",
        "emoji",
        "rickroll",
        "paywall",
        "situationship",
    ],
    "compound": [
        "deadass",
        "lowkey",
        "hangry",
        "ghosting",
        "mansplain",
        "sideeye",
        "thirsttrap",
    ],
    "haiku-ambiguous": [
        "fire",
        "hour",
        "flower",
        "every",
        "interesting",
        "poem",
        "being",
        "prayer",
        "family",
        "camera",
    ],
}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None  # None -> packaged
    m = model.Model.load(ckpt)

    for group, words in PROBE.items():
        print(f"\n=== {group} ===")
        for w, p in zip(words, m.probs(words), strict=True):
            top = sorted(enumerate(p, 1), key=lambda t: -t[1])[:3]
            top_s = " ".join(f"{c}:{pr:.2f}" for c, pr in top)
            n = count_syllables(w)
            box = "ABSTAIN" if n is None else str(n)
            print(f"  {w:14s} -> {box:8s}  model[{top_s}]")


if __name__ == "__main__":
    main()
