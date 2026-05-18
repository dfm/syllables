"""Stream the kaikki English Wiktionary extract -> data/kaikki_counts.tsv.

The source is ~3 GB of JSONL; we never hold or store it -- iterate it
line by line over HTTP and write only:

    word \\t ipa_counts \\t hyph_counts

ipa_counts:  counts from IPA strings carrying boundary dots (noisy, 61%
             vs CMU -- kept for reference; data.py ignores it).
hyph_counts: counts from the hyphenation field (orthographic, 93.5% vs
             CMU, orthogonal errors -- the signal we actually use).

CC-BY-SA (Wiktionary-derived); data/ is gitignored so not redistributed.

    uv run python scripts/fetch_kaikki.py                # stream the URL
    uv run python scripts/fetch_kaikki.py sample.jsonl   # parse a local file
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
OUT = Path(__file__).resolve().parent.parent / "data" / "kaikki_counts.tsv"

_STRIP = "ˈˌːˑ ̯̩͡‿/[]()|"  # stress/length/tie/delim marks dropped before counting


def ipa_syllables(ipa: str) -> int | None:
    s = "".join(c for c in ipa if c not in _STRIP)
    if "." not in s:
        return None
    groups = [g for g in s.split(".") if g]
    return len(groups) if 1 <= len(groups) <= 12 else None


def extract(entry: dict):
    if entry.get("lang_code") != "en":
        return None
    word = (entry.get("word") or "").strip().lower()
    if not word:
        return None
    ipa: set[int] = set()
    for snd in entry.get("sounds", []):
        if "ipa" in snd:
            c = ipa_syllables(snd["ipa"])
            if c:
                ipa.add(c)
    hyph: set[int] = set()
    for h in entry.get("hyphenations") or entry.get("hyphenation") or []:
        parts = h.get("parts") if isinstance(h, dict) else h
        if parts:
            n = len([p for p in parts if p])
            if 1 <= n <= 12:
                hyph.add(n)
    if not ipa and not hyph:
        return None
    return word, ipa, hyph


def _lines(source):
    if source:
        with open(source, encoding="utf-8") as f:
            yield from f
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "syllables/0.1"})
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                yield raw.decode("utf-8", "ignore")


def run(source=None):
    ipa_by: dict[str, set] = {}
    hyph_by: dict[str, set] = {}
    n = bad = 0
    t0 = time.time()
    for line in _lines(source):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            entry = json.loads(line)
        except Exception:
            bad += 1
            continue
        got = extract(entry)
        if got:
            w, ip, hy = got
            if ip:
                ipa_by.setdefault(w, set()).update(ip)
            if hy:
                hyph_by.setdefault(w, set()).update(hy)
        if n % 200_000 == 0:
            print(
                f"  {n:>9,} lines  {len(ipa_by):>7,} ipa  "
                f"{len(hyph_by):>7,} hyph  {time.time() - t0:.0f}s",
                flush=True,
            )

    words = sorted(set(ipa_by) | set(hyph_by))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for w in words:
            ip = ",".join(map(str, sorted(ipa_by.get(w, ()))))
            hy = ",".join(map(str, sorted(hyph_by.get(w, ()))))
            f.write(f"{w}\t{ip}\t{hy}\n")
    print(
        f"\ndone: {n:,} lines ({bad} unparsable), {len(words):,} words "
        f"written -> {OUT}  ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
