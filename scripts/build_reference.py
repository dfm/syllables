"""Precompute the shipped reference lexicon.

`syllables/lexicon/reference.tsv.xz` is the **single runtime source of
truth** — `heuristics._cmu()` always loads it (dev and installed alike;
no runtime call to `data.build_sources()`). So rerun this whenever the
reconciliation or the raw `data/` extracts change:

    uv run python scripts/build_reference.py   # raw data/ must be present

Reconciles CMUdict + WikiPron + kaikki via `data.build_sources()` and
writes the *minimal* `word\\tprimary` table, lzma-compressed (~0.46 MB,
~169k words). `valid`/`ambiguous` are intentionally dropped — the shipped
path emits `primary` only.
"""

from __future__ import annotations

import lzma
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syllables import data  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "syllables" / "lexicon" / "reference.tsv.xz"


def main() -> None:
    tbl = data.build_sources()
    body = "\n".join(f"{w}\t{r['primary']}" for w, r in sorted(tbl.items())).encode()
    blob = lzma.compress(body, preset=9 | lzma.PRESET_EXTREME)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(blob)
    print(f"{len(tbl):,} words -> {OUT}  ({len(blob) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
