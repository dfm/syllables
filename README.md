# syllables

Conservative English syllable counting for messy real-world text
(Bluesky posts, slang, emoji, numbers, URLs). It returns a count **or
abstains** — it never silently guesses.

```python
from syllables import count_syllables, count_batch, is_haiku_line

# count_syllables is just count_batch of one line -- same kwargs:
count_syllables("five syllable line", max_syllables=7, fuzziness=0.0)  # -> 5
count_syllables("zqxwvb??", max_syllables=7)                           # -> None
count_batch(posts,          max_syllables=7, fuzziness=0.0)  # -> list[int|None]
is_haiku_line("an old silent pond")                          # -> 5 | 7 | None
```

`max_syllables`/`fuzziness` are optional (defaults: `None`, `0.0`) and
identical for `count_syllables` and `count_batch`. `None` means "not
confidently known" (uncertain, or over `max_syllables`) — the caller
discards it. `fuzziness` 0→1 trades precision for coverage.

## How it works

1. **Normalize** — NFKC, emoji/emoticons, numbers/currency, URLs, @/#,
   elongation, contractions (`heuristics.py`).
2. **Resolve deterministically** — curated slang/initialisms, then the
   reconciled CMUdict + WikiPron + Wiktionary reference (`data.py`);
   abstain if unknown.
3. **Model fallback** — OOV tokens are batched through a small char model
   (`model.py`) with a `fuzziness`-tuned abstention gate (`harness.py`).

The deterministic layer never touches the model; `harness.py` owns all
model integration and policy. Docs: [`docs/heuristics.md`](docs/heuristics.md)
— operational reference (tiers, harness API, per-feature decisions);
[`docs/architecture.md`](docs/architecture.md) — why the system is shaped
this way, the data, and the experiments.

## Develop

```
uv run python -m unittest discover -s tests        # tests
uv run python -m syllables.heuristics "text…"      # per-token trace
```
