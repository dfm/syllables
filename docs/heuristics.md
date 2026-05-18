# Design notes: deterministic heuristics + harness

> Scope: the **operational** reference for the deterministic layer and the
> harness. For *why* the system is shaped this way — model architecture,
> the reconciled data set, and the experiments — see
> [`architecture.md`](architecture.md) (canonical for those).

The system is two layers with a hard boundary:

- **`heuristics.py`** — conservative, deterministic normalization +
  lexical counting. Pure stdlib + the reconciled reference (read-only via
  `data`). It resolves a token or **abstains**. *No neural model here.*
- **`harness.py`** — the public API. Runs the deterministic layer, then
  batches the tokens it abstained on through the char model (`model.py`)
  with a single `fuzziness`-tuned gate. Owns all model integration/policy.

```
text ─▶ heuristics: tokenize ─▶ count_token ─▶ analyze   (det. count | abstain)
     ─▶ harness:    early-stop ─▶ batch OOV ─▶ model gate ─▶ count_batch
```

Why the split: keeping the net out of `heuristics.py` is what makes
batching, a single abstention policy, and hermetic testing possible.

Run it:

```
uv run python -m syllables.heuristics "I have 5 cats and $19.99 lol 😂"  # trace
uv run python -m unittest discover -s tests    # hermetic tests (no jax)
python -c "from syllables import count_batch"   # public API
```

## The contract

Precision over recall. Every tier that fires is trusted; the instant none
fires we **abstain** (`confidence="none"`) instead of guessing.
`analyze()` returns a best-effort `total` *and* a `confident` flag —
`confident` is `True` only if every non-silent unit resolved through a
trusted tier with no abstentions. The per-unit trace makes every count
auditable. (*Why* precision-over-recall, and why we deliberately keep the
near-ambiguous cases like `fire`=1/2: `architecture.md` §1.)

## Tier order (a word)

| # | tier | when | confidence |
|---|------|------|------------|
| 1 | curated SLANG | modern word w/ one defensible count | exact |
| 2 | INITIALISM | texting acronym; *ambiguous ones abstain* | exact |
| 3 | reference | reconciled CMUdict+WikiPron+kaikki (via `data`; counts/sources → `architecture.md` §4) — PRIMARY | exact |
| 4 | ACRONYM | said as a word (nasa, scuba) | exact |
| 5 | letters | ALL-CAPS short non-word (`NDA`) or vowelless (`btw`) → spelled aloud | exact |
| 6 | spell-fix | *(disabled — manufactured confident-wrong counts)* | — |
| 7 | — | **abstain** → the harness may ask the model | none |

Numbers / URLs / symbols are expanded into plain words that are all in the
reference, so their count is **exact**, not modelled. `heuristics.py`
itself never reaches a model — tier 7 is a plain abstention; the harness
decides what to do with it.

## Design decisions worth knowing

- **Emoji *and ASCII emoticons* are silent (0 syllables).** Nobody
  pronounces 😂 or `<3` / `:)` / `xD` when reading a tweet aloud. They're
  detected only so they get split off glued-on words (`great😂job`) and
  named in the trace; they never add/remove syllables or make a text
  "uncertain".
- **Interpreted readings are `approx`, not `exact`.** Dates (`5/17/2026` →
  "may seventeenth twenty twenty-six"), times (`10:45pm`), fractions
  (`1/2` → "one half", `½`), and ratios (`3-2` → "three two") are
  deterministic but a *guess about intent*, so they show `~` in the trace
  (still confident — not abstained — the words themselves are exact).
  Slash-shorthand `w/`/`w/o`/`b/c` → with/without/because is unambiguous,
  so it stays `exact`.
- **Magnitude suffixes multiply only with currency.** `$1.2B` → "one
  point two billion dollars". A bare number keeps the letter literal:
  `20k` → "twenty k", `100K` → "one hundred k", `4K` → "four k" — the
  form people actually say, and it sidesteps 20k-views vs 4K-TV.
- **`#` is context-sensitive.** `#` directly before a number is the
  number sign — `#3` → "number three", `Art Collection #3` → "…number
  three". `#` before a word is a hashtag — `#ThrowbackThursday` reads the
  words, hash silent. A lone `#` → "hash".
- **Acronyms / initialisms.** Common ones (`IBM`, `FBI`, `CEO`) are in
  cmudict with their spelled pronunciation → exact. Ones cmudict lacks
  (`NDA`, `ACLU`) hit the ALL-CAPS letters tier → spelled out. Said-as-a-
  word acronyms (`NASA`, `SCUBA`) resolve as words *before* that tier, so
  they aren't spelled. Vowelless slang (`fr`, `rn`) → letters too ("eff
  arr" = 2, which also matches "for real"). Dotted Latin abbrevs (`e.g.`,
  `i.e.`) split into letters and read spelled ("ee jee" = 2); `etc.` →
  cmudict "et cetera" = 4.
- **Small Roman numerals get a default cardinal reading; larger ones
  abstain.** An all-caps ≥2-char valid-Roman token that isn't a real
  cmudict word and is **≤ 10** reads as the cardinal ("World War II" →
  *two*, "Star Wars IV" → *four*), marked `approx` because cardinal-vs-
  ordinal is a guess about intent — though for 3–10 both readings have
  the same syllable count, so only `II` ("two"=1 vs "second"=2) actually
  trades a little precision for coverage. **> 10** still abstains (the
  ambiguity compounds and big numerals are rarer / more numeral-like).
  (`MIX` is a real cmudict word → counted normally, not as 1009.)
- **Diacritic folding.** A NFKD accent-stripped candidate is tried in the
  cmudict tier, so `café`/`naïve`/`jalapeño` resolve **exact**. Words like
  `résumé` whose fold (`resume`) has disagreeing cmudict variants still
  abstain — the conservative contract.
- **URL recognition: TLD-validated, or path/query overrides.** A *bare*
  dotted host is a URL only if its last label is a known TLD (or a ccTLD
  with a registry second-level: `bit.ly`, `foo.co.uk`) — this keeps
  `e.g.`/`U.S.` from being URLs. But a path or query after a plausible
  host (`youtu.be/…?…`, `t.co/x`) is an unambiguous URL **regardless of
  TLD** (the harness then discards it as a full URL). Email local-part
  dots are voiced ("first **dot** last at gmail dot com").
- **Elongation collapse.** Any letter run ≥3 is emphasis, not spelling:
  `noooooo → no`. We can't know if the base keeps a double (`cooool →
  cool` vs `sooo → so`), so both single- and double-collapsed spellings
  are tried in order; the first a tier knows wins.
- **Years always use the year reading.** A bare 4-digit number in the
  plausible-year band (1000–2099) is *always* read as a year — `1999` →
  "nineteen ninety-nine" (4), never the spelled-out "one thousand nine
  hundred ninety-nine" (8). No abstention for years; it's a committed,
  confident choice. A comma (`1,234`) disqualifies the year reading and
  falls back to the cardinal.
- **Typo correction is timid by choice.** ≥4 chars only (short tokens are
  usually slang, not typos), first letter must survive, and a *unique*
  closest cmudict word is required — a tie abstains. A wrong correction is
  a silent wrong count, which is worse than no answer.
- **Unified reference, not cmudict alone.** Tier 3 is the reconciled
  CMUdict+WikiPron+kaikki table, consumed read-only via `data`
  (composition, word count, source priority, and the soft-uniform loss:
  `architecture.md` §4 — the canonical data writeup). It falls back to
  cmudict-only if the WikiPron/kaikki extracts are absent. The
  deterministic-layer payoff, measured on real firehose posts: confident
  coverage 54.7% → 60.6% with the model off (`rizz`, `skibidi`, `yeet`,
  `bsky` resolve deterministically).
- **Primary even when ambiguous, not fail-closed.** The reference flags
  genuine ambiguity (`fire`={1,2}, `every`={2,3}; ~1.4% of words). Policy
  here: still emit `primary` for max coverage (the 1.4% risk is accepted)
  rather than abstaining.
- **Smart quotes → ASCII `'` before anything else.** iOS/macOS rewrite `'`
  to `’` (U+2019), which NFKC does *not* fold; cmudict keys are ASCII, so
  without this every `I'm`/`don't`/`he's` missed cmudict (~11% of real-
  post abstentions).
- **Spell-fix tier is disabled** (`ENABLE_SPELL=False`). On real posts it
  converted honest OOV abstentions into confident-*wrong* counts
  (`rational`→`rationale`, `collabs`→`collars`); the char model is the
  intended OOV path. Flip the flag to restore it.
- **NFKC folding** maps stylized unicode (𝓱𝓮𝓵𝓵𝓸, ｆｕｌｌｗｉｄｔｈ, ﬁ-ligatures)
  back to ASCII-ish so it isn't all spuriously OOV.

## The harness (public API)

`from syllables import count_syllables, count_batch, is_haiku_line, analyze`

```
count_batch(lines, *, max_syllables=None, fuzziness=0.0) -> list[int | None]
count_syllables(line, **kwargs) -> int | None     # = count_batch of one
is_haiku_line(line, *, fuzziness=0.0) -> 5 | 7 | None
```

- **`None` = discard.** Folds together "uncertain" and "> max_syllables" —
  the caller (e.g. a haiku matcher) drops the post either way. `int` is
  always a confident exact count ≤ `max_syllables`.
- **Composition, zero edits to `heuristics.py`.** The harness runs
  `analyze()` (deterministic), reads which tokens abstained from the
  trace, and resolves only those.
- **Batched, fixed-size model.** All OOV tokens across a `count_batch`
  call go through `model.Model.logits` in `ceil(N / MODEL_BATCH)` forwards
  padded to a fixed `MODEL_BATCH=64` (one XLA compile). Sized from real
  firehose data: ~3.7% of posts ever reach the model, ~1 OOV token each.
- **Early-stop.** A line whose running-min already exceeds
  `max_syllables`, or that contains a hard abstention, is rejected
  *before* the model — this is why only a sliver of traffic hits JAX.
- **URL policy.** A *bare domain* (`ebay.com`, `www.x.org` — no
  scheme/path/query) stays spoken-counted ("ebay dot com"); a *full URL*
  (scheme, or a real path/query: `music.apple.com/us/album/x`,
  `https://…`) **discards the whole line** — it isn't a spoken haiku. The
  deterministic layer already drops the path; this is the discard rule.
- **Junk discards the line.** An OOV token that isn't a plausible spoken
  word (16+ chars, random interior caps like `hhFDWkdAxDjkbJfXdOwv`, or a
  long consonant run) is never sent to the model — the line is dropped.
  Kills the `macro: …` firehose garbage at the source.
- **`fuzziness` ∈ [0,1] is the only abstention policy.** It linearly
  relaxes the model gate from strict (`top≥0.85, margin≥0.35`) at `0` to
  argmax-always at `1` — i.e. the precision↔coverage operating point.
- **Model optional / graceful.** No jax or no checkpoint → OOV lines just
  stay `None`; the deterministic results still flow. Tests use a fake
  model (no jax, no network).

## Files

| file | role |
|------|------|
| `syllables/heuristics.py` | **deterministic** layer: numbers, emoji, lexicon, extras, spell-fix (off), tokenizer/normalize, tiered counter, trace entrypoint |
| `syllables/harness.py` | **public API**: composition, batched model, fuzziness gate, early-stop |
| `syllables/__init__.py` | re-exports the public API |
| `tests/test_heuristics.py` | deterministic-layer unit tests |
| `tests/test_harness.py` | API tests (model forced off / faked) |

`model.py` / `data.py` are consumed read-only by the harness/heuristics.
Section banners inside `heuristics.py`: `[numbers] [emoji] [lexicon]
[spell] [normalize] [extras] [count]`.

## Known sharp edges / open forks

- Bare digit runs (`911`) read as the cardinal "nine hundred eleven", not
  digit-by-digit "nine one one"; phone numbers (`555-1234`) fall through
  the ratio matcher imperfectly — deeper, genuinely ambiguous, not yet
  handled.
- Slash dates assume **US m/d/y** (unless the first field >12). 2-digit
  years assume 19xx for >68 else 20xx.
- `n/a`, `a/c`, `c/o`, Reddit `r/`/`u/` are *not* curated shorthand (only
  the unambiguous `w/`,`w/o`,`b/c` are) — they read literally with "slash".
- The letter tier triggers on any vowelless ≤6 token; an unusual vowelless
  real spelling would be spelled out. Fork: gate on all-caps too.
- Spell-fix uses cmudict membership, not word frequency — uniqueness at
  min edit distance is the only frequency proxy.
- The year band is 1000–2099; a 4-digit quantity in that range (e.g.
  `1500 people`) is read as a year. Accepted trade-off of "always prefer
  year". `24/7` matches the fraction reading ("twenty-four sevenths" = 5,
  coincidentally the same count as the idiom).
