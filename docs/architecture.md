# Architecture & motivation

A write-up of *why* this syllable counter is shaped the way it is, the
experiments that justified each choice, and the references it draws on.
For the specific shipped artifact (params, eval table, limitations) see
[`MODEL_CARD.md`](../MODEL_CARD.md); for the operational reference —
deterministic tiers, the harness API, per-feature normalization decisions
— see [`heuristics.md`](heuristics.md). This document is the canonical
record of the design reasoning and the experiments behind it.

## 1. The problem

Harvest haiku candidates (5–7–5) from the Bluesky firehose. Two
consequences shape everything:

- **A line is a conjunction.** 5-7-5 is right only if *every* word's
  count is right and they sum exactly. One silent ±1 ruins the poem, so
  per-word accuracy must be high and, where it can't be, the system must
  *know* it's unsure.
- **The interesting words are OOV.** Firehose text is saturated with
  slang, neologisms, and coinages (`rizz`, `skibidi`, `doomscroll`) that
  no pronunciation dictionary contains. Generalization to never-seen
  spellings *is* the task.

We want **conservative** behaviour — fail closed, but *keep* the
genuinely near-ambiguous cases (`fire` = 1 or 2), because those are the
fun haiku.

## 2. Why a tiny learned model

- **Dictionary-only (CMUdict)** is the gold standard for known words
  (syllables = stress-marked vowel phonemes) but fails exactly on the
  slang/neologisms we care about. It's a tier, not the answer.
- **Rule-based vowel-group counting** is fast but brittle (~70-85%); the
  model is best understood as the *learned, context-aware
  generalization* of that heuristic.
- **Full grapheme-to-phoneme** (ByT5 / CharsiuG2P, Zhu et al. 2022) is
  overkill: we need a *count*, not a pronunciation, so a model two
  orders of magnitude smaller suffices.
- **LLMs are the wrong tool.** PhonologyBench (Suvarna et al. 2024)
  found LLMs do not outperform humans and are unreliable at syllable
  counting; a dictionary-first pipeline with a small specialist model is
  the well-supported design.

The orthographic-syllabification literature converged on **character-level
sequence models**: Bartlett, Kondrak & Cherry (2008/2009, structured
SVM-HMM, ~98–99% word accuracy on CELEX), Trogkanis & Elkan (2010, CRF
for hyphenation, operating directly on characters), and Krantz, Dulin &
De Palma (2019, BiLSTM + CNN + CRF, language-agnostic). Their plateau is
~98% on clean dictionaries; the residual is the slang/OOV tail — our
regime.

## 3. The model: shape = the task

`char embed → 1-D conv (widths 2,3) → BiGRU → masked mean → MLP →
softmax over counts 1–8`. Pure JAX, ~7.4k params. The design mirrors what
a person does sounding out an unseen word:

- **Characters, not words** — generalization to OOV lives at the
  sub-word level; novel words are unseen *as words* but built of
  familiar pieces.
- **Conv (small widths)** = a translation-invariant detector of local
  vowel-group / spelling motifs (digraphs, silent-e context, `-le`):
  the learned, fuzzy version of "count the vowel groups."
- **BiGRU** = the part that *tallies* and applies order-dependent
  corrections (silent-e, `-ed`, morphology) using whole-word context
  from both directions. Counting is intrinsically a running tally;
  this is why pure max-pooled CNNs lose on long words.
- **Masked mean-pool**, not max — for a *counting* task you want the
  accumulated quantity, not "did this pattern occur." Padding is
  provably inert (zeroed pre-conv, carry-through GRU).
- **Softmax over counts**, not regression — yields a calibrated
  distribution the harness `fuzziness` gate turns into a keep/abstain.

The input contract is a **pinned 26-letter alphabet** (`a–z`; `encode()`
lowercases and drops everything else). Stripping punctuation is
syllable-count-neutral and makes train/serve a single total function.

## 4. Data

One reconciled table, keyed by the `a–z`-stripped word (collisions like
`it's`/`its` merged by count-union) — **169,069 words**:

| source | role | vs CMUdict |
|---|---|---|
| **CMUdict** | gold, phonological (stress-vowel count) | — |
| **kaikki** hyphenation | orthogonal orthographic vote; fixes WikiPron's hiatus bias | 93.5% |
| **WikiPron** | breadth/slang (+~43k words: `rizz`, `skibidi`, `yeet` …) | ~92% (systematic −1 hiatus undercount) |

Reliable = CMUdict + kaikki-hyph; WikiPron supplies coverage but never
overrides a reliable source (its lone disagreement is the hiatus
artifact, not signal). Genuine ambiguity (`fire`→{1,2}) is ~1.4% and is
trained with a **soft-uniform valid-set loss** so the model keeps the
spread instead of collapsing it. kaikki IPA-with-dots was evaluated and
dropped (61% vs CMU — too noisy).

Also evaluated and **not** used (so future work need not re-investigate):
**CELEX** (highest-quality hand-annotated, but LDC-licensed/paid);
**Bartlett-annotated CMUdict** (per-character boundary labels — only
relevant if we ever switch to the boundary/sequence-labeling framing,
which we did not); **Moby Hyphenator II** (public-domain hyphenation,
older and idiosyncratic, superseded by kaikki); **NETtalk** (small,
largely a CMUdict subset).

## 5. What the experiments showed

(Numbers were measured under evolving splits/labels as the project
matured; the *conclusions* are stable. The shipped-artifact eval table is
in `MODEL_CARD.md`.)

**Architecture bake-off** (controlled, matched params) selected
CNN→BiGRU and showed pooling + the recurrence's counting bias matter:

| model | params | exact | tail (≥4 syl) |
|---|---|---|---|
| CNN + max-pool | 30k | 0.943 | 0.865 |
| CNN + mean⊕max | 42k | 0.951 | 0.890 |
| BiGRU | 29k | 0.954 | 0.926 |
| **CNN→BiGRU** | 44k | **0.960** | **0.926** |

Transformer was deprioritized (short ≤24-char sequences, weak
inductive-bias match, no pretraining payoff).

**Capacity is slack, not the engine.** A shrink ladder showed a **20×
parameter cut costs only ~1.7pp** with *no cliff*:

| size | params | val exact |
|---|---|---|
| s44k | 43,920 | 0.9649 |
| s22k | 21,528 | 0.9599 |
| **s7k** | **7,372** | **0.9549** |
| s4k | 4,382 | 0.9516 |
| s2k | 2,176 | 0.9481 |

(Param counts are exact, recomputed from `init_params`; the val-exact
column is illustrative — see the §5 preamble.) This is evidence the
*inductive bias*, not memorized capacity, does the work — so we ship the
tiny **s7k** (~7.4k params, 36 KB); no quantization is needed.

**It genuinely generalizes.** 3-seed error bars gave per-seed std
**0.0007** (so sub-1pp deltas above are real signal). A lemma-disjoint
**hard split** (no morphological leakage) scored **0.927 ± 0.0007** vs
**~0.930** random — removing leakage costs ~0.3pp, i.e. the model is not
coasting on memorized inflections.

**The honest OOV number.** Trained dictionary-only, tested on ~37k
WikiPron-only words: raw exact **0.823**, within-1 **0.99**. But error
analysis showed **~82% of "errors" are `pred = label + 1`** — the known
WikiPron −1 *hiatus label* noise (`gonial`, `abulia`: model right, label
wrong). True OOV generalization is substantially higher; the metric was
label-noise-dominated, not model-limited.

**Calibration is a known limitation.** Soft-uniform loss keeps mass on
valid counts (`amb_mass` ≈ 0.99) but only partially spreads it
(`cover@0.9` ≈ 0.5, flat across training). Conclusion: a single tiny
softmax won't be perfectly calibrated, and that's acceptable because
conservatism is **system-level**, not the model alone.

## 6. Where the model sits

The public `harness` layer owns *integration and policy* and composes two
lower layers it does not modify:

```
lines → heuristics.analyze()   deterministic only: normalize
                               (numbers/symbols/elongation/emoji),
                               lexicon lookup, spell-out; resolves a
                               token or marks it OOV in an audit trace
      → OOV tokens, batched across the whole call, → model.Model
                               (ONE fixed-size padded forward, compiles
                               once; the firehose-justified design)
      → fuzziness gate         single abstention knob: 0 = strict, 1 =
                               best-guess (relaxes the softmax
                               confidence/margin requirement)
      → count_batch(lines) -> list[int | None]   None = discard
```

The model is the **OOV specialist**, not a dictionary replacement.
Known and known-ambiguous words are resolved deterministically; the
model only has to be honest on novel words; the contract is
precision-over-recall — a confident exact `int`, or `None` (the caller
discards). `is_haiku_line` returns 5/7 only when *confidently* exact.
The shipped model is a single self-contained ~36 KB artifact loadable
with only `jax`+`numpy` (`from syllables import model;
model.Model.load()`); `model.Model` exposes only the raw softmax
(`.logits`/`.probs`) — the abstention policy is the harness, not a
second policy on the model.

Operational specifics — exact gate thresholds, `MODEL_BATCH`, early-stop,
URL/junk policy, the `count_batch`/`is_haiku_line` signatures — live in
[`heuristics.md` → "The harness (public API)"](heuristics.md), the
canonical operational doc; deliberately not restated here.

## References

The works the design draws on:

1. Bartlett, Kondrak & Cherry (2008). *Automatic Syllabification with
   Structured SVMs for Letter-to-Phoneme Conversion.*
2. Bartlett, Kondrak & Cherry (2009). *On the Syllabification of
   Phonemes.*
3. Trogkanis & Elkan (2010). *Conditional Random Fields for Word
   Hyphenation.*
4. Krantz, Dulin & De Palma (2019). *Language-Agnostic Syllabification
   with Neural Sequence Labeling.*
5. Suvarna et al. (2024). *PhonologyBench* — LLM phonological evaluation.
6. Zhu et al. (2022). *CharsiuG2P* — ByT5 multilingual grapheme-to-phoneme.

Data sources: CMU Pronouncing Dictionary (CMUdict); WikiPron
(CUNY-CL); Wiktextract / kaikki.org (Ylönen) — Wiktionary-derived,
CC-BY-SA.
