# Model card — `syllables/weights/model.npz`

A tiny char-level syllable counter shipped inside the package. It is the
OOV specialist behind the public `syllables` API; the model itself only
exposes the raw softmax, with the abstention policy living one layer up:

```python
from syllables import count_syllables   # the public, policy-applied API
count_syllables("skibidi")              # -> 3

from syllables import model             # the raw primitive, for probes
m = model.Model.load()                  # zero-config: packaged weights
m.probs("skibidi")                      # -> [[p1 … p8]] softmax row
```

## Architecture

char embedding → 1-D conv (widths 2, 3) → BiGRU → masked mean → MLP →
softmax over counts 1–8. Pure JAX (no neural-net library). Input contract
is the pinned 26-letter alphabet `a–z` (`encode()` lowercases and drops
everything else); padding is provably inert.

- **caps** `s7k`: embed 12, conv_filters 16, gru_hidden 16, head 32
- **7,372 params**, single self-validating `.npz` (~36 KB), no sidecar
- Class index = count − 1; `Model.probs` is the only output (softmax) —
  the conservative keep/abstain decision is the harness `fuzziness` gate

## Training

- **Data:** all 169,069 words of the reconciled table — CMUdict (gold,
  phonological) ∪ kaikki hyphenation (orthogonal, fixes WikiPron hiatus) ∪
  WikiPron (breadth/slang), keyed by `a–z` with collisions merged.
- **Objective:** soft-uniform cross-entropy over the *valid-count set*
  (one-hot for ~98% single-valid words; a spread for genuinely ambiguous).
- **Optim:** AdamW, warmup→cosine LR, dropout 0.05, 100 epochs, seed 0.
- **No holdout** — this is the deployable run; *every* word trains
  (deliberate: don't split out the hard examples).
- **Date:** 2026-05-17.

### Source snapshots (NOT pinned in-repo — numbers drift if upstream changes)

- CMUdict — `cmusphinx/cmudict` `cmudict.dict`, fetched 2026-05-17
- WikiPron — `CUNY-CL/wikipron` `eng_latn_us_broad.tsv`, 2026-05-17
- kaikki — English Wiktionary extract, streamed 2026-05-17

**Licensing:** WikiPron/kaikki are Wiktionary-derived → CC-BY-SA; attribute
Wiktionary. Trained weights are generally not treated as derivative works,
but attribution is courteous. CMUdict is BSD-style.

## Evaluation

Measured on *held-out* runs before this all-data ship run (which itself
has no holdout by design):

| split | exact | notes |
|---|---|---|
| random | ≈0.930 | in-distribution |
| hard (lemma-disjoint) | 0.927 ± 0.0007 (3 seeds) | ≈ random ⇒ genuine generalization, not memorization |
| source_oov (train dict-only, test ~37k WikiPron-only) | 0.823 raw | **but ~82% of "errors" are the WikiPron −1 hiatus *label* noise** ⇒ true OOV generalization is substantially higher |

within-1 ≈ 0.99 everywhere. Calibration: `cover@0.9` ≈ 0.5 on ambiguous
words — a **known limitation**. System conservatism is not the model's
solo softmax; it comes from the surrounding pipeline (deterministic
CMUdict/kaikki lookups, the heuristics tiers + abstention, and the
harness `fuzziness` gate over the model's softmax).

## Intended use & limitations

Per-word syllable counting as one tier of the Bluesky-firehose
haiku-candidate harvester. Conservative, fail-closed.

Known failure modes (mostly handled upstream, not by this model):

- **Elongation** (`noooo`, `aaawesome`) inflates counts → the
  normalization layer collapses runs before the model.
- **Syllabic endings on exotic Greek/Latin/loan morphology** undercount
  by ~1 (`cyclopes`, `menarche`) — ~2.4% of OOV.
- **Long/rare/proper-noun tail** (`schopenhauerian`) — ~1%.
- **Initialisms** (`idk`, `omg`) read as words — handled by a separate
  heuristics tier with letter-spelling, not the NN.

## Reproduce

```sh
uv run python scripts/train.py --config s7k --split all --epochs 100
```

(Requires the three source files in `data/`; results will differ if the
upstream lexica have changed since 2026-05-17.)
