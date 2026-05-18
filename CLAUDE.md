# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Conservative English syllable counting for messy real-world text (Bluesky
firehose: slang, emoji, numbers, URLs). It returns an exact count **or
abstains (`None`)** — it never silently guesses. The downstream goal is
harvesting haiku candidates (5-7-5), so per-word precision and honest
abstention matter more than coverage.

## Commands

Everything runs through `uv` (the package is installed via the hatchling
build-system, so `from syllables import …` works from any script — **do
not add `sys.path` manipulation**).

```sh
uv run python -m unittest discover -s tests          # all tests
uv run python -m unittest tests.test_heuristics.CLASS.test_x   # one test
uv run python -m syllables.heuristics "text…"        # per-token audit trace
uv run python scripts/explain.py ["text…" | -]       # full cross-layer pipeline tour
uv run python scripts/explain.py --firehose          # same, on live Bluesky (opt. atproto)
uv run python scripts/infer.py                       # slang probe (packaged weights)

# Train the deployable model (writes the packaged artifact, no holdout):
uv run python scripts/train.py --config s7k --split all --epochs 100
# Experiments (write to results/, gitignored):
uv run python scripts/train.py --split hard|source_oov|random [--tune|--shrink|--ensemble N]
uv run python scripts/train.py --summarize results/<run>.jsonl
uv run python scripts/plot_curves.py results/<run>.jsonl
uv run python scripts/fetch_kaikki.py                # stream 3GB -> data/kaikki_counts.tsv
```

Source lexica (`data/`) and experiment outputs (`results/`) are
gitignored. The shipped model `syllables/weights/model.npz` **is**
tracked and ships in the wheel.

## Architecture (the big picture)

Three layers with a **hard boundary** — keep them decoupled:

- **`syllables/model.py`** — the neural model *and* the serving contract.
  Tiny char model (embed → conv widths 2,3 → BiGRU → masked-mean → MLP →
  softmax over counts 1–8), **pure JAX, no neural-net framework**. Owns
  the pinned input contract (`ALPHABET = a–z`, `encode`, `MAX_LEN`,
  `NUM_CLASSES`) and the artifact format: a single self-validating
  `.npz` (`p*` leaves + a `meta` JSON; no pickle, no sidecar) that loads
  loudly on any train/serve skew. `Model.load()` is zero-config (→
  `syllables/weights/model.npz`); `.logits()`/`.probs()` are the only
  primitives (raw softmax) — there is **no policy on the model**, the
  single abstention policy is the harness `fuzziness` gate. Serving
  needs only `jax`+`numpy` and never imports `data.py`.
- **`syllables/heuristics.py`** — conservative *deterministic* layer:
  NFKC/emoji/number/URL/elongation normalization, curated
  slang/initialisms, reconciled-lexicon lookup, spell-out. Produces an
  auditable per-token trace; **no model tier lives here**; abstains
  rather than guess.
- **`syllables/harness.py`** — the **public API**
  (`count_syllables`, `count_batch`, `is_haiku_line`, `analyze`,
  re-exported from `__init__`). Owns *integration and policy only*; it
  composes the other two without modifying them: `heuristics.analyze()`
  resolves tokens or marks them OOV; OOV tokens batch into **one
  fixed-size `Model` forward** (compiles once); a single `fuzziness`
  knob (0 = strict … 1 = best-guess) is the entire abstention policy;
  contract is `count_batch(lines) -> list[int | None]` (`None` =
  discard). The model is the **OOV specialist, not a dictionary
  replacement** — conservatism is system-level, not the model's solo
  softmax.

- **`syllables/data.py`** — training-side only: parse CMUdict / WikiPron
  / kaikki, reconcile into one table (`build_sources`), encode + split +
  `load`. It *imports the input contract from `model.py`* (dependency
  inverted on purpose — don't reintroduce a `data.py` dependency into
  the serving path).

**Multi-source reconciliation (non-obvious, evidence-grounded — see
`docs/`):** CMUdict = gold (stress-vowel count); kaikki *hyphenation* =
orthogonal reliable source (fixes WikiPron's systematic −1 hiatus
undercount); WikiPron = breadth/slang only and **never overrides a
reliable source** (its lone disagreement is the hiatus artifact, not
signal); kaikki IPA-with-dots is deliberately **unused** (too noisy).
Words are stripped to `a–z` and collisions merged by count-union (~169k
words). Genuine ambiguity (~1.4%, e.g. `fire`→{1,2}) is trained with a
soft-uniform valid-set loss so the spread is *kept*, not collapsed.

**Splits** (`scripts/train.py --split`): `random` (in-dist), `hard`
(lemma-disjoint — the OOV-generalization proxy), `source_oov` (train
dict-only, test WikiPron-only), `all` (no holdout — the ship model;
disables eval/early-stop, saves final).

## Conventions & gotchas

- **Pure JAX for the model — no Flax/NN library.** `optax` is a *dev*
  dependency (training only); runtime deps are just `jax` + `numpy`.
- **The `a–z` alphabet is pinned and is the train==serve contract.**
  Changing it invalidates checkpoints (`model.load` asserts); checkpoints
  are self-validating and fail loudly rather than skew silently.
- **Don't over-read raw OOV exact-match.** The `source_oov` ~0.82 is
  *label-noise-dominated* (~82% of "errors" are the known WikiPron −1
  hiatus, model often correct). Hard-split ≈ random (~0.93) → the model
  genuinely generalizes, not memorizes. Seed std ≈0.0007 (sub-1pp deltas
  are real signal).
- **Training data is not pinned in-repo** (lexica gitignored; kaikki
  regenerated upstream) → numbers drift if upstream changes. Provenance
  lives in the artifact `meta` and `MODEL_CARD.md`.
- Docs: `docs/architecture.md` = model + experiment rationale, the
  literature, and datasets considered/rejected (the canonical design
  record — the original research log `CHAT.md` was folded in here and
  deleted); `docs/heuristics.md` = deterministic-layer design;
  `MODEL_CARD.md` = the shipped artifact.
