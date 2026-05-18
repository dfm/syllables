"""Tests for the public harness API.

Hermetic: the neural model is either forced unavailable (deterministic
path) or replaced with a tiny fake (OOV-fill + fuzziness path). No jax,
no checkpoint, no network.

    uv run python -m unittest tests.test_harness
"""

from __future__ import annotations

import unittest

import numpy as np

from syllables import harness


def _model_off():
    harness._MDL, harness._MDL_TRIED = None, True


def _use(fake):
    harness._MDL, harness._MDL_TRIED = fake, True


class Base(unittest.TestCase):
    """Every test starts with the model OFF and the deterministic-pass
    cache cleared, so model-global / lru_cache state never leaks across
    classes (run order is otherwise alphabetical and fragile)."""

    def setUp(self):
        _model_off()
        harness._resolve_line.cache_clear()

    tearDown = setUp


class FakeModel:
    """`.probs(words)` -> [n, 8]; every row peaked at `peak` (count
    peak+1) with mass `p`, the rest uniform. Records batch sizes seen."""

    def __init__(self, peak=1, p=0.95):
        self.peak, self.p, self.sizes = peak, p, []

    def probs(self, words):
        self.sizes.append(len(words))
        row = np.full(8, (1 - self.p) / 7.0)
        row[self.peak] = self.p
        return np.tile(row, (len(words), 1))


class WordModel:
    """`.probs(words)` -> a *distinct* near-one-hot row per word, keyed by
    `word -> count`. A transposition / mis-scatter in the batched path
    yields the wrong word's count, which identical-row fakes can't catch."""

    def __init__(self, table):
        self.table = table

    def probs(self, words):
        out = np.full((len(words), 8), 0.001)
        for i, w in enumerate(words):
            out[i, self.table.get(w, 1) - 1] = 0.99
        return out / out.sum(1, keepdims=True)


class Gate(Base):
    def test_strict_vs_loose(self):
        peaked = np.array([0.02, 0.9, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0])
        flat = np.array([0.3, 0.28, 0.22, 0.1, 0.05, 0.03, 0.01, 0.01])
        self.assertEqual(harness._gate(peaked, 0.0), 2)  # confident -> ok
        self.assertIsNone(harness._gate(flat, 0.0))  # unsure -> abstain
        self.assertEqual(harness._gate(flat, 1.0), 1)  # fuzzy -> argmax
        # monotone: looser never abstains where stricter accepted
        self.assertEqual(harness._gate(peaked, 0.5), 2)


class ProbsFixed(Base):
    def test_padding_one_compile(self):
        fake = FakeModel()
        words = [f"w{i}" for i in range(harness.MODEL_BATCH + 5)]
        out = harness._probs_fixed(fake, words)
        self.assertEqual(out.shape[0], len(words))  # sliced back
        # every forward saw exactly MODEL_BATCH rows (single XLA shape)
        self.assertTrue(all(s == harness.MODEL_BATCH for s in fake.sizes))
        self.assertEqual(len(fake.sizes), 2)  # ceil(69/64)


class DeterministicPath(Base):
    def test_exact_line(self):
        self.assertEqual(harness.count_syllables("the quick brown fox"), 4)

    def test_batch_alignment_and_pure_fastpath(self):
        out = harness.count_batch(["the cat", "qwertzuiop", "a dog"])
        self.assertEqual(out, [2, None, 2])  # OOV line -> None, model off

    def test_oov_is_oov(self):  # self-validate the gibberish really is OOV
        t = harness.analyze("qwertzuiop").tokens[0]
        self.assertEqual(t.source, "abstain")

    def test_hard_abstain_even_if_short(self):
        # 'lol' is an ambiguous initialism -> hard abstain, never modelled
        self.assertIsNone(harness.count_syllables("lol", max_syllables=7))

    def test_max_syllables_early_stop(self):
        long = "cat " * 9  # 9 deterministic syllables
        self.assertEqual(harness.count_syllables(long.strip()), 9)
        self.assertIsNone(harness.count_syllables(long.strip(), max_syllables=7))

    def test_is_haiku_line(self):
        self.assertEqual(harness.is_haiku_line("cat cat cat cat cat"), 5)
        self.assertEqual(harness.is_haiku_line("cat cat cat cat cat cat cat"), 7)
        self.assertIsNone(harness.is_haiku_line("cat cat cat cat"))  # 4


class UrlPolicy(Base):
    def test_bare_domain_is_spoken_counted(self):
        self.assertEqual(harness.count_syllables("ebay.com"), 4)  # ebay·dot·com
        self.assertEqual(harness.count_syllables("check ebay.com please"), 6)

    def test_url_with_oov_label_abstains(self):
        # a bare domain whose label is genuinely OOV must NOT come back as
        # a confident undercount (the OOV sub-word silently counted 0) --
        # it violates the contract; abstain instead. (regression)
        for s in (
            "check out zxqwflarb.com",
            "visit zxqwflarb.com",
            "mail me at bob@zxqwflarb.com",
        ):
            self.assertIsNone(harness.count_syllables(s), s)
        # ...while a fully-resolved bare domain still counts.
        self.assertEqual(harness.count_syllables("ebay.com"), 4)

    def test_full_url_discards_line(self):
        for u in (
            "ebay.us/m/3WDD0T",
            "music.youtube.com/watch?v=-xVH",
            "music.apple.com/us/album/ble",
            "https://example.com",
            "listen here music.apple.com/us/album/x",
            # path/query => URL even when the TLD isn't allowlisted
            "youtu.be/4rWCsedkPAs?si=abc",
            "t.co/xY9",
            "foo.bar/baz",
        ):
            self.assertIsNone(harness.count_syllables(u), u)

    def test_unknown_tld_needs_a_path(self):
        # bare host with an unlisted TLD and NO path is not a URL (so it
        # won't be wrongly treated/discarded as one); numeric/abbrev
        # shapes never match.
        self.assertFalse(harness.heuristics._is_url("youtu.be"))
        self.assertFalse(harness.heuristics._is_url("3.14/2"))
        self.assertFalse(harness.heuristics._is_url("e.g."))
        self.assertTrue(harness.heuristics._is_url("youtu.be/x?si=1"))

    def test_is_full_url(self):
        for u in (
            "ebay.us/m/3",
            "x.com/p",
            "a.com?q=1",
            "https://x.com",
            "http://x.com/y",
        ):
            self.assertTrue(harness._is_full_url(u), u)
        for d in (
            "ebay.com",
            "www.example.org",
            "music.apple.com",
            "foo.co.uk",
            "x.com/",
        ):
            self.assertFalse(harness._is_full_url(d), d)


class Junk(Base):
    def test_junk_token_discards_line(self):
        # the 'macro:' firehose garbage: never modelled, line discarded
        self.assertIsNone(harness.count_syllables("macro: hhFDWkdAxDjkbJfXdOwv"))

    def test_looks_junk(self):
        for j in ("hhFDWkdAxDjkbJfXdOwv", "abcdefghijklmnop", "qwrtplkjhg"):
            self.assertTrue(harness._looks_junk(j), j)
        for ok in ("banana", "rangefinders", "skibidi", "yeet"):
            self.assertFalse(harness._looks_junk(ok), ok)


class ModelPath(Base):
    def test_oov_filled_by_model(self):
        _use(FakeModel(peak=1, p=0.95))  # count 2, confident
        # the(1) + quick(1) + <oov>=2  -> 4
        self.assertEqual(harness.count_syllables("the quick qwertzuiop"), 4)

    def test_fuzziness_gates_low_confidence(self):
        _use(FakeModel(peak=1, p=0.30))  # flat-ish -> unsure
        self.assertIsNone(harness.count_syllables("the qwertzuiop"))
        # same text, loosen fuzziness -> accept argmax (count 2)
        _use(FakeModel(peak=1, p=0.30))
        self.assertEqual(harness.count_syllables("the qwertzuiop", fuzziness=1.0), 3)

    def test_any_oov_abstain_kills_line(self):
        _use(FakeModel(peak=1, p=0.30))  # model abstains (strict)
        self.assertIsNone(harness.count_syllables("the qwertzuiop", max_syllables=7))


class ModelBatching(Base):
    """The batched OOV path: scatter-back alignment, input normalization,
    and the max_syllables early-stop when OOV is present."""

    def test_multiline_scatterback(self):
        # distinct count per word; repeated/interleaved OOV across lines.
        # A transposition in the flat<->line mapping mis-assigns counts.
        _use(WordModel({"florbnix": 4, "quambo": 6, "drindle": 2}))
        lines = ["florbnix", "the quambo", "drindle and florbnix", "cat"]
        # 4 | the(1)+quambo(6)=7 | drindle(2)+and(1)+florbnix(4)=7 | cat(1)
        self.assertEqual(harness.count_batch(lines), [4, 7, 7, 1])

    def test_elongated_oov_is_normalized_for_the_model(self):
        # the model is trained on clean a-z words; it must receive the
        # heuristics-collapsed form, not raw emphasis-spam. (regression)
        _use(WordModel({"blargh": 2}))  # only the collapsed key
        # omg(initialism=3) + blarghhhh->blargh(model=2) + today(cmu=2)
        self.assertEqual(harness.count_syllables("omg blarghhhh today"), 7)

    def test_max_syllables_skips_model_when_floor_blows_budget(self):
        fake = FakeModel(peak=1, p=0.99)
        _use(fake)
        # cat(1) + 3 OOV (floor 1 each) = 4 > 2  -> None, model untouched
        self.assertIsNone(
            harness.count_syllables("cat florbnix quambo drindle", max_syllables=2)
        )
        self.assertEqual(fake.sizes, [])  # never forwarded

    def test_max_syllables_rechecked_after_model(self):
        _use(WordModel({"florbnix": 8}))
        # the(1) + florbnix(8) = 9 > 3  -> None (post-model budget recheck)
        self.assertIsNone(harness.count_syllables("the florbnix", max_syllables=3))


class ApiSurface(Base):
    def test_reexport(self):
        import syllables

        self.assertEqual(
            set(syllables.__all__),
            {"count_syllables", "count_batch", "is_haiku_line", "analyze"},
        )


if __name__ == "__main__":
    unittest.main()
