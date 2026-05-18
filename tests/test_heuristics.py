"""Hermetic tests for the heuristics layer.

Run:  uv run python -m unittest discover -s tests

`heuristics.py` is deterministic and has no model tier (the neural net
lives one layer up, in `harness.py`), so these tests need no stub: they
assert the *heuristic* behaviour directly -- cmudict / curated tiers /
normalization. The model path is covered by `tests/test_harness.py`.
"""

from __future__ import annotations

import unittest

from syllables import heuristics as h


def total(text):
    return h.analyze(text)


class Numbers(unittest.TestCase):
    def test_cardinal(self):
        self.assertEqual(h.cardinal(0), ["zero"])
        self.assertEqual(h.cardinal(100), ["one", "hundred"])
        self.assertEqual(
            h.cardinal(1234),
            ["one", "thousand", "two", "hundred", "thirty", "four"],
        )
        self.assertEqual(h.cardinal(-5), ["negative", "five"])
        self.assertEqual(h.cardinal(1_000_000), ["one", "million"])
        # huge digit strings (crypto/IDs) exceed short-scale words ->
        # read digit-by-digit instead of crashing (real firehose bug)
        big = int("1" + "0" * 30)
        self.assertEqual(h.cardinal(big)[:2], ["one", "zero"])
        self.assertEqual(len(h.cardinal(big)), 31)

    def test_year(self):
        self.assertEqual(h.year(1999), ["nineteen", "ninety", "nine"])
        self.assertEqual(h.year(2000), ["two", "thousand"])
        self.assertEqual(h.year(2007), ["two", "thousand", "seven"])
        self.assertEqual(h.year(1905), ["nineteen", "oh", "five"])
        self.assertEqual(h.year(1900), ["nineteen", "hundred"])

    def test_ordinal(self):
        self.assertEqual(h.ordinal(1), ["first"])
        self.assertEqual(h.ordinal(21), ["twenty", "first"])
        self.assertEqual(h.ordinal(42), ["forty", "second"])

    def test_expand_token(self):
        self.assertEqual(h.expand("5"), ["five"])
        self.assertEqual(
            h.expand("$19.99"),
            ["nineteen", "dollars", "and", "ninety", "nine", "cents"],
        )
        self.assertEqual(h.expand("3.14"), ["three", "point", "one", "four"])
        # 4-digit year band -> ALWAYS the year reading, never spelled out.
        self.assertEqual(h.expand("1999"), ["nineteen", "ninety", "nine"])
        # a comma disqualifies the year reading -> plain cardinal.
        self.assertEqual(
            h.expand("1,234"),
            ["one", "thousand", "two", "hundred", "thirty", "four"],
        )
        self.assertEqual(h.expand("nope"), [])


class Elongation(unittest.TestCase):
    def test_collapse(self):
        self.assertEqual(h.collapse_elongation("noooooo")[0], "no")
        self.assertEqual(h.collapse_elongation("sooo")[0], "so")
        self.assertEqual(h.collapse_elongation("ahhhhh")[0], "ah")
        self.assertIn("cool", h.collapse_elongation("cooool"))
        self.assertEqual(h.collapse_elongation("hi"), ["hi"])

    def test_pipeline_elongation(self):
        r = total("noooooo")
        self.assertEqual(r.total, 1)
        self.assertTrue(r.confident)


class Emoji(unittest.TestCase):
    def test_detect_and_split(self):
        self.assertTrue(h.is_emoji_char("😂"))
        self.assertFalse(h.is_emoji_char("a"))
        self.assertEqual(
            h.split_emoji("a😂b"),
            [("a", False), ("😂", True), ("b", False)],
        )

    def test_emoji_is_silent_not_uncertain(self):
        r = total("great😂job")
        self.assertEqual(r.total, total("great job").total)
        self.assertTrue(r.confident)


class Spell(unittest.TestCase):
    def setUp(self):
        self.fx = h.SpellFixer(
            ["definitely", "brown", "born", "quick", "the", "absolutely"]
        )

    def test_corrects_unique(self):
        self.assertEqual(self.fx.correct("definately"), ("definitely", 1))

    def test_short_token_abstains(self):
        self.assertIsNone(self.fx.correct("teh"))  # < 4 chars

    def test_ambiguous_abstains(self):
        self.assertIsNone(self.fx.correct("borwn"))  # brown vs born


class Counting(unittest.TestCase):
    def test_plain_words(self):
        r = total("the quick brown fox")
        self.assertEqual(r.total, 4)
        self.assertTrue(r.confident)

    def test_numbers_exact(self):
        self.assertEqual(total("5 cats").total, 2)
        self.assertEqual(total("$19.99").total, 9)
        self.assertEqual(total("3.14%").total, 6)

    def test_slang_and_initialism(self):
        self.assertEqual(total("bussin").total, 2)
        self.assertEqual(total("idk").total, 3)
        self.assertFalse(total("lol").confident)  # ambiguous -> abstain

    def test_year_uses_year_reading(self):
        # "in 1999" -> in(1) nineteen(2) ninety(2) nine(1) = 6, confident
        r = total("in 1999")
        self.assertTrue(r.confident)
        self.assertEqual(r.total, 6)

    def test_oov_abstains(self):
        r = total("zyqof")  # has vowels, no close cmudict word
        self.assertFalse(r.confident)
        self.assertEqual(r.n_uncertain, 1)

    def test_letters_for_vowelless(self):
        self.assertEqual(total("xkcd").total, 4)

    def test_compound_split(self):
        self.assertEqual(total("well-known").total, 2)

    def test_mention_and_hashtag(self):
        self.assertEqual(total("@bob_smith").total, 3)
        self.assertEqual(total("#ThrowbackThursday").total, 4)

    def test_url_spoken(self):
        self.assertEqual(total("https://www.example.com").total, 15)


class Extras(unittest.TestCase):
    def test_emoticons_silent(self):
        r = total("hi <3 xD :) T_T :-( </3")
        self.assertEqual(r.total, total("hi").total)
        self.assertTrue(r.confident)

    def test_currency_signs(self):
        # £/€/¥ parallel the $ path: plural/singular, sub-unit, and the
        # magnitude suffix MUST multiply (not be read as a literal letter
        # -- that was a silent confident-wrong undercount). (regression)
        self.assertEqual(h.expand("£5"), ["five", "pounds"])
        self.assertEqual(h.expand("£1"), ["one", "pound"])
        self.assertEqual(h.expand("€20"), ["twenty", "euros"])
        self.assertEqual(
            h.expand("£5.99"),
            ["five", "pounds", "and", "ninety", "nine", "pence"],
        )
        self.assertEqual(
            h.expand("€5.99"),
            ["five", "euros", "and", "ninety", "nine", "cents"],
        )
        self.assertEqual(h.expand("£0.01"), ["zero", "pounds", "and", "one", "penny"])
        self.assertEqual(
            h.expand("£1.2B"), ["one", "point", "two", "billion", "pounds"]
        )
        self.assertEqual(h.expand("€3M"), ["three", "million", "euros"])
        # yen has no spoken sub-unit -> decimal reads "point x y"
        self.assertEqual(h.expand("¥500"), ["five", "hundred", "yen"])
        self.assertEqual(h.expand("¥1.5"), ["one", "point", "five", "yen"])
        # sign composes with currency + sub-unit
        self.assertEqual(
            h.expand("-£2.50"),
            ["negative", "two", "pounds", "and", "fifty", "pence"],
        )

    def test_magnitude_suffix(self):
        self.assertEqual(
            h.expand("$1.2B"),
            ["one", "point", "two", "billion", "dollars"],
        )
        # bare number keeps the letter literal (no multiply)
        self.assertEqual(h.expand("20k"), ["twenty", "k"])
        self.assertEqual(h.expand("100K"), ["one", "hundred", "k"])
        self.assertEqual(h.expand("4K"), ["four", "k"])

    def test_shorthand(self):
        self.assertEqual(total("w/o").total, 2)  # without
        self.assertEqual(total("b/c").total, 2)  # because
        self.assertTrue(total("w/ b/c").confident)

    def test_allcaps_acronym_spelled(self):
        # cmudict has IBM/CEO already; an acronym it LACKS still spells
        # out via the all-caps tier, not the model.
        self.assertEqual(total("IBM").total, 3)  # cmudict
        r = total("NDA")
        self.assertEqual(r.total, 3)  # n-d-a, letters
        self.assertTrue(r.confident)
        self.assertEqual(r.tokens[0].source, "letters")
        # all-caps real words / said-as-word acronyms are NOT spelled
        self.assertEqual(total("STOP").total, 1)  # cmudict word
        self.assertEqual(total("NASA").total, 2)  # acronym, as word
        # dotted Latin abbrevs stay spelled (e.g. -> "ee jee" = 2)
        self.assertEqual(total("e.g.").total, 2)

    def test_apostrophe_dropped_contractions(self):
        for w, n in [
            ("dont", 1),
            ("ive", 1),
            ("thats", 1),
            ("theyre", 1),
            ("doesnt", 2),
            ("youre", 1),
            ("im", 1),
            ("wouldnt", 2),
        ]:
            r = total(w)
            self.assertTrue(r.confident, w)
            self.assertEqual(r.total, n, w)
        # collision-safe: real-word twins keep the (identical) count
        self.assertEqual(total("cant were its lets").total, 4)

    def test_apostrophe_bearing_contractions(self):
        # the ASCII-apostrophe form must hit cmudict directly...
        for w, n in [
            ("I'm", 1),
            ("don't", 1),
            ("he's", 1),
            ("you're", 1),
            ("wouldn't", 2),
        ]:
            r = total(w)
            self.assertTrue(r.confident, w)
            self.assertEqual(r.total, n, w)
        # ...and the iOS/macOS smart quote (U+2019) must fold to it first
        # (NFKC does NOT do this) -- else every smart-quoted contraction
        # would silently miss cmudict and abstain.
        for smart, n in [("I’m", 1), ("don’t", 1), ("you’re", 1)]:
            r = total(smart)
            self.assertTrue(r.confident, smart)
            self.assertEqual(r.total, n, smart)

    def test_roman_numerals(self):
        # small (<= ROMAN_MAX) -> default cardinal reading, confident but
        # marked approx (the cardinal/ordinal choice is a guess).
        for s, n in [("World War II", 3),   # world war two
                     ("Henry VIII", 3),     # hen-ry eight
                     ("Star Wars IV", 3),   # star wars four
                     ("Final Fantasy VII", 7)]:  # fi-nal fan-ta-sy se-ven
            r = total(s)
            self.assertTrue(r.confident, s)
            self.assertEqual(r.total, n, s)
        rom = next(c for c in total("World War II").tokens if c.raw == "II")
        self.assertEqual(rom.confidence, "approx")
        self.assertEqual(rom.count, 1)                  # "two"
        # larger numerals stay an abstention (ambiguity compounds)
        self.assertFalse(total("Super Bowl LVIII").confident)  # 58
        self.assertFalse(total("section XV").confident)        # 15
        self.assertFalse(total("year MMXXIV").confident)       # 2024
        # a real word that happens to be valid Roman is NOT treated so
        self.assertEqual(total("MIX").total, 1)
        self.assertTrue(total("MIX").confident)

    def test_diacritics_fold_to_cmudict(self):
        r = total("café")
        self.assertEqual(r.total, 2)
        self.assertTrue(r.confident)  # exact via folded cmudict
        self.assertTrue(total("naïve").confident)

    def test_url_recognition(self):
        self.assertTrue(h._is_url("bit.ly/abc"))
        self.assertTrue(h._is_url("foo.co.uk"))
        self.assertTrue(h._is_url("https://x"))
        self.assertFalse(h._is_url("hello.world"))  # 'world' not a TLD
        self.assertTrue(total("bit.ly/abc").confident)

    def test_email_dots_voiced(self):
        toks = h.tokenize("a.b@x.io")
        url = next(t for t in toks if t.kind == "url")
        self.assertEqual(url.words.count("dot"), 2)  # local + host dots

    def test_dates_times_fractions(self):
        self.assertEqual(
            h._date_words(5, 17, 2026),
            ["may", "seventeenth", "twenty", "twenty", "six"],
        )
        self.assertEqual(h._time_words(10, 45, "p"), ["ten", "forty", "five", "p", "m"])
        self.assertEqual(h._time_words(9, 0, None), ["nine", "o", "clock"])
        self.assertEqual(h._fraction_words(3, 4), ["three", "quarters"])
        self.assertEqual(h._fraction_words(1, 2), ["one", "half"])
        self.assertEqual(h._fraction_words(2, 3), ["two", "thirds"])
        self.assertEqual(total("½").total, 2)  # one half

    def test_interpreted_readings_marked_approx(self):
        # dates/times/fractions are deterministic but a guess -> not
        # "exact", though still confident (not abstained).
        cs = {c.raw: c for c in h.analyze("on 5/17/2026").tokens}
        self.assertEqual(cs["5/17/2026"].confidence, "approx")


if __name__ == "__main__":
    unittest.main()
