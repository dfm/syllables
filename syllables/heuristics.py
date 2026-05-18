"""Conservative deterministic heuristics + normalization layer.

Turns raw internet text into a reliable syllable count *or an abstention*
-- no neural model here. It reads the reconciled reference lexicon via
`data` (read-only); the OOV tokens it abstains on are handed to the model
by the `harness` layer, which owns all model integration and policy.

    raw text
      -> tokenize()      text  -> [Token]      (NFKC, elongation, numbers,
      -> count_token()   Token -> Counted       URLs/@/#, emoji, symbols)
      -> analyze()       text  -> Result        (tiered count + abstention)

Run it:

    uv run python -m syllables.heuristics "I have 5 cats and $19.99 lol 😂"
    uv run python -m syllables.heuristics            # built-in demo set
    uv run python -m unittest discover -s tests      # hermetic tests

Contract: precision over recall. Every tier that fires is trusted; the
moment none fires we abstain rather than guess. `analyze()` reports a
best-effort total *and* a `confident` flag (true only if every non-silent
unit resolved through a trusted tier). The per-unit trace makes every
count auditable.

Sections below: [numbers] [emoji] [lexicon] [spell] [normalize]
[extras] [count].
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from . import data

# ===========================================================================
# [numbers]  numeric tokens -> spoken words (all land in cmudict => exact)
# ===========================================================================

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]
_ORD_ONES = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def _under_1000(n: int) -> list[str]:
    w: list[str] = []
    if n >= 100:
        w += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        w.append(_TENS[n // 10])
        n %= 10
    if n:
        w.append(_ONES[n])
    return w


def cardinal(n: int) -> list[str]:
    """123456 -> [one hundred twenty three thousand four hundred fifty six]."""
    if n < 0:
        return ["negative", *cardinal(-n)]
    if n < 20:
        return [_ONES[n]]
    chunks: list[int] = []
    m = n
    while m:
        chunks.append(m % 1000)
        m //= 1000
    if len(chunks) > len(_SCALES):
        # beyond quadrillion has no short-scale word -- and a 20-digit
        # crypto/ID string is read digit-by-digit anyway, not as a count.
        return [_ONES[int(d)] for d in str(n)]
    words: list[str] = []
    for i in range(len(chunks) - 1, -1, -1):
        if not chunks[i]:
            continue
        words += _under_1000(chunks[i])
        if _SCALES[i]:
            words.append(_SCALES[i])
    return words or ["zero"]


def ordinal(n: int) -> list[str]:
    """21 -> [twenty first]; transforms only the final word."""
    words = cardinal(n)
    last = words[-1]
    if last in _ORD_ONES:
        words[-1] = _ORD_ONES[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "tieth"  # twenty -> twentieth
    else:
        words[-1] = last + "th"  # hundred -> hundredth, six -> sixth
    return words


def year(n: int) -> list[str]:
    """Conventional spoken year: 1999 -> nineteen ninety nine, 2007 ->
    two thousand seven, 1905 -> nineteen oh five, 2000 -> two thousand."""
    if not (1000 <= n <= 2099):
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        # round thousand -> "two thousand"; else "nineteen hundred"
        return cardinal(n) if n % 1000 == 0 else cardinal(hi) + ["hundred"]
    if 2000 <= n < 2010:
        return ["two", "thousand", _ONES[lo]]
    head = cardinal(hi)
    tail = ["oh", _ONES[lo]] if lo < 10 else _under_1000(lo)
    return head + tail


def _digit_tail(frac: str) -> list[str]:
    return ["point", *(_ONES[int(c)] for c in frac if c.isdigit())]


_MAG = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}

# (unit sg, unit pl, sub-unit sg, sub-unit pl). Yen has no spoken
# sub-unit, so a yen decimal reads "point x y", not "... and N sen".
_CURRENCY = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "", ""),
}


def expand(token: str) -> list[str]:
    """Expand one numeric token to spoken words; [] if not numeric.

    Handles a leading currency sign ($/£/€/¥), trailing %, commas, a
    decimal point, a sign, ordinal suffixes, and magnitude suffixes
    (k/m/b/t). For a bare 4-digit number in the plausible-year band
    (1000-2099) we *always* use the year reading ("nineteen ninety
    nine"), never the spelled-out cardinal.

    Magnitude rule: a suffix multiplies *only* with a currency sign
    ("$1.2B" -> one point two billion dollars). A bare number keeps the
    letter literal ("20k" -> twenty k, "4K" -> four k) -- the spoken form
    people actually use, and it sidesteps the 20k-views vs 4K-TV ambiguity.
    """
    t = token.strip()
    neg = t[:1] in "-−"
    if neg:
        t = t[1:]
    cur = _CURRENCY.get(t[:1])
    money = cur is not None
    if money:
        t = t[1:]
    pct = t[-1:] == "%"
    if pct:
        t = t[:-1]

    mag = lit = ""
    suf, core = t[-1:].lower(), t[:-1]
    if suf in _MAG and core.replace(",", "").replace(".", "", 1).isdigit():
        if money:  # "$1.2B" -> ... billion dollars (multiplier)
            mag, t = _MAG[suf], core
        else:  # "20k" -> twenty k (literal letter)
            lit, t = suf, core

    had_comma = "," in t

    ord_m = ""
    for suf in ("st", "nd", "rd", "th"):
        if t.lower().endswith(suf) and t[: -len(suf)].isdigit():
            ord_m, t = suf, t[: -len(suf)]
            break

    t = t.replace(",", "")
    if not t or not t.replace(".", "", 1).isdigit():
        return []

    int_part, _, frac = t.partition(".")
    n = int(int_part or "0")

    if ord_m:
        out = (["negative"] if neg else []) + ordinal(n)
        return out + (["percent"] if pct else [])

    # Bare 4-digit year-band integer -> year reading, always preferred.
    if (
        not frac
        and not money
        and not pct
        and not neg
        and not had_comma
        and not mag
        and not lit
        and int_part.isdigit()
        and len(int_part) == 4
        and 1000 <= n <= 2099
    ):
        return year(n)

    words = cardinal(n)
    if frac:
        words = words + _digit_tail(frac)
    if mag:
        words = words + [mag]
    if lit:
        words = words + [lit]
    if money:
        unit_sg, unit_pl, sub_sg, sub_pl = cur
        unit = unit_sg if n == 1 else unit_pl
        if mag:  # "$1.2B" -> one point two billion dollars
            words = words + [unit_pl]
        elif sub_sg and frac and len(frac) <= 2:
            cents = int((frac + "00")[:2])
            words = cardinal(n) + [unit]
            if cents:
                words += ["and", *cardinal(cents), sub_sg if cents == 1 else sub_pl]
        else:
            words = words + [unit]
    if neg:
        words = ["negative", *words]
    if pct:
        words = words + ["percent"]
    return words


# ===========================================================================
# [emoji]  detect / segment / name -- count contribution is always 0
# ===========================================================================
#
# When a human reads a tweet aloud they do not pronounce emoji. So an emoji
# contributes ZERO syllables. We detect them only to (a) split them off
# glued-on words ("great😂job") and (b) name them in the trace.

_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x1F1E6, 0x1F1FF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0x1F000, 0x1F0FF),
    (0xFE00, 0xFE0F),
    (0x1F3FB, 0x1F3FF),
)
_ZWJ = 0x200D
EMOJI_NAMES: dict[str, str] = {
    "😂": "tears of joy",
    "🤣": "rolling laughing",
    "😭": "loudly crying",
    "❤️": "red heart",
    "❤": "red heart",
    "🔥": "fire",
    "💀": "skull",
    "😍": "heart eyes",
    "🥺": "pleading",
    "😎": "cool",
    "👀": "eyes",
    "🙏": "folded hands",
    "💯": "hundred",
    "✨": "sparkles",
    "🎉": "party",
    "😅": "sweat smile",
    "😊": "smiling",
    "🤔": "thinking",
    "👍": "thumbs up",
    "👎": "thumbs down",
    "🤡": "clown",
    "💅": "nail polish",
    "🫠": "melting",
    "😤": "huffing",
    "🥹": "holding back tears",
    "😩": "weary",
    "🤌": "pinched",
    "🫡": "saluting",
    "🙃": "upside down",
    "😬": "grimacing",
    "🤝": "handshake",
}


def is_emoji_char(ch: str) -> bool:
    o = ord(ch)
    return o == _ZWJ or any(lo <= o <= hi for lo, hi in _EMOJI_RANGES)


def split_emoji(s: str) -> list[tuple[str, bool]]:
    """Segment into runs, tagging each (text, is_emoji); adjacent emoji
    codepoints (ZWJ / skin tone / variation selectors) group greedily."""
    out: list[tuple[str, bool]] = []
    buf, buf_emoji = "", None
    for ch in s:
        e = is_emoji_char(ch)
        if buf and e == buf_emoji:
            buf += ch
        else:
            if buf:
                out.append((buf, bool(buf_emoji)))
            buf, buf_emoji = ch, e
    if buf:
        out.append((buf, bool(buf_emoji)))
    return out


def emoji_name(glyph: str) -> str:
    g = glyph.replace("️", "")
    return EMOJI_NAMES.get(glyph) or EMOJI_NAMES.get(g) or "emoji"


# ===========================================================================
# [lexicon]  hand-curated counts cmudict and the model both get wrong
# ===========================================================================

# Spoken syllable count of each letter name. Only "w" (double-u) is >1.
LETTER: dict[str, int] = {c: 1 for c in "abcdefghijklmnopqrstuvxyz"}
LETTER["w"] = 3

SYMBOL: dict[str, list[str]] = {
    "%": ["percent"],
    "&": ["and"],
    "@": ["at"],
    "#": ["hash"],
    "+": ["plus"],
    "=": ["equals"],
    "/": ["slash"],
    "~": ["about"],
    "*": ["star"],
    "^": ["caret"],
    "°": ["degrees"],
    "·": [],
    "$": ["dollars"],
    "€": ["euros"],
    "£": ["pounds"],
    "¥": ["yen"],
    "₿": ["bitcoin"],
    "©": ["copyright"],
    "®": ["registered"],
    "™": ["trademark"],
    "…": [],
    "—": [],
    "–": [],
}

# Modern / internet words with one defensible count -- overrides cmudict
# and the model.
SLANG: dict[str, int] = {
    "yeet": 1,
    "rizz": 1,
    "sus": 1,
    "bussin": 2,
    "skibidi": 3,
    "gyat": 1,
    "gyatt": 1,
    "delulu": 3,
    "sigma": 2,
    "sheesh": 1,
    "slay": 1,
    "cap": 1,
    "bougie": 2,
    "mid": 1,
    "based": 1,
    "cringe": 1,
    "simp": 1,
    "yap": 1,
    "goated": 2,
    "ick": 1,
    "mogging": 2,
    "mog": 1,
    "npc": 3,
    "pog": 1,
    "poggers": 2,
    "ratio": 3,
    "cope": 1,
    "gigachad": 3,
    "chad": 1,
    "bruh": 1,
    "fam": 1,
    "yolo": 2,
    "lit": 1,
    "bae": 1,
    "vibe": 1,
    "vibes": 1,
    "vibing": 2,
    "drip": 1,
    "dripped": 1,
    "stan": 1,
    "thicc": 1,
    "snatched": 1,
    "salty": 2,
    "shook": 1,
    "extra": 2,
    "ghosted": 2,
    "flex": 1,
    "flexing": 2,
    "clout": 1,
    "tea": 1,
    "shade": 1,
    "woke": 1,
    "wokeness": 2,
    "boomer": 2,
    "zoomer": 2,
    "doomer": 2,
    "incel": 2,
    "normie": 2,
    "noob": 1,
    "n00b": 1,
    "pwn": 1,
    "pwned": 1,
    "rekt": 1,
    "owned": 1,
    "glowup": 2,
    "glowdown": 2,
    "situationship": 5,
    "talking": 2,
    "doomscroll": 2,
    "doomscrolling": 3,
    "rickroll": 2,
    "paywall": 2,
    "mansplain": 2,
    "manspread": 2,
    "hangry": 2,
    "hangrily": 3,
    "deadass": 2,
    "lowkey": 2,
    "highkey": 2,
    "sideeye": 2,
    "thirsttrap": 2,
    "thirsty": 2,
    "thirst": 1,
    "swole": 1,
    "tiktok": 2,
    "tiktoker": 3,
    "insta": 2,
    "selfie": 2,
    "podcast": 2,
    "crypto": 2,
    "blockchain": 2,
    "emoji": 3,
    "wifi": 2,
    "blog": 1,
    "vlog": 1,
    "vlogger": 2,
    "meme": 1,
    "memes": 1,
    "memed": 1,
    "gif": 1,
    "gifs": 1,
    "app": 1,
    "apps": 1,
    "dox": 1,
    "doxx": 1,
    "doxxed": 1,
    "fomo": 2,
    "finsta": 2,
    "subtweet": 2,
    "screenshot": 2,
    "unfollow": 3,
    "unfriend": 2,
    "retweet": 2,
    "hashtag": 2,
    "livestream": 2,
    "livestreaming": 3,
    "streamer": 2,
    "twitch": 1,
    "discord": 2,
    "gonna": 2,
    "wanna": 2,
    "gotta": 2,
    "lemme": 2,
    "gimme": 2,
    "kinda": 2,
    "dunno": 2,
    "imma": 2,
    "finna": 2,
    "tryna": 2,
    "prolly": 2,
    "shoulda": 2,
    "coulda": 2,
    "woulda": 2,
    "musta": 2,
    "outta": 2,
    "sorta": 2,
    "betcha": 2,
    "gotcha": 2,
    "cuppa": 2,
    "hella": 2,
    "fella": 2,
    "buncha": 2,
    "yall": 1,
    "aint": 1,
    "cuz": 1,
    "tho": 1,
    "thru": 1,
    "nah": 1,
    "yea": 1,
    "yeah": 1,
    "yep": 1,
    "yup": 1,
    "nope": 1,
    "meh": 1,
    "eh": 1,
    "duh": 1,
    "hmm": 1,
    "psst": 1,
    "tsk": 1,
    "ugh": 1,
    "oof": 1,
    "yikes": 1,
    "womp": 1,
    "bruv": 1,
    "innit": 2,
    # NB: "fr" / "rn" deliberately NOT here -- with no vowel they fall
    # through to the letter tier ("ef ar" = 2), the right spoken count.
    # ("istg" is a (non-ambiguous) INITIALISM entry, resolved there.)
    "no": 1,
    "yas": 1,
    "yass": 1,
    "ah": 1,
    "ahh": 1,
    "bro": 1,
    "broo": 1,
    "periodt": 2,
    "aight": 1,
    "ya": 1,
}

# Texting initialisms: (count, ambiguous). ambiguous=True => no single
# spoken truth ("lol" = "lawl"/1 or spelled/3) so the counter abstains.
INITIALISM: dict[str, tuple[int, bool]] = {
    "lol": (3, True),
    "lmao": (3, True),
    "lmfao": (5, True),
    "rofl": (2, True),
    "idk": (3, False),
    "idc": (3, False),
    "tbh": (3, False),
    "ngl": (3, False),
    "omg": (3, False),
    "omfg": (4, False),
    "wtf": (3, False),
    "wth": (3, False),
    "btw": (4, False),
    "brb": (3, False),
    "smh": (3, False),
    "imo": (3, False),
    "imho": (4, False),
    "fyi": (3, False),
    "asap": (2, True),
    "aka": (3, False),
    "diy": (3, False),
    "faq": (3, True),
    "rsvp": (4, False),
    "eta": (3, False),
    "dm": (2, False),
    "dms": (3, False),
    "pfp": (3, False),
    "gg": (2, False),
    "ggs": (3, False),
    "gn": (2, False),
    "gm": (2, False),
    "ong": (3, True),
    "ofc": (3, True),
    "istg": (4, False),
    "iykyk": (5, False),
    "tldr": (4, True),
    "tl;dr": (4, True),
    "afaik": (4, True),
    "irl": (3, False),
    "nsfw": (5, True),
    "op": (2, True),
    "pov": (3, False),
    "wyd": (3, False),
    "hbu": (3, False),
    "wbu": (3, False),
    "ily": (3, False),
    "ttyl": (4, False),
    "nvm": (3, True),
    "jk": (2, True),
    "ftw": (4, False),
    "tfw": (3, False),
    "mfw": (3, False),
    "smfh": (4, False),
    "lmk": (3, False),
    "wfh": (3, False),
    "eli5": (4, True),
}

# Acronyms said as words, not spelled out (count is just the word).
ACRONYM: dict[str, int] = {
    "nasa": 2,
    "scuba": 2,
    "laser": 2,
    "radar": 2,
    "asap": 2,
    "fomo": 2,
    "yolo": 2,
    "gif": 1,
    "jpeg": 2,
    "png": 3,
    "url": 3,
    "html": 4,
    "http": 4,
    "https": 5,
    "json": 2,
    "sql": 3,
    "api": 3,
    "ui": 2,
    "ux": 2,
    "css": 3,
    "ascii": 2,
    "wysiwyg": 4,
    "captcha": 2,
    "covid": 2,
    "potus": 2,
    "scotus": 2,
    "swat": 1,
    "naacp": 5,
    "aids": 1,
    "unicef": 3,
    "opec": 2,
    "fifa": 2,
    "ikea": 3,
}

# Apostrophe-DROPPED contractions (don't -> dont) -- pervasive in posts.
# We map to the ASCII-apostrophe form, appended as a *lower-priority*
# candidate: a bare form that is itself a real word (cant, wont, were,
# its, lets) still resolves as that word first via the reference (same
# syllable count anyway), so this is collision-safe. The forms with no
# real-word twin (dont, ive, thats, theyre, doesnt) were the #1 residual
# failure on real Bluesky posts.
CONTRACTION: dict[str, str] = {
    f.replace("'", ""): f
    for f in (
        "ain't aren't can't could've couldn't didn't doesn't don't hadn't "
        "hasn't haven't he'd he'll he's here's how'd how's i'd i'll i'm "
        "i've isn't it'd it'll it's let's might've mustn't must've needn't "
        "she'd she'll she's shouldn't should've that'll that's there's "
        "they'd they'll they're they've wasn't we'd we'll we're we've "
        "weren't what'd what're what's when's where's who'd who'll who's "
        "who've why's won't would've wouldn't y'all you'd you'll you're "
        "you've"
    ).split()
}


# ===========================================================================
# [spell]  conservative SymSpell-style typo correction vs cmudict
# ===========================================================================
#
# Timid by policy: a wrong correction is a silent wrong count, worse than
# abstaining. >=4 chars only, first letter must survive, unique best
# candidate at the minimal edit distance required (a tie abstains).


def _osa_distance(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance; early-exits past `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            row_min = min(row_min, v)
        if row_min > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[len(b)]


def _deletes(word: str, edits: int) -> set[str]:
    out = {word}
    frontier = {word}
    for _ in range(edits):
        nxt: set[str] = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1 :])
        out |= nxt
        frontier = nxt
    return out


class SpellFixer:
    """Built once from the cmudict word list; delete-index is lazy."""

    def __init__(self, words):
        self._words = [w for w in words if w.isalpha()]
        self._index: dict[str, list[str]] | None = None

    def _build(self) -> None:
        idx: dict[str, list[str]] = {}
        for w in self._words:
            for d in _deletes(w, 2 if len(w) >= 8 else 1):
                idx.setdefault(d, []).append(w)
        self._index = idx

    def correct(self, token: str) -> tuple[str, int] | None:
        """Best safe correction as (word, distance), or None to abstain."""
        t = token.lower()
        if len(t) < 4 or not t.isalpha():
            return None
        if self._index is None:
            self._build()
        assert self._index is not None

        max_edits = 2 if len(t) >= 8 else 1
        cands: set[str] = set()
        for d in _deletes(t, max_edits):
            cands.update(self._index.get(d, ()))
        cands.discard(t)
        if not cands:
            return None

        best: list[str] = []
        best_d = max_edits + 1
        for c in cands:
            if c[0] != t[0] and not (
                len(c) > 1 and len(t) > 1 and c[:2] == t[1::-1][:2]
            ):
                continue  # first letter must survive (or be a transpose)
            dist = _osa_distance(t, c, max_edits)
            if dist > max_edits:
                continue
            if dist < best_d:
                best_d, best = dist, [c]
            elif dist == best_d:
                best.append(c)
        return (best[0], best_d) if len(best) == 1 else None


# ===========================================================================
# [normalize]  raw internet text -> stream of countable Tokens
# ===========================================================================

_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_NUM = re.compile(
    r"[-−]?[$£€¥]?\d[\d,]*(?:\.\d+)?%?(?:st|nd|rd|th)?"
    r"(?:[kmbt](?![A-Za-z]))?",
    re.IGNORECASE,
)
# A host is only a URL if its last label is a known TLD (or a ccTLD whose
# second-level is registry-ish, e.g. foo.co.uk). This is checked in code
# after a permissive grab, so bit.ly and foo.co.uk work without an
# ever-growing alternation.
_URLISH = re.compile(
    r"\b(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)+(?:/\S*)?", re.IGNORECASE
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MENTION = re.compile(r"@(\w+)")
_HASHTAG = re.compile(r"#(\w+)")
# "#3" is the number sign ("number three"), not a hashtag like #FooBar.
_HASHNUM = re.compile(r"\d+(?:st|nd|rd|th)?\Z", re.I)
TLDS = set(
    "com org net io gov edu co ai dev app me tv gg xyz info biz online "
    "site blog news ly uk us ca eu de fr es it nl ru cn jp kr in br au "
    "nz mx za ch se no fi pl tr so sh to cc gl fm am".split()
)
_CC2 = {"co", "com", "org", "ac", "gov", "net", "edu"}  # foo.co.uk
# Spoken form for TLDs people don't say as a word; the rest fall through
# to _split_handle (read as a word, e.g. "info", "blog").
_TLD = {
    "io": ["i", "o"],
    "edu": ["e", "d", "u"],
    "ai": ["a", "i"],
    "tv": ["t", "v"],
    "gg": ["g", "g"],
    "xyz": ["x", "y", "z"],
    "uk": ["u", "k"],
    "us": ["u", "s"],
    "eu": ["e", "u"],
    "fm": ["f", "m"],
    "cc": ["c", "c"],
    "fr": ["f", "r"],
}

# ===========================================================================
# [extras]  emoticons, slash-shorthand, roman numerals, dates/times,
#           fractions/ratios, diacritics
# ===========================================================================

# ASCII emoticons / kaomoji are punctuation in speech -> silent (0), just
# like emoji. Nobody voices ":)" or "<3". Longest plausible match.
_EMOTICON = re.compile(
    r"</?3|x+D+|X+D+|>?[:;=]'?-?[)(\[\]DPpoO0/|\\3c]|\^[-_.]?\^|-_-|"
    r"[oO0]_[oO0]|[oO]\.[oO]|[tT]_[tT]|[tT]\.[tT]|;_;"
)

# Strict Roman numeral (1-3999). Only treated as one when the raw token
# is all-uppercase, length>=2, and not a real cmudict word. Small values
# (<= ROMAN_MAX) get a default *cardinal* reading ("World War II" -> two,
# "Star Wars IV" -> four), marked approx since the cardinal/ordinal
# choice is a guess about intent -- though for 3..10 both readings have
# the same syllable count anyway, so only II ("two"=1 vs "second"=2) is
# a real coverage-for-precision trade. Larger values stay an abstention
# (the ambiguity compounds and big numerals are rarer / more numeral-y).
_ROMAN = re.compile(r"M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\Z")
ROMAN_MAX = 10

# Unambiguous slash-shorthand -> words (then exact via cmudict).
SHORTHAND = {"w/o": ["without"], "w/": ["with"], "b/c": ["because"]}
_SHORTHAND = re.compile(r"(?i)(w/o|w/|b/c)(?![\w/])")

_MONTHS = (
    "",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_TIME = re.compile(r"(\d{1,2}):([0-5]\d)(?::\d\d)?\s*([ap])\.?\s?m\.?", re.I)
_TIME2 = re.compile(r"\b(\d{1,2}):([0-5]\d)(?![\d:])")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_RATIO = re.compile(r"\b(\d{1,3})-(\d{1,3})(?![\w-])")
_FRACTION = re.compile(r"\b(\d{1,2})/(\d{1,2})(?![\d/])")
_DENOM = {2: "half", 4: "quarter"}
# Vulgar fraction codepoints -> spoken words (substituted before NFKC,
# which would otherwise shatter ½ into "1 / 2").
VULGAR = {
    "½": "one half",
    "⅓": "one third",
    "⅔": "two thirds",
    "¼": "one quarter",
    "¾": "three quarters",
    "⅕": "one fifth",
    "⅖": "two fifths",
    "⅗": "three fifths",
    "⅘": "four fifths",
    "⅙": "one sixth",
    "⅚": "five sixths",
    "⅐": "one seventh",
    "⅛": "one eighth",
    "⅜": "three eighths",
    "⅝": "five eighths",
    "⅞": "seven eighths",
    "⅑": "one ninth",
    "⅒": "one tenth",
}


def _strip_diacritics(s: str) -> str:
    """café -> cafe, naïve -> naive (NFKD, drop combining marks)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _roman_value(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        total += -v if i + 1 < len(s) and vals[s[i + 1]] > v else v
    return total


def _date_words(mo: int, day: int, yr: int) -> list[str] | None:
    if not (1 <= mo <= 12 and 1 <= day <= 31):
        return None
    if yr < 100:
        yr += 1900 if yr > 68 else 2000
    return [_MONTHS[mo], *ordinal(day), *year(yr)]


def _time_words(hr: int, mn: int, ap: str | None) -> list[str] | None:
    if not (0 <= hr <= 23 and 0 <= mn <= 59):
        return None
    w = cardinal(hr if 1 <= hr <= 12 or not ap else hr)
    if mn == 0:
        w += ["o", "clock"]
    elif mn < 10:
        w += ["oh", *cardinal(mn)]
    else:
        w += cardinal(mn)
    if ap:
        w += [ap.lower(), "m"]  # "a m" / "p m" -> spelled, 2 syllables
    return w


def _fraction_words(num: int, den: int) -> list[str]:
    base = _DENOM.get(den) or ordinal(den)[-1]
    if num != 1:
        base = base[:-2] + "ves" if base == "half" else base + "s"
    return cardinal(num) + [base]


@dataclass
class Token:
    raw: str
    kind: str  # word|number|symbol|url|emoji|punct
    candidates: list[str] = field(default_factory=list)  # for kind=word
    words: list[str] | None = None  # pre-expanded plain words (count each)
    silent: bool = False  # contributes 0 syllables
    approx: bool = False  # deterministic but interpreted reading
    note: str = ""


def collapse_elongation(word: str) -> list[str]:
    """noooooo -> [no]; cooool -> [col, cool]; soooo -> [so, soo].

    Any letter run >=3 is emphasis, not spelling. We can't know if the
    base keeps a double, so return single- then double-collapsed forms;
    the counter takes the first a tier knows.
    """
    one = re.sub(r"(.)\1{2,}", r"\1", word)
    two = re.sub(r"(.)\1{2,}", r"\1\1", word)
    if one == word:
        return [word]
    return [one] if one == two else [one, two]


def _split_handle(h: str) -> list[str]:
    """fooBar_baz42 -> [foo, bar, baz, forty, two]."""
    out: list[str] = []
    for p in re.split(r"[_\-]+", h):
        for chunk in re.findall(r"[A-Za-z]+|\d+", p):
            if chunk[0].isdigit():
                out += expand(chunk)
            else:
                out += re.findall(
                    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+", chunk
                ) or [chunk]
    return [w.lower() for w in out if w]


def _expand_url(u: str) -> list[str]:
    """example.com/x -> [example, dot, com]; www -> w w w (double-u x3)."""
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = u.split("/")[0].split("?")[0].rstrip(".,!?)]}'\"")
    out: list[str] = []
    for i, part in enumerate(p for p in u.split(".") if p):
        if i:
            out.append("dot")
        low = part.lower()
        if low == "www":
            out += ["w", "w", "w"]
        elif low in _TLD:
            out += _TLD[low]
        else:
            out += _split_handle(part)
    return out


def _classify_word(raw: str) -> Token:
    base = raw.strip("'’-")
    # Roman numeral: only when ALL-CAPS, >=2 chars, and not a real word.
    # Small values -> a default cardinal reading (approx); larger ones
    # stay an abstention -- the cardinal/ordinal ambiguity compounds.
    if (
        len(base) >= 2
        and base.isupper()
        and base.isalpha()
        and _ROMAN.match(base)
        and _cmu_count(base.lower()) is None
    ):
        v = _roman_value(base)
        if 1 <= v <= ROMAN_MAX:
            return Token(
                raw=raw, kind="number", words=cardinal(v), approx=True,
                note="roman numeral",
            )
        return Token(
            raw=raw, kind="word", candidates=[], note="roman numeral (ambiguous)"
        )
    w = raw.lower().strip("'’-")
    cands = collapse_elongation(w)  # always non-empty (>=1 candidate)
    for c in list(cands):  # accent-folded variant -> cmudict can hit it
        d = _strip_diacritics(c)
        if d != c and d not in cands:
            cands.append(d)
    note = "elongation" if cands[0] != w else ""
    for c in list(cands):  # a-z-stripped: the reference is a-z-keyed, so
        s = re.sub(r"[^a-z]", "", c)  # an apostrophe form ("don't") only
        if s and s != c and s not in cands:  # hits as its key ("dont")
            cands.append(s)
    if w in CONTRACTION:  # appended last: real-word twins win first
        cands.append(CONTRACTION[w])
        note = note or "contraction"
    return Token(raw=raw, kind="word", candidates=cands, note=note)


# Unicode apostrophe / single-quote variants -> ASCII '. iOS/macOS auto-
# convert ' to ' (U+2019); cmudict keys use ASCII, and NFKC does NOT fold
# these, so without this every smart-quote contraction (I'm, don't, he's)
# misses cmudict and abstains. ~11% of real-post abstentions were this.
_APOS = {ord(c): "'" for c in "’‘ʼ′´`"}


def tokenize(text: str) -> list[Token]:
    """The whole normalization pass. NFKC folds stylized unicode (𝓱𝓮𝓵𝓵𝓸,
    ﬁ-ligatures, full-width) so it isn't spuriously OOV; smart quotes are
    folded to ASCII ' first so contractions match cmudict."""
    text = text.translate(_APOS)
    for k, v in VULGAR.items():  # before NFKC shatters ½ into "1 / 2"
        if k in text:
            text = text.replace(k, f" {v} ")
    text = unicodedata.normalize("NFKC", text)
    toks: list[Token] = []
    for chunk, is_emoji in split_emoji(text):
        if is_emoji:
            names = ", ".join(
                emoji_name(g)
                for g in chunk
                if is_emoji_char(g)
                and ord(g) != _ZWJ
                and not 0xFE00 <= ord(g) <= 0xFE0F
            )
            toks.append(
                Token(raw=chunk, kind="emoji", silent=True, note=names or "emoji")
            )
        else:
            toks.extend(_tokenize_text(chunk))
    return toks


def _is_url(body: str) -> bool:
    if body.lower().startswith(("http://", "https://", "www.")):
        return True
    host = body.split("/")[0].split("?")[0].rstrip(".,!?)]}'\"")
    labels = [x for x in host.split(".") if x]
    if len(labels) < 2:
        return False
    tld = labels[-1].lower()
    # A path or query after a plausible dotted host is an unambiguous URL
    # regardless of TLD (youtu.be/x, t.co/x, foo.bar/x). The TLD allowlist
    # only guards the *bare* host case (so "e.g."/"U.S." aren't URLs).
    if ("/" in body or "?" in body) and tld.isalpha() and len(tld) >= 2:
        return True
    return tld in TLDS or (
        len(labels) >= 3 and len(tld) == 2 and labels[-2].lower() in _CC2
    )


def _tokenize_text(text: str) -> list[Token]:
    toks: list[Token] = []
    spans: list[tuple[int, int, str]] = []
    for m in _URLISH.finditer(text):
        if _is_url(m.group(0)):
            spans.append((m.start(), m.end(), "url"))
    for m in _EMAIL.finditer(text):
        spans.append((m.start(), m.end(), "email"))
    spans.sort()
    keep: list[tuple[int, int, str]] = []
    last = -1
    for s, e, k in spans:
        if s >= last:
            keep.append((s, e, k))
            last = e

    pos = 0
    for s, e, k in keep:
        toks.extend(_tokenize_plain(text[pos:s]))
        body = text[s:e]
        if k == "email":
            local, _, host = body.partition("@")
            words: list[str] = []
            for j, piece in enumerate(local.split(".")):
                if j:
                    words.append("dot")  # "first dot last at ..."
                words += _split_handle(piece)
            words += ["at"] + _expand_url(host)
        else:
            words = _expand_url(body)
        toks.append(Token(raw=body, kind="url", words=words, note="spoken url"))
        pos = e
    toks.extend(_tokenize_plain(text[pos:]))
    return toks


def _date_slash(g) -> list[str] | None:
    a, b, y = int(g[0]), int(g[1]), int(g[2])
    mo, day = (b, a) if a > 12 and b <= 12 else (a, b)  # US m/d default
    return _date_words(mo, day, y)


def _extra_digit_token(text: str, i: int) -> tuple[Token, int] | None:
    """Date / time / ratio / fraction starting at a digit; None if none.

    These are *interpreted* readings (approx) -- deterministic but a guess
    about what the writer meant -- so they show ~ in the trace, not exact.
    """
    for rx, kind, build in (
        (_DATE_ISO, "date", lambda g: _date_words(int(g[1]), int(g[2]), int(g[0]))),
        (_DATE_SLASH, "date", _date_slash),
        (_TIME, "time", lambda g: _time_words(int(g[0]), int(g[1]), g[2])),
        (_TIME2, "time", lambda g: _time_words(int(g[0]), int(g[1]), None)),
        (_RATIO, "ratio", lambda g: cardinal(int(g[0])) + cardinal(int(g[1]))),
        (_FRACTION, "fraction", lambda g: _fraction_words(int(g[0]), int(g[1]))),
    ):
        m = rx.match(text, i)
        if not m:
            continue
        w = build(m.groups())
        if w:
            return Token(
                raw=m.group(0), kind=kind, words=w, approx=True, note=kind
            ), m.end()
    return None


def _tokenize_plain(text: str) -> list[Token]:
    toks: list[Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        em = _EMOTICON.match(text, i)
        if em and (em.end() == len(text) or not text[em.end()].isalnum()):
            toks.append(
                Token(raw=em.group(0), kind="emoji", silent=True, note="emoticon")
            )
            i = em.end()
            continue
        sh = _SHORTHAND.match(text, i)
        if sh:
            toks.append(
                Token(
                    raw=sh.group(0),
                    kind="word",
                    words=SHORTHAND[sh.group(1).lower()],
                    note="shorthand",
                )
            )
            i = sh.end()
            continue
        if ch.isdigit():
            extra = _extra_digit_token(text, i)
            if extra:
                tok, end = extra
                toks.append(tok)
                i = end
                continue
        m = _MENTION.match(text, i) or _HASHTAG.match(text, i)
        if m and m.start() == i:
            handle = m.group(1)
            if text[i] == "@":
                toks.append(
                    Token(
                        raw=m.group(0),
                        kind="word",
                        words=["at"] + _split_handle(handle),
                        note="mention",
                    )
                )
            elif _HASHNUM.match(handle):  # "#3" -> "number three"
                toks.append(
                    Token(
                        raw=m.group(0),
                        kind="word",
                        words=["number"] + expand(handle),
                        note="number sign",
                    )
                )
            else:  # '#': in speech the words are read, the hash is silent
                toks.append(
                    Token(
                        raw=m.group(0),
                        kind="word",
                        words=_split_handle(handle),
                        note="hashtag",
                    )
                )
            i = m.end()
            continue
        nm = _NUM.match(text, i)
        if nm and any(c.isdigit() for c in nm.group(0)):
            words = expand(nm.group(0))
            if words:
                toks.append(
                    Token(raw=nm.group(0), kind="number", words=words, note="number")
                )
                i = nm.end()
                continue
        wm = _WORD.match(text, i)
        if wm:
            raw = wm.group(0)
            if "-" in raw and len(raw) > 3:  # compound: split & sum
                for part in (p for p in raw.split("-") if p):
                    toks.append(_classify_word(part))
            else:
                toks.append(_classify_word(raw))
            i = wm.end()
            continue
        if ch in SYMBOL:
            w = SYMBOL[ch]
            toks.append(
                Token(raw=ch, kind="symbol", words=w, silent=not w, note="symbol")
            )
        else:
            toks.append(Token(raw=ch, kind="punct", silent=True))
        i += 1
    return toks


# ===========================================================================
# [count]  tiered conservative counter + aggregation -- DETERMINISTIC ONLY
# ===========================================================================
#
# This module never touches the neural model. It resolves a word through
# deterministic tiers or abstains; the OOV tokens it abstains on are the
# `harness` layer's job (it batches them through the model). Keeping the
# net out of here is what makes batching / a single fuzziness policy /
# clean testing possible.
#
# Tier order for a word (first confident hit wins; elongation candidates
# tried in order within each tier):
#   1 SLANG       curated single count            -> exact
#   2 INITIALISM  texting acronym; ambiguous abstain -> exact
#   3 reference   CMUdict+WikiPron+kaikki (sources.build); PRIMARY -> exact
#   4 ACRONYM     said as a word (nasa, scuba)    -> exact
#   5 letters     vowelless/short, spelled aloud  -> exact
#   6 spell-fix   unique close cmudict word (OFF) -> approx
#   7 (none)      abstain  (-> harness asks the model)
#
# Tier 6 is disabled by default: measured on real Bluesky posts it turned
# honest abstentions into confident-WRONG counts (rational->rationale,
# collabs->collars) -- the exact failure the conservative contract bans.
# Flip ENABLE_SPELL to revert.

ENABLE_SPELL = False
_VOWELS = set("aeiouy")


@lru_cache(maxsize=1)
def _cmu() -> dict[str, dict]:
    """The reference lexicon: the other session's reconciled CMUdict +
    WikiPron + kaikki(Wiktionary) table (`sources.build()`), ~173k words
    incl. slang/proper/loanword breadth, sharing the model's training
    ground truth. Falls back to cmudict-only if the WikiPron/kaikki
    extracts aren't present, so the pipeline still works without them."""
    try:
        return data.build_sources()
    except Exception:
        return {
            w: {
                "primary": e["primary"],
                "valid": frozenset(e["counts"]),
                "ambiguous": len(set(e["counts"])) > 1,
                "source": "cmu",
            }
            for w, e in data.parse_cmudict().items()
        }


@lru_cache(maxsize=1)
def _speller() -> SpellFixer:
    return SpellFixer(_cmu().keys())


def _cmu_count(word: str) -> int | None:
    """Reference count via the PRIMARY (canonical) pronunciation, or None
    if unknown. We do NOT fail-closed on multi-pronunciation / ambiguous
    words (`fire`, `every`): policy is to emit `primary` for max coverage
    (the genuine-ambiguity rate is ~1.4% and accepted)."""
    e = _cmu().get(word)
    return e["primary"] if e else None


@dataclass
class Counted:
    raw: str
    count: int  # best-effort; 0 for silent / abstained
    source: str  # slang|initialism|cmudict|acronym|letters|spell|
    #                     number|url|symbol|emoji|punct|abstain
    confidence: str  # exact | approx | none
    note: str = ""


def _looks_spelled_out(tok: str) -> bool:
    """idk / btw / fbi: short, no vowel sound -> read letter by letter."""
    return 2 <= len(tok) <= 6 and tok.isalpha() and not (_VOWELS & set(tok))


def _spell_letters(s: str) -> tuple[int, str, str, str]:
    return (sum(LETTER.get(c, 1) for c in s), "letters", "exact", "-".join(s))


def _count_one_word(cands: list[str], raw: str = "") -> tuple[int, str, str, str]:
    for w in cands:
        if w in SLANG:
            return SLANG[w], "slang", "exact", ""
        if w in INITIALISM:
            n, amb = INITIALISM[w]
            if amb:
                return 0, "abstain", "none", "ambiguous initialism"
            return n, "initialism", "exact", ""
        c = _cmu_count(w)
        if c is not None:
            return c, "cmudict", "exact", ""
        if w in ACRONYM:
            return ACRONYM[w], "acronym", "exact", ""
    # An ALL-CAPS short non-word (after cmudict/ACRONYM missed) is an
    # initialism cmudict just lacks -> spell it (IBM, NDA, ACLU). Strong,
    # conservative cue; NASA/STOP already resolved above as words. The
    # `cands` guard preserves the Roman-numeral abstain (empty candidates).
    if cands and 2 <= len(raw) <= 6 and raw.isalpha() and raw.isupper():
        return _spell_letters(raw.lower())
    for w in cands:
        if _looks_spelled_out(w):
            return _spell_letters(w)
    if ENABLE_SPELL:
        for w in cands:
            fix = _speller().correct(w)
            if fix:
                corrected, dist = fix
                c = _cmu_count(corrected)
                if c is not None:
                    return c, "spell", "approx", f"{w}→{corrected} (d{dist})"
    return 0, "abstain", "none", "OOV"  # OOV -> harness may ask the model


def count_token(tok: Token) -> Counted:
    if tok.silent:
        return Counted(tok.raw, 0, tok.kind, "exact", tok.note)
    if tok.words is not None:
        total, miss = 0, []
        for w in tok.words:
            c = _cmu_count(w)
            if c is None:  # rare: a handle word not in cmudict
                cc = _count_one_word([w])
                c = cc[0]
                if cc[1] == "abstain":
                    miss.append(w)
            total += c
        note = tok.note
        if miss:
            return Counted(
                tok.raw,
                total,
                "abstain",
                "approx",
                f"{note} [OOV: {' '.join(miss)}]".strip(),
            )
        return Counted(
            tok.raw, total, tok.kind, "approx" if tok.approx else "exact", note
        )
    n, src, conf, note = _count_one_word(tok.candidates, tok.raw.strip("'’-"))
    note = f"{tok.note}; {note}" if note and tok.note else note or tok.note
    return Counted(tok.raw, n, src, conf, note)


@dataclass
class Result:
    total: int  # best-effort sum (abstained units -> 0)
    confident: bool  # no abstentions among non-silent units
    n_uncertain: int
    tokens: list[Counted]

    def __str__(self) -> str:
        tag = "OK    " if self.confident else "UNSURE"
        extra = "" if self.confident else f"  ({self.n_uncertain} uncertain)"
        return f"[{tag}] total={self.total} syllables{extra}"


def analyze(text: str) -> Result:
    """Full pipeline: normalize -> tier-count each unit -> aggregate.

    `confident` is True only if every non-silent unit resolved through a
    trusted tier with no abstentions -- the conservative contract.
    """
    counted = [count_token(t) for t in tokenize(text)]
    total = sum(c.count for c in counted)
    uncertain = [c for c in counted if c.confidence == "none" or c.source == "abstain"]
    return Result(total, not uncertain, len(uncertain), counted)


# ===========================================================================
# entrypoint -- human-readable, auditable trace
# ===========================================================================

DEMO = [
    "I have 5 cats and $19.99 to spend lol",
    "noooooo this is sooo bussin fr 😭🔥",
    "check out https://www.example.com/cats for more",
    "in 1999 the team scored 21st 🎉",
    "tbh idk why @CoolUser42 posted #ThrowbackThursday",
    "teh quick borwn fox (definately a typo)",
    "it's 3.14% colder today, ~5 degrees",
    "she said 𝓱𝓮𝓵𝓵𝓸 in 𝐛𝐨𝐥𝐝 and ﬁ-ligatures",
    "heart u <3 xD :) lol so funny",
    "$1.2B raised, 20k users, café w/ résumé",
    "meeting 5/17/2026 at 10:45pm, ½ done, World War II",
    "email me at first.last@gmail.com or bit.ly/abc",
]


def describe(text: str) -> str:
    r = analyze(text)
    lines = [f"{text!r}", str(r)]
    for c in r.tokens:
        if c.source == "punct":
            continue
        mark = {"exact": " ", "approx": "~", "none": "!"}[c.confidence]
        body = f" {c.note}" if c.note else ""
        lines.append(f"  {mark} {c.raw!r:24s} {c.count:>2d}  {c.source:<10s}{body}")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    if args == ["-"]:
        args = [sys.stdin.read()]
    for text in args or DEMO:
        print(describe(text))
        print()


if __name__ == "__main__":
    main()
