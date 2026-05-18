"""Explain how a line flows through the whole pipeline.

A guided tour of the layers, line by line:

  1. the deterministic per-token trace (`heuristics.describe`)
  2. how the harness splits it: syllables resolved deterministically,
     which tokens fall through to the model, or a hard abstention
  3. the `fuzziness` sweep -- the final public count at 0 .. 1; the
     columns diverge *only* when a line hinges on a modelled token,
     which is the quickest way to see what the model actually decides
  4. the haiku verdict (`is_haiku_line`)

    uv run python scripts/explain.py                 # built-in corpus
    uv run python scripts/explain.py "noooo bussin"  # your own text
    echo "some text" | uv run python scripts/explain.py -
    uv run python scripts/explain.py --firehose --limit 40   # live Bluesky

`--firehose` streams real Bluesky posts. It lazily imports the official
`atproto` client; that is NOT a project dependency (runtime is just
jax+numpy) -- install it only for this mode (`uv pip install atproto`).
The self-contained modes need nothing extra.
"""

from __future__ import annotations

import argparse
import sys

from syllables import count_syllables, harness, heuristics, is_haiku_line

FUZZ = (0.0, 0.25, 0.5, 0.75, 1.0)

# Representative messy lines, grouped so the default run is a tour AND
# actually demonstrates the fuzziness knob. NOTE: deterministic and
# hard-abstain lines are *expected* to be flat across fuzziness -- the
# model is never consulted there, so the knob physically cannot move
# them. Only the model-routed group below makes the sweep diverge.
CORPUS = [
    # -- deterministic (numbers/currency/year, emoji, slang, contraction,
    #    elongation, roman expansion): flat by design --
    "I spent $1.2B and £5.99 on 1999 vibes",
    "noooo this is sooo bussin fr 😭🔥",
    "I'm gonna doomscroll, don't @ me",
    "World War II started a long time ago",
    # -- hard-abstain (ambiguous initialism, full URL, junk): all-dashes
    #    by design, model never consulted --
    "tbh idk why lol",
    "check ebay.com then https://example.com/post?x=1",
    "macro: hhFDWkdAxDjkbJfXdOwv",
    # -- model-routed: the fuzziness sweep visibly moves, at DIFFERENT
    #    points (the per-OOV "model :" line shows why) --
    "doomscrollery is my whole personality now",     # accepts from f=0.25
    "that situationshippy energy is so exhausting",  # accepts from f=0.25
    "the quick borwn fox jumps over",                # accepts from f=0.75
    # -- a real haiku line (deterministic, exactly 5) --
    "an old silent pond",
]


def explain_line(line: str) -> None:
    """Print the full cross-layer breakdown for one line."""
    print("=" * 72)
    print(repr(line))
    print("-" * 72)
    # 1. deterministic per-token trace
    print(heuristics.describe(line))
    # 2. how the harness routes it
    det, oov, hard = harness._resolve_line(line)
    if hard:
        print("\n  routing : HARD ABSTAIN "
              "(roman / ambiguous initialism / junk) -> None")
    elif not oov:
        print(f"\n  routing : fully deterministic, "
              f"det={det} syllables (no model)")
    else:
        print(f"\n  routing : det={det} syllables + {len(oov)} OOV "
              f"-> model: {', '.join(oov)}")
        # Show each OOV token's model confidence and the lowest fuzziness
        # that accepts it. The shipped model is deliberately tiny and
        # often overconfident, so most tokens accept already at f=0 --
        # which is *why* the sweep below is usually flat (not a bug).
        mdl = harness._model()
        if mdl is None:
            print("  model   : UNAVAILABLE (no jax / weights) -> every "
                  "OOV line abstains at all fuzziness")
        else:
            probs = harness._probs_fixed(mdl, list(oov))
            for w, p in zip(oov, probs, strict=True):
                order = p.argsort()[::-1]
                top, second = float(p[order[0]]), float(p[order[1]])
                acc = next(f for f in FUZZ
                           if harness._gate(p, f) is not None)
                gate = ("accepted at strict f=0" if acc == 0.0
                        else f"abstains at f=0, accepts from f={acc}")
                print(f"  model   : {w!r} -> count {int(order[0]) + 1}  "
                      f"(p={top:.2f}, margin={top - second:.2f})  "
                      f"{gate}")
    # 3. fuzziness sweep of the final public count
    cells = []
    for f in FUZZ:
        n = count_syllables(line, fuzziness=f)
        cells.append(f"{f:.2f}:{'-' if n is None else n}")
    print(f"  fuzz    : {'  '.join(cells)}   (- = abstain/discard)")
    # 4. haiku verdict
    hk = is_haiku_line(line)
    print(f"  haiku   : {hk if hk else 'no'}")


def _post_text(record) -> tuple[str, bool] | None:
    """(text, is_english_toplevel) from a decoded post record, or None."""
    kind = record.get("$type") or record.get("py_type")
    if kind != "app.bsky.feed.post":
        return None
    text = (record.get("text") or "").strip()
    if not text:
        return None
    langs = record.get("langs") or []
    english = ("en" in langs or any(str(x).startswith("en") for x in langs))
    return text, bool(english) and record.get("reply") is None


def run_firehose(limit: int, secs: int) -> None:
    try:
        from atproto import (
            CAR,
            FirehoseSubscribeReposClient,
            models,
            parse_subscribe_repos_message,
        )
    except ImportError:
        raise SystemExit(
            "--firehose needs the optional `atproto` client (not a "
            "project dependency). Install it just for this mode:\n"
            "    uv pip install atproto"
        ) from None

    client = FirehoseSubscribeReposClient()
    seen = haiku_hits = 0
    timer = None

    def stop():
        nonlocal timer
        if timer is not None:
            timer.cancel()
            timer = None
        client.stop()

    if secs:
        import threading

        timer = threading.Timer(secs, stop)
        timer.daemon = True
        timer.start()

    def on_message(message) -> None:
        nonlocal seen, haiku_hits
        commit = parse_subscribe_repos_message(message)
        if not isinstance(
            commit, models.ComAtprotoSyncSubscribeRepos.Commit
        ):
            return
        try:
            car = CAR.from_bytes(commit.blocks)
        except Exception:
            return
        for op in commit.ops:
            if op.action != "create" or "app.bsky.feed.post" not in op.path:
                continue
            got = _post_text(car.blocks.get(op.cid) or {})
            if not got:
                continue
            text, ok = got
            if not ok:
                continue
            seen += 1
            hk = is_haiku_line(text)
            if hk:
                haiku_hits += 1
                print(f"\n*** HAIKU LINE ({hk} syllables) ***")
                explain_line(text)
            elif seen % 25 == 0:  # periodic sample so you see the flow
                n = count_syllables(text)
                print(f"[{seen:5d} seen, {haiku_hits} haiku]  "
                      f"count={n if n is not None else '-'}  {text[:60]!r}")
            if limit and seen >= limit:
                stop()
                return

    print("streaming Bluesky (Ctrl-C to stop)...", flush=True)
    try:
        client.start(on_message)
    except KeyboardInterrupt:
        stop()
    print(f"\ndone: {seen} english posts seen, {haiku_hits} haiku lines.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Explain how a line flows through the pipeline."
    )
    ap.add_argument("text", nargs="*",
                    help="line(s) to explain; '-' reads stdin; "
                         "omit for the built-in corpus")
    ap.add_argument("--firehose", action="store_true",
                    help="stream live Bluesky posts (needs `atproto`)")
    ap.add_argument("--limit", type=int, default=50,
                    help="--firehose: stop after N english posts (0 = inf)")
    ap.add_argument("--secs", type=int, default=0,
                    help="--firehose: also stop after S seconds")
    args = ap.parse_args()

    if args.firehose:
        run_firehose(args.limit, args.secs)
        return

    if args.text == ["-"]:
        lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    else:
        lines = args.text or CORPUS
    for line in lines:
        explain_line(line)


if __name__ == "__main__":
    main()
