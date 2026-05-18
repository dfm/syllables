"""Plot training curves from a streamed results JSONL.

    uv run python scripts/plot_curves.py [results/tune-<stamp>.jsonl]

Saves results/curves.png: training loss, val exact-match, and the two
calibration metrics (amb_cover, amb_mass) vs epoch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(Path("results").glob("tune-*.jsonl"))[-1]
    )
    recs = (json.loads(line) for line in Path(path).read_text().splitlines() if line)
    vals = [r for r in recs if r["event"] == "val"]
    ep = [v["epoch"] for v in vals]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(ep, [v["loss"] for v in vals], label="train loss", color="C3")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("soft-CE train loss")
    ax0b = ax[0].twinx()
    ax0b.plot(ep, [v["exact"] for v in vals], label="val exact", color="C0")
    ax0b.set_ylabel("val exact-match")
    ax[0].set_title("loss vs accuracy")
    ax[0].legend(loc="upper left")
    ax0b.legend(loc="lower right")

    ax[1].plot(
        ep, [v.get("amb_cover", 0) for v in vals], label="amb_cover@0.9", color="C2"
    )
    ax[1].plot(ep, [v.get("amb_mass", 0) for v in vals], label="amb_mass", color="C1")
    ax[1].plot(
        ep,
        [v["exact"] for v in vals],
        label="val exact",
        color="C0",
        ls="--",
        alpha=0.6,
    )
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title("calibration vs accuracy")
    ax[1].legend(loc="lower right")

    fig.tight_layout()
    out = Path("results") / "curves.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}  ({len(vals)} eval points, up to epoch {ep[-1]})")


if __name__ == "__main__":
    main()
