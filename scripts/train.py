"""Train / tune the syllable-count model.

    uv run python scripts/train.py --tune              # sweep + converge
    uv run python scripts/train.py --split all         # ship model: all data
    uv run python scripts/train.py --summarize results/<file>.jsonl

Recipe: warmup -> cosine LR, AdamW weight decay, dropout on the pooled
representation, best-val checkpointing with early stopping. `--split all`
trains on every word (no holdout, no early stop) for the deployable
model. Labels are counts in [1, MAX_COUNT]; class index = count - 1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from syllables import data, model

RESULTS_DIR = Path("results")

SMALL = dict(
    embed_dim=24, conv_widths=[2, 3], conv_filters=32, gru_hidden=48, head_hidden=64
)
MEDIUM = dict(
    embed_dim=32, conv_widths=[2, 3], conv_filters=64, gru_hidden=96, head_hidden=128
)

# Targeted sweep: capacity x dropout, fixed improved recipe.
CONFIGS = {
    "small_d10": dict(caps=SMALL, drop=0.10),
    "small_d25": dict(caps=SMALL, drop=0.25),
    "medium_d10": dict(caps=MEDIUM, drop=0.10),
    "medium_d25": dict(caps=MEDIUM, drop=0.25),
}

# "How small can it go?" ladder -- find the accuracy/size knee. The GRU is
# ~3/4 of the params, so gru_hidden shrinks hardest.
SHRINK = {
    "s44k": dict(caps=SMALL, drop=0.10),
    "s22k": dict(
        caps=dict(
            embed_dim=16,
            conv_widths=[2, 3],
            conv_filters=24,
            gru_hidden=32,
            head_hidden=48,
        ),
        drop=0.10,
    ),
    "s7k": dict(
        caps=dict(
            embed_dim=12,
            conv_widths=[2, 3],
            conv_filters=16,
            gru_hidden=16,
            head_hidden=32,
        ),
        drop=0.05,
    ),
    "s4k": dict(
        caps=dict(
            embed_dim=10,
            conv_widths=[2, 3],
            conv_filters=12,
            gru_hidden=12,
            head_hidden=24,
        ),
        drop=0.05,
    ),
    "s2k": dict(
        caps=dict(
            embed_dim=8,
            conv_widths=[2, 3],
            conv_filters=8,
            gru_hidden=8,
            head_hidden=16,
        ),
        drop=0.0,
    ),
}


def _jsonl(path, record):
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _predict_probs(forward, params, X, batch=2048):
    n = len(X)
    pad = (-n) % batch
    Xp = np.concatenate([X, np.zeros((pad, X.shape[1]), X.dtype)]) if pad else X
    out = []
    for i in range(0, len(Xp), batch):
        logits = forward(params, jnp.asarray(Xp[i : i + batch]))
        out.append(np.asarray(jax.nn.softmax(logits, -1)))
    return np.concatenate(out)[:n]  # [n, C], class c-1 = count c


def _covers(p, valid, thr=0.9):
    """Does the smallest top-prob set summing to >=thr cover the valid set?"""
    order = np.argsort(-p)
    cum, support = 0.0, set()
    for c in order:
        support.add(int(c) + 1)
        cum += p[c]
        if cum >= thr:
            break
    return valid <= support


def _metrics(P, split):
    """All metrics from a [n, C] probability matrix (single model or
    ensemble-averaged)."""
    pred = P.argmax(1) + 1
    y, amb = split["y"], split["amb"]
    res = {
        "exact": float((pred == y).mean()),
        "within1": float((np.abs(pred - y) <= 1).mean()),
        "mae": float(np.abs(pred - y).mean()),
        "lenient": float(
            np.mean([p in v for p, v in zip(pred, split["valid"], strict=True)])
        ),
        "_pred": pred,
    }
    # Calibration on the genuinely-ambiguous words: does the model keep the
    # spread (top-mass support covers the valid set), and how much mass sits
    # on valid counts.
    ai = np.where(amb)[0]
    if len(ai):
        res["amb_cover"] = float(
            np.mean([_covers(P[i], split["valid"][i]) for i in ai])
        )
        res["amb_mass"] = float(
            np.mean([sum(P[i, c - 1] for c in split["valid"][i]) for i in ai])
        )
        res["amb_n"] = int(len(ai))
    return res


def evaluate(forward, params, split):
    return _metrics(_predict_probs(forward, params, split["X"]), split)


def train(
    ds,
    cfg,
    name="run",
    max_epochs=200,
    batch=512,
    lr=2e-3,
    wd=1e-4,
    warmup_epochs=3,
    eval_every=2,
    patience=8,
    seed=0,
    results_log=None,
    ckpt_path=None,
):
    caps, drop = cfg["caps"], cfg["drop"]
    eval_forward = jax.jit(model.forward)
    tr = ds["train"]
    Xtr, Ytr = jnp.asarray(tr["X"]), jnp.asarray(tr["Y"])
    n_full = (len(Xtr) // batch) * batch
    steps_per_epoch = n_full // batch
    if steps_per_epoch == 0:
        raise SystemExit(
            f"[{name}] train split has {len(Xtr)} examples < batch={batch}; "
            f"there is nothing to train (would silently ship an untrained "
            f"model). Use a larger split or a smaller batch."
        )

    params = model.init_params(
        jax.random.PRNGKey(seed), ds["vocab_size"], data.NUM_CLASSES, **caps
    )
    n_params = sum(a.size for a in jax.tree_util.tree_leaves(params))

    total_steps = max_epochs * steps_per_epoch
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps // 5)
    sched = optax.warmup_cosine_decay_schedule(
        0.0,
        lr,
        warmup_steps,
        total_steps,
        lr * 0.05,
    )
    opt = optax.adamw(sched, weight_decay=wd)
    opt_state = opt.init(params)

    def loss_fn(params, x, Y, key):
        logits = model.forward(params, x, key=key, drop=drop)
        # soft cross-entropy against the uniform-over-valid-set target
        return -(Y * jax.nn.log_softmax(logits)).sum(-1).mean()

    @jax.jit
    def step(params, opt_state, x, Y, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, Y, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    print(f"[{name}] params={n_params}  caps={caps}  drop={drop}", flush=True)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    best_val, best_epoch, since = -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(len(Xtr))
        for i in range(0, n_full, batch):
            idx = perm[i : i + batch]
            key, sk = jax.random.split(key)
            params, opt_state, loss = step(params, opt_state, Xtr[idx], Ytr[idx], sk)
        if epoch % eval_every == 0 or epoch == max_epochs:
            if len(ds["val"]["y"]):
                v = evaluate(eval_forward, params, ds["val"])
                _jsonl(
                    results_log,
                    {
                        "event": "val",
                        "config": name,
                        "epoch": epoch,
                        "elapsed": round(time.time() - t0, 1),
                        "loss": float(loss),
                        **{
                            k: v[k]
                            for k in (
                                "exact",
                                "within1",
                                "mae",
                                "lenient",
                                "amb_cover",
                                "amb_mass",
                            )
                        },
                    },
                )
                print(
                    f"  [{name}] ep {epoch:3d}  loss={float(loss):.3f}  "
                    f"val_exact={v['exact']:.4f}  "
                    f"amb_cover={v.get('amb_cover', 0):.3f}  "
                    f"best={max(best_val, 0):.4f}",
                    flush=True,
                )
                if v["exact"] > best_val:
                    best_val, best_epoch, since = v["exact"], epoch, 0
                    if ckpt_path is not None:
                        model.save(
                            ckpt_path,
                            params,
                            caps,
                            provenance={
                                "name": name,
                                "epoch": epoch,
                                "best_val": round(best_val, 4),
                                "date": time.strftime("%Y-%m-%d"),
                            },
                        )
                else:
                    since += 1
                    if since >= patience:
                        print(
                            f"  [{name}] early stop (no gain {patience} evals)",
                            flush=True,
                        )
                        break
            else:  # all-data run: no holdout -> log loss, no early stop
                print(
                    f"  [{name}] ep {epoch:3d}  loss={float(loss):.3f} (all-data)",
                    flush=True,
                )
                _jsonl(
                    results_log,
                    {
                        "event": "train",
                        "config": name,
                        "epoch": epoch,
                        "loss": float(loss),
                        "elapsed": round(time.time() - t0, 1),
                    },
                )
    secs = time.time() - t0
    if ckpt_path is not None and not len(ds["val"]["y"]):
        model.save(
            ckpt_path,
            params,
            caps,
            provenance={
                "name": name,
                "epochs": max_epochs,
                "all_data": True,
                "date": time.strftime("%Y-%m-%d"),
                "train_words": int(len(tr["y"])),
            },
        )
    _jsonl(
        results_log,
        {
            "event": "result",
            "config": name,
            "params": int(n_params),
            "secs": round(secs, 1),
            "best_val": best_val,
            "best_epoch": best_epoch,
        },
    )
    return {
        "name": name,
        "params": n_params,
        "secs": secs,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }


def _final_report(ds, ckpt, log):
    """Reload best checkpoint, report test + per-count."""
    params, _ = model.load(ckpt)
    fwd = jax.jit(model.forward)
    t = evaluate(fwd, params, ds["test"])
    yt = ds["test"]["y"]
    tail = yt >= 4
    per = {
        int(c): round(float((t["_pred"][yt == c] == c).mean()), 3)
        for c in range(1, data.NUM_CLASSES + 1)
        if (yt == c).sum()
    }
    rec = {
        "event": "test",
        "exact": t["exact"],
        "within1": t["within1"],
        "mae": t["mae"],
        "lenient": t["lenient"],
        "tail4plus": float((t["_pred"][tail] == yt[tail]).mean()),
        "amb_cover": t.get("amb_cover"),
        "amb_mass": t.get("amb_mass"),
        "amb_n": t.get("amb_n"),
        "per_count": per,
    }
    _jsonl(log, rec)
    print(
        f"\nTEST  exact={t['exact']:.4f}  within1={t['within1']:.4f}  "
        f"mae={t['mae']:.4f}  lenient={t['lenient']:.4f}  "
        f"tail4+={rec['tail4plus']:.4f}"
    )
    if t.get("amb_n"):
        print(
            f"  CALIBRATION (ambiguous n={t['amb_n']}): "
            f"cover@0.9={t['amb_cover']:.3f}  "
            f"valid_mass={t['amb_mass']:.3f}"
        )
    print("  per-count exact:", per)


def ensemble_report(ds, ckpts, log, split_name):
    """Per-seed mean+/-std (error bars) and the averaged-softmax ensemble.

    Under split='hard' the ambiguous test words are unseen shapes, i.e.
    the OOV-ambiguous case the model's calibration actually has to own.
    """
    fwd = jax.jit(model.forward)
    sp = ds["test"]
    Ps, per = [], []
    for cp in ckpts:
        p, _ = model.load(cp)
        P = _predict_probs(fwd, p, sp["X"])
        Ps.append(P)
        per.append(_metrics(P, sp))
    ens = _metrics(np.mean(Ps, axis=0), sp)

    def ms(k):
        v = [m[k] for m in per]
        return float(np.mean(v)), float(np.std(v))

    print(
        f"\n=== {len(ckpts)}-seed ensemble | split={split_name} | "
        f"ambiguous n={ens.get('amb_n')} "
        f"({'OOV-ambiguous' if split_name == 'hard' else 'mixed'}) ==="
    )
    print(f"  {'metric':10s} {'per-seed mean±std':>20s}  {'ensemble':>9s}")
    rec = {"event": "ensemble", "split": split_name, "seeds": len(ckpts)}
    for k in ("exact", "amb_cover", "amb_mass"):
        m, s = ms(k)
        print(f"  {k:10s} {m:8.4f} ± {s:.4f}      {ens.get(k, 0):.4f}")
        rec[k] = {"mean": m, "std": s, "ensemble": ens.get(k)}
    _jsonl(log, rec)


def summarize(path):
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r["event"] == "val":
            print(
                f"  [{r['config']}] ep {r['epoch']:3d}  "
                f"val_exact={r['exact']:.4f}  ({r['elapsed']}s)"
            )
        elif r["event"] == "result":
            print(
                f"== {r['config']}: best_val={r['best_val']:.4f} "
                f"@ep{r['best_epoch']}  params={r['params']}  {r['secs']}s"
            )
        elif r["event"] == "train":
            print(
                f"  [{r['config']}] ep {r['epoch']:3d}  loss={r['loss']:.3f} (all-data)"
            )
        elif r["event"] == "test":
            print(f"\nTEST {r}")
        elif r["event"] == "ensemble":
            body = "  ".join(
                f"{k}={r[k]}" for k in ("exact", "amb_cover", "amb_mass") if k in r
            )
            print(f"\nENSEMBLE[{r['split']}, {r['seeds']} seeds]  {body}")
        else:
            print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--shrink", action="store_true")
    ap.add_argument("--config", default="s7k", choices=list(SHRINK))
    ap.add_argument(
        "--split", default="random", choices=["random", "hard", "source_oov", "all"]
    )
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--search-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", metavar="PATH")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return

    if args.tune and args.split == "all":
        ap.error(
            "--tune selects a config by held-out val accuracy; "
            "--split all has no val set, so the choice would be "
            "arbitrary. Tune on random/hard, then ship with --split all."
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = RESULTS_DIR / f"tune-{stamp}.jsonl"
    # An all-data run produces the deployable artifact -> write it into the
    # package (ships in the wheel). Experiment runs stay in results/.
    ckpt = (
        model.DEFAULT_WEIGHTS
        if args.split == "all"
        else RESULTS_DIR / "ckpt" / "model.npz"
    )
    print(f"streaming to {log}", flush=True)

    ds = data.load(seed=args.seed, split=args.split)
    print(
        f"split={args.split}  train={len(ds['train']['y'])}  "
        f"val={len(ds['val']['y'])}  test={len(ds['test']['y'])}  "
        f"vocab={ds['vocab_size']}",
        flush=True,
    )

    if args.ensemble > 1:
        ckpts = []
        for s in range(args.ensemble):
            cp = RESULTS_DIR / "ckpt" / f"ens_{args.config}_s{s}.npz"
            train(
                ds,
                SHRINK[args.config],
                name=f"{args.config}_s{s}",
                max_epochs=args.epochs,
                patience=10,
                seed=s,
                results_log=log,
                ckpt_path=cp,
            )
            ckpts.append(cp)
        ensemble_report(ds, ckpts, log, args.split)
        print(f"\nresults: {log}")
        return

    if args.shrink:
        rows = []
        for nm, cfg in SHRINK.items():
            r = train(
                ds,
                cfg,
                name=nm,
                max_epochs=args.search_epochs,
                seed=args.seed,
                results_log=log,
                ckpt_path=RESULTS_DIR / "ckpt" / f"{nm}.npz",
            )
            rows.append(r)
        print("\n" + "=" * 56)
        print(f"{'config':8s} {'params':>8s} {'best_val':>9s} {'@ep':>4s} {'sec':>6s}")
        print("-" * 56)
        for r in sorted(rows, key=lambda r: -r["params"]):
            print(
                f"{r['name']:8s} {r['params']:8d} {r['best_val']:9.4f} "
                f"{r['best_epoch']:4d} {r['secs']:6.0f}"
            )
        print("=" * 56)
        print(f"results: {log}  (checkpoints in {RESULTS_DIR / 'ckpt'})")
        return

    if args.tune:
        results = []
        for nm, cfg in CONFIGS.items():
            r = train(
                ds,
                cfg,
                name=nm,
                max_epochs=args.search_epochs,
                seed=args.seed,
                results_log=log,
                ckpt_path=RESULTS_DIR / "ckpt" / f"search_{nm}.npz",
            )
            results.append(r)
        best = max(results, key=lambda r: r["best_val"])
        print(
            f"\nwinner: {best['name']} (val={best['best_val']:.4f}); "
            f"converging to {args.epochs} epochs",
            flush=True,
        )
        train(
            ds,
            CONFIGS[best["name"]],
            name=f"final_{best['name']}",
            max_epochs=args.epochs,
            patience=15,
            seed=args.seed,
            results_log=log,
            ckpt_path=ckpt,
        )
    else:
        train(
            ds,
            SHRINK[args.config],
            name=args.config,
            max_epochs=args.epochs,
            patience=15,
            seed=args.seed,
            results_log=log,
            ckpt_path=ckpt,
        )

    if args.split != "all":  # no held-out test on an all-data run
        _final_report(ds, ckpt, log)
    print(f"\ncheckpoint: {ckpt}   results: {log}")


if __name__ == "__main__":
    main()
