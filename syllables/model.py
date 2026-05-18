"""The syllable-count model: char CNN -> BiGRU -> masked mean -> MLP.

Pure JAX + numpy at serve time -- no neural-net library, no optax (that
is training-only). ~7.4k params for the shipped `s7k` config.

Intuition: the conv spots local vowel-group / spelling motifs; the BiGRU
scans the word both ways to tally them and apply context-dependent
corrections (silent-e, -ed, digraphs, morphology); masked mean pools the
accumulated evidence; the head emits a distribution over syllable counts.

Padding (id 0) is fully inert: its embedding is zeroed before the conv
(so conv windows at the word edge see clean zero-padding), and the GRU
*carries its state through* padded steps in both directions, so a padded
position never updates the recurrence and never enters the mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# --- inference contract (pinned; serving never imports data.py) ----------- #
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOCAB = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = PAD
VOCAB_SIZE = len(ALPHABET) + 1  # 27
MAX_LEN = 24
MAX_COUNT = 8
NUM_CLASSES = MAX_COUNT  # class index = count - 1
PAD = 0
FORMAT_VERSION = 1


def encode(word: str) -> np.ndarray:
    """Word -> fixed-length char ids. Total function: lowercase, keep only
    a-z (the pinned alphabet), truncate/pad to MAX_LEN. This *is* the
    train/serve input contract -- the only text->model mapping anywhere."""
    ids = [VOCAB[c] for c in word.lower() if c in VOCAB][:MAX_LEN]
    ids += [PAD] * (MAX_LEN - len(ids))
    return np.array(ids, dtype=np.int32)


EMBED_DIM = 24
CONV_WIDTHS = (2, 3)
CONV_FILTERS = 32
GRU_HIDDEN = 48
HEAD_HIDDEN = 64


def _scaled_init(key, shape, fan_in):  # std = 1/sqrt(fan_in) (LeCun-style)
    return jax.random.normal(key, shape) * (1.0 / fan_in) ** 0.5


def _mask(x):  # [B, L] int -> [B, L] float, 1 where real char
    return (x != 0).astype(jnp.float32)


def _masked_mean(h, m):  # h: [B, L, F], m: [B, L]
    return (h * m[..., None]).sum(1) / jnp.clip(m.sum(1, keepdims=True), 1.0)


def _conv_stack(params, e):  # e: [B, L, D] -> [B, L, F*nwidths]
    feats = []
    for w in sorted(params["convs"]):  # widths inferred from params
        c = params["convs"][w]
        z = jax.lax.conv_general_dilated(
            e, c["w"], (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC")
        )
        feats.append(jax.nn.relu(z + c["b"]))
    return jnp.concatenate(feats, axis=-1)


def _init_gru(key, in_dim, hid):
    k = jax.random.split(key, 2)
    return {
        "W": _scaled_init(k[0], (in_dim, 3 * hid), in_dim),
        "U": _scaled_init(k[1], (hid, 3 * hid), hid),
        "b": jnp.zeros(3 * hid),
        "h0": jnp.zeros(hid),
    }


def _gru_scan(gp, seq, mask):  # seq [B,L,in], mask [B,L] -> outputs [B,L,hid]
    hid = gp["h0"].shape[0]
    seq_t = jnp.transpose(seq, (1, 0, 2))  # [L, B, in]
    mask_t = jnp.transpose(mask, (1, 0))  # [L, B]

    def cell(h, xm):
        x, m = xm
        gx, gh = x @ gp["W"], h @ gp["U"]
        zr = jax.nn.sigmoid(gx[:, : 2 * hid] + gh[:, : 2 * hid] + gp["b"][: 2 * hid])
        z, r = zr[:, :hid], zr[:, hid:]
        n = jnp.tanh(gx[:, 2 * hid :] + r * gh[:, 2 * hid :] + gp["b"][2 * hid :])
        h_new = (1 - z) * n + z * h
        m = m[:, None]
        h = m * h_new + (1 - m) * h  # padded step: carry state unchanged
        return h, h

    h0 = jnp.broadcast_to(gp["h0"], (seq.shape[0], hid))
    _, outs = jax.lax.scan(cell, h0, (seq_t, mask_t))
    return jnp.transpose(outs, (1, 0, 2))


def _bigru(params, seq, mask):
    fwd = _gru_scan(params["gru_f"], seq, mask)
    bwd = jnp.flip(
        _gru_scan(params["gru_b"], jnp.flip(seq, axis=1), jnp.flip(mask, axis=1)),
        axis=1,
    )
    return jnp.concatenate([fwd, bwd], axis=-1)  # [B, L, 2*hid]


CAP_DEFAULTS = dict(
    embed_dim=EMBED_DIM,
    conv_widths=list(CONV_WIDTHS),
    conv_filters=CONV_FILTERS,
    gru_hidden=GRU_HIDDEN,
    head_hidden=HEAD_HIDDEN,
)


def init_params(key, vocab_size: int, num_classes: int, **caps) -> dict:
    c = {**CAP_DEFAULTS, **caps}
    ed, fil, gh, hh = (
        c["embed_dim"],
        c["conv_filters"],
        c["gru_hidden"],
        c["head_hidden"],
    )
    widths = list(c["conv_widths"])
    k = jax.random.split(key, 5 + len(widths))
    p = {"emb": _scaled_init(k[0], (vocab_size, ed), ed), "convs": {}}
    for w, ck in zip(widths, k[5:], strict=True):
        p["convs"][w] = {
            "w": _scaled_init(ck, (w, ed, fil), w * ed),
            "b": jnp.zeros(fil),
        }
    fc = fil * len(widths)
    p["gru_f"] = _init_gru(k[1], fc, gh)
    p["gru_b"] = _init_gru(k[2], fc, gh)
    p["W1"] = _scaled_init(k[3], (2 * gh, hh), 2 * gh)
    p["b1"] = jnp.zeros(hh)
    p["W2"] = _scaled_init(k[4], (hh, num_classes), hh)
    p["b2"] = jnp.zeros(num_classes)
    return p


def forward(params, x, key=None, drop=0.0):
    """x: int32 [B, L] -> logits [B, num_classes].

    Pass (key, drop>0) only during training for dropout on the pooled
    representation; eval/inference use the defaults (no dropout).
    """
    m = _mask(x)
    e = params["emb"][x] * m[..., None]  # zero out PAD embeddings
    h = _conv_stack(params, e)
    h = _bigru(params, h, m)
    z = _masked_mean(h, m)
    z = jax.nn.relu(z @ params["W1"] + params["b1"])
    if drop > 0.0 and key is not None:
        keep = 1.0 - drop
        z = z * jax.random.bernoulli(key, keep, z.shape) / keep
    return z @ params["W2"] + params["b2"]


def save(path, params, caps=None, provenance=None):
    """One self-contained .npz: positional param leaves + a `meta` JSON
    (alphabet, num_classes, max_len, caps, param paths+shapes, provenance).
    No pickle, no sidecar."""
    caps = {**CAP_DEFAULTS, **(caps or {})}
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    leaves = [np.asarray(v) for _, v in flat]
    meta = {
        "format_version": FORMAT_VERSION,
        "alphabet": ALPHABET,
        "num_classes": NUM_CLASSES,
        "max_len": MAX_LEN,
        "caps": caps,
        "param_paths": [jax.tree_util.keystr(p) for p, _ in flat],
        "param_shapes": [list(a.shape) for a in leaves],
        "provenance": provenance or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        meta=np.array(json.dumps(meta)),
        **{f"p{i}": a for i, a in enumerate(leaves)},
    )


def _skeleton(caps):
    return init_params(jax.random.PRNGKey(0), VOCAB_SIZE, NUM_CLASSES, **caps)


# The shipped model lives inside the package, so `Model.load()` is
# zero-config and the weights travel in the wheel.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "model.npz"


def load(path=None):
    """Validated load -> (params, meta). Loud on any train/serve skew.
    `path=None` loads the weights shipped inside the package."""
    path = DEFAULT_WEIGHTS if path is None else path
    z = np.load(str(path), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta["alphabet"] != ALPHABET:
        raise ValueError("artifact alphabet != code contract")
    if meta["num_classes"] != NUM_CLASSES or meta["max_len"] != MAX_LEN:
        raise ValueError("artifact num_classes/max_len != code contract")
    flat, treedef = jax.tree_util.tree_flatten_with_path(_skeleton(meta["caps"]))
    if [jax.tree_util.keystr(p) for p, _ in flat] != meta["param_paths"]:
        raise ValueError("artifact param structure != current init_params")
    leaves = []
    for i, (path_i, ref) in enumerate(flat):
        a = jnp.asarray(z[f"p{i}"])
        if list(a.shape) != list(ref.shape):
            raise ValueError(f"shape mismatch at {jax.tree_util.keystr(path_i)}")
        leaves.append(a)
    return jax.tree_util.tree_unflatten(treedef, leaves), meta


class Model:
    """Serving API. Self-contained: needs only the .npz + jax/numpy; does
    NOT import data.py or parse any lexicon.

    `logits`/`probs` are the only primitives -- the raw softmax. The
    abstention policy (the `fuzziness` gate) and all line aggregation /
    normalization / lookups live one layer up in `harness.py`, so there
    is exactly one policy in the system, not a second one buried here."""

    def __init__(self, params, meta):
        self.params, self.meta = params, meta
        self._fwd = jax.jit(forward)

    @classmethod
    def load(cls, path=None, **kw):
        params, meta = load(path)
        return cls(params, meta, **kw)

    def logits(self, words) -> np.ndarray:
        """words (str or iterable) -> raw logits [n, NUM_CLASSES];
        column c-1 corresponds to syllable count c."""
        if isinstance(words, str):
            words = [words]
        words = list(words)
        if not words:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        X = jnp.asarray(np.stack([encode(w) for w in words]))
        return np.asarray(self._fwd(self.params, X))

    def probs(self, words) -> np.ndarray:
        z = self.logits(words)
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
