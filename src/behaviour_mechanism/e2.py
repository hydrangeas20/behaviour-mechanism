"""E2: offset-randomised repeat task (generator version ``e2-v1``).

This is a **new protocol**. Its results must not be pooled with the historical
fixed-offset experiment.

Sequence construction
---------------------
Every sequence has ``seq_len`` tokens drawn uniformly from the vocabulary. With probability
``frac_repeat`` it is a *repeat sequence*: a chunk length ``L`` is drawn uniformly from
``train_lengths``, a start ``s`` uniformly from ``[0, seq_len - 2L]``, and
``tokens[s+L : s+2L] = tokens[s : s+L]``. Otherwise it stays fully random.

The repeat offset therefore equals ``L`` and varies per sequence, and the absolute
position of the repeat varies with ``s``. No constant offset and no absolute position
predicts the repeated tokens (see :func:`baseline_table`).

Labels
------
Next-token prediction. The *scored* targets are positions ``s+L+1 .. s+2L-1``: the current
token (at ``p-1``) has an earlier occurrence at ``p-1-L`` whose successor is the target.
The first repeated token (``s+L``) is excluded, because token matching cannot predict it.
In training, scored positions are up-weighted by the historical factor
(``config.INDUCTION_LOSS_WEIGHT``).

Splits and leakage controls
---------------------------
All data are generated, never stored. Each split has its own RNG stream:
``default_rng([data_seed, split_id, index])`` with ``split_id`` 0 = train (index = step),
1 = validation, 2 = test. Chunk lengths in ``heldout_lengths`` never appear in training
and are evaluated separately, as are ``extrapolation_lengths`` outside the training range.
:func:`sequence_hashes` supports exact-duplicate checks between splits.

Remaining shortcuts (documented, not eliminated)
------------------------------------------------
* The repeat is adjacent, so offset = chunk length. A model could learn a family of
  positional copiers plus a length detector; the held-out and extrapolation lengths and
  the non-adjacent *gap probe* (:func:`make_gap_probe_batch`) test for that.
* Tokens are sampled with replacement, so the current token can have several earlier
  occurrences. Single-token matching is ambiguous then; see the reference solutions.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

GENERATOR_VERSION = "e2-v1"
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class E2Config:
    vocab_size: int = 100
    seq_len: int = 64
    len_min: int = 8
    len_max: int = 24
    heldout_lengths: tuple[int, ...] = (11, 17, 22)
    extrapolation_lengths: tuple[int, ...] = (4, 5, 6, 28, 30, 32)
    frac_repeat: float = 0.7
    data_seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.len_min <= self.len_max <= self.seq_len // 2:
            raise ValueError("need 2 <= len_min <= len_max <= seq_len // 2")
        if any(not self.len_min <= h <= self.len_max for h in self.heldout_lengths):
            raise ValueError("held-out lengths must lie inside [len_min, len_max]")
        if any(self.len_min <= e <= self.len_max for e in self.extrapolation_lengths):
            raise ValueError("extrapolation lengths must lie outside [len_min, len_max]")
        if any(not 2 <= e <= self.seq_len // 2 for e in self.extrapolation_lengths):
            raise ValueError("extrapolation lengths must fit in the sequence")
        if not self.train_lengths:
            raise ValueError("no training lengths left")

    @property
    def train_lengths(self) -> tuple[int, ...]:
        return tuple(
            l for l in range(self.len_min, self.len_max + 1) if l not in self.heldout_lengths
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "generator_version": GENERATOR_VERSION}


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray  # (B, T) int
    is_repeat: np.ndarray  # (B,) bool
    start: np.ndarray  # (B,) int, -1 for non-repeat
    length: np.ndarray  # (B,) int, 0 for non-repeat
    scored: np.ndarray  # (B, T) bool: target positions token matching can predict


def split_rng(e2: E2Config, split: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng([e2.data_seed, SPLIT_IDS[split], index])


def make_batch(
    rng: np.random.Generator,
    e2: E2Config,
    n: int,
    lengths: tuple[int, ...] | None = None,
    frac_repeat: float | None = None,
    gap: int = 0,
) -> Batch:
    """``lengths`` defaults to the training lengths. ``gap > 0`` inserts filler between the
    chunk and its repeat (offset = L + gap); it is used only by the gap probe."""
    lengths = e2.train_lengths if lengths is None else lengths
    frac = e2.frac_repeat if frac_repeat is None else frac_repeat
    t = e2.seq_len
    if any(2 * l + gap > t for l in lengths):
        raise ValueError("chunk, gap and repeat do not fit in the sequence")
    tokens = rng.integers(0, e2.vocab_size, size=(n, t))
    is_rep = rng.random(n) < frac
    length = np.where(is_rep, rng.choice(np.asarray(lengths), size=n), 0)
    start = np.where(is_rep, rng.integers(0, t - 2 * length - gap + 1), -1)
    scored = np.zeros((n, t), dtype=bool)
    for i in np.flatnonzero(is_rep):
        s, l = int(start[i]), int(length[i])
        r = s + l + gap
        tokens[i, r : r + l] = tokens[i, s : s + l]
        scored[i, r + 1 : r + l] = True
    return Batch(tokens, is_rep, start, length, scored)


def make_gap_probe_batch(rng: np.random.Generator, e2: E2Config, n: int, gap: int) -> Batch:
    """Repeat sequences whose repeat is *not* adjacent (never seen in training)."""
    lengths = tuple(l for l in e2.train_lengths if 2 * l + gap <= e2.seq_len)
    return make_batch(rng, e2, n, lengths=lengths, frac_repeat=1.0, gap=gap)


def make_shuffled_repeat_batch(rng: np.random.Generator, e2: E2Config, n: int) -> Batch:
    """Control: the second chunk is a random permutation of the first. Same tokens, wrong
    order, so successor copying should mostly fail. ``scored`` marks the same positions as
    an ordinary repeat; the *label* there is the actual (shuffled) token."""
    b = make_batch(rng, e2, n, frac_repeat=1.0)
    tokens = b.tokens.copy()
    for i in range(n):
        s, l = int(b.start[i]), int(b.length[i])
        tokens[i, s + l : s + 2 * l] = rng.permutation(tokens[i, s : s + l])
    return Batch(tokens, b.is_repeat, b.start, b.length, b.scored)


def sequence_hashes(tokens: np.ndarray) -> set[bytes]:
    return {hashlib.blake2b(row.tobytes(), digest_size=12).digest() for row in tokens}


# ---------------------------------------------------------------------------
# Rule-based predictors: shortcut baselines and reference solutions
# ---------------------------------------------------------------------------
# Each returns predictions for positions 1..T-1 given positions 0..p-1, shape (B, T-1),
# aligned with ``tokens[:, 1:]``.


def predict_fixed_offset(tokens: np.ndarray, k: int) -> np.ndarray:
    """Predict ``tokens[p - k]`` for target ``p`` (the historical shortcut uses k = T/2)."""
    t = tokens.shape[1]
    pred = np.full((tokens.shape[0], t - 1), -1)
    if k < t:
        pred[:, k - 1 :] = tokens[:, : t - k]
    return pred


def predict_constant(tokens: np.ndarray, token: int) -> np.ndarray:
    return np.full((tokens.shape[0], tokens.shape[1] - 1), token)


def predict_by_position(tokens: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``table[p]`` is the most frequent training token at position ``p``."""
    return np.broadcast_to(table[1:], (tokens.shape[0], tokens.shape[1] - 1)).copy()


def predict_bigram(tokens: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``table[v]`` is the most frequent training successor of token ``v``."""
    pred: np.ndarray = table[tokens[:, :-1]]
    return pred


def predict_previous_token(tokens: np.ndarray) -> np.ndarray:
    return tokens[:, :-1].copy()


def predict_last_occurrence(tokens: np.ndarray) -> np.ndarray:
    """Reference A (single-token induction): find the most recent earlier occurrence of the
    current token and predict its successor; -1 if there is none."""
    b, t = tokens.shape
    pred = np.full((b, t - 1), -1)
    for i in range(b):
        last: dict[int, int] = {}
        row = tokens[i]
        for q in range(t - 1):
            j = last.get(int(row[q]))
            if j is not None:
                pred[i, q] = row[j + 1]
            last[int(row[q])] = q
    return pred


def predict_longest_suffix(tokens: np.ndarray) -> np.ndarray:
    """Reference B: among earlier positions holding the current token, choose the one whose
    preceding context matches the current context for the longest stretch (ties: most
    recent); predict its successor."""
    b, t = tokens.shape
    pred = np.full((b, t - 1), -1)
    for i in range(b):
        row = tokens[i]
        for q in range(1, t - 1):
            best_j, best_len = -1, 0
            for j in range(q - 1, -1, -1):
                if row[j] != row[q]:
                    continue
                k = 1
                while j - k >= 0 and row[j - k] == row[q - k]:
                    k += 1
                if k > best_len:
                    best_j, best_len = j, k
            if best_j >= 0:
                pred[i, q] = row[best_j + 1]
    return pred


def fit_tables(e2: E2Config, n_batches: int = 200, batch_size: int = 64) -> dict[str, Any]:
    """Fit the count-based baselines on training batches ``0 .. n_batches-1``."""
    v, t = e2.vocab_size, e2.seq_len
    uni = np.zeros(v, dtype=np.int64)
    pos = np.zeros((t, v), dtype=np.int64)
    big = np.zeros((v, v), dtype=np.int64)
    for step in range(n_batches):
        tok = make_batch(split_rng(e2, "train", step), e2, batch_size).tokens
        uni += np.bincount(tok.ravel(), minlength=v)
        for p in range(t):
            pos[p] += np.bincount(tok[:, p], minlength=v)
        np.add.at(big, (tok[:, :-1].ravel(), tok[:, 1:].ravel()), 1)
    return {
        "unigram_token": int(uni.argmax()),
        "position_table": pos.argmax(axis=1),
        "bigram_table": big.argmax(axis=1),
        "n_sequences": n_batches * batch_size,
    }


def accuracy(pred: np.ndarray, batch: Batch) -> float:
    sel = batch.scored[:, 1:]
    return float((pred == batch.tokens[:, 1:])[sel].mean()) if sel.any() else float("nan")


def baseline_table(
    e2: E2Config, batch: Batch, tables: dict[str, Any], max_offset: int = 40
) -> list[dict[str, Any]]:
    """Accuracy of every shortcut baseline and reference solution on ``batch``'s scored
    positions."""
    tok = batch.tokens
    fixed = {k: accuracy(predict_fixed_offset(tok, k), batch) for k in range(1, max_offset + 1)}
    best_k = max(fixed, key=lambda k: fixed[k])
    half = e2.seq_len // 2
    rows = [
        ("chance (1 / vocab)", "shortcut", 1 / e2.vocab_size),
        (
            "token-frequency (most common token)",
            "shortcut",
            accuracy(predict_constant(tok, tables["unigram_token"]), batch),
        ),
        (
            "position-only (most common token per position)",
            "shortcut",
            accuracy(predict_by_position(tok, tables["position_table"]), batch),
        ),
        (
            "bigram (most common successor)",
            "shortcut",
            accuracy(predict_bigram(tok, tables["bigram_table"]), batch),
        ),
        ("previous-token copy", "shortcut", accuracy(predict_previous_token(tok), batch)),
        (f"fixed-offset copy, historical offset {half}", "shortcut", fixed[half]),
        (f"fixed-offset copy, best single offset ({best_k})", "shortcut", fixed[best_k]),
        (
            "reference A: last-occurrence induction",
            "reference",
            accuracy(predict_last_occurrence(tok), batch),
        ),
        (
            "reference B: longest-suffix induction",
            "reference",
            accuracy(predict_longest_suffix(tok), batch),
        ),
    ]
    return [{"predictor": n, "kind": k, "accuracy": a} for n, k, a in rows]


# ---------------------------------------------------------------------------
# Clean/corrupted pairs for the corrected measurement
# ---------------------------------------------------------------------------

E2_CORRUPTIONS = ("earlier_occurrence", "copy_source", "query_token", "random_repeat")


def make_e2_pairs(
    rng: np.random.Generator,
    e2: E2Config,
    n: int,
    corruption: str,
    lengths: tuple[int, ...] | None = None,
    distinct: bool = False,
) -> dict[str, np.ndarray]:
    """Pairs for a read position ``q`` inside the repeat (its output predicts ``q + 1``).

    With repeat start ``r = s + L``, ``q`` is uniform on ``[r, r + L - 2]``. The query token
    (at ``q``) first occurred at ``q - L``; the target equals the token at ``q - L + 1``.

    ``earlier_occurrence``  change the token at ``q - L`` only (breaks the match, keeps the
                            query token and the copy source);
    ``copy_source``         change the token at ``q - L + 1`` only (changes what is copied);
    ``query_token``         change the token at ``q`` only;
    ``random_repeat``       resample everything from ``r`` on (the analogue of the
                            historical corruption; it also changes the query token).

    ``distinct=True`` draws the base sequence without replacement and resamples to unused
    tokens, so the only duplicates are the repeat itself (needs ``vocab > seq_len + 1``).
    Returns ``clean``, ``corrupt``, ``target``, ``read_pos``, ``length``.
    """
    if corruption not in E2_CORRUPTIONS:
        raise ValueError(f"unknown corruption {corruption!r}")
    lengths = e2.train_lengths if lengths is None else lengths
    t, v = e2.seq_len, e2.vocab_size
    if distinct and v < t + 2:
        raise ValueError("distinct pairs need vocab_size >= seq_len + 2")
    clean = np.empty((n, t), dtype=np.int64)
    corrupt = np.empty((n, t), dtype=np.int64)
    target = np.empty(n, dtype=np.int64)
    read_pos = np.empty(n, dtype=np.int64)
    length = rng.choice(np.asarray(lengths), size=n)
    for i in range(n):
        l = int(length[i])
        s = int(rng.integers(0, t - 2 * l + 1))
        r = s + l
        if distinct:
            perm = rng.permutation(v)
            row, spare = perm[:t].copy(), perm[t:]
        else:
            row, spare = rng.integers(0, v, size=t), None
        row[r : r + l] = row[s : s + l]
        q = int(rng.integers(r, r + l - 1))

        def resample(orig: int, row: np.ndarray = row, spare: np.ndarray | None = spare) -> int:
            if spare is not None:
                return int(rng.choice(spare))
            return int((orig + rng.integers(1, v)) % v)

        bad = row.copy()
        if corruption == "earlier_occurrence":
            bad[q - l] = resample(int(row[q - l]))
        elif corruption == "copy_source":
            bad[q - l + 1] = resample(int(row[q - l + 1]))
        elif corruption == "query_token":
            bad[q] = resample(int(row[q]))
        else:
            if spare is not None:
                bad[r:] = rng.choice(spare, size=t - r)
            else:
                bad[r:] = rng.integers(0, v, size=t - r)
        clean[i], corrupt[i], target[i], read_pos[i] = row, bad, row[q + 1], q
    return {
        "clean": clean,
        "corrupt": corrupt,
        "target": target,
        "read_pos": read_pos,
        "length": length,
    }
