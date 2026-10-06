"""Pre-compute the tokenization of all T_ws so similarity_eval.py can analyze
bigrams without re-tokenizing.

Writes data/t_ws/llama/bigrams.npz (offsets + concatenated token ids) and
data/t_ws/llama/bigrams_meta.json.

Usage:
    python -m src.experiments.main.compute_t_w_bigrams
"""

import json
import argparse
from src.util.experiment_profile import ExperimentProfile, add_profile_argument
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
T_WS_PATH = REPO_ROOT / "data" / "t_ws" / "llama" / "combined_t_ws.json"
OUT_NPZ = REPO_ROOT / "data" / "t_ws" / "llama" / "bigrams.npz"
OUT_META = REPO_ROOT / "data" / "t_ws" / "llama" / "bigrams_meta.json"

TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 512


def main():
    global T_WS_PATH, OUT_NPZ, OUT_META, TOKENIZER_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    profile = ExperimentProfile(parser.parse_args().profile)
    T_WS_PATH = REPO_ROOT / profile.corpus_dir / "combined_t_ws.json"
    OUT_NPZ = REPO_ROOT / profile.corpus_dir / "bigrams.npz"
    OUT_META = REPO_ROOT / profile.corpus_dir / "bigrams_meta.json"
    TOKENIZER_NAME = profile.model
    print(f"loading T_ws from {T_WS_PATH} ...")
    with open(T_WS_PATH) as f:
        t_ws = json.load(f)
    print(f"  -> {len(t_ws)} texts")

    print(f"loading tokenizer {TOKENIZER_NAME} ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    t0 = time.time()
    token_ids = []
    for i in range(0, len(t_ws), BATCH_SIZE):
        chunk = t_ws[i:i + BATCH_SIZE]
        enc = tok(chunk, add_special_tokens=False)
        token_ids.extend(enc["input_ids"])
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  tokenized {i + len(chunk)}/{len(t_ws)}")
    print(f"tokenization took {time.time() - t0:.1f}s")

    lengths = np.array([len(t) for t in token_ids], dtype=np.int64)
    total = int(lengths.sum())
    offsets = np.zeros(len(token_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    tokens_flat = np.empty(total, dtype=np.int32)
    for i, t in enumerate(token_ids):
        tokens_flat[offsets[i]:offsets[i + 1]] = t

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, offsets=offsets, tokens=tokens_flat)
    print(f"wrote {OUT_NPZ}  ({OUT_NPZ.stat().st_size / 1e6:.1f} MB)")

    n_bigrams = np.maximum(lengths - 1, 0).astype(np.int64)
    meta = {
        "tokenizer": TOKENIZER_NAME,
        "add_special_tokens": False,
        "n_samples": int(len(token_ids)),
        "n_tokens_total": total,
        "n_bigrams_total": int(n_bigrams.sum()),
        "token_len_mean": float(lengths.mean()),
        "token_len_min": int(lengths.min()),
        "token_len_max": int(lengths.max()),
        "bigram_len_mean": float(n_bigrams.mean()),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT_META}")


if __name__ == "__main__":
    main()
