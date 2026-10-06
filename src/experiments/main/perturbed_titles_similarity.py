"""
Computes cosine similarity between original titles and perturbed variants.

Writes results (overall and per variant) to
results/auxiliary/perturbed_titles_similarity/mpnet_similarity.json.

Usage:
    python -m src.experiments.main.perturbed_titles_similarity
"""

import sys
import argparse
from src.util.experiment_profile import ExperimentProfile, add_profile_argument

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.util.filereader import (
    load_path_file,
    load_titles,
    write_path_file,
)


EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE = 128
PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
N_SAMPLES = 64000
VARIANTS = ("titles_1", "titles_2", "titles_3")


def embed(model, texts):
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def summarise(scores):
    pct = np.asarray(PERCENTILES, dtype=np.float64)
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "percentiles": pct.tolist(),
        "percentile_values": np.percentile(scores, pct)
            .astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    profile = ExperimentProfile(parser.parse_args().profile)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {EMBED_MODEL_NAME} on {device} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    originals = load_titles(N_SAMPLES)
    variants = {
        name: load_path_file(["data", "prompts", "shared"], f"{name}.json")
        for name in VARIANTS
    }
    for name, v in variants.items():
        assert len(v) == len(originals), (
            f"{name} length {len(v)} != originals {len(originals)}"
        )
    print(f"loaded {len(originals)} originals and {len(VARIANTS)} variants")

    print("embedding originals ...")
    orig_emb = embed(model, originals)

    per_variant = {}
    all_scores = []
    for name in VARIANTS:
        print(f"embedding {name} ...")
        v_emb = embed(model, variants[name])
        scores = (orig_emb * v_emb).sum(axis=1)
        per_variant[name] = summarise(scores)
        all_scores.append(scores)
        print(f"  {name}: mean={per_variant[name]['mean']:.4f} "
              f"min={per_variant[name]['min']:.4f} "
              f"max={per_variant[name]['max']:.4f}")

    combined = np.concatenate(all_scores)
    overall = summarise(combined)

    result = {
        "embed_model": EMBED_MODEL_NAME,
        "n_originals": len(originals),
        "variants": list(VARIANTS),
        "overall": overall,
        "per_variant": per_variant,
    }

    print("\noverall (n={}): mean={:.4f}  min={:.4f}  max={:.4f}  std={:.4f}"
          .format(overall["n"], overall["mean"], overall["min"],
                  overall["max"], overall["std"]))

    write_path_file(
        ["results", "auxiliary", "perturbed_titles_similarity" + profile.suffix],
        "mpnet_similarity.json",
        result,
    )
    print(f"\nwrote results/auxiliary/perturbed_titles_similarity{profile.suffix}/mpnet_similarity.json")


if __name__ == "__main__":
    main()
