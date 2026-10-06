"""Sample the seeded dataset from the unarXive open subset.

First, we need to download the open subset of unarXive into data/unarxive_open first. 
Then use this script to first count the number of samples in all subsets of the dataset,
and create a seeded random sample of n abstracts in data/seeded_dataset.jsonl. 
Smaller samples are prefixes or larger samples, as the first
k lines of an n-sample run are equal to the full output of a k-sample run with the
same seed.

usage:
    python src/data_creation/sample_dataset.py --n 165000 --seed 42
"""

import argparse
import bisect
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def open_jsonl(path):
    """Open .jsonl or .jsonl.gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def list_dataset_files(root):
    """Lists all dataset files, in a deterministic order (sorted by year dirs and then
    sorted by filenames)."""
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"Dataset root does not exist or is not a directory: {root}")

    files = []
    for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for fp in sorted(year_dir.iterdir()):
            name = fp.name
            if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                files.append(str(fp))
    return files


def build_index(root, manifest_path):
    """Writes path, line count and offset of every file, so a
    global index can be translated back to (file, local line number)."""
    files = list_dataset_files(root)
    if not files:
        sys.exit(f"No .jsonl(.gz) files found under {root}")

    entries = []
    cum = 0
    for i, fp in enumerate(files, 1):
        count = 0
        with open_jsonl(fp) as f:
            for _ in f:
                count += 1
        entries.append({"path": fp, "count": count, "offset": cum})
        cum += count
        print(f"[{i}/{len(files)}] {fp}: {count} lines (cumulative {cum})")

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"total": cum, "files": entries}, f, indent=2)
    print(f"\nManifest written to {manifest_path}. Total samples: {cum}")


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    return m["files"], m["total"]


def selected_indices(total, n, seed):
    """First n entries of a deterministic permutation of [0, total)."""
    if n > total:
        raise ValueError(f"Requested n={n} but only {total} samples available")
    rng = random.Random(seed)
    perm = list(range(total))
    rng.shuffle(perm)
    return perm[:n]


def find_file(global_idx, offsets, entries):
    """Binary-search the manifest for the file containing a global index."""
    i = bisect.bisect_right(offsets, global_idx) - 1
    return entries[i], global_idx - offsets[i]


def process_sample(raw, global_idx=None):
    """Keep only abstract.text, metadata.title, metadata.id."""
    try:
        return {
            "id": raw["metadata"]["id"],
            "title": raw["metadata"]["title"],
            "abstract": raw["abstract"]["text"],
        }
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Sample at global index {global_idx} is missing a required field: {e}"
        )


def draw_samples(manifest_path, n, seed, out_path):
    entries, total = load_manifest(manifest_path)
    offsets = [e["offset"] for e in entries]

    selected = selected_indices(total, n, seed)

    by_file = defaultdict(dict)
    for out_pos, gidx in enumerate(selected):
        entry, local = find_file(gidx, offsets, entries)
        by_file[entry["path"]][local] = out_pos

    samples_by_pos = {}
    files_to_read = sorted(by_file.keys())
    for j, fp in enumerate(files_to_read, 1):
        wanted = by_file[fp]
        max_line = max(wanted)
        with open_jsonl(fp) as f:
            for i, line in enumerate(f):
                if i in wanted:
                    raw = json.loads(line)
                    samples_by_pos[wanted[i]] = process_sample(
                        raw, global_idx=selected[wanted[i]]
                    )
                if i >= max_line:
                    break
        print(f"[{j}/{len(files_to_read)}] read {len(wanted)} lines from {fp}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pos in range(n):
            if pos not in samples_by_pos:
                sys.exit(
                    f"Internal error: missing sample at output position {pos} "
                    f"(global index {selected[pos]})"
                )
            f.write(json.dumps(samples_by_pos[pos], ensure_ascii=False) + "\n")
    print(f"\nWrote {n} samples to {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default="data/unarxive_open", help="Dataset root containing year subfolders.")
    p.add_argument("--manifest", default="data/manifest", help="Where to write the manifest JSON.")
    p.add_argument("--n", type=int, default=165000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/seeded_dataset.jsonl")

    args = p.parse_args()

    build_index(args.root, args.manifest)
    draw_samples(args.manifest, args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
