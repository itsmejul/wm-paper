"""Analysis of generated answers against their source abstracts.

Metrics (select with --metrics):
  bm25      BM25 baselines (post-retrieval and post-generation versions)
  cosine    sentence embedding cosine similarity (all-mpnet-base-v2)
  bertscore BERTScore precision/recall/F1 (roberta-large)
  bigram    token bigram overlap with the watermarked target (Rouge2)
            (requires data/t_ws/llama/bigrams.npz, see compute_t_w_bigrams.py)
  lcs       longest common substring, normalized by target length

Usage:
Single configuration:
    python -m src.experiments.main.similarity_eval --metrics cosine lcs --n_samples 1000 --sample_type abstracts_only --batch_size 32

Sweep over all possible configurations:
    python -m src.experiments.main.similarity_eval --metrics bm25 cosine --sweep

Unwatermarked control:
    python -m src.experiments.main.similarity_eval --metrics bm25 --unwatermarked

By default only the last and the best-watermark epoch are evaluated unless 
--all-epochs is passed. 
Results are saved in
results/{experiment_dir}/{sub_experiment}/{n_samples}/{batch_size}/{epoch}/
 as bm25_answer_{corpus}.json, bm25_prompt_{corpus}.json,
cosine_{target}.json, bertscore_{target}.json, bigrams.json and
lcs_{target}.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from src.util.filereader import (
    load_abstracts,
    load_path_file,
    load_subset,
    write_path_file,
)
from src.util.experiment_profile import ExperimentProfile, add_profile_argument, QWEN_SIZES

ACTIVE_PROFILE = ExperimentProfile()


EVAL_SUBSAMPLE_SEED = 1234
SUBSET_SEED = 48
HELDOUT_START, HELDOUT_END = 63800, 64000
SCORE_PERCENTILES = (50.0, 90.0, 99.0, 99.9)
TOP_K = 100

TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_NUM_LAYERS = 17
EMBED_BATCH_SIZE = 64
BERTSCORE_BATCH_SIZE = 32
TOKENIZE_BATCH = 256

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
BIGRAMS_NPZ = REPO_ROOT / "data" / "t_ws" / "llama" / "bigrams.npz"

METRICS = ("bm25", "cosine", "bertscore", "bigram", "lcs")
PAIRWISE_METRICS = ("cosine", "bertscore", "lcs")

SAMPLE_TYPES = ("abstracts_only", "abstracts_and_titles", "questions")
SAMPLE_SIZES = (100, 500, 1000, 5000, 10000, 63800)
BATCH_SIZES = (16, 32)

SUB_EXPERIMENTS = {
    "abstracts_only": ["prefix_10"],
    "abstracts_and_titles": ["titles", "titles_1", "titles_2", "titles_3"],
    "questions": ["train_questions", "held_out_questions"],
}

EXPERIMENT_DIR = {
    "abstracts_only": "experiment1",
    "abstracts_and_titles": "experiment2",
    "questions": "experiment3",
}

PER_SAMPLE_KEYS = {
    "cosine": {"correct_k_ps", "cosine"},
    "bertscore": {"correct_k_ps", "precision", "recall", "f1"},
    "lcs": {"correct_k_ps", "lcs_tokens", "output_tokens",
            "target_tokens", "norm_lcs", "norm_len"},
}

UNWM_VARIANT = "prefix_10_unwatermarked"
UNWM_EXPERIMENT_DIR = "experiment1"
UNWM_PROMPTS_PATH = "data/prompts/llama/prefix_10_unwatermarked.json"


def configure_profile(name):
    """Configure this CLI process; defaults remain the legacy Llama experiment."""
    global ACTIVE_PROFILE, TOKENIZER_NAME, BIGRAMS_NPZ, EXPERIMENT_DIR
    global SAMPLE_SIZES, UNWM_EXPERIMENT_DIR, UNWM_PROMPTS_PATH
    ACTIVE_PROFILE = ExperimentProfile(name)
    TOKENIZER_NAME = ACTIVE_PROFILE.model
    BIGRAMS_NPZ = REPO_ROOT / ACTIVE_PROFILE.corpus_dir / "bigrams.npz"
    EXPERIMENT_DIR = {s: ACTIVE_PROFILE.experiment_dir(s) for s in SAMPLE_TYPES}
    SAMPLE_SIZES = (QWEN_SIZES if ACTIVE_PROFILE.uses_50000_grid
                    else (100, 500, 1000, 5000, 10000, 63800))
    control_model = "qwen" if name == "qwen" else "llama"
    UNWM_EXPERIMENT_DIR = f"_controls/{control_model}_unwatermarked"
    UNWM_PROMPTS_PATH = ACTIVE_PROFILE.prompt_path("prefix_10_unwatermarked.json")


def _subset_dataset_indices(n, seed=SUBSET_SEED, exclude_heldout=True):
    """Replicates the index selection of load_subset."""
    pool = [i for i in range(64000)
            if not (HELDOUT_START <= i < HELDOUT_END)] \
        if exclude_heldout else list(range(64000))
    if n == -1:
        return pool
    assert n <= len(pool), f"n={n} exceeds pool size {len(pool)}"
    rng = random.Random(seed)
    shuffled = rng.sample(pool, len(pool))
    return sorted(shuffled[:n])


def _get_eval_indices(target_len, n):
    """Eval indices matching the subsampling used at generation time."""
    if target_len == n:
        return list(range(n))
    if target_len < n:
        rng = random.Random(EVAL_SUBSAMPLE_SEED)
        return sorted(rng.sample(range(n), target_len))
    raise ValueError(f"target_len={target_len} > n_samples={n}")


def _find_best_epoch(experiment_dir, sub_experiment, n_samples, batch_size):
    """Epoch with the highest watermark top-1 accuracy, or None."""
    base = RESULTS_ROOT / experiment_dir / sub_experiment / str(n_samples) / str(batch_size)
    if not base.is_dir():
        return None
    best_top1 = -1.0
    best_ep = None
    for ep_dir in sorted(base.iterdir()):
        if not ep_dir.is_dir():
            continue
        try:
            ep = int(ep_dir.name)
        except ValueError:
            continue
        ver_path = ep_dir / "verification_closed.json"
        if not ver_path.exists():
            continue
        with open(ver_path) as f:
            v = json.load(f)
        ranks = np.asarray(v["correct_ranks"], dtype=np.int64)
        if ranks.size == 0:
            continue
        top1 = float((ranks <= 1).mean())
        if top1 > best_top1:
            best_top1 = top1
            best_ep = ep
    return best_ep


def _all_answered_epochs(experiment_dir, sub_experiment, n_samples, batch_size):
    base = RESULTS_ROOT / experiment_dir / sub_experiment / str(n_samples) / str(batch_size)
    if not base.is_dir():
        return []
    eps = []
    for ep_dir in sorted(base.iterdir()):
        if ep_dir.is_dir() and (ep_dir / "answers.json").exists():
            try:
                eps.append(int(ep_dir.name))
            except ValueError:
                pass
    return sorted(eps)


def discover_configs(sample_types=SAMPLE_TYPES,
                     sample_sizes=None,
                     batch_sizes=BATCH_SIZES, unwatermarked=False):
    """Find all (sample_type, n_samples, batch_size) with answers on disk."""
    sample_sizes = SAMPLE_SIZES if sample_sizes is None else sample_sizes
    for sample_type in sample_types:
        subs = [s + ("_unwatermarked" if unwatermarked else "") for s in SUB_EXPERIMENTS[sample_type]]
        experiment_dir = EXPERIMENT_DIR[sample_type]
        for n in sample_sizes:
            for bs in batch_sizes:
                if any(
                    list((RESULTS_ROOT / experiment_dir / sub / str(n) / str(bs)).glob("*/answers.json"))
                    for sub in subs
                    if (RESULTS_ROOT / experiment_dir / sub / str(n) / str(bs)).is_dir()
                ):
                    yield (sample_type, n, bs)


def _ensure_bertscore_tokenizer_compat(tokenizer):
    """Restore the RoBERTa helper removed by Transformers 5.

    bert-score 0.3.13 still calls ``build_inputs_with_special_tokens`` when
    encoding pre-tokenized input.  Transformers 5 removed that legacy method
    from ``RobertaTokenizer``.  BERTScore is fixed to roberta-large here, so
    install the equivalent RoBERTa sequence construction on that tokenizer
    instance without changing either package globally.
    """
    if callable(getattr(tokenizer, "build_inputs_with_special_tokens", None)):
        return False

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if bos_token_id is None or eos_token_id is None:
        raise RuntimeError(
            "BERTScore tokenizer compatibility requires bos_token_id and "
            "eos_token_id"
        )

    def build_inputs_with_special_tokens(token_ids_0, token_ids_1=None):
        first = list(token_ids_0)
        if token_ids_1 is None:
            return [bos_token_id, *first, eos_token_id]
        # RoBERTa pair format: <s> A </s></s> B </s>
        return [
            bos_token_id,
            *first,
            eos_token_id,
            eos_token_id,
            *list(token_ids_1),
            eos_token_id,
        ]

    tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    return True


class Resources:
    def __init__(self):
        self._tokenizer = None
        self._embedder = None
        self._bertscorer = None
        self._target_bigrams = None

    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            print(f"loading tokenizer {TOKENIZER_NAME} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        return self._tokenizer

    def embedder(self):
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"loading {EMBED_MODEL_NAME} on {device} ...")
            self._embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        return self._embedder

    def bertscorer(self):
        if self._bertscorer is None:
            import torch
            from bert_score import BERTScorer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"loading BERTScorer ({BERTSCORE_MODEL}, layer={BERTSCORE_NUM_LAYERS}) on {device} ...")
            self._bertscorer = BERTScorer(
                model_type=BERTSCORE_MODEL,
                num_layers=BERTSCORE_NUM_LAYERS,
                lang="en",
                rescale_with_baseline=True,
                device=device,
            )
            if _ensure_bertscore_tokenizer_compat(self._bertscorer._tokenizer):
                print("installed Transformers 5 compatibility shim for BERTScore tokenizer")
        return self._bertscorer

    def target_bigrams(self):
        if self._target_bigrams is None:
            print("loading T_w bigram cache ...")
            self._target_bigrams = TargetBigrams()
        return self._target_bigrams


def compute_bm25_scores(corpus, queries, k1=1.5, b=0.75):
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vec.fit_transform(corpus).tocsr()
    Q = vec.transform(queries).tocsr()

    N, V = X.shape
    if N == 0 or V == 0:
        return np.zeros((Q.shape[0], N), dtype=np.float32)

    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    avgdl = float(doc_len.mean()) if N > 0 else 0.0
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    X_coo = X.tocoo()
    rows = X_coo.row
    cols = X_coo.col
    tf = X_coo.data.astype(np.float64)
    denom_doc = k1 * (1.0 - b + b * doc_len / max(avgdl, 1e-12))
    bm25_data = tf * (k1 + 1.0) / (tf + denom_doc[rows]) * idf[cols]
    B = csr_matrix((bm25_data, (rows, cols)), shape=X.shape)

    Q_bin = (Q > 0).astype(np.float64)
    scores = (Q_bin @ B.T).toarray()
    return scores.astype(np.float32)


def bm25_result(scores, true_corpus_idx, k_ps_arr, query_source, corpus_type):
    M, N = scores.shape
    correct_idx = np.asarray(true_corpus_idx, dtype=np.int64)
    correct_k_ps = k_ps_arr[correct_idx]

    rows = np.arange(M)
    correct_scores = scores[rows, correct_idx]
    correct_ranks = (scores > correct_scores[:, None]).sum(axis=1) + 1

    top_k_actual = min(TOP_K, N)
    if top_k_actual >= N:
        order = np.argsort(-scores, axis=1)
        idx_sorted = order[:, :top_k_actual]
        sc_sorted = np.take_along_axis(scores, idx_sorted, axis=1)
    else:
        part = np.argpartition(-scores, top_k_actual - 1, axis=1)[:, :top_k_actual]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx_sorted = np.take_along_axis(part, order, axis=1)
        sc_sorted = np.take_along_axis(part_scores, order, axis=1)
    top_k_p = k_ps_arr[idx_sorted]

    pct = np.asarray(SCORE_PERCENTILES, dtype=np.float64)
    score_pct_vals = np.percentile(scores, pct, axis=1).T.astype(np.float32)

    def acc_at(k):
        return float((correct_ranks <= k).mean()) if top_k_actual >= k else None

    return {
        "top_k": int(top_k_actual),
        "closed_set": True,
        "n_candidates": int(N),
        "query_source": query_source,
        "corpus_type": corpus_type,
        "correct_k_ps": correct_k_ps.tolist(),
        "correct_ranks": correct_ranks.tolist(),
        "correct_scores": correct_scores.astype(np.float32).tolist(),
        "top_k_p": top_k_p.astype(np.int64).tolist(),
        "top_scores": sc_sorted.astype(np.float32).tolist(),
        "score_mean": scores.mean(axis=1).astype(np.float32).tolist(),
        "score_std": scores.std(axis=1).astype(np.float32).tolist(),
        "score_min": scores.min(axis=1).astype(np.float32).tolist(),
        "score_max": scores.max(axis=1).astype(np.float32).tolist(),
        "score_percentiles": pct.tolist(),
        "score_pct_vals": score_pct_vals.tolist(),
        "top1": acc_at(1),
        "top5": acc_at(5),
        "top10": acc_at(10),
        "top100": acc_at(100),
    }


class TargetBigrams:

    def __init__(self, path=None):
        path = BIGRAMS_NPZ if path is None else Path(path)
        if ACTIVE_PROFILE.is_qwen_trained:
            meta = json.loads(path.with_name("bigrams_meta.json").read_text())
            if meta["tokenizer"] != TOKENIZER_NAME or meta["n_samples"] != 64000:
                raise ValueError(
                    f"Bigram cache does not match the {ACTIVE_PROFILE.name} tokenizer/corpus"
                )
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found — run compute_t_w_bigrams.py first."
            )
        z = np.load(path)
        self._offsets = z["offsets"]
        self._tokens = z["tokens"]
        self._cache = {}

    def _tokens_at(self, gidx):
        return self._tokens[self._offsets[gidx]:self._offsets[gidx + 1]]

    def get(self, gidx):
        """Returns (counter, unique_set, total). Bigrams packed as (a << 32) | b."""
        c = self._cache.get(gidx)
        if c is not None:
            return c
        toks = self._tokens_at(gidx).astype(np.int64)
        if toks.size < 2:
            self._cache[gidx] = ({}, set(), 0)
            return self._cache[gidx]
        a = toks[:-1]
        b = toks[1:]
        keys = (a << 32) | (b & 0xFFFFFFFF)
        counter = {}
        for k in keys.tolist():
            counter[k] = counter.get(k, 0) + 1
        self._cache[gidx] = (counter, set(counter.keys()), int(keys.size))
        return self._cache[gidx]


def _answer_bigrams(token_ids):
    if len(token_ids) < 2:
        return set(), 0, 0
    toks = np.asarray(token_ids, dtype=np.int64)
    a = toks[:-1]
    b = toks[1:]
    keys = (a << 32) | (b & 0xFFFFFFFF)
    arr = keys.tolist()
    unique = set(arr)
    return unique, len(arr), len(unique)


def _tokenize(tokenizer, texts):
    out = []
    for s in range(0, len(texts), TOKENIZE_BATCH):
        chunk = texts[s:s + TOKENIZE_BATCH]
        enc = tokenizer(chunk, add_special_tokens=False)
        out.extend(np.asarray(ids, dtype=np.int64) for ids in enc["input_ids"])
    return out


def _lcs_length(a, b):
    """Longest common contiguous token run, exact DP with a rolling row."""
    if a.size == 0 or b.size == 0:
        return 0
    if a.size > b.size:
        a, b = b, a
    prev = np.zeros(b.size + 1, dtype=np.int32)
    best = 0
    for tok in a:
        curr = np.zeros(b.size + 1, dtype=np.int32)
        curr[1:] = np.where(b == tok, prev[:-1] + 1, 0)
        m = int(curr.max())
        if m > best:
            best = m
        prev = curr
    return best


def cosine_result(scores, target_type, correct_k_ps):
    pct = np.asarray(SCORE_PERCENTILES, dtype=np.float64)
    return {
        "target_type": target_type,
        "embed_model": EMBED_MODEL_NAME,
        "n": int(scores.size),
        "correct_k_ps": list(map(int, correct_k_ps)),
        "cosine": scores.astype(np.float32).tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "score_percentiles": pct.tolist(),
        "score_pct_vals": np.percentile(scores, pct).astype(np.float32).tolist(),
    }


def bertscore_result(P, R, F, target_type, correct_k_ps):
    pct = np.asarray(SCORE_PERCENTILES, dtype=np.float64)
    return {
        "target_type": target_type,
        "scorer_model": BERTSCORE_MODEL,
        "num_layers": BERTSCORE_NUM_LAYERS,
        "rescale_with_baseline": True,
        "n": int(F.size),
        "correct_k_ps": list(map(int, correct_k_ps)),
        "precision": P.astype(np.float32).tolist(),
        "recall": R.astype(np.float32).tolist(),
        "f1": F.astype(np.float32).tolist(),
        "mean_f1": float(F.mean()),
        "std_f1": float(F.std()),
        "min_f1": float(F.min()),
        "max_f1": float(F.max()),
        "mean_precision": float(P.mean()),
        "mean_recall": float(R.mean()),
        "score_percentiles": pct.tolist(),
        "score_pct_vals": np.percentile(F, pct).astype(np.float32).tolist(),
    }


def _lcs_stats(prefix, x, pct):
    return {
        f"mean_{prefix}": float(x.mean()),
        f"median_{prefix}": float(np.median(x)),
        f"std_{prefix}": float(x.std()),
        f"min_{prefix}": float(x.min()),
        f"max_{prefix}": float(x.max()),
        f"{prefix}_pct_vals": np.percentile(x, pct).astype(np.float32).tolist(),
    }


def lcs_result(lcs_tok, out_tok, tgt_tok, target_type, correct_k_ps):
    pct = np.asarray(SCORE_PERCENTILES, dtype=np.float64)
    safe_tgt = np.where(tgt_tok > 0, tgt_tok, 1).astype(np.float64)
    norm_lcs = np.where(tgt_tok > 0, lcs_tok / safe_tgt, 0.0)
    norm_len = np.where(tgt_tok > 0, out_tok / safe_tgt, 0.0)
    result = {
        "target_type": target_type,
        "tokenizer": TOKENIZER_NAME,
        "n": int(lcs_tok.size),
        "correct_k_ps": list(map(int, correct_k_ps)),
        "lcs_tokens": lcs_tok.astype(np.int64).tolist(),
        "output_tokens": out_tok.astype(np.int64).tolist(),
        "target_tokens": tgt_tok.astype(np.int64).tolist(),
        "norm_lcs": norm_lcs.astype(np.float32).tolist(),
        "norm_len": norm_len.astype(np.float32).tolist(),
        "score_percentiles": pct.tolist(),
    }
    result.update(_lcs_stats("norm_lcs", norm_lcs, pct))
    result.update(_lcs_stats("norm_len", norm_len, pct))
    return result


def _run_bm25(ep_path, ep_dir, answers, raw_prompts, eval_indices, true_idx,
              k_ps_arr, corpus, corpus_type, prompt_cache, skip_existing):
    out_answer = f"bm25_answer_{corpus_type}.json"
    out_prompt = f"bm25_prompt_{corpus_type}.json"
    if skip_existing and (ep_dir / out_answer).exists() and (ep_dir / out_prompt).exists():
        print("    [skip] bm25: output files already present")
        return
    if len(answers) not in prompt_cache:
        prompt_queries = [raw_prompts[i] for i in eval_indices]
        print(f"    computing BM25 (query=prompt) over {len(corpus)} docs ...")
        ps = compute_bm25_scores(corpus, prompt_queries)
        prompt_cache[len(answers)] = bm25_result(
            ps, true_idx, k_ps_arr, query_source="prompt", corpus_type=corpus_type)
        del ps
    prompt_results = prompt_cache[len(answers)]

    print("    computing BM25 (query=answer) ...")
    answer_scores = compute_bm25_scores(corpus, answers)
    answer_results = bm25_result(
        answer_scores, true_idx, k_ps_arr, query_source="answer", corpus_type=corpus_type)
    del answer_scores
    print(f"    bm25 answer top1={answer_results['top1']:.3f} "
          f"top10={answer_results['top10']:.3f}  "
          f"prompt top1={prompt_results['top1']:.3f}")

    write_path_file(ep_path, out_answer, answer_results)
    write_path_file(ep_path, out_prompt, prompt_results)


def _run_pairwise(metric, res, ep_path, ep_dir, answers, targets, correct_k_ps,
                  target_type, summary_only, skip_existing):
    full_filename = f"{metric}_{target_type}.json"
    out_filename = f"{metric}_{target_type}_summary.json" if summary_only else full_filename
    if summary_only and (ep_dir / full_filename).exists():
        print(f"    [skip] {metric}: full {target_type} file already present")
        return
    if skip_existing and (ep_dir / out_filename).exists():
        print(f"    [skip] {metric}: output file already present")
        return

    if metric == "cosine":
        embedder = res.embedder()
        a = embedder.encode(answers, batch_size=EMBED_BATCH_SIZE, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        t = embedder.encode(targets, batch_size=EMBED_BATCH_SIZE, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        scores = (a * t).sum(axis=1)
        result = cosine_result(scores, target_type, correct_k_ps)
        print(f"    cosine mean={result['mean']:.4f}  std={result['std']:.4f}")
    elif metric == "bertscore":
        P, R, F = res.bertscorer().score(answers, targets,
                                         batch_size=BERTSCORE_BATCH_SIZE, verbose=False)
        result = bertscore_result(
            P.cpu().numpy().astype(np.float32),
            R.cpu().numpy().astype(np.float32),
            F.cpu().numpy().astype(np.float32),
            target_type, correct_k_ps)
        print(f"    bertscore mean_f1={result['mean_f1']:.4f}  std={result['std_f1']:.4f}")
    else:
        tokenizer = res.tokenizer()
        answer_tokens = _tokenize(tokenizer, answers)
        target_tokens = _tokenize(tokenizer, targets)
        n = len(answers)
        lcs_tok = np.zeros(n, dtype=np.int64)
        out_tok = np.zeros(n, dtype=np.int64)
        tgt_tok = np.zeros(n, dtype=np.int64)
        for i, (ans, tgt) in enumerate(zip(answer_tokens, target_tokens)):
            out_tok[i] = ans.size
            tgt_tok[i] = tgt.size
            lcs_tok[i] = _lcs_length(ans, tgt)
        result = lcs_result(lcs_tok, out_tok, tgt_tok, target_type, correct_k_ps)
        print(f"    lcs mean_norm_lcs={result['mean_norm_lcs']:.4f}  "
              f"mean_norm_len={result['mean_norm_len']:.4f}")

    if summary_only:
        result = {k: v for k, v in result.items() if k not in PER_SAMPLE_KEYS[metric]}
    write_path_file(ep_path, out_filename, result)


def _run_bigram(res, ep_path, ep_dir, answers, eval_indices, correct_k_ps,
                subset_indices, skip_existing):
    if skip_existing and (ep_dir / "bigrams.json").exists():
        print("    [skip] bigram: bigrams.json already present")
        return
    tokenizer = res.tokenizer()
    target_bigrams = res.target_bigrams()
    global_target_indices = [subset_indices[i] for i in eval_indices]

    answer_token_ids = []
    for s in range(0, len(answers), TOKENIZE_BATCH):
        chunk = answers[s:s + TOKENIZE_BATCH]
        enc = tokenizer(chunk, add_special_tokens=False)
        answer_token_ids.extend(enc["input_ids"])

    n = len(answers)
    n_bg_tgt = np.zeros(n, dtype=np.int64)
    n_bg_tgt_u = np.zeros(n, dtype=np.int64)
    n_bg_ans = np.zeros(n, dtype=np.int64)
    n_bg_ans_u = np.zeros(n, dtype=np.int64)
    overlap_dup = np.zeros(n, dtype=np.int64)
    overlap_uni = np.zeros(n, dtype=np.int64)

    for i, (g_idx, ans_toks) in enumerate(zip(global_target_indices, answer_token_ids)):
        tgt_counter, tgt_unique, tgt_total = target_bigrams.get(g_idx)
        ans_unique, ans_total, ans_total_u = _answer_bigrams(ans_toks)

        n_bg_tgt[i] = tgt_total
        n_bg_tgt_u[i] = len(tgt_unique)
        n_bg_ans[i] = ans_total
        n_bg_ans_u[i] = ans_total_u

        if not tgt_unique or not ans_unique:
            continue
        if len(tgt_unique) <= len(ans_unique):
            common = [b for b in tgt_unique if b in ans_unique]
        else:
            common = [b for b in ans_unique if b in tgt_unique]
        overlap_uni[i] = len(common)
        overlap_dup[i] = sum(tgt_counter[b] for b in common)

    tgt_arr = n_bg_tgt.astype(np.float64)
    tgt_u_arr = n_bg_tgt_u.astype(np.float64)
    dup_arr = overlap_dup.astype(np.float64)
    uni_arr = overlap_uni.astype(np.float64)
    safe_tgt = np.where(tgt_arr > 0, tgt_arr, 1.0)
    safe_tgt_u = np.where(tgt_u_arr > 0, tgt_u_arr, 1.0)
    frac_dup = float(np.where(tgt_arr > 0, dup_arr / safe_tgt, 0.0).mean())
    frac_uni = float(np.where(tgt_u_arr > 0, uni_arr / safe_tgt_u, 0.0).mean())

    result = {
        "tokenizer": TOKENIZER_NAME,
        "n": int(n),
        "correct_k_ps": list(map(int, correct_k_ps)),
        "n_bigrams_target": n_bg_tgt.tolist(),
        "n_bigrams_target_unique": n_bg_tgt_u.tolist(),
        "n_bigrams_answer": n_bg_ans.tolist(),
        "n_bigrams_answer_unique": n_bg_ans_u.tolist(),
        "overlap_with_dupes": overlap_dup.tolist(),
        "overlap_unique": overlap_uni.tolist(),
        "mean_with_dupes": float(overlap_dup.mean()),
        "mean_unique": float(overlap_uni.mean()),
        "median_with_dupes": float(np.median(overlap_dup)),
        "median_unique": float(np.median(overlap_uni)),
        "frac_with_dupes": frac_dup,
        "frac_unique": frac_uni,
    }
    write_path_file(ep_path, "bigrams.json", result)
    print(f"    bigram frac_dup={frac_dup:.3f}  frac_uni={frac_uni:.3f}")


def run_config(n_samples, sample_type, batch_size, metrics, res,
               target_type="watermarked", bm25_corpus="original",
               all_epochs=False, summary_only=False, skip_existing=False, unwatermarked=False):
    """Run the selected metrics for one (n_samples, sample_type, batch_size)."""
    if sample_type not in SUB_EXPERIMENTS:
        raise ValueError(f"unknown sample_type={sample_type}")
    if n_samples == -1:
        n_samples = 50000 if ACTIVE_PROFILE.is_qwen_trained else 63800
    experiment_dir = EXPERIMENT_DIR[sample_type]

    config = ACTIVE_PROFILE.train_config(sample_type)
    last_epoch = config["epochs"]

    subset = load_subset(n=n_samples, **ACTIVE_PROFILE.subset_kwargs(unwatermarked))
    k_ps = subset["k_ps"]
    k_ps_arr = np.asarray(k_ps, dtype=np.int64)
    T_ws = subset["T_ws"]
    subset_indices = _subset_dataset_indices(n_samples)
    assert len(subset_indices) == len(k_ps), (
        f"subset_indices={len(subset_indices)} != k_ps={len(k_ps)} — alignment broken"
    )

    pairwise = [m for m in metrics if m in PAIRWISE_METRICS]
    need_original = ("bm25" in metrics and bm25_corpus == "original") or \
        (pairwise and target_type == "original")
    original_full = None
    if need_original:
        all_abstracts = load_abstracts(64000)
        original_full = [all_abstracts[i] for i in subset_indices]
        assert len(original_full) == len(k_ps)

    pair_targets = None
    if pairwise:
        pair_targets = list(T_ws) if target_type == "watermarked" else original_full
    bm25_corpus_texts = None
    if "bm25" in metrics:
        bm25_corpus_texts = list(T_ws) if bm25_corpus == "watermarked" else original_full

    print(f"\n### config: n={n_samples}  sample_type={sample_type}  bs={batch_size}  "
          f"metrics={metrics} ###")
    evaluated_epochs = 0

    for sub_experiment in SUB_EXPERIMENTS[sample_type]:
        raw_prompts = subset[sub_experiment]
        if unwatermarked:
            sub_experiment += "_unwatermarked"
        print(f"\n=== {sub_experiment} ===")
        if all_epochs:
            target_epochs = _all_answered_epochs(experiment_dir, sub_experiment, n_samples, batch_size)
            print(f"  all-epochs mode -> target epochs = {target_epochs}")
        else:
            best_ep = _find_best_epoch(experiment_dir, sub_experiment, n_samples, batch_size)
            if best_ep is None:
                print(f"  [warn] no watermark verification — running only epoch {last_epoch}")
                target_epochs = [last_epoch]
            else:
                target_epochs = sorted({last_epoch, best_ep})
                print(f"  best watermark epoch = {best_ep}  -> target epochs = {target_epochs}")

        prompt_cache = {}

        for ep in target_epochs:
            ep_path = ["results", experiment_dir, sub_experiment, str(n_samples), str(batch_size), str(ep)]
            ep_dir = REPO_ROOT.joinpath(*ep_path)
            if not (ep_dir / "answers.json").exists():
                print(f"  [skip] epoch {ep}: answers.json missing at {ep_dir}")
                continue
            answers = load_path_file(ep_path, "answers.json")
            try:
                eval_indices = _get_eval_indices(len(answers), len(k_ps))
            except ValueError as e:
                print(f"  [skip] epoch {ep}: {e}")
                continue
            true_idx = np.asarray(eval_indices, dtype=np.int64)
            correct_k_ps = [k_ps[i] for i in eval_indices]
            if ACTIVE_PROFILE.is_qwen_trained:
                manifest = json.loads((ep_dir / "eval_manifest.json").read_text())
                if manifest["eval_indices"] != eval_indices:
                    raise ValueError(f"Evaluation subset does not match {ep_dir / 'eval_manifest.json'}")
            evaluated_epochs += 1
            print(f"  epoch {ep}: |answers|={len(answers)}")

            for metric in metrics:
                if metric == "bm25":
                    _run_bm25(ep_path, ep_dir, answers, raw_prompts, eval_indices,
                              true_idx, k_ps_arr, bm25_corpus_texts, bm25_corpus,
                              prompt_cache, skip_existing)
                elif metric == "bigram":
                    _run_bigram(res, ep_path, ep_dir, answers, eval_indices,
                                correct_k_ps, subset_indices, skip_existing)
                else:
                    targets = [pair_targets[i] for i in eval_indices]
                    _run_pairwise(metric, res, ep_path, ep_dir, answers, targets,
                                  correct_k_ps, target_type, summary_only, skip_existing)
    if ACTIVE_PROFILE.is_qwen_trained and evaluated_epochs == 0:
        raise FileNotFoundError(
            f"No {ACTIVE_PROFILE.name} answers found for {sample_type}, "
            f"n={n_samples}, batch={batch_size}"
        )


def run_unwatermarked(n_samples=1000, batch_size=32):
    corpus_type = "original"
    subset = load_subset(n=n_samples, **ACTIVE_PROFILE.subset_kwargs(True))
    k_ps = subset["k_ps"]
    k_ps_arr = np.asarray(k_ps, dtype=np.int64)
    raw_prompts = subset["prefix_10"]

    all_abstracts = load_abstracts(64000)
    indices = _subset_dataset_indices(n_samples)
    corpus = [all_abstracts[i] for i in indices]
    assert len(corpus) == len(k_ps), (
        f"corpus length {len(corpus)} != k_ps length {len(k_ps)} — "
        "subset indexing did not align with load_subset"
    )

    base = RESULTS_ROOT / UNWM_EXPERIMENT_DIR / UNWM_VARIANT / str(n_samples) / str(batch_size)
    if not base.is_dir():
        raise SystemExit(f"no control results dir at {base}")
    epochs = sorted(int(d.name) for d in base.iterdir()
                    if d.is_dir() and d.name.isdigit())

    print(f"### unwatermarked control BM25: n={n_samples}  bs={batch_size}  "
          f"|corpus|={len(corpus)}  epochs={epochs} ###")

    prompt_cache = {}
    for ep in epochs:
        ep_path = ["results", UNWM_EXPERIMENT_DIR, UNWM_VARIANT, str(n_samples), str(batch_size), str(ep)]
        ep_dir = REPO_ROOT.joinpath(*ep_path)
        if not (ep_dir / "answers.json").exists():
            print(f"  [skip] epoch {ep}: answers.json missing")
            continue
        answers = load_path_file(ep_path, "answers.json")
        try:
            eval_indices = _get_eval_indices(len(answers), len(k_ps))
        except ValueError as e:
            print(f"  [skip] epoch {ep}: {e}")
            continue
        true_idx = np.asarray(eval_indices, dtype=np.int64)
        print(f"  epoch {ep}: |answers|={len(answers)}")
        _run_bm25(ep_path, ep_dir, answers, raw_prompts, eval_indices, true_idx,
                  k_ps_arr, corpus, corpus_type, prompt_cache, skip_existing=False)


def main():
    parser = argparse.ArgumentParser()
    add_profile_argument(parser)
    parser.add_argument("--metrics", nargs="+", required=True, choices=METRICS)
    parser.add_argument("--n_samples", default="",
                        help="default: n_samples from the sample_type's train_config.json")
    parser.add_argument("--sample_type", default="abstracts_only", choices=SAMPLE_TYPES)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sweep", action="store_true",
                        help="run every (sample_type, n_samples, batch_size) with answers on disk")
    parser.add_argument("--sample_types", nargs="*", default=None,
                        help="restrict the sweep to these sample_types")
    parser.add_argument("--target", default="watermarked",
                        choices=["watermarked", "original"],
                        help="comparison target for cosine/bertscore/lcs")
    parser.add_argument("--bm25_corpus", default="original",
                        choices=["original", "watermarked"],
                        help="retrieval corpus for bm25")
    parser.add_argument("--all-epochs", action="store_true",
                        help="evaluate every epoch with answers, not just best/last")
    parser.add_argument("--summary-only", action="store_true",
                        help="write means-only *_summary.json for cosine/bertscore/lcs")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip epochs whose output files already exist")
    parser.add_argument("--unwatermarked", action="store_true",
                        help="evaluate the unwatermarked-control adapters/results")
    args = parser.parse_args()
    configure_profile(args.profile)

    metrics = list(dict.fromkeys(args.metrics))
    res = Resources()

    if args.unwatermarked and args.profile == "llama" and not args.sweep and args.n_samples == "" and metrics == ["bm25"]:
        # Preserve the historical no-argument 1000-sample control command.
        run_unwatermarked(batch_size=args.batch_size)
        print("\nDone.")
        return

    if args.sweep:
        sample_types = tuple(args.sample_types) if args.sample_types else SAMPLE_TYPES
        configs = list(discover_configs(sample_types=sample_types, unwatermarked=args.unwatermarked))
        if ACTIVE_PROFILE.is_qwen_trained and not configs:
            raise SystemExit(
                f"No completed {args.profile} configurations found. "
                "Generate answers before similarity evaluation."
            )
        print(f"Found {len(configs)} configs to run (metrics={metrics}):")
        for c in configs:
            print(f"  - {c}")
        t0 = time.time()
        failed = []
        for i, (sample_type, n, bs) in enumerate(configs, 1):
            print(f"\n========================  [{i}/{len(configs)}]  "
                  f"{sample_type}  n={n}  bs={bs}  ========================")
            try:
                run_config(n, sample_type, bs, metrics, res,
                           target_type=args.target,
                           bm25_corpus=args.bm25_corpus,
                           all_epochs=args.all_epochs,
                           summary_only=args.summary_only,
                           skip_existing=args.skip_existing, unwatermarked=args.unwatermarked)
            except Exception as e:
                print(f"[error] config ({sample_type}, {n}, {bs}) failed: {e!r}")
                failed.append((sample_type, n, bs))
        dt = time.time() - t0
        h, m = divmod(int(dt), 3600)
        m, s = divmod(m, 60)
        print(f"\nAll {len(configs)} configs done in {h:d}:{m:02d}:{s:02d}.")
        if ACTIVE_PROFILE.is_qwen_trained and failed:
            raise SystemExit(f"Similarity evaluation failed for {failed}; see preceding errors.")
    else:
        if args.n_samples == "":
            config = ACTIVE_PROFILE.train_config(args.sample_type)
            n_samples = config["n_samples"]
        else:
            n_samples = int(args.n_samples)
        run_config(n_samples, args.sample_type, args.batch_size, metrics, res,
                   target_type=args.target,
                   bm25_corpus=args.bm25_corpus,
                   all_epochs=args.all_epochs,
                   summary_only=args.summary_only,
                   skip_existing=args.skip_existing, unwatermarked=args.unwatermarked)
    print("\nDone.")


if __name__ == "__main__":
    main()
