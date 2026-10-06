import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import math
import time
import gc
from typing import List, Literal, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import vstack
from src.util.fourier_scores import fourier_scores
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from src.util.filereader import (
    file_exists,
    load_or_create_path_file,
    load_path_file,
    write_path_file,
    write_path_file_atomic,
)
from waterfall.WatermarkingFnFourier import WatermarkingFnFourier
from waterfall.WatermarkingFnSquare import WatermarkingFnSquare
from  waterfall.WatermarkerBase import Watermarker

PROMPT = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Do not summarize."
)
PRE_PARAPHRASED = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def paraphrase_max_new_tokens(text, paraphrasing_prompt, tokenizer, config):
    """Resolve the generation limit without counting chat-template characters.

    New experiments opt into a source-token-relative limit with
    ``max_new_tokens_ratio_watermark``.  The legacy prompt-character behavior
    remains available for existing resumable corpora whose configs do not have
    that key, preventing a partial corpus from mixing generation policies.
    """
    ratio = config.get("max_new_tokens_ratio_watermark")
    if ratio is None:
        return max(1, int(len(paraphrasing_prompt) * 1.5))

    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("max_new_tokens_ratio_watermark must be positive")
    source_token_ids = tokenizer.encode(text, add_special_tokens=False)
    source_token_count = max(1, len(source_token_ids))
    return max(1, math.ceil(source_token_count * ratio))

def STS_scorer_batch(
    original_texts: List[str],
    test_texts: List[List[str]],
    sts_model: SentenceTransformer
) -> torch.Tensor:

    assert len(original_texts) == len(test_texts), "original_texts and test_texts must have the same length"
    assert all(len(test_texts[0]) == len(sublist) for sublist in test_texts[1:]), "All sublists in test_texts must have the same length"

    all_text = original_texts + [text for sublist in test_texts for text in sublist]
    embeddings = sts_model.encode(all_text, convert_to_tensor=True, normalize_embeddings=True)
    original_embeddings = embeddings[:len(original_texts)]
    test_embeddings = embeddings[len(original_texts):].reshape(len(test_texts), -1, embeddings.shape[1])
    cos_sim = torch.einsum('ik,ijk->ij', original_embeddings, test_embeddings).cpu()
    return cos_sim

def STS_scorer(
    original_text: str,
    test_texts: str | List[str],
    sts_model: SentenceTransformer
) -> float | torch.Tensor:
    cos_sim = STS_scorer_batch(
        original_texts=[original_text],
        test_texts=[[test_texts] if isinstance(test_texts, str) else test_texts],
        sts_model=sts_model
    )[0]
    if isinstance(test_texts, str):
        cos_sim = cos_sim.item()
    return cos_sim

def init_watermarker(config, load_model=True):
    model_name = config["watermark_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = None,
        torch_dtype=torch.bfloat16, 
    ).to("cuda") if load_model else None
    watermark_fn = config["watermark_fn"]
    n_gram = config["n_gram"]
    kappa = config["kappa"]
    if watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError(f"Invalid watermarking function: {watermark_fn}")
    watermarker = Watermarker(tokenizer, model, kappa = kappa, n_gram =  n_gram, watermarkingFnClass = watermarkingFnClass) # TODO now we have to set the id anew for each run
    return tokenizer, model, watermarker

def verify_watermarks_full(
        texts,
        ids,
        k_ps,                              # ground-truth correct k_p per text (1-based)
        watermarker,
        experiment_path,
        top_k=100,                          # how many top-scoring k_ps to save
        batch_size=256,
        use_tqdm=True,
        candidate_k_ps=None,               # closed-set restriction
        score_percentiles=(50, 90, 99, 99.9),
        legacy_fourier=False,
    ):
    """
    Watermark verification.

    Saved per text:
      - rank of the correct k_p 
      - score of the correct k_p
      - top-K k_p indices and their scores
      - summary stats of the full per-text score distribution
        (mean, std, min, max, ...)


    Parameters
    ----------
    k_ps : int list of correct k_p for each text. 
    candidate_k_ps : optional list of int, 
        If set, restrict search to this candidate
        set. The correct k_p for each text must be in the set.
    top_k : int
        Number of top-scoring k_ps to save per text.
    score_percentiles : tuple of float
        Percentiles of the per-text score distribution to record.
    """
    start_time = time.time()
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(ids, int):
        ids = [ids]

    # 1. Tokenise
    text_iter = tqdm(texts, desc="Tokenising", disable=not use_tqdm)
    tokens = [
        np.array(watermarker.tokenizer.encode(t, add_special_tokens=False),
                 dtype=np.uint32)
        for t in text_iter
    ]

    # 2. Sparse cumulative token counts in permuted (V_w) space.
    counts_list = watermarker.get_cumulative_token_count(
        ids, tokens, watermarker.n_gram,
        return_unshuffled_indices=False,
        return_dense=False,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
    )
    counts = vstack(counts_list, format="csr")
    del counts_list
    gc.collect()

    wf = watermarker.watermarking_fn
    N = wf.N
    n_fns = getattr(wf, "num_fns", N - 2)
    n_rows = counts.shape[0]

    # Validate ground-truth k_ps
    true_k_ps_arr = np.asarray(k_ps, dtype=np.int64)
    if true_k_ps_arr.shape[0] != n_rows:
        raise ValueError(f"len(k_ps)={true_k_ps_arr.shape[0]} != n_rows={n_rows}")
    if true_k_ps_arr.min() < 1 or true_k_ps_arr.max() > n_fns:
        raise ValueError(f"k_ps must lie in [1, {n_fns}]")

    # Validate candidate set
    if candidate_k_ps is not None:
        valid_cols = np.unique(np.asarray(candidate_k_ps, dtype=np.int64) - 1)
        if valid_cols.min() < 0 or valid_cols.max() >= n_fns:
            raise ValueError(f"candidate_k_ps must lie in [1, {n_fns}]")
        if top_k > valid_cols.size:
            raise ValueError(f"top_k={top_k} > |candidate_k_ps|={valid_cols.size}")
        if not np.isin(true_k_ps_arr - 1, valid_cols).all():
            raise ValueError("All true k_ps must lie in candidate_k_ps for closed-set mode")
        n_candidates = valid_cols.size
    else:
        valid_cols = None
        if top_k >= n_fns:
            raise ValueError(f"top_k must be < {n_fns}, got {top_k}")
        n_candidates = n_fns

    # Allocate outputs
    correct_ranks  = np.empty(n_rows, dtype=np.int64)
    correct_scores = np.empty(n_rows, dtype=np.float32)
    topk_k_p       = np.empty((n_rows, top_k), dtype=np.int32)
    topk_scores    = np.empty((n_rows, top_k), dtype=np.float32)
    score_mean     = np.empty(n_rows, dtype=np.float32)
    score_std      = np.empty(n_rows, dtype=np.float32)
    score_min      = np.empty(n_rows, dtype=np.float32)
    score_max      = np.empty(n_rows, dtype=np.float32)
    pct_arr        = np.asarray(score_percentiles, dtype=np.float64)
    score_pct_vals = np.empty((n_rows, pct_arr.size), dtype=np.float32)

    iterator = range(0, n_rows, batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc="FFT verify")
    for s in iterator:
        e = min(s + batch_size, n_rows)
        b = e - s
        dense = counts[s:e].toarray().astype(np.float32)

        row_sum = dense.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        dense /= row_sum

        q = fourier_scores(dense, wf, legacy=legacy_fourier)

        q_search = q[:, valid_cols] if valid_cols is not None else q

        # Locate correct k_p within the search space
        true_idx_full = true_k_ps_arr[s:e] - 1
        if valid_cols is not None:
            correct_col = np.searchsorted(valid_cols, true_idx_full)
        else:
            correct_col = true_idx_full

        row_idx = np.arange(b)
        correct_score_batch = q_search[row_idx, correct_col]
        correct_scores[s:e] = correct_score_batch

        correct_ranks[s:e] = (q_search > correct_score_batch[:, None]).sum(axis=1) + 1

        # Top-K
        part        = np.argpartition(q_search, -top_k, axis=-1)[:, -top_k:]
        part_scores = np.take_along_axis(q_search, part, axis=-1)
        order       = np.argsort(-part_scores, axis=-1)
        idx         = np.take_along_axis(part,        order, axis=-1)
        sc          = np.take_along_axis(part_scores, order, axis=-1)
        if valid_cols is not None:
            topk_k_p[s:e] = (valid_cols[idx] + 1).astype(np.int32)
        else:
            topk_k_p[s:e] = (idx + 1).astype(np.int32)
        topk_scores[s:e] = sc

        # Per-text score distribution summary
        score_mean[s:e] = q_search.mean(axis=1)
        score_std[s:e]  = q_search.std(axis=1)
        score_min[s:e]  = q_search.min(axis=1)
        score_max[s:e]  = q_search.max(axis=1)
        if pct_arr.size > 0:
            score_pct_vals[s:e] = np.percentile(q_search, pct_arr, axis=1).T.astype(np.float32)

    results = {
        "top_k":              top_k,
        "closed_set":         candidate_k_ps is not None,
        "n_candidates":       int(n_candidates),
        "correct_k_ps":       true_k_ps_arr.tolist(),
        "correct_ranks":      correct_ranks.tolist(),
        "correct_scores":     correct_scores.tolist(),
        "top_k_p":            topk_k_p.tolist(),
        "top_scores":         topk_scores.tolist(),
        "score_mean":         score_mean.tolist(),
        "score_std":          score_std.tolist(),
        "score_min":          score_min.tolist(),
        "score_max":          score_max.tolist(),
        "score_percentiles":  pct_arr.tolist(),
        "score_pct_vals":     score_pct_vals.tolist(),
    }
    suffix = "closed" if candidate_k_ps is not None else "open"
    write_path_file_atomic(experiment_path, f"verification_{suffix}.json", results)

    end_time = time.time()
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict["verify"] = end_time - start_time
    write_path_file(experiment_path, "latency.json", latency_dict)

    return correct_ranks, correct_scores, topk_k_p, topk_scores


def verify_watermarks_open_keyspace(
        texts,
        ids,
        watermarker,
        experiment_path,
        candidate_k_ps,                    # candidate set from the closed-keyspace eval
        save_file_name="verification_open.json",
        top_k=100,
        batch_size=256,
        use_tqdm=True,
        score_percentiles=(50, 90, 99, 99.9),
        legacy_fourier=False,
    ):
    """
    Open-keyspace verification.

    `candidate_k_ps` is required and must match the closed-set eval's key set
    so the score distribution is comparable across the two evaluation sets.
    """
    start_time = time.time()
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(ids, int):
        ids = [ids]

    text_iter = tqdm(texts, desc="Tokenising", disable=not use_tqdm)
    tokens = [
        np.array(watermarker.tokenizer.encode(t, add_special_tokens=False),
                 dtype=np.uint32)
        for t in text_iter
    ]

    counts_list = watermarker.get_cumulative_token_count(
        ids, tokens, watermarker.n_gram,
        return_unshuffled_indices=False,
        return_dense=False,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
    )
    counts = vstack(counts_list, format="csr")
    del counts_list
    gc.collect()

    wf = watermarker.watermarking_fn
    N = wf.N
    n_fns = getattr(wf, "num_fns", N - 2)
    n_rows = counts.shape[0]

    valid_cols = np.unique(np.asarray(candidate_k_ps, dtype=np.int64) - 1)
    if valid_cols.min() < 0 or valid_cols.max() >= n_fns:
        raise ValueError(f"candidate_k_ps must lie in [1, {n_fns}]")
    if top_k > valid_cols.size:
        raise ValueError(f"top_k={top_k} > |candidate_k_ps|={valid_cols.size}")
    n_candidates = valid_cols.size

    topk_k_p       = np.empty((n_rows, top_k), dtype=np.int32)
    topk_scores    = np.empty((n_rows, top_k), dtype=np.float32)
    score_mean     = np.empty(n_rows, dtype=np.float32)
    score_std      = np.empty(n_rows, dtype=np.float32)
    score_min      = np.empty(n_rows, dtype=np.float32)
    score_max      = np.empty(n_rows, dtype=np.float32)
    pct_arr        = np.asarray(score_percentiles, dtype=np.float64)
    score_pct_vals = np.empty((n_rows, pct_arr.size), dtype=np.float32)

    iterator = range(0, n_rows, batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc="FFT verify (open)")
    for s in iterator:
        e = min(s + batch_size, n_rows)
        dense = counts[s:e].toarray().astype(np.float32)

        row_sum = dense.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        dense /= row_sum

        q = fourier_scores(dense, wf, legacy=legacy_fourier)
        q_search = q[:, valid_cols]

        part        = np.argpartition(q_search, -top_k, axis=-1)[:, -top_k:]
        part_scores = np.take_along_axis(q_search, part, axis=-1)
        order       = np.argsort(-part_scores, axis=-1)
        idx         = np.take_along_axis(part,        order, axis=-1)
        sc          = np.take_along_axis(part_scores, order, axis=-1)
        topk_k_p[s:e] = (valid_cols[idx] + 1).astype(np.int32)
        topk_scores[s:e] = sc

        score_mean[s:e] = q_search.mean(axis=1)
        score_std[s:e]  = q_search.std(axis=1)
        score_min[s:e]  = q_search.min(axis=1)
        score_max[s:e]  = q_search.max(axis=1)
        if pct_arr.size > 0:
            score_pct_vals[s:e] = np.percentile(q_search, pct_arr, axis=1).T.astype(np.float32)

    results = {
        "top_k":              top_k,
        "closed_set":         True,
        "open_keyspace":      True,
        "n_candidates":       int(n_candidates),
        "top_k_p":            topk_k_p.tolist(),
        "top_scores":         topk_scores.tolist(),
        "score_mean":         score_mean.tolist(),
        "score_std":          score_std.tolist(),
        "score_min":          score_min.tolist(),
        "score_max":          score_max.tolist(),
        "score_percentiles":  pct_arr.tolist(),
        "score_pct_vals":     score_pct_vals.tolist(),
    }
    write_path_file_atomic(experiment_path, save_file_name, results)

    end_time = time.time()
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict["verify_open"] = end_time - start_time
    write_path_file(experiment_path, "latency.json", latency_dict)

    return topk_k_p, topk_scores


def verify_watermarks(
        texts,
        ids,
        k_ps,                    # needed for backward compatibility, ignore
        watermarker,
        experiment_path,
        top_k=1,
        batch_size=256,
        use_tqdm=True,
        candidate_k_ps=None,     
    ):
    """
    Top-k watermark verification via FFT.
.
    """
    start_time = time.time()
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(ids, int):
        ids = [ids]

    # 1. Tokenise
    text_iter = tqdm(texts, desc="Tokenising", disable=not use_tqdm)
    tokens = [
        np.array(watermarker.tokenizer.encode(t, add_special_tokens=False),
                 dtype=np.uint32)
        for t in text_iter
    ]

    # 2. Sparse cumulative token counts in permuted (V_w) space.
    counts_list = watermarker.get_cumulative_token_count(
        ids, tokens, watermarker.n_gram,
        return_unshuffled_indices=False,
        return_dense=False,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
    )
    counts = vstack(counts_list, format="csr")
    del counts_list
    gc.collect()

    wf = watermarker.watermarking_fn
    N = wf.N
    n_fns = getattr(wf, "num_fns", N - 2)
    n_rows = counts.shape[0]

    # Prepare / validate candidate column indices (1-based k_p -> 0-based col).
    if candidate_k_ps is not None:
        valid_cols = np.unique(np.asarray(candidate_k_ps, dtype=np.int64) - 1)
        if valid_cols.min() < 0 or valid_cols.max() >= n_fns:
            raise ValueError(
                f"candidate_k_ps must lie in [1, {n_fns}]"
            )
        if top_k > valid_cols.size:
            raise ValueError(
                f"top_k={top_k} > |candidate_k_ps|={valid_cols.size}"
            )
    else:
        valid_cols = None
        if top_k >= n_fns:
            raise ValueError(f"top_k must be < {n_fns}, got {top_k}")

    topk_k_p    = np.empty((n_rows, top_k), dtype=np.int32)
    topk_scores = np.empty((n_rows, top_k), dtype=np.float32)

    iterator = range(0, n_rows, batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc="FFT top-k verify")
    for s in iterator:
        e = min(s + batch_size, n_rows)
        dense = counts[s:e].toarray().astype(np.float32)

        row_sum = dense.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        dense /= row_sum

        # One rFFT -> all q-scores in [1 .. N-2].
        q = fourier_scores(dense, wf)

        # Restrict search to candidate k_ps, if provided.
        q_search = q[:, valid_cols] if valid_cols is not None else q

        # Partial sort for top-k on whatever we're searching over.
        part        = np.argpartition(q_search, -top_k, axis=-1)[:, -top_k:]
        part_scores = np.take_along_axis(q_search, part, axis=-1)
        order       = np.argsort(-part_scores, axis=-1)                # desc
        idx         = np.take_along_axis(part,        order, axis=-1)  # indices into q_search
        sc          = np.take_along_axis(part_scores, order, axis=-1)

        if valid_cols is not None:
            # Map indices-into-valid_cols back to 1-based k_p values.
            topk_k_p[s:e] = (valid_cols[idx] + 1).astype(np.int32)
        else:
            topk_k_p[s:e] = (idx + 1).astype(np.int32)
        topk_scores[s:e] = sc

    topk_k_p    = topk_k_p.reshape(len(texts), len(ids), top_k)
    topk_scores = topk_scores.reshape(len(texts), len(ids), top_k)

    results = {
        "top_k":      top_k,
        "top_k_p":    topk_k_p.tolist(),
        "top_scores": topk_scores.tolist(),
        "closed_set": candidate_k_ps is not None,
        "n_candidates": int(valid_cols.size) if valid_cols is not None else n_fns,
    }
    if candidate_k_ps is not None:
        write_path_file_atomic(experiment_path, "scores_closed.json", results)
    else:
        write_path_file_atomic(experiment_path, "scores_open.json", results)
    end_time = time.time()
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict["verify"] = end_time - start_time
    write_path_file(experiment_path, "latency.json", latency_dict)
    return topk_k_p, topk_scores


def _disable_waterfall_custom_beam_search_for_sampling():
    """Keep Waterfall 0.3.4's group-beam shim out of sampling calls.

    Waterfall 0.3.4 adds ``custom_generate`` globally on Transformers >=5.3,
    including for ordinary sampling.  The bundled implementation only accepts
    group beam search and otherwise rejects the generation configuration.
    """
    import waterfall.WatermarkerBase as waterfall_base

    removed = []
    for key in ("custom_generate", "trust_remote_code"):
        if key in waterfall_base.additional_generation_config:
            waterfall_base.additional_generation_config.pop(key)
            removed.append(key)
    if removed:
        print(
            "Disabled Waterfall's group-beam custom_generate hook for sampling "
            f"({', '.join(removed)})."
        )


def watermark(
    T_os,
    ids,
    k_ps,
    config,
    experiment_path,
    return_sts_scores=False,
    resume=False,
    checkpoint_every=0,
    progress_metadata=None,
    max_new_texts=None,
):
    print("Starting watermarking")
    start_time = time.time()
    if not (len(T_os) == len(ids) == len(k_ps)):
        raise ValueError("T_os, ids, and k_ps must have the same length")

    T_ws = []
    all_sts_scores = []
    if resume and file_exists(experiment_path, "watermarked_texts.json"):
        T_ws = load_path_file(experiment_path, "watermarked_texts.json")
        if not isinstance(T_ws, list) or len(T_ws) > len(T_os):
            raise ValueError(
                "Existing watermarked_texts.json is not a valid partial result "
                f"for this batch ({len(T_ws) if isinstance(T_ws, list) else 'not a list'})."
            )
        if return_sts_scores and T_ws:
            if not file_exists(experiment_path, "sts_scores.json"):
                raise ValueError("Cannot resume: sts_scores.json is missing")
            all_sts_scores = load_path_file(experiment_path, "sts_scores.json")
            if len(all_sts_scores) != len(T_ws):
                raise ValueError("Cannot resume: text and STS checkpoint lengths differ")
        print(f"Resuming after {len(T_ws)} of {len(T_os)} texts")

    initial_completed_count = len(T_ws)

    def save_checkpoint():
        write_path_file_atomic(experiment_path, "watermarked_texts.json", T_ws)
        if return_sts_scores:
            write_path_file_atomic(experiment_path, "sts_scores.json", all_sts_scores)
        progress = dict(progress_metadata or {})
        elapsed_seconds = time.time() - start_time
        generated_this_run = len(T_ws) - initial_completed_count
        progress.update(
            {
                "completed_count": len(T_ws),
                "expected_count": len(T_os),
                "complete": len(T_ws) == len(T_os),
                "started_at_count": initial_completed_count,
                "generated_this_run": generated_this_run,
                "elapsed_seconds_this_run": elapsed_seconds,
                "seconds_per_new_text": (
                    elapsed_seconds / generated_this_run
                    if generated_this_run
                    else None
                ),
            }
        )
        write_path_file_atomic(experiment_path, "progress.json", progress)

    if len(T_ws) == len(T_os):
        print("Batch is already complete; no model load needed.")
        save_checkpoint()
        if return_sts_scores:
            return T_ws, all_sts_scores
        return T_ws

    if max_new_texts is not None and max_new_texts <= 0:
        raise ValueError("max_new_texts must be positive")

    resume_index = len(T_ws)
    run_end_index = len(T_os)
    if max_new_texts is not None:
        run_end_index = min(run_end_index, resume_index + max_new_texts)

    model_name = config["watermark_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    print("Loading model...")
    if not torch.cuda.is_available():
        raise RuntimeError("Watermark generation requires a CUDA GPU")
    model_dtype_kwargs = (
        {"dtype": torch.bfloat16}
        if tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 56)
        else {"torch_dtype": torch.bfloat16}
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        **model_dtype_kwargs,
    ).to("cuda")
    model.eval()

    watermark_fn = config["watermark_fn"]
    n_gram = config["n_gram"]
    kappa = config["kappa"]
    do_sample = config["do_sample_watermark"]
    if do_sample:  # sampling
        _disable_waterfall_custom_beam_search_for_sampling()
        temperature = config["temperature_watermark"]
        top_k = config.get("top_k_watermark", 50)
        top_p = config["top_p_watermark"]
        num_return_sequences = config["num_return_sequences"]
    else:          # beam search 
        num_beam_groups = config["num_beam_groups"]
        beams_per_group = config["beams_per_group"]
        diversity_penalty = config["diversity_penalty"]

    multiple_versions = not do_sample or num_return_sequences>1
    if multiple_versions:
        num_paraphrased_versions = config["num_paraphrased_versions"]
        STS_scale = config["STS_scale"]
        sts_model_name = config["sts_model"]
        sts_model = SentenceTransformer(sts_model_name, device=device)

    if watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError(f"Invalid watermarking function: {watermark_fn}")

    if return_sts_scores:
        sts_model_name = config["sts_model"]
        sts_model = SentenceTransformer(sts_model_name, device=device)

    first_id = int(ids[resume_index])
    first_k_p = int(k_ps[resume_index])
    watermarker = Watermarker(
        tokenizer,
        model,
        first_id,
        kappa,
        first_k_p,
        n_gram,
        watermarkingFnClass,
    )
    chat_template_kwargs = config.get("chat_template_kwargs", {})
    generation_seed = config.get("generation_seed")

    remaining = zip(
        T_os[resume_index:run_end_index],
        ids[resume_index:run_end_index],
        k_ps[resume_index:run_end_index],
    )
    for local_index, (text, watermark_id, k_p) in enumerate(
        tqdm(
            remaining,
            total=run_end_index - resume_index,
            desc="Watermarking texts",
        ),
        start=resume_index,
    ):
        if generation_seed is not None:
            # A per-text seed makes resumed runs reproducible and permits
            # paired generation ablations where only one setting changes.
            transformers.set_seed(int(generation_seed) + local_index)
        watermark_id = int(watermark_id)
        k_p = int(k_p)
        watermarker.k_p = k_p
        if watermarker.id != watermark_id:
            watermarker.set_id(watermark_id)
        else:
            # Preserve the per-instance LRU permutation cache while changing k_p.
            watermarker.compute_phi(watermarkingFnClass)
        
        paraphrasing_prompt = tokenizer.apply_chat_template(
            [
                {"role":"system", "content":PROMPT},
                {"role":"user", "content":text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        ) + PRE_PARAPHRASED
        max_new_tokens = paraphrase_max_new_tokens(
            text, paraphrasing_prompt, tokenizer, config
        )
    
        if do_sample: # sampling
            watermarked = watermarker.generate(
                paraphrasing_prompt,
                return_scores=multiple_versions,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
                num_beams=1,
                num_beam_groups=1,
                diversity_penalty=0.0,
                num_return_sequences=num_return_sequences,
                logits_processor=[],
            )
        else: # beam search
            watermarked = watermarker.generate(
                paraphrasing_prompt,
                return_scores = True,
                max_new_tokens=max_new_tokens,
                do_sample = False, 
                temperature=None, 
                top_p=None,
                num_beams = num_beam_groups * beams_per_group,
                num_beam_groups = num_beam_groups,
                num_return_sequences = num_beam_groups * beams_per_group,
                diversity_penalty = diversity_penalty,    # Only for beam search to make different beams more diverse
                logits_processor = [],
                )
        if multiple_versions:
            if num_paraphrased_versions == 1:
                # Select best paraphrasing based on q_score and semantic similarity
                sts_scores = STS_scorer(text, watermarked["text"], sts_model)
                selection_score = sts_scores * STS_scale + torch.from_numpy(watermarked["q_score"]).squeeze() 
                selection = torch.argmax(selection_score).item() 
                T_w = watermarked["text"][selection]
                T_ws.append(T_w)
            else:
                raise NotImplementedError("Currently, only paraphrased_versions=1 is supported")
        else:  # only one version was returned from watermarked, no need for STS scoring
            T_w = watermarked[0]
            T_ws.append(T_w)
            if return_sts_scores:
                sts_score = STS_scorer(text, T_w, sts_model)
                all_sts_scores.append(sts_score)

        completed_this_batch = local_index + 1
        if checkpoint_every and (
            completed_this_batch % checkpoint_every == 0
            or completed_this_batch == run_end_index
        ):
            save_checkpoint()

    save_checkpoint()
    end_time = time.time()
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict["watermark"] = end_time - start_time
    write_path_file_atomic(experiment_path, "latency.json", latency_dict)
    if return_sts_scores:
        return T_ws, all_sts_scores
    else:
        return T_ws
