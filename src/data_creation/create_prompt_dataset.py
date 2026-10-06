"""Generate prefix prompts locally and optional LLM prompts via OpenAI Batch.

Closed keyspace (64000 samples):
    prefix_10.json    first 10 tokens of each watermarked text T_w
    titles_1.json     titles with exactly 1 word replaced
    titles_2.json     titles with exactly 2 words replaced
    titles_3.json     titles with exactly 3 words replaced
    questions.json    [q1, q2] pairs of questions

Open keyspace (1000 held-out negatives):
    prefix_10_open.json, titles_{1,2,3}_open.json, questions_open.json

Each task is split into batches of <50,000 requests due to OpenAI rate limits, 
all batches are started simultaneously and polled until done. 
Batch IDs and intermediate outputs are cached in _batch_work/ in case the run fails.
Generated questions that are empty or too long are regenerated.

Prefix-only generation does not require an OpenAI API key. Titles are derived
only from original titles, and questions are derived only from original
unwatermarked abstracts, so both can be reused when only the watermark model
and watermarked corpus change.

usage:
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py # both sets
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py --set open
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py --set closed
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py \
        --profile data/prompts/config_qwen.json --tasks prefix
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.util.filereader import (
    REPO_ROOT,
    load_abstracts,
    load_open_keyspace_set,
    load_titles,
    write_path_file,
    write_path_file_atomic,
)

sys.stdout.reconfigure(line_buffering=True)
client = None

N_CLOSED = 64000
N_OPEN = 1000
MAX_BATCH_REQUESTS = 50000
MODEL = "gpt-4o-mini"
POLL_INTERVAL_S = 60
PREFIX_LEN = 10

MAX_QUESTION_LEN = 250
REPAIR_WORKERS = 50
MAX_REPAIR_ROUNDS = 3

DEFAULT_WORK_DIR = Path(__file__).parent / "_batch_work"
WORK_DIR = DEFAULT_WORK_DIR
BATCH_IDS_FILE = WORK_DIR / "batch_ids.json"

PERTURB_PROMPTS = {
    "titles_1": """You are given a single paper title. Replace exactly one word with a suitable synonym or similar word.
Rules:
- The replacement must be a different word (not just different capitalization or punctuation)
- You must replace exactly one word
- Return only the modified title, nothing else. No explanation.""",
    "titles_2": """You are given a single paper title. Replace exactly two words with suitable synonyms or similar words.
Rules:
- Each replacement must be a different word (not just different capitalization or punctuation)
- You must replace exactly two words
- Return only the modified title, nothing else. No explanation.""",
    "titles_3": """You are given a single paper title. Replace exactly three words with suitable synonyms or similar words.
Rules:
- Each replacement must be a different word (not just different capitalization or punctuation)
- you must replace exactly three words
- Return only the modified title, nothing else. No explanation.""",
}

QUESTIONS_TASK = "questions"
QUESTIONS_PROMPT = """You are given the abstract of a research paper.
Generate exactly TWO distinct questions whose answers are contained in this paper.

Rules for each question:
- Sound like a natural knowledge or exam question, NOT a question about the paper itself.
- Never reference "the paper", "the authors", "the study", or ask about methodology or experimental details.
- Ask about the underlying concepts and findings.

The two questions must be clearly different from each other in both content and phrasing
(do not paraphrase the same question twice).

Return a JSON object with exactly two keys, "q1" and "q2", whose values are the two
questions as plain strings. Do not include any other keys or commentary."""

PROMPT_FOR = {**PERTURB_PROMPTS, QUESTIONS_TASK: QUESTIONS_PROMPT}
TITLE_TASKS = ("titles_1", "titles_2", "titles_3")
REUSED_CLOSED_FILES = (
    "titles_1.json",
    "titles_2.json",
    "titles_3.json",
    "questions.json",
)
REUSED_OPEN_FILES = (
    "titles_1_open.json",
    "titles_2_open.json",
    "titles_3_open.json",
    "questions_open.json",
)


def _get_client():
    global client
    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for title/question generation, "
                "but not for --tasks prefix"
            )
        client = OpenAI(api_key=api_key)
    return client


def _repo_path(value, argument_name):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{argument_name} must be inside the repository: {value}")
    return path


def _load_json(relative_path):
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_reused_prompts(reuse_dir, filenames, expected_count):
    for filename in filenames:
        relative_path = reuse_dir / filename
        data = _load_json(relative_path)
        if not isinstance(data, list) or len(data) != expected_count:
            raise ValueError(
                f"Cannot reuse {relative_path}: expected {expected_count} "
                f"entries, found {len(data) if isinstance(data, list) else 'non-list data'}"
            )
        print(f"reusing {relative_path} ({len(data)} entries)")


def _load_batch_ids():
    if BATCH_IDS_FILE.is_file():
        return json.loads(BATCH_IDS_FILE.read_text())
    return {}


def _save_batch_ids(batch_ids):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))


def split_chunks(items):
    """Split into the fewest balanced chunks of at most MAX_BATCH_REQUESTS."""
    k = max(1, math.ceil(len(items) / MAX_BATCH_REQUESTS))
    size = math.ceil(len(items) / k)
    return [items[i: i + size] for i in range(0, len(items), size)]


def submit_batch(key, task, inputs, json_mode):
    """Write the request file for one chunk and submit it."""
    openai_client = _get_client()
    body_extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    jsonl_path = WORK_DIR / f"batch_input_{key}.jsonl"
    with open(jsonl_path, "w") as f:
        for i, text in enumerate(inputs):
            line = {
                "custom_id": f"{key}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": PROMPT_FOR[task]},
                        {"role": "user", "content": text},
                    ],
                    **body_extra,
                },
            }
            f.write(json.dumps(line) + "\n")

    with open(jsonl_path, "rb") as request_file:
        batch_file = openai_client.files.create(file=request_file, purpose="batch")
    batch = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[{key}] submitted: {batch.id}")
    return batch.id


def poll_until_done(batch_ids):
    """Poll every batch until all are finished. Returns dict: key -> batch."""
    openai_client = _get_client()
    pending = dict(batch_ids)
    done = {}
    while pending:
        for key, batch_id in list(pending.items()):
            batch = openai_client.batches.retrieve(batch_id)
            c = batch.request_counts
            print(f"[{key}] {batch.status} — {c.completed}/{c.total} completed, {c.failed} failed")
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                done[key] = batch
                del pending[key]
        if pending:
            time.sleep(POLL_INTERVAL_S)
    return done


def collect_raw(batch, expected_len):
    """Collect a completed batch and return raw contents ordered by index."""
    openai_client = _get_client()
    if batch.status != "completed":
        if batch.error_file_id:
            print(openai_client.files.content(batch.error_file_id).text[:1000])
        raise RuntimeError(f"batch ended with status: {batch.status}")
    if batch.error_file_id:
        print("partial errors:")
        print(openai_client.files.content(batch.error_file_id).text[:500])

    result_text = openai_client.files.content(batch.output_file_id).text
    by_idx = {}
    for line in result_text.strip().split("\n"):
        r = json.loads(line)
        idx = int(r["custom_id"].split("_")[-1])
        by_idx[idx] = r["response"]["body"]["choices"][0]["message"]["content"].strip()

    assert len(by_idx) == expected_len, f"result count mismatch: got {len(by_idx)}, expected {expected_len}"
    return [by_idx[i] for i in range(expected_len)]


def parse_two_questions(content):
    """Parse a JSON response into [q1, q2]."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return [None, None]
    q1 = obj.get("q1") or obj.get("question1") or obj.get("Q1")
    q2 = obj.get("q2") or obj.get("question2") or obj.get("Q2")
    return [
        q1.strip() if isinstance(q1, str) else None,
        q2.strip() if isinstance(q2, str) else None,
    ]


def question_is_valid(entry):
    """A valid entry is [q1, q2] with two non-empty strings under the max length."""
    if not isinstance(entry, list) or len(entry) != 2:
        return False
    for q in entry:
        if not isinstance(q, str) or q.strip() == "" or len(q) > MAX_QUESTION_LEN:
            return False
    return True


def regenerate_one(abstract):
    """Single call to regenerate a [q1, q2] pair for one abstract."""
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QUESTIONS_PROMPT},
            {"role": "user", "content": abstract},
        ],
        response_format={"type": "json_object"},
    )
    return parse_two_questions(resp.choices[0].message.content.strip())


def repair_questions(questions, abstracts):
    """Regenerate any invalid entry, repeating until max num of retries is reached."""
    for round_num in range(1, MAX_REPAIR_ROUNDS + 1):
        bad = [i for i, e in enumerate(questions) if not question_is_valid(e)]
        if not bad:
            print("all question entries valid")
            return questions
        print(f"repair round {round_num}: regenerating {len(bad)} invalid entries")
        with ThreadPoolExecutor(max_workers=REPAIR_WORKERS) as pool:
            futures = {pool.submit(regenerate_one, abstracts[i]): i for i in bad}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    questions[i] = fut.result()
                except Exception as e:
                    print(f"  idx={i} failed: {e}")
    leftover = sum(1 for e in questions if not question_is_valid(e))
    print(f"repair finished with {leftover} still-invalid entries (left as-is)")
    return questions


def run_task(task, suffix, inputs, json_mode, batch_ids):
    """Submit (if needed), poll, collect and combine all chunks for one task."""
    chunks = split_chunks(inputs)
    keys = [f"{task}{suffix}_chunk{c}" for c in range(len(chunks))]

    for key, chunk in zip(keys, chunks):
        if key in batch_ids or (WORK_DIR / f"{key}.json").is_file():
            continue
        batch_ids[key] = submit_batch(key, task, chunk, json_mode)
        _save_batch_ids(batch_ids)

    to_poll = {
        key: batch_ids[key]
        for key in keys
        if key in batch_ids and not (WORK_DIR / f"{key}.json").is_file()
    }
    completed = poll_until_done(to_poll) if to_poll else {}

    combined = []
    for key, chunk in zip(keys, chunks):
        cached = WORK_DIR / f"{key}.json"
        if cached.is_file():
            part = json.loads(cached.read_text())
        else:
            raw = collect_raw(completed[key], expected_len=len(chunk))
            part = [parse_two_questions(c) for c in raw] if json_mode else raw
            cached.write_text(json.dumps(part, ensure_ascii=False))
        combined += part
    return combined


def build_prefix(texts, watermark_config):
    """First PREFIX_LEN tokens of each text as a string, using
    the watermark model's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(watermark_config["watermark_model"])
    tokenizer.pad_token = tokenizer.eos_token
    return [
        tokenizer.decode(
            tokenizer.encode(text, max_length=PREFIX_LEN, truncation=True),
            skip_special_tokens=True,
        )
        for text in texts
    ]


def generate_api_tasks(suffix, titles, abstracts, batch_ids, tasks, output_dir):
    """Run the requested OpenAI tasks and write their output files."""
    output_parts = list(output_dir.parts)
    generated = []
    if "titles" in tasks:
        for task in TITLE_TASKS:
            combined = run_task(
                task, suffix, titles, json_mode=False, batch_ids=batch_ids
            )
            write_path_file(output_parts, f"{task}{suffix}.json", combined)
            generated.append(f"{task}{suffix}.json")
            print(
                f"wrote {output_dir}/{task}{suffix}.json "
                f"({len(combined)} entries)"
            )

    if "questions" in tasks:
        questions = run_task(
            QUESTIONS_TASK,
            suffix,
            abstracts,
            json_mode=True,
            batch_ids=batch_ids,
        )
        questions = repair_questions(questions, abstracts)
        write_path_file(
            output_parts, f"{QUESTIONS_TASK}{suffix}.json", questions
        )
        generated.append(f"{QUESTIONS_TASK}{suffix}.json")
        print(
            f"wrote {output_dir}/{QUESTIONS_TASK}{suffix}.json "
            f"({len(questions)} entries)"
        )
    return generated


def _load_profile(profile_path):
    if profile_path is None:
        return {}
    return _load_json(_repo_path(profile_path, "--profile"))


def _resolve_tasks(values):
    values = set(values)
    if "all" in values:
        return {"prefix", "titles", "questions"}
    return values


def main():
    global WORK_DIR, BATCH_IDS_FILE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["closed", "open", "both"], default="both")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", "prefix", "titles", "questions"],
        default=["all"],
        help="prompt groups to generate; Qwen only needs prefix",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="JSON profile defining watermark config, corpus, and output paths",
    )
    parser.add_argument("--watermark-config", default=None)
    parser.add_argument("--watermarked-texts", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-prompts-dir", default=None)
    parser.add_argument("--shared-prompts-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--include-unwatermarked-prefix", action="store_true",
                        help="also prepare tokenizer-specific control prefixes from original abstracts")
    args = parser.parse_args()

    profile = _load_profile(args.profile)
    watermark_config_path = _repo_path(
        args.watermark_config
        or profile.get("watermark_config", "data/t_ws/config_llama.json"),
        "--watermark-config",
    )
    watermarked_texts_path = _repo_path(
        args.watermarked_texts
        or profile.get("watermarked_texts", "data/t_ws/llama/combined_t_ws.json"),
        "--watermarked-texts",
    )
    output_dir = _repo_path(
        args.output_dir or profile.get("output_dir", "data/prompts/llama"),
        "--output-dir",
    )
    reuse_prompts_dir = _repo_path(
        args.reuse_prompts_dir
        or profile.get("reuse_prompts_dir", "data/prompts/shared"),
        "--reuse-prompts-dir",
    )
    shared_prompts_dir = _repo_path(
        args.shared_prompts_dir
        or profile.get("shared_prompts_dir", "data/prompts/shared"),
        "--shared-prompts-dir",
    )
    work_dir_value = args.work_dir or profile.get("work_dir")
    if work_dir_value:
        WORK_DIR = REPO_ROOT / _repo_path(work_dir_value, "--work-dir")
    else:
        WORK_DIR = DEFAULT_WORK_DIR
    BATCH_IDS_FILE = WORK_DIR / "batch_ids.json"

    tasks = _resolve_tasks(args.tasks)
    watermark_config = _load_json(watermark_config_path)
    reused_files = []
    if tasks == {"prefix"}:
        if args.set in ("closed", "both"):
            _validate_reused_prompts(
                reuse_prompts_dir, REUSED_CLOSED_FILES, N_CLOSED
            )
            reused_files.extend(REUSED_CLOSED_FILES)
        if args.set in ("open", "both"):
            _validate_reused_prompts(
                reuse_prompts_dir, REUSED_OPEN_FILES, N_OPEN
            )
            reused_files.extend(REUSED_OPEN_FILES)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    batch_ids = _load_batch_ids()
    generated_files = []

    if args.set in ("closed", "both"):
        print("=== closed set (64000 samples) ===")
        if args.include_unwatermarked_prefix:
            prefixes = build_prefix(load_abstracts(N_CLOSED), watermark_config)
            if len(prefixes) != N_CLOSED:
                raise ValueError("Insufficient original abstracts for control prefixes")
            write_path_file_atomic(list(output_dir.parts), "prefix_10_unwatermarked.json", prefixes)
            generated_files.append("prefix_10_unwatermarked.json")
        if "prefix" in tasks:
            t_ws = _load_json(watermarked_texts_path)
            if len(t_ws) != N_CLOSED:
                raise ValueError(
                    f"{watermarked_texts_path} has {len(t_ws)} texts; "
                    f"expected {N_CLOSED}"
                )
            prefix = build_prefix(t_ws, watermark_config)
            write_path_file_atomic(
                list(output_dir.parts), "prefix_10.json", prefix
            )
            generated_files.append("prefix_10.json")
            print(
                f"wrote {output_dir}/prefix_10.json ({len(prefix)} entries)"
            )

        if tasks & {"titles", "questions"}:
            titles = load_titles(N_CLOSED) if "titles" in tasks else None
            abstracts = (
                load_abstracts(N_CLOSED) if "questions" in tasks else None
            )
            generated_files.extend(
                generate_api_tasks(
                    "", titles, abstracts, batch_ids, tasks, shared_prompts_dir
                )
            )

    if args.set in ("open", "both"):
        print("=== open keyspace (1000 held-out samples) ===")
        open_set = load_open_keyspace_set(n_samples=N_OPEN)
        titles, abstracts = open_set["titles"], open_set["abstracts"]

        if "prefix" in tasks:
            prefix = build_prefix(abstracts, watermark_config)
            write_path_file_atomic(
                list(output_dir.parts), "prefix_10_open.json", prefix
            )
            generated_files.append("prefix_10_open.json")
            print(
                f"wrote {output_dir}/prefix_10_open.json "
                f"({len(prefix)} entries)"
            )

        if tasks & {"titles", "questions"}:
            generated_files.extend(
                generate_api_tasks(
                    "_open", titles, abstracts, batch_ids, tasks, shared_prompts_dir
                )
            )

    manifest = {
        "watermark_config": str(watermark_config_path),
        "watermark_model": watermark_config["watermark_model"],
        "watermarked_texts": str(watermarked_texts_path),
        "generated_tasks": sorted(tasks),
        "generated_files": generated_files,
        "reuse_prompts_dir": str(reuse_prompts_dir),
        "shared_prompts_dir": str(shared_prompts_dir),
        "reused_files": list(reused_files),
        "question_source": "original unwatermarked abstracts",
        "title_source": "original titles",
    }
    write_path_file_atomic(
        list(output_dir.parts), "prompt_manifest.json", manifest
    )


if __name__ == "__main__":
    main()
