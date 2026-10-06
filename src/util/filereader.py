from pathlib import Path
import json
import os
import random
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]

def _resolve(parts):
    path = REPO_ROOT
    for part in parts:
        path = path / part
    return path

def load_jsonl_dataset(path=None, n_samples=None):
    dataset_path = _resolve(path) if path is not None else REPO_ROOT / 'data' / 'seeded_dataset.jsonl'
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset not found. Did you forget to download it to the correct directory?")
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_samples is not None and i >= n_samples:
                break
            data.append(json.loads(line))
    return data

def load_abstracts(n_samples: int):
    data = load_jsonl_dataset(n_samples=n_samples)
    abstracts = [d["abstract"] for d in data]
    if n_samples < 0 :
        return abstracts
    else:
        return abstracts[:n_samples]

def load_titles(n_samples):
    data = load_jsonl_dataset(n_samples=n_samples)
    titles = [d["title"] for d in data]
    if n_samples < 0 :
        return titles
    else:
        return titles[:n_samples]

def load_questions(n_samples):
    questions = load_path_file(["data", "prompts", "shared"], "questions.json")
    questions = [[q if q is not None else "" for q in sublist] for sublist in questions] # for None questions replace with empty string
    if n_samples < 0:
        return questions
    else:
        return questions[:n_samples]

def file_exists(path, filename):
    '''
    Checks whether a file exists at a given path (list of strings) with a given filename.
    Returns True if the file exists, False otherwise.
    '''
    return (_resolve(path) / filename).is_file()

def load_path_file(path, filename):
    '''
    Loads a json file from a path defined by a list of strings.
    E.g. path = ["results", "experiment1", "sub_experiment1", "sub_sub_experiment1"]
    '''
    base_path = _resolve(path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"No directory found at path: {'/'.join(path)}")
    file_path = base_path / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Directory found, but doesn't contain file: {filename}")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_path_file(file_path, file_name, data):
    experiment_path = _resolve(file_path)
    if not experiment_path.is_dir():
        experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / file_name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_path_file_atomic(file_path, file_name, data):
    """Write JSON without leaving a truncated file if a job is interrupted."""
    experiment_path = _resolve(file_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    destination = experiment_path / file_name
    fd, temporary_name = tempfile.mkstemp(
        dir=experiment_path,
        prefix=f".{file_name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

def load_or_create_path_file(path, filename):
    '''
    Loads a json file from a path defined by a list of strings.
    If the file does not exist, creates it with an empty dict and returns it.
    '''
    if file_exists(path, filename):
        return load_path_file(path, filename)
    else:
        write_path_file(path, filename, {})
        return {}

HELDOUT_START, HELDOUT_END = 63800, 64000
def load_subset(texts_path: str = "data/t_ws/llama/combined_t_ws.json",
                keys_path: str = "data/keys.json",
                n: int = -1,
                seed: int = 48,
                prompts_path: str = "data/prompts/llama/prefix_10.json", exclude_heldout: bool = True):
    with open(REPO_ROOT / texts_path, "r") as f:
        texts = json.load(f)
    with open(REPO_ROOT / keys_path, "r") as f:
        data = json.load(f)
    with open(REPO_ROOT / prompts_path, "r") as f:
        prompts = json.load(f)
    ids = data["ids"]
    k_ps = data["k_ps"]

    titles = load_titles(64000)

    titles_1 = load_path_file(["data", "prompts", "shared"], "titles_1.json")
    titles_2 = load_path_file(["data", "prompts", "shared"], "titles_2.json")
    titles_3 = load_path_file(["data", "prompts", "shared"], "titles_3.json")
    questions = load_questions(64000)
    train_questions = [q[0] for q in questions]
    held_out_questions = [q[1] for q in questions]

    assert len(texts) == len(ids) == len(k_ps) == len(titles) == len(prompts) == len(train_questions) == len(held_out_questions) == len(titles_1) == len(titles_2) == len(titles_3), "All lists must be the same length."
    if n == -1:
        indices = [i for i in range(len(texts))
                if not (HELDOUT_START <= i < HELDOUT_END)] \
                if exclude_heldout else list(range(len(texts)))
    else:
        pool = [i for i in range(len(texts))
                if not (HELDOUT_START <= i < HELDOUT_END)] \
            if exclude_heldout else list(range(len(texts)))
        assert n <= len(pool), f"n={n} exceeds pool size {len(pool)} (exclude_heldout={exclude_heldout})."
        rng = random.Random(seed)
        shuffled = rng.sample(pool, len(pool))
        indices = sorted(shuffled[:n])

    res = {
        "T_ws" : [texts[i] for i in indices],
        "ids" : [ids[i] for i in indices],
        "k_ps" : [k_ps[i] for i in indices],
        "titles": [titles[i] for i in indices],
        "titles_1": [titles_1[i] for i in indices],
        "titles_2": [titles_2[i] for i in indices],
        "titles_3": [titles_3[i] for i in indices],
        "prompts": [prompts[i] for i in indices],
        "train_questions" : [train_questions[i] for i in indices],
        "held_out_questions" : [held_out_questions[i] for i in indices],
        "prefix_10" : [prompts[i] for i in indices]
    }
    return res



def load_held_out_set(n_samples: int = 200, **subset_kwargs):
    indices = list(range(HELDOUT_START, HELDOUT_END))[:n_samples]
    data = load_subset(n=-1, exclude_heldout=False, **subset_kwargs)
    return {k: [v[i] for i in indices] for k, v in data.items()}


OPEN_KEYSPACE_START, OPEN_KEYSPACE_END = 64000, 65000
def load_open_keyspace_set(n_samples: int = 1000):
    assert n_samples <= OPEN_KEYSPACE_END - OPEN_KEYSPACE_START, (
        f"n_samples={n_samples} exceeds open-keyspace pool size "
        f"{OPEN_KEYSPACE_END - OPEN_KEYSPACE_START}"
    )
    data = load_jsonl_dataset(n_samples=OPEN_KEYSPACE_START + n_samples)
    data = data[OPEN_KEYSPACE_START:OPEN_KEYSPACE_START + n_samples]
    return {
        "abstracts": [d["abstract"] for d in data],
        "titles":    [d["title"] for d in data],
    }

def get_lora_adapter_path(file_path):
    return _resolve(file_path) / 'lora_adapter'
