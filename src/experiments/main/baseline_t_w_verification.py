"""Baseline: verify the watermarked texts (T_ws) themselves.
Usage:
    python -m src.experiments.main.baseline_t_w_verification [n_samples] [top_k] [t_ws_dir] [keys_file_name]
"""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

from src.util.filereader import load_subset, load_path_file
from src.util.watermark import init_watermarker, verify_watermarks
from src.util.experiment_profile import ExperimentProfile, add_profile_argument

parser = argparse.ArgumentParser()
add_profile_argument(parser)
parser.add_argument("n_samples_exp", nargs="?", default="-1")
parser.add_argument("top_k", nargs="?", default="1")
parser.add_argument("t_ws_dir", nargs="?", default=None)
parser.add_argument("keys_file_name", nargs="?", default="keys.json")
args = parser.parse_args()
profile = ExperimentProfile(args.profile)
profile.check_waterfall()
config = profile.watermark_config

n_samples = int(args.n_samples_exp)
t_ws_dir = args.t_ws_dir or profile.corpus_dir.removeprefix("data/")
top_k = int(args.top_k)
keys_file_name = args.keys_file_name

texts_path = "data/" + t_ws_dir + "/combined_t_ws.json"
keys_path = "data/" + keys_file_name
subset = load_subset(texts_path=texts_path, keys_path=keys_path, n=n_samples,
                     prompts_path=profile.prompt_path("prefix_10.json"))
T_ws = subset["T_ws"]
ids = subset["ids"]
k_ps = subset["k_ps"]

_, _, watermarker = init_watermarker(config, load_model=False)
experiment_path = ["results", "baseline_t_w_verification" + profile.suffix, str(n_samples) + "_samples", "top_" + str(top_k)]
verify_watermarks(T_ws, [ids[0]], k_ps, watermarker, experiment_path, top_k=top_k)
