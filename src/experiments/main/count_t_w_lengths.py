"""Count the token length of every watermarked text and save the list to
data/t_ws/llama/token_lengths.json.

Usage:
    python -m src.experiments.main.count_t_w_lengths
"""

import argparse
from src.util.experiment_profile import ExperimentProfile, add_profile_argument
from transformers import AutoTokenizer

from src.util.filereader import load_path_file, write_path_file

parser = argparse.ArgumentParser(description=__doc__)
add_profile_argument(parser)
profile = ExperimentProfile(parser.parse_args().profile)
T_ws = load_path_file([profile.corpus_dir], "combined_t_ws.json")
print(len(T_ws))

config = profile.watermark_config
model_name = config["watermark_model"]
print(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lengths = [len(tokenizer.encode(t_w)) for t_w in T_ws]
write_path_file([profile.corpus_dir], "token_lengths.json", lengths)
