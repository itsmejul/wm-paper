from src.util.filereader import load_abstracts, load_path_file, write_path_file
from src.util.key_generator import generate_keys
from src.util.watermark import watermark, init_watermarker, verify_watermarks
import json 
import os 
from pathlib import Path 
import sys
sys.stdout.reconfigure(line_buffering=True)

experiment_name = "3_ngram_ablation"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "sub_experiment_name",
    nargs="?",              # makes it optional
    default=""
)
args = parser.parse_args()
print(f"Using: {args.sub_experiment_name}")
sub_experiment_name = args.sub_experiment_name
if sub_experiment_name == "":
    sub_experiment_names = ["one", "two", "three", "four", "five"]
else:
    sub_experiment_names = [sub_experiment_name]

default_config = load_path_file(["results", "ablations", experiment_name], "default_config.json")

n_samples = default_config["n_samples"]

T_os = load_abstracts(n_samples)
print(T_os)

# generate keys
for sub_experiment_name in sub_experiment_names:
    config = load_path_file(["results", "ablations", experiment_name, sub_experiment_name], "config.json")
    config = config | default_config # combine configs
    print(config)
    id_gen_type = config["id_gen"]
    kp_gen_type = config["k_p_gen"]
    keys = generate_keys(n_samples, id_type=id_gen_type, kp_type=kp_gen_type)
    ids = keys["ids"]
    k_ps = keys["k_ps"]
    write_path_file(["results", "ablations", experiment_name, sub_experiment_name], "keys.json", keys)

    tokenizer, model, watermarker = init_watermarker(config)
    # watermark 
    experiment_path = ["results", "ablations", experiment_name, sub_experiment_name]
    T_ws = watermark(T_os, ids, k_ps, config, experiment_path)
    # verify
    
    verify_watermarks(T_ws, [ids[0]], k_ps, watermarker, experiment_path)
