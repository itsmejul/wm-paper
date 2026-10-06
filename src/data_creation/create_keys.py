"""Generate the watermark keys from the canonical Llama watermark config.

Each of the n_samples texts is assigned an id and a k_p key
based on the config generation strategy. 
See src/util/key_generator.py for the generation strategies.
"""
import sys

from src.util.filereader import load_path_file, write_path_file
from src.util.key_generator import generate_keys

sys.stdout.reconfigure(line_buffering=True)

config = load_path_file(["data", "t_ws"], "config_llama.json")

keys = generate_keys(
    config["n_samples"],
    id_type=config["id_gen"],
    kp_type=config["k_p_gen"],
)
write_path_file(["data"], "keys.json", keys)

k_ps = keys["k_ps"]
print(
    f"Wrote {len(k_ps)} keys: {len(set(k_ps))} distinct k_p "
    f"in [{min(k_ps)}, {max(k_ps)}]"
)
