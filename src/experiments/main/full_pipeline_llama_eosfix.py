"""Run the historical Llama experiments with isolated EOS-fixed outputs.

This is a convenience entrypoint for ``full_pipeline --eos-fix``. It keeps the
old default pipeline unchanged while making it difficult to accidentally place
new adapters or results in the historical directories.

Usage:
    python -m src.experiments.main.full_pipeline_llama_eosfix \
        [n_samples] [sample_type] [batch_size] [n_eval_samples] [--eval-only]
"""

import runpy
import sys


if "--eos-fix" not in sys.argv:
    sys.argv.append("--eos-fix")

runpy.run_module("src.experiments.main.full_pipeline", run_name="__main__")
