from functools import partial
import argparse
import os
from pathlib import Path
from config import ExperimentContext, RunOptions
import data
import evaluate
import model
import experiments
import reports


def free_gpu(ctx):
    ctx.gc.collect()
    ctx.torch.cuda.empty_cache()


def initialize(ctx):
    import os

    ctx.os = os
    ctx.os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ctx.os.environ["WANDB_DISABLED"] = "true"
    ctx.os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ctx.os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from unsloth import FastLanguageModel, is_bfloat16_supported

    ctx.FastLanguageModel = FastLanguageModel
    ctx.is_bfloat16_supported = is_bfloat16_supported
    import torch

    ctx.torch = torch
    if not ctx.torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA GPU visible. Use a supported NVIDIA GPU on Linux or Kaggle. On Kaggle: right panel > Session options > Accelerator > GPU T4 x2 (not P100), then Run All again. Greyed out? Verify your phone number in Kaggle settings or check your weekly GPU quota. (torch {ctx.torch.__version__}, CUDA build {ctx.torch.version.cuda})"
        )
    print(
        f"GPU OK: {ctx.torch.cuda.get_device_name(0)} | torch {ctx.torch.__version__} | CUDA {ctx.torch.version.cuda}"
    )
    from unsloth import FastLanguageModel, is_bfloat16_supported

    ctx.FastLanguageModel = FastLanguageModel
    ctx.is_bfloat16_supported = is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    ctx.train_on_responses_only = train_on_responses_only
    import re, json, sqlite3, random, time, math, gc, zlib, collections, platform, glob, shutil

    ctx.re = re
    ctx.json = json
    ctx.sqlite3 = sqlite3
    ctx.random = random
    ctx.time = time
    ctx.math = math
    ctx.gc = gc
    ctx.zlib = zlib
    ctx.collections = collections
    ctx.platform = platform
    ctx.glob = glob
    ctx.shutil = shutil
    import numpy as np, pandas as pd

    ctx.np = np
    ctx.pd = pd
    import matplotlib.pyplot as plt

    ctx.plt = plt
    from matplotlib.colors import LinearSegmentedColormap

    ctx.LinearSegmentedColormap = LinearSegmentedColormap
    from utils import display, Markdown

    ctx.display = display
    ctx.Markdown = Markdown
    from tqdm.auto import tqdm

    ctx.tqdm = tqdm
    from datasets import load_dataset, Dataset

    ctx.load_dataset = load_dataset
    ctx.Dataset = Dataset
    from trl import SFTTrainer, SFTConfig

    ctx.SFTTrainer = SFTTrainer
    ctx.SFTConfig = SFTConfig
    from scipy.stats import binomtest, chi2_contingency

    ctx.binomtest = binomtest
    ctx.chi2_contingency = chi2_contingency
    import config
    import reports
    import utils

    config.configure(ctx)
    reports.initialize_plotting(ctx)
    utils.initialize_cache(ctx)
    ctx.free_gpu = partial(free_gpu, ctx)


STAGES = (
    ("Setup and configuration", initialize),
    ("Dataset preparation", data.load_data),
    ("Exploratory data analysis", reports.run_exploration),
    ("Evaluation metrics and databases", evaluate.prepare_evaluation),
    ("Model and training services", model.check_token_lengths),
    ("Hyperparameter tuning", experiments.run_tuning),
    ("Main comparison and seed robustness", experiments.run_comparison),
    ("Error analysis", reports.run_errors),
    ("Cross-dataset generalisation", experiments.run_generalisation),
    ("Model scaling", experiments.run_scaling),
    ("Training-size ablation", experiments.run_ablation),
    ("Results discussion", reports.run_discussion),
    ("Saved adapter verification", reports.check_adapter),
    ("Export artifacts", reports.run_artifacts),
)


def run(options: RunOptions) -> ExperimentContext:
    """Run every section in notebook order and return the measured results.

    SQLite connections are closed before returning; saved results remain usable.
    GPU dependencies load only when the runtime stage starts.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    context = ExperimentContext(options)
    try:
        for index, (name, stage) in enumerate(STAGES, 1):
            print(f"\n[{index}/{len(STAGES)}] {name}", flush=True)
            stage(context)
        return context
    finally:
        context.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Use notebook smoke-test sizes."
    )
    parser.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[
            "tuning",
            "fewshot",
            "default_test",
            "datasets",
            "scaling",
            "size_ablation",
        ],
    )
    args = parser.parse_args(argv)
    if args.resume_from and (not args.resume_from.is_dir()):
        parser.error("--resume-from must be an existing directory")
    run(
        RunOptions(
            args.quick, args.model, args.output_dir, args.resume_from, tuple(args.skip)
        )
    )


if __name__ == "__main__":
    main()
