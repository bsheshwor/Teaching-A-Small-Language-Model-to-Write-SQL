"""Train a single adapter using the notebook's data splits and LoRA recipe."""

import argparse
import os
from pathlib import Path
from config import RunOptions
from model import train_adapter
from utils import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--dataset", choices=["a", "b", "mixed"], default="a")
    parser.add_argument(
        "--hp-config", type=Path, help="JSON hyperparameters or saved best_config.json."
    )
    args = parser.parse_args()
    hp = read_json(args.hp_config) if args.hp_config else None
    if hp is not None:
        hp = hp.get("best_hp", hp)
    os.environ.setdefault("MPLBACKEND", "Agg")
    adapter = train_adapter(
        RunOptions(quick=args.quick, model=args.model, output_dir=args.output_dir),
        dataset=args.dataset,
        hyperparameters=hp,
    )
    print(f"Saved adapter: {adapter}")


if __name__ == "__main__":
    main()
