from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils import write_json


@dataclass(frozen=True)
class RunOptions:
    quick: bool = False
    model: str = "unsloth/Qwen3.5-0.8B"
    output_dir: Path = Path("outputs")
    resume_from: Path | None = None
    skip: tuple[str, ...] = ()


class ExperimentContext:
    """Mutable experiment state with deterministic resource cleanup."""

    def __init__(self, options: RunOptions):
        self.options = options

    def close(self) -> None:
        for _, database_sets, _ in getattr(self, "EVAL_SETS", {}).values():
            for databases in database_sets:
                for connection in databases:
                    connection.close()
        torch: Any = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def configure(ctx):
    ctx.QUICK_RUN = ctx.options.quick
    ctx.RUN_TUNING = "tuning" not in ctx.options.skip
    ctx.RUN_FEWSHOT = "fewshot" not in ctx.options.skip
    ctx.RUN_DEFAULT_TEST = "default_test" not in ctx.options.skip
    ctx.RUN_DATASETS = "datasets" not in ctx.options.skip
    ctx.RUN_SCALING = "scaling" not in ctx.options.skip
    ctx.RUN_SIZE_ABLATION = "size_ablation" not in ctx.options.skip
    ctx.MODEL_NAME = ctx.options.model
    ctx.SCALING_MODELS = {"2B": "unsloth/Qwen3.5-2B"}
    ctx.RUN_4B_4BIT = False
    ctx.MODEL_4B = "unsloth/Qwen3.5-4B"
    ctx.DS_A = "b-mc2/sql-create-context"
    ctx.DS_B = "gretelai/synthetic_text_to_sql"
    ctx.N_TRAIN = 200 if ctx.QUICK_RUN else 2000
    ctx.N_VAL = 50 if ctx.QUICK_RUN else 100
    ctx.N_TUNE = 60 if ctx.QUICK_RUN else 300
    ctx.N_TEST = 50 if ctx.QUICK_RUN else 200
    ctx.N_SHOTS = 3
    ctx.B_POOL = ctx.N_TRAIN + ctx.N_VAL + 400
    ctx.MAX_PROMPT_CHARS = 2600
    ctx.ABLATION_SIZES = [50, 100] if ctx.QUICK_RUN else [250, 500, 1000]
    ctx.SEEDS = [3407, 42] if ctx.QUICK_RUN else [3407, 42, 1234]
    if ctx.QUICK_RUN:
        ctx.SCALING_MODELS = {"2B": ctx.SCALING_MODELS["2B"]}
    ctx.BASE_HP = {
        "lora_r": 16,
        "alpha_mult": 1,
        "lr": 0.0002,
        "epochs": 1,
        "targets": "all",
    }
    ctx.TUNE_GRID = (
        {"lora_r": [8, 16], "lr": [0.0002, 0.0005]}
        if ctx.QUICK_RUN
        else {
            "lora_r": [4, 8, 16, 32, 64],
            "lr": [5e-05, 0.0001, 0.0002, 0.0005],
            "epochs": [1, 2, 3],
            "targets": ["attn", "all"],
            "alpha_mult": [1, 2],
        }
    )
    ctx.TUNE_TOL = 1.0
    ctx.PREFER = {
        "lora_r": "low",
        "epochs": "low",
        "lr": "base",
        "targets": "base",
        "alpha_mult": "base",
    }
    ctx.HP_LABELS = {
        "lora_r": "LoRA rank r",
        "lr": "learning rate",
        "epochs": "epochs",
        "targets": "target modules",
        "alpha_mult": "alpha / r",
    }
    ctx.MAX_SEQ_LEN = 1024
    ctx.BATCH_BY_SIZE = {"0.8B": (4, 4), "2B": (4, 4), "4B-4bit": (2, 8)}
    ctx.GEN_BATCH_SIZE = 16
    ctx.MAX_NEW_TOKENS = 192
    ctx.N_EXEC_DBS, ctx.ROWS_PER_TABLE = (3, 20)
    ctx.N_BOOT = 2000
    ctx.SEED = ctx.SEEDS[0]
    ctx.OUT_DIR = str(ctx.options.output_dir.resolve())
    ctx.FIG_DIR, ctx.RUN_DIR, ctx.ADAPTER_DIR = (
        f"{ctx.OUT_DIR}/figures",
        f"{ctx.OUT_DIR}/runs",
        f"{ctx.OUT_DIR}/adapters",
    )
    for ctx.d in (ctx.FIG_DIR, ctx.RUN_DIR, ctx.ADAPTER_DIR):
        ctx.os.makedirs(ctx.d, exist_ok=True)
    ctx.SAVE_ALL_ADAPTERS = False
    ctx.PFX = "quick_" if ctx.QUICK_RUN else ""
    ctx.random.seed(ctx.SEED)
    ctx.np.random.seed(ctx.SEED)
    ctx.torch.manual_seed(ctx.SEED)
    ctx.CONFIG = {
        k: v
        for k, v in vars(ctx).items()
        if k.isupper() and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }
    write_json(f"{ctx.OUT_DIR}/config.json", ctx.CONFIG, indent=2, default=str)
    print(
        f"GPU: {ctx.torch.cuda.get_device_name(0)} | bf16: {ctx.is_bfloat16_supported()} | Python {ctx.platform.python_version()}"
    )
    ctx.pd.DataFrame(
        [(k, str(v)) for k, v in ctx.CONFIG.items()], columns=["setting", "value"]
    ).set_index("setting")
