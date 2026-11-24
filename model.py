from functools import partial
from experiments import experiment, is_oom, ex_of
import json
import os
from pathlib import Path
from utils import build_prompt, extract_sql
import random
from config import ExperimentContext, RunOptions
import data
from utils import write_json


def load_model(ctx, model_name, four_bit=False):
    model, tokenizer = ctx.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=ctx.MAX_SEQ_LEN,
        load_in_4bit=four_bit,
        load_in_16bit=not four_bit,
        full_finetuning=False,
    )
    ctx.tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    if ctx.tok.pad_token is None:
        ctx.tok.pad_token = ctx.tok.eos_token
    ctx.IM_END_ID = ctx.tok.convert_tokens_to_ids("<|im_end|>")
    return (model, tokenizer)


def to_text(ctx, row):
    return {"text": ctx.build_prompt(row) + row["answer"] + "<|im_end|>"}


def finetune(
    ctx,
    model,
    tokenizer,
    train_rows,
    val_rows,
    hp,
    seed,
    size_tag="0.8B",
    tag="run",
    verbose=True,
):
    ctx.FastLanguageModel.for_training(model)
    model = ctx.FastLanguageModel.get_peft_model(
        model,
        r=hp["lora_r"],
        lora_alpha=hp["lora_r"] * hp["alpha_mult"],
        lora_dropout=0,
        bias="none",
        target_modules=ctx.TARGETS[hp["targets"]],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=ctx.MAX_SEQ_LEN,
    )
    tr = ctx.Dataset.from_list([ctx.to_text(r) for r in train_rows])
    va = ctx.Dataset.from_list([ctx.to_text(r) for r in val_rows]) if val_rows else None
    bsz, acc = ctx.BATCH_BY_SIZE.get(size_tag, (4, 4))
    steps = ctx.math.ceil(len(tr) / (bsz * acc)) * hp["epochs"]
    args = ctx.SFTConfig(
        dataset_text_field="text",
        max_length=ctx.MAX_SEQ_LEN,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=acc,
        num_train_epochs=hp["epochs"],
        learning_rate=hp["lr"],
        lr_scheduler_type="linear",
        warmup_steps=max(1, min(10, steps // 10)),
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not ctx.is_bfloat16_supported(),
        bf16=ctx.is_bfloat16_supported(),
        logging_steps=max(1, steps // 40),
        eval_strategy="steps" if va is not None else "no",
        eval_steps=max(1, steps // 8),
        per_device_eval_batch_size=8,
        output_dir=f"{ctx.OUT_DIR}/outputs_tmp",
        save_strategy="no",
        report_to="none",
        seed=seed,
        dataset_num_proc=1,
    )
    trainer = ctx.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tr,
        eval_dataset=va,
        args=args,
    )
    trainer = ctx.train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        tokenizer=ctx.tok,
    )
    kept = ctx.tok.decode([t for t in trainer.train_dataset[0]["labels"] if t != -100])
    assert "<|im_start|>" not in kept and "Schema:" not in kept and kept, (
        f"Masking failed: {kept!r}"
    )
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total = sum((p.numel() for p in model.parameters()))
    if verbose:
        print(
            f"[{tag}] {len(tr)} ex, {steps} steps, hp={hp}, seed={seed} | trainable {trainable / 1000000.0:.2f}M ({100 * trainable / total:.2f}%)"
        )
    ctx.torch.cuda.reset_peak_memory_stats()
    t0 = ctx.time.time()
    stats = trainer.train()
    assert not ctx.math.isnan(stats.training_loss), (
        "Loss is NaN (fp16 overflow) - lower the learning rate"
    )
    ev = [x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x]
    info = {
        "train_examples": len(tr),
        "steps": trainer.state.global_step,
        "train_time_min": (ctx.time.time() - t0) / 60,
        "peak_vram_gb": ctx.torch.cuda.max_memory_allocated() / 1000000000.0,
        "final_train_loss": stats.training_loss,
        "final_val_loss": ev[-1] if ev else None,
        "trainable_params_M": trainable / 1000000.0,
        "total_params_M": total / 1000000.0,
    }
    return (model, trainer, info)


def generate(ctx, model, rows, shots=None, batch_size=None, max_new_tokens=None):
    if batch_size is None:
        batch_size = ctx.GEN_BATCH_SIZE
    if max_new_tokens is None:
        max_new_tokens = ctx.MAX_NEW_TOKENS
    ctx.FastLanguageModel.for_inference(model)
    ctx.tok.padding_side = "left"
    raws, t0 = ([], ctx.time.time())
    bs = max(1, batch_size // 2) if shots else batch_size
    for i in ctx.tqdm(range(0, len(rows), bs), desc="generating", leave=False):
        batch = rows[i : i + bs]
        enc = ctx.tok(
            [ctx.build_prompt(r, shots) for r in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=ctx.tok.pad_token_id,
            eos_token_id=ctx.IM_END_ID,
        )
        raws += ctx.tok.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
    secs = (ctx.time.time() - t0) / len(rows)
    ctx.tok.padding_side = "right"
    return ([ctx.extract_sql(r) for r in raws], raws, secs)


def initialize(ctx):
    """Register model, generation, training and experiment services."""
    ctx.TARGETS = {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    ctx.load_model = partial(load_model, ctx)
    ctx.to_text = partial(to_text, ctx)
    ctx.generate = ctx.torch.no_grad()(partial(generate, ctx))
    ctx.finetune = partial(finetune, ctx)
    ctx.experiment = partial(experiment, ctx)
    ctx.is_oom = partial(is_oom, ctx)
    ctx.ex_of = partial(ex_of, ctx)


def check_token_lengths(ctx):
    """Check training token lengths and save the original notebook figure."""
    initialize(ctx)
    ctx.model, ctx.tokenizer = ctx.load_model(ctx.MODEL_NAME)
    ctx.lens = {
        name: ctx.np.array(
            [
                len(
                    ctx.tok(ctx.to_text(r)["text"], add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                for r in rows
            ]
        )
        for name, rows in [("A train", ctx.train_A), ("B train", ctx.train_B)]
    }
    print(
        "Example prompt fed to the model:\n"
        + "-" * 60
        + "\n"
        + ctx.build_prompt(ctx.test_A[0])
    )
    del ctx.model, ctx.tokenizer
    ctx.free_gpu()
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(8.5, 3.8))
    ctx.hi = max((v.max() for v in ctx.lens.values()))
    ctx.bins = ctx.np.linspace(0, ctx.hi, 50)
    for (ctx.name, ctx.v), ctx.c, ctx.kw in zip(
        ctx.lens.items(),
        [ctx.A_C, ctx.B_C],
        [{"alpha": 0.55}, {"histtype": "step", "linewidth": 2}],
    ):
        ctx.ax.hist(
            ctx.v,
            bins=ctx.bins,
            color=ctx.c,
            label=f"{ctx.name} (median {ctx.np.median(ctx.v):.0f})",
            **ctx.kw,
        )
    ctx.ax.axvline(ctx.MAX_SEQ_LEN, color=ctx.INK, linestyle="--", linewidth=1)
    ctx.ax.annotate(
        f"limit {ctx.MAX_SEQ_LEN}",
        (ctx.MAX_SEQ_LEN, ctx.ax.get_ylim()[1] * 0.9),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    ctx.ax.set_xlabel("tokens per training example (prompt + answer)")
    ctx.ax.set_ylabel("examples")
    ctx.ax.legend()
    ctx.ax.set_title("Figure 6 - Training sequence lengths")
    ctx.savefig(ctx.fig, "fig06_token_lengths", "Training sequence lengths in tokens")
    for ctx.name, ctx.v in ctx.lens.items():
        print(
            f"{ctx.name}: mean {ctx.v.mean():.0f}, max {ctx.v.max()}, truncated: {(ctx.v > ctx.MAX_SEQ_LEN).mean() * 100:.2f}%"
        )


class Text2SQL:
    def __init__(self, model_path, max_seq_length=1024, four_bit=None):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from unsloth import FastLanguageModel
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Inference requires a supported NVIDIA CUDA GPU with Unsloth."
            )
        metadata = Path(model_path) / "run_info.json"
        info = json.loads(metadata.read_text()) if metadata.is_file() else {}
        if four_bit is None:
            four_bit = info.get("four_bit", False)
        self.model, processor = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            load_in_4bit=four_bit,
            load_in_16bit=not four_bit,
            full_finetuning=False,
        )
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.max_seq_length = max_seq_length
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        FastLanguageModel.for_inference(self.model)

    def predict(self, question, schema, max_new_tokens=192):
        """Return generated SQL. This method does not execute the query."""
        import torch

        if not question.strip() or not schema.strip():
            raise ValueError("Question and schema must both be nonempty.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        prompt = build_prompt(self.tokenizer, {"question": question, "context": schema})
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = inputs["input_ids"].shape[1]
        if length + max_new_tokens > self.max_seq_length:
            raise ValueError(
                "Prompt plus generation exceeds max_seq_length; increase it or shorten the schema."
            )
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.end_id,
            )
        raw = self.tokenizer.decode(output[0, length:], skip_special_tokens=True)
        return extract_sql(raw)


def train_adapter(options: RunOptions, dataset="a", hyperparameters=None):
    """Use notebook splits and defaults, then save one adapter and its training log."""
    if dataset not in ("a", "b", "mixed"):
        raise ValueError("dataset must be a, b or mixed")
    ctx = ExperimentContext(options)
    model = tokenizer = trainer = None
    try:
        from train import initialize as initialize_runtime

        initialize_runtime(ctx)
        data.load_data(ctx)
        initialize(ctx)
        ctx.build_prompt = lambda row, shots=None: build_prompt(ctx.tok, row, shots)
        hp = dict(ctx.BASE_HP)
        if hyperparameters:
            unknown = set(hyperparameters) - set(hp)
            if unknown:
                raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
            hp.update(hyperparameters)
        if dataset == "a":
            rows, validation = (ctx.train_A, ctx.val_A)
        elif dataset == "b":
            rows, validation = (ctx.train_B, ctx.val_B)
        else:
            half = ctx.N_TRAIN // 2
            rows = ctx.train_A[:half] + ctx.train_B[: ctx.N_TRAIN - half]
            random.Random(ctx.SEED).shuffle(rows)
            validation = ctx.val_A + ctx.val_B
        model, tokenizer = ctx.load_model(ctx.MODEL_NAME)
        model, trainer, info = ctx.finetune(
            model, tokenizer, rows, validation, hp, ctx.SEED
        )
        adapter = Path(ctx.ADAPTER_DIR) / "main"
        model.save_pretrained(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        write_json(
            adapter / "run_info.json",
            {
                "base_model": ctx.MODEL_NAME,
                "hp": hp,
                "seed": ctx.SEED,
                "four_bit": False,
                "n_train": len(rows),
                "dataset": dataset,
            },
            indent=2,
        )
        write_json(
            Path(ctx.OUT_DIR) / "main_adapter.json",
            {"adapter_path": str(adapter)},
            indent=2,
        )
        write_json(Path(ctx.OUT_DIR) / "training_metrics.json", info, indent=2)
        write_json(
            Path(ctx.OUT_DIR) / "training_log.json", trainer.state.log_history, indent=2
        )
        return adapter
    finally:
        del model, tokenizer, trainer
        ctx.close()
