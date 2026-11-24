from utils import write_json
from utils import read_json
from functools import partial


def experiment(
    ctx,
    key,
    model_name=None,
    size_tag="0.8B",
    train_rows=None,
    hp=None,
    seed=None,
    evals=("A_test",),
    shots=False,
    val_rows=None,
    four_bit=False,
    save_adapter=False,
):
    if model_name is None:
        model_name = ctx.MODEL_NAME
    if seed is None:
        seed = ctx.SEED
    "Train (optional) + evaluate one configuration. Cached: re-running returns the saved result."
    key = ctx.safe_key(ctx.PFX + key)
    rec = ctx.load_run(key)
    save_adapter = bool(train_rows) and (save_adapter or ctx.SAVE_ALL_ADAPTERS)
    adapter_path = f"{ctx.ADAPTER_DIR}/{key}"
    adapter_ok = not save_adapter or ctx.os.path.exists(
        f"{adapter_path}/adapter_config.json"
    )
    if rec is not None and all((e in rec["eval"] for e in evals)) and adapter_ok:
        print(f"[cached] {key}")
        if save_adapter:
            rec["adapter_path"] = adapter_path
        return rec
    if rec is not None and (not adapter_ok):
        print(
            f"[retrain] {key}: results are cached but the adapter was not saved yet - retraining to save it"
        )
    t0 = ctx.time.time()
    ctx.random.seed(seed)
    ctx.np.random.seed(seed)
    ctx.torch.manual_seed(seed)
    model = tokenizer = trainer = None
    try:
        model, tokenizer = ctx.load_model(model_name, four_bit=four_bit)
        rec = {
            "key": key,
            "model": model_name,
            "size": size_tag,
            "hp": hp,
            "seed": seed,
            "shots": shots,
            "four_bit": four_bit,
            "n_train": len(train_rows) if train_rows else 0,
            "eval": {},
            "train": None,
            "log": [],
            "total_params_M": sum((p.numel() for p in model.parameters())) / 1000000.0,
        }
        if train_rows:
            model, trainer, info = ctx.finetune(
                model, tokenizer, train_rows, val_rows, hp, seed, size_tag, tag=key
            )
            rec["train"] = info
            rec["log"] = [
                {k: float(v) for k, v in x.items() if isinstance(v, (int, float))}
                for x in trainer.state.log_history
            ]
            if save_adapter:
                model.save_pretrained(adapter_path)
                tokenizer.save_pretrained(adapter_path)
                write_json(
                    f"{adapter_path}/run_info.json",
                    {
                        "base_model": model_name,
                        "hp": hp,
                        "seed": seed,
                        "four_bit": four_bit,
                        "n_train": len(train_rows),
                        "key": key,
                    },
                    indent=2,
                )
                rec["adapter_path"] = adapter_path
                print(f"[saved adapter] {adapter_path}")
        for name in evals:
            rows, dbs, shot_rows = ctx.EVAL_SETS[name]
            preds, raws, secs = ctx.generate(
                model, rows, shots=shot_rows if shots else None
            )
            df = ctx.score(preds, raws, rows, dbs)
            rec["eval"][name] = {
                "summary": ctx.summarize(df),
                "secs_per_example": secs,
                "df": df,
            }
        rec["wall_min"] = (ctx.time.time() - t0) / 60
        ctx.save_run(rec)
        print(
            f"[done] {key}: "
            + " | ".join(
                (f"{n} EX={rec['eval'][n]['summary']['ex'][0]:.1f}" for n in evals)
            )
            + f" ({rec['wall_min']:.1f} min)"
        )
        return rec
    finally:
        del model, tokenizer, trainer
        ctx.free_gpu()


def is_oom(ctx, e):
    return "out of memory" in str(e).lower()


def ex_of(ctx, rec, name="A_test", metric="ex"):
    return rec["eval"][name]["summary"][metric][0]


def run_tuning(ctx):
    ctx.best_cfg_path = f"{ctx.RUN_DIR}/{ctx.PFX}best_config.json"
    ctx.tune_rows = []
    if ctx.RUN_TUNING:
        ctx.variants = [("baseline", "-", dict(ctx.BASE_HP))]
        for ctx.factor, ctx.vals in ctx.TUNE_GRID.items():
            for ctx.v in ctx.vals:
                if ctx.v != ctx.BASE_HP[ctx.factor]:
                    ctx.variants.append(
                        (ctx.factor, ctx.v, {**ctx.BASE_HP, ctx.factor: ctx.v})
                    )
        print(f"Stage 1: {len(ctx.variants)} configurations (+1 combined)")
        for ctx.factor, ctx.val, ctx.hp in ctx.variants:
            ctx.rec = ctx.experiment(
                f"tune_{ctx.hp_key(ctx.hp)}",
                train_rows=ctx.train_A,
                hp=ctx.hp,
                seed=ctx.SEED,
                evals=("A_tune",),
                val_rows=ctx.val_A,
            )
            ctx.s = ctx.rec["eval"]["A_tune"]["summary"]
            ctx.tune_rows.append(
                {
                    "factor": ctx.factor,
                    "value": ctx.val,
                    **ctx.hp,
                    "tune_EX": ctx.s["ex"][0],
                    "EX_lo": ctx.s["ex"][1],
                    "EX_hi": ctx.s["ex"][2],
                    "tune_EM": ctx.s["em"][0],
                    "tune_valid": ctx.s["valid"][0],
                    "val_loss": ctx.rec["train"]["final_val_loss"],
                    "train_min": ctx.rec["train"]["train_time_min"],
                    "trainable_M": ctx.rec["train"]["trainable_params_M"],
                    "peak_vram_gb": ctx.rec["train"]["peak_vram_gb"],
                }
            )
        ctx.tune_df = ctx.pd.DataFrame(ctx.tune_rows)

        def choose(factor):
            sub = ctx.tune_df[
                (ctx.tune_df.factor == factor) | (ctx.tune_df.factor == "baseline")
            ].copy()
            sub["v"] = [
                ctx.BASE_HP[factor] if f == "baseline" else v
                for f, v in zip(sub.factor, sub.value)
            ]
            best = sub.tune_EX.max()
            ties = sub[sub.tune_EX >= best - ctx.TUNE_TOL]
            if ctx.PREFER[factor] == "low":
                return min(ties.v)
            if ctx.BASE_HP[factor] in list(ties.v):
                return ctx.BASE_HP[factor]
            return ties.sort_values("tune_EX", ascending=False).v.iloc[0]

        ctx.choose = choose
        ctx.chosen = {f: ctx.choose(f) for f in ctx.TUNE_GRID}
        ctx.combined = {**ctx.BASE_HP, **ctx.chosen}
        ctx.rec_c = ctx.experiment(
            f"tune_{ctx.hp_key(ctx.combined)}",
            train_rows=ctx.train_A,
            hp=ctx.combined,
            seed=ctx.SEED,
            evals=("A_tune",),
            val_rows=ctx.val_A,
        )
        ctx.comb_ex = ctx.ex_of(ctx.rec_c, "A_tune")
        ctx.best_single = ctx.tune_df.sort_values("tune_EX", ascending=False).iloc[0]
        ctx.best_single_hp = {k: ctx.best_single[k] for k in ctx.BASE_HP}
        if ctx.comb_ex >= ctx.best_single.tune_EX - ctx.TUNE_TOL:
            ctx.BEST_HP, ctx.how = (ctx.combined, "combined per-factor choices")
        else:
            ctx.BEST_HP, ctx.how = (
                ctx.best_single_hp,
                "best single configuration (combination did not help)",
            )
        ctx.BEST_HP = {
            k: int(v)
            if isinstance(v, (ctx.np.integer,))
            else float(v)
            if isinstance(v, ctx.np.floating)
            else v
            for k, v in ctx.BEST_HP.items()
        }
        write_json(
            ctx.best_cfg_path,
            {
                "best_hp": ctx.BEST_HP,
                "how": ctx.how,
                "combined_tune_EX": ctx.comb_ex,
                "baseline_tune_EX": float(
                    ctx.tune_df.loc[ctx.tune_df.factor == "baseline", "tune_EX"].iloc[0]
                ),
            },
            indent=2,
        )
    elif ctx.os.path.exists(ctx.best_cfg_path):
        ctx.BEST_HP, ctx.how = (
            read_json(ctx.best_cfg_path)["best_hp"],
            "loaded from an earlier tuning run",
        )
    else:
        ctx.BEST_HP, ctx.how = (dict(ctx.BASE_HP), "default (tuning disabled)")
    print(f"\nSELECTED CONFIGURATION ({ctx.how}): {ctx.BEST_HP}")
    if ctx.tune_rows:
        ctx.factors = list(ctx.TUNE_GRID)
        ctx.ncol = min(3, len(ctx.factors))
        ctx.nrow = ctx.math.ceil(len(ctx.factors) / ctx.ncol)
        ctx.fig, ctx.axes = ctx.plt.subplots(
            ctx.nrow,
            ctx.ncol,
            figsize=(4.2 * ctx.ncol, 3.6 * ctx.nrow),
            sharey=True,
            squeeze=False,
        )
        ctx.ylo = max(0, ctx.tune_df.EX_lo.min() - 5)
        ctx.yhi = min(100, ctx.tune_df.EX_hi.max() + 6)
        ctx.base_ex = ctx.tune_df.loc[ctx.tune_df.factor == "baseline", "tune_EX"].iloc[
            0
        ]
        for ctx.ax, ctx.f in zip(ctx.axes.ravel(), ctx.factors):
            ctx.sub = ctx.tune_df[
                (ctx.tune_df.factor == ctx.f) | (ctx.tune_df.factor == "baseline")
            ].copy()
            ctx.sub["v"] = [
                ctx.BASE_HP[ctx.f] if x == "baseline" else v
                for x, v in zip(ctx.sub.factor, ctx.sub.value)
            ]
            ctx.order = (
                ctx.TUNE_GRID[ctx.f]
                if all((v in ctx.TUNE_GRID[ctx.f] for v in ctx.sub.v))
                else sorted(ctx.sub.v, key=str)
            )
            ctx.sub = ctx.sub.set_index("v").loc[
                [v for v in ctx.order if v in set(ctx.sub.v)]
            ]
            ctx.xs = ctx.np.arange(len(ctx.sub))
            ctx.ax.errorbar(
                ctx.xs,
                ctx.sub.tune_EX,
                yerr=[ctx.sub.tune_EX - ctx.sub.EX_lo, ctx.sub.EX_hi - ctx.sub.tune_EX],
                color=ctx.FT_C,
                marker="o",
                markersize=7,
                capsize=3,
                elinewidth=1,
                ecolor=ctx.INK2,
            )
            for ctx.x, (ctx.v, ctx.r) in zip(ctx.xs, ctx.sub.iterrows()):
                ctx.ax.annotate(
                    f"{ctx.r.tune_EX:.1f}",
                    (ctx.x, ctx.r.tune_EX),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=ctx.INK2,
                )
                if ctx.v == ctx.BEST_HP[ctx.f]:
                    ctx.ax.scatter(
                        [ctx.x],
                        [ctx.r.tune_EX],
                        s=220,
                        facecolors="none",
                        edgecolors=ctx.ACCENT,
                        linewidths=2,
                        zorder=5,
                    )
            ctx.ax.axhline(ctx.base_ex, color=ctx.GOLD_C, linestyle=":", linewidth=1)
            ctx.ax.set_xticks(ctx.xs)
            ctx.ax.set_xticklabels(
                [f"{v:g}" if isinstance(v, float) else str(v) for v in ctx.sub.index]
            )
            ctx.ax.set_xlabel(
                ctx.HP_LABELS[ctx.f] + ("  (default " + str(ctx.BASE_HP[ctx.f]) + ")")
            )
            ctx.ax.set_ylim(ctx.ylo, ctx.yhi)
            ctx.ax.grid(axis="x", visible=False)
        for ctx.ax in ctx.axes.ravel()[len(ctx.factors) :]:
            ctx.ax.axis("off")
        ctx.axes[0, 0].set_ylabel("tuning-set EX (%)")
        ctx.fig.suptitle(
            "Figure 7 - Hyperparameter study (tuning set; green ring = selected value, dotted = default config)",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=12,
        )
        ctx.savefig(
            ctx.fig, "fig07_hparam_study", "Hyperparameter study on the tuning set"
        )
        ctx.show = ctx.tune_df[
            [
                "factor",
                "value",
                "tune_EX",
                "EX_lo",
                "EX_hi",
                "tune_EM",
                "tune_valid",
                "val_loss",
                "trainable_M",
                "train_min",
                "peak_vram_gb",
            ]
        ].round(3)
        print("Table 2 - All Stage 1 runs (tuning set)")
        ctx.display(ctx.show)
        ctx.sens = {
            f: ctx.pd.concat(
                [
                    ctx.tune_df[ctx.tune_df.factor == f].tune_EX,
                    ctx.pd.Series([ctx.base_ex]),
                ]
            ).max()
            - ctx.pd.concat(
                [
                    ctx.tune_df[ctx.tune_df.factor == f].tune_EX,
                    ctx.pd.Series([ctx.base_ex]),
                ]
            ).min()
            for f in ctx.factors
        }
        ctx.sens_df = (
            ctx.pd.Series(ctx.sens, name="EX range across values (pp)")
            .sort_values(ascending=False)
            .round(1)
        )
        print(
            "Table 3 - Sensitivity: how much tuning-set EX changes across each hyperparameter's values"
        )
        ctx.display(ctx.sens_df.to_frame())
        ctx.chosen_tbl = ctx.pd.DataFrame(
            {"default": ctx.BASE_HP, "selected": ctx.BEST_HP}
        )
        print("Table 4 - Default vs selected configuration")
        ctx.display(ctx.chosen_tbl)
        ctx.tune_df.to_csv(f"{ctx.OUT_DIR}/hparam_study.csv", index=False)


def row_for(ctx, label, rec, name="A_test"):
    s = rec["eval"][name]["summary"]
    return {
        "Model": label,
        **{n: f"{s[m][0]:.1f} [{s[m][1]:.1f}, {s[m][2]:.1f}]" for m, n in ctx.METRICS},
    }


def run_comparison(ctx):
    ctx.EVALS = ("A_test", "B_test")
    ctx.R = {}
    ctx.R["zero"] = ctx.experiment("zero_0.8B", evals=ctx.EVALS)
    if ctx.RUN_FEWSHOT:
        ctx.R["few"] = ctx.experiment("fewshot_0.8B", evals=ctx.EVALS, shots=True)
    if ctx.RUN_DEFAULT_TEST and ctx.BEST_HP != ctx.BASE_HP:
        ctx.R["default"] = ctx.experiment(
            f"ftA_{ctx.hp_key(ctx.BASE_HP)}_s{ctx.SEEDS[0]}",
            train_rows=ctx.train_A,
            hp=ctx.BASE_HP,
            seed=ctx.SEEDS[0],
            evals=ctx.EVALS,
            val_rows=ctx.val_A,
        )
    ctx.R["seeds"] = [
        ctx.experiment(
            f"ftA_{ctx.hp_key(ctx.BEST_HP)}_s{s}",
            train_rows=ctx.train_A,
            hp=ctx.BEST_HP,
            seed=s,
            evals=ctx.EVALS,
            val_rows=ctx.val_A,
            save_adapter=s == ctx.SEEDS[0],
        )
        for s in ctx.SEEDS
    ]
    if "default" not in ctx.R:
        ctx.R["default"] = ctx.R["seeds"][0]
    ctx.main = ctx.R["seeds"][0]
    ctx.zero_df, ctx.ft_df = (
        ctx.R["zero"]["eval"]["A_test"]["df"],
        ctx.main["eval"]["A_test"]["df"],
    )
    ctx.few_df = ctx.R["few"]["eval"]["A_test"]["df"] if "few" in ctx.R else None
    ctx.log = ctx.pd.DataFrame(ctx.main["log"])
    ctx.tr_log = (
        ctx.log.dropna(subset=["loss"]) if "loss" in ctx.log else ctx.pd.DataFrame()
    )
    ctx.ev_log = (
        ctx.log.dropna(subset=["eval_loss"])
        if "eval_loss" in ctx.log
        else ctx.pd.DataFrame()
    )
    ctx.fig, (ctx.ax1, ctx.ax2) = ctx.plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]}
    )
    for ctx.s, ctx.rec in zip(ctx.SEEDS, ctx.R["seeds"]):
        ctx.lg = ctx.pd.DataFrame(ctx.rec["log"])
        if "loss" in ctx.lg:
            ctx.lg = ctx.lg.dropna(subset=["loss"])
            ctx.lab = (
                f"training loss (seed {ctx.s})"
                if ctx.rec is ctx.main
                else "other seeds"
                if ctx.s == ctx.SEEDS[1]
                else None
            )
            ctx.ax1.plot(
                ctx.lg["step"],
                ctx.lg["loss"],
                color=ctx.FT_C,
                alpha=1 if ctx.rec is ctx.main else 0.35,
                linewidth=1.5 if ctx.rec is ctx.main else 1,
                label=ctx.lab,
            )
    if len(ctx.ev_log):
        ctx.ax1.plot(
            ctx.ev_log["step"],
            ctx.ev_log["eval_loss"],
            color=ctx.ZERO_C,
            marker="o",
            markersize=5,
            label="validation loss",
        )
    ctx.ax1.set_xlabel("optimiser step")
    ctx.ax1.set_ylabel("cross-entropy loss (answer tokens)")
    ctx.ax1.legend()
    ctx.ax1.set_title("Figure 8a - Loss curves, tuned config (faint = other seeds)")
    if "learning_rate" in ctx.tr_log:
        ctx.ax2.plot(ctx.tr_log["step"], ctx.tr_log["learning_rate"], color=ctx.GOLD_C)
    ctx.ax2.set_xlabel("optimiser step")
    ctx.ax2.set_ylabel("learning rate")
    ctx.ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ctx.ax2.set_title("Figure 8b - Learning-rate schedule")
    ctx.savefig(
        ctx.fig,
        "fig08_training_curves",
        "Training/validation loss and learning-rate schedule",
    )
    ctx.row_for = partial(row_for, ctx)
    ctx.models = [("Zero-shot", ctx.R["zero"])] + (
        [("3-shot prompting", ctx.R["few"])] if "few" in ctx.R else []
    )
    if ctx.R["default"] is not ctx.main:
        ctx.models.append(("LoRA default config", ctx.R["default"]))
    ctx.models.append((f"LoRA tuned config (seed {ctx.SEEDS[0]})", ctx.main))
    ctx.t5 = ctx.pd.DataFrame([ctx.row_for(l, r) for l, r in ctx.models])
    ctx.seed_ex = {
        m: [r["eval"]["A_test"]["summary"][m][0] for r in ctx.R["seeds"]]
        for m, _ in ctx.METRICS
    }
    ctx.t5.loc[len(ctx.t5)] = {
        "Model": f"LoRA tuned, mean ± SD over {len(ctx.SEEDS)} seeds",
        **{
            n: f"{ctx.np.mean(ctx.seed_ex[m]):.1f} ± {ctx.np.std(ctx.seed_ex[m], ddof=1):.1f}"
            for m, n in ctx.METRICS
        },
    }
    ctx.t5["B-test EX"] = [f"{ctx.ex_of(r, 'B_test'):.1f}" for _, r in ctx.models] + [
        f"{ctx.np.mean([ctx.ex_of(r, 'B_test') for r in ctx.R['seeds']]):.1f} ± {ctx.np.std([ctx.ex_of(r, 'B_test') for r in ctx.R['seeds']], ddof=1):.1f}"
    ]
    ctx.t5["s / example"] = [
        f"{r['eval']['A_test']['secs_per_example']:.2f}" for _, r in ctx.models
    ] + [""]
    print(
        f"Table 5 - Test results on A-test (n = {len(ctx.test_A)}), % with 95% bootstrap CI; last column: B-test EX"
    )
    ctx.display(ctx.t5.set_index("Model"))
    ctx.tests = []
    for ctx.s, ctx.rec in zip(ctx.SEEDS, ctx.R["seeds"]):
        ctx.n01, ctx.n10, ctx.p = ctx.mcnemar(
            ctx.zero_df["correct"], ctx.rec["eval"]["A_test"]["df"]["correct"]
        )
        ctx.tests.append(
            {
                "comparison": f"tuned (seed {ctx.s}) vs zero-shot",
                "fixed": ctx.n01,
                "broken": ctx.n10,
                "p-value": ctx.p,
            }
        )
    if ctx.few_df is not None:
        ctx.n01, ctx.n10, ctx.p = ctx.mcnemar(
            ctx.few_df["correct"], ctx.ft_df["correct"]
        )
        ctx.tests.append(
            {
                "comparison": "tuned vs 3-shot",
                "fixed": ctx.n01,
                "broken": ctx.n10,
                "p-value": ctx.p,
            }
        )
    if ctx.R["default"] is not ctx.main:
        ctx.n01, ctx.n10, ctx.p = ctx.mcnemar(
            ctx.R["default"]["eval"]["A_test"]["df"]["correct"], ctx.ft_df["correct"]
        )
        ctx.tests.append(
            {
                "comparison": "tuned vs default LoRA config",
                "fixed": ctx.n01,
                "broken": ctx.n10,
                "p-value": ctx.p,
            }
        )
    ctx.tests = ctx.pd.DataFrame(ctx.tests)
    ctx.tests["significant (alpha=0.05)"] = ctx.tests["p-value"] < 0.05
    print(
        "Table 6 - McNemar exact tests on A-test (fixed = only the second model correct)"
    )
    ctx.display(
        ctx.tests.assign(**{"p-value": ctx.tests["p-value"].map(lambda p: f"{p:.2e}")})
    )
    ctx.p_mcnemar = ctx.tests.iloc[0]["p-value"]
    ctx.n01, ctx.n10 = (ctx.tests.iloc[0]["fixed"], ctx.tests.iloc[0]["broken"])
    ctx.colors = {
        "Zero-shot": ctx.ZERO_C,
        "3-shot prompting": ctx.FEW_C,
        "LoRA default config": ctx.DEF_C,
    }
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(11, 4.8))
    ctx.x = ctx.np.arange(len(ctx.METRICS))
    ctx.k = len(ctx.models)
    ctx.w = 0.8 / ctx.k
    for ctx.i, (ctx.lab, ctx.rec) in enumerate(ctx.models):
        ctx.s = ctx.rec["eval"]["A_test"]["summary"]
        ctx.vals = ctx.np.array([ctx.s[m][0] for m, _ in ctx.METRICS])
        ctx.err = ctx.np.array(
            [
                [ctx.s[m][0] - ctx.s[m][1], ctx.s[m][2] - ctx.s[m][0]]
                for m, _ in ctx.METRICS
            ]
        ).T
        ctx.bars = ctx.ax.bar(
            ctx.x + (ctx.i - (ctx.k - 1) / 2) * ctx.w,
            ctx.vals,
            ctx.w - 0.02,
            color=ctx.colors.get(ctx.lab, ctx.FT_C),
            label=ctx.lab,
            yerr=ctx.err,
            error_kw={"ecolor": ctx.INK2, "elinewidth": 1, "capsize": 2},
        )
        for ctx.b, ctx.v, ctx.hi in zip(ctx.bars, ctx.vals, ctx.vals + ctx.err[1]):
            ctx.ax.annotate(
                f"{ctx.v:.0f}",
                (ctx.b.get_x() + ctx.b.get_width() / 2, ctx.hi),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=ctx.INK2,
            )
    ctx.ax.set_xticks(ctx.x)
    ctx.ax.set_xticklabels([n for _, n in ctx.METRICS])
    ctx.ax.set_ylim(0, 118)
    ctx.ax.set_ylabel("% of A-test examples")
    ctx.ax.legend(loc="upper left", ncol=ctx.k, fontsize=9)
    ctx.ax.grid(axis="x", visible=False)
    ctx.ax.set_title(
        "Figure 9 - Zero-shot vs 3-shot vs LoRA on A-test (error bars = 95% bootstrap CI)"
    )
    ctx.savefig(ctx.fig, "fig09_main_results", "Main results on A-test with 95% CIs")
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(8, 4))
    ctx.show_m = [
        ("ex", "Execution Acc."),
        ("em", "Exact Match"),
        ("valid", "Valid SQL"),
        ("format_ok", "Format"),
    ]
    for ctx.j, (ctx.m, ctx.n) in enumerate(ctx.show_m):
        ctx.vals = ctx.seed_ex[ctx.m]
        ctx.ax.scatter(
            ctx.np.full(len(ctx.vals), ctx.j)
            + ctx.np.linspace(-0.08, 0.08, len(ctx.vals)),
            ctx.vals,
            color=ctx.FT_C,
            s=55,
            zorder=3,
            label="tuned LoRA, one seed" if ctx.j == 0 else None,
        )
        ctx.ax.hlines(
            ctx.np.mean(ctx.vals),
            ctx.j - 0.22,
            ctx.j + 0.22,
            color=ctx.INK,
            linewidth=2,
            label="mean over seeds" if ctx.j == 0 else None,
        )
        ctx.ax.scatter(
            [ctx.j + 0.32],
            [ctx.R["zero"]["eval"]["A_test"]["summary"][ctx.m][0]],
            marker="D",
            color=ctx.ZERO_C,
            s=45,
            label="zero-shot" if ctx.j == 0 else None,
        )
        ctx.ax.annotate(
            f"SD {ctx.np.std(ctx.vals, ddof=1):.1f}",
            (ctx.j, max(ctx.vals)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=ctx.INK2,
        )
    ctx.ax.set_xticks(range(len(ctx.show_m)))
    ctx.ax.set_xticklabels([n for _, n in ctx.show_m])
    ctx.ax.set_ylim(0, 108)
    ctx.ax.set_ylabel("% of A-test examples")
    ctx.ax.legend(loc="lower right", fontsize=8)
    ctx.ax.grid(axis="x", visible=False)
    ctx.ax.set_title(
        f"Figure 10 - Run-to-run variability over {len(ctx.SEEDS)} training seeds"
    )
    ctx.savefig(
        ctx.fig, "fig10_seed_variability", "Seed variability of the tuned configuration"
    )


def run_generalisation(ctx):
    ctx.grid = {"Zero-shot": ctx.R["zero"], f"Train A ({ctx.N_TRAIN})": ctx.main}
    if ctx.RUN_DATASETS:
        ctx.half = ctx.N_TRAIN // 2
        ctx.grid[f"Train B ({ctx.N_TRAIN})"] = ctx.experiment(
            f"ftB_{ctx.hp_key(ctx.BEST_HP)}_s{ctx.SEED}",
            train_rows=ctx.train_B,
            hp=ctx.BEST_HP,
            seed=ctx.SEED,
            evals=ctx.EVALS,
            val_rows=ctx.val_B,
            save_adapter=True,
        )
        ctx.mix = ctx.train_A[: ctx.half] + ctx.train_B[: ctx.half]
        ctx.random.Random(ctx.SEED).shuffle(ctx.mix)
        ctx.grid[f"Train A+B ({ctx.half}+{ctx.half})"] = ctx.experiment(
            f"ftAB_{ctx.hp_key(ctx.BEST_HP)}_s{ctx.SEED}",
            train_rows=ctx.mix,
            hp=ctx.BEST_HP,
            seed=ctx.SEED,
            evals=ctx.EVALS,
            val_rows=ctx.val_A[: ctx.N_VAL // 2] + ctx.val_B[: ctx.N_VAL // 2],
            save_adapter=True,
        )
    ctx.grid_df = ctx.pd.DataFrame(
        {
            k: {"A-test": ctx.ex_of(r, "A_test"), "B-test": ctx.ex_of(r, "B_test")}
            for k, r in ctx.grid.items()
        }
    ).T
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(6.5, 0.7 * len(ctx.grid_df) + 1.6))
    ctx.ax.imshow(ctx.grid_df.values, cmap=ctx.BLUES, vmin=0, vmax=100, aspect="auto")
    for ctx.i in range(ctx.grid_df.shape[0]):
        for ctx.j in range(ctx.grid_df.shape[1]):
            ctx.v = ctx.grid_df.values[ctx.i, ctx.j]
            ctx.ax.text(
                ctx.j,
                ctx.i,
                f"{ctx.v:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if ctx.v > 60 else ctx.INK,
            )
    ctx.ax.set_xticks(range(ctx.grid_df.shape[1]))
    ctx.ax.set_xticklabels(ctx.grid_df.columns)
    ctx.ax.set_yticks(range(ctx.grid_df.shape[0]))
    ctx.ax.set_yticklabels(ctx.grid_df.index)
    ctx.ax.grid(False)
    ctx.ax.set_title("Figure 16 - Execution accuracy (%): training data x test set")
    ctx.savefig(ctx.fig, "fig16_cross_dataset", "Cross-dataset execution accuracy")
    print("Table 8 - Cross-dataset execution accuracy (%)")
    ctx.display(ctx.grid_df.round(1))


def run_scaling(ctx):
    ctx.scale = [("0.8B", ctx.R["zero"], ctx.main)]
    if ctx.RUN_SCALING:
        ctx.todo = [(size, name, False) for size, name in ctx.SCALING_MODELS.items()]
        if ctx.RUN_4B_4BIT:
            ctx.todo.append(("4B-4bit", ctx.MODEL_4B, True))
        for ctx.size, ctx.name, ctx.fb in ctx.todo:
            try:
                ctx.z = ctx.experiment(
                    f"zero_{ctx.size}",
                    model_name=ctx.name,
                    size_tag=ctx.size,
                    evals=ctx.EVALS,
                    four_bit=ctx.fb,
                )
                ctx.f = ctx.experiment(
                    f"ftA_{ctx.size}_{ctx.hp_key(ctx.BEST_HP)}_s{ctx.SEED}",
                    model_name=ctx.name,
                    size_tag=ctx.size,
                    train_rows=ctx.train_A,
                    hp=ctx.BEST_HP,
                    seed=ctx.SEED,
                    evals=ctx.EVALS,
                    val_rows=ctx.val_A,
                    four_bit=ctx.fb,
                    save_adapter=True,
                )
                ctx.scale.append((ctx.size, ctx.z, ctx.f))
            except RuntimeError as e:
                if not ctx.is_oom(e):
                    raise
                print(
                    f"SKIPPED {ctx.size}: out of GPU memory on this hardware ({str(e)[:90]}...)"
                )
    ctx.scale_df = ctx.pd.DataFrame(
        [
            {
                "model": f"Qwen3.5-{s}",
                "params (M)": f["total_params_M"]
                if f["train"] is None
                else f["train"]["total_params_M"],
                "trainable (M)": f["train"]["trainable_params_M"],
                "zero-shot EX A": ctx.ex_of(z, "A_test"),
                "fine-tuned EX A": ctx.ex_of(f, "A_test"),
                "zero-shot EX B": ctx.ex_of(z, "B_test"),
                "fine-tuned EX B": ctx.ex_of(f, "B_test"),
                "train min": f["train"]["train_time_min"],
                "peak VRAM (GB)": f["train"]["peak_vram_gb"],
                "s / example (ft)": f["eval"]["A_test"]["secs_per_example"],
            }
            for s, z, f in ctx.scale
        ]
    ).set_index("model")
    print("Table 9 - Model scaling")
    ctx.display(ctx.scale_df.round(2))
    ctx.fig, ctx.axes = ctx.plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ctx.xs = ctx.np.arange(len(ctx.scale))
    for ctx.ax, ctx.col, ctx.title in zip(
        ctx.axes, ["A", "B"], ["A-test (in-distribution)", "B-test (other dataset)"]
    ):
        for ctx.kind, ctx.c, ctx.mk in [
            ("zero-shot", ctx.ZERO_C, "D"),
            ("fine-tuned", ctx.FT_C, "o"),
        ]:
            ctx.v = ctx.scale_df[f"{ctx.kind} EX {ctx.col}"].values
            ctx.ax.plot(
                ctx.xs, ctx.v, color=ctx.c, marker=ctx.mk, markersize=7, label=ctx.kind
            )
            for ctx.xi, ctx.vi in zip(ctx.xs, ctx.v):
                ctx.ax.annotate(
                    f"{ctx.vi:.1f}",
                    (ctx.xi, ctx.vi),
                    xytext=(0, 8 if ctx.kind == "fine-tuned" else -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=ctx.c,
                )
        ctx.ax.set_xticks(ctx.xs)
        ctx.ax.set_xticklabels([f"{s}" for s, _, _ in ctx.scale])
        ctx.ax.set_xlabel("model size (parameters)")
        ctx.ax.set_ylim(0, 105)
        ctx.ax.set_title(ctx.title)
        ctx.ax.grid(axis="x", visible=False)
    ctx.axes[0].set_ylabel("execution accuracy (%)")
    ctx.axes[0].legend(loc="lower right")
    ctx.fig.suptitle(
        "Figure 17 - Zero-shot vs fine-tuned across model sizes",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    ctx.savefig(ctx.fig, "fig17_scaling", "Model scaling")


def run_ablation(ctx):
    ctx.abl = [
        {
            "train_examples": 0,
            **{
                m: ctx.R["zero"]["eval"]["A_test"]["summary"][m][0]
                for m, _ in ctx.METRICS
            },
        },
        {
            "train_examples": ctx.N_TRAIN,
            **{m: ctx.main["eval"]["A_test"]["summary"][m][0] for m, _ in ctx.METRICS},
        },
    ]
    if ctx.RUN_SIZE_ABLATION:
        for ctx.n in ctx.ABLATION_SIZES:
            ctx.rec = ctx.experiment(
                f"ftA_n{ctx.n}_{ctx.hp_key(ctx.BEST_HP)}_s{ctx.SEED}",
                train_rows=ctx.train_A[: ctx.n],
                hp=ctx.BEST_HP,
                seed=ctx.SEED,
                evals=("A_test",),
            )
            ctx.abl.append(
                {
                    "train_examples": ctx.n,
                    **{
                        m: ctx.rec["eval"]["A_test"]["summary"][m][0]
                        for m, _ in ctx.METRICS
                    },
                }
            )
    ctx.abl_df = (
        ctx.pd.DataFrame(ctx.abl).sort_values("train_examples").reset_index(drop=True)
    )
    ctx.display(
        ctx.abl_df.rename(columns=dict(ctx.METRICS))
        .round(1)
        .set_index("train_examples")
    )
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(8.5, 4.3))
    ctx.xs = ctx.np.arange(len(ctx.abl_df))
    for ctx.m, ctx.name, ctx.col, ctx.mk in [
        ("ex", "Execution Accuracy", ctx.VIOLET, "o"),
        ("em", "Exact Match", ctx.ACCENT, "s"),
        ("valid", "Valid SQL", ctx.GOLD_C, "^"),
    ]:
        ctx.ax.plot(
            ctx.xs,
            ctx.abl_df[ctx.m],
            marker=ctx.mk,
            markersize=7,
            color=ctx.col,
            label=ctx.name,
        )
    for ctx.xi, ctx.v in zip(ctx.xs, ctx.abl_df["ex"]):
        ctx.ax.annotate(
            f"{ctx.v:.1f}",
            (ctx.xi, ctx.v),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=ctx.VIOLET,
        )
    ctx.ax.set_xticks(ctx.xs)
    ctx.ax.set_xticklabels(
        ["0\n(zero-shot)" if n == 0 else f"{n:,}" for n in ctx.abl_df.train_examples]
    )
    ctx.ax.set_xlabel("training examples")
    ctx.ax.set_ylabel("% of A-test examples")
    ctx.ax.set_ylim(0, 105)
    ctx.ax.legend(loc="lower right")
    ctx.ax.set_xlim(-0.3, len(ctx.xs) - 0.5)
    ctx.ax.set_title("Figure 18 - Learning curve: test accuracy vs training-set size")
    ctx.savefig(ctx.fig, "fig18_size_ablation", "Accuracy vs training-set size")
