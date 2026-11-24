from functools import partial
import data
from pathlib import Path
from utils import read_json, write_json


def savefig(ctx, fig, name, caption):
    fig.tight_layout()
    fig.savefig(f"{ctx.FIG_DIR}/{name}.png", bbox_inches="tight")
    ctx.FIGS.append((name, caption))
    ctx.plt.close(fig)


def label_bars(ctx, ax, bars, fmt="{:.1f}", horizontal=False):
    for b in bars:
        if horizontal:
            w = b.get_width()
            ax.annotate(
                fmt.format(w),
                (w, b.get_y() + b.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=ctx.INK2,
            )
        else:
            h = b.get_height()
            ax.annotate(
                fmt.format(h),
                (b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=ctx.INK2,
            )


def initialize_plotting(ctx):
    ctx.ZERO_C, ctx.FEW_C, ctx.DEF_C, ctx.FT_C = (
        "#eb6834",
        "#1baf7a",
        "#86b6ef",
        "#2a78d6",
    )
    ctx.BASE_C, ctx.GOLD_C, ctx.ACCENT, ctx.VIOLET = (
        ctx.ZERO_C,
        "#6b6a66",
        "#1baf7a",
        "#6250d6",
    )
    ctx.A_C, ctx.B_C = ("#2a78d6", "#6250d6")
    ctx.INK, ctx.INK2, ctx.GRID = ("#0b0b0b", "#52514e", "#e4e3df")
    ctx.BLUES = ctx.LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"]
    )
    ctx.plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": ctx.GRID,
            "axes.labelcolor": ctx.INK2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": ctx.INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": ctx.GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": ctx.INK2,
            "ytick.color": ctx.INK2,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 2,
        }
    )
    ctx.FIGS = []
    ctx.savefig = partial(savefig, ctx)
    ctx.label_bars = partial(label_bars, ctx)


def describe(ctx, df):
    d = (
        df[["question_words", "sql_tokens", "n_tables", "n_columns"]]
        .agg(["mean", "median", lambda x: x.quantile(0.9), "max"])
        .T
    )
    d.columns = ["mean", "median", "90th pct", "max"]
    return d


def leakage(ctx, train, test, name):
    tq = set((ctx.norm_q(r["question"]) for r in train))
    tp = set(((ctx.norm_q(r["question"]), r["context"]) for r in train))
    lq = sum((ctx.norm_q(r["question"]) in tq for r in test))
    lp = sum(((ctx.norm_q(r["question"]), r["context"]) in tp for r in test))
    return {
        "dataset": name,
        "test questions also in train": lq,
        "identical (question, schema) pairs": lp,
        "test size": len(test),
    }


def run_exploration(ctx):
    data.initialize_features(ctx)
    ctx.t0 = ctx.time.time()
    ctx.full_A = ctx.add_eda_columns(ctx.full_A)
    ctx.eda_B = ctx.add_eda_columns(ctx.pd.DataFrame(ctx.pool_B))
    ctx.eda_B["split"] = [
        "train"
        if i < ctx.N_TRAIN
        else "validation"
        if i < ctx.N_TRAIN + ctx.N_VAL
        else "unused pool"
        for i in range(len(ctx.eda_B))
    ]
    ctx.eda_Bt = ctx.add_eda_columns(ctx.pd.DataFrame(ctx.test_B))
    ctx.eda_Bt["split"] = "test"
    print(
        f"Analysed {len(ctx.full_A):,} A + {len(ctx.eda_B) + len(ctx.eda_Bt):,} B examples in {ctx.time.time() - ctx.t0:.1f}s"
    )
    ctx.describe = partial(describe, ctx)
    ctx.stats = ctx.pd.concat(
        {"A (full)": ctx.describe(ctx.full_A), "B (pool)": ctx.describe(ctx.eda_B)},
        axis=1,
    ).round(1)
    ctx.stats.index = [
        "Question length (words)",
        "SQL length (tokens)",
        "Tables per schema",
        "Columns per schema",
    ]
    print("Table 1 - Descriptive statistics: dataset A vs dataset B")
    ctx.display(ctx.stats)
    for ctx.name, ctx.df in [("A", ctx.full_A), ("B", ctx.eda_B)]:
        ctx.dq = ctx.df["question"].str.lower().str.strip().duplicated().sum()
        print(
            f"{ctx.name}: duplicate questions {ctx.dq:,} ({100 * ctx.dq / len(ctx.df):.2f}%) | contexts with sample rows {100 * ctx.df['has_sample_rows'].mean():.0f}%"
        )
    ctx.fig, ctx.axes = ctx.plt.subplots(2, 2, figsize=(11, 7))
    ctx.panels = [
        ("question_words", "Question length (words)"),
        ("sql_tokens", "Gold SQL length (tokens)"),
        ("n_tables", "Tables per schema"),
        ("n_columns", "Columns per schema"),
    ]
    for ctx.ax, (ctx.col, ctx.title) in zip(ctx.axes.ravel(), ctx.panels):
        ctx.hi = int(
            max(
                ctx.np.percentile(ctx.full_A[ctx.col], 99.5),
                ctx.np.percentile(ctx.eda_B[ctx.col], 99.5),
            )
        )
        ctx.lo = int(min(ctx.full_A[ctx.col].min(), ctx.eda_B[ctx.col].min()))
        ctx.bins = (
            ctx.np.arange(ctx.lo, ctx.hi + 2) - 0.5
            if ctx.hi - ctx.lo < 40
            else ctx.np.linspace(ctx.lo, ctx.hi, 40)
        )
        for ctx.df, ctx.c, ctx.lab, ctx.kw in [
            (ctx.full_A, ctx.A_C, "A: sql-create-context", {"alpha": 0.55}),
            (
                ctx.eda_B,
                ctx.B_C,
                "B: Gretel synthetic",
                {"histtype": "step", "linewidth": 2},
            ),
        ]:
            ctx.w = ctx.np.full(len(ctx.df), 100 / len(ctx.df))
            ctx.ax.hist(
                ctx.df[ctx.col].clip(upper=ctx.hi),
                bins=ctx.bins,
                weights=ctx.w,
                color=ctx.c,
                label=ctx.lab,
                **ctx.kw,
            )
        ctx.ax.set_title(ctx.title)
        ctx.ax.set_ylabel("% of examples")
        ctx.ax.xaxis.set_major_locator(ctx.plt.MaxNLocator(integer=True))
    ctx.axes[0, 0].legend(loc="upper right")
    ctx.fig.suptitle(
        "Figure 1 - Length and schema-size distributions, dataset A vs B",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    ctx.savefig(
        ctx.fig, "fig01_distributions", "Length and schema-size distributions, A vs B"
    )
    ctx.feat_cols = [f"has_{k}" for k in ctx.FEATURES]
    ctx.prev = ctx.pd.DataFrame(
        {
            "A": ctx.full_A[ctx.feat_cols].mean() * 100,
            "B": ctx.eda_B[ctx.feat_cols].mean() * 100,
        }
    )
    ctx.prev.index = [c[4:] for c in ctx.prev.index]
    ctx.prev = ctx.prev[(ctx.prev > 0).any(axis=1)].sort_values("A")
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(9, 0.5 * len(ctx.prev) + 1.5))
    ctx.y = ctx.np.arange(len(ctx.prev))
    ctx.h = 0.38
    ctx.label_bars(
        ctx.ax,
        ctx.ax.barh(
            ctx.y + ctx.h / 2,
            ctx.prev["A"],
            ctx.h - 0.03,
            color=ctx.A_C,
            label="A: sql-create-context",
        ),
        "{:.1f}%",
        True,
    )
    ctx.label_bars(
        ctx.ax,
        ctx.ax.barh(
            ctx.y - ctx.h / 2,
            ctx.prev["B"],
            ctx.h - 0.03,
            color=ctx.B_C,
            label="B: Gretel synthetic",
        ),
        "{:.1f}%",
        True,
    )
    ctx.ax.set_yticks(ctx.y)
    ctx.ax.set_yticklabels(ctx.prev.index)
    ctx.ax.set_xlim(0, 112)
    ctx.ax.grid(axis="y", visible=False)
    ctx.ax.set_xlabel("% of queries using the feature")
    ctx.ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ctx.ax.set_title("Figure 2 - SQL feature prevalence, dataset A vs B")
    ctx.savefig(ctx.fig, "fig02_sql_features", "SQL feature prevalence, A vs B")
    ctx.diff_rows = {
        "A full": ctx.full_A["difficulty"],
        "A train": ctx.full_A.loc[ctx.full_A.split == "train", "difficulty"],
        "A test": ctx.full_A.loc[ctx.full_A.split == "test", "difficulty"],
        "B train": ctx.eda_B.loc[ctx.eda_B.split == "train", "difficulty"],
        "B test": ctx.eda_Bt["difficulty"],
    }
    ctx.diff_tab = ctx.pd.DataFrame(
        {
            k: v.value_counts(normalize=True).reindex(ctx.DIFF_ORDER).fillna(0) * 100
            for k, v in ctx.diff_rows.items()
        }
    ).T
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(9.5, 4.3))
    ctx.w = 0.16
    ctx.colors = ["#b7d3f6", "#5598e7", "#184f95", "#b9b2f0", ctx.B_C]
    for ctx.i, (ctx.g, ctx.c) in enumerate(zip(ctx.diff_tab.index, ctx.colors)):
        ctx.label_bars(
            ctx.ax,
            ctx.ax.bar(
                ctx.np.arange(3) + (ctx.i - 2) * ctx.w,
                ctx.diff_tab.loc[ctx.g],
                ctx.w - 0.02,
                color=ctx.c,
                label=ctx.g,
            ),
            "{:.0f}",
        )
    ctx.ax.set_xticks(range(3))
    ctx.ax.set_xticklabels(ctx.DIFF_ORDER)
    ctx.ax.set_ylabel("% of examples")
    ctx.ax.set_ylim(0, ctx.diff_tab.values.max() * 1.2)
    ctx.ax.legend(ncol=5, loc="upper right", fontsize=8)
    ctx.ax.grid(axis="x", visible=False)
    ctx.ax.set_title("Figure 3 - Query difficulty by dataset and split")
    ctx.savefig(
        ctx.fig,
        "fig03_difficulty_splits",
        "Difficulty distribution across datasets and splits",
    )
    ctx.counts = (
        ctx.pd.crosstab(ctx.full_A["split"], ctx.full_A["difficulty"])
        .reindex(index=["train", "test"], columns=ctx.DIFF_ORDER)
        .fillna(0)
    )
    ctx.counts = ctx.counts.loc[:, ctx.counts.sum() > 0]
    ctx.chi2, ctx.p_split, *ctx._ = ctx.chi2_contingency(ctx.counts.values)
    print(
        f"Chi-square, A train vs A test difficulty: chi2={ctx.chi2:.2f}, p={ctx.p_split:.3f} -> {('no significant difference (representative split)' if ctx.p_split > 0.05 else 'distributions differ')}"
    )
    ctx.top = ctx.full_A["skeleton"].value_counts().head(10)
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(10, 4.8))
    ctx.labels = [s if len(s) < 70 else s[:67] + "..." for s in ctx.top.index][::-1]
    ctx.label_bars(
        ctx.ax,
        ctx.ax.barh(
            ctx.labels,
            (100 * ctx.top.values / len(ctx.full_A))[::-1],
            color=ctx.A_C,
            height=0.6,
        ),
        "{:.1f}%",
        True,
    )
    ctx.ax.set_xlabel("% of all queries in A")
    ctx.ax.grid(axis="y", visible=False)
    ctx.ax.tick_params(axis="y", labelsize=8)
    for ctx.t in ctx.ax.get_yticklabels():
        ctx.t.set_fontfamily("monospace")
    ctx.ax.set_title("Figure 4 - Ten most common query templates in dataset A")
    ctx.savefig(ctx.fig, "fig04_templates", "Most common query templates in A")
    print(
        f"Distinct templates - A: {ctx.full_A['skeleton'].nunique():,} (top 10 cover {100 * ctx.top.sum() / len(ctx.full_A):.1f}%) | B pool: {ctx.eda_B['skeleton'].nunique():,} of {len(ctx.eda_B):,} queries"
    )
    ctx.fig, (ctx.ax1, ctx.ax2) = ctx.plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    ctx.cx = ctx.eda_B["complexity"].value_counts(normalize=True).sort_values() * 100
    ctx.label_bars(
        ctx.ax1,
        ctx.ax1.barh(ctx.cx.index, ctx.cx.values, color=ctx.B_C, height=0.6),
        "{:.0f}%",
        True,
    )
    ctx.ax1.set_xlabel("% of B pool")
    ctx.ax1.grid(axis="y", visible=False)
    ctx.ax1.set_xlim(0, ctx.cx.max() * 1.2)
    ctx.ax1.set_title("Figure 5a - B: Gretel complexity labels")
    ctx.dm = ctx.eda_B["domain"].value_counts().head(10).sort_values()
    ctx.label_bars(
        ctx.ax2,
        ctx.ax2.barh(ctx.dm.index, ctx.dm.values, color=ctx.B_C, height=0.6),
        "{:.0f}",
        True,
    )
    ctx.ax2.set_xlabel("examples")
    ctx.ax2.grid(axis="y", visible=False)
    ctx.ax2.tick_params(axis="y", labelsize=8)
    ctx.ax2.set_title(
        f"Figure 5b - B: top {len(ctx.dm)} of {ctx.eda_B['domain'].nunique()} domains"
    )
    ctx.savefig(
        ctx.fig, "fig05_gretel_profile", "Dataset B complexity labels and domains"
    )
    ctx.norm_q = lambda s: ctx.re.sub("\\s+", " ", s.lower().strip())
    ctx.leakage = partial(leakage, ctx)
    ctx.display(
        ctx.pd.DataFrame(
            [
                ctx.leakage(ctx.train_A + ctx.val_A + ctx.tune_A, ctx.test_A, "A"),
                ctx.leakage(ctx.pool_B, ctx.test_B, "B"),
                ctx.leakage(ctx.train_A, ctx.tune_A, "A (train vs tuning set)"),
            ]
        )
    )


def run_errors(ctx):
    ctx.comp = (
        [("Zero-shot", ctx.zero_df, ctx.ZERO_C)]
        + ([("3-shot", ctx.few_df, ctx.FEW_C)] if ctx.few_df is not None else [])
        + [("Tuned LoRA", ctx.ft_df, ctx.FT_C)]
    )
    ctx.diff_n = (
        ctx.ft_df["difficulty"]
        .value_counts()
        .reindex(ctx.DIFF_ORDER)
        .fillna(0)
        .astype(int)
    )
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(8.5, 4.3))
    ctx.k = len(ctx.comp)
    ctx.w = 0.8 / ctx.k
    for ctx.i, (ctx.lab, ctx.df, ctx.c) in enumerate(ctx.comp):
        ctx.acc = (
            ctx.df.groupby("difficulty")["correct"]
            .mean()
            .reindex(ctx.DIFF_ORDER)
            .fillna(0)
            * 100
        )
        ctx.label_bars(
            ctx.ax,
            ctx.ax.bar(
                ctx.np.arange(3) + (ctx.i - (ctx.k - 1) / 2) * ctx.w,
                ctx.acc,
                ctx.w - 0.02,
                color=ctx.c,
                label=ctx.lab,
            ),
        )
    ctx.ax.set_xticks(range(3))
    ctx.ax.set_xticklabels([f"{d}\n(n={ctx.diff_n[d]})" for d in ctx.DIFF_ORDER])
    ctx.ax.set_ylim(0, 112)
    ctx.ax.set_ylabel("accuracy (%)")
    ctx.ax.legend(ncol=ctx.k, loc="upper right")
    ctx.ax.grid(axis="x", visible=False)
    ctx.ax.set_title("Figure 11 - Accuracy by query difficulty (A-test)")
    ctx.savefig(ctx.fig, "fig11_by_difficulty", "Accuracy by query difficulty")
    ctx.b_d = (
        ctx.zero_df.groupby("difficulty")["correct"].mean().reindex(ctx.DIFF_ORDER)
        * 100
    )
    ctx.f_d = (
        ctx.ft_df.groupby("difficulty")["correct"].mean().reindex(ctx.DIFF_ORDER) * 100
    )
    ctx.feat_rows = []
    for ctx.kf in ctx.FEATURES:
        ctx.m = ctx.ft_df[f"has_{ctx.kf}"].astype(bool)
        if ctx.m.sum() >= 5:
            ctx.feat_rows.append(
                {
                    "feature": ctx.kf,
                    "n": int(ctx.m.sum()),
                    "base": ctx.zero_df.loc[ctx.m, "correct"].mean() * 100,
                    "fine-tuned": ctx.ft_df.loc[ctx.m, "correct"].mean() * 100,
                }
            )
    ctx.by_feat = ctx.pd.DataFrame(ctx.feat_rows).sort_values("fine-tuned")
    ctx.by_feat["Δ (pp)"] = ctx.by_feat["fine-tuned"] - ctx.by_feat["base"]
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(9, 0.55 * len(ctx.by_feat) + 1.6))
    ctx.y = ctx.np.arange(len(ctx.by_feat))
    ctx.h = 0.38
    ctx.label_bars(
        ctx.ax,
        ctx.ax.barh(
            ctx.y + ctx.h / 2,
            ctx.by_feat["base"],
            ctx.h - 0.03,
            color=ctx.ZERO_C,
            label="Zero-shot",
        ),
        horizontal=True,
    )
    ctx.label_bars(
        ctx.ax,
        ctx.ax.barh(
            ctx.y - ctx.h / 2,
            ctx.by_feat["fine-tuned"],
            ctx.h - 0.03,
            color=ctx.FT_C,
            label="Tuned LoRA",
        ),
        horizontal=True,
    )
    ctx.ax.set_yticks(ctx.y)
    ctx.ax.set_yticklabels(
        [f"{f} (n={n})" for f, n in zip(ctx.by_feat.feature, ctx.by_feat.n)]
    )
    ctx.ax.set_xlim(0, 112)
    ctx.ax.set_xlabel("accuracy (%)")
    ctx.ax.grid(axis="y", visible=False)
    ctx.ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ctx.ax.set_title(
        "Figure 12 - Accuracy on A-test queries using each SQL feature (n >= 5)"
    )
    ctx.savefig(ctx.fig, "fig12_by_feature", "Accuracy by SQL feature")
    ctx.display(ctx.by_feat.round(1).set_index("feature"))
    ctx.b_ok, ctx.f_ok = (
        ctx.zero_df["correct"].values.astype(bool),
        ctx.ft_df["correct"].values.astype(bool),
    )
    ctx.cont = ctx.pd.crosstab(
        ctx.pd.Series(ctx.b_ok, name="Zero-shot"), ctx.pd.Series(ctx.f_ok, name="Tuned")
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(5.2, 4.4))
    ctx.ax.imshow(ctx.cont.values, cmap=ctx.BLUES)
    ctx.ax.set_xticks([0, 1])
    ctx.ax.set_xticklabels(["Tuned wrong", "Tuned correct"])
    ctx.ax.set_yticks([0, 1])
    ctx.ax.set_yticklabels(["Zero-shot wrong", "Zero-shot correct"])
    ctx.ax.grid(False)
    ctx.vmax = ctx.cont.values.max()
    for ctx.i in range(2):
        for ctx.j in range(2):
            ctx.v = ctx.cont.values[ctx.i, ctx.j]
            ctx.ax.text(
                ctx.j,
                ctx.i,
                f"{ctx.v}\n({100 * ctx.v / ctx.cont.values.sum():.0f}%)",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if ctx.v > ctx.vmax * 0.55 else ctx.INK,
                fontweight="bold",
            )
    ctx.ax.set_title("Figure 13 - Paired outcomes (A-test)")
    ctx.savefig(ctx.fig, "fig13_contingency", "Paired zero-shot vs tuned outcomes")
    ctx.err_cols = [
        (lab, df["error_type"].value_counts().reindex(ctx.ERROR_ORDER).fillna(0), c)
        for lab, df, c in ctx.comp
    ]
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(9, 4.6))
    ctx.y = ctx.np.arange(len(ctx.ERROR_ORDER))[::-1]
    ctx.k = len(ctx.err_cols)
    ctx.h = 0.8 / ctx.k
    for ctx.i, (ctx.lab, ctx.e, ctx.c) in enumerate(ctx.err_cols):
        ctx.label_bars(
            ctx.ax,
            ctx.ax.barh(
                ctx.y + ((ctx.k - 1) / 2 - ctx.i) * ctx.h,
                ctx.e.values,
                ctx.h - 0.03,
                color=ctx.c,
                label=ctx.lab,
            ),
            "{:.0f}",
            True,
        )
    ctx.ax.set_yticks(ctx.y)
    ctx.ax.set_yticklabels(ctx.ERROR_ORDER)
    ctx.ax.set_xlabel("number of wrong A-test answers")
    ctx.ax.set_xlim(0, max((e.max() for _, e, _ in ctx.err_cols)) * 1.15 + 1)
    ctx.ax.legend(ncol=ctx.k, loc="lower right")
    ctx.ax.grid(axis="y", visible=False)
    ctx.ax.set_title(
        "Figure 14 - Error taxonomy (first matching category per wrong answer)"
    )
    ctx.savefig(ctx.fig, "fig14_error_taxonomy", "Error taxonomy")
    ctx.err_f = ctx.err_cols[-1][1]
    ctx.fig, ctx.ax = ctx.plt.subplots(figsize=(8, 3.6 + 0.4 * len(ctx.comp)))
    ctx.golds_A = [r["answer"] for r in ctx.test_A]
    ctx.data = [df["pred_tokens"] for _, df, _ in ctx.comp] + [
        [len(ctx.sql_tokens(g)) for g in ctx.golds_A]
    ]
    ctx.bp = ctx.ax.boxplot(
        ctx.data,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": ctx.INK, "linewidth": 1.5},
        flierprops={"markersize": 3, "markeredgecolor": ctx.INK2},
    )
    for ctx.patch, ctx.c in zip(
        ctx.bp["boxes"], [c for _, _, c in ctx.comp] + [ctx.GOLD_C]
    ):
        ctx.patch.set_facecolor(ctx.c)
        ctx.patch.set_alpha(0.85)
        ctx.patch.set_edgecolor("white")
    ctx.ax.set_yticklabels([lab for lab, _, _ in ctx.comp] + ["Gold"])
    ctx.ax.set_xlabel("SQL tokens in the answer")
    ctx.ax.grid(axis="y", visible=False)
    ctx.ax.set_title("Figure 15 - Answer length: predictions vs gold (A-test)")
    ctx.savefig(ctx.fig, "fig15_output_length", "Answer length distributions")
    ctx.pd.set_option("display.max_colwidth", 120)
    ctx.cmp = ctx.pd.DataFrame(
        {
            "question": ctx.zero_df.question,
            "gold": ctx.zero_df.gold,
            "zero_pred": ctx.zero_df.pred,
            "ft_pred": ctx.ft_df.pred,
            "zero_ok": ctx.b_ok,
            "ft_ok": ctx.f_ok,
            "ft_error": ctx.ft_df.error_type,
            "difficulty": ctx.ft_df.difficulty,
        }
    )
    print("Table 7a - Fixed by fine-tuning")
    ctx.display(
        ctx.cmp[~ctx.cmp.zero_ok & ctx.cmp.ft_ok][
            ["question", "gold", "zero_pred", "ft_pred"]
        ].head(5)
    )
    print("Table 7b - Still wrong after fine-tuning")
    ctx.display(
        ctx.cmp[~ctx.cmp.ft_ok][
            ["question", "gold", "ft_pred", "ft_error", "difficulty"]
        ].head(6)
    )
    print(
        "Table 7c - Correct by execution but not by exact match (equivalent SQL written differently)"
    )
    ctx.display(
        ctx.cmp[(ctx.ft_df.ex == 1) & ~ctx.ft_df.em.astype(bool)][
            ["question", "gold", "ft_pred"]
        ].head(4)
    )


def pp(ctx, x):
    return f"{x:+.1f} pp"


def run_discussion(ctx):
    ctx.pp = partial(pp, ctx)
    ctx.Z, ctx.F = (
        ctx.R["zero"]["eval"]["A_test"]["summary"],
        ctx.main["eval"]["A_test"]["summary"],
    )
    ctx.ex_mean, ctx.ex_sd = (
        ctx.np.mean(ctx.seed_ex["ex"]),
        ctx.np.std(ctx.seed_ex["ex"], ddof=1),
    )
    ctx.paras = []
    ctx.paras.append(
        f"**Main finding.** LoRA fine-tuning of Qwen3.5-0.8B on {ctx.N_TRAIN:,} examples raised execution accuracy on A-test from **{ctx.Z['ex'][0]:.1f}%** (zero-shot, 95% CI {ctx.Z['ex'][1]:.1f}–{ctx.Z['ex'][2]:.1f}) to **{ctx.ex_mean:.1f}% ± {ctx.ex_sd:.1f}** (mean ± SD over {len(ctx.SEEDS)} seeds; first-seed CI {ctx.F['ex'][1]:.1f}–{ctx.F['ex'][2]:.1f}), a gain of **{ctx.pp(ctx.ex_mean - ctx.Z['ex'][0])}** (Table 5, Figure 9). McNemar tests against zero-shot give p = "
        + ", ".join(
            (
                f"{p:.1e}"
                for p in ctx.tests[ctx.tests.comparison.str.contains("zero")]["p-value"]
            )
        )
        + f" for the {len(ctx.SEEDS)} seeds, so the improvement is "
        + (
            "**significant for every seed**."
            if (
                ctx.tests[ctx.tests.comparison.str.contains("zero")]["p-value"] < 0.05
            ).all()
            else "**not significant for every seed**."
        )
        + f" Format compliance went from {ctx.Z['format_ok'][0]:.1f}% to {ctx.F['format_ok'][0]:.1f}% and valid-SQL rate from {ctx.Z['valid'][0]:.1f}% to {ctx.F['valid'][0]:.1f}%."
    )
    if "few" in ctx.R:
        ctx.Fw = ctx.R["few"]["eval"]["A_test"]["summary"]
        ctx.r_few = ctx.tests[ctx.tests.comparison == "tuned vs 3-shot"].iloc[0]
        ctx.paras.append(
            f"**Is fine-tuning necessary?** 3-shot prompting reached {ctx.Fw['ex'][0]:.1f}% EX ({ctx.pp(ctx.Fw['ex'][0] - ctx.Z['ex'][0])} over zero-shot) without any training, while the tuned LoRA model scored {ctx.pp(ctx.F['ex'][0] - ctx.Fw['ex'][0])} above 3-shot (McNemar p = {ctx.r_few['p-value']:.1e}). "
            + (
                "Prompting closes part of the gap, but fine-tuning is clearly better."
                if ctx.r_few["p-value"] < 0.05 and ctx.F["ex"][0] > ctx.Fw["ex"][0]
                else "The difference between prompting and fine-tuning is not significant here."
            )
        )
    if ctx.tune_rows:
        ctx.top_f = ctx.sens_df.index[0]
        ctx.low_f = ctx.sens_df.index[-1]
        ctx.paras.append(
            f"**Hyperparameters (Stage 1).** On the tuning set the most sensitive hyperparameter was **{ctx.HP_LABELS[ctx.top_f]}** (EX range {ctx.sens_df.iloc[0]:.1f} pp across its values) and the least sensitive was **{ctx.HP_LABELS[ctx.low_f]}** ({ctx.sens_df.iloc[-1]:.1f} pp) (Figure 7, Table 3). LoRA rank changed tuning EX by {ctx.sens.get('lora_r', float('nan')):.1f} pp between r = {min(ctx.TUNE_GRID.get('lora_r', [16]))} and r = {max(ctx.TUNE_GRID.get('lora_r', [16]))}, "
            + (
                "consistent with Hu et al. (2021): a small rank already captures this task."
                if ctx.sens.get("lora_r", 0) <= 3
                else "so rank does matter for this model and data."
            )
            + f" The selected configuration is {ctx.BEST_HP} ({ctx.how})."
        )
        if ctx.R["default"] is not ctx.main:
            ctx.r_def = ctx.tests[
                ctx.tests.comparison == "tuned vs default LoRA config"
            ].iloc[0]
            ctx.paras.append(
                f"On the **test set**, the tuned configuration scored {ctx.pp(ctx.F['ex'][0] - ctx.ex_of(ctx.R['default']))} versus the default configuration (McNemar p = {ctx.r_def['p-value']:.1e}), "
                + (
                    "so tuning gave a real improvement."
                    if ctx.r_def["p-value"] < 0.05
                    and ctx.F["ex"][0] > ctx.ex_of(ctx.R["default"])
                    else "so the gain from tuning is within noise - the default values were already reasonable."
                )
            )
    ctx.paras.append(
        f"**Robustness.** Across {len(ctx.SEEDS)} seeds EX varied by SD {ctx.ex_sd:.1f} pp (range {min(ctx.seed_ex['ex']):.1f}–{max(ctx.seed_ex['ex']):.1f}; Figure 10). Differences between configurations smaller than about {2 * ctx.ex_sd:.1f} pp should therefore not be over-interpreted."
    )
    if len(ctx.grid) > 2:
        ctx.gA, ctx.gB = (ctx.grid_df["A-test"], ctx.grid_df["B-test"])
        ctx.kA, ctx.kB, ctx.kAB = (
            [k for k in ctx.grid_df.index if k.startswith("Train A (")][0],
            [k for k in ctx.grid_df.index if k.startswith("Train B")][0],
            [k for k in ctx.grid_df.index if k.startswith("Train A+B")][0],
        )
        ctx.paras.append(
            f"**Generalisation (Stage 3).** Training on A improved B-test EX from {ctx.gB['Zero-shot']:.1f}% to {ctx.gB[ctx.kA]:.1f}%, while training on B reached {ctx.gB[ctx.kB]:.1f}% on B-test and {ctx.gA[ctx.kB]:.1f}% on A-test (Figure 16). The A+B mix scored {ctx.gA[ctx.kAB]:.1f}% / {ctx.gB[ctx.kAB]:.1f}% with the same total data, "
            + (
                "the best balance across both datasets."
                if ctx.gA[ctx.kAB] + ctx.gB[ctx.kAB]
                >= max(ctx.gA[ctx.kA] + ctx.gB[ctx.kA], ctx.gA[ctx.kB] + ctx.gB[ctx.kB])
                - 1
                else "not better on average than training on a single dataset."
            )
            + " Gains transfer only partly across datasets: the model learns the dataset's style as well as SQL."
        )
    if len(ctx.scale) > 1:
        ctx.big = ctx.scale_df.index[-1]
        ctx.paras.append(
            "**Scale (Stage 4).** Zero-shot EX on A-test was "
            + ", ".join(
                (
                    f"{r['zero-shot EX A']:.1f}% ({i[8:]})"
                    for i, r in ctx.scale_df.iterrows()
                )
            )
            + "; after fine-tuning: "
            + ", ".join(
                (f"{r['fine-tuned EX A']:.1f}%" for _, r in ctx.scale_df.iterrows())
            )
            + " (Figure 17). "
            + f"The fine-tuned 0.8B model {('beats' if ctx.scale_df['fine-tuned EX A'].iloc[0] > ctx.scale_df['zero-shot EX A'].iloc[-1] else 'does not beat')} the zero-shot {ctx.big[8:]} model, at {ctx.scale_df['s / example (ft)'].iloc[0] / max(ctx.scale_df['s / example (ft)'].iloc[-1], 1e-09):.2f}x its generation time per example."
        )
        ctx.neg = [
            f"{i[8:]} ({r['zero-shot EX B']:.1f}% -> {r['fine-tuned EX B']:.1f}%)"
            for i, r in ctx.scale_df.iterrows()
            if r["fine-tuned EX B"] < r["zero-shot EX B"] - 2
        ]
        if ctx.neg:
            ctx.paras[-1] += (
                " **Negative transfer:** fine-tuning on A *lowered* B-test accuracy for "
                + ", ".join(ctx.neg)
                + ". The model adopts A's narrow query style (short single-table queries), which hurts on B's multi-domain questions - a form of over-specialisation worth discussing."
            )
    if ctx.RUN_SIZE_ABLATION and len(ctx.abl_df) > 2:
        ctx.a = ctx.abl_df.sort_values("train_examples")
        ctx.half_gain = ctx.a["ex"].iloc[0] + 0.5 * (
            ctx.a["ex"].iloc[-1] - ctx.a["ex"].iloc[0]
        )
        ctx.n_half = int(
            ctx.a.loc[ctx.a["ex"] >= ctx.half_gain, "train_examples"].iloc[0]
        )
        ctx.paras.append(
            f"**Data efficiency (Stage 5).** Half of the total EX gain is reached with about **{ctx.n_half:,}** examples (Figure 18); "
            + (
                "the curve is still rising at the largest size."
                if ctx.a["ex"].iloc[-1] - ctx.a["ex"].iloc[-2] > max(2, ctx.ex_sd)
                else "beyond that, gains are within seed noise."
            )
        )
    ctx.paras.append(
        f"**Errors.** The most common remaining error type after fine-tuning is **{ctx.err_f.idxmax()}** ({int(ctx.err_f.max())} cases, Figure 14). {int(((ctx.ft_df.ex == 1) & ~ctx.ft_df.em.astype(bool)).sum())} predictions were correct by execution but not by exact match (equivalent SQL written differently), which is why EX is the primary metric."
    )
    ctx.md_text = "### Results\n\n" + "\n\n".join(ctx.paras)
    ctx.display(ctx.Markdown(ctx.md_text))
    Path(ctx.OUT_DIR, "results_discussion.md").write_text(ctx.md_text, encoding="utf-8")


def run_artifacts(ctx):
    ctx.cmp.to_csv(f"{ctx.OUT_DIR}/predictions_A_test.csv", index=False)
    ctx.results_json = {
        "config": ctx.CONFIG,
        "selected_hp": ctx.BEST_HP,
        "selection": ctx.how,
        "main_A_test": {
            m: ctx.main["eval"]["A_test"]["summary"][m] for m, _ in ctx.METRICS
        },
        "zero_A_test": {
            m: ctx.R["zero"]["eval"]["A_test"]["summary"][m] for m, _ in ctx.METRICS
        },
        "seeds_A_test_EX": ctx.seed_ex["ex"],
        "mcnemar": ctx.tests.to_dict("records"),
        "cross_dataset": ctx.grid_df.to_dict(),
        "scaling": ctx.scale_df.reset_index().to_dict("records"),
        "size_ablation": ctx.abl_df.to_dict("records"),
    }
    write_json(f"{ctx.OUT_DIR}/results.json", ctx.results_json, indent=2, default=float)
    ctx.t5.to_csv(f"{ctx.OUT_DIR}/table5_main_results.csv", index=False)
    ctx.grid_df.to_csv(f"{ctx.OUT_DIR}/table8_cross_dataset.csv")
    ctx.scale_df.to_csv(f"{ctx.OUT_DIR}/table9_scaling.csv")
    ctx.abl_df.to_csv(f"{ctx.OUT_DIR}/size_ablation.csv", index=False)
    ctx.budget = []
    for ctx.path in sorted(ctx.glob.glob(f"{ctx.RUN_DIR}/{ctx.PFX}*.json")):
        if ctx.path.endswith("best_config.json"):
            continue
        ctx.r = read_json(ctx.path)
        if not ctx.r.get("key", "").startswith(ctx.PFX) or (
            ctx.PFX == "" and ctx.r["key"].startswith("quick_")
        ):
            continue
        ctx.budget.append(
            {
                "run": ctx.r["key"],
                "model": ctx.r["size"],
                "train min": (ctx.r["train"] or {}).get("train_time_min", 0),
                "total min": ctx.r.get("wall_min", 0),
            }
        )
    ctx.budget = ctx.pd.DataFrame(ctx.budget)
    print(
        f"Compute budget: {len(ctx.budget)} runs, {ctx.budget['total min'].sum() / 60:.2f} GPU-hours (training {ctx.budget['train min'].sum() / 60:.2f} h)"
    )
    ctx.display(ctx.budget.round(1))
    print("\nFigures:")
    for ctx.name, ctx.cap in ctx.FIGS:
        print(f"  figures/{ctx.name}.png - {ctx.cap}")
    print(
        "\nTo resume after a crash: zip and download runs/ AND adapters/, upload them as a Kaggle Dataset, attach it with Add Input, then Run All (finished runs and saved adapters are reused)."
    )


def check_adapter(ctx):
    ctx.saved = sorted(
        (
            d
            for d in ctx.glob.glob(f"{ctx.ADAPTER_DIR}/*")
            if ctx.os.path.exists(f"{d}/adapter_config.json")
        )
    )
    ctx.size_mb = lambda d: (
        sum(
            (
                ctx.os.path.getsize(ctx.os.path.join(r, f))
                for r, _, fs in ctx.os.walk(d)
                for f in fs
            )
        )
        / 1000000.0
    )
    ctx.display(
        ctx.pd.DataFrame(
            [
                {
                    "adapter": ctx.os.path.basename(d),
                    "size (MB)": round(ctx.size_mb(d), 1),
                    **{
                        k: v
                        for k, v in read_json(f"{d}/run_info.json").items()
                        if k in ("base_model", "n_train", "seed")
                    },
                }
                for d in ctx.saved
            ]
        )
    )
    ctx.MAIN_ADAPTER = (
        ctx.main.get("adapter_path") or f"{ctx.ADAPTER_DIR}/{ctx.main['key']}"
    )
    write_json(
        f"{ctx.OUT_DIR}/main_adapter.json",
        {"adapter_path": ctx.MAIN_ADAPTER, "base_model": ctx.MODEL_NAME},
        indent=2,
    )
    if ctx.os.path.exists(f"{ctx.MAIN_ADAPTER}/adapter_config.json"):
        ctx.model, ctx.tokenizer = ctx.load_model(ctx.MAIN_ADAPTER)
        ctx.demo = [
            {
                "context": "CREATE TABLE students (name VARCHAR, gpa REAL, major VARCHAR)",
                "question": "Which computer science students have a GPA above 3.5?",
                "answer": "",
            },
            {
                "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, total REAL, year INTEGER)",
                "question": "What is the total revenue per customer in 2025, highest first?",
                "answer": "",
            },
        ]
        ctx.preds, ctx._, ctx._ = ctx.generate(ctx.model, ctx.demo)
        for ctx.d, ctx.p in zip(ctx.demo, ctx.preds):
            print(f"Q: {ctx.d['question']}\nSQL: {ctx.p}\n")
        ctx.chk_rows = ctx.test_A[:16]
        ctx.re_preds, ctx._, ctx._ = ctx.generate(ctx.model, ctx.chk_rows)
        ctx.agree = (
            ctx.np.mean(
                [
                    a == b
                    for a, b in zip(
                        ctx.re_preds, ctx.ft_df["pred"].fillna("").tolist()[:16]
                    )
                ]
            )
            * 100
        )
        print(
            f"Reloaded adapter reproduces {ctx.agree:.0f}% of the original predictions on 16 test questions (greedy decoding; small differences can come from batching)"
        )
        del ctx.model, ctx.tokenizer
        ctx.free_gpu()
    else:
        print(
            "Main adapter not found - re-run Stage 2 (it retrains the main model once to save it)."
        )
