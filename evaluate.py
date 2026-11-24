from functools import partial


def _user_msg(ctx, row):
    return {
        "role": "user",
        "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}",
    }


def build_prompt(ctx, row, shots=None):
    msgs = [{"role": "system", "content": ctx.SYSTEM}]
    for s in shots or []:
        msgs += [ctx._user_msg(s), {"role": "assistant", "content": s["answer"]}]
    msgs.append(ctx._user_msg(row))
    return ctx.tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def pick_shots(ctx, rows, k=None):
    if k is None:
        k = ctx.N_SHOTS
    "One example per difficulty level (Easy, Medium, Hard), shortest first, so the prompt stays small."
    chosen = []
    for d in ctx.DIFF_ORDER:
        cands = sorted(
            [r for r in rows if ctx.difficulty(r["answer"]) == d],
            key=lambda r: len(r["context"]),
        )
        if cands:
            chosen.append(cands[0])
    for r in rows:
        if len(chosen) >= k:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen[:k]


def clean_output(ctx, text):
    return ctx.re.sub("<think>.*?</think>", "", text, flags=ctx.re.S).strip()


def extract_sql(ctx, text):
    text = ctx.clean_output(text)
    m = ctx.re.search("```(?:sql|sqlite)?\\s*(.*?)```", text, flags=ctx.re.S | ctx.re.I)
    if m:
        text = m.group(1)
    else:
        m = ctx.re.search("\\b(select|with)\\b.*", text, flags=ctx.re.S | ctx.re.I)
        text = m.group(0) if m else text
    return text.strip().split(";")[0].strip()


def format_compliant(ctx, raw):
    t = ctx.clean_output(raw).strip().rstrip(";").strip()
    return (
        bool(ctx.re.match("(?is)^(select|with)\\b", t))
        and "```" not in t
        and (t == ctx.extract_sql(raw))
    )


def normalize(ctx, sql):
    sql = sql.replace('"', "'").lower()
    sql = ctx.re.sub("\\s+", " ", sql)
    sql = ctx.re.sub("\\s*([(),=<>*])\\s*", "\\1", sql)
    return sql.strip().rstrip(";").strip()


def token_f1(ctx, pred, gold):
    p, g = (
        ctx.collections.Counter(ctx.sql_tokens(ctx.normalize(pred))),
        ctx.collections.Counter(ctx.sql_tokens(ctx.normalize(gold))),
    )
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = (overlap / sum(p.values()), overlap / sum(g.values()))
    return 2 * prec * rec / (prec + rec)


def is_valid_sql(ctx, sql, schema):
    return ctx.sqlite_ok(schema, sql)


def literals(ctx, sql):
    strs = [s[1:-1] for s in ctx.STR_RE.findall(sql)]
    nums = ctx.NUM_RE.findall(ctx.STR_RE.sub(" ", sql))
    return (strs, nums)


def _as_number(ctx, v):
    f = float(v)
    return int(f) if f.is_integer() else f


def _rand_value(ctx, rng, ctype, strs, nums):
    if rng.random() < 0.1:
        return None
    if (strs or nums) and rng.random() < 0.5:
        v = rng.choice(strs + nums)
        return ctx._as_number(v) if v in nums else v
    if any((t in ctype.upper() for t in ctx.NUMERIC_TYPES)):
        return rng.randint(0, 100)
    return rng.choice(ctx.WORDS + [str(rng.randint(0, 100))])


def build_dbs(ctx, context, gold, n_dbs=None, rows=None):
    if n_dbs is None:
        n_dbs = ctx.N_EXEC_DBS
    if rows is None:
        rows = ctx.ROWS_PER_TABLE
    strs, nums = ctx.literals(gold)
    has_rows = bool(ctx.re.search("\\binsert\\b", context, ctx.re.I))
    dbs = []
    for k in range(n_dbs):
        rng = ctx.random.Random(
            ctx.zlib.crc32(f"{context}|{gold}|{k}|{ctx.SEED}".encode())
        )
        con = ctx.sqlite3.connect(":memory:")
        try:
            con.executescript(context)
            if not (has_rows and k == 0):
                tables = [
                    r[0]
                    for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ]
                for t in tables:
                    cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                    q = f'''INSERT INTO "{t}" VALUES ({",".join("?" * len(cols))})'''
                    for _ in range(rows):
                        try:
                            con.execute(
                                q,
                                [
                                    ctx._rand_value(rng, c[2] or "", strs, nums)
                                    for c in cols
                                ],
                            )
                        except Exception:
                            pass
        except Exception:
            pass
        dbs.append(con)
    return dbs


def _run(ctx, con, sql, timeout=2.0):
    start = ctx.time.time()
    con.set_progress_handler(
        lambda: 1 if ctx.time.time() - start > timeout else 0, 1000
    )
    try:
        rows = con.execute(sql).fetchmany(5000)
    finally:
        con.set_progress_handler(None, 1000)
    return [
        tuple((round(v, 6) if isinstance(v, float) else v for v in r)) for r in rows
    ]


def exec_match(ctx, pred, gold, dbs):
    """True/False, or None if the gold query cannot run on the first database (excluded from EX)."""
    ordered = bool(ctx.re.search("\\border\\s+by\\b", gold, ctx.re.I))
    for k, con in enumerate(dbs):
        try:
            g = ctx._run(con, gold)
        except Exception:
            if k == 0:
                return None
            continue
        try:
            p = ctx._run(con, pred)
        except Exception:
            return False
        if (
            g != p
            if ordered
            else ctx.collections.Counter(g) != ctx.collections.Counter(p)
        ):
            return False
    return True


def gold_nonempty(ctx, gold, dbs):
    try:
        return any((len(ctx._run(con, gold)) > 0 for con in dbs))
    except Exception:
        return False


def _idents(ctx, sql):
    s = ctx.STR_RE.sub(" ", sql.lower())
    return {
        t for t in ctx.re.findall("[a-z_][a-z0-9_]*", s) if t not in ctx.SQL_KEYWORDS
    }


def _tables(ctx, sql):
    return {
        t.lower().strip('`"')
        for t in ctx.re.findall(
            '\\b(?:from|join)\\s+([`\\"]?\\w+)', ctx.STR_RE.sub(" ", sql), ctx.re.I
        )
    }


def error_type(ctx, pred, gold, valid):
    if not valid:
        return "Invalid SQL"
    if ctx._tables(pred) != ctx._tables(gold):
        return "Wrong table"
    if ctx._idents(pred) - ctx._tables(pred) != ctx._idents(gold) - ctx._tables(gold):
        return "Wrong column"
    lp, lg = (ctx.literals(pred), ctx.literals(gold))
    if sorted(lp[0] + lp[1]) != sorted(lg[0] + lg[1]):
        return "Wrong literal value"
    if sorted((m.lower() for m in ctx.AGG_RE.findall(pred))) != sorted(
        (m.lower() for m in ctx.AGG_RE.findall(gold))
    ):
        return "Wrong aggregation"
    return "Wrong clause / structure"


def score(ctx, preds, raws, rows, dbs_list):
    recs = []
    for p, raw, r, dbs in zip(preds, raws, rows, dbs_list):
        valid = ctx.is_valid_sql(p, r["context"])
        ex = ctx.exec_match(p, r["answer"], dbs)
        em = ctx.normalize(p) == ctx.normalize(r["answer"])
        correct = ex if ex is not None else em
        recs.append(
            {
                "question": r["question"],
                "gold": r["answer"],
                "pred": p,
                "raw_output": raw,
                "em": em,
                "ex": ex,
                "valid": valid,
                "token_f1": ctx.token_f1(p, r["answer"]),
                "format_ok": ctx.format_compliant(raw),
                "correct": bool(correct),
                "pred_tokens": len(ctx.sql_tokens(p)),
                "difficulty": ctx.difficulty(r["answer"]),
                "error_type": "Correct"
                if correct
                else ctx.error_type(p, r["answer"], valid),
                **{f"has_{k}": fn(r["answer"]) for k, fn in ctx.FEATURES.items()},
            }
        )
    return ctx.pd.DataFrame(recs)


def bootstrap_ci(ctx, x, n=None, seed=None):
    if n is None:
        n = ctx.N_BOOT
    if seed is None:
        seed = ctx.SEED
    x = ctx.np.asarray(
        [
            v
            for v in x
            if v is not None and (not (isinstance(v, float) and ctx.np.isnan(v)))
        ],
        float,
    )
    if len(x) == 0:
        return (ctx.np.nan, ctx.np.nan, ctx.np.nan)
    means = ctx.np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return (
        100 * x.mean(),
        100 * ctx.np.percentile(means, 2.5),
        100 * ctx.np.percentile(means, 97.5),
    )


def summarize(ctx, df):
    return {m: ctx.bootstrap_ci(df[m].tolist()) for m, _ in ctx.METRICS}


def mcnemar(ctx, a, b):
    """a, b: boolean arrays (model 1 correct, model 2 correct). Exact binomial McNemar test."""
    a, b = (ctx.np.asarray(a, bool), ctx.np.asarray(b, bool))
    n01, n10 = (int((~a & b).sum()), int((a & ~b).sum()))
    p = ctx.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return (n01, n10, p)


def prepare_evaluation(ctx):
    ctx.SYSTEM = "You are a text-to-SQL assistant. Given a database schema and a question, reply with a single SQLite query and nothing else."
    ctx._user_msg = partial(_user_msg, ctx)
    ctx.build_prompt = partial(build_prompt, ctx)
    ctx.pick_shots = partial(pick_shots, ctx)
    ctx.SHOTS_A = ctx.pick_shots(ctx.train_A[:500])
    ctx.SHOTS_B = ctx.pick_shots(ctx.pool_B[ctx.N_TRAIN + ctx.N_VAL :])
    ctx.clean_output = partial(clean_output, ctx)
    ctx.extract_sql = partial(extract_sql, ctx)
    ctx.format_compliant = partial(format_compliant, ctx)
    ctx.normalize = partial(normalize, ctx)
    ctx.token_f1 = partial(token_f1, ctx)
    ctx.is_valid_sql = partial(is_valid_sql, ctx)
    ctx.NUMERIC_TYPES = ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUB")
    ctx.WORDS = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "north",
        "south",
        "red",
        "blue",
        "x1",
        "y2",
    ]
    ctx.literals = partial(literals, ctx)
    ctx._as_number = partial(_as_number, ctx)
    ctx._rand_value = partial(_rand_value, ctx)
    ctx.build_dbs = partial(build_dbs, ctx)
    ctx._run = partial(_run, ctx)
    ctx.exec_match = partial(exec_match, ctx)
    ctx.gold_nonempty = partial(gold_nonempty, ctx)
    ctx._idents = partial(_idents, ctx)
    ctx._tables = partial(_tables, ctx)
    ctx.ERROR_ORDER = [
        "Invalid SQL",
        "Wrong table",
        "Wrong column",
        "Wrong literal value",
        "Wrong aggregation",
        "Wrong clause / structure",
    ]
    ctx.error_type = partial(error_type, ctx)
    ctx.score = partial(score, ctx)
    ctx.bootstrap_ci = partial(bootstrap_ci, ctx)
    ctx.METRICS = [
        ("ex", "Execution Accuracy"),
        ("em", "Exact Match"),
        ("valid", "Valid SQL"),
        ("token_f1", "Token F1"),
        ("format_ok", "Format compliance"),
    ]
    ctx.summarize = partial(summarize, ctx)
    ctx.mcnemar = partial(mcnemar, ctx)
    ctx.t0 = ctx.time.time()
    ctx.EVAL_SETS = {}
    for ctx.name, ctx.rows, ctx.shots in [
        ("A_test", ctx.test_A, ctx.SHOTS_A),
        ("B_test", ctx.test_B, ctx.SHOTS_B),
        ("A_tune", ctx.tune_A, ctx.SHOTS_A),
    ]:
        ctx.EVAL_SETS[ctx.name] = (
            ctx.rows,
            [ctx.build_dbs(r["context"], r["answer"]) for r in ctx.rows],
            ctx.shots,
        )
    ctx.sanity = []
    for ctx.name, (ctx.rows, ctx.dbs, ctx._) in ctx.EVAL_SETS.items():
        ctx.golds = [r["answer"] for r in ctx.rows]
        ctx.gold_ex = [ctx.exec_match(g, g, d) for g, d in zip(ctx.golds, ctx.dbs)]
        ctx.const_ex = [
            ctx.exec_match("SELECT 1", g, d) for g, d in zip(ctx.golds, ctx.dbs)
        ]
        ctx.sanity.append(
            {
                "eval set": ctx.name,
                "gold executable (%)": 100
                * ctx.np.mean([x is not None for x in ctx.gold_ex]),
                "gold vs itself EX (must be 100)": 100
                * ctx.np.mean([x for x in ctx.gold_ex if x is not None]),
                "'SELECT 1' EX (should be ~0)": 100
                * ctx.np.mean([x for x in ctx.const_ex if x is not None]),
                "gold non-empty on >= 1 DB (%)": 100
                * ctx.np.mean(
                    [ctx.gold_nonempty(g, d) for g, d in zip(ctx.golds, ctx.dbs)]
                ),
            }
        )
    ctx.sanity = ctx.pd.DataFrame(ctx.sanity).set_index("eval set").round(1)
    ctx.display(ctx.sanity)
    print(
        f"Built {sum((len(v[1]) for v in ctx.EVAL_SETS.values())) * ctx.N_EXEC_DBS} databases in {ctx.time.time() - ctx.t0:.1f}s"
    )
    assert (ctx.sanity["gold vs itself EX (must be 100)"] == 100).all(), (
        "EX of gold vs gold must be 100%"
    )
    print("\nFew-shot examples for A:", [s["answer"] for s in ctx.SHOTS_A])
