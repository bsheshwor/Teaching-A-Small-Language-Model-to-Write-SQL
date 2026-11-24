from functools import partial


def sqlite_ok(ctx, context, sql, timeout=2.0):
    con = ctx.sqlite3.connect(":memory:")
    try:
        con.executescript(context)
        start = ctx.time.time()
        con.set_progress_handler(
            lambda: 1 if ctx.time.time() - start > timeout else 0, 1000
        )
        con.execute(sql).fetchmany(10)
        return True
    except Exception:
        return False
    finally:
        con.close()


def gretel_rows(ctx, split, n_keep, seed, scan_limit):
    ds = ctx.load_dataset(ctx.DS_B, split=split).shuffle(seed=seed)
    out, scanned, why = ([], 0, ctx.collections.Counter())
    for r in ds:
        if len(out) >= n_keep or scanned >= scan_limit:
            break
        scanned += 1
        sql = r["sql"].strip().rstrip(";").strip()
        schema_context = r["sql_context"].strip()
        if not ctx.re.match("(?is)^(select|with)\\b", sql) or ";" in sql:
            why["not a single read-only query"] += 1
            continue
        if len(schema_context) + len(r["sql_prompt"]) + len(sql) > ctx.MAX_PROMPT_CHARS:
            why["too long"] += 1
            continue
        if not ctx.sqlite_ok(schema_context, sql):
            why["does not run in SQLite"] += 1
            continue
        out.append(
            {
                "question": r["sql_prompt"],
                "context": schema_context,
                "answer": sql,
                "complexity": r["sql_complexity"],
                "domain": r["domain"],
            }
        )
    return (out, scanned, why)


def load_data(ctx):
    ctx.sqlite_ok = partial(sqlite_ok, ctx)
    ctx.raw_A = ctx.load_dataset(ctx.DS_A, split="train").shuffle(seed=ctx.SEED)
    ctx.cuts = ctx.np.cumsum([0, ctx.N_TRAIN, ctx.N_VAL, ctx.N_TUNE, ctx.N_TEST])
    ctx.train_A, ctx.val_A, ctx.tune_A, ctx.test_A = (
        ctx.raw_A.select(range(ctx.cuts[i], ctx.cuts[i + 1])).to_list()
        for i in range(4)
    )
    ctx.full_A = ctx.raw_A.to_pandas()
    ctx.full_A["split"] = "unused pool"
    for ctx.name, (ctx.lo, ctx.hi) in zip(
        ["train", "validation", "tuning", "test"], zip(ctx.cuts[:-1], ctx.cuts[1:])
    ):
        ctx.full_A.loc[ctx.lo : ctx.hi - 1, "split"] = ctx.name
    ctx.gretel_rows = partial(gretel_rows, ctx)
    ctx.t0 = ctx.time.time()
    ctx.pool_B, ctx.scanned_B, ctx.why_B = ctx.gretel_rows(
        "train", ctx.B_POOL, ctx.SEED, scan_limit=40 * ctx.B_POOL
    )
    ctx.test_B, ctx.scanned_Bt, ctx.why_Bt = ctx.gretel_rows(
        "test", ctx.N_TEST, ctx.SEED, scan_limit=40 * ctx.N_TEST
    )
    ctx.train_B, ctx.val_B = (
        ctx.pool_B[: ctx.N_TRAIN],
        ctx.pool_B[ctx.N_TRAIN : ctx.N_TRAIN + ctx.N_VAL],
    )
    print(
        f"Dataset B: kept {len(ctx.pool_B)} of {ctx.scanned_B} scanned train rows and {len(ctx.test_B)} of {ctx.scanned_Bt} test rows ({ctx.time.time() - ctx.t0:.0f}s). Dropped: {dict(ctx.why_B)}"
    )
    assert len(ctx.train_B) == ctx.N_TRAIN and len(ctx.test_B) == ctx.N_TEST, (
        "Not enough usable Gretel rows - raise scan_limit"
    )
    ctx.split_table = ctx.pd.DataFrame(
        [
            ["A", "full dataset", len(ctx.raw_A)],
            ["A", "train", len(ctx.train_A)],
            ["A", "validation", len(ctx.val_A)],
            ["A", "tuning", len(ctx.tune_A)],
            ["A", "test", len(ctx.test_A)],
            ["B", "usable pool (filtered)", len(ctx.pool_B)],
            ["B", "train", len(ctx.train_B)],
            ["B", "validation", len(ctx.val_B)],
            ["B", "test (from official test split)", len(ctx.test_B)],
        ],
        columns=["dataset", "split", "examples"],
    )
    ctx.display(ctx.split_table)
    print("\nExample from A:")
    print(ctx.json.dumps(ctx.train_A[0], indent=2))
    print("\nExample from B:")
    print(ctx.json.dumps({k: v for k, v in ctx.train_B[0].items()}, indent=2)[:1200])


def _has(ctx, pattern):
    rx = ctx.re.compile(pattern, ctx.re.I)
    return lambda s: bool(rx.search(ctx.STR_RE.sub(" ", s)))


def n_conditions(ctx, sql):
    s = ctx.STR_RE.sub(" ", sql)
    m = ctx.re.search(
        "\\bwhere\\b(.*?)(\\bgroup\\s+by\\b|\\border\\s+by\\b|\\blimit\\b|\\bhaving\\b|$)",
        s,
        ctx.re.I | ctx.re.S,
    )
    return (
        0 if not m else 1 + len(ctx.re.findall("\\b(and|or)\\b", m.group(1), ctx.re.I))
    )


def difficulty(ctx, sql):
    f = {k: fn(sql) for k, fn in ctx.FEATURES.items()}
    if (
        f["JOIN"]
        or f["Nested query"]
        or f["Set operation"]
        or f["HAVING"]
        or f["Window / CTE"]
    ):
        return "Hard"
    if f["GROUP BY"] or f["ORDER BY"] or ctx.n_conditions(sql) >= 2:
        return "Medium"
    return "Easy"


def parse_schema(ctx, context):
    tables = {}
    for name, cols in ctx.CREATE_RE.findall(context):
        cols = ctx.re.sub("\\([^()]*\\)", "", cols)
        names = [
            c.strip().split()[0].strip('`"[]')
            for c in cols.split(",")
            if c.strip() and (not ctx.CONSTRAINT_RE.match(c.strip()))
        ]
        tables[name.strip('`"[] ')] = names
    return tables


def sql_skeleton(ctx, sql):
    """Query template: literals -> 'val'/'num', identifiers -> 'x'."""
    s = ctx.STR_RE.sub(" val ", sql.lower())
    s = ctx.NUM_RE.sub(" num ", s)
    s = ctx.re.sub(
        "[a-z_][a-z0-9_]*",
        lambda m: (
            m.group(0) if m.group(0) in ctx.SQL_KEYWORDS | {"val", "num"} else "x"
        ),
        s,
    )
    return ctx.re.sub(
        "\\s+", " ", ctx.re.sub("\\s*([(),=<>*])\\s*", " \\1 ", s)
    ).strip()


def sql_tokens(ctx, sql):
    return ctx.re.findall("'[^']*'|\\\"[^\\\"]*\\\"|\\w+|[^\\s\\w]", sql)


def add_eda_columns(ctx, df):
    df = df.copy()
    df["question_words"] = df["question"].str.split().str.len()
    df["sql_tokens"] = df["answer"].map(lambda s: len(ctx.sql_tokens(s)))
    schema = df["context"].map(ctx.parse_schema)
    df["n_tables"] = schema.map(len)
    df["n_columns"] = schema.map(lambda t: sum((len(c) for c in t.values())))
    df["has_sample_rows"] = df["context"].str.contains(
        "\\binsert\\b", case=False, regex=True
    )
    df["difficulty"] = df["answer"].map(ctx.difficulty)
    for k, fn in ctx.FEATURES.items():
        df[f"has_{k}"] = df["answer"].map(fn)
    df["skeleton"] = df["answer"].map(ctx.sql_skeleton)
    return df


def initialize_features(ctx):
    ctx.SQL_KEYWORDS = set(
        "select from where and or not in like between is null join inner left right outer on as\ngroup by order having limit distinct count sum avg min max asc desc union intersect except exists case when\nthen else end cross natural using all any with over partition row_number rank dense_rank".split()
    )
    ctx.STR_RE = ctx.re.compile("'(?:[^']|'')*'|\\\"[^\\\"]*\\\"")
    ctx.NUM_RE = ctx.re.compile("(?<![\\w.])-?\\d+(?:\\.\\d+)?(?![\\w.])")
    ctx.AGG_RE = ctx.re.compile("\\b(count|sum|avg|min|max)\\s*\\(", ctx.re.I)
    ctx.CREATE_RE = ctx.re.compile(
        'CREATE TABLE\\s+(?:IF NOT EXISTS\\s+)?([`\\"\\[]?[\\w ]+?[`\\"\\]]?)\\s*\\((.*?)\\)\\s*(?:;|$)',
        ctx.re.I | ctx.re.S,
    )
    ctx.CONSTRAINT_RE = ctx.re.compile(
        "^(primary|foreign|unique|constraint|check|key|index)\\b", ctx.re.I
    )
    ctx._has = partial(_has, ctx)
    ctx.FEATURES = {
        "WHERE": ctx._has("\\bwhere\\b"),
        "Aggregate": lambda s: bool(ctx.AGG_RE.search(ctx.STR_RE.sub(" ", s))),
        "ORDER BY": ctx._has("\\border\\s+by\\b"),
        "GROUP BY": ctx._has("\\bgroup\\s+by\\b"),
        "JOIN": ctx._has("\\bjoin\\b"),
        "DISTINCT": ctx._has("\\bdistinct\\b"),
        "LIMIT": ctx._has("\\blimit\\b"),
        "HAVING": ctx._has("\\bhaving\\b"),
        "Nested query": lambda s: (
            len(ctx.re.findall("\\bselect\\b", ctx.STR_RE.sub(" ", s), ctx.re.I)) > 1
        ),
        "Set operation": ctx._has("\\b(union|intersect|except)\\b"),
        "Window / CTE": ctx._has("\\bover\\s*\\(|^\\s*with\\b"),
    }
    ctx.DIFF_ORDER = ["Easy", "Medium", "Hard"]
    ctx.n_conditions = partial(n_conditions, ctx)
    ctx.difficulty = partial(difficulty, ctx)
    ctx.parse_schema = partial(parse_schema, ctx)
    ctx.sql_skeleton = partial(sql_skeleton, ctx)
    ctx.sql_tokens = partial(sql_tokens, ctx)
    ctx.add_eda_columns = partial(add_eda_columns, ctx)
