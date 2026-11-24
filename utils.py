import json
from pathlib import Path
import re
from functools import partial


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, **kwargs):
    Path(path).write_text(json.dumps(value, **kwargs), encoding="utf-8")


class Markdown(str):
    pass


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else str(value))


SYSTEM = "You are a text-to-SQL assistant. Given a database schema and a question, reply with a single SQLite query and nothing else."


def _user_msg(row):
    return {
        "role": "user",
        "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}",
    }


def build_prompt(tokenizer, row, shots=None):
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots or []:
        msgs += [_user_msg(s), {"role": "assistant", "content": s["answer"]}]
    msgs.append(_user_msg(row))
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def clean_output(text):
    return re.sub("<think>.*?</think>", "", text, flags=re.S).strip()


def extract_sql(text):
    text = clean_output(text)
    m = re.search("```(?:sql|sqlite)?\\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        text = m.group(1)
    else:
        m = re.search("\\b(select|with)\\b.*", text, flags=re.S | re.I)
        text = m.group(0) if m else text
    return text.strip().split(";")[0].strip()


def safe_key(ctx, s):
    return ctx.re.sub("[^\\w.-]", "_", s)


def hp_key(ctx, hp):
    return f"r{hp['lora_r']}_a{hp['alpha_mult']}_lr{hp['lr']:g}_e{hp['epochs']}_{hp['targets']}"


def load_run(ctx, key):
    path = f"{ctx.RUN_DIR}/{key}.json"
    if not ctx.os.path.exists(path):
        return None
    rec = read_json(path)
    for name in rec.get("eval", {}):
        rec["eval"][name]["df"] = ctx.pd.read_csv(
            f"{ctx.RUN_DIR}/{key}__{name}.csv", keep_default_na=False, na_values=[""]
        )
    return rec


def save_run(ctx, rec):
    slim = {k: v for k, v in rec.items() if k != "eval"}
    slim["eval"] = {}
    for name, ev in rec.get("eval", {}).items():
        ev["df"].to_csv(f"{ctx.RUN_DIR}/{rec['key']}__{name}.csv", index=False)
        slim["eval"][name] = {k: v for k, v in ev.items() if k != "df"}
    write_json(f"{ctx.RUN_DIR}/{rec['key']}.json", slim, indent=1, default=float)


def initialize_cache(ctx):
    ctx.RUN_FILE_RE = ctx.re.compile(
        "^(quick_)?((tune|zero|fewshot|ftA|ftB|ftAB)_.*\\.(json|csv)|best_config\\.json)$"
    )
    ctx.restored, ctx.sources = (0, set())
    for ctx.src in (
        ctx.glob.glob(str(ctx.options.resume_from / "**/*"), recursive=True)
        if ctx.options.resume_from
        else []
    ):
        ctx.name = ctx.os.path.basename(ctx.src)
        if ctx.os.path.isfile(ctx.src) and ctx.RUN_FILE_RE.match(ctx.name):
            ctx.dst = f"{ctx.RUN_DIR}/{ctx.name}"
            if not ctx.os.path.exists(ctx.dst):
                ctx.shutil.copy(ctx.src, ctx.dst)
                ctx.restored += 1
                ctx.sources.add(ctx.os.path.dirname(ctx.src))
    for ctx.cfg in (
        ctx.glob.glob(
            str(ctx.options.resume_from / "**/adapter_config.json"), recursive=True
        )
        if ctx.options.resume_from
        else []
    ):
        ctx.src_dir = ctx.os.path.dirname(ctx.cfg)
        ctx.dst_dir = f"{ctx.ADAPTER_DIR}/{ctx.os.path.basename(ctx.src_dir)}"
        if not ctx.os.path.exists(ctx.dst_dir):
            ctx.shutil.copytree(ctx.src_dir, ctx.dst_dir)
            ctx.restored += 1
            ctx.sources.add(ctx.os.path.dirname(ctx.src_dir))
    print(
        f"Restored {ctx.restored} cached files/adapters from attached inputs {sorted(ctx.sources)[:3]} | cached runs available: {len(ctx.glob.glob(f'{ctx.RUN_DIR}/*.json'))}"
    )
    ctx.safe_key = partial(safe_key, ctx)
    ctx.hp_key = partial(hp_key, ctx)
    ctx.load_run = partial(load_run, ctx)
    ctx.save_run = partial(save_run, ctx)
