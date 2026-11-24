import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Adapter folder or Hugging Face repository ID."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question")
    mode.add_argument(
        "--input-file", type=Path, help="JSONL rows with question and context fields."
    )
    schema = parser.add_mutually_exclusive_group()
    schema.add_argument("--schema")
    schema.add_argument("--schema-file", type=Path)
    parser.add_argument("--output-file", type=Path, help="Write predictions as JSONL.")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--four-bit", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args(argv)
    if args.max_seq_length < 1 or args.max_new_tokens < 1:
        parser.error("Token limits must be positive.")
    try:
        if args.input_file:
            if args.schema or args.schema_file:
                parser.error("JSONL input supplies the schema in each context field.")
            rows = [
                json.loads(line)
                for line in args.input_file.read_text().splitlines()
                if line.strip()
            ]
        else:
            if args.schema is None and args.schema_file is None:
                parser.error("--question requires --schema or --schema-file.")
            rows = [
                {
                    "question": args.question,
                    "context": args.schema_file.read_text()
                    if args.schema_file
                    else args.schema,
                }
            ]
        if not rows or any(
            (
                not isinstance(r, dict)
                or not all(
                    (
                        isinstance(r.get(k), str) and r[k].strip()
                        for k in ("question", "context")
                    )
                )
                for r in rows
            )
        ):
            parser.error(
                "Each input row must contain nonempty question and context strings."
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    from model import Text2SQL

    model = Text2SQL(args.model, args.max_seq_length, args.four_bit)
    predictions = [
        dict(
            row, sql=model.predict(row["question"], row["context"], args.max_new_tokens)
        )
        for row in rows
    ]
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            "".join((json.dumps(r) + "\n" for r in predictions))
        )
    elif args.question:
        print(predictions[0]["sql"])
    else:
        for row in predictions:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
