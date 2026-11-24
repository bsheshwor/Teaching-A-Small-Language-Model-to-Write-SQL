"""Optional local merged export or Hugging Face upload, matching notebook cell 46."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--merged-dir", help="Save a merged model locally.")
    destination.add_argument(
        "--hub-repo", help="Explicitly upload adapter and tokenizer to this repository."
    )
    args = parser.parse_args()
    if args.hub_repo and not os.environ.get("HF_TOKEN"):
        parser.error("Set HF_TOKEN in the environment before requesting an upload.")
    from model import Text2SQL

    loaded = Text2SQL(args.adapter)
    if args.merged_dir:
        loaded.model.save_pretrained_merged(
            args.merged_dir, loaded.processor, save_method="merged_16bit"
        )
    else:
        loaded.model.push_to_hub(args.hub_repo, token=os.environ["HF_TOKEN"])
        loaded.processor.push_to_hub(args.hub_repo, token=os.environ["HF_TOKEN"])


if __name__ == "__main__":
    main()
