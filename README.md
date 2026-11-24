# Text-to-SQL with Qwen3.5

Python version of the Kaggle notebook in `notebooks/`. Uses Unsloth and LoRA to fine-tune Qwen3.5 for SQLite query generation.

## Files

| File | Contents |
|---|---|
| `config.py` | Settings, dataset sizes and hyperparameters |
| `data.py` | Dataset loading, splits and SQL features |
| `model.py` | Model loading, LoRA training and generation |
| `evaluate.py` | SQL metrics, execution checks and statistical tests |
| `experiments.py` | Tuning, baselines, generalisation, scaling and ablation |
| `reports.py` | Plots, error analysis and result tables |
| `utils.py` | Prompts, SQL extraction, JSON and cached runs |
| `train.py` | Full notebook experiment workflow |
| `finetune.py` | Train one adapter |
| `infer.py` | Generate SQL using a saved adapter |
| `export_model.py` | Optional merged export or Hugging Face upload |

Start with `infer.py`, then `model.py`. For training, read `finetune.py` and `train_adapter()` in `model.py`. `train.py` lists the order of the full research experiments. Functions receive `ctx`, which stores settings, datasets and results shared during a run.

## Setup

Run model training and inference in a supported NVIDIA CUDA environment, such as Kaggle. The code is not configured for macOS CPU/MPS.

```bash
python -m pip install -r requirements.txt
```

Dependencies follow the notebook and are unpinned. Save the package versions from your working GPU environment for reproducibility.

## Train

Train one adapter with smaller data:

```bash
python finetune.py --quick --output-dir outputs/training
```

Remove `--quick` for 2,000 training examples. Use `--dataset b` or `--dataset mixed` for the other training datasets. This command prepares both notebook datasets but skips the research comparisons.

Run the complete notebook study:

```bash
python train.py --quick --output-dir outputs/quick
python train.py --output-dir outputs/full
```

The full study includes tuning, baselines, multiple seeds, generalisation, scaling, ablation, plots and evaluation. Settings are in `config.py`.

Completed experiments are reused when the same output directory is used. To restore Kaggle outputs:

```bash
python train.py --output-dir outputs/restored --resume-from /path/to/previous-output
```

Keep both `runs/` and `adapters/`. Use a new output directory when changing models or data settings because cache keys do not capture every setting.

## Inference

Download the complete adapter directory from Kaggle, including its weights, `adapter_config.json` and tokenizer files. Loading also requires the base model.

```bash
python infer.py --model /path/to/adapter \
  --schema-file examples/schema.sql \
  --question 'How many students have a GPA above 3.5?'
```

For the adapter from the single training command, use `--model outputs/training/adapters/main`.

```bash
python infer.py --model /path/to/adapter \
  --input-file examples/questions.jsonl --output-file outputs/predictions.jsonl
```

Python usage:

```python
from model import Text2SQL

model = Text2SQL('/path/to/adapter')
sql = model.predict('How many students?', 'CREATE TABLE students(name TEXT, gpa REAL);')
print(sql)
```

Inference returns SQL text without executing it. For a local 4-bit adapter, `run_info.json` supplies the quantization setting; `--four-bit` can also be passed explicitly.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

CPU tests cover data filtering, prompts, SQL scoring, caching, inference file handling and cleanup. GPU training and generation still need to be checked in the target CUDA environment. The original notebook's evaluation limitations, including synthetic execution databases, remain applicable.
