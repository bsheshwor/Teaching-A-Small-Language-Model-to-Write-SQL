### Results

**Main finding.** LoRA fine-tuning of Qwen3.5-0.8B on 2,000 examples raised execution accuracy on A-test from **34.0%** (zero-shot, 95% CI 27.3–40.7) to **81.6% ± 1.2** (mean ± SD over 3 seeds; first-seed CI 75.3–86.1), a gain of **+47.6 pp** (Table 5, Figure 9). McNemar tests against zero-shot give p = 1.9e-24, 6.0e-26, 6.2e-28 for the 3 seeds, so the improvement is **significant for every seed**. Format compliance went from 100.0% to 100.0% and valid-SQL rate from 90.5% to 95.0%.

**Is fine-tuning necessary?** 3-shot prompting reached 26.3% EX (-7.7 pp over zero-shot) without any training, while the tuned LoRA model scored +54.6 pp above 3-shot (McNemar p = 9.0e-29). Prompting closes part of the gap, but fine-tuning is clearly better.

**Hyperparameters (Stage 1).** On the tuning set the most sensitive hyperparameter was **target modules** (EX range 14.2 pp across its values) and the least sensitive was **alpha / r** (1.0 pp) (Figure 7, Table 3). LoRA rank changed tuning EX by 7.8 pp between r = 4 and r = 64, so rank does matter for this model and data. The selected configuration is {'lora_r': 16, 'alpha_mult': 1, 'lr': 0.0002, 'epochs': 3, 'targets': 'all'} (combined per-factor choices).

On the **test set**, the tuned configuration scored +1.5 pp versus the default configuration (McNemar p = 5.8e-01), so the gain from tuning is within noise - the default values were already reasonable.

**Robustness.** Across 3 seeds EX varied by SD 1.2 pp (range 80.9–83.0; Figure 10). Differences between configurations smaller than about 2.4 pp should therefore not be over-interpreted.

**Generalisation (Stage 3).** Training on A improved B-test EX from 41.0% to 23.0%, while training on B reached 54.5% on B-test and 38.7% on A-test (Figure 16). The A+B mix scored 79.4% / 57.0% with the same total data, the best balance across both datasets. Gains transfer only partly across datasets: the model learns the dataset's style as well as SQL.

**Scale (Stage 4).** Zero-shot EX on A-test was 34.0% (0.8B), 37.1% (2B); after fine-tuning: 80.9%, 83.0% (Figure 17). The fine-tuned 0.8B model beats the zero-shot 2B model, at 1.20x its generation time per example. **Negative transfer:** fine-tuning on A *lowered* B-test accuracy for 0.8B (41.0% -> 23.0%), 2B (50.5% -> 32.5%). The model adopts A's narrow query style (short single-table queries), which hurts on B's multi-domain questions - a form of over-specialisation worth discussing.

**Data efficiency (Stage 5).** Half of the total EX gain is reached with about **250** examples (Figure 18); beyond that, gains are within seed noise.

**Errors.** The most common remaining error type after fine-tuning is **Wrong literal value** (15 cases, Figure 14). 16 predictions were correct by execution but not by exact match (equivalent SQL written differently), which is why EX is the primary metric.