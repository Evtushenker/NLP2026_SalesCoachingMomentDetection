# Actionable Sales-Coaching Moment Detection

This repository contains a compact NLP pipeline for detecting actionable coaching moments in English sales-dialogue fragments. The system works with short speaker-tagged windows and predicts whether a fragment should be reviewed by a sales coach, with a separate signal for unresolved customer concerns.

The project uses the public Hugging Face dataset [`gwenshap/sales-transcripts`](https://huggingface.co/datasets/gwenshap/sales-transcripts). Source transcripts are converted into dialogue windows, labeled for coaching-oriented signals, evaluated with grouped cross-validation, and ranked for human review.

## Repository Structure

```text
.
|-- main.tex                 # self-contained Overleaf report
|-- README.md                # project overview and run instructions
|-- project/
|   |-- requirements.txt     # Python dependencies
|   `-- src/                 # data, modeling, evaluation, recommendation code
`-- .gitignore
```

Raw transcripts, intermediate files, reports, and model artifacts are intentionally not stored in Git. The repository includes a compact reviewed gold sample and focused-task prediction artifacts so the main evaluation scripts can be run without rebuilding the full dataset first.

## Main Task

The main input is a dialogue window such as:

```text
manager: ...
client: ...
manager: ...
client: ...
```

The main prediction targets are:

- `actionable_coaching_needed`: the window contains a concrete coaching opportunity;
- `unresolved_customer_concern`: the customer has an open concern or objection that still needs a manager response.

Auxiliary labels used for analysis and recommendations:

- `needs_discovery`
- `value_articulation`
- `objection_handling`
- `next_step_fixed`
- `early_presentation_before_discovery`

## Results

Grouped cross-validation by `dialogue_id` on the reviewed 100-window gold sample:

| Target | Best approach | F1 |
| --- | --- | ---: |
| `actionable_coaching_needed` | TF-IDF + Logistic Regression | 0.862 |
| `unresolved_customer_concern` | open-concern dialogue rule | 0.810 |

Review-queue evaluation at a top-20% review budget:

| Target | Ranking approach | Precision | Recall | Lift |
| --- | --- | ---: | ---: | ---: |
| `actionable_coaching_needed` | TF-IDF + Logistic Regression | 0.950 | 0.322 | 1.610x |
| `unresolved_customer_concern` | open-concern dialogue rule | 0.900 | 0.462 | 2.308x |

## Included Data

The following small artifacts are committed for reproducibility:

- `project/data/samples/gwenshap_focused_gold_tasks.csv`
- `project/data/processed/focused_task_cv_results.json`
- `project/data/processed/focused_task_cv_predictions.csv`

They are enough to rerun focused-task evaluation, prioritization evaluation, and recommendation demos. Raw transcripts and intermediate datasets can still be regenerated from the public Hugging Face dataset.

## Setup

Install dependencies:

```bash
pip install -r project/requirements.txt
```

The dataset loader uses Hugging Face `datasets`, so the first data-preparation run requires internet access.

## Run

Run the focused-task evaluation directly from the included gold sample:

```bash
python project/src/models/train_focused_tasks.py \
  --input project/data/samples/gwenshap_focused_gold_tasks.csv \
  --output project/data/processed/focused_task_cv_results.json \
  --predictions-output project/data/processed/focused_task_cv_predictions.csv
```

Evaluate prioritization quality from the included out-of-fold predictions:

```bash
python project/src/evaluation/evaluate_prioritization.py \
  --predictions project/data/processed/focused_task_cv_predictions.csv \
  --output project/data/processed/prioritization_results.json \
  --report-output project/outputs/prioritization_results.md
```

Build evidence and confidence demos from the included out-of-fold predictions:

```bash
python project/src/recommendations/build_evidence_review.py \
  --predictions project/data/processed/focused_task_cv_predictions.csv \
  --output project/data/samples/gwenshap_evidence_review_30.csv \
  --target unresolved_customer_concern \
  --model open_concern_rule \
  --limit 30

python project/src/recommendations/build_confidence_demo.py \
  --predictions project/data/processed/focused_task_cv_predictions.csv \
  --output project/data/samples/gwenshap_confidence_demo_30.csv \
  --limit 30
```

To rebuild windows from the public source dataset, run:

Prepare public transcripts and build dialogue windows:

```bash
python project/src/data/prepare_gwenshap_sales_transcripts.py \
  --raw-dir project/data/raw/gwenshap_sales_transcripts/transcripts \
  --output project/data/processed/gwenshap_candidate_windows.csv \
  --sample-output project/data/samples/gwenshap_sample_100_windows.csv
```

Then create weak labels for a window sample:

```bash
python project/src/data/weak_label_sales_windows.py \
  --input project/data/samples/gwenshap_sample_100_windows.csv \
  --output project/data/samples/gwenshap_sample_100_windows_weak_labeled.csv
```

Build focused binary labels from a reviewed gold-label file:

```bash
python project/src/data/build_focused_tasks.py \
  --input project/data/samples/gwenshap_sample_100_windows_gold_labeled.csv \
  --output project/data/samples/gwenshap_focused_gold_tasks.csv
```

Run the window-size ablation:

```bash
python project/src/data/build_window_size_ablation.py \
  --gold project/data/samples/gwenshap_focused_gold_tasks.csv \
  --candidate-windows project/data/processed/gwenshap_candidate_windows.csv \
  --output-dir project/data/processed/window_ablation \
  --window-sizes 2,4,6

python project/src/evaluation/evaluate_window_ablation.py \
  --input-dir project/data/processed/window_ablation \
  --output project/data/processed/window_ablation/window_ablation_results.json \
  --report-output project/outputs/window_size_ablation_results.md \
  --window-sizes 2,4,6
```
