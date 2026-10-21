# Predicting Cardiovascular Risk using Electronic Health Records

UE24CS352A Machine Learning - Mini-Project. Team: `Madhumitha Dhondi (PES1UG24AM923), Anjali Arun (PES1UG24AM421)`

Predicts presence of cardiovascular disease from routine clinical data and compares regularised logistic regression
(Ridge / Lasso / Elastic Net) against a gradient boosting model (XGBoost), with AUROC evaluation and a gender/age subgroup check.
The approach follows Nguyen's Stanford CS229 EHR work; because the original STRIDE data is private, we use the public
Kaggle Cardiovascular Disease dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset as described in [`data/README.md`](data/README.md) (place it at `data/cardio_train.csv`).

## Run

```bash
python src/train.py --data data/cardio_train.csv     # full run (10-fold CV grid search)
python src/train.py --data data/cardio_train.csv --fast   # quick run, smaller grids
```

`python src/train.py --synthetic --fast` runs the pipeline on generated data, for checking the setup only (do not report those numbers).

## Outputs (`results/`)

`metrics.csv` (CV/test AUROC, accuracy, sensitivity, specificity), `roc.png`, `subgroups.csv`, `feature_importance.png`,
`best_params.json`, `data_summary.json`.

## Structure

```
src/data.py         loading, cleaning, feature engineering
src/train.py        model tuning, evaluation, plots, subgroup analysis
report/             writeup.pdf
slides/             deck.pptx
```
