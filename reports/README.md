Generated reports go here. Nothing in this folder is a result unless it was produced by a logged run on real data:

- `dataset_verification/`  from `scripts/inspect_dataset.py` (pseudonymised; names exist only in `data/interim/private/`)
- `training/`              from `scripts/train.py` (model comparison, split assignment, training-only OOF scores)
- `evaluation/`            from `scripts/evaluate.py` (held-out metrics, plots, confounding analysis, feature importance)

The pipeline outputs are git-ignored on purpose: regenerate them locally and review them before publishing anything.
