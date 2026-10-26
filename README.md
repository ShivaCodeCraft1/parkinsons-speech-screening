# AI-Assisted Parkinson's Disease Screening Using Speech Signals

> **Research prototype. NOT a diagnostic medical device.** It does not diagnose Parkinson's disease, makes no treatment
> recommendation, and has no demonstrated clinical validity. Performance on controlled dataset recordings does not
> guarantee performance on arbitrary microphone recordings.

## Status (what has and has not been verified)

| Component | Code | Verified how |
|---|---|---|
| Dataset verifier (Phase 1, preserved) + summaries | done | 22 original synthetic-data tests + CLI run on a synthetic dataset. **Not yet run on the real dataset.** |
| Audio loading / validation / preprocessing | done | unit tests (valid, stereo, empty, corrupt, silent, NaN, short, clipping, resampling) |
| Feature extraction (50 features, saved schema) | done | unit tests; identical code path for training and API (tested) |
| Subject-grouped split, grouped CV, tuning, threshold | done | unit + synthetic end-to-end tests, CLI run on synthetic data |
| Held-out evaluation, bootstrap CIs, age/group confounding, importance | done | CLI run on synthetic data only |
| FastAPI (`/health`, `/model-info`, `/predict`, `/analyze`, `/metrics`, demo) | done | TestClient tests + a live `uvicorn` run (WAV, OGG, WebM uploads) on a synthetic-data model |
| Browser UI (record / upload / waveform / spectrogram / prediction) | done | served and JS syntax-checked; **microphone recording NOT verified in a real browser** |
| Docker / docker-compose / GitHub Actions | written | **never executed** (no Docker daemon or GitHub runner was available) |
| Cloud deployment | not done | instructions only |
| **Real-data features, training, held-out evaluation** | code ready | **NOT RUN. No trained model ships with this repo.** |

Everything "synthetic" above means generated sine/harmonic signals used only to test the software. They say nothing about
medical accuracy. **No dataset statistic or performance metric exists in this repository.** Any AUROC/sensitivity/specificity on
a CV or resume must come from your own `scripts/evaluate.py` run; claims such as AUROC 0.91 / 90% / 87% are unverified until then.

## Dataset
Primary candidate: *Italian Parkinson's Voice and Speech* (IEEE DataPort; free login to download). Licence: **unknown** from the
supplied files (the page asks users to cite the associated papers), so no recordings are bundled and `.gitignore` blocks `*.wav`.
Keep the dataset **outside** the repo and point to it with `PD_DATA_ROOT`. Group names and task codes in
`configs/dataset_italian_pvs.yaml` were confirmed by hand by the project owner; the verifier cross-checks them against the real files.

## Architecture
```
PD_DATA_ROOT (your WAVs) --scripts/inspect_dataset.py--> full manifest + excluded-recordings audit + reports
                         --scripts/extract_features.py--> data/processed/features.csv  (+ feature_schema.json)
features.csv --scripts/train.py--> subject holdout -> grouped CV model comparison -> threshold from OOF scores -> models/current/
models/current + held-out subjects --scripts/evaluate.py--> reports/evaluation/*
audio (mic / upload) --api/main.py--> decode -> validate -> preprocess -> SAME feature code -> saved pipeline -> score
```
| Folder | Contents |
|---|---|
| `src/audio`, `src/preprocessing`, `src/validation` | decoding (soundfile, ffmpeg fallback), resampling/DC/peak-normalise, audio + feature checks |
| `src/features/extract.py` | the single feature implementation (Praat via Parselmouth + librosa) |
| `src/data` | verifier (Phase 1), manifest helpers, summaries, subject-level splits |
| `src/models`, `src/evaluation`, `src/explainability` | training, artifacts, inference, metrics/confounding/plots, importance |
| `api/` + `frontend/` | FastAPI backend; plain HTML/CSS/JS UI (design tokens in `frontend/style.css`) |
| `configs/` | `dataset_italian_pvs.yaml`, `features.yaml`, `training.yaml` |

## Key design decisions
- **Positive class = Parkinson's (`is_pd=1`); metrics are binary** (PD vs young + elderly controls). Age is a known shortcut risk.
- **Leakage prevention:** the unit of splitting is the participant (`subject_id`). Holdout split by subject (stratified by group), `StratifiedGroupKFold` for tuning, imputer + scaler inside the sklearn Pipeline (refit per fold), explicit overlap assertions (tested). The test set is not touched by `train.py`; threshold selection uses only out-of-fold *training* scores (Youden J).
- **Dataset audit / Option B:** Parkinson's participants may be nested under numeric range folders (for example `1-5/participant`); the verifier resolves the participant folder instead of treating the range as a person. The full manifest preserves all WAVs. Recordings with unparsed task codes are also written to `data/interim/excluded_recordings.csv` with a reason; codes are never guessed. `src/data/dataset.py` only allows readable, labelled recordings with `task_parse_status == ok` into training. Multiple distinct files for the same subject/task are flagged for review, but are not automatically treated as duplicate content; byte-identical / decoded-audio-identical files remain hard-check failures.
- **Task choice:** `configs/training.yaml` defaults to `task_families: [sustained_vowel]` (easy to reproduce live). This is a design choice made without seeing your data; change it (or set `null` for all tasks), then re-train. The model only makes sense for the task it was trained on.
- **Undefined measurements** (e.g. jitter on unvoiced audio) stay `NaN` in the feature table; the training pipeline imputes with training medians and the API lists which features were imputed.
- **Scores are not probabilities.** The UI/API say "score (uncalibrated)" and a threshold; never a percentage risk.
- **No noise reduction** (it alters jitter/shimmer/HNR), peak-normalisation to 0.95 for gain invariance. Browser echo-cancel/noise-suppress/auto-gain are switched off during recording.
- **Age/confounding analysis** (`evaluate.py`): age-only vs acoustic CV AUROC, PD vs elderly-only AUROC, score-vs-age correlation among controls, per-group/task/sex rates; subgroups with fewer than 6 subjects are suppressed. Adding age as a feature does not remove confounding.
- **Explainability:** permutation importance for any model, plus |coef| (logistic regression) or SHAP (tree models); per-prediction contributions via `?explain=true` for those model types only. Importance describes the model, not biology.

## Windows quick start (PowerShell, VS Code)
```powershell
# 1-2. Extract and open
Expand-Archive .\parkinsons-speech-screening-complete.zip -DestinationPath C:\Projects
cd C:\Projects\parkinsons-speech-screening
code .

# 3-5. Virtual environment + dependencies (Python 3.11 or 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements\dev.txt

# 6. Point to your dataset (kept outside the repo)
Copy-Item .env.example .env
notepad .env                           # set PD_DATA_ROOT=D:/path/to/Italian Parkinson's Voice and Speech

# 7. Inspect + verify dataset  (read reports\dataset_verification\dataset_report.md; exit code 1 = a hard check failed)
python scripts\inspect_dataset.py

# 8. Features (all tasks; add --task-families sustained_vowel to be faster)
python scripts\extract_features.py --n-jobs 4

# 9. Train (optional: --nested-cv for a cross-validated estimate)
python scripts\train.py

# 10. Evaluate ONCE on the held-out subjects
python scripts\evaluate.py

# 11-12. API + web UI
uvicorn api.main:app --port 8000
start http://localhost:8000/app/        # Swagger: http://localhost:8000/docs

# 15. Tests (synthetic data only)
pytest -q
```
`--print-tree-depth 3` on `inspect_dataset.py` lists folder names (console only) if the folder regexes in the config need adjusting.

**13. Recording:** open `http://localhost:8000/app/`, allow the microphone, press *Start recording*, say a steady "aaah" for ~5 s, *Stop*, check playback, *Analyze*. Use `localhost` (browsers only allow microphones on localhost/HTTPS).
**14. Upload:** choose a WAV (other formats are converted to WAV in the browser when possible; the server falls back to ffmpeg if installed).
**Demo mode:** set `PD_ENABLE_DEMO_MODE=1` in `.env`, restart the API; a sample picker appears (held-out subjects by default). The model result is shown first; the dataset label appears separately, labelled "not a model output", with the sample's train/test partition.

### Training vs inference
*Training* (offline, your machine): features from the dataset → subject split → tuning → `models/current/{model.joblib, feature_schema.json, metadata.json}`.
*Inference* (API): loads those artifacts once; for each request decodes the audio **in memory**, runs the identical preprocessing + feature code, orders features by the saved schema, applies the saved pipeline. It never retrains. Without artifacts, `/predict` and `/model-info` return `503 model_not_available`; `/analyze` (validation, plots, features) still works.
Regenerate artifacts any time with `extract_features.py` then `train.py`. Changing `configs/features.yaml` requires re-extracting and retraining (the API warns on a hash mismatch).

### API
`GET /health` · `GET /model-info` · `POST /predict[?explain=true]` (multipart `file`) · `POST /analyze` · `GET /metrics` · `GET /demo/samples`, `POST /demo/predict/{id}` (demo mode only). Errors are `{"error": {"code", "message"}}` with 400/413/415/422/503. Upload limit 20 MB, 1–60 s.

## Docker (optional)
```powershell
python scripts\train.py                      # artifacts must exist in models\current first
docker build -t pd-speech .
docker run --rm -p 8000:8000 -v ${PWD}\models\current:/app/models/current:ro pd-speech
# or: docker compose up --build
```
Not executed during development; treat as untested until you run it.

## Deployment notes
Any container host (Render, Fly.io, Azure/GCP container services, Hugging Face Docker Space) can run the Dockerfile. Plan for: ≥1 GB RAM (librosa/numba/SHAP imports); artifact size from <1 MB (logistic regression/SVM) to tens of MB (random forest); no persistent storage needed (audio is processed in memory; the ffmpeg fallback uses a temp file that is deleted); env vars `PD_MODEL_DIR`, `PD_LOG_LEVEL` (leave `PD_ENABLE_DEMO_MODE=0`); HTTPS is required for microphone access off localhost; set a request-size limit at the proxy. Because the dataset licence is unknown, check it before publishing a model trained on it. No secrets are needed or stored. Do not call a deployment successful until you have opened it and run a prediction.

## Troubleshooting
- **Python/pip errors:** use Python 3.11/3.12 (64-bit). `pip install -r requirements\dev.txt` inside the activated `.venv`.
- **`Dataset path not configured`:** set `PD_DATA_ROOT` in `.env` or pass `--data-root`.
- **`path_lookup_PRIVATE.csv not found`:** run `inspect_dataset.py` on this machine first.
- **Audio won't decode:** upload WAV; install ffmpeg for other containers; check the file isn't empty/corrupt.
- **Mic blocked:** allow it in the browser address bar; use `http://localhost:8000`, not an IP address.
- **`model_not_available`:** run extract + train. **Port in use:** `uvicorn api.main:app --port 8010`.
- **Config changed warning:** re-extract features and retrain.

## Testing
`pytest -q` → 61 tests, all passing at packaging time (22 original verifier tests + 39 new). They use synthetic signals only; they test software behaviour, not clinical accuracy. `ruff check .` is clean.

## Limitations
Small single-site dataset (65 subjects expected → a ~20% test split is only ~13 subjects, so metrics and CIs will be unstable); possible age/sex/recording-condition shortcuts; domain shift to microphone recordings; scores uncalibrated; one person stored under two folder names cannot be detected automatically; no external validation; not clinically validated.

## Disclaimer
Research prototype for screening-support research. Not a medical device; must not be used for clinical decisions.
