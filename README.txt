Price Benchmark MLLM
====================

Reproducibility baseline
------------------------
- Install from lockfile:
  pip install -r requirements.txt
- Editable dependency list:
  requirements.in
- Regenerate lockfile after editing requirements.in:
  pip-compile --output-file=requirements.txt requirements.in

- Copy environment template once:
  copy .env.example .env
  (or on bash: cp .env.example .env)

- Shared project config:
  config/defaults.yaml
  All major scripts accept:
  --config config/defaults.yaml

Pipeline order
--------------
1) Capture screenshots
   python src/main_scraper.py --run-id 20260305T180000Z

2) Prime Cloudflare state (optional but recommended)
   python src/cloudflare_handler.py --url https://www.ufurnish.com

3) Extract structured features from screenshots
   python src/feature_extractor_MLLM.py ^
     --screenshots-dir artifacts/scraping/20260305T180000Z/screenshots ^
     --output-jsonl data/processed/MLLM_extracted_features.jsonl ^
     --overwrite-output

4) Split into train/holdout
   python src/split_data.py

5) Train
   python src/train.py

6) Evaluate
   python src/evaluate.py

7) Predict (batch inference)
   python src/predict.py ^
     --input data/processed/single_row_test.jsonl ^
     --output data/processed/single_test_pred.jsonl

Single-command orchestration
----------------------------
- End-to-end pipeline (recommended default):
  python src/pipeline.py --config config/defaults.yaml

- Include manual Cloudflare priming stage:
  python src/pipeline.py --config config/defaults.yaml --include-cloudflare

- Run only part of the pipeline:
  python src/pipeline.py --config config/defaults.yaml --from split --to evaluate

- Recreate train/holdout split safely:
  python src/pipeline.py --config config/defaults.yaml --from split --to predict --force-resplit

- Preview commands without executing:
  python src/pipeline.py --config config/defaults.yaml --dry-run

- Pipeline run metadata/logs are written to:
  artifacts/pipeline/<run_id>/run_summary.json
  artifacts/pipeline/<run_id>/pipeline.log

Online serving (FastAPI)
------------------------
- Start API server:
  python src/serve_api.py --config config/defaults.yaml

- Open minimal web UI:
  http://127.0.0.1:8000/

- Health and readiness:
  GET /healthz
  GET /readyz

- Serving contract:
  GET /v1/schema
    Returns numeric fields (must be > 0), plus:
    - categorical_fields_top: top-K most common values per categorical field
      (K configured by `serving.top_k_categories`, default 10)
    - categorical_fields_all: all allowed values observed in training metadata
      (`models/category_meta.json`)

  POST /v1/predict
    Body:
      {
        "numeric_field": "width",
        "numeric_value": 210.0,
        "categorical_field": "brand",
        "categorical_value": "Julian Bowen"
      }
    Response includes predicted price plus model lineage identifiers.

- UI integration pattern:
  1) Call /v1/schema to populate dropdowns (use top values by default).
  2) Let user pick one numeric field + positive value.
  3) Let user pick one categorical field + allowed value.
  4) Send POST /v1/predict.

Notes
-----
- Scraper and extraction logs/artifacts are written under artifacts/.
- Extraction writes success-only records to data/processed/MLLM_extracted_features.jsonl.
- Batch inference entrypoint: src/predict.py (accepts JSONL/JSON input).
- Set HF_TOKEN environment variable if your selected model requires authentication.
- Track local secrets in .env only (never commit .env).
- Fill these values first:
  - .env: HF_TOKEN (if model is gated), optionally MLFLOW_TRACKING_URI.
  - config/defaults.yaml paths: update if your directories differ.
  - config/defaults.yaml model_id: choose your target MLLM.
