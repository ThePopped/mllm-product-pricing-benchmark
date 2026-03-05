Price Benchmark MLLM
====================

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

Notes
-----
- Scraper and extraction logs/artifacts are written under artifacts/.
- Extraction writes success-only records to data/processed/MLLM_extracted_features.jsonl.
- Set HF_TOKEN environment variable if your selected model requires authentication.
