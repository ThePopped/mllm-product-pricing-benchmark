from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from src import pipeline


def test_pipeline_dry_run_respects_stage_range_and_writes_summary(tmp_path, monkeypatch) -> None:
    cfg = {
        "paths": {
            "scraping_artifacts_dir": str((tmp_path / "scraping").resolve()),
            "train_jsonl": str((tmp_path / "train.jsonl").resolve()),
            "holdout_jsonl": str((tmp_path / "holdout.jsonl").resolve()),
            "inference_input_jsonl": str((tmp_path / "inference_input.jsonl").resolve()),
            "inference_output_jsonl": str((tmp_path / "inference_output.jsonl").resolve()),
        }
    }
    cfg_path = tmp_path / "defaults.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline.py",
            "--config",
            str(cfg_path),
            "--run-id",
            "test_run",
            "--from",
            "train",
            "--to",
            "predict",
            "--dry-run",
            "--pipeline-artifacts-dir",
            str((tmp_path / "pipeline_artifacts").resolve()),
        ],
    )

    pipeline.main()

    summary_path = tmp_path / "pipeline_artifacts" / "test_run" / "run_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["selected_stages"] == ["train", "evaluate", "predict"]
    assert len(summary["stages"]) == 3
    assert all(stage["status"] == "skipped_dry_run" for stage in summary["stages"])
