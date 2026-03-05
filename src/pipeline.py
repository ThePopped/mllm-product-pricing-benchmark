from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_config import DEFAULT_CONFIG, cfg_path, load_config

STAGES: list[str] = [
    "scrape",
    "cloudflare",
    "extract",
    "split",
    "train",
    "evaluate",
    "predict",
]


@dataclass(frozen=True)
class StageCommand:
    name: str
    command: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    paths_cfg = cfg.get("paths", {})
    default_predict_input = cfg_path(
        ROOT,
        paths_cfg.get("inference_input_jsonl"),
        ROOT / "data" / "processed" / "holdout.jsonl",
    )
    default_predict_output = cfg_path(
        ROOT,
        paths_cfg.get("inference_output_jsonl"),
        ROOT / "data" / "processed" / "predictions.jsonl",
    )
    default_pipeline_dir = ROOT / "artifacts" / "pipeline"
    default_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    p = argparse.ArgumentParser(
        description="Run the end-to-end sofa price pipeline in one command."
    )
    p.add_argument("--config", type=Path, default=pre_args.config)
    p.add_argument("--run-id", type=str, default=default_run_id)
    p.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    p.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    p.add_argument(
        "--include-cloudflare",
        action="store_true",
        help="Include manual Cloudflare priming stage in the run.",
    )
    p.add_argument(
        "--force-resplit",
        action="store_true",
        help="Delete existing train/holdout split files before running split stage.",
    )
    p.add_argument("--predict-input", type=Path, default=default_predict_input)
    p.add_argument("--predict-output", type=Path, default=default_predict_output)
    p.add_argument("--pipeline-artifacts-dir", type=Path, default=default_pipeline_dir)
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args()


def selected_stages(from_stage: str, to_stage: str, include_cloudflare: bool) -> list[str]:
    from_idx = STAGES.index(from_stage)
    to_idx = STAGES.index(to_stage)
    if from_idx > to_idx:
        raise ValueError(f"Invalid stage range: --from {from_stage} comes after --to {to_stage}.")
    chosen = STAGES[from_idx : to_idx + 1]
    if not include_cloudflare:
        chosen = [s for s in chosen if s != "cloudflare"]
    return chosen


def build_stage_commands(args: argparse.Namespace, cfg: dict) -> list[StageCommand]:
    paths_cfg = cfg.get("paths", {})
    scraping_artifacts_dir = cfg_path(
        ROOT, paths_cfg.get("scraping_artifacts_dir"), ROOT / "artifacts" / "scraping"
    )
    screenshots_dir = scraping_artifacts_dir / args.run_id / "screenshots"

    commands_by_stage: dict[str, list[str]] = {
        "scrape": [
            sys.executable,
            str(ROOT / "src" / "main_scraper.py"),
            "--config",
            str(args.config),
            "--run-id",
            args.run_id,
        ],
        "cloudflare": [
            sys.executable,
            str(ROOT / "src" / "cloudflare_handler.py"),
            "--config",
            str(args.config),
        ],
        "extract": [
            sys.executable,
            str(ROOT / "src" / "feature_extractor_MLLM.py"),
            "--config",
            str(args.config),
            "--run-id",
            args.run_id,
            "--screenshots-dir",
            str(screenshots_dir),
            "--overwrite-output",
        ],
        "split": [
            sys.executable,
            str(ROOT / "src" / "split_data.py"),
            "--config",
            str(args.config),
        ],
        "train": [
            sys.executable,
            str(ROOT / "src" / "train.py"),
            "--config",
            str(args.config),
        ],
        "evaluate": [
            sys.executable,
            str(ROOT / "src" / "evaluate.py"),
            "--config",
            str(args.config),
        ],
        "predict": [
            sys.executable,
            str(ROOT / "src" / "predict.py"),
            "--config",
            str(args.config),
            "--input",
            str(args.predict_input),
            "--output",
            str(args.predict_output),
        ],
    }
    return [StageCommand(name=stage, command=commands_by_stage[stage]) for stage in selected_stages(args.from_stage, args.to_stage, args.include_cloudflare)]


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def maybe_clear_split_files(args: argparse.Namespace, cfg: dict, log_path: Path) -> None:
    if not args.force_resplit:
        return

    paths_cfg = cfg.get("paths", {})
    train_path = cfg_path(ROOT, paths_cfg.get("train_jsonl"), ROOT / "data" / "processed" / "train.jsonl")
    holdout_path = cfg_path(ROOT, paths_cfg.get("holdout_jsonl"), ROOT / "data" / "processed" / "holdout.jsonl")
    for path in (train_path, holdout_path):
        if path.exists():
            path.unlink()
            msg = f"Deleted split file: {path}"
            print(msg)
            append_log(log_path, msg)


def run_command(command: Sequence[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(ROOT), check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_plan = build_stage_commands(args, cfg)

    pipeline_run_dir = args.pipeline_artifacts_dir / args.run_id
    pipeline_run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = pipeline_run_dir / "run_summary.json"
    log_path = pipeline_run_dir / "pipeline.log"

    summary: dict = {
        "pipeline_run_id": args.run_id,
        "config_path": str(args.config),
        "dry_run": args.dry_run,
        "selected_stages": [s.name for s in stage_plan],
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "status": "running",
        "stages": [],
    }
    write_summary(summary_path, summary)

    append_log(log_path, f"Pipeline run started: {summary['started_at_utc']}")
    append_log(log_path, f"Config: {args.config}")
    append_log(log_path, f"Stages: {', '.join(summary['selected_stages'])}")

    for stage in stage_plan:
        if stage.name == "split":
            maybe_clear_split_files(args, cfg, log_path)

        stage_entry = {
            "stage": stage.name,
            "command": stage.command,
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "status": "running",
            "error": None,
        }
        summary["stages"].append(stage_entry)
        write_summary(summary_path, summary)
        append_log(log_path, f"Starting stage: {stage.name}")

        try:
            run_command(stage.command, dry_run=args.dry_run)
            stage_entry["status"] = "skipped_dry_run" if args.dry_run else "completed"
            append_log(log_path, f"Completed stage: {stage.name}")
        except subprocess.CalledProcessError as exc:
            stage_entry["status"] = "failed"
            stage_entry["error"] = f"Exit code {exc.returncode}"
            stage_entry["completed_at_utc"] = utc_now()
            summary["status"] = "failed"
            summary["completed_at_utc"] = utc_now()
            write_summary(summary_path, summary)
            append_log(log_path, f"Failed stage: {stage.name} (exit code {exc.returncode})")
            raise
        finally:
            if stage_entry["completed_at_utc"] is None:
                stage_entry["completed_at_utc"] = utc_now()
            write_summary(summary_path, summary)

    summary["status"] = "completed" if not args.dry_run else "dry_run"
    summary["completed_at_utc"] = utc_now()
    write_summary(summary_path, summary)
    append_log(log_path, f"Pipeline run finished: {summary['completed_at_utc']}")
    print(f"Pipeline summary: {summary_path}")
    print(f"Pipeline log: {log_path}")


if __name__ == "__main__":
    main()
