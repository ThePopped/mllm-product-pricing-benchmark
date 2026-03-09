from __future__ import annotations

"""Production MLLM feature extraction pipeline.
Unlikely to work locally - initial runs orchestrated on Kaggle.
Run:
    python src/feature_extractor_MLLM.py --screenshots-dir <path>

Outputs:
  - data/processed/MLLM_extracted_features.jsonl (success-only records)
  - artifacts/feature_extraction/<run_id>/logs/extract.log
  - artifacts/feature_extraction/<run_id>/outputs/results_full.jsonl
  - artifacts/feature_extraction/<run_id>/outputs/failures.jsonl
  - artifacts/feature_extraction/<run_id>/outputs/summary.json
  - artifacts/feature_extraction/<run_id>/outputs/run_config.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mlflow
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lineage_utils import runtime_metadata
from project_config import DEFAULT_CONFIG, cfg_path, load_config

# Should probably replace with langchain implementation

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SCREENSHOTS_DIR = ROOT / "artifacts" / "scraping"
DEFAULT_OUTPUT_JSONL = ROOT / "data" / "processed" / "MLLM_extracted_features.jsonl"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "feature_extraction"
DEFAULT_EXPERIMENT = "sofa_price_regression"

log = logging.getLogger("feature_extractor_mllm")


SOFA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "price": {"type": ["number", "null"]},
        "discount_amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "brand": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "colour_1": {"type": ["string", "null"]},
        "colour_2": {"type": ["string", "null"]},
        "colour_3": {"type": ["string", "null"]},
        "material_1": {"type": ["string", "null"]},
        "material_2": {"type": ["string", "null"]},
        "reclining": {"type": ["boolean", "null"]},
        "pull_out_bed": {"type": ["boolean", "null"]},
        "storage": {"type": ["boolean", "null"]},
        "seat_number": {"type": ["integer", "null"]},
        "height": {"type": ["number", "null"]},
        "width": {"type": ["number", "null"]},
        "depth": {"type": ["number", "null"]},
        "product_count": {"type": "integer", "minimum": 1},
        "additional_features": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "price",
        "discount_amount",
        "currency",
        "brand",
        "model",
        "colour_1", # need to make these numbered categories more clearly divided
        "colour_2",
        "colour_3",
        "material_1",
        "material_2",
        "reclining",
        "pull_out_bed",
        "storage",
        "seat_number",
        "height",
        "width",
        "depth",
        "product_count",
        "additional_features",
        "evidence",
    ],
}

PROMPT_TEXT = """You are extracting structured product attributes from a sofa product-page screenshot.
Return ONLY a single valid JSON object that matches the schema exactly.
Do not include keys other than: price,discount_amount,currency,brand,model,colour_1,colour_2,colour_3,material_1,material_2,reclining,pull_out_bed,storage,seat_number,height,width,depth,product_count,additional_features,evidence.
Rules:
- No markdown, no commentary, no trailing commas.
- Use null when a value is unknown/unreadable.
- Booleans must be true/false.
- Numbers must be numeric (no currency symbols, no units inside numbers).
- "additional_features" should capture rare/long-tail features you see (USB/AC chargers, cup holders, wall-hugger, power headrest, storage chaise, console, etc.)
- "evidence" must contain short snippets/icons that justify rare features and price when possible.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_KEYS = set(SOFA_SCHEMA["properties"].keys())


try:
    import jsonschema
except Exception:  # pragma: no cover - optional dependency fallback
    jsonschema = None


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs_dir: Path
    outputs_dir: Path
    full_results_jsonl: Path
    failures_jsonl: Path
    summary_json: Path
    run_config_json: Path


@dataclass
class ExtractionRecord:
    image_path: str
    image_file: str
    success: bool
    error: Optional[str]
    runtime_s: float
    attempts_used: int
    prompt_text: str
    raw_text: Optional[str]
    parsed_json: Optional[dict]


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    paths_cfg = cfg.get("paths", {})
    extraction_cfg = cfg.get("extraction", {})
    default_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    p = argparse.ArgumentParser(description="Extract structured sofa features from screenshots with an MLLM.")
    p.add_argument("--config", type=Path, default=pre_args.config)
    p.add_argument(
        "--screenshots-dir",
        type=Path,
        default=cfg_path(ROOT, paths_cfg.get("scraping_artifacts_dir"), DEFAULT_SCREENSHOTS_DIR),
    )
    p.add_argument("--glob-pattern", type=str, default=str(extraction_cfg.get("glob_pattern", "*.png")))
    p.add_argument("--model-id", type=str, default=str(extraction_cfg.get("model_id", DEFAULT_MODEL_ID)))
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=cfg_path(ROOT, paths_cfg.get("extraction_output_jsonl"), DEFAULT_OUTPUT_JSONL),
    )
    p.add_argument("--overwrite-output", action="store_true")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=cfg_path(ROOT, paths_cfg.get("extraction_artifacts_dir"), DEFAULT_ARTIFACTS_DIR),
    )
    p.add_argument("--run-id", type=str, default=default_run_id)
    p.add_argument("--experiment", type=str, default=str(extraction_cfg.get("experiment", DEFAULT_EXPERIMENT)))
    p.add_argument("--run-name", type=str, default=str(extraction_cfg.get("run_name", "feature_extraction")))
    p.add_argument("--max-new-tokens", type=int, default=int(extraction_cfg.get("max_new_tokens", 300)))
    p.add_argument("--temperature", type=float, default=float(extraction_cfg.get("temperature", 0.2)))
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--retry-on-invalid-json", type=int, default=int(extraction_cfg.get("retry_on_invalid_json", 1)))
    p.add_argument("--limit", type=int, default=0, help="0 means process all matching files.")
    p.add_argument("--min-pixels", type=int, default=int(extraction_cfg.get("min_pixels", 256 * 256)))
    p.add_argument("--max-pixels", type=int, default=int(extraction_cfg.get("max_pixels", 1024 * 1024)))
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization on CUDA.")
    p.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    p.add_argument("--seed", type=int, default=int(extraction_cfg.get("seed", 1001)))
    return p.parse_args()


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def build_run_paths(artifacts_dir: Path, run_id: str) -> RunPaths:
    run_dir = artifacts_dir / run_id
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        logs_dir=logs_dir,
        outputs_dir=outputs_dir,
        full_results_jsonl=outputs_dir / "results_full.jsonl",
        failures_jsonl=outputs_dir / "failures.jsonl",
        summary_json=outputs_dir / "summary.json",
        run_config_json=outputs_dir / "run_config.json",
    )


def configure_runtime(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))


def make_crops(image_path: Path) -> tuple[Image.Image, Image.Image]:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, int(h * 0.62)))
    specs = im.crop((int(w * 0.48), int(h * 0.22), w, int(h * 0.52)))
    return top, specs


def _extract_json_obj(text: str) -> str:
    t = text.strip()
    m = _JSON_FENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = _FIRST_OBJ_RE.search(t)
    if m:
        return m.group(0).strip()
    raise ValueError("No JSON object found in model output.")


def _loads_json(s: str) -> dict:
    s2 = re.sub(r",\s*([}\]])", r"\1", s.strip())
    return json.loads(s2)


def _validate_schema(obj: dict) -> None:
    if jsonschema is not None:
        jsonschema.validate(instance=obj, schema=SOFA_SCHEMA)


def _as_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        s = re.sub(r"[,$£€]", "", x.strip())
        s = re.sub(r"\s*(cm|mm|in|inch|inches|\"|')\s*$", "", s, flags=re.IGNORECASE)
        try:
            return float(s)
        except Exception:
            return None
    return None


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+", x.replace(",", ""))
        return int(m.group(0)) if m else None
    return None


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


def _infer_currency_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    if "£" in text:
        return "GBP"
    if "€" in text:
        return "EUR"
    if "$" in text:
        return "USD"
    return None


def _normalize_to_schema(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise TypeError("Model output JSON must be an object (dict).")

    out: dict[str, Any] = {}
    for key in _ALLOWED_KEYS:
        if key in obj:
            out[key] = obj[key]

    if out.get("model") in (None, "") and isinstance(obj.get("name"), str):
        out["model"] = obj["name"].strip() or None

    specs = obj.get("specifications")
    if isinstance(specs, dict):
        mats = specs.get("materials") or specs.get("material")
        if out.get("material_1") in (None, "") and mats:
            if isinstance(mats, str):
                parts = [p.strip() for p in re.split(r"[/,;+]| and ", mats) if p.strip()]
            elif isinstance(mats, list):
                parts = [str(p).strip() for p in mats if str(p).strip()]
            else:
                parts = []
            out["material_1"] = parts[0] if len(parts) > 0 else None
            out["material_2"] = parts[1] if len(parts) > 1 else out.get("material_2")

        dims = specs.get("dimensions")
        if isinstance(dims, dict):
            if out.get("width") is None:
                out["width"] = _as_number(dims.get("width"))
            if out.get("depth") is None:
                out["depth"] = _as_number(dims.get("depth"))
            if out.get("height") is None:
                out["height"] = _as_number(dims.get("height"))

        if out.get("seat_number") is None:
            sizes = specs.get("sofa_sizes") or specs.get("size") or specs.get("seater")
            if isinstance(sizes, list) and sizes:
                sizes = sizes[0]
            if isinstance(sizes, str):
                match = re.search(r"(\d+)\s*seater", sizes.lower())
                if match:
                    out["seat_number"] = int(match.group(1))

    if out.get("currency") is None:
        out["currency"] = _infer_currency_from_text(
            f"{obj.get('description') or ''} {obj.get('delivery') or ''}"
        )

    out["price"] = _as_number(out.get("price"))
    out["discount_amount"] = _as_number(out.get("discount_amount"))
    out["height"] = _as_number(out.get("height"))
    out["width"] = _as_number(out.get("width"))
    out["depth"] = _as_number(out.get("depth"))
    out["seat_number"] = _as_int(out.get("seat_number"))
    out["reclining"] = _as_bool(out.get("reclining"))
    out["pull_out_bed"] = _as_bool(out.get("pull_out_bed"))
    out["storage"] = _as_bool(out.get("storage"))

    pc = _as_int(out.get("product_count"))
    out["product_count"] = pc if (pc is not None and pc >= 1) else 1

    af = out.get("additional_features")
    out["additional_features"] = af if isinstance(af, list) else ([] if af in (None, "") else [str(af)])
    ev = out.get("evidence")
    out["evidence"] = ev if isinstance(ev, list) else ([] if ev in (None, "") else [str(ev)])

    for key in ("brand", "model", "colour_1", "colour_2", "colour_3", "material_1", "material_2", "currency"):
        value = out.get(key)
        if value is None:
            out[key] = None
        elif isinstance(value, str):
            out[key] = value.strip() or None
        else:
            out[key] = str(value)

    for required in SOFA_SCHEMA["required"]:
        if required not in out:
            if required in ("additional_features", "evidence"):
                out[required] = []
            elif required == "product_count":
                out[required] = 1
            else:
                out[required] = None

    return {key: out[key] for key in SOFA_SCHEMA["properties"].keys()}


def load_model_and_processor(
    model_id: str,
    min_pixels: int,
    max_pixels: int,
    use_4bit: bool,
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    if torch.cuda.is_available():
        kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, processor


@torch.inference_mode()
def extract_one_image(
    image_path: Path,
    *,
    model: Any,
    processor: Any,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    retry_on_invalid_json: int,
) -> tuple[dict, str, dict]:
    started = time.perf_counter()
    top_img, _ = make_crops(image_path)
    image_for_prompt = top_img

    def _run_once(extra_user_text: Optional[str] = None) -> tuple[str, str]:
        user_text = PROMPT_TEXT if extra_user_text is None else f"{PROMPT_TEXT}\n\n{extra_user_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image_for_prompt},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[image_for_prompt], return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        out = processor.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return out.strip(), user_text

    attempts_used = 0
    prompt_text_used = ""
    last_raw = ""
    last_error: Optional[Exception] = None

    for attempt in range(retry_on_invalid_json + 1):
        attempts_used = attempt + 1
        if attempt == 0:
            last_raw, prompt_text_used = _run_once()
        else:
            correction = (
                "Your previous response was not a single valid JSON object matching the schema.\n"
                "Return only the corrected JSON object and no extra text.\n"
                f"Previous response:\n{last_raw}"
            )
            last_raw, prompt_text_used = _run_once(extra_user_text=correction)

        try:
            parsed = _loads_json(_extract_json_obj(last_raw))
            normalized = _normalize_to_schema(parsed)
            _validate_schema(normalized)
            runtime_s = time.perf_counter() - started
            return normalized, last_raw, {
                "runtime_s": runtime_s,
                "attempts_used": attempts_used,
                "prompt_text": prompt_text_used,
            }
        except Exception as exc:
            last_error = exc

    runtime_s = time.perf_counter() - started
    raise ValueError(
        "Model did not produce valid schema-matching JSON. "
        f"runtime_s={runtime_s:.3f}, attempts_used={attempts_used}, last_error={last_error}"
    )


def find_images(screenshots_dir: Path, glob_pattern: str, limit: int) -> list[Path]:
    if not screenshots_dir.exists():
        raise FileNotFoundError(f"Input screenshots directory does not exist: {screenshots_dir}")

    if "**" in glob_pattern:
        images = sorted(p for p in screenshots_dir.glob(glob_pattern) if p.is_file())
    else:
        images = sorted(p for p in screenshots_dir.glob(glob_pattern) if p.is_file())

    if not images:
        raise FileNotFoundError(
            f"No image files found in {screenshots_dir} using pattern '{glob_pattern}'."
        )

    if limit > 0:
        return images[:limit]
    return images


def _build_record(
    image_path: Path,
    *,
    success: bool,
    error: Optional[str],
    runtime_s: float,
    attempts_used: int,
    prompt_text: str,
    raw_text: Optional[str],
    parsed_json: Optional[dict],
) -> ExtractionRecord:
    return ExtractionRecord(
        image_path=str(image_path),
        image_file=image_path.name,
        success=success,
        error=error,
        runtime_s=runtime_s,
        attempts_used=attempts_used,
        prompt_text=prompt_text,
        raw_text=raw_text,
        parsed_json=parsed_json,
    )


def extract_batch(
    image_paths: list[Path],
    *,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
) -> tuple[list[ExtractionRecord], list[ExtractionRecord]]:
    all_records: list[ExtractionRecord] = []
    success_records: list[ExtractionRecord] = []

    for idx, image_path in enumerate(image_paths, start=1):
        log.info(f"[{idx}/{len(image_paths)}] extracting {image_path.name}")
        try:
            parsed_json, raw_text, meta = extract_one_image(
                image_path,
                model=model,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                retry_on_invalid_json=args.retry_on_invalid_json,
            )
            record = _build_record(
                image_path,
                success=True,
                error=None,
                runtime_s=float(meta["runtime_s"]),
                attempts_used=int(meta["attempts_used"]),
                prompt_text=meta["prompt_text"],
                raw_text=raw_text,
                parsed_json=parsed_json,
            )
            success_records.append(record)
        except Exception as exc:
            log.exception(f"failed extraction for {image_path.name}: {exc}")
            record = _build_record(
                image_path,
                success=False,
                error=str(exc),
                runtime_s=0.0,
                attempts_used=0,
                prompt_text="",
                raw_text=None,
                parsed_json=None,
            )
        all_records.append(record)

    return all_records, success_records


def write_jsonl(path: Path, records: list[ExtractionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_summary(path: Path, *, total: int, success: int, failed: int, runtime_s: float) -> dict[str, Any]:
    summary = {
        "total_images": total,
        "successful_extractions": success,
        "failed_extractions": failed,
        "success_rate": (success / total) if total else 0.0,
        "runtime_s": runtime_s,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    run_paths = build_run_paths(args.artifacts_dir, args.run_id)
    configure_logging(run_paths.logs_dir / "extract.log")

    configure_runtime(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["screenshots_dir"] = str(args.screenshots_dir)
    run_config["output_jsonl"] = str(args.output_jsonl)
    run_config["artifacts_dir"] = str(args.artifacts_dir)
    run_config.update(runtime_metadata(root=ROOT, requirements_path=ROOT / "requirements.txt"))
    run_paths.run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    hf_token = os.getenv(args.hf_token_env)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    image_paths = find_images(args.screenshots_dir, args.glob_pattern, args.limit)
    log.info(f"images discovered: {len(image_paths)} from {args.screenshots_dir}")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("stage", "feature_extraction")
        mlflow.log_params(
            {
                "model_id": args.model_id,
                "glob_pattern": args.glob_pattern,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "retry_on_invalid_json": args.retry_on_invalid_json,
                "seed": args.seed,
                "screenshots_dir": str(args.screenshots_dir),
            }
        )

        model, processor = load_model_and_processor(
            args.model_id,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_4bit=(not args.no_4bit),
        )

        started = time.perf_counter()
        all_records, success_records = extract_batch(image_paths, model=model, processor=processor, args=args)
        runtime_s = time.perf_counter() - started

        failed_records = [r for r in all_records if not r.success]
        write_jsonl(run_paths.full_results_jsonl, all_records)
        write_jsonl(run_paths.failures_jsonl, failed_records)

        if args.output_jsonl.exists() and not args.overwrite_output:
            raise FileExistsError(
                f"Output file already exists: {args.output_jsonl}. "
                "Pass --overwrite-output to replace it."
            )
        write_jsonl(args.output_jsonl, success_records)

        summary = write_summary(
            run_paths.summary_json,
            total=len(all_records),
            success=len(success_records),
            failed=len(failed_records),
            runtime_s=runtime_s,
        )

        mlflow.log_metric("num_images_total", len(all_records))
        mlflow.log_metric("num_images_success", len(success_records))
        mlflow.log_metric("num_images_failed", len(failed_records))
        mlflow.log_metric("success_rate", summary["success_rate"])
        mlflow.log_metric("runtime_s", runtime_s)
        mlflow.log_artifact(str(run_paths.run_config_json), artifact_path="feature_extraction")
        mlflow.log_artifact(str(run_paths.summary_json), artifact_path="feature_extraction")
        mlflow.log_artifact(str(run_paths.failures_jsonl), artifact_path="feature_extraction")

    log.info("run complete")
    log.info(f"processed output jsonl: {args.output_jsonl.resolve()}")
    log.info(f"run artifacts: {run_paths.run_dir.resolve()}")


if __name__ == "__main__":
    main()
