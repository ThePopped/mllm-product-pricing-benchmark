from __future__ import annotations

import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    """Read a JSONL file of ExtractionRecords and return the parsed_json dicts.

    Each line in the file is a full ExtractionRecord produced by the MLLM
    extraction pipeline. Only the `parsed_json` field is returned, as that
    contains the structured sofa features used for modelling.

    Skips any lines where `parsed_json` is null or missing (failed extractions).
    """
    records: list[dict] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            parsed = entry.get("parsed_json")
            if parsed is not None:
                records.append(parsed)
    return records
