"""Validate a future saved response for review. No inference or gallery mutation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from companion.native_parser import inspect_response, strict_json

ROOT = Path(__file__).resolve().parents[1]


def validate_record(record, manifest, audio_root):
    required = {"condition_id", "input_sha256", "model", "model_revision", "raw_response",
                "prompt", "settings", "seed", "timestamp"}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError("Missing response provenance fields")
    for key in ("condition_id", "input_sha256", "model", "model_revision", "prompt", "timestamp"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"Nonempty {key} required")
    if not isinstance(record["raw_response"], str):
        raise ValueError("raw_response must preserve the complete response as a string")
    if not isinstance(record["settings"], dict) or not record["settings"]:
        raise ValueError("Nonempty prompt/settings provenance is required")
    if type(record["seed"]) is not int or record["seed"] < 0:
        raise ValueError("An explicit nonnegative integer seed is required")
    timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    key = record["condition_id"]
    if key not in manifest["files"]:
        raise ValueError("Unknown condition ID")
    info = manifest["files"][key]
    root = Path(audio_root).resolve()
    path = (root / info["path"]).resolve()
    if root not in path.parents:
        raise ValueError("Audio path escapes manifest directory")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", record["input_sha256"]):
        raise ValueError("input_sha256 must be an exact lowercase SHA-256")
    if record["input_sha256"] != info["sha256"] or actual != info["sha256"]:
        raise ValueError("Input hash mismatch: response cannot attach to this condition")
    return {"evidence_status": "imported_response_pending_review", "record": record,
            "inspection": inspect_response(record["raw_response"]),
            "human_review_required": True, "executed_action": None,
            "note": "Provenance validated, not authenticated. Parser heuristics are not a validated judge; no action execution inferred."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "prepared/audio_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_record(strict_json(args.record.read_text()),
                                 strict_json(args.manifest.read_text()), args.manifest.parent)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    # Preserve prior evidence; choose a new path for each import.
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved for human review: {args.output}. No gallery card or timer changed.")


if __name__ == "__main__":
    main()
