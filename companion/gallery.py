"""Verify saved first-response evidence for display, without model or tool calls."""
import hashlib
from html import escape
import json
from pathlib import Path

from companion.import_response import validate_record
from companion.native_parser import inspect_response, strict_json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_loader(raw, condition_id):
    return {"condition_id": condition_id, "input_sha256": raw["input_wav_sha256"],
            "model": raw["source_model"], "model_revision": raw["model_revision"],
            "raw_response": raw["raw_response"], "prompt": raw["rendered_prompt"],
            "settings": raw["generation_config"], "seed": raw["seed"],
            "timestamp": raw["created_at"]}


def validate_loader(raw, condition_id, frozen):
    """Check the full frozen request, not just its audio identity.

    This verifies consistency of saved provenance, not model authentication,
    semantic correctness, perceptual availability, or completed task execution.
    """
    inputs = {item["condition_id"]: item for item in frozen["inputs_in_order"]}
    require(condition_id in inputs, "Condition is absent from frozen requests")
    request = inputs[condition_id]
    contract = frozen["runtime_contract"]
    require(raw["status"] == "completed" and raw["historical"] is False, "Not a completed current observation")
    require(raw["source_model"] == frozen["model"], "Model identity mismatch")
    require(raw["model_revision"] == frozen["model_revision"], "Model revision mismatch")
    require(raw["processor_revision"] == frozen["processor_revision"], "Processor revision mismatch")
    require(type(raw["seed"]) is int and raw["seed"] == request["seed"] == 1234, "Seed mismatch")
    require(raw["input_wav_sha256"] == request["input_sha256"], "Frozen input hash mismatch")
    require(raw["input_sample_rate"] == request["sample_rate"] == 16000
            and raw["input_samples"] == request["samples"] == 28989, "Input format mismatch")
    require(raw["rendered_prompt"] == frozen["rendered_prompt"], "Rendered prompt mismatch")
    require(digest(raw["rendered_prompt"]) == raw["prompt_sha256_canonical_json_string"]
            == contract["prompt_sha256_canonical_json_string"], "Prompt hash mismatch")
    require(digest(raw["generation_config"]) == digest(frozen["generation_config"])
            == contract["generation_sha256_canonical_json"], "Resolved generation settings mismatch")
    require(raw["messages"] == frozen["messages"] and raw["tools"] == frozen["tools"], "Task/tool or neutral audio message mismatch")
    require(raw["model_input_contains_evaluator_metadata"] is False, "Unexpected evaluator input flag")
    require(isinstance(raw["raw_response"], str), "Raw response must be a string")
    require(hashlib.sha256(raw["raw_response"].encode()).hexdigest() == raw["raw_response_sha256_utf8"], "Raw response hash mismatch")
    tokens = raw["token_ids"]
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens)
            and raw["generated_tokens"] == len(tokens), "Token count or token IDs malformed")
    eos = raw["generation_config"]["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    stopped = bool(tokens and tokens[-1] in eos_ids)
    truncated = bool(tokens and len(tokens) >= raw["generation_config"]["max_new_tokens"] and not stopped)
    require(raw["stopped_at_eos"] is stopped and raw["truncated"] is truncated, "Termination metadata mismatch")
    runtime = raw["runtime"]
    for actual, expected in [("torch", "expected_torch"), ("transformers", "expected_transformers"),
                             ("architecture_sha256", "architecture_sha256"), ("feature_extractor_sha256", "feature_extractor_sha256")]:
        require(runtime[actual] == contract[expected], f"Runtime {actual} mismatch")
    require(runtime["python"] == "3.10.20" and runtime["cuda"] == "12.8", "Python/CUDA runtime mismatch")
    require("RTX PRO 6000" in runtime["gpu_name"] and runtime["gpu_total_bytes"] >= 94 * 1024**3, "GPU contract mismatch")
    require(bool(raw["input_tensors"]) and all(
        isinstance(value.get("shape"), list) and len(value.get("sha256", "")) == 64
        for value in raw["input_tensors"].values()), "Input tensor provenance missing")
    return normalize_loader(raw, condition_id)


def read_observation(root, condition, manifest, frozen):
    root = Path(root)
    key = condition["id"]
    require(condition["evidence_status"] == "observed_first_response", "Unsupported evidence status")
    require(condition["generation_status"] == "completed", "Generation is not complete")
    require(condition["human_review_required"] is True and condition["audio_review_status"] == "pending", "Review flags changed without reviewed evidence")
    paths = {kind: root / f"evidence/gallery/{kind}/{key}.json" for kind in ("raw", "normalized", "validated")}
    records = {kind: strict_json(path.read_text()) for kind, path in paths.items()}
    require(condition["evidence_file"] == str(paths["validated"].relative_to(root)), "Evidence link mismatch")
    normalized = validate_loader(records["raw"], key, frozen)
    require(canonical(normalized) == canonical(records["normalized"]), "Normalization changed loader values")
    validated = validate_record(normalized, manifest, root / "prepared")
    require(canonical(validated) == canonical(records["validated"]), "Saved validation record mismatch")
    require(condition["response"] == normalized["raw_response"], "Gallery text differs from saved response")
    require(condition.get("executed_action") is None and validated["executed_action"] is None, "A proposal must not be treated as executed")
    inspected = inspect_response(normalized["raw_response"])
    require(condition["proposed_action"] == proposed_action(inspected), "Proposed action differs from corrected parser")
    return {"raw": records["raw"], "normalized": normalized, "validated": validated,
            "inspection": inspected, "files": paths}


def proposed_action(inspection):
    if not inspection["would_accept_native_call"]:
        return None
    return {"name": "set_timer", "duration_seconds": inspection["proposed_duration_seconds"],
            "arguments": inspection["parsed"]["calls"][0]["arguments"], "status": "proposal_only"}


def format_duration(seconds):
    """Use minutes for whole-minute timers, matching the spoken instruction."""
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f'{minutes} minute{"s" if minutes != 1 else ""}'
    return f'{seconds} second{"s" if seconds != 1 else ""}'


def observed_fields(raw, inspection, key):
    """All generated content is escaped. Exact special tokens remain inspectable."""
    action = proposed_action(inspection)
    response = raw["raw_response"]
    visible = response.strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        if visible.endswith(token):
            visible = visible[:-len(token)].rstrip()
    if action:
        accompanying = inspection["parsed"]["accompanying_text"]
        summary = "The model returned a timer proposal."
        if accompanying:
            summary += "\n" + accompanying
        action_text = f'Set a timer for {format_duration(action["duration_seconds"])} (proposal only).'
    else:
        summary = visible or "Empty textual response."
        action_text = ("Malformed or unsupported proposal; no valid action parsed."
                       if inspection["parsed"]["attempted"] else "No timer proposal parsed.")
    termination = "Token limit reached; the saved response is truncated." if raw["truncated"] else "Saved first response; no follow-up conversation was run."
    return f'''<dl class="response-fields"><dt>Actual first response</dt>
      <dd class="response-text">{escape(summary)}</dd><dt>Proposed action</dt><dd>{escape(action_text)}</dd></dl>
      <details class="technical response-detail"><summary>Exact response and provenance</summary>
      <pre class="raw-response"><code>{escape(response)}</code></pre>
      <p>{escape(termination)}</p><p>Seed {raw["seed"]} · {escape(raw["created_at"])}<br>No timer was executed.</p>
      <p><a href="data/gallery/{escape(key)}_raw_response.txt" download>Raw response text</a> ·
      <a href="data/gallery/{escape(key)}_loader.json" download>Complete saved observation</a> ·
      <a href="data/gallery/{escape(key)}_validated.json" download>Validation record</a></p></details>'''
