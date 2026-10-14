"""Synthetic record tests only: never model calls or gallery observations."""
from copy import deepcopy
import hashlib
from html import unescape
import json
from pathlib import Path
import tempfile
import unittest

from companion.gallery import normalize_loader, observed_fields, proposed_action, read_observation, validate_loader
from companion.import_response import validate_record
from companion.native_parser import inspect_response

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "evidence/gallery/run_manifest.json").read_text())
MANIFEST = json.loads((ROOT / "prepared/audio_manifest.json").read_text())
CALL = '<tool_call>\n{"name":"set_timer","arguments":{"hours":0,"minutes":9,"seconds":0}}\n</tool_call><|im_end|>'


def synthetic_record(text=CALL):
    contract = FROZEN["runtime_contract"]
    return {"status": "completed", "historical": False, "source_model": FROZEN["model"],
            "model_revision": FROZEN["model_revision"], "processor_revision": FROZEN["processor_revision"],
            "seed": 1234, "input_wav_sha256": FROZEN["inputs_in_order"][0]["input_sha256"],
            "input_sample_rate": 16000, "input_samples": 28989, "rendered_prompt": FROZEN["rendered_prompt"],
            "prompt_sha256_canonical_json_string": contract["prompt_sha256_canonical_json_string"],
            "generation_config": deepcopy(FROZEN["generation_config"]), "messages": FROZEN["messages"],
            "tools": FROZEN["tools"], "model_input_contains_evaluator_metadata": False,
            "raw_response": text, "raw_response_sha256_utf8": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": [10, FROZEN["generation_config"]["eos_token_id"]], "generated_tokens": 2,
            "stopped_at_eos": True, "truncated": False, "created_at": "2026-10-14T00:00:00Z",
            "input_tensors": {"input_features": {"shape": [1, 128, 181], "sha256": "0" * 64}},
            "runtime": {"python": "3.10.20", "cuda": "12.8", "torch": contract["expected_torch"],
                        "transformers": contract["expected_transformers"], "architecture_sha256": contract["architecture_sha256"],
                        "feature_extractor_sha256": contract["feature_extractor_sha256"],
                        "gpu_name": "NVIDIA RTX PRO 6000", "gpu_total_bytes": 96 * 1024**3}}


class GalleryTests(unittest.TestCase):
    def test_full_provenance_gate(self):
        raw = synthetic_record()
        self.assertEqual(validate_loader(raw, "reference", FROZEN), normalize_loader(raw, "reference"))
        for field, value in [("input_wav_sha256", "0" * 64), ("model_revision", "wrong"),
                             ("processor_revision", "wrong"), ("seed", 5678),
                             ("rendered_prompt", "different"), ("input_samples", 1),
                             ("source_model", "wrong"), ("raw_response", "changed")]:
            changed = deepcopy(raw)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_loader(changed, "reference", FROZEN)

    def test_settings_and_runtime_mismatch_rejected(self):
        for field, key, value in [("generation_config", "temperature", .7),
                                  ("runtime", "architecture_sha256", "0" * 64),
                                  ("runtime", "gpu_total_bytes", 80 * 1024**3),
                                  ("runtime", "transformers", "wrong")]:
            raw = synthetic_record()
            raw[field][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_loader(raw, "reference", FROZEN)

    def test_validated_observation_can_render_with_pending_review(self):
        raw = synthetic_record()
        record = normalize_loader(raw, "reference")
        validated = validate_record(record, MANIFEST, ROOT / "prepared")
        condition = {"id": "reference", "evidence_status": "observed_first_response",
                     "generation_status": "completed", "human_review_required": True,
                     "audio_review_status": "pending", "evidence_file": "evidence/gallery/validated/reference.json",
                     "response": CALL, "proposed_action": proposed_action(inspect_response(CALL)), "executed_action": None}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared/audio").mkdir(parents=True)
            (root / "prepared/audio/reference.wav").write_bytes((ROOT / "prepared/audio/reference.wav").read_bytes())
            for name, value in [("raw", raw), ("normalized", record), ("validated", validated)]:
                path = root / f"evidence/gallery/{name}/reference.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(value))
            result = read_observation(root, condition, MANIFEST, FROZEN)
            self.assertTrue(result["validated"]["human_review_required"])
            self.assertIsNone(result["validated"]["executed_action"])
            self.assertFalse(result["inspection"]["real_action_executed"])
            self.assertIn("9 minutes (proposal only)", observed_fields(raw, result["inspection"], "reference"))
            condition["executed_action"] = {"timer": 540}
            with self.assertRaisesRegex(ValueError, "executed"):
                read_observation(root, condition, MANIFEST, FROZEN)

    def test_generated_markup_is_escaped_and_special_tokens_preserved(self):
        text = '<script>alert("test")</script> How many minutes?<|im_end|>'
        raw = synthetic_record(text)
        html = observed_fields(raw, inspect_response(text), "reference")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn(text, unescape(html))
        self.assertIn("How many minutes?", html)

    def test_empty_malformed_and_ambiguous_responses_remain_visible(self):
        for text, expected in [("", "Empty textual response"),
                               ("<tool_callBROKEN", "Malformed or unsupported proposal"),
                               ("Can you elaborate?<|im_end|>", "Can you elaborate?")]:
            with self.subTest(text=text):
                html = observed_fields(synthetic_record(text), inspect_response(text), "reference")
                self.assertIn(expected, html)
                self.assertIn("No timer was executed.", html)

    def test_legacy_interpretation_cannot_override_corrected_parser(self):
        text = CALL + "<tool_callBROKEN"
        raw = synthetic_record(text)
        raw["proposal_inspection"] = {"would_accept_native_call": True, "proposed_duration_seconds": 540}
        html = observed_fields(raw, inspect_response(text), "reference")
        self.assertIn("Malformed or unsupported proposal", html)
        self.assertNotIn("9 minutes (proposal only)", html)
