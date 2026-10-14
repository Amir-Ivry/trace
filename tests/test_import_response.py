from copy import deepcopy
import hashlib
import tempfile
from pathlib import Path
import unittest

from companion.import_response import validate_record


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input.wav").write_bytes(b"hash test fixture, not audio evidence")
        digest = hashlib.sha256((self.root / "input.wav").read_bytes()).hexdigest()
        self.manifest = {"files": {"reference": {"path": "input.wav", "sha256": digest}}}
        self.record = dict(condition_id="reference", input_sha256=digest, model="test-fixture",
                           model_revision="test", raw_response=" \n<tool_callBROKEN\n ",
                           prompt="test prompt", settings={"do_sample": False}, seed=1,
                           timestamp="2026-10-13T00:00:00Z")

    def test_malformed_raw_is_preserved_without_execution(self):
        result = validate_record(self.record, self.manifest, self.root)
        self.assertEqual(result["record"]["raw_response"], self.record["raw_response"])
        self.assertIsNone(result["executed_action"])
        self.assertTrue(result["human_review_required"])
        self.assertFalse(result["inspection"]["would_accept_native_call"])

    def test_wrong_record_hash_blocks_attachment(self):
        self.record["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_record(self.record, self.manifest, self.root)

    def test_changed_audio_blocks_attachment(self):
        (self.root / "input.wav").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_record(self.record, self.manifest, self.root)

    def test_missing_or_invalid_provenance_rejected(self):
        for key in self.record:
            record = deepcopy(self.record)
            del record[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(record, self.manifest, self.root)
        for key, value in [("condition_id", "unknown"), ("seed", True),
                           ("timestamp", "2026-10-13T00:00:00"), ("settings", {})]:
            record = dict(self.record, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(record, self.manifest, self.root)

    def test_ambiguous_clarification_requires_review(self):
        self.record["raw_response"] = "Can you elaborate?"
        result = validate_record(self.record, self.manifest, self.root)
        self.assertTrue(result["inspection"]["clarification"]["unresolved"])
        self.assertTrue(result["human_review_required"])
