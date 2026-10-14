"""CPU-only regression tests for the companion parser; no model or tool calls."""
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "companion_native_parser", ROOT / "companion" / "native_parser.py"
)
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)
FIXTURES = json.loads(
    (ROOT / "source" / "TRACE_COMPANION_TRANSFER" / "configs" /
     "native_parser_fixtures.json").read_text()
)["fixtures"]
CALL = (
    '<tool_call>{"name":"set_timer","arguments":'
    '{"hours":0,"minutes":9,"seconds":0}}</tool_call>'
)
PHI_CALL = (
    '<|tool_call|>[{"name":"set_timer","arguments":'
    '{"hours":0,"minutes":9,"seconds":0}}]<|/tool_call|>'
)


class ParserTests(unittest.TestCase):
    def assert_rejected(self, raw):
        result = PARSER.score_native(raw, 540)
        self.assertEqual(result["category"], "other")
        self.assertTrue(result["parsed"]["attempted"])
        self.assertIsNotNone(result["parsed"]["error"])
        self.assertFalse(result["simulation"]["validation"]["valid"])
        self.assertIsNone(result["simulation"]["execution_event"])
        self.assertEqual(result["simulation"]["resulting_state"], {"timers": []})
        self.assertEqual(result["parsed"]["raw_response"], raw)

    def test_valid_block_with_malformed_tail(self):
        for tail in ("<tool_callBROKEN", "</tool_callBROKEN", "<tool_call"):
            with self.subTest(tail=tail):
                self.assert_rejected(CALL + tail)

    def test_mixed_format_blocks_and_partial_markers(self):
        for extra in (PHI_CALL, '<|tool_call|>{"foo":"bar"}', "<|tool_call"):
            with self.subTest(extra=extra):
                self.assert_rejected(CALL + extra)
                self.assert_rejected(extra + CALL)

    def test_stray_closing_markers(self):
        for marker in ("</tool_call>", "<|/tool_call|>", "<|/tool_call"):
            with self.subTest(marker=marker):
                self.assert_rejected(CALL + marker)
                self.assert_rejected(marker)

    def test_phi_parser_also_rejects_mixed_qwen_marker(self):
        parsed = PARSER.proposed_calls(PHI_CALL + CALL, "phi")
        self.assertIsNotNone(parsed["error"])
        self.assertFalse(PARSER.simulate(parsed, 540)["validation"]["valid"])

    def test_normal_prose_with_valid_call(self):
        raw = "I can set that timer.\n" + CALL + "\nThe requested duration is nine minutes."
        result = PARSER.score_native(raw, 540)
        self.assertEqual(result["category"], "correct_action")
        self.assertIsNone(result["parsed"]["error"])
        self.assertEqual(result["duration_seconds"], 540)
        self.assertEqual(result["parsed"]["raw_response"], raw)

    def test_clarification_alone(self):
        result = PARSER.score_native("How long should I set the timer for?", None)
        self.assertEqual(result["category"], "clarification")
        self.assertFalse(result["parsed"]["attempted"])
        self.assertIsNone(result["simulation"]["execution_event"])

    def test_original_raw_response_is_preserved(self):
        raw = " \n" + CALL + "<|im_end|>\n "
        self.assertEqual(PARSER.proposed_calls(raw, "qwen")["raw_response"], raw)
        self.assertEqual(PARSER.inspect_response(raw)["parsed"]["raw_response"], raw)

    def test_proposal_inspection_does_not_claim_real_execution(self):
        result = PARSER.inspect_response(CALL)
        self.assertEqual(result["status"], "proposal_inspection_only")
        self.assertFalse(result["real_action_executed"])
        self.assertTrue(result["would_accept_native_call"])
        self.assertEqual(result["proposed_duration_seconds"], 540)
        self.assertEqual(result["correctness"], "not_evaluated")
        self.assertNotIn("simulation", result)


def fixture_test(item):
    def test(self):
        result = PARSER.score_native(item["raw_response"], item.get("gold_duration_seconds"))
        self.assertEqual(result["category"], item["expected"])
        if "expected_duration_seconds" in item:
            self.assertEqual(result["duration_seconds"], item["expected_duration_seconds"])
        if "expected_exact_match" in item:
            self.assertEqual(result["exact_match"], item["expected_exact_match"])
        if item.get("mixed_behavior"):
            self.assertTrue(result["mixed_behavior"])
        self.assertFalse(PARSER.inspect_response(item["raw_response"])["real_action_executed"])
        self.assertEqual(result["parsed"]["raw_response"], item["raw_response"])
    return test


if len(FIXTURES) != 12:
    raise RuntimeError("Expected the original twelve unchanged transfer fixtures.")
for index, item in enumerate(FIXTURES, 1):
    setattr(ParserTests, f"test_original_fixture_{index:02d}", fixture_test(item))


if __name__ == "__main__":
    unittest.main()
