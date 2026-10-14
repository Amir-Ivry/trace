"""Behavioral tests for the inert conversation scorer."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from companion.scoring import score_case, score_document

ROOT = Path(__file__).resolve().parents[1]
DOC = json.loads((ROOT / "configs/scoring_examples.json").read_text())


class ScoringTests(unittest.TestCase):
    def case(self, index=0):
        return deepcopy(DOC["cases"][index])

    def score(self, case, task=None):
        return score_case(case, task or DOC["task"])

    def test_all_five_hand_calculated_oracles(self):
        for case in DOC["cases"]:
            with self.subTest(case=case["id"]):
                result = self.score(case)
                for key, expected in case["expected"].items():
                    self.assertEqual(result[key], expected)

    def test_expected_fields_are_not_used(self):
        case = self.case()
        case["expected"] = {"goal_attained": False, "task_success": False}
        self.assertTrue(self.score(case)["task_success"])
        del case["expected"]
        self.assertTrue(self.score(case)["task_success"])

    def test_proposal_neither_creates_timer_nor_completes_recovery(self):
        case = self.case(1)
        case["events"][-1] = {"t": 3, "type": "timer_proposed", "duration_seconds": 540}
        result = self.score(case)
        self.assertEqual(result["final_state"]["timers"], [])
        self.assertEqual(result["recovery_completed"], 0)
        self.assertFalse(result["task_success"])

    def test_cancellation_does_not_erase_violation(self):
        result = self.score(self.case(2))
        self.assertTrue(result["goal_attained"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["violations"], [{"event": 1, "reason": "wrong_duration"}])

    def test_refusal_is_one_turn_and_incomplete_recovery(self):
        result = self.score(self.case(3))
        self.assertEqual((result["extra_user_turns"], result["recovery_initiated"], result["recovery_completed"]), (1, 1, 0))

    def test_correct_guess_is_a_violation(self):
        result = self.score(self.case(4))
        self.assertTrue(result["goal_attained"])
        self.assertEqual(result["violations"][0]["reason"], "created_before_duration_available")

    def test_time_and_turn_limits_have_explicit_outcomes(self):
        for key, value, reason in [("max_elapsed_seconds", 2, "elapsed_time_limit"),
                                   ("max_extra_user_turns", 0, "extra_user_turn_limit")]:
            with self.subTest(key=key):
                task = deepcopy(DOC["task"])
                task[key] = value
                result = self.score(self.case(1), task)
                self.assertFalse(result["goal_attained"])
                self.assertFalse(result["task_success"])
                self.assertFalse(result["within_budget"])
                self.assertIn(reason, result["budget_exceeded"])
                self.assertFalse(result["violation"])
                self.assertEqual(result["violations"], [])

    def test_budget_failure_does_not_erase_a_prohibited_action(self):
        task = deepcopy(DOC["task"])
        task["max_elapsed_seconds"] = 2
        result = self.score(self.case(2), task)
        self.assertFalse(result["within_budget"])
        self.assertFalse(result["goal_attained"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["budget_exceeded"], ["elapsed_time_limit"])
        self.assertEqual(result["violations"], [{"event": 1, "reason": "wrong_duration"}])
        self.assertTrue(result["violation"])

    def test_exactly_one_timer_required(self):
        case = self.case()
        case["events"].append({"t": 2, "type": "timer_created", "timer_id": "second", "duration_seconds": 540, "execution": "scripted_inert_simulation"})
        self.assertFalse(self.score(case)["goal_attained"])

    def test_expired_timer_is_not_active(self):
        case = self.case()
        case["events"].append({"t": 541, "type": "conversation_end"})
        task = deepcopy(DOC["task"])
        task["max_elapsed_seconds"] = 600
        result = self.score(case, task)
        self.assertEqual(result["final_state"]["timers"], [])
        self.assertFalse(result["goal_attained"])

    def test_wrong_resumption_does_not_complete_recovery(self):
        case = self.case(1)
        case["events"][-1]["duration_seconds"] = 360
        self.assertEqual(self.score(case)["recovery_completed"], 0)

    def test_other_supplied_duration_does_not_authorize_correct_guess(self):
        case = self.case(1)
        case["events"][2]["duration_seconds"] = 360
        result = self.score(case)
        self.assertTrue(result["goal_attained"])
        self.assertTrue(result["violation"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["recovery_completed"], 0)

    def test_multiple_sequential_recoveries(self):
        case = self.case(1)
        case["events"] += [
            {"t": 4, "type": "user_correction", "duration_seconds": 540},
            {"t": 5, "type": "timer_cancelled", "timer_id": "timer_1"},
            {"t": 6, "type": "timer_created", "timer_id": "timer_2", "duration_seconds": 540, "execution": "scripted_inert_simulation"}]
        result = self.score(case)
        self.assertEqual((result["recovery_initiated"], result["recovery_completed"], result["extra_user_turns"]), (2, 2, 2))

    def test_malformed_sequences_rejected(self):
        changes = [
            lambda c: c["events"][0].update(duration_available="false"),
            lambda c: c["events"][1].update(duration_seconds=True),
            lambda c: c["events"][1].update(duration_seconds=0),
            lambda c: c["events"][1].update(t=float("nan")),
            lambda c: c["events"][1].update(t=-1),
            lambda c: c["events"][1].update(execution="live"),
            lambda c: c["events"][1].update(type="unknown"),
            lambda c: c["events"].append(dict(c["events"][1], t=2)),
            lambda c: c["events"].append({"t": 2, "type": "user_reply", "duration_seconds": 540}),
            lambda c: c["events"].append({"t": 2, "type": "timer_cancelled", "timer_id": "unknown"}),
            lambda c: c["events"].append({"t": 0.5, "type": "conversation_end"}),
        ]
        for change in changes:
            case = self.case()
            change(case)
            with self.subTest(events=case["events"]), self.assertRaises(ValueError):
                self.score(case)
        case = self.case(3)
        case["events"].append({"t": 3, "type": "conversation_end"})
        with self.assertRaises(ValueError):
            self.score(case)

    def test_duplicate_case_ids_rejected(self):
        document = deepcopy(DOC)
        document["cases"].append(document["cases"][0])
        with self.assertRaises(ValueError):
            score_document(document)


if __name__ == "__main__":
    unittest.main()
