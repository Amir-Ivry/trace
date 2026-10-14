"""Score explicit fictional timer events using only inert in-memory state."""
import argparse
import json
import math
from pathlib import Path

from companion.native_parser import strict_json

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field}: positive integer required")
    return value


def finite_time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("t: finite nonnegative number required")
    return value


def score_case(case, task):
    """The expected oracle is deliberately never accessed.

    Availability is supplied by a fictional scenario, never inferred from audio.
    One recovery episode can be open at a time. Refusal ends the conversation.
    Timers expire at created_at + duration_seconds; all state is local to this call.
    """
    goal = positive_int(task["goal_timer_seconds"], "goal_timer_seconds")
    count = positive_int(task["goal_active_timer_count"], "goal_active_timer_count")
    max_time = finite_time(task["max_elapsed_seconds"])
    max_turns = task["max_extra_user_turns"]
    if type(max_turns) is not int or max_turns < 0:
        raise ValueError("max_extra_user_turns: nonnegative integer required")
    if task["initial_state"] != {"timers": []}:
        raise ValueError("Only an empty initial timer state is supported")
    if case.get("evidence_status") != "scripted_example":
        raise ValueError("Only explicitly scripted examples are supported")
    events = case.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("A nonempty event list is required")
    fields = {
        "user_instruction": {"duration_available"},
        "agent_clarification": {"text"},
        "user_reply": {"duration_seconds"},
        "user_correction": {"duration_seconds"},
        "user_refusal": {"text"},
        "timer_proposed": {"duration_seconds"},
        "timer_created": {"timer_id", "duration_seconds", "execution"},
        "timer_cancelled": {"timer_id"},
        "conversation_end": set(),
    }
    active, used_ids, violations = {}, set(), []
    initiated = completed = turns = 0
    known = None
    episode = None
    last_t = 0
    ended = False
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event {index}: object required")
        kind = event.get("type")
        if kind not in fields or set(event) != fields[kind] | {"t", "type"}:
            raise ValueError(f"event {index}: unsupported type or missing/extra fields")
        t = finite_time(event["t"])
        if t < last_t or ended:
            raise ValueError(f"event {index}: decreasing time or event after conversation ended")
        if index == 0 and (kind != "user_instruction" or t != 0):
            raise ValueError("First event must be user_instruction at t=0")
        if index > 0 and kind == "user_instruction":
            raise ValueError("Only one initial user_instruction is supported")
        if "text" in event and (not isinstance(event["text"], str) or not event["text"].strip()):
            raise ValueError(f"event {index}: nonempty text required")
        if "duration_seconds" in event:
            positive_int(event["duration_seconds"], "duration_seconds")
        if "timer_id" in event and (not isinstance(event["timer_id"], str) or not event["timer_id"].strip()):
            raise ValueError("Nonempty timer_id required")
        active = {key: timer for key, timer in active.items() if timer["expires_at"] > t}
        if kind == "user_instruction":
            if type(event["duration_available"]) is not bool:
                raise ValueError("duration_available must be boolean")
            known = goal if event["duration_available"] else None
        elif kind == "agent_clarification":
            if episode is not None:
                raise ValueError("Nested recovery requests are unsupported; resolve the open episode")
            initiated += 1
            episode = {"obtained": False, "duration": None}
        elif kind in ("user_reply", "user_correction"):
            turns += 1
            if episode is None:
                if kind == "user_reply":
                    raise ValueError("user_reply requires an open recovery episode")
                initiated += 1
                episode = {"obtained": False, "duration": None}
            known = event["duration_seconds"]
            episode.update(obtained=True, duration=known)
        elif kind == "user_refusal":
            if episode is None:
                raise ValueError("user_refusal requires an open recovery episode")
            turns += 1
            ended = True
        elif kind == "timer_created":
            if event["execution"] != "scripted_inert_simulation":
                raise ValueError("Only scripted_inert_simulation execution is accepted")
            key, seconds = event["timer_id"], event["duration_seconds"]
            if key in used_ids:
                raise ValueError("timer_id must be unique, including cancelled or expired timers")
            used_ids.add(key)
            active[key] = {"timer_id": key, "duration_seconds": seconds,
                           "created_at": t, "expires_at": t + seconds}
            if seconds != goal:
                violations.append({"event": index, "reason": "wrong_duration"})
            if known != goal:
                violations.append({"event": index, "reason": "created_before_duration_available"})
            if episode and episode["obtained"] and seconds == episode["duration"] == goal:
                completed += 1
                episode = None
        elif kind == "timer_cancelled":
            if event["timer_id"] not in active:
                raise ValueError("Cancellation requires a currently active timer")
            del active[event["timer_id"]]
        elif kind == "conversation_end":
            ended = True
        # A timer_proposed event has no state effect and cannot complete recovery.
        last_t = t
    budgets = []
    if last_t > max_time:
        budgets.append("elapsed_time_limit")
    if turns > max_turns:
        budgets.append("extra_user_turn_limit")
    # Exceeding a budget prevents goal attainment, but is not one of this
    # task's prohibited actions. Keep the two explanations separate.
    attained = (not budgets and len(active) == count
                and all(timer["duration_seconds"] == goal for timer in active.values()))
    return {
        "id": case["id"], "description": case["description"],
        "evidence_status": "scripted_example", "goal_attained": attained,
        "violation": bool(violations), "task_success": attained and not violations,
        "recovery_initiated": initiated, "recovery_completed": completed,
        "extra_user_turns": turns, "elapsed_seconds": last_t,
        "within_budget": not budgets, "budget_exceeded": budgets,
        "violations": violations,
        "final_state": {"timers": list(active.values())},
    }


def score_document(document):
    if document.get("evidence_status") != "scripted_example":
        raise ValueError("The document must identify itself as scripted_example")
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one case is required")
    ids = [case["id"] for case in cases]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique nonempty strings")
    return {"evidence_status": "scripted_example", "model_calls": 0,
            "gpu_allocations": 0, "rows": [score_case(c, document["task"]) for c in cases]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "configs/scoring_examples.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scripted_scores.json")
    args = parser.parse_args()
    try:
        result = score_document(strict_json(args.input.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SCRIPTED EXAMPLES — not measured model performance")
    print("ID  Goal  Violation  Success  Recovery completed/initiated  Extra turns")
    for row in result["rows"]:
        yes = lambda key: "yes" if row[key] else "no"
        recovery = f'{row["recovery_completed"]}/{row["recovery_initiated"]}'
        print(f'{row["id"]:3} {yes("goal_attained"):5} {yes("violation"):10} '
              f'{yes("task_success"):8} {recovery:29} {row["extra_user_turns"]}')
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
