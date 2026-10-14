# Companion parser change

The unchanged reference is `source/TRACE_COMPANION_TRANSFER/code/native_parser.py`.
Its SHA-256 is `7978271e1282766edbff92bffa19b13c5c40bba179186ccc9cfea8782d5ee155`.
The reference transfer, inference runner, prompt, and generation settings are unchanged.

The companion copy rejects residual native delimiter prefixes outside complete
blocks: `<tool_call`, `</tool_call`, `<|tool_call`, and `<|/tool_call`. This closes
the case where a valid block followed by a malformed or mixed-format call marker
was accepted. A rejected proposal creates no simulated timer. Ordinary prose
accompanying a valid single call still passes the existing rules. No malformed
response is repaired. The parsed record also includes `raw_response`, preserving
the complete supplied string, including whitespace and termination tokens.

`inspect_response()` reports a proposal only. `score_native()` performs an
in-memory simulation with separately supplied evaluator ground truth; neither
function calls a real timer, model, API, scheduler, or device. Scripted examples
remain scripted examples, and historical responses remain historical evidence.

This targeted change is not a comprehensive semantic validator. Clarification
and mixed-behavior detection remain English-language heuristics. Literal native
delimiter examples in accompanying prose are conservatively treated as call
attempts. Evaluator ground truth must remain separate from model input; missing
duration labels require an explicit evaluation assumption and audio review.

Run the standard-library CPU checks from the project root:

```bash
python -B -m unittest discover -s tests -p test_parser.py -v
```

The tests include all twelve original fixtures plus malformed tails, mixed
formats, stray closing markers, ordinary prose, clarification, exact raw-text
preservation, and proposal-only inspection.
