"""Native Qwen timer proposals: parsing and in-memory simulation only.

No device, scheduler, API, or real timer is called here. The inference CLI only
uses inspect_response(), which describes a proposed call without executing it.
score_native() is the historical evaluator and is NEVER used as model input.
"""
import json
import re

def duration(args):
    if not isinstance(args, dict) or set(args) != {'hours', 'minutes', 'seconds'}:
        raise ValueError('exactly_three_required_fields')
    if any(type(v) is not int or v < 0 for v in args.values()):
        raise ValueError('nonnegative_integers_required')
    total = args['hours'] * 3600 + args['minutes'] * 60 + args['seconds']
    if total <= 0:
        raise ValueError('positive_total_required')
    return total

def strict_json(text):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError('duplicate_json_key')
            obj[k] = v
        return obj
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite_json')))

def response_text(raw):
    # Only remove documented termination tokens; retain call delimiters.
    text = raw.strip()
    for token in ['<|im_end|>', '<|endoftext|>', '<|end|>']:
        if text.endswith(token):
            text = text[:-len(token)].rstrip()
    # Reasoning is not a request addressed to the user.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    # A truncated analysis block also cannot create an action or clarification.
    text = re.sub(r'<think>.*$', '', text, flags=re.S).strip()
    return text

def proposed_calls(raw, model):
    text = response_text(raw)
    start, end = ('<tool_call>', '</tool_call>') if model == 'qwen' else ('<|tool_call|>', '<|/tool_call|>')
    attempted = any(t in text for t in (start, end, '<tool_call', '<|tool_call', 'functools['))
    blocks = re.findall(re.escape(start) + '(.*?)' + re.escape(end), text, flags=re.S)
    residue = re.sub(re.escape(start) + '.*?' + re.escape(end), '', text, flags=re.S).strip()
    calls = []
    error = None
    if attempted and (text.count(start) != len(blocks) or text.count(end) != len(blocks) or not blocks):
        error = 'malformed_call_delimiters'
    try:
        for block in blocks:
            obj = strict_json(block)
            if model == 'phi':
                if not isinstance(obj, list):
                    raise ValueError('phi_native_call_requires_json_array')
                calls.extend(obj)
            else:
                calls.append(obj)
    except (ValueError, TypeError) as e:
        error = 'malformed_call_json:' + str(e)
    # Plain JSON/code proposing a call is invalid, never repaired or executed.
    outside_attempt = bool(re.search(r'\bset_timer\s*\(', residue) or re.search(r'"(?:name|function)"\s*:', residue))
    if outside_attempt:
        attempted, error = True, ('additional_call_outside_native_delimiters' if blocks else 'call_outside_native_delimiters')
    return dict(attempted=attempted, raw_blocks=blocks, raw_proposal_text=text if attempted else None, calls=calls, error=error, accompanying_text=residue)

CLARIFICATION_RULES = [
    ('repeat_request', r'\b(?:could|can|would|will) you\b[^.!?]{0,70}\b(?:repeat|restate|say (?:that|it) again|clarify (?:your|the) request)\b'),
    ('please_repeat', r'\bplease\b[^.!?]{0,35}\b(?:repeat|restate|say (?:that|it) again)\b'),
    ('duration_question', r'\b(?:how long|what duration|which duration|what timer duration)\b[^.!?]{0,110}\?'),
    ('component_question', r'\bhow many (?:hours|minutes|seconds)\b[^.!?]{0,100}\?'),
    ('provide_duration', r'\b(?:please|could you|can you|would you)\b[^.!?]{0,60}\b(?:specify|provide|tell me|clarify|confirm)\b[^.!?]{0,80}\b(?:duration|how long|hours|minutes|seconds)\b'),
    ('confirm_duration_question', r'\b(?:did you (?:say|mean)|do you mean|should i set|would you like (?:me to set|a timer for)|is (?:that|it))\b[^.!?]{0,100}\b(?:seconds?|minutes?|hours?)\b[^.!?]{0,50}\?'),
]

def clarification(text):
    text = response_text(text)
    for name, pattern in CLARIFICATION_RULES:
        if re.search(pattern, text, re.I):
            return dict(qualifying=True, unresolved=False, reason=name)
    # Explicit non-information offers are not clarification. Other questions need review.
    remainder = re.sub(r'(?:is there anything else|can i help (?:you )?with anything else|would you like (?:anything else|to set another timer))[^.!?]*\?', '', text, flags=re.I)
    unresolved = any(mark in remainder for mark in ['?', '？', '؟']) or bool(re.search(r'\b(?:repeat|restate|clarify|confirm|provide|specify|tell me|let me know|say again|need (?:more|the|your))\b', remainder, re.I))
    return dict(qualifying=False, unresolved=unresolved, reason='ambiguous_information_request' if unresolved else 'no_qualifying_information_request')

def simulate(parsed, gold):
    result = dict(virtual_time=0, initial_state={'timers': []}, validation={'valid': False, 'reason': 'no_call'}, execution_event=None, resulting_state={'timers': []}, exact_match=None)
    if not parsed['attempted']:
        return result
    reason = parsed['error']
    if not reason and len(parsed['calls']) != 1:
        reason = 'multiple_calls' if len(parsed['calls']) > 1 else 'empty_call_attempt'
    if not reason:
        call = parsed['calls'][0]
        if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
            reason = 'invalid_call_object_fields'
        elif call['name'] != 'set_timer':
            reason = 'unsupported_function'
        else:
            try:
                total = duration(call['arguments'])
            except ValueError as e:
                reason = str(e)
    if reason:
        result['validation']['reason'] = reason
        return result
    timer = dict(duration_seconds=total, created_at=0, expires_at=total)
    result.update(validation={'valid': True, 'reason': 'valid_single_call'}, execution_event={'type': 'timer_created', 'timer': timer}, resulting_state={'timers': [timer]}, exact_match=total == gold)
    return result

def score_native(raw,gold):
    parsed=proposed_calls(raw,'qwen');request=clarification(parsed['accompanying_text']);sim=simulate(parsed,gold)
    if gold is None:sim['exact_match']=None
    sim['gold_duration_seconds']=gold
    accepted=sim['validation']['valid'];chosen=sim['execution_event']['timer']['duration_seconds'] if accepted else None
    if accepted:label='unsupported_action' if gold is None else 'correct_action' if sim['exact_match'] else 'wrong_action';reason='accepted simulated timer'
    elif parsed['attempted']:label='other';reason=sim['validation']['reason']
    elif request['qualifying']:label='clarification';reason=request['reason']
    else:label='other';reason=request['reason'] if request['unresolved'] else 'no_execution_or_clear_information_request'
    uncertainty=bool(re.search(r"\b(?:unsure|uncertain|not sure|cannot determine|can't determine|might|maybe)\b",parsed['accompanying_text'],re.I))
    mixed=accepted and (request['qualifying'] or request['unresolved'] or uncertainty)
    return dict(category=label,duration_seconds=chosen,exact_match=sim['exact_match'],gold_duration_seconds=gold,parsed=parsed,simulation=sim,clarification=request,mixed_behavior=mixed,requests_clarification=request['qualifying'],reason=reason,review_required=mixed or request['unresolved'] or (label=='other' and not parsed['attempted']))

def inspect_response(raw):
    parsed = proposed_calls(raw, "qwen")
    validation = simulate(parsed, None)
    accepted = validation["validation"]["valid"]
    proposed_seconds = validation["execution_event"]["timer"]["duration_seconds"] if accepted else None
    return {
        "status": "proposal_inspection_only",
        "real_action_executed": False,
        "parsed": parsed,
        "clarification": clarification(parsed["accompanying_text"]),
        "would_accept_native_call": accepted,
        "validation": validation["validation"],
        "proposed_duration_seconds": proposed_seconds,
        "correctness": "not_evaluated",
    }
