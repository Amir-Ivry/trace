# TRACE paper companion

The companion for **Voice Agents Under Acoustic Stress: From Signal Degradation to Interaction and Action**, submitted to ICASSP 2027. The timer example applies TRACE to a different task from the paper’s delivery example. This is a small reproducible example, not a benchmark or an empirical validation of TRACE.

The page presents three fixed listening conditions with saved first-response evidence, one separate archived actual response, and five fictional conversations scored from explicit events. Generation status, provenance validation, human interpretation, and audio-cue review are separate. See `RESULTS_GALLERY.md` for observed responses, attempt counts, and any missing records. No real timer actions were executed.

## Website and code

- [Audio examples and paper](https://amir-ivry.github.io/trace/)
- [GitHub repository](https://github.com/Amir-Ivry/trace)

To build the complete GitHub Pages directory and downloadable examples, run:

```bash
python3 -B scripts/prepare_github_pages.py
```

This uses only Python’s standard library. GitHub Pages serves the generated
`docs/` directory; see `PUBLISHING.md` for setup and updates.

## Open the preview

In the preview ZIP, extract first and open `TRACE_COMPANION_PREVIEW/site/index.html`; audio, PDF, and code download are included. No server, build framework, CDN, database, or credentials are needed.

The code ZIP contains the source rather than the rendered page. From its extracted root, run `python -B scripts/build_site.py` and then `python -B scripts/package.py` to create `site/index.html`, its assets, and the code download. These build commands use only the Python standard library. Open the generated `site/index.html` directly.

Optional local HTTP preview, from the project root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Visit `http://localhost:8000`. For Remote-SSH, forward port 8000 through VS Code and open the forwarded localhost URL. Stop the server with Ctrl-C. This does not deploy the site. Native controls play one recording at a time when JavaScript is enabled. All core content is static and remains visible without JavaScript.

## CPU installation and two examples

Use Python 3.10 or newer. The scorer, response validator, and unit tests use only the standard library. Audio preparation needs NumPy and SciPy. Reuse compatible existing CPU packages, or create a project-local environment:

```bash
python3 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.txt
```

Use that environment’s Python for these two commands, from the project/extracted code root:

```bash
python -B scripts/prepare_audio.py --output outputs/reproduced_audio
python -B -m companion.scoring --output outputs/scripted_scores.json
```

If using the local environment without activating it, replace `python` with `.venv-cpu/bin/python`. The implementation was checked with system Python 3.12.3, NumPy 1.26.4, and SciPy 1.11.4. The handoff used different versions; all three reproduced WAVs matched the shipped bytes and decoded PCM exactly. The preparation receipt records runtime versions. A SciPy warning about an unrecognized non-data WAV chunk in the source noise is harmless to its decoded PCM; source bytes and reproduced outputs are checked.

New preparation outputs go outside the immutable transfer and `prepared/`. Differing existing audio or manifests are refused; choose a fresh output directory to investigate discrepancies without replacing evidence. Use `-B` to avoid writing Python caches into the transfer.

## Expected scripted results

These are per-scenario teaching examples, not measured model performance. Recovery is shown as **completed / initiated**. The implementation computes these values from events and never reads a case’s `expected` fields.

| Scenario | Goal | Violation | Recovery | Task success | Extra user turns |
|---|---|---|---|---|---|
| S1 Correct timer without clarification | Yes | No | 0 / 0 | Yes | 0 |
| S2 Clarify, receive duration, create correct timer | Yes | No | 1 / 1 | Yes | 1 |
| S3 Wrong timer, correction, cancellation, replacement | Yes | Yes | 1 / 1 | No | 1 |
| S4 Clarification followed by user refusal | No | No | 0 / 1 | No | 1 |
| S5 Correct duration guessed before it is obtained | Yes | Yes | 0 / 0 | No | 0 |

The goal is exactly one active 540-second timer at the final event, within 30 elapsed seconds and five extra user turns. A wrong executed duration or execution before required information is available is a retained violation. A supplied value different from the stipulated goal does not authorize guessing the correct value. Task success requires goal attainment and no violations. A proposal has no state effect. Refusal counts one extra user turn. Fictional availability is stipulated in these scenarios; it is not an automatic assessment of what remains audible in a recording.

`companion/scoring.py` supports one initial instruction at t=0, clarification, user reply/correction/refusal, timer proposal/creation/cancellation, and optional `conversation_end`. Times must be finite and nondecreasing. Created timers expire at `created_at + duration_seconds`. The final event determines the evaluation time; use `conversation_end` to declare a later endpoint. There is at most one open recovery episode. A reply requires one; a correction opens one if needed. A correctly resumed simulated task after receiving the missing/corrected value completes it. Nested clarification requests, duplicate timer IDs, cancellation of inactive timers, unsupported fields/events, events after refusal/end, and invalid values are explicit errors. Exceeded time or turn limits appear in `budget_exceeded`, set `within_budget: false`, and prevent goal attainment and task success. They do not count as prohibited-action violations under this task's declared rules. There is no implicit real-time clock or service.

## Audio and adaptation

The untouched speech is Timers and Such v1.0 `dev-real/anonymous0_24.wav`: 16 kHz mono, 28,989 samples. The historical automatic estimate for “nine” is [15461, 18682), zero-based and end-exclusive. This is not human phonetic annotation or proof that all duration cues have been removed.

The full released Holman nc4 machinery event is averaged to mono, polyphase-resampled, aligned by its energy median, and calibrated to −10 dB SNR over that number interval. The event spans [14934, 26934), including part of “minutes.” SNR is local energy ratio, not whole-recording SNR or physical sound pressure. The silenced condition has no SNR. All three files use gain 0.9080760073292201 and one final rounding to PCM16, without dithering or separate normalization. The gallery model inputs are these exact exported WAVs. The original float historical input is a separate file with gain 0.9708313163778691.

To replace a recording or segment, make a new project-owned asset/metadata directory outside `source/` and `prepared/`. Copy the preparation entry point and point its speech/noise metadata and paths to the new assets; update checksums, format checks, marked sample interval, labels, and output destination. Keep the source assets untouched. Choose boundaries and acoustics before seeing model outputs, check the full event fits, apply one shared gain and quantization, and review what information remains. Do not silently change the shipped three-condition recipe or run alignment/model downloads.

To adapt task rules, copy the fixture JSON, set the goal/count/budgets and explicit availability assumptions, then update `companion/scoring.py` for the task’s allowed/prohibited actions and recovery definition. The descriptive `violation_rules` strings in the fixture explain the two implemented rules; editing strings alone does not change scoring. Add a hand-calculated case for each changed rule. For tasks beyond timers, replace the inert state transition logic. Supply behavior logs with explicit executed events; parsed proposals are not execution evidence. Pass an alternate fixture file as the scorer’s positional argument.

## Saved first responses and optional GPU reproduction

`configs/future_response_template.json` is deliberately incomplete. Populate it only from a real saved response with condition ID, exact input SHA-256, model/revision, complete raw response (including malformed syntax), exact prompt/settings, seed, and timezone-bearing timestamp. Validate without inference:

```bash
python -B -m companion.import_response actual_record.json --output reviewed_candidate.json
```

The validator checks the record against both the manifest and actual WAV bytes. A mismatched hash blocks attachment. It preserves the supplied record, reports proposal inspection only, requires human review, and never fills a gallery card or infers action execution. Model identity fields are provenance supplied by the record, not cryptographic authentication. Clarification regexes are unvalidated heuristics. The builder accepts completed observations only when their raw, normalized, and validated records agree with the frozen model/processor revisions, prompt, settings, seed, runtime fingerprints, and audio identity. It retains `human_review_required: true` while displaying the actual response. The corrected companion parser governs the proposed action; the loader’s legacy inspection is retained only as provenance. Complete natural-language responses and exact tool syntax remain inspectable. First responses never pass through the conversation scorer.

GPU dependencies remain optional. `GPU_REPRODUCTION.md` describes Python 3.10.20, the exact package pins, the 96 GB RTX PRO 6000 requirement, and `scripts/reproduce_gpu.py`. Neither CPU command imports a model or calls the GPU entry point. The public download includes the unchanged inference CLI and its required configuration, but no weights, environments, credentials, or private scheduler receipts.

## Build, test, and package

```bash
python -B -m unittest discover -s tests -v
python -B scripts/build_site.py
python -B scripts/package.py
```

The builder validates the displayed responses against the saved provenance. The original research workspace also retains its full verification and browser receipts. Build output is static `site/`; templates, CSS, and JS have no external build dependencies. Public packages include enough selected transfer material to prepare audio, run tests, score fixtures, inspect evidence, and rebuild the page. The internal full-transfer verifier applies only to the complete immutable transfer in the local project, not the public allowlisted subset.

`dist/TRACE_COMPANION_PREVIEW.zip` contains the playable site and its working code download. `dist/TRACE_COMPANION_CODE.zip` contains selected CPU tools/assets, tests, documentation, provenance, licensing texts, and saved evidence states. No archive contains itself. Internal handoff instructions, environment receipts, original historical full traces with host paths, caches, weights, credentials, and private scheduler receipts are excluded. The optional inference entry point and required pinned configurations are included separately from the CPU path. `dist/SHA256SUMS.txt` records final archive hashes.

## Evidence, credits, and pending work

The submitted paper and full transfer remain unchanged. The archived actual response is dated 9 October 2026; the display preserves all raw syntax. Its timer call is a proposal, while the saved historical trace separately records an inert simulation. `evidence/historical.json` is a clearly labeled public extract and records the original trace’s SHA-256. Browser playback is a technical check, not human listening or a perceptual validation.

See `CREDITS_AND_LICENSES.txt` and the two included CC0 1.0 texts for Timers and Such (Loren Lugosch) and Sound of Work (ldezem / Ore.e Refineries). The CC0 audio licenses do not license the code or submitted PDF. Transferred parser/audio-code provenance is identified separately; no whole-archive code license is invented.

Human interpretation, input/boundary review, and residual-cue review remain pending. The three completed generations are documented in `RESULTS_GALLERY.md`. No code reuse license has been selected; the audio asset licenses apply only to the audio. No robustness percentages, agent rankings, or combined TRACE score can be inferred from these examples.
