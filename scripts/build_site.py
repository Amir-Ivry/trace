"""Build static HTML and copy exact assets. No network, models, or JavaScript build."""
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.scoring import score_document
from companion.gallery import read_observation, observed_fields

TRANSFER = ROOT / "source/TRACE_COMPANION_TRANSFER"
SITE = ROOT / "site"


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def copy(source, destination):
    dest = SITE / destination
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def build():
    content = read_json("configs/site_content.json")
    gallery = read_json("configs/gallery.json")
    manifest = read_json("prepared/audio_manifest.json")
    historical = read_json("evidence/historical.json")
    fixtures = read_json("configs/scoring_examples.json")
    scores = score_document(fixtures)
    if historical["evidence_status"] != "historical_actual":
        raise ValueError("Archived evidence must retain its historical status")
    cards, hashes = [], []
    frozen = read_json("evidence/gallery/run_manifest.json")
    observed = 0
    for index, condition in enumerate(gallery["conditions"], 1):
        key = condition["id"]
        audio = ROOT / condition["audio"]
        info = manifest["files"][key]
        if hashlib.sha256(audio.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Audio checksum mismatch: {key}")
        copy(audio, f"audio/{key}.wav")
        status = condition["evidence_status"]
        if status == "pending_observation":
            if condition["response"] is not None or condition["proposed_action"] is not None:
                raise ValueError("Pending card contains an unverified response or action")
            fields = '<dl class="response-fields"><dt>Agent response</dt><dd class="pending">Agent response not yet generated</dd><dt>Proposed action</dt><dd class="pending">Pending agent response</dd></dl>'
        elif status == "observed_first_response":
            evidence = read_observation(ROOT, condition, manifest, frozen)
            fields = observed_fields(evidence["raw"], evidence["inspection"], key)
            for kind, source in evidence["files"].items():
                suffix = "loader" if kind == "raw" else kind
                copy(source, f"data/gallery/{key}_{suffix}.json")
            (SITE / f"data/gallery/{key}_raw_response.txt").write_text(evidence["raw"]["raw_response"])
            observed += 1
        else:
            raise ValueError(f"Unsupported gallery evidence status: {status}")
        cards.append(f'''<article class="audio-card" data-evidence-status="{escape(status)}" data-condition="{escape(key)}">
          <span class="condition">Condition {index:02d}</span><h3>{escape(condition['title'])}</h3>
          <p class="description">{escape(condition['description'])}</p>
          <audio controls preload="metadata" aria-label="{escape(condition['title'])}"><source src="audio/{escape(key)}.wav" type="audio/wav"><a href="audio/{escape(key)}.wav">Download WAV</a></audio>
          {fields}
          <p class="observation">{escape(condition['observation'])}</p></article>''')
        hashes.append(f'<li>{escape(condition["title"])}: <code>{info["sha256"]}</code></li>')
    rows, logs = [], []
    columns = [("goal_attained", "Goal attained"), ("violation", "Violation"),
               ("recovery", "Recovery completed / initiated"), ("task_success", "Task success"),
               ("extra_user_turns", "Extra user turns")]
    for row, case in zip(scores["rows"], fixtures["cases"]):
        cells = []
        for key, label in columns:
            if key == "recovery":
                value = f'{row["recovery_completed"]} / {row["recovery_initiated"]}'
                css = ""
            elif key == "extra_user_turns":
                value, css = str(row[key]), ""
            else:
                value = "Yes" if row[key] else "No"
                css = "warning" if key == "violation" and row[key] else "yes" if row[key] else "no"
            cells.append(f'<td data-label="{label}" class="{css}">{value}</td>')
        rows.append(f'<tr data-scenario="{row["id"]}"><td><span class="scenario"><small>{row["id"]}</small>{escape(row["description"])}</span><span class="assumption">{escape(case["assumption"])}</span></td>{"".join(cells)}</tr>')
        logs.append(f'<h3>{escape(case["id"])} · {escape(case["description"])}</h3><pre><code>{escape(json.dumps(case["events"], indent=2, ensure_ascii=False))}</code></pre>')
    citation = ("@unpublished{ivry2026trace,\n"
                "  title = {" + content["title"] + "},\n"
                "  author = {Ivry, Amir and Chang, Kai-Wei and Zhang, Lin and Gannot, Sharon and Busso, Carlos},\n"
                "  year = {2026},\n  note = {Submitted to ICASSP 2027}\n}")
    replacements = {
        "STATUS": escape(content["status"]), "TITLE": escape(content["title"]),
        "GITHUB_URL": escape(content["github_repository_url"], quote=True),
        "PAGE_URL": escape(content["intended_page_url"], quote=True),
        "SUMMARY": escape(content["summary"]), "REVISION": escape(gallery["model_revision"]),
        "AUTHORS": " ".join(f'<span>{escape(a["name"])}<sup>{i}</sup></span>' for i, a in enumerate(content["authors"], 1)),
        "AFFILIATIONS": " ".join(f'<span><sup>{i}</sup> {escape(a["affiliation"])}</span>' for i, a in enumerate(content["authors"], 1)),
        "STEPS": "".join(f'<li><span class="letter" aria-hidden="true">{s["letter"]}</span><h3>{escape(s["name"])}</h3><p>{escape(s["description"])}</p></li>' for s in content["trace_steps"]),
        "CARDS": "\n".join(cards), "HASHES": "\n".join(hashes),
        "RAW_RESPONSE": escape(historical["raw_response"]), "SCORE_ROWS": "\n".join(rows),
        "SCRIPTED_LOGS": "\n".join(logs), "BIBTEX": escape(citation),
        "GALLERY_NOTE": (f"<strong>{observed} actual first responses.</strong> "
                         + ("Each response belongs to the exact WAV above. " if observed else "The fixed requests are prepared; model outputs remain pending. ")
                         + "Human interpretation and audio-cue review remain pending. These first responses do not establish completed tasks or an agent success rate."),
        "RESPONSE_IMPORT_NOTE": (f"{observed} of 3 first-response records are available. "
                                 "Saved records are checked against the exact audio hash, model and processor revisions, generic prompt, resolved settings, seed 1234, and pinned runtime. "
                                 "Human-review flags remain pending. No parsed timer proposal is executed, and first responses are not passed through the conversation scorer."),
        "FOOTER_EVIDENCE": f"Paper companion · {observed} actual first responses",
    }
    page = (ROOT / "templates/index.html").read_text()
    for key, value in replacements.items():
        page = page.replace(f"@@{key}@@", value)
    if "@@" in page:
        raise ValueError("Unresolved template field")
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(page)
    copy(ROOT / "reference/paper.pdf", "paper.pdf")
    copy(TRANSFER / "assets/speech/dev-real/anonymous0_24.wav", "audio/untouched_source.wav")
    copy(TRANSFER / "historical/pilot_recording.wav", "audio/pilot_recording.wav")
    for src, dest in [
        (ROOT / "prepared/audio_manifest.json", "data/audio_manifest.json"),
        (ROOT / "configs/gallery.json", "data/gallery.json"),
        (ROOT / "configs/scoring_examples.json", "data/scoring_examples.json"),
        (ROOT / "evidence/historical.json", "data/historical.json"),
        (ROOT / "evidence/gallery/run_manifest.json", "data/gallery/run_manifest.json"),
        (ROOT / "CREDITS_AND_LICENSES.txt", "data/CREDITS_AND_LICENSES.txt"),
        (TRANSFER / "historical/raw_response.txt", "data/historical_raw_response.txt"),
        (TRANSFER / "historical/rendered_prompt.txt", "data/historical_prompt.txt"),
        (TRANSFER / "historical/generation_config.json", "data/historical_generation_config.json"),
        (TRANSFER / "configs/native_prompt.txt", "data/native_prompt.txt"),
        (TRANSFER / "configs/generation_config.json", "data/generation_config.json"),
        (TRANSFER / "assets/speech/CC0_1.0.txt", "licenses/speech_CC0_1.0.txt"),
        (TRANSFER / "assets/noise/CC0_1.0.txt", "licenses/noise_CC0_1.0.txt"),
    ]:
        copy(src, dest)
    (SITE / "data/scripted_scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    audio_hashes = {str(p.relative_to(SITE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((SITE / "audio").glob("*.wav"))}
    (SITE / "data/audio_hashes.json").write_text(json.dumps(audio_hashes, indent=2) + "\n")
    print(f"Built {SITE / 'index.html'}; {observed} observed and {3-observed} pending cards, 1 archived response, 5 scripted rows.")


if __name__ == "__main__":
    build()
