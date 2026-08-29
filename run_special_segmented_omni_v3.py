#!/usr/bin/env python3
import argparse, json, math, os, re, subprocess, tempfile, time
from collections import defaultdict
from pathlib import Path

import run_special_segmented_omni as base

END_RE = re.compile(r"\b(truly|truth|actually|ultimate|ultimately|ending|end|final|last|twist|real|really|eventually|fate|in the end|shoot|dies?|death)\b", re.I)
BEGIN_RE = re.compile(r"\b(initially|initial|beginning|first|enter|starts?|at the beginning)\b", re.I)
CAUSE_RE = re.compile(r"\b(why|how does .* know|how .* truth|cause|because)\b", re.I)


def fconf(x):
    try:
        return float(x)
    except Exception:
        return -1.0


def quantize(x, q=4.0):
    return max(0.0, round(float(x) / q) * q)


def make_micro_clip(src, out, start, seconds):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(src),
        "-vf", "fps=8,scale=-2:480",
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "430k",
        "-maxrate", "520k", "-bufsize", "1040k",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    raw = os.path.getsize(out)
    if math.ceil(raw * 4 / 3) >= 9_500_000:
        raise RuntimeError(f"micro clip too large after Base64 expansion: {raw} bytes")


def select_windows(qs, notes, dur, seconds):
    """Question-conditioned revisit schedule.

    Revisit V1 clips that already emitted evidence for a question, then add
    deterministic bookend windows for truth/ending/beginning/causal questions.
    Windows are grouped so one API call can serve several questions.
    """
    by_q = defaultdict(list)
    for n in notes:
        ns = float(n.get("start") or 0.0)
        ne = float(n.get("end") or (ns + 32.0))
        center_start = max(0.0, (ns + ne) / 2.0 - seconds / 2.0)
        for e in n.get("evidence") or []:
            qid = str(e.get("question_id", "")).strip()
            if qid:
                by_q[qid].append((fconf(e.get("confidence")), center_start))

    schedule = defaultdict(set)
    for q in qs:
        qid, text = q["question_id"], q["question"]
        # Top two V1-localized revisits, if available.
        seen = set()
        for conf, st in sorted(by_q.get(qid, []), reverse=True):
            s = quantize(min(max(0.0, st), max(0.0, dur - seconds)))
            if s in seen:
                continue
            schedule[s].add(qid)
            seen.add(s)
            if len(seen) >= 2:
                break

        # Bookend priors are deterministic and label-free.
        if END_RE.search(text) or CAUSE_RE.search(text):
            for off in (72.0, 48.0, 24.0):
                s = quantize(max(0.0, dur - off))
                s = min(s, max(0.0, dur - seconds))
                schedule[s].add(qid)
        if BEGIN_RE.search(text):
            for s in (0.0, 24.0, 48.0):
                if s < dur:
                    schedule[quantize(min(s, max(0.0, dur - seconds)))].add(qid)

    # Ensure questions with no localized/risk windows get one beginning and one late probe.
    assigned = set().union(*schedule.values()) if schedule else set()
    for q in qs:
        qid = q["question_id"]
        if qid not in assigned:
            schedule[0.0].add(qid)
            schedule[quantize(max(0.0, dur - 36.0))].add(qid)
    return dict(sorted(schedule.items()))


def micro_prompt(qs, start, end):
    qtxt = "\n".join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    return f"""Inspect ONLY movie time {start:.1f}s-{end:.1f}s using picture AND audio.
You are not summarizing the movie. You are collecting literal evidence for these exact questions:
{qtxt}

For each listed question, report evidence ONLY if this clip directly supports or contradicts an answer.
Be concrete: identify visible objects, interfaces, people, deaths, real-vs-virtual status, exact actions, dialogue, locations and causal transitions. Do not replace a concrete event with a theme such as 'immersive experience', 'realization', or 'addiction'.
If this clip reveals that an earlier interpretation was false, mark overturns=true.
Return ONLY JSON:
{{"evidence":[{{"question_id":"id","fact":"literal observed fact","candidate":"short direct answer justified by this clip","confidence":0.0,"overturns":false}}]}}
Evidence may be empty."""


def build_packets(qs, notes, micro):
    packets = []
    for q in qs:
        qid = q["question_id"]
        ordinary, strong = [], []
        for n in notes:
            for e in n.get("evidence") or []:
                if str(e.get("question_id", "")).strip() == qid:
                    ordinary.append({
                        "time": [n.get("start"), n.get("end")],
                        "candidate": e.get("candidate", ""),
                        "fact": e.get("fact", ""),
                        "confidence": e.get("confidence"),
                    })
        for n in micro:
            for e in n.get("evidence") or []:
                if str(e.get("question_id", "")).strip() == qid:
                    strong.append({
                        "time": [n.get("start"), n.get("end")],
                        "candidate": e.get("candidate", ""),
                        "fact": e.get("fact", ""),
                        "confidence": e.get("confidence"),
                        "overturns": bool(e.get("overturns", False)),
                    })
        packets.append({
            "question_id": qid,
            "question": q["question"],
            "micro_evidence": strong,
            "ordinary_evidence": ordinary,
        })
    return packets


def final_prompt(qs, notes, micro):
    packets = build_packets(qs, notes, micro)
    return f"""Answer the movie questions from evidence packets below.

STRICT RULES:
1. MICRO_EVIDENCE is a high-resolution, question-conditioned revisit and has priority over ordinary evidence.
2. Later explicit evidence overrides earlier apparent/game/simulation interpretations.
3. Do not answer with vague abstractions ('realization', 'immersive experience', 'someone') when evidence contains a concrete object/person/action.
4. For WHY questions, state the concrete discovered event/cause, not a generic theme.
5. For ENDING questions, state what physically happens, not that the fate is uncertain or sealed.
6. For WHO questions, identify role/name when evidence supports it.
7. Never invent unsupported names or numbers. Keep the final answer short and reference-like.

PACKETS:
{json.dumps(packets, ensure_ascii=False)}

Return ONLY JSON array with exactly {len(qs)} objects in original order:
[{{"question_id":"...","prediction":"..."}}]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--video-dir", default="data/public_videos")
    ap.add_argument("--output", default="outputs/special_segmented_omni7b_v3.csv")
    ap.add_argument("--cache-dir", default="cache/segmented_omni7b")
    ap.add_argument("--micro-cache-dir", default="cache/segmented_omni7b_v3_micro")
    ap.add_argument("--model", default=base.MODEL)
    ap.add_argument("--micro-seconds", type=float, default=20.0)
    ap.add_argument("--limit-movies", type=int, default=0)
    args = ap.parse_args()

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY")

    rows = base.load_questions(args.questions)
    order = [r["question_id"] for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[r["video_id"]].append(r)
    preds = base.load_submission(args.output)
    completed = 0

    for vid, qs in groups.items():
        if all(q["question_id"] in preds for q in qs):
            continue
        if args.limit_movies and completed >= args.limit_movies:
            break
        src = base.find_video(args.video_dir, vid)
        dur = base.duration(src)

        # Reuse paid V1 full-movie evidence cache.
        notes = []
        for p in sorted((Path(args.cache_dir) / vid).glob("clip_*.json")):
            notes.append(json.loads(p.read_text(encoding="utf-8")))
        if not notes:
            raise RuntimeError(f"No V1 cache for {vid}; run V1 full movie first")

        schedule = select_windows(qs, notes, dur, args.micro_seconds)
        qmap = {q["question_id"]: q for q in qs}
        micro_notes = []
        print(f"MOVIE_V3 {vid} duration={dur:.1f}s ordinary={len(notes)} micro_windows={len(schedule)}", flush=True)

        for wi, (start, qids) in enumerate(schedule.items()):
            end = min(dur, start + args.micro_seconds)
            tag = f"{int(round(start*10)):07d}"
            cpj = Path(args.micro_cache_dir) / vid / f"micro_{tag}.json"
            wanted = [qmap[qid] for qid in sorted(qids) if qid in qmap]
            if cpj.exists():
                obj = json.loads(cpj.read_text(encoding="utf-8"))
                micro_notes.append(obj)
                continue
            cpj.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                clip = Path(td) / "micro.mp4"
                make_micro_clip(src, clip, start, end-start)
                messages = [{"role":"user","content":[
                    {"type":"video_url","video_url":{"url":base.video_data_url(clip)}},
                    {"type":"text","text":micro_prompt(wanted, start, end)},
                ]}]
                last = None
                for attempt in range(3):
                    try:
                        text, usage, dbg = base.sse_chat(key, messages, args.model, 1200, 0.0)
                        obj = base.json_object(text)
                        obj.update({"start":round(start,2),"end":round(end,2),"question_ids":[q["question_id"] for q in wanted],"usage":usage})
                        base.atomic_json(cpj, obj)
                        micro_notes.append(obj)
                        print(f"  MICRO {wi+1}/{len(schedule)} {start:.0f}-{end:.0f}s q={len(wanted)} evidence={len(obj.get('evidence') or [])}", flush=True)
                        last = None
                        break
                    except Exception as e:
                        last = e
                        print(f"  MICRO_ERR {start:.0f}s attempt={attempt+1}: {e}", flush=True)
                        time.sleep(2**attempt)
                if last:
                    raise last

        messages = [{"role":"user","content":final_prompt(qs, notes, micro_notes)}]
        text, usage, dbg = base.sse_chat(key, messages, args.model, 1900, 0.0)
        arr = base.json_array(text)
        got = {str(x.get("question_id","")).strip():str(x.get("prediction","")).strip() for x in arr if isinstance(x,dict)}
        expected = {q["question_id"] for q in qs}
        if set(got) != expected:
            raise RuntimeError(f"Final ID mismatch missing={sorted(expected-set(got))} extra={sorted(set(got)-expected)} raw={text[:1800]!r}")
        preds.update(got)
        base.write_submission(args.output, order, preds)
        completed += 1
        print(f"OK_V3 {vid}: +{len(got)} total={len(preds)}/{len(rows)} final_usage={usage}", flush=True)

    base.write_submission(args.output, order, preds)
    print(f"FINAL_V3 {len(preds)}/{len(rows)} -> {args.output}")

if __name__ == "__main__":
    main()
