#!/usr/bin/env python3
import argparse, json, math, os, re, tempfile, time
from collections import defaultdict
from pathlib import Path

import run_special_segmented_omni as base

RISK_RE = re.compile(r"\b(truly|truth|actually|ultimately|ending|end|final|last|twist|initially|why|real|really|eventually|in the end)\b", re.I)


def risky(q):
    return bool(RISK_RE.search(q.get("question", "")))


def verify_prompt(qs, start, end, position):
    qtxt = "\n".join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    return f"""You are doing a contradiction/plot-twist verification pass on ONLY movie time {start:.1f}s-{end:.1f}s ({position}).
Use both picture and audio. The listed questions are high-risk because they ask about truth, causality, identity, beginnings, endings, or twists.

QUESTIONS:
{qtxt}

Do NOT give generic movie-theme answers. Extract only concrete facts visibly shown or audibly stated in THIS clip. Pay special attention to revelations that overturn what the protagonist or viewer believed earlier.
Return ONLY JSON:
{{"reveal":"single most important literal revelation/event in this clip", "evidence":[{{"question_id":"id","fact":"literal supporting fact","candidate":"short candidate answer if this clip supports one","confidence":0.0,"overturns_earlier_assumption":false}}]}}
Evidence may be empty. Preserve names, objects, deaths, actions, causal links, and whether something is real versus virtual."""


def build_question_packets(qs, notes, verify_notes):
    packets = []
    for q in qs:
        qid = q["question_id"]
        ev = []
        for n in notes:
            for x in n.get("evidence") or []:
                if str(x.get("question_id", "")).strip() == qid:
                    ev.append({
                        "source": f'ordinary {n.get("start")}s-{n.get("end")}s',
                        "candidate": x.get("candidate", ""),
                        "fact": x.get("fact", ""),
                        "confidence": x.get("confidence", None),
                    })
        for n in verify_notes:
            for x in n.get("evidence") or []:
                if str(x.get("question_id", "")).strip() == qid:
                    ev.append({
                        "source": f'VERIFY {n.get("start")}s-{n.get("end")}s',
                        "candidate": x.get("candidate", ""),
                        "fact": x.get("fact", ""),
                        "confidence": x.get("confidence", None),
                        "overturns_earlier_assumption": x.get("overturns_earlier_assumption", False),
                    })
        packets.append({"question_id": qid, "question": q["question"], "evidence": ev})
    return packets


def final_prompt(qs, notes, verify_notes):
    packets = build_question_packets(qs, notes, verify_notes)
    timeline = []
    for n in notes:
        timeline.append({"start": n.get("start"), "end": n.get("end"), "summary": n.get("clip_summary", "")})
    reveals = [{"start": n.get("start"), "end": n.get("end"), "reveal": n.get("reveal", "")} for n in verify_notes]
    return f"""Answer every movie question from structured evidence.

DECISION RULES:
1. Literal video/audio evidence beats generic world knowledge.
2. A later explicit reveal can overturn an earlier apparent interpretation.
3. For questions containing words like truly, truth, actually, ultimately, ending, final, why, initially, or twist, inspect VERIFY evidence first.
4. Do not soften a concrete event into vague language. If the evidence shows death, a real-world target, a named person, or a physical mechanism, say it directly.
5. Do not invent unsupported names/numbers. Keep answers concise and reference-like.

QUESTION PACKETS:
{json.dumps(packets, ensure_ascii=False)}

MOVIE TIMELINE:
{json.dumps(timeline, ensure_ascii=False)}

BOOKEND/TWIST REVEALS:
{json.dumps(reveals, ensure_ascii=False)}

Return ONLY a JSON array with exactly {len(qs)} objects in original order:
[{{"question_id":"...","prediction":"..."}}]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--video-dir", default="data/public_videos")
    ap.add_argument("--output", default="outputs/special_segmented_omni7b_v2.csv")
    ap.add_argument("--cache-dir", default="cache/segmented_omni7b")
    ap.add_argument("--verify-cache-dir", default="cache/segmented_omni7b_v2_verify")
    ap.add_argument("--model", default=base.MODEL)
    ap.add_argument("--clip-seconds", type=float, default=32.0)
    ap.add_argument("--overlap", type=float, default=4.0)
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
        step = max(1.0, args.clip_seconds - args.overlap)
        starts = [i * step for i in range(int(math.ceil(max(0.1, dur - args.overlap) / step)))]
        starts = [s for s in starts if s < dur]

        # Reuse V1 ordinary evidence cache; do not pay for it twice.
        notes = []
        missing = []
        for ci, start in enumerate(starts):
            p = Path(args.cache_dir) / vid / f"clip_{ci:04d}.json"
            if p.exists():
                notes.append(json.loads(p.read_text(encoding="utf-8")))
            else:
                missing.append((ci, start))
        if missing:
            raise RuntimeError(f"V1 cache incomplete for {vid}: {len(missing)} clips missing. Run V1 full movie first.")

        risk_qs = [q for q in qs if risky(q)]
        # Always verify two beginning + three ending windows. The ending is weighted more heavily
        # because SF20K questions often hinge on a final reveal.
        verify_indices = sorted(set(list(range(min(2, len(starts)))) + list(range(max(0, len(starts)-3), len(starts)))))
        verify_notes = []
        print(f"MOVIE {vid} duration={dur:.1f}s ordinary_cache={len(notes)} risk_q={len(risk_qs)} verify_windows={verify_indices}", flush=True)

        if risk_qs:
            for vi in verify_indices:
                start = starts[vi]
                end = min(dur, start + args.clip_seconds)
                vp = Path(args.verify_cache_dir) / vid / f"verify_{vi:04d}.json"
                if vp.exists():
                    verify_notes.append(json.loads(vp.read_text(encoding="utf-8")))
                    continue
                vp.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as td:
                    cp = Path(td) / "clip.mp4"
                    base.make_clip(src, cp, start, end-start)
                    pos = "BEGINNING" if vi < 2 else "ENDING/FINAL ACT"
                    messages = [{"role":"user","content":[
                        {"type":"video_url","video_url":{"url":base.video_data_url(cp)}},
                        {"type":"text","text":verify_prompt(risk_qs, start, end, pos)},
                    ]}]
                    last = None
                    for attempt in range(3):
                        try:
                            text, usage, dbg = base.sse_chat(key, messages, args.model, 1300, 0.0)
                            obj = base.json_object(text)
                            obj.update({"clip_index":vi,"start":round(start,2),"end":round(end,2),"usage":usage})
                            base.atomic_json(vp, obj)
                            verify_notes.append(obj)
                            print(f"  VERIFY {vi} {start:.0f}-{end:.0f}s evidence={len(obj.get('evidence') or [])} reveal={obj.get('reveal','')[:120]}", flush=True)
                            last = None
                            break
                        except Exception as e:
                            last = e
                            print(f"  VERIFY_ERR {vi} attempt={attempt+1}: {e}", flush=True)
                            time.sleep(2**attempt)
                    if last:
                        raise last

        prompt = final_prompt(qs, notes, verify_notes)
        messages = [{"role":"user","content":prompt}]
        text, usage, dbg = base.sse_chat(key, messages, args.model, 1900, 0.0)
        arr = base.json_array(text)
        got = {str(x.get("question_id","")).strip():str(x.get("prediction","")).strip() for x in arr if isinstance(x,dict)}
        expected = {q["question_id"] for q in qs}
        if set(got) != expected:
            raise RuntimeError(f"Final ID mismatch missing={sorted(expected-set(got))} extra={sorted(set(got)-expected)} raw={text[:1500]!r}")
        preds.update(got)
        base.write_submission(args.output, order, preds)
        completed += 1
        print(f"OK_V2 {vid}: +{len(got)} total={len(preds)}/{len(rows)} final_usage={usage}", flush=True)

    base.write_submission(args.output, order, preds)
    print(f"FINAL_V2 {len(preds)}/{len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
