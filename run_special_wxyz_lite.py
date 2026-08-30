#!/usr/bin/env python3
import argparse, csv, json, math, os, re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import run_special_50first as base

VL_MODEL = 'qwen2.5-vl-7b-instruct'
TEXT_MODEL = 'qwen2.5-7b-instruct'


def scan_video(path, step_sec=1.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n / fps if fps else 0.0
    times = np.arange(0, max(0.1, dur), step_sec)
    recs, prev_gray, prev_hist = [], None, None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        small = cv2.resize(fr, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            motion = float(np.mean(np.linalg.norm(flow, axis=2)))
        recs.append({'t': float(t), 'cut': cut, 'motion': motion})
        prev_gray, prev_hist = gray, hist
    cap.release()
    return recs, dur


def z01(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return a
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    if hi <= lo + 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def select_keyframes(path, n=48):
    recs, dur = scan_video(path)
    if not recs:
        return [], dur
    cuts = z01([r['cut'] for r in recs])
    motions = z01([r['motion'] for r in recs])
    for i, r in enumerate(recs):
        pos = r['t'] / max(dur, 1.0)
        bookend = 1.0 if pos < 0.08 or pos > 0.88 else 0.0
        r['score'] = 0.52 * float(cuts[i]) + 0.33 * float(motions[i]) + 0.15 * bookend

    # Stage 1: temporal coverage, one representative per bin.
    cover_n = min(max(12, n // 2), len(recs))
    chosen = []
    for bi in range(cover_n):
        lo, hi = dur * bi / cover_n, dur * (bi + 1) / cover_n
        bucket = [r for r in recs if lo <= r['t'] < hi]
        if bucket:
            chosen.append(max(bucket, key=lambda r: r['score']))

    # Stage 2: narrative-rhythm peaks from scene change + optical flow.
    for r in sorted(recs, key=lambda x: x['score'], reverse=True):
        if len(chosen) >= n:
            break
        if all(abs(r['t'] - x['t']) >= 3.0 for x in chosen):
            chosen.append(r)

    # Explicitly protect ending reveal and opening setup.
    for t in [0, 5, 15, max(0, dur - 90), max(0, dur - 45), max(0, dur - 12), max(0, dur - 3)]:
        nearest = min(recs, key=lambda r: abs(r['t'] - t))
        if all(abs(nearest['t'] - x['t']) >= 1.5 for x in chosen):
            chosen.append(nearest)

    chosen = sorted(chosen, key=lambda r: r['t'])
    if len(chosen) > n:
        # Keep high-score frames while preserving chronology.
        keep = sorted(chosen, key=lambda r: r['score'], reverse=True)[:n]
        chosen = sorted(keep, key=lambda r: r['t'])
    return chosen, dur


def make_shot_sheets(video, selected, outdir, per_sheet=6):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video)); cards = []
    for i, r in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_MSEC, r['t'] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(fr); im.thumbnail((640, 360))
        card = Image.new('RGB', (656, 398), 'white')
        card.paste(im, ((656-im.width)//2, 8))
        d = ImageDraw.Draw(card)
        d.text((8, 372), f'F{i:02d}  t={r["t"]:.1f}s  rhythm={r["score"]:.2f}', fill='black')
        cards.append(card)
    cap.release()
    paths = []
    for si in range(0, len(cards), per_sheet):
        chunk = cards[si:si+per_sheet]
        sheet = Image.new('RGB', (656*3, 398*2), 'white')
        for j, im in enumerate(chunk):
            sheet.paste(im, ((j % 3)*656, (j // 3)*398))
        p = outdir / f'shot_sheet_{si//per_sheet:02d}.jpg'
        sheet.save(p, quality=92)
        paths.append(p)
    return paths


def shot_caption_prompt(qs, sheet_index):
    qtxt = '\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    return f'''You are the shot-description agent for a long-form movie QA system. This is storyboard sheet {sheet_index}. Each frame has an exact timestamp and frame ID.

QUESTIONS FOR THE WHOLE MOVIE:
{qtxt}

Describe each visible frame literally and compactly. Track recurring people, clothing, objects, locations, physical actions, injuries/deaths, screens/interfaces, and anything that could reveal a twist. Do not guess names unless text/subtitles establish them. Pay special attention to final-act reversals.

Return ONLY JSON object:
{{"frames":[{{"frame_id":"F00","time":0.0,"description":"literal visual fact","question_ids":["..."]}}],"possible_reveals":["..."]}}'''


def story_prompt(qs, source, shot_notes, dur):
    qtxt = '\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    transcript = (source.get('transcript') or '')[:42000]
    desc = (source.get('description') or '')[:6000]
    return f'''You are the narrative reconstruction agent. Build a factual movie memory from shot-level visual observations and subtitles. Later evidence overrides earlier appearances.

DURATION: {dur:.1f}s
PUBLIC DESCRIPTION (weak evidence):
{desc}

SUBTITLES / TRANSCRIPT:
{transcript}

SHOT-LEVEL VISUAL OBSERVATIONS:
{json.dumps(shot_notes, ensure_ascii=False)}

QUESTIONS:
{qtxt}

Return ONLY JSON with:
characters: names/roles/relationships;
locations: recurring settings;
timeline: 10-30 chronological concrete events with timestamps when possible;
ending: exact final event and reveal;
twists: explicit belief reversals;
question_memory: one entry per question_id containing direct_fact, supporting_evidence, confidence 0-1.
Never replace a concrete event with a generic theme.'''


def refine_prompt(qs, ledger):
    qtxt = '\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    return f'''Act as a verifier for a <8B Special Track long-video QA system. Critique this movie ledger for contradictions, vague facts, mistaken identities, and missed ending reversals. Repair it using ONLY evidence already contained in the ledger; do not invent external facts.

QUESTIONS:
{qtxt}

LEDGER:
{json.dumps(ledger, ensure_ascii=False)}

Return ONLY the corrected JSON ledger with the same top-level structure. For every question_memory entry, make direct_fact as concrete as the evidence permits.'''


def answer_group_prompt(group, ledger):
    qtxt = '\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in group)
    return f'''Answer these SF20K open-ended questions from the verified movie ledger.

Rules: 2-10 words normally; direct person/location/object/number when asked; Yes/No first for binary questions; direct cause for why/how; ending reveal overrides earlier appearance; no hedging; no unsupported details.

QUESTIONS:
{qtxt}

VERIFIED LEDGER:
{json.dumps(ledger, ensure_ascii=False)}

Return ONLY JSON array: [{{"question_id":"...","prediction":"..."}}]'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--video-dir', default='data/public_videos')
    ap.add_argument('--source-dir', default='data/source_meta')
    ap.add_argument('--cache-dir', default='cache/wxyz_lite')
    ap.add_argument('--output', default='outputs/special_wxyz_lite.csv')
    ap.add_argument('--frames', type=int, default=48)
    ap.add_argument('--limit-movies', type=int, default=0)
    ap.add_argument('--vl-model', default=VL_MODEL)
    ap.add_argument('--text-model', default=TEXT_MODEL)
    args = ap.parse_args()

    key = os.getenv('DASHSCOPE_API_KEY')
    if not key:
        raise SystemExit('Set DASHSCOPE_API_KEY')
    rows = base.load_questions(args.questions)
    order = [r['question_id'] for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[r['video_id']].append(r)
    preds = base.load_existing(args.output)
    done = 0

    for vid, qs in groups.items():
        if all(q['question_id'] in preds for q in qs):
            continue
        if args.limit_movies and done >= args.limit_movies:
            break
        video = base.find_video(args.video_dir, vid)
        source = base.load_source(args.source_dir, vid)
        cdir = Path(args.cache_dir) / vid; cdir.mkdir(parents=True, exist_ok=True)
        ledger_p = cdir / 'verified_ledger.json'

        if ledger_p.exists():
            ledger = json.loads(ledger_p.read_text(encoding='utf-8'))
        else:
            selected, dur = select_keyframes(video, args.frames)
            sheets = make_shot_sheets(video, selected, cdir / 'sheets')
            shot_notes = []
            print(f'MOVIE {vid} q={len(qs)} dur={dur:.1f}s selected={len(selected)} sheets={len(sheets)} transcript={len(source.get("transcript", ""))}', flush=True)
            for si, sheet in enumerate(sheets):
                p = cdir / f'shot_notes_{si:02d}.json'
                if p.exists():
                    obj = json.loads(p.read_text(encoding='utf-8'))
                else:
                    text = base.vl_call(key, [sheet], shot_caption_prompt(qs, si), args.vl_model)
                    obj = base.extract_json(text, 'object')
                    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
                shot_notes.append(obj)
                print(f'  SHOTS {si+1}/{len(sheets)} frames={len(obj.get("frames") or [])}', flush=True)

            raw_p = cdir / 'raw_ledger.json'
            if raw_p.exists():
                raw_ledger = json.loads(raw_p.read_text(encoding='utf-8'))
            else:
                text = base.text_call(key, story_prompt(qs, source, shot_notes, dur), args.text_model, 5000)
                raw_ledger = base.extract_json(text, 'object')
                raw_p.write_text(json.dumps(raw_ledger, ensure_ascii=False, indent=2), encoding='utf-8')

            text = base.text_call(key, refine_prompt(qs, raw_ledger), args.text_model, 5000)
            ledger = base.extract_json(text, 'object')
            ledger['_selected_times'] = [round(r['t'], 1) for r in selected]
            ledger_p.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')

        got = {}
        for i in range(0, len(qs), 8):
            chunk = qs[i:i+8]
            text = base.text_call(key, answer_group_prompt(chunk, ledger), args.text_model, 1800)
            arr = base.extract_json(text, 'array')
            for x in arr:
                if isinstance(x, dict):
                    qid = str(x.get('question_id', '')).strip(); pred = str(x.get('prediction', '')).strip()
                    if qid and pred: got[qid] = pred
        exp = {q['question_id'] for q in qs}
        if set(got) != exp:
            raise RuntimeError(f'ID mismatch {vid}: missing={sorted(exp-set(got))} extra={sorted(set(got)-exp)}')
        preds.update(got); base.write_csv(args.output, order, preds); done += 1
        print(f'OK_WXYZ {vid}: +{len(got)} total={len(preds)}/{len(rows)}', flush=True)

    base.write_csv(args.output, order, preds)
    print(f'FINAL_WXYZ {len(preds)}/{len(rows)} -> {args.output}')

if __name__ == '__main__':
    main()
