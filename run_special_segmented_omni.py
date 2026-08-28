#!/usr/bin/env python3
import argparse, base64, csv, json, math, os, re, subprocess, tempfile, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen2.5-omni-7b"


def sse_chat(key, messages, model=MODEL, max_tokens=1100, temperature=0.01, timeout=900):
    payload = {
        "model": model,
        "messages": messages,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream"
    })
    parts, usage, raw_debug = [], None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                text = line[5:].strip()
                if text == "[DONE]":
                    break
                if len(raw_debug) < 8:
                    raw_debug.append(text[:1000])
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                parts.append(item["text"])
        return "".join(parts), usage, raw_debug
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:5000]}")


def json_object(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b+1])
    raise ValueError(f"No JSON object: {text[:1200]!r}")


def json_array(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        v = json.loads(t)
        if isinstance(v, list): return v
    except Exception: pass
    a, b = t.find("["), t.rfind("]")
    if a >= 0 and b > a:
        return json.loads(t[a:b+1])
    raise ValueError(f"No JSON array: {text[:1200]!r}")


def load_questions(path):
    rows=[]
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({k:(v or "").strip() for k,v in r.items()})
    return rows


def find_video(root, vid):
    exts={".mp4",".mkv",".mov",".avi",".flv",".wmv"}
    candidates=[]
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and (p.stem == vid or p.stem.startswith(vid)):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No local video for {vid} under {root}")
    return min(candidates, key=lambda p: len(str(p)))


def duration(path):
    out=subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)], text=True)
    return float(out.strip())


def make_clip(src, out, start, seconds):
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-ss",f"{start:.3f}","-t",f"{seconds:.3f}","-i",str(src),
         "-vf","fps=6,scale=-2:360","-c:v","libx264","-preset","veryfast","-b:v","260k","-maxrate","320k","-bufsize","640k",
         "-c:a","aac","-b:a","48k","-ac","1","-ar","16000","-movflags","+faststart",str(out)]
    subprocess.run(cmd, check=True)
    n=os.path.getsize(out)
    if math.ceil(n*4/3) >= 9_500_000:
        raise RuntimeError(f"Compressed clip still too large for Base64 API: {n} bytes")


def video_data_url(path):
    return "data:;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def clip_prompt(qs, start, end):
    qtxt="\n".join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    return f"""You are inspecting ONLY movie time {start:.1f}s-{end:.1f}s. Use both picture and audio.\nQuestions for the whole movie:\n{qtxt}\n\nExtract evidence from THIS CLIP only. Do not guess answers not supported here. Return only JSON object:\n{{\"clip_summary\":\"literal events/dialogue in this clip\",\"evidence\":[{{\"question_id\":\"id\",\"candidate\":\"short answer/evidence\",\"confidence\":0.0}}]}}\nEvidence may be empty. Preserve names, quoted words, numbers, locations, causal and temporal facts."""


def synth_prompt(qs, notes):
    qtxt="\n".join(f'{i+1}. [{q["question_id"]}] {q["question"]}' for i,q in enumerate(qs))
    ntxt="\n".join(json.dumps(n, ensure_ascii=False) for n in notes)
    return f"""Answer all questions about one movie using the time-stamped clip evidence below. Resolve chronology and cross-clip consistency. If evidence is indirect, make the most likely literal inference, but never invent specific names/numbers. Answers should be concise because an LLM judge compares semantic correctness.\n\nQUESTIONS:\n{qtxt}\n\nCLIP EVIDENCE:\n{ntxt}\n\nReturn ONLY a JSON array with exactly {len(qs)} objects, in question order: [{{\"question_id\":\"...\",\"prediction\":\"...\"}}]."""


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp,path)


def write_submission(path, order, preds):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["question_id","prediction"])
        for qid in order:
            if qid in preds: w.writerow([qid,preds[qid]])
    os.replace(tmp,path)


def load_submission(path):
    d={}
    if Path(path).exists():
        with open(path,newline="",encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("question_id") and r.get("prediction"): d[r["question_id"]]=r["prediction"]
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions",required=True)
    ap.add_argument("--video-dir",default="data/public_videos")
    ap.add_argument("--output",default="outputs/special_segmented_omni7b.csv")
    ap.add_argument("--cache-dir",default="cache/segmented_omni7b")
    ap.add_argument("--model",default=MODEL)
    ap.add_argument("--clip-seconds",type=float,default=32.0)
    ap.add_argument("--overlap",type=float,default=4.0)
    ap.add_argument("--limit-movies",type=int,default=0)
    ap.add_argument("--max-clips",type=int,default=0,help="Smoke-test cap per movie; 0=all")
    args=ap.parse_args()
    key=os.getenv("DASHSCOPE_API_KEY")
    if not key: raise SystemExit("Set DASHSCOPE_API_KEY")
    rows=load_questions(args.questions); order=[r["question_id"] for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[r["video_id"]].append(r)
    preds=load_submission(args.output); completed=0
    for vid,qs in groups.items():
        if all(q["question_id"] in preds for q in qs): continue
        if args.limit_movies and completed>=args.limit_movies: break
        src=find_video(args.video_dir,vid); dur=duration(src)
        step=max(1.0,args.clip_seconds-args.overlap)
        starts=[i*step for i in range(int(math.ceil(max(0.1,dur-args.overlap)/step)))]
        starts=[s for s in starts if s<dur]
        if args.max_clips: starts=starts[:args.max_clips]
        print(f"MOVIE {vid} file={src} duration={dur:.1f}s clips={len(starts)} questions={len(qs)}", flush=True)
        notes=[]
        for ci,start in enumerate(starts):
            end=min(dur,start+args.clip_seconds)
            cache=Path(args.cache_dir)/vid/f"clip_{ci:04d}.json"
            if cache.exists():
                notes.append(json.loads(cache.read_text(encoding="utf-8"))); continue
            cache.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                cp=Path(td)/"clip.mp4"; make_clip(src,cp,start,end-start)
                messages=[{"role":"user","content":[{"type":"video_url","video_url":{"url":video_data_url(cp)}},{"type":"text","text":clip_prompt(qs,start,end)}]}]
                last=None
                for attempt in range(3):
                    try:
                        text,usage,dbg=sse_chat(key,messages,args.model,1100,0.01)
                        obj=json_object(text); obj.update({"clip_index":ci,"start":round(start,2),"end":round(end,2),"usage":usage})
                        atomic_json(cache,obj); notes.append(obj)
                        print(f"  CLIP {ci+1}/{len(starts)} {start:.0f}-{end:.0f}s evidence={len(obj.get('evidence') or [])} usage={usage}",flush=True)
                        last=None; break
                    except Exception as e:
                        last=e; print(f"  CLIP_ERR {ci} attempt={attempt+1}: {e}",flush=True); time.sleep(2**attempt)
                if last: raise last
        if args.max_clips and len(starts)*step+args.overlap < dur-1:
            print("SMOKE_ONLY: clip cap did not cover full movie; skipping final answers.",flush=True)
            completed+=1; continue
        messages=[{"role":"user","content":synth_prompt(qs,notes)}]
        text,usage,dbg=sse_chat(key,messages,args.model,1800,0.01)
        arr=json_array(text); got={str(x.get("question_id","")).strip():str(x.get("prediction","")).strip() for x in arr if isinstance(x,dict)}
        expected={q["question_id"] for q in qs}
        if set(got)!=expected: raise RuntimeError(f"Final ID mismatch missing={sorted(expected-set(got))} extra={sorted(set(got)-expected)} raw={text[:1500]!r}")
        preds.update(got); write_submission(args.output,order,preds); completed+=1
        print(f"OK {vid}: +{len(got)} total={len(preds)}/{len(rows)} final_usage={usage}",flush=True)
    write_submission(args.output,order,preds)
    print(f"FINAL {len(preds)}/{len(rows)} -> {args.output}")

if __name__=="__main__": main()
