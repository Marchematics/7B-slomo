#!/usr/bin/env python3
import argparse, csv, json, os, time, urllib.request, urllib.error
from collections import defaultdict

DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

SYSTEM = """You answer questions about one long movie. Use only evidence from the supplied video, including its audio. Answer every question concisely and literally. Preserve names, numbers, places, temporal order, causes, and negation. Do not explain unless the question requires it. Return ONLY valid JSON: an array of objects with keys question_id and prediction. Exactly one object per requested question, in the same order. Never omit a question."""


def post_json(url, key, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:4000]}")


def post_stream_json(url, key, payload, timeout=900):
    """Read an OpenAI-compatible SSE stream and concatenate assistant text."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    parts = []
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                text = line[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "".join(parts), usage
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:4000]}")


def extract_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        return json.loads(text[a:b+1])
    raise ValueError(f"No JSON array found in model output: {text[:1000]}")


def load_questions(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qid = str(r.get("question_id", "")).strip()
            vid = str(r.get("video_id", "")).strip()
            q = str(r.get("question", "")).strip()
            url = str(r.get("video_url", "")).strip()
            if not qid or not vid or not q:
                raise ValueError(f"bad row: {r}")
            rows.append({"question_id": qid, "video_id": vid, "question": q, "video_url": url})
    return rows


def write_csv(path, order, preds):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "prediction"])
        for qid in order:
            if qid in preds:
                w.writerow([qid, preds[qid]])
    os.replace(tmp, path)


def load_existing(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qid = str(r.get("question_id", "")).strip()
            p = str(r.get("prediction", "")).strip()
            if qid and p:
                out[qid] = p
    return out


def build_payload(model, video_url, prompt, temperature, max_tokens, fps):
    # Qwen-Omni uses streaming only and can consume both video and its audio.
    # For VL models, fps is a sibling of video_url according to the OpenAI-compatible API.
    video_item = {"type": "video_url", "video_url": {"url": video_url}}
    if "omni" not in model.lower() and fps is not None:
        video_item["fps"] = fps

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                video_item,
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "omni" in model.lower():
        payload.update({
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
        })
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", default="outputs/special_qwen25omni7b.csv")
    ap.add_argument("--model", default="qwen2.5-omni-7b")
    ap.add_argument("--endpoint", default=os.getenv("DASHSCOPE_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--fps", type=float, default=0.2, help="Used for VL models; Omni controls its own A/V sampling")
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit-movies", type=int, default=0)
    args = ap.parse_args()

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY in the Codespace environment; never commit it.")

    rows = load_questions(args.questions)
    order = [r["question_id"] for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[r["video_id"]].append(r)

    preds = load_existing(args.output)
    done_movies = 0
    print(f"questions={len(rows)} movies={len(groups)} resume_predictions={len(preds)} model={args.model}")

    for vid, qs in groups.items():
        if all(x["question_id"] in preds for x in qs):
            continue
        if args.limit_movies and done_movies >= args.limit_movies:
            break
        video_url = next((x["video_url"] for x in qs if x["video_url"]), "")
        if not video_url:
            print(f"SKIP {vid}: missing video_url")
            continue

        qtext = "\n".join(f'{i+1}. [{x["question_id"]}] {x["question"]}' for i, x in enumerate(qs))
        prompt = f"Movie questions ({len(qs)} total):\n{qtext}\n\nReturn exactly {len(qs)} JSON objects. Predictions should usually be short noun phrases or short sentences, not explanations."
        payload = build_payload(args.model, video_url, prompt, args.temperature, args.max_tokens, args.fps)

        last_err = None
        for attempt in range(args.retries):
            try:
                if payload.get("stream"):
                    text, usage = post_stream_json(args.endpoint, key, payload)
                else:
                    resp = post_json(args.endpoint, key, payload)
                    text = resp["choices"][0]["message"]["content"]
                    usage = resp.get("usage")
                arr = extract_json_array(text)
                got = {}
                for item in arr:
                    qid = str(item.get("question_id", "")).strip()
                    p = str(item.get("prediction", "")).strip()
                    if qid and p:
                        got[qid] = p
                expected = {x["question_id"] for x in qs}
                if set(got) != expected:
                    missing = sorted(expected - set(got))
                    extra = sorted(set(got) - expected)
                    raise ValueError(f"ID mismatch missing={missing[:8]} extra={extra[:8]}")
                preds.update(got)
                write_csv(args.output, order, preds)
                done_movies += 1
                print(f"OK {vid}: +{len(got)} total={len(preds)}/{len(rows)} usage={usage}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                delay = min(10, 2 ** attempt)
                print(f"ERR {vid} attempt={attempt+1}/{args.retries}: {e}")
                time.sleep(delay)
        if last_err is not None:
            print(f"FAILED {vid}: {last_err}")

    write_csv(args.output, order, preds)
    print(f"FINAL {len(preds)}/{len(rows)} -> {args.output}")

if __name__ == "__main__":
    main()
