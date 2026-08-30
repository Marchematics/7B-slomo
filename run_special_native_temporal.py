#!/usr/bin/env python3
import argparse, json, math, os
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import dashscope
import run_special_50first as base

MODEL='qwen2.5-vl-7b-instruct'

def extract_uniform(video, outdir, target=128, max_frames=240):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dur,_=base.duration_and_fps(video)
    n=max(target, int(math.ceil(dur*0.1))+1)
    n=min(max_frames, n)
    times=np.linspace(0,max(0.1,dur-0.1),n)
    cap=cv2.VideoCapture(str(video)); paths=[]
    for i,t in enumerate(times):
        p=outdir/f'f_{i:04d}.jpg'
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
            ok,fr=cap.read()
            if not ok: continue
            h,w=fr.shape[:2]
            scale=min(1.0,384/max(w,1),216/max(h,1))
            if scale<1.0:
                fr=cv2.resize(fr,(max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)))
            cv2.imwrite(str(p),fr,[int(cv2.IMWRITE_JPEG_QUALITY),82])
        paths.append(p)
    cap.release()
    fps=max(0.1,min(10.0,(len(paths)-1)/max(dur,1.0)))
    return paths,fps,dur

def mm_text(resp):
    try:
        c=resp['output']['choices'][0]['message']['content']
        if isinstance(c,str): return c
        if isinstance(c,list): return ''.join(x.get('text','') for x in c if isinstance(x,dict))
    except Exception: pass
    raise RuntimeError(f'Bad response: {resp}')

def direct_call(key, model, frames, fps, prompt):
    content=[
        {'video':['file://'+str(Path(p).resolve()) for p in frames], 'fps':fps, 'min_pixels':3136, 'max_pixels':100352},
        {'text':prompt},
    ]
    resp=dashscope.MultiModalConversation.call(
        api_key=key, model=model, messages=[{'role':'user','content':content}], temperature=0.0)
    if getattr(resp,'status_code',200)!=200:
        raise RuntimeError(f'VL failed code={getattr(resp,"code",None)} msg={getattr(resp,"message",None)} raw={resp}')
    return mm_text(resp)

def prompt_for(group, source, dur):
    qtxt='\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in group)
    tr=(source.get('transcript') or '')[:38000]
    desc=(source.get('description') or '')[:4000]
    return f"""You are answering questions about ONE complete short film. The attached input is a uniformly sampled VIDEO FRAME SEQUENCE in chronological order; its fps metadata preserves approximate movie time. Use the frame sequence and subtitles together. Do not replace concrete facts with a generic plot summary.

DURATION: {dur:.1f}s
DESCRIPTION (weak evidence):
{desc}

SUBTITLES / TRANSCRIPT:
{tr}

QUESTIONS:
{qtxt}

For each question independently, locate the relevant time/event before answering. Be especially literal for names, identities, objects, locations, counts, causes, and endings. Later reveals override earlier appearances. If a question asks WHO/WHERE/WHAT OBJECT, return the exact entity rather than an explanation.

Return ONLY JSON array with exactly {len(group)} objects:
[{{\"question_id\":\"...\",\"prediction\":\"concise answer, usually 2-10 words\",\"evidence_time\":\"approx seconds or range\",\"confidence\":0.0}}]"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--video-dir',default='data/public_videos')
    ap.add_argument('--source-dir',default='data/source_meta')
    ap.add_argument('--cache-dir',default='cache/native_temporal')
    ap.add_argument('--output',default='outputs/special_native_temporal.csv')
    ap.add_argument('--model',default=MODEL)
    ap.add_argument('--target-frames',type=int,default=128)
    ap.add_argument('--max-frames',type=int,default=240)
    ap.add_argument('--batch-questions',type=int,default=6)
    ap.add_argument('--limit-movies',type=int,default=0)
    args=ap.parse_args()
    key=os.getenv('DASHSCOPE_API_KEY')
    if not key: raise SystemExit('Set DASHSCOPE_API_KEY')
    rows=base.load_questions(args.questions); order=[r['question_id'] for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[r['video_id']].append(r)
    preds=base.load_existing(args.output); done=0
    for vid,qs in groups.items():
        if all(q['question_id'] in preds for q in qs): continue
        if args.limit_movies and done>=args.limit_movies: break
        video=base.find_video(args.video_dir,vid); source=base.load_source(args.source_dir,vid)
        cdir=Path(args.cache_dir)/vid
        frames,fps,dur=extract_uniform(video,cdir/'frames',args.target_frames,args.max_frames)
        print(f'MOVIE {vid} q={len(qs)} frames={len(frames)} temporal_fps={fps:.4f} transcript={len(source.get("transcript", ""))}',flush=True)
        got={}
        for bi in range(0,len(qs),args.batch_questions):
            chunk=qs[bi:bi+args.batch_questions]
            cp=cdir/f'answers_{bi:03d}.json'
            if cp.exists():
                arr=json.loads(cp.read_text(encoding='utf-8'))
            else:
                text=direct_call(key,args.model,frames,fps,prompt_for(chunk,source,dur))
                arr=base.extract_json(text,'array')
                cp.write_text(json.dumps(arr,ensure_ascii=False,indent=2),encoding='utf-8')
            for x in arr:
                if isinstance(x,dict):
                    qid=str(x.get('question_id','')).strip(); pred=str(x.get('prediction','')).strip()
                    if qid and pred: got[qid]=pred
            print(f'  BATCH {bi//args.batch_questions+1} +{len(chunk)}',flush=True)
        exp={q['question_id'] for q in qs}
        if set(got)!=exp:
            raise RuntimeError(f'ID mismatch {vid}: missing={sorted(exp-set(got))} extra={sorted(set(got)-exp)}')
        preds.update(got); base.write_csv(args.output,order,preds); done+=1
        print(f'OK_NATIVE {vid}: +{len(got)} total={len(preds)}/{len(rows)}',flush=True)
    base.write_csv(args.output,order,preds)
    print(f'FINAL_NATIVE {len(preds)}/{len(rows)} -> {args.output}')
if __name__=='__main__': main()
