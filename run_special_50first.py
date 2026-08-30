#!/usr/bin/env python3
import argparse, base64, csv, json, math, os, re, tempfile, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dashscope

VL_MODEL='qwen2.5-vl-7b-instruct'
TEXT_MODEL='qwen2.5-7b-instruct'


def load_questions(path):
    rows=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): rows.append({k:(v or '').strip() for k,v in r.items()})
    return rows


def find_video(root,vid):
    exts={'.mp4','.mkv','.mov','.avi','.flv','.wmv'}
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in exts and (p.stem==vid or p.stem.startswith(vid)):
            return p
    raise FileNotFoundError(f'No video for {vid}')


def load_source(root,vid):
    p=Path(root)/vid/'source.json'
    if p.exists(): return json.loads(p.read_text(encoding='utf-8'))
    return {'transcript':'','description':''}


def load_v1_notes(cache_root,vid):
    d=Path(cache_root)/vid
    if not d.exists(): return []
    out=[]
    for p in sorted(d.glob('clip_*.json')):
        try:
            x=json.loads(p.read_text(encoding='utf-8'))
            out.append({'start':x.get('start'),'end':x.get('end'),'summary':x.get('clip_summary',''),'evidence':x.get('evidence',[])})
        except Exception: pass
    return out


def duration_and_fps(path):
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; n=cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur=n/fps if fps else 0; cap.release(); return dur,fps


def read_frame(cap,t):
    cap.set(cv2.CAP_PROP_POS_MSEC,max(0,t)*1000)
    ok,fr=cap.read(); return fr if ok else None


def choose_times(path,n=24):
    dur,_=duration_and_fps(path)
    cap=cv2.VideoCapture(str(path))
    scan=np.linspace(0,max(0,dur-0.1),min(240,max(30,int(dur/2)+1)))
    feats=[]; times=[]
    prev=None
    for t in scan:
        fr=read_frame(cap,float(t))
        if fr is None: continue
        small=cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY),(96,54))
        diff=0.0 if prev is None else float(np.mean(cv2.absdiff(small,prev)))
        feats.append(diff); times.append(float(t)); prev=small
    cap.release()
    uniform=list(np.linspace(0,max(0,dur-0.1),max(8,n//2)))
    peaks=[]
    if feats:
        order=np.argsort(feats)[::-1]
        for idx in order:
            t=times[int(idx)]
            if all(abs(t-x)>max(5,dur/80) for x in peaks): peaks.append(t)
            if len(peaks)>=n//2: break
    extra=[0,min(10,dur*.03),max(0,dur*.5),max(0,dur-60),max(0,dur-25),max(0,dur-5)]
    cand=sorted(uniform+peaks+extra)
    chosen=[]
    for t in cand:
        if not chosen or abs(t-chosen[-1])>1.5: chosen.append(float(t))
    if len(chosen)>n:
        idx=np.linspace(0,len(chosen)-1,n).round().astype(int); chosen=[chosen[i] for i in idx]
    return chosen,dur


def make_sheets(video,times,outdir,per_sheet=6):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video)); imgs=[]
    for t in times:
        fr=read_frame(cap,t)
        if fr is None: continue
        fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        im=Image.fromarray(fr); im.thumbnail((560,315))
        canvas=Image.new('RGB',(576,345),'white'); canvas.paste(im,((576-im.width)//2,8))
        d=ImageDraw.Draw(canvas); d.text((8,325),f't={t:.1f}s',fill='black')
        imgs.append(canvas)
    cap.release(); paths=[]
    for si in range(0,len(imgs),per_sheet):
        chunk=imgs[si:si+per_sheet]
        sheet=Image.new('RGB',(576*3,345*2),'white')
        for j,im in enumerate(chunk): sheet.paste(im,((j%3)*576,(j//3)*345))
        p=outdir/f'sheet_{si//per_sheet:02d}.jpg'; sheet.save(p,quality=90); paths.append(p)
    return paths


def mm_text(resp):
    try:
        content=resp['output']['choices'][0]['message']['content']
        if isinstance(content,str): return content
        if isinstance(content,list):
            return ''.join(x.get('text','') for x in content if isinstance(x,dict))
    except Exception: pass
    raise RuntimeError(f'Bad multimodal response: {resp}')


def vl_call(key,images,prompt,model):
    content=[{'image':'file://'+str(Path(p).resolve())} for p in images]+[{'text':prompt}]
    resp=dashscope.MultiModalConversation.call(api_key=key,model=model,messages=[{'role':'user','content':content}],temperature=0.0)
    if getattr(resp,'status_code',200)!=200:
        raise RuntimeError(f'VL call failed: code={getattr(resp,"code",None)} msg={getattr(resp,"message",None)} raw={resp}')
    return mm_text(resp)


def text_call(key,prompt,model,max_tokens=2400):
    resp=dashscope.Generation.call(api_key=key,model=model,messages=[{'role':'user','content':prompt}],result_format='message',temperature=0.0,max_tokens=max_tokens)
    if getattr(resp,'status_code',200)!=200:
        raise RuntimeError(f'Text call failed: code={getattr(resp,"code",None)} msg={getattr(resp,"message",None)} raw={resp}')
    return resp['output']['choices'][0]['message']['content']


def extract_json(text,want='object'):
    t=text.strip(); t=re.sub(r'^```(?:json)?\s*','',t); t=re.sub(r'\s*```$','',t)
    try: return json.loads(t)
    except Exception: pass
    a=t.find('{' if want=='object' else '['); b=t.rfind('}' if want=='object' else ']')
    if a>=0 and b>a: return json.loads(t[a:b+1])
    raise ValueError(f'No JSON {want}: {t[:1500]}')


def ledger_prompt(qs,source,v1,dur):
    qtxt='\n'.join(f'- [{q["question_id"]}] {q["question"]}' for q in qs)
    transcript=(source.get('transcript') or '')[:30000]
    desc=(source.get('description') or '')[:5000]
    v1txt=json.dumps(v1,ensure_ascii=False)[:18000]
    return f'''You are building a factual movie ledger for SF20K. The attached storyboard frames span the WHOLE movie from beginning to ending and include timestamps. Do not answer from genre stereotypes. Reconstruct identity, relationships, major actions, causes, twists, and the ending. Later evidence overrides earlier appearances when there is a reveal.

MOVIE TITLE: {source.get('movie_title') or source.get('provider_title') or ''}
DURATION: {dur:.1f}s
PUBLIC DESCRIPTION (may be incomplete; do not trust it over video):
{desc}

SUBTITLES / TRANSCRIPT (may contain ASR errors):
{transcript}

OPTIONAL PREVIOUS 7B CLIP NOTES (use only as weak evidence):
{v1txt}

QUESTIONS:
{qtxt}

Return ONLY JSON object with keys:
characters: concise list of names/roles/relationships;
timeline: 8-20 chronological literal events;
ending: exact final event/reveal;
twists: concrete reversals of earlier beliefs;
question_evidence: one object per question with question_id, best_fact, source (storyboard/subtitle/description/clip-note/inference), confidence 0-1.
For who/where/what-object/count/name questions preserve exact concrete details. For why/how questions preserve causal links. Do not leave generic phrases if a concrete fact is visible or stated.'''


def answer_prompt(qs,ledger):
    qtxt='\n'.join(f'{i+1}. [{q["question_id"]}] {q["question"]}' for i,q in enumerate(qs))
    return f'''Answer all SF20K open-ended questions using ONLY this movie ledger. The evaluator is GPT-4.1-nano comparing semantic correctness.

STYLE RULES:
- Usually 2-8 words; maximum 15 unless needed.
- WHO: give the person/role directly. WHERE: location directly. COUNT: exact number. YES/NO: start with Yes or No.
- WHY/HOW: state the direct causal/action fact, not a theme.
- Never hedge with "might", "possibly", "not clear", or "based on this clip" when the ledger contains an answer.
- A final reveal overrides an earlier apparent interpretation.
- Do not invent names or details absent from the ledger.

QUESTIONS:
{qtxt}

LEDGER:
{json.dumps(ledger,ensure_ascii=False)}

Return ONLY JSON array in original order with exactly {len(qs)} objects: [{{"question_id":"...","prediction":"..."}}].'''


def write_csv(path,order,preds):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['question_id','prediction'])
        for qid in order:
            if qid in preds: w.writerow([qid,preds[qid]])
    os.replace(tmp,path)


def load_existing(path):
    d={}
    if Path(path).exists():
        with open(path,newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('question_id') and r.get('prediction'): d[r['question_id']]=r['prediction']
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--video-dir',default='data/public_videos')
    ap.add_argument('--source-dir',default='data/source_meta')
    ap.add_argument('--v1-cache-dir',default='cache/segmented_omni7b')
    ap.add_argument('--cache-dir',default='cache/50first')
    ap.add_argument('--output',default='outputs/special_50first.csv')
    ap.add_argument('--vl-model',default=VL_MODEL)
    ap.add_argument('--text-model',default=TEXT_MODEL)
    ap.add_argument('--frames',type=int,default=24)
    ap.add_argument('--limit-movies',type=int,default=0)
    args=ap.parse_args()
    key=os.getenv('DASHSCOPE_API_KEY')
    if not key: raise SystemExit('Set DASHSCOPE_API_KEY')
    rows=load_questions(args.questions); order=[r['question_id'] for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[r['video_id']].append(r)
    preds=load_existing(args.output); done=0
    for vid,qs in groups.items():
        if all(q['question_id'] in preds for q in qs): continue
        if args.limit_movies and done>=args.limit_movies: break
        video=find_video(args.video_dir,vid); source=load_source(args.source_dir,vid); v1=load_v1_notes(args.v1_cache_dir,vid)
        cdir=Path(args.cache_dir)/vid; cdir.mkdir(parents=True,exist_ok=True)
        lp=cdir/'ledger.json'
        if lp.exists():
            ledger=json.loads(lp.read_text(encoding='utf-8'))
        else:
            times,dur=choose_times(video,args.frames); sheets=make_sheets(video,times,cdir/'sheets')
            print(f'MOVIE {vid} q={len(qs)} frames={len(times)} sheets={len(sheets)} transcript={len(source.get("transcript", ""))} v1notes={len(v1)}',flush=True)
            text=vl_call(key,sheets,ledger_prompt(qs,source,v1,dur),args.vl_model)
            ledger=extract_json(text,'object'); ledger['_frame_times']=[round(x,1) for x in times]
            lp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
        text=text_call(key,answer_prompt(qs,ledger),args.text_model,2400)
        arr=extract_json(text,'array')
        got={str(x.get('question_id','')).strip():str(x.get('prediction','')).strip() for x in arr if isinstance(x,dict)}
        exp={q['question_id'] for q in qs}
        if set(got)!=exp: raise RuntimeError(f'ID mismatch {vid}: missing={exp-set(got)} extra={set(got)-exp}')
        preds.update(got); write_csv(args.output,order,preds); done+=1
        print(f'OK50 {vid}: +{len(got)} total={len(preds)}/{len(rows)}',flush=True)
    write_csv(args.output,order,preds); print(f'FINAL50 {len(preds)}/{len(rows)} -> {args.output}')

if __name__=='__main__': main()
