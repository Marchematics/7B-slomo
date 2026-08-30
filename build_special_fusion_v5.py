#!/usr/bin/env python3
import argparse, csv, json, os, re
from collections import defaultdict
from pathlib import Path
import dashscope

MODEL='qwen2.5-7b-instruct'


def load_csv(path):
    out={}
    with open(path,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            qid=(r.get('question_id') or '').strip(); p=(r.get('prediction') or '').strip()
            if qid and p: out[qid]=p
    return out


def load_questions(path):
    rows=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): rows.append({k:(v or '').strip() for k,v in r.items()})
    return rows


def source_text(root,vid):
    p=Path(root)/vid/'source.json'
    if not p.exists(): return ''
    try:
        x=json.loads(p.read_text(encoding='utf-8'))
        return ((x.get('description') or '')+'\n'+(x.get('transcript') or ''))[:24000]
    except Exception: return ''


def ledger_obj(root,vid):
    p=Path(root)/vid/'ledger.json'
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}


def text_call(key,prompt,model,max_tokens=2600):
    resp=dashscope.Generation.call(api_key=key,model=model,messages=[{'role':'user','content':prompt}],result_format='message',temperature=0.0,max_tokens=max_tokens)
    if getattr(resp,'status_code',200)!=200:
        raise RuntimeError(f'call failed code={getattr(resp,"code",None)} msg={getattr(resp,"message",None)}')
    return resp['output']['choices'][0]['message']['content']


def parse_array(text):
    t=text.strip(); t=re.sub(r'^```(?:json)?\s*','',t); t=re.sub(r'\s*```$','',t)
    try:
        x=json.loads(t)
        if isinstance(x,list): return x
    except Exception: pass
    a,b=t.find('['),t.rfind(']')
    if a>=0 and b>a: return json.loads(t[a:b+1])
    raise ValueError(f'No JSON array: {t[:1200]}')


def prompt_for(qs,a,b,ledger,source):
    rows=[]
    for q in qs:
        qid=q['question_id']
        rows.append({'question_id':qid,'question':q['question'],'A':a.get(qid,''),'B':b.get(qid,'')})
    return f'''You are a conservative evidence-based selector for SF20K open-ended movie QA.
Candidate A and Candidate B were independently produced by two different <=7B pipelines. Select the candidate best supported by the supplied movie evidence. Do NOT prefer B merely because it is newer or longer.

RULES:
1. Concrete transcript/storyboard/ledger evidence beats generic plausible language.
2. Prefer exact names, objects, places, counts, causal actions, and explicit endings.
3. For why/how questions choose the candidate with the direct causal/action fact, not a theme.
4. If A and B are semantically equivalent, choose the shorter/more literal one.
5. If BOTH are clearly contradicted by supplied evidence, you may use choice="REPAIR" and write a concise repaired answer using ONLY evidence below. Do not invent details.
6. Never use outside knowledge.
7. Output answers normally 2-12 words; yes/no questions should start Yes/No.

QUESTIONS_AND_CANDIDATES:
{json.dumps(rows,ensure_ascii=False)}

V4_MOVIE_LEDGER:
{json.dumps(ledger,ensure_ascii=False)[:26000]}

PUBLIC_DESCRIPTION_AND_SUBTITLES:
{source}

Return ONLY JSON array with exactly {len(rows)} objects in original order:
[{{"question_id":"...","choice":"A|B|REPAIR","prediction":"final concise answer","reason":"very short evidence reason"}}]
'''


def write_csv(path,order,preds):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['question_id','prediction'])
        for qid in order:
            if qid in preds: w.writerow([qid,preds[qid]])
    os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--v1',required=True)
    ap.add_argument('--v4',required=True)
    ap.add_argument('--source-dir',default='data/source_meta')
    ap.add_argument('--ledger-dir',default='cache/50first')
    ap.add_argument('--output',default='outputs/special_fusion_v5.csv')
    ap.add_argument('--audit',default='outputs/special_fusion_v5_audit.jsonl')
    ap.add_argument('--model',default=MODEL)
    ap.add_argument('--limit-movies',type=int,default=0)
    args=ap.parse_args()
    key=os.getenv('DASHSCOPE_API_KEY')
    if not key: raise SystemExit('Set DASHSCOPE_API_KEY')
    qs=load_questions(args.questions); order=[r['question_id'] for r in qs]
    A=load_csv(args.v1); B=load_csv(args.v4)
    groups=defaultdict(list)
    for r in qs:
        if r['question_id'] in A and r['question_id'] in B: groups[r['video_id']].append(r)
    preds={}; audits=[]; done=0
    for vid,mqs in groups.items():
        if args.limit_movies and done>=args.limit_movies: break
        led=ledger_obj(args.ledger_dir,vid); src=source_text(args.source_dir,vid)
        text=text_call(key,prompt_for(mqs,A,B,led,src),args.model)
        arr=parse_array(text)
        got={str(x.get('question_id','')).strip():x for x in arr if isinstance(x,dict)}
        exp={q['question_id'] for q in mqs}
        if set(got)!=exp: raise RuntimeError(f'{vid} ID mismatch missing={exp-set(got)} extra={set(got)-exp}')
        for q in mqs:
            qid=q['question_id']; x=got[qid]
            choice=str(x.get('choice','')).upper(); pred=str(x.get('prediction','')).strip()
            if choice=='A' and A.get(qid): pred=A[qid]
            elif choice=='B' and B.get(qid): pred=B[qid]
            elif choice!='REPAIR':
                # deterministic conservative fallback
                pred=B.get(qid) or A.get(qid) or pred
                choice='B_FALLBACK'
            if not pred: pred=B.get(qid) or A.get(qid) or 'Unknown'
            preds[qid]=pred
            audits.append({'video_id':vid,'question_id':qid,'choice':choice,'A':A.get(qid,''),'B':B.get(qid,''),'prediction':pred,'reason':x.get('reason','')})
        done+=1
        print(f'FUSE {vid}: +{len(mqs)} total={len(preds)}',flush=True)
    write_csv(args.output,order,preds)
    Path(args.audit).parent.mkdir(parents=True,exist_ok=True)
    with open(args.audit,'w',encoding='utf-8') as f:
        for x in audits: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(f'FINAL_FUSION {len(preds)} -> {args.output}; audit={args.audit}')

if __name__=='__main__': main()
