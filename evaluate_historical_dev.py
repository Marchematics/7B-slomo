#!/usr/bin/env python3
import argparse, csv, json, os, re
from collections import defaultdict
import dashscope

MODEL='qwen2.5-7b-instruct'
LETTERS='ABCDE'

def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def extract_json(text):
    t=text.strip(); t=re.sub(r'^```(?:json)?\s*','',t); t=re.sub(r'\s*```$','',t)
    try: return json.loads(t)
    except Exception:
        a,b=t.find('['),t.rfind(']')
        if a>=0 and b>a: return json.loads(t[a:b+1])
    raise ValueError(text[:1500])

def call(key,items,model):
    prompt='''Map each free-form prediction to the multiple-choice option that has the SAME SEMANTIC ANSWER. This is evaluation only. Do not answer the question yourself from world knowledge; compare the prediction to the five options. If no option expresses the prediction, output X. Return only JSON array [{"question_id":"...","letter":"A|B|C|D|E|X"}].\n\nITEMS:\n'''+json.dumps(items,ensure_ascii=False)
    r=dashscope.Generation.call(api_key=key,model=model,messages=[{'role':'user','content':prompt}],result_format='message',temperature=0.0,max_tokens=1800)
    if getattr(r,'status_code',200)!=200: raise RuntimeError(r)
    return extract_json(r['output']['choices'][0]['message']['content'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gold',required=True); ap.add_argument('--pred',required=True); ap.add_argument('--model',default=MODEL); ap.add_argument('--batch',type=int,default=16); args=ap.parse_args()
    key=os.getenv('DASHSCOPE_API_KEY')
    if not key: raise SystemExit('Set DASHSCOPE_API_KEY')
    gold=read_csv(args.gold); pred={r['question_id']:r.get('prediction','').strip() for r in read_csv(args.pred)}
    rows=[r for r in gold if r['question_id'] in pred]
    chosen={}
    for i in range(0,len(rows),args.batch):
        batch=rows[i:i+args.batch]; items=[]
        for r in batch:
            items.append({'question_id':r['question_id'],'question':r['question'],'prediction':pred[r['question_id']], 'options':{LETTERS[j]:r.get(f'option_{j}','') for j in range(5)}})
        arr=call(key,items,args.model)
        for x in arr:
            if isinstance(x,dict): chosen[str(x.get('question_id',''))]=str(x.get('letter','')).upper().strip()
    per=defaultdict(lambda:[0,0])
    correct=0
    for r in rows:
        qid=r['question_id']; ok=chosen.get(qid)==str(r.get('correct_letter','')).upper().strip(); correct+=ok; per[r['video_id']][0]+=int(ok); per[r['video_id']][1]+=1
    print(f'PROXY {correct}/{len(rows)} = {100*correct/max(1,len(rows)):.2f}%')
    for vid,(c,n) in per.items(): print(f'  {vid}: {c}/{n} = {100*c/n:.1f}%')
    missing=set(pred)-{r['question_id'] for r in gold}
    if missing: print('warning predictions not in gold=',len(missing))

if __name__=='__main__': main()
