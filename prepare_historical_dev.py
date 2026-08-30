#!/usr/bin/env python3
import argparse, csv, os, subprocess, sys
from collections import Counter
from pathlib import Path


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--current-questions',default='data/sf20k_public_test_questions.csv')
    ap.add_argument('--movies',type=int,default=5)
    ap.add_argument('--out-dir',default='dev/historical5')
    ap.add_argument('--download-videos',action='store_true')
    ap.add_argument('--prepare-sources',action='store_true')
    args=ap.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        raise SystemExit('Install huggingface_hub first: python -m pip install -U huggingface_hub')

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    hist=hf_hub_download(repo_id='rghermi/sf20k',repo_type='dataset',filename='test_silent.csv')
    old=read_rows(hist); cur=read_rows(args.current_questions)
    current_ids={r.get('video_id','').strip() for r in cur}
    safe=[r for r in old if r.get('video_id','').strip() and r.get('video_id','').strip() not in current_ids]
    counts=Counter(r['video_id'].strip() for r in safe)
    selected=[v for v,_ in sorted(counts.items(), key=lambda x:(-x[1],x[0]))[:args.movies]]
    rank={v:i for i,v in enumerate(selected)}
    rows=sorted([r for r in safe if r['video_id'].strip() in rank], key=lambda r:(rank[r['video_id'].strip()],r['question_id']))

    qpath=out/'questions.csv'; gpath=out/'gold.csv'
    with open(qpath,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['question_id','video_id','question','video_url'])
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in w.fieldnames})
    gold_fields=['question_id','video_id','question','answer','option_0','option_1','option_2','option_3','option_4','correct_answer','correct_letter','video_url']
    with open(gpath,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=gold_fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in gold_fields})

    print('selected_movies=',selected)
    print('questions=',len(rows),'current_overlap=',len(set(selected)&current_ids))
    print('questions_file=',qpath,'gold_file=',gpath)

    if args.download_videos:
        vdir=out/'videos'; vdir.mkdir(parents=True,exist_ok=True)
        urls={r['video_id'].strip():r.get('video_url','').strip() for r in rows}
        for i,vid in enumerate(selected,1):
            if any(p.is_file() and p.stem.startswith(vid) for p in vdir.iterdir()):
                print(f'[{i}/{len(selected)}] CACHE video {vid}'); continue
            url=urls[vid]
            cmd=[sys.executable,'-m','yt_dlp','-f','bv*+ba/b','--merge-output-format','mp4','--no-playlist','-o',str(vdir/'%(id)s.%(ext)s'),url]
            print(f'[{i}/{len(selected)}] DOWNLOAD {vid}',flush=True)
            subprocess.run(cmd,check=True)

    if args.prepare_sources:
        cmd=[sys.executable,'prepare_special_sources.py','--questions',str(qpath),'--out-dir',str(out/'source_meta'),'--limit-movies',str(args.movies)]
        subprocess.run(cmd,check=True)

if __name__=='__main__': main()
