#!/usr/bin/env python3
import argparse, csv, html, json, re, subprocess, sys
from pathlib import Path


def load_movies(path):
    out=[]; seen=set()
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            vid=(r.get('video_id') or '').strip(); url=(r.get('video_url') or '').strip(); title=(r.get('movie_title') or '').strip()
            if vid and vid not in seen:
                seen.add(vid); out.append((vid,url,title))
    return out


def vtt_to_text(path):
    lines=[]; last=''
    for raw in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        s=raw.strip()
        if not s or s.startswith('WEBVTT') or '-->' in s or re.fullmatch(r'\d+',s):
            continue
        s=re.sub(r'<[^>]+>','',s)
        s=html.unescape(s)
        s=re.sub(r'\s+',' ',s).strip()
        if s and s != last:
            lines.append(s); last=s
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--out-dir',default='data/source_meta')
    ap.add_argument('--limit-movies',type=int,default=0)
    args=ap.parse_args()
    movies=load_movies(args.questions)
    if args.limit_movies: movies=movies[:args.limit_movies]
    root=Path(args.out_dir); root.mkdir(parents=True,exist_ok=True)
    ok_subs=0
    for i,(vid,url,title) in enumerate(movies,1):
        d=root/vid; d.mkdir(parents=True,exist_ok=True)
        out_json=d/'source.json'
        if out_json.exists():
            obj=json.loads(out_json.read_text(encoding='utf-8'))
            print(f'[{i}/{len(movies)}] CACHE {vid} transcript_chars={len(obj.get("transcript", ""))}')
            ok_subs += bool(obj.get('transcript'))
            continue
        tmpl=str(d/'source.%(ext)s')
        cmd=[sys.executable,'-m','yt_dlp','--skip-download','--write-info-json','--write-description',
             '--write-subs','--write-auto-subs','--sub-langs','en.*,en','--sub-format','vtt',
             '--convert-subs','vtt','--no-playlist','-o',tmpl,url]
        err=''
        try:
            p=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
            if p.returncode!=0: err=(p.stderr or p.stdout)[-4000:]
        except Exception as e:
            err=repr(e)
        desc=''
        desc_files=list(d.glob('*.description'))
        if desc_files: desc=desc_files[0].read_text(encoding='utf-8',errors='replace').strip()
        info={}
        infos=list(d.glob('*.info.json'))
        if infos:
            try: info=json.loads(infos[0].read_text(encoding='utf-8'))
            except Exception: pass
        vtts=sorted(d.glob('*.vtt'), key=lambda p:(0 if '.en.' in p.name or p.name.endswith('.en.vtt') else 1, len(p.name)))
        transcript=''
        chosen=''
        for vp in vtts:
            t=vtt_to_text(vp)
            if len(t)>len(transcript): transcript=t; chosen=vp.name
        obj={
            'video_id':vid,'url':url,'movie_title':title,
            'provider_title':info.get('title',''),'uploader':info.get('uploader',''),
            'description':desc or info.get('description','') or '',
            'transcript':transcript,'subtitle_file':chosen,'yt_dlp_error':err,
        }
        out_json.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        ok_subs += bool(transcript)
        print(f'[{i}/{len(movies)}] {vid} transcript_chars={len(transcript)} desc_chars={len(obj["description"])} err={bool(err)}')
    print(f'DONE movies={len(movies)} with_transcript={ok_subs}')

if __name__=='__main__': main()
