#!/usr/bin/env bash
set -euo pipefail

mkdir -p data outputs

python -m pip install -q --upgrade pip
python -m pip install -q gdown pandas requests

QUESTIONS_ID="1y524i-A8eUaPgsPcnA0yT2m6wF5sMLZ8"
QUESTIONS="data/sf20k_public_test_questions.csv"

if [[ ! -s "$QUESTIONS" ]]; then
  echo "[1/2] Downloading SF20K 2026 public questions from Google Drive..."
  gdown --id "$QUESTIONS_ID" -O "$QUESTIONS"
else
  echo "[1/2] Questions already present: $QUESTIONS"
fi

echo "[2/2] Validating public questions..."
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('data/sf20k_public_test_questions.csv')
df = pd.read_csv(p)
required = {'question_id','video_id','question','video_url'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f'Missing columns: {sorted(missing)}; got={list(df.columns)}')
print('rows=', len(df))
print('movies=', df['video_id'].nunique())
print('columns=', list(df.columns))
print('blank_questions=', int(df['question'].isna().sum()))
print('blank_video_urls=', int(df['video_url'].isna().sum()))
if len(df) != 538:
    raise SystemExit(f'Expected 538 public questions, got {len(df)}')
if df['video_id'].nunique() != 50:
    raise SystemExit(f'Expected 50 public movies, got {df["video_id"].nunique()}')
print('VALID: 538 questions / 50 movies')
PY

echo
echo "Ready. Next command:"
echo "python run_special_dashscope.py --questions $QUESTIONS --output outputs/special_qwen3vl4b.csv --model qwen3-vl-4b-instruct --fps 0.2 --limit-movies 1"
