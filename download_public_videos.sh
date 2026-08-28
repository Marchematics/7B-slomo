#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/public_videos
python -m pip install -q gdown

ZIP_ID="1HlDh-zwjMo2OFX-bBUKxbYow0KyEqM6Z"
ZIP_PATH="data/sf20k_public_test_videos.zip"
OUT_DIR="data/public_videos"

if [[ ! -s "$ZIP_PATH" ]]; then
  echo "[1/3] Downloading official SF20K public test videos (~1.68 GB)..."
  python -m gdown "$ZIP_ID" -O "$ZIP_PATH"
else
  echo "[1/3] Zip already present: $ZIP_PATH"
fi

echo "[2/3] Testing archive..."
unzip -tq "$ZIP_PATH" >/dev/null

echo "[3/3] Extracting..."
unzip -oq "$ZIP_PATH" -d "$OUT_DIR"

COUNT=$(find "$OUT_DIR" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' \) | wc -l | tr -d ' ')
echo "video_files=$COUNT"
if [[ "$COUNT" -lt 50 ]]; then
  echo "WARNING: fewer than 50 video files found; inspect archive layout with: find $OUT_DIR -maxdepth 3 -type f | head -100"
fi

echo "Ready: $OUT_DIR"
