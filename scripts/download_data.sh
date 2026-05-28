#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

curl -L --fail --retry 5 -C - -o data/raw/train.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/train.zip"
curl -L --fail --retry 5 -C - -o data/raw/val.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/val.zip"
curl -L --fail --retry 5 -C - -o data/raw/test.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/test.zip"

unzip -q -o data/raw/train.zip -d data/raw/train
unzip -q -o data/raw/val.zip -d data/raw/val
unzip -q -o data/raw/test.zip -d data/raw/test
