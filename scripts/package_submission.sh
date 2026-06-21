#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash scripts/package_submission.sh FirstName LastName [checkpoint_path]"
  exit 1
fi

zip_name="${1}_${2}_GalaxEye.zip"
staging="outputs/submission"
checkpoint_path="${3:-outputs/checkpoints_multitask_late_fusion/best.pth}"

mkdir -p "$staging"
cp "$checkpoint_path" "$staging/best.pth"
cp reports/technical_report.pdf "$staging/technical_report.pdf"
cp reports/time_resource_log.txt "$staging/time_resource_log.txt"
rm -f "$zip_name"
cd "$staging"
zip -r "../../$zip_name" .
echo "Created $zip_name"
