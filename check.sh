#!/usr/bin/env bash

set -euo pipefail

required_files=(
  "H.avif"
  "_H.json"
  "H.json"
  "theme.json"
  "V.avif"
  "_V.json"
  "V.json"
)

failed=0

for dir in */; do
  [[ -d "$dir" ]] || continue

  missing=()

  for file in "${required_files[@]}"; do
    if [[ ! -f "${dir}${file}" ]]; then
      missing+=("$file")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "❌ ${dir%/}"
    for file in "${missing[@]}"; do
      echo "   - $file"
    done
    failed=1
  else
    echo "✅ ${dir%/}"
  fi
done

if (( failed )); then
  echo
  echo "Some directories are missing required files."
  exit 1
else
  echo
  echo "All directories are complete."
fi

