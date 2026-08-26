#!/usr/bin/env bash

set -euo pipefail

VENV_DIR="${HOME}/.venv"
GIT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
DATASET_DIR="${HOME}/datasets/raw"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install huggingface_hub pyarrow
mkdir -p "${DATASET_DIR}"

"${VENV_DIR}/bin/hf" download \
  HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --include "sample/10BT/*.parquet" \
  --local-dir "${DATASET_DIR}"
