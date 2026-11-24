#!/bin/bash
# Install only the small frozen-encoder stack when an older shared environment
# lacks it. Avoids the full environment reinstall on every diagnostic job.
set -euo pipefail

PYTHON=${PYTHON:-python}
if "$PYTHON" -c 'import transformers, tokenizers, safetensors' 2>/dev/null; then
  exit 0
fi

"$PYTHON" -m pip install --disable-pip-version-check \
  transformers==4.44.2 \
  tokenizers==0.19.1 \
  safetensors==0.8.0 \
  huggingface-hub==0.36.2
