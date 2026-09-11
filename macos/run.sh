#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x ".venv/bin/python" ]; then
  echo "Run bash macos/setup.sh first."
  exit 1
fi
exec ".venv/bin/python" -m bsense_p300_pilot "$@"
