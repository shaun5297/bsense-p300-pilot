#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${P300_PYTHON:-python3}"
if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -c 'import sys, tkinter; assert (3,11) <= sys.version_info[:2] < (3,14), "Python 3.11-3.13 with Tk is required"'
  "$PYTHON_BIN" -m venv ".venv"
fi
".venv/bin/python" -m pip install -e ".[test]"
".venv/bin/python" -m bsense_p300_pilot --plan
