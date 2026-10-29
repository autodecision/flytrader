#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python -m flytrader.gui "$@"
else
  exec python3 -m flytrader.gui "$@"
fi
