#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

source "${SC_VENV_PATH:-./venv}/bin/activate"

python './main.py'
