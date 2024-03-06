#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

if ! command -v paru &> /dev/null; then
	echo "You need to install the paru AUR helper first!"
	exit 1
fi

paru -S python-torchvision-rocm

python -m venv venv --system-site-packages

source venv/bin/activate

pip install --upgrade pip

# The current diffusers package is broken, so we install this commit
pip install git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887

pip install accelerate transformers gradio opencv-python pillow
