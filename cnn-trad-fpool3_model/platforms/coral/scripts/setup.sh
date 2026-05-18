#!/usr/bin/env bash
# One-time setup: install edgetpu_compiler from the Coral apt repository.
# Run this interactively (needs sudo password).

set -euo pipefail

if command -v edgetpu_compiler &>/dev/null; then
    echo "edgetpu_compiler already installed: $(edgetpu_compiler --version 2>&1 | head -1)"
    exit 0
fi

echo "Installing edgetpu_compiler from Coral apt repository..."

curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg

echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] \
https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update
sudo apt-get install -y edgetpu-compiler

echo ""
echo "Installed: $(edgetpu_compiler --version 2>&1 | head -1)"
echo "Run ./compile_edgetpu.sh to compile the model."
