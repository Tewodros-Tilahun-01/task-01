#!/bin/bash
# setup.sh — Automated setup for Task-01 Red Team Assessment
# Platform: Linux / macOS

set -e

echo "Task-01 Setup"
echo "============="
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null && python --version 2>&1 | grep -q "Python 3"; then
    PYTHON=python
else
    echo "Error: Python 3 is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create directories
echo "Creating directory structure..."
mkdir -p target/{data,weights}
mkdir -p evidence/{adversarial/{fgsm,pgd,square},extraction,logs/{fgsm,pgd,square,extraction}}
mkdir -p reports video

# Train model only if weights don't exist
if [ ! -f "target/weights/cifar10_cnn.pth" ]; then
    echo "Training model (this will take 10-20 minutes on CPU)..."
    $PYTHON -m target.training.train
else
    echo "Model weights already exist, skipping training..."
fi

# Get hashes for Docker
if [ -f "target/weights/cifar10_cnn.pth" ]; then
    MODEL_SHA=$(sha256sum target/weights/cifar10_cnn.pth | awk '{print $1}')
    MODEL_INFO_SHA=$(sha256sum target/weights/model_info.json | awk '{print $1}')
fi

# Build Docker image
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build -t cifar10-target:latest ./target -q
    echo ""
    echo "Setup complete!"
    echo ""
    echo "Start the server with:"
    echo "  docker run -p 127.0.0.1:8000:8000 \\"
    echo "    -e MODEL_SHA256=$MODEL_SHA \\"
    echo "    -e MODEL_INFO_SHA256=$MODEL_INFO_SHA \\"
    echo "    cifar10-target:latest"
    echo ""
    echo "Then run attacks:"
    echo "  python -m attacks.run_all_attacks"
else
    echo ""
    echo "Setup complete (Docker not found, skipping image build)"
    echo ""
    echo "Install Docker, then build the image:"
    echo "  docker build -t cifar10-target:latest ./target"
fi

echo ""
