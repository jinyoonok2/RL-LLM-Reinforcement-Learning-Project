#!/bin/bash
# Simple activation script for RL-LLM project
# Usage: ./activate.sh

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run './setup.sh' first to create the environment."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export PROJECT_ROOT="$(pwd)"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment variables loaded from .env"
fi

echo ""
echo "🚀 RL-LLM Environment Activated!"
echo "================================="
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check if we have CUDA available
echo -n "CUDA status: "
python -c "import torch; print(f'Available (devices: {torch.cuda.device_count()})' if torch.cuda.is_available() else 'Not available')" 2>/dev/null || echo "PyTorch not installed yet"

echo ""
echo "Environment is ready! You can now:"
echo "  - Run ./download.sh to get datasets and models"
echo "  - Start development with the modules in src/"
echo "  - Use 'deactivate' to exit the virtual environment"