#!/bin/bash

# RL-LLM Reinforcement Learning Project Setup Script
# Creates virtual environment and installs Python dependencies
# Usage: ./setup.sh [--restart] [--help]

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
RESTART=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --restart)
            RESTART=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --restart : Remove existing venv and recreate from scratch"
            echo "  -h, --help Show this help message"
            echo ""
            echo "This script creates a Python virtual environment and installs"
            echo "all required dependencies for the RL-LLM project."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Starting RL-LLM Project Setup..."

# Handle restart option
if [ "$RESTART" = true ]; then
    if [ -d "venv" ]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
        print_success "Existing virtual environment removed"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed. Please install it first:"
        print_error "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        exit 1
    fi
    
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        print_error "Make sure python3-venv is installed: sudo apt install python3-venv"
        exit 1
    fi
    print_success "Virtual environment created successfully"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    print_status "Installing Python dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
    print_success "All Python dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Create project directory structure
print_status "Creating project directory structure..."
mkdir -p datasets/finqa outputs/finqa_rl configs src tests notebooks docs
print_success "Project directories created"

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export PROJECT_ROOT="$(pwd)"

print_success "Setup completed successfully!"
echo ""
echo "🚀 RL-LLM Project Setup Complete!"
echo "=================================="
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(python -m pip --version)"

# Check CUDA availability
echo -n "CUDA status: "
python -c "import torch; print(f'Available (devices: {torch.cuda.device_count()})' if torch.cuda.is_available() else 'Not available')" 2>/dev/null || echo "PyTorch installed, checking CUDA..."

echo ""
echo "Next steps:"
echo "1. Use './activate.sh' to activate the environment for development"
echo "2. Run './download.sh' to download datasets and models"
echo "3. Start development with the module files (00_*.py, etc.)"
echo ""
echo "The virtual environment is ready. Use 'deactivate' to exit."