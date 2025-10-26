#!/bin/bash

# RL-LLM Project - Unified Data & Model Download Script
# This script handles all downloads for datasets and models using modern APIs

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default settings
DOWNLOAD_DATASET=true
DOWNLOAD_MODEL=true
MODEL_NAME="microsoft/DialoGPT-medium"  # Small model for testing
FULL_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"  # Full research model
USE_FULL_MODEL=false
OUTPUT_DIR="datasets/finqa"
MODEL_DIR="models"
HF_TOKEN=""
FORCE_DOWNLOAD=false

show_help() {
    echo "RL-LLM Project Download Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset-only       Download only FinQA dataset"
    echo "  --model-only         Download only the model"
    echo "  --full-model         Download Llama-3-8B (~15GB) instead of test model"
    echo "  --model NAME         Specify custom model name"
    echo "  --token TOKEN        HuggingFace authentication token"
    echo "  --output-dir DIR     Dataset output directory (default: datasets/finqa)"
    echo "  --model-dir DIR      Model output directory (default: models)"
    echo "  --force              Force re-download even if files exist"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Download dataset + small test model"
    echo "  $0 --full-model                      # Download dataset + full research model"
    echo "  $0 --dataset-only                    # Only download FinQA dataset"
    echo "  $0 --model-only --model mistralai/Mistral-7B-Instruct-v0.1"
    echo "  $0 --force                           # Re-download everything even if exists"
    echo ""
    echo "Popular Models:"
    echo "  microsoft/DialoGPT-medium             # Small test model (~500MB)"
    echo "  meta-llama/Meta-Llama-3-8B-Instruct  # Full research model (~15GB)"
    echo "  mistralai/Mistral-7B-Instruct-v0.1   # Alternative 7B model"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-only)
            DOWNLOAD_MODEL=false
            shift
            ;;
        --model-only)
            DOWNLOAD_DATASET=false
            shift
            ;;
        --full-model)
            USE_FULL_MODEL=true
            MODEL_NAME=$FULL_MODEL
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation checks
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Virtual environment not activated!"
    print_status "Please run: source activate.sh"
    exit 1
fi

if ! command -v python &> /dev/null; then
    print_error "Python not found in PATH"
    exit 1
fi

# Check required Python packages
check_python_package() {
    if ! python -c "import $1" 2>/dev/null; then
        print_warning "Installing missing package: $1"
        pip install $1
    fi
}

print_status "RL-LLM Project Download Starting..."
print_status "Configuration:"
print_status "  - Download Dataset: $DOWNLOAD_DATASET"
print_status "  - Download Model: $DOWNLOAD_MODEL"
print_status "  - Model: $MODEL_NAME"
print_status "  - Dataset Dir: $OUTPUT_DIR"
print_status "  - Model Dir: $MODEL_DIR"
print_status "  - Force Download: $FORCE_DOWNLOAD"

# Create directories
mkdir -p "$OUTPUT_DIR" "$MODEL_DIR"

#=============================================================================
# SMART CHECK FUNCTIONS
#=============================================================================
check_dataset_exists() {
    # Check if dataset already exists and is valid
    if [ -f "$OUTPUT_DIR/train.json" ] && [ -f "$OUTPUT_DIR/dev.json" -o -f "$OUTPUT_DIR/val.json" ] && [ -f "$OUTPUT_DIR/test.json" ]; then
        # Validate files are not empty and contain JSON
        for file in "$OUTPUT_DIR"/*.json; do
            if [ -f "$file" ] && [ -s "$file" ]; then
                # Quick JSON validation
                if python -c "import json; json.load(open('$file'))" 2>/dev/null; then
                    continue
                else
                    return 1  # Invalid JSON found
                fi
            else
                return 1  # Empty or missing file
            fi
        done
        return 0  # All files exist and valid
    else
        return 1  # Missing files
    fi
}

check_model_exists() {
    local model_path="$MODEL_DIR/$(basename $MODEL_NAME)"
    
    # Check if model directory exists with required files
    if [ -d "$model_path" ]; then
        # Check for essential model files
        if [ -f "$model_path/config.json" ] && [ -f "$model_path/tokenizer_config.json" ]; then
            # Check if model weights exist (either pytorch_model.bin or model.safetensors)
            if [ -f "$model_path/pytorch_model.bin" ] || [ -f "$model_path/model.safetensors" ] || ls "$model_path"/pytorch_model-*.bin >/dev/null 2>&1; then
                return 0  # Model exists and looks complete
            fi
        fi
    fi
    return 1  # Model missing or incomplete
}

#=============================================================================
# DATASET DOWNLOAD FUNCTION
#=============================================================================
download_finqa_dataset() {
    print_status "Checking FinQA dataset..."
    
    # Check if dataset already exists (skip if force download)
    if [ "$FORCE_DOWNLOAD" = false ] && check_dataset_exists; then
        print_success "✅ FinQA dataset already exists and is valid"
        print_status "Files found:"
        for file in "$OUTPUT_DIR"/*.json; do
            if [ -f "$file" ]; then
                count=$(python -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "?")
                size=$(du -sh "$file" | cut -f1)
                print_status "  - $(basename $file): $count examples ($size)"
            fi
        done
        print_status "(Use --force to re-download)"
        return 0
    fi
    
    print_status "Dataset not found or incomplete. Downloading FinQA dataset..."
    
    # Download FinQA dataset via git clone (reliable method)
    if command -v git >/dev/null 2>&1; then
        TEMP_DIR=$(dirname "$OUTPUT_DIR")/temp_finqa
        print_status "Cloning FinQA repository from GitHub..."
        
        if git clone https://github.com/czyssrs/FinQA.git "$TEMP_DIR" >/dev/null 2>&1; then
            print_status "Git clone successful, copying dataset files..."
            mkdir -p "$OUTPUT_DIR"
            cp "$TEMP_DIR/dataset"/*.json "$OUTPUT_DIR/"
            
            # Rename dev.json to val.json for consistency
            if [ -f "$OUTPUT_DIR/dev.json" ]; then
                mv "$OUTPUT_DIR/dev.json" "$OUTPUT_DIR/val.json"
            fi
            
            # Clean up temporary directory
            rm -rf "$TEMP_DIR"
            
            print_success "✅ FinQA dataset downloaded successfully to $OUTPUT_DIR"
            
            # Show summary
            for file in "$OUTPUT_DIR"/*.json; do
                if [ -f "$file" ]; then
                    count=$(python -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "?")
                    size=$(du -sh "$file" | cut -f1)
                    print_status "  - $(basename $file): $count examples ($size)"
                fi
            done
        else
            print_error "Git clone failed"
            print_status "Manual download instructions:"
            echo "1. Visit: https://github.com/czyssrs/FinQA"
            echo "2. Clone or download the repository"
            echo "3. Copy dataset files to $OUTPUT_DIR/"
            echo "   Required: train.json, dev.json (rename to val.json), test.json"
            return 1
        fi
    else
        print_error "Git not available - please install git first"
        print_status "Install git: sudo apt install git"
        return 1
    fi
}

#=============================================================================
# MODEL DOWNLOAD FUNCTION
#=============================================================================
download_model() {
    print_status "Checking model: $MODEL_NAME"
    
    model_output_dir="$MODEL_DIR/$(basename $MODEL_NAME)"
    
    # Check if model already exists (skip if force download)
    if [ "$FORCE_DOWNLOAD" = false ] && check_model_exists; then
        print_success "✅ Model already exists and is complete"
        
        # Show model info
        size=$(du -sh "$model_output_dir" 2>/dev/null | cut -f1 || echo "Unknown")
        print_status "Model info:"
        print_status "  - Location: $model_output_dir"
        print_status "  - Size: $size"
        
        # Try to get model parameters if possible
        if [ -f "$model_output_dir/config.json" ]; then
            params=$(python -c "
import json
try:
    config = json.load(open('$model_output_dir/config.json'))
    if 'n_parameters' in config:
        print(f\"{config['n_parameters'] / 1e9:.1f}B\")
    elif 'hidden_size' in config and 'num_hidden_layers' in config:
        # Rough estimate for transformer models
        h = config['hidden_size']
        l = config['num_hidden_layers']
        approx = (12 * h * h * l) / 1e9
        print(f\"~{approx:.1f}B (estimated)\")
    else:
        print('Unknown')
except:
    print('Unknown')
" 2>/dev/null || echo "Unknown")
            print_status "  - Parameters: $params"
        fi
        
        print_status "(Use --force to re-download)"
        return 0
    fi
    
    print_status "Model not found or incomplete. Downloading model: $MODEL_NAME"
    
    if [ "$USE_FULL_MODEL" = true ]; then
        print_warning "Downloading full model (~15GB). This may take a while..."
    fi
    
    # Check for required packages
    check_python_package "transformers"
    check_python_package "torch"
    
    model_output_dir="$MODEL_DIR/$(basename $MODEL_NAME)"
    
    # Create Python script inline for model download
    python << EOF
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "$MODEL_NAME"
        output_dir = "$model_output_dir"
        token = "$HF_TOKEN" if "$HF_TOKEN" else None
        
        logger.info(f"Downloading model: {model_name}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        # Download model with appropriate settings
        logger.info("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token=token,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Save locally
        logger.info(f"Saving model to {output_dir}")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model downloaded successfully!")
        logger.info(f"  - Parameters: ~{param_count / 1e9:.1f}B")
        logger.info(f"  - Vocab size: {tokenizer.vocab_size}")
        logger.info(f"  - Location: {output_dir}")
        
        print("SUCCESS")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        print("FAILED")

download_model()
EOF

    if [ $? -eq 0 ]; then
        if [ -d "$model_output_dir" ]; then
            print_success "✅ Model downloaded to $model_output_dir"
            
            # Show model size
            size=$(du -sh "$model_output_dir" 2>/dev/null | cut -f1 || echo "Unknown")
            print_status "  - Size: $size"
        else
            print_error "Model download failed"
            print_status "Alternative: Model will download automatically during training"
        fi
    fi
}

#=============================================================================
# MAIN EXECUTION
#=============================================================================

# Download dataset if requested
if [ "$DOWNLOAD_DATASET" = true ]; then
    download_finqa_dataset
fi

# Download model if requested
if [ "$DOWNLOAD_MODEL" = true ]; then
    download_model
fi

# Final summary
print_status "Checking downloads..."

# Check storage usage
total_size=$(du -sh . 2>/dev/null | cut -f1 || echo "Unknown")
print_status "Current project size: $total_size"

# Create download manifest
cat > download_manifest.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "dataset_downloaded": $DOWNLOAD_DATASET,
    "model_downloaded": $DOWNLOAD_MODEL,
    "model_name": "$MODEL_NAME",
    "dataset_dir": "$OUTPUT_DIR",
    "model_dir": "$MODEL_DIR",
    "project_size": "$total_size"
}
EOF

print_success "Download process completed!"
print_status "Manifest saved to: download_manifest.json"
print_status ""
print_status "Next steps:"
echo "  1. Validate data:    python src/00_check_data.py --data_root $OUTPUT_DIR"
echo "  2. Prepare dataset:  python src/01_prepare_dataset.py"
echo "  3. Start training:   python src/03_sft_train.py"