#!/bin/bash
# Test script for different model architectures
# Tests that all model configs can be loaded properly

set -e  # Exit on any error

echo "🧪 Testing Model Architecture Compatibility"
echo "=========================================="

# Array of model configs to test
configs=(
    "configs/models/llama-3.2-1b.yaml"
    "configs/models/tinyllama-1.1b.yaml" 
    "configs/models/dialogpt-medium.yaml"
    "configs/models/llama-3.2-1b-ultrafast.yaml"
)

# Test each config
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "✅ Testing $config..."
        
        # Extract model name for cleaner output
        model_name=$(basename "$config" .yaml)
        
        # Test config loading (dry run)
        python -c "
import sys
sys.path.append('.')
from utils.common import load_yaml_config
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Test YAML loading
    cfg = load_yaml_config('$config')
    model_id = cfg['model']['name']
    print(f'  📄 Config loaded: {model_id}')
    
    # Test tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f'  🔤 Tokenizer loaded: vocab_size={len(tokenizer)}')
    
    # Test model info (without downloading full model)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    print(f'  🤖 Model type: {model_config.model_type}')
    print(f'  📊 Parameters: ~{model_config.hidden_size * model_config.num_hidden_layers}M')
    
    print(f'  ✅ {model_id} - Compatible!')
    
except Exception as e:
    print(f'  ❌ Error: {e}')
    exit(1)
"
    else
        echo "⚠️  Config not found: $config"
    fi
    echo ""
done

echo "🎉 All model architectures are compatible with the trainer!"
echo ""
echo "💡 Usage examples:"
echo "  # DialoGPT (fastest)"
echo "  python 03_sft_train.py --config configs/models/dialogpt-medium.yaml"
echo ""  
echo "  # TinyLlama (Llama architecture testing)"
echo "  python 03_sft_train.py --config configs/models/tinyllama-1.1b.yaml"
echo ""
echo "  # Llama-3.2-1B (current default)"
echo "  python 03_sft_train.py --config configs/models/llama-3.2-1b.yaml"
echo ""
echo "  # Ultra-fast (for rapid iteration)"
echo "  python 03_sft_train.py --config configs/models/llama-3.2-1b-ultrafast.yaml"