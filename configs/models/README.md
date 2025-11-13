# Model-Specific Configuration Files

This directory contains model-specific YAML configuration files for training with different LLM architectures.

## Available Configs

### DialoGPT Models (GPT-2 Architecture)
- **`config_microsoft_DialoGPT_medium.yaml`**
  - Model: microsoft/DialoGPT-medium
  - Size: ~863MB
  - LoRA targets: `c_attn`, `c_proj`
  - Batch size: 4, Gradient accum: 4
  - Good for: Fast testing and development

### Llama Models
- **`config_TinyLlama_1.1B_Chat.yaml`**
  - Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Size: ~2.2GB
  - LoRA targets: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - Batch size: 4, Gradient accum: 4
  - Good for: Testing Llama architecture with lower memory

- **`config_meta_llama_Llama_3_8B_Instruct.yaml`**
  - Model: meta-llama/Llama-3-8B-Instruct
  - Size: ~15GB
  - LoRA targets: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - Batch size: 2, Gradient accum: 8
  - Good for: Production/research results
  - **Note**: Requires HuggingFace authentication and model access approval

## Usage

### 1. Using Config File
```bash
# Train with DialoGPT config
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml

# Train with TinyLlama config
python 03_sft_train.py --config configs/models/config_TinyLlama_1.1B_Chat.yaml

# Quick test with config
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --quick_test --skip_validation
```

### 2. Overriding Config Parameters
Command-line arguments override YAML values:
```bash
# Use config but change batch size
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --batch_size 8

# Use config but change epochs
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --epochs 5
```

### 3. Generate New Configs
Use the inspection tool to generate configs for new models:
```bash
# Inspect and generate config for a specific model
python inspect_model_architecture.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Generate configs for all default models
python inspect_model_architecture.py --all

# Just inspect without generating config
python inspect_model_architecture.py --model microsoft/DialoGPT-medium --no_config
```

## Config Structure

Each YAML file contains:

```yaml
model:
  name: model/path           # HuggingFace model identifier
  type: gpt2/llama/etc       # Model architecture type
  architecture: ModelClass   # Specific class name
  hidden_size: 1024          # Hidden dimension size
  num_layers: 24             # Number of transformer layers
  vocab_size: 50257          # Vocabulary size

lora:
  use_lora: true             # Enable LoRA fine-tuning
  r: 16                      # LoRA rank (lower = fewer params)
  alpha: 32                  # LoRA scaling factor
  dropout: 0.05              # LoRA dropout rate
  target_modules:            # Which layers to apply LoRA
    - q_proj                 # (model-specific)
    - v_proj

training:
  epochs: 3                  # Number of training epochs
  batch_size: 4              # Per-device batch size
  gradient_accumulation_steps: 4  # Effective batch = batch_size * this
  learning_rate: 2.0e-05     # Learning rate
  warmup_steps: 100          # LR warmup steps
  max_length: 512            # Max sequence length
  fp16: true                 # Use mixed precision

validation:
  eval_steps: 500            # Validate every N steps
  save_steps: 1000           # Save checkpoint every N steps
  logging_steps: 100         # Log metrics every N steps

generation:
  max_new_tokens: 128        # Max tokens to generate
  temperature: 0.7           # Sampling temperature
  top_p: 0.9                 # Nucleus sampling threshold

paths:
  data_dir: datasets/finqa_processed
  output_dir: outputs/finqa_rl/run_001/03_sft_model_name
  reward_spec: outputs/finqa_rl/02_rewards/reward_spec.yaml

metadata:
  generated_by: inspect_model_architecture.py
  all_available_layers:      # All layer types in the model
    - layer_name_1
    - layer_name_2
```

## Key Differences Between Architectures

### GPT-2 Style (DialoGPT)
- **LoRA Targets**: `c_attn`, `c_proj`
- **Attention**: Combined QKV projection (`c_attn`)
- **Memory**: Lower requirements
- **Speed**: Faster training

### Llama Style (Llama, TinyLlama, Mistral)
- **LoRA Targets**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Attention**: Separate Q, K, V projections
- **Memory**: Higher requirements (especially Llama-3-8B)
- **Quality**: Often better for complex reasoning

## Adjusting for Your Hardware

### If you have limited GPU memory:
```yaml
training:
  batch_size: 1              # Reduce batch size
  gradient_accumulation_steps: 16  # Increase grad accum
  fp16: true                 # Use mixed precision
```

### If you have plenty of GPU memory:
```yaml
training:
  batch_size: 8              # Increase batch size
  gradient_accumulation_steps: 2  # Reduce grad accum
lora:
  r: 32                      # Increase LoRA rank for better quality
```

## Tips

1. **Start with DialoGPT** for fast iteration and testing
2. **Use TinyLlama** to test Llama-specific code without huge downloads
3. **Use Llama-3-8B** for final research results
4. **Adjust batch_size × gradient_accumulation_steps** to match your GPU memory
5. **Effective batch size** = batch_size × gradient_accumulation_steps × num_gpus
6. Keep **effective batch size around 16-32** for stable training
