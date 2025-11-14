# Configuration Files

This directory contains all configuration files for the RL-LLM project using a **two-tier system** that separates model architecture from algorithm hyperparameters.

## Directory Structure

```
configs/
├── README.md              # This file
├── schema.json            # FinQA output schema (JSON format specification)
├── models/                # Model-specific configurations
│   ├── config_microsoft_DialoGPT_medium.yaml
│   ├── config_TinyLlama_1.1B_Chat.yaml
│   └── config_meta_llama_Llama_3_8B_Instruct.yaml
└── algorithms/            # RL algorithm hyperparameters
    ├── ppo.yaml           # PPO settings
    ├── grpo.yaml          # GRPO settings
    ├── rloo.yaml          # RLOO settings
    └── dpo.yaml           # DPO settings
```

## Design Philosophy

### Two-Tier System

**Tier 1: Model Configs** (`models/`)
- Model architecture (GPT-2, Llama, etc.)
- LoRA configuration
- Training parameters (batch size, learning rate, epochs)
- Paths (data_dir, output_dir)
- Model-specific metadata

**Tier 2: Algorithm Configs** (`algorithms/`)
- RL method hyperparameters only
- Independent of model choice
- Algorithm-specific settings (PPO clip_range, GRPO group_size, etc.)

**Why separate?**
- **No redundancy**: Each config has a single responsibility
- **Flexibility**: Mix any model with any algorithm
- **Maintainability**: Update model or algorithm independently
- **Clarity**: Clear separation of concerns

## Usage Patterns

### 1. Using Both Configs

Most common pattern - specify both model and algorithm:

```bash
# SFT training (model config only)
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml

# PPO training (model + algorithm configs)
python 05_train_ppo.py \
  --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --algo_config configs/algorithms/ppo.yaml

# GRPO training with different model
python 06_train_grpo.py \
  --config configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml \
  --algo_config configs/algorithms/grpo.yaml
```

### 2. Override Parameters

CLI arguments override config values:

```bash
# Use config but change batch size
python 05_train_ppo.py \
  --config configs/models/config_TinyLlama_1.1B_Chat.yaml \
  --algo_config configs/algorithms/ppo.yaml \
  --batch_size 8 \
  --clip_range 0.3
```

### 3. Mix and Match

Try different combinations:

```bash
# Same model, different algorithms
python 05_train_ppo.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --algo_config configs/algorithms/ppo.yaml
python 06_train_grpo.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --algo_config configs/algorithms/grpo.yaml
python 07_train_rloo.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --algo_config configs/algorithms/rloo.yaml

# Same algorithm, different models
python 05_train_ppo.py --config configs/models/config_microsoft_DialoGPT_medium.yaml --algo_config configs/algorithms/ppo.yaml
python 05_train_ppo.py --config configs/models/config_TinyLlama_1.1B_Chat.yaml --algo_config configs/algorithms/ppo.yaml
python 05_train_ppo.py --config configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml --algo_config configs/algorithms/ppo.yaml
```

## Model Configs

Each model config (`configs/models/*.yaml`) includes:

```yaml
model:
  name: "..."              # HuggingFace model ID
  type: "gpt2|llama|..."   # Architecture type
  # ... model specs

lora:
  use_lora: true
  r: 16
  target_modules: [...]    # Model-specific LoRA targets

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-5
  # ... training params

paths:
  data_dir: "..."
  output_dir: "..."
```

**Available Models:**
- `config_microsoft_DialoGPT_medium.yaml` - Fast testing (863MB)
- `config_TinyLlama_1.1B_Chat.yaml` - Llama testing (2.2GB)
- `config_meta_llama_Llama_3_8B_Instruct.yaml` - Production (15GB)

## Algorithm Configs

Each algorithm config (`configs/algorithms/*.yaml`) includes only method-specific hyperparameters:

**PPO** (`ppo.yaml`):
```yaml
clip_range: 0.2
kl_coef: 0.05
ppo_epochs: 4
# ... PPO-specific params
```

**GRPO** (`grpo.yaml`):
```yaml
group_size: 4
use_batch_bonus: true
group_baseline: "mean"
# ... GRPO-specific params
```

**RLOO** (`rloo.yaml`):
```yaml
num_samples: 4
baseline_type: "loo"
use_reward_whitening: true
# ... RLOO-specific params
```

**DPO** (`dpo.yaml`):
```yaml
beta: 0.1
loss_type: "sigmoid"
reference_free: false
# ... DPO-specific params
```

## Schema File

`schema.json` defines the expected JSON output format for FinQA:

```json
{
  "type": "object",
  "properties": {
    "answer": {...},      // Required
    "program": {...},     // Optional
    "reasoning": {...}    // Optional
  },
  "required": ["answer"]
}
```

Used by:
- Reward function (`02_build_rewards.py`)
- Dataset preparation (`01_prepare_dataset.py`)
- Evaluation (`10_evaluate.py`)

## Adding New Configs

### New Model Config

1. Use inspection tool:
   ```bash
   python inspect_model_architecture.py --model <hf_model_name>
   ```

2. Or manually create `configs/models/config_<model_name>.yaml` following the template

### New Algorithm Config

1. Create `configs/algorithms/<method>.yaml`
2. Include only algorithm-specific hyperparameters
3. Document key parameters in comments
4. Add usage example in this README

## Best Practices

1. **Don't duplicate**: Keep model info in model configs, algorithm info in algorithm configs
2. **Use comments**: Document non-obvious hyperparameters
3. **Set defaults**: Choose sensible defaults that work out of the box
4. **Version control**: Commit config changes with clear messages
5. **Document experiments**: Note which configs were used in run READMEs

## Migration Notes

**Old structure** (deprecated):
```
configs/
├── base.yaml        # ❌ Redundant
├── sft.yaml         # ❌ Redundant
└── ppo.yaml         # ❌ Mixed model + algorithm
```

**New structure**:
```
configs/
├── models/          # ✅ Model-specific only
└── algorithms/      # ✅ Algorithm-specific only
```

If you have old configs, migrate by:
1. Model params → `configs/models/*.yaml`
2. Algorithm params → `configs/algorithms/*.yaml`
3. Delete `base.yaml` and `sft.yaml`
