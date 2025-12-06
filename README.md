# RL-LLM: Reinforcement Learning for Financial Question Answering

**Optimizing LLMs with PPO for Mathematical Reasoning on FinQA**

This project trains language models using Proximal Policy Optimization (PPO) to improve accuracy on financial question-answering tasks. It uses a classification-based approach where models learn to select the best answer from multiple candidates, optimized for speed and memory efficiency on multi-GPU setups.

> **Goal**: Train Llama models with supervised fine-tuning + PPO to achieve high accuracy on mathematical reasoning in financial contexts.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 2-4 CUDA GPUs with 24GB VRAM each (tested on 2× RTX 4090)
- Linux environment

### Installation

```bash
# 1. Clone repository
git clone https://github.com/jinyoonok2/RL-LLM-Reinforcement-Learning-Project.git
cd RL-LLM-Reinforcement-Learning-Project

# 2. Setup environment
./setup.sh        # Creates venv and installs dependencies
source activate.sh # Activates environment

# 3. Download FinQA dataset and Llama-3.2-3B model
bash download.sh
```

### Training Pipeline

Run the complete pipeline or individual steps:

```bash
# Full pipeline (data prep → SFT → PPO)
./run_full_pipeline.sh

# Or run steps individually:
python 00_check_data.py              # Verify dataset integrity
python 01_prepare_dataset.py         # Create train/val/test splits
python 04_generate_candidates.py     # Generate answer candidates
python 02_build_rewards.py           # Calculate rewards for candidates
python 04_sft_train.py               # Supervised fine-tuning (3B model)
python 05_train_ppo.py --policy_ckpt outputs/run_001/04_sft_llama3b/best_model
```

### Model Options

**Llama-3.2-3B** (default, recommended):
- Speed: ~2-3 hours SFT, ~4 hours PPO on 2 GPUs
- Quality: Good balance of performance and speed
- Config: `configs/models/llama-3.2-3b.yaml`

**Llama-3-8B** (higher quality):
- Speed: ~3.5 hours SFT, ~9 hours PPO on 4 GPUs
- Quality: Best performance
- Config: `configs/models/llama-3-8b.yaml`
- Usage: `python 04_sft_train.py --config configs/models/llama-3-8b.yaml`

## 📁 Project Structure

### Core Training Scripts
```
00_check_data.py              # Validate FinQA dataset integrity
01_prepare_dataset.py         # Split data into train/val/test
04_generate_candidates.py     # Generate answer candidates from SFT model
02_build_rewards.py           # Calculate rewards for candidate answers
04_sft_train.py               # Supervised fine-tuning (classification mode)
05_train_ppo.py               # PPO reinforcement learning training
```

### Configuration Files
```
configs/
├── models/
│   ├── llama-3.2-3b.yaml     # 3B model (default, optimized for 2 GPUs)
│   └── llama-3-8b.yaml       # 8B model (higher quality, needs 4 GPUs)
└── algorithms/
    ├── ppo.yaml              # PPO hyperparameters
    └── grpo.yaml             # GRPO configuration (future)
```

### Utilities
```
utils/
├── common.py                 # Logging, config loading, file I/O
├── evaluation.py             # Metrics and evaluation functions
├── rewards.py                # Reward calculation logic
└── trainer.py                # Shared training utilities
```

### Key Helper Scripts
```
run_full_pipeline.sh          # Execute complete training pipeline
check_token_lengths.py        # Analyze dataset token distributions
setup.sh                      # Environment setup
download.sh                   # Download dataset and models
```
│   │   └── llama-3-8b.yaml        # Production: 8B model
│   └── algorithms/                # RL algorithm configs
│       ├── ppo.yaml
│       ├── grpo.yaml
│       ├── rloo.yaml
│       └── dpo.yaml
│
├── 00_check_data.py               # ✅ Data validation
├── 01_prepare_dataset.py          # ✅ Dataset preparation  
├── 02_build_rewards.py            # ✅ Reward function implementation
├── 03_sft_train.py                # ✅ Supervised fine-tuning
├── 04_generate_candidates.py      # ✅ Candidate generation for RL
├── 05_train_ppo.py                # 🚧 PPO training (in progress)
├── 06_train_grpo.py               # 🚧 GRPO training (planned)
├── 07_train_rloo.py               # 🚧 RLOO training (planned)
├── 08_build_prefs.py              # 🚧 Preference construction (planned)
├── 09_train_dpo.py                # 🚧 DPO training (planned)
├── 10_evaluate.py                 # 🚧 Evaluation (planned)
├── 11_compare_runs.py             # 🚧 Results comparison (planned)
│
├── utils/                         # Shared utilities
│   ├── rewards.py                 # FinQA reward calculator
│   └── common.py                  # Logging, manifest writing
│
├── datasets/finqa/                # Original FinQA data
├── datasets/finqa_processed/      # Preprocessed JSONL files
└── outputs/                       # Experiment outputs
    └── run_001/                   # First experimental run
        ├── 02_rewards/            # Reward spec
        ├── 03_sft/                # SFT checkpoints
        └── 04_candidates/         # Generated candidates
```

## 🔬 Implementation Details

### Architecture: Classification-Based Approach

Unlike traditional generation-based methods, this project uses a **classification approach** for efficiency:

**Input**: Question + 8 candidate answers  
**Output**: Score distribution over candidates  
**Training**: Cross-entropy to select highest-reward candidate  
**Benefits**: 
- Lower memory usage (no generation)
- Faster training (fixed sequence length)
- Better multi-GPU scaling
- Easier to optimize with RL

### Training Pipeline

1. **Data Preparation** (`00_check_data.py`, `01_prepare_dataset.py`)
   - Validates FinQA dataset (8,281 examples)
   - Creates train/val/test splits
   - Output: `datasets/finqa_with_rewards/`

2. **Candidate Generation** (`04_generate_candidates.py`)
   - Generates 8 diverse answer candidates per question
   - Uses sampling with temperature=0.7
   - Output: 66,248 total candidates

3. **Reward Calculation** (`02_build_rewards.py`)
   - Evaluates each candidate for correctness
   - Metrics: exact match, numeric tolerance, program match
   - Average reward: 0.637 (gold: 1.30, corrupted: 0.54)

4. **Supervised Fine-Tuning** (`04_sft_train.py`)
   - Trains model to select best candidate from pool
   - Uses LoRA (rank=16, alpha=32) for efficiency
   - 3 epochs, batch_size=2, max_length=256
   - Output: Policy initialization for PPO

5. **PPO Training** (`05_train_ppo.py`)
   - Optimizes candidate selection with PPO
   - 10 epochs, batch_size=12, single PPO inner epoch
   - Caches reference model outputs for speed
   - Expected: 4 hours on 2× RTX 4090

### Optimizations Applied

**Model Size**: Llama-3.2-3B (2.6× faster than 8B, better than 1B)  
**Sequence Length**: 256 tokens (99.9% coverage, 2× speedup vs 512)  
**Batch Size**: 12 for 2 GPUs (optimized for 48GB total VRAM)  
**LoRA**: rank=16, 4 target modules (0.27% trainable params)  
**PPO Epochs**: 1 inner epoch (50% speedup, still converges)  
**Reference Caching**: Compute ref logprobs once per batch  

**Combined Speedup**: ~6-8× vs baseline 8B/512 configuration

## 📊 Expected Results

### Performance Metrics

**SFT Baseline** (after supervised fine-tuning):
- Accuracy: ~90-92%
- Average reward: ~0.63-0.64

**PPO** (after reinforcement learning):
- Target accuracy: 93-95%
- Target avg reward: 0.70-0.80
- KL divergence: <0.01 (stays close to SFT policy)

### Training Time (2× RTX 4090)

| Stage | Duration | Memory |
|-------|----------|--------|
| Data Prep (00-02) | ~5 min | Minimal |
| SFT Training | ~2-3 hours | 22GB/GPU |
| PPO Training | ~4 hours | 23GB/GPU |
| **Total** | **~7 hours** | **48GB total** |

### Output Structure

```
outputs/run_001/
├── 04_sft_llama3b/
│   ├── best_model/              # Best SFT checkpoint
│   ├── final_model/             # Final SFT checkpoint
│   └── training_manifest.json   # Training metrics
└── 05_ppo/
    ├── best_model/              # Best PPO checkpoint
    ├── checkpoint_epoch_N/      # Periodic checkpoints
    └── final_model/             # Final PPO model
```
## ⚙️ Configuration

### Model Configs (`configs/models/`)

**llama-3.2-3b.yaml** (default):
```yaml
model:
  name: meta-llama/Llama-3.2-3B
  
lora:
  r: 16              # LoRA rank
  alpha: 32          # Scaling factor
  target_modules: [q_proj, v_proj, o_proj, gate_proj]

training:
  epochs: 3
  batch_size: 2      # For 2 GPUs
  gradient_accumulation_steps: 4
  max_length: 256    # Optimized for speed
  learning_rate: 2e-5
  bf16: true
```

**llama-3-8b.yaml** (high quality):
- Same structure, larger model
- Requires 4 GPUs or reduces batch size
- Longer training time (~3× slower)

### Algorithm Configs (`configs/algorithms/`)

**ppo.yaml**:
```yaml
ppo:
  learning_rate: 1e-5
  batch_size: 12
  ppo_epochs: 1           # Inner optimization steps
  clip_range: 0.2
  kl_coef: 0.05
  entropy_coef: 0.01
  total_epochs: 10
```

## 🔧 Model Configuration

### Available Models

1. **Llama-3.2-1B-Instruct** (Default)
   - Size: ~2.5GB
   - VRAM: ~4-6GB with LoRA
   - Speed: ~8x faster than 8B
   - Config: `configs/models/llama-3.2-1b.yaml`

2. **Meta-Llama-3-8B-Instruct** (Production)
   - Size: ~15GB  
   - VRAM: ~18-20GB with LoRA
   - Quality: Better performance
   - Config: `configs/models/llama-3-8b.yaml`

### Training Configuration

Both configs include:
- **LoRA**: r=32, alpha=64, dropout=0.05
- **Batch**: size=1, gradient_accumulation=4
- **Learning rate**: 5e-6 (stable for LoRA)
- **Epochs**: 3
- **FP16**: Enabled

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Use 1B model: `python 03_sft_train.py` (default)
- Or reduce batch size in config

**2. Model Download Fails**
```bash
## 🛠️ Troubleshooting

### Common Issues

**OOM Error during training**:
```bash
# Reduce batch size in config
training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Keep effective batch size
```

**Device mismatch errors**:
- Fixed in latest version
- Run `git pull origin main` to get updates

**Training won't resume from checkpoint**:
- PPO automatically detects and resumes from `checkpoint_epoch_N/`
- Check `outputs/run_001/05_ppo/` for existing checkpoints

**Model download issues**:
```bash
# Authenticate with HuggingFace
huggingface-cli login
# Accept Llama license at https://huggingface.co/meta-llama/Llama-3.2-3B
```

**Slow training**:
- Use 3B model instead of 8B (2.6× faster)
- Verify `max_length: 256` in config
- Check GPU utilization with `nvidia-smi`

### Performance Tips

1. **Token length optimization**: Run `python check_token_lengths.py` to verify max_length setting
2. **Multi-GPU setup**: Automatic with `device_map="auto"`
3. **Checkpoint resumption**: Enabled by default, saves progress every 2 epochs
4. **Memory monitoring**: Watch `nvidia-smi` during training

---

## 📝 Citation

```bibtex
@misc{rl-llm-finqa,
  author = {Jin Yoon Ok},
  title = {RL-LLM: Reinforcement Learning for Financial Question Answering},
  year = {2025},
  url = {https://github.com/jinyoonok2/RL-LLM-Reinforcement-Learning-Project}
}
```

## 📄 License

MIT License - see LICENSE file for details.

---

**Status**: Active development  
**Last Updated**: December 2025
- `samples/`: selected generations with targets and reward components
- `manifest.json`

**Verify**
- Metrics are reproducible given the same seed/ckpt
- EM and format metrics match (within tolerance) spot recomputation by 02's scorer
- Qualitative failures are interpretable (e.g., numeric off by rounding)

---

### 11_compare_runs.py — Reports & Ablations

**Goal**: Aggregate metrics across methods/seeds for the ablation study.

**Inputs**
- One or more `…/10_eval/metrics.json` paths
- Optional: glob like `outputs/finqa_rl/run_*/10_eval/metrics.json`

**Saves (in `outputs/finqa_rl/11_reports/`)**
- `summary.csv` (method, seed, EM, format, program validity, tokens/GPU-hrs if logged)
- `plots/` (score vs seed, compute vs score)
- `report.md` (bulleted comparisons + tables)
- `manifest.json`

**Verify**
- Numbers in `summary.csv` match source `metrics.json`
- Plots reflect expected trends (e.g., GRPO group size G vs stability)
- Report highlights best method and discusses stability/efficiency trade-offs

---

### 12_driver.py — Orchestrate a Full Run (Optional)

**Goal**: Automate a full cycle: 01 → 02 → 03 → (04) → [05|06|07|09] → 10 → 11.

**Inputs**
- `--method ppo|grpo|rloo|dpo`
- Passthrough flags to submodules or `--config configs/<method>.yaml`
- `--run_id run_00X`

**Saves (in `outputs/finqa_rl/<run_id>/`)**
- Re-creates the folder structure used by submodules
- `run_manifest.json`: pointers to all sub-manifests for provenance

**Verify**
- After one invocation, all expected subfolders exist
- Final metrics appear in `10_eval/metrics.json` and are included in `11_reports/summary.csv`

---

## 🔗 Module Dependencies

### Data Flow Pipeline

```
00_check_data.py → validates raw data
        ↓
01_prepare_dataset.py → train/val/test.jsonl
        ↓                        ↓
02_build_rewards.py → reward_spec.yaml
        ↓                        ↓
03_sft_train.py ← train/val.jsonl → ckpt_sft/
        ↓                        ↓
04_sample_candidates.py → candidates.jsonl
        ↓                        ↓
[05_ppo/06_grpo/07_rloo] → ckpt_<method>/
        ↓                        ↓
08_build_prefs.py → pairs.jsonl
        ↓
09_train_dpo.py → ckpt_dpo/
        ↓
10_evaluate.py → metrics.json
        ↓
11_compare_runs.py → final report
```

### Key File Dependencies

- `01_prepare_dataset.py` → produces `train/val/test.jsonl` used by training and evaluation modules
- `02_build_rewards.py` → `reward_spec.yaml` used by all RL methods and evaluation
- `03_sft_train.py` → `ckpt_sft/` used as starting point for all RL methods
- `04_sample_candidates.py` → `candidates.jsonl` used by DPO and can warm-start GRPO/RLOO
- `08_build_prefs.py` → `pairs.jsonl` used by DPO training
- `05/06/07/09_*` → `ckpt_<method>/` used by evaluation and comparison

## 📂 Output Directory Structure

All experiment outputs follow a unified hierarchy for consistency and easy comparison across runs.

### Structure Overview

```
outputs/
├── run_001/                    # First experimental run
│   ├── 00_data_validation/     # Data validation results
│   ├── 01_preprocessing/       # Dataset preprocessing outputs
│   ├── 02_rewards/             # Reward function specification
│   ├── 03_sft/                 # Supervised fine-tuning
│   ├── 04_candidates/          # Candidate generation
│   ├── 05_ppo/                 # PPO training
│   ├── 06_grpo/                # GRPO training
│   ├── 07_rloo/                # RLOO training
│   ├── 08_preferences/         # Preference pair construction
│   ├── 09_dpo/                 # DPO training
│   ├── 10_evaluation/          # Final evaluation
│   ├── 11_comparison/          # Cross-method comparison
│   └── README.md               # Run-specific notes
│
├── run_002/                    # Second experimental run (e.g., different model)
│   └── ...                     # Same structure
│
└── run_003/                    # Third experimental run (e.g., different hyperparams)
    └── ...                     # Same structure
```

### Unified Format

**Pattern**: `outputs/run_XXX/YY_module_name/`

Where:
- `run_XXX`: Run identifier (001, 002, 003, ...)
- `YY`: Module number (00-12)
- `module_name`: Descriptive name of the module

**Benefits**:
1. **Consistency**: All modules follow the same pattern
2. **Clarity**: Easy to find outputs for any module in any run
3. **Comparison**: Simple to compare same module across different runs
4. **Organization**: Keeps related experiments together
5. **Scalability**: Easy to add new runs or modules

### Module Output Details

**02_rewards/**
```
02_rewards/
├── reward_spec.yaml         # Reward function configuration
├── unit_tests/              # Unit test results
│   ├── test_results.json
│   └── test_cases.json
└── manifest.json            # Module metadata
```

**03_sft/** (or **03_sft_<model_name>/**)
```
03_sft/
├── ckpt_sft/                # Model checkpoints
│   ├── best/                # Best validation checkpoint
│   ├── final/               # Final checkpoint
│   └── step_N/              # Intermediate checkpoints
├── logs/                    # Training logs
│   └── training_log.json
├── valid_samples/           # Validation samples
│   ├── step_500.json
│   ├── step_1000.json
│   └── ...
└── manifest.json            # Module metadata
```

**10_evaluation/**
```
10_evaluation/
├── metrics.json             # Final metrics
├── breakdown.csv            # Per-item breakdown
├── samples/                 # Sample predictions
│   └── predictions.jsonl
└── manifest.json            # Module metadata
```

**11_comparison/**
```
11_comparison/
├── summary.csv              # Cross-method summary
├── plots/                   # Comparison plots
│   ├── reward_vs_method.png
│   └── efficiency_vs_quality.png
├── report.md                # Analysis report
└── manifest.json            # Module metadata
```

### Run Management

**Creating a New Run**

```bash
# Default: run_001
python 02_build_rewards.py
python 03_sft_train.py

# Specific run ID
python 02_build_rewards.py --output_dir outputs/run_002/02_rewards
python 03_sft_train.py --output_dir outputs/run_002/03_sft
```

**Using Configs**

Model configs automatically set the correct paths:

```bash
# Uses outputs/run_001/03_sft_microsoft_DialoGPT_medium/
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml

# Override to use different run
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --output_dir outputs/run_002/03_sft_dialogpt
```

**Run Organization Tips**

1. **run_001**: Initial baseline experiments
2. **run_002**: Hyperparameter tuning
3. **run_003**: Different model architecture
4. **run_004**: Final production run
5. Add `README.md` in each run directory to document experiment goals

**Example run_001/README.md**:
```markdown
# Run 001 - Baseline Experiments

**Goal**: Establish baseline performance with DialoGPT-medium

**Config**:
- Model: microsoft/DialoGPT-medium
- LoRA rank: 16
- Batch size: 4
- Learning rate: 2e-5
- Epochs: 3

**Results**:
- Best validation reward: 0.65
- Parse rate: 85%
- Training time: 2.5 hours

**Notes**:
- Good baseline for comparison with Llama models
```

**Cross-Run Comparisons**

```bash
# Compare reward specs
diff outputs/run_001/02_rewards/reward_spec.yaml \
     outputs/run_002/02_rewards/reward_spec.yaml

# Compare final metrics
python 11_compare_runs.py \
  outputs/run_001/10_evaluation/metrics.json \
  outputs/run_002/10_evaluation/metrics.json \
  outputs/run_003/10_evaluation/metrics.json
```

## 🤖 Configuration System

The project uses a two-tier configuration system that separates model architecture from algorithm hyperparameters.

### Configuration Overview

**Two-Tier Design:**
1. **Model Configs** (`configs/models/*.yaml`): Model architecture, LoRA settings, training parameters, paths
2. **Algorithm Configs** (`configs/algorithms/*.yaml`): RL method-specific hyperparameters only

**Benefits:**
- Mix and match any model with any algorithm
- No redundancy between configs
- Clear separation of concerns
- Easy to add new models or algorithms

### Model Configs

Model configurations define the architecture, LoRA settings, and training parameters.

### Available Model Configs

**1. DialoGPT (GPT-2 Architecture)**
- **File**: `configs/models/config_microsoft_DialoGPT_medium.yaml`
- **Model**: microsoft/DialoGPT-medium
- **Size**: ~863MB
- **LoRA targets**: `c_attn`, `c_proj`
- **Batch size**: 4, Gradient accum: 4
- **Use case**: Fast testing and development

**2. TinyLlama (Small Llama Architecture)**
- **File**: `configs/models/config_TinyLlama_1.1B_Chat.yaml`
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Size**: ~2.2GB
- **LoRA targets**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Batch size**: 4, Gradient accum: 4
- **Use case**: Testing Llama architecture with lower memory

**3. Llama-3-8B (Production Llama Architecture)**
- **File**: `configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml`
- **Model**: meta-llama/Llama-3-8B-Instruct
- **Size**: ~15GB
- **LoRA targets**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Batch size**: 2, Gradient accum: 8
- **Use case**: Production/research results
- **Note**: Requires HuggingFace authentication and model access approval

### Usage

**1. Using Config File**
```bash
# Train with DialoGPT config
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml

# Train with TinyLlama config
python 03_sft_train.py --config configs/models/config_TinyLlama_1.1B_Chat.yaml

# Quick test with config
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --quick_test --skip_validation
```

**2. Overriding Config Parameters**

Command-line arguments override YAML values:
```bash
# Use config but change batch size
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --batch_size 8

# Use config but change epochs
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --epochs 5
```

**3. Generate New Configs**

Use the inspection tool to generate configs for new models:
```bash
# Inspect and generate config for a specific model
python inspect_model_architecture.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Generate configs for all default models
python inspect_model_architecture.py --all

# Just inspect without generating config
python inspect_model_architecture.py --model microsoft/DialoGPT-medium --no_config
```

### Config Structure

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
  output_dir: outputs/run_001/03_sft_<model_name>
  reward_spec: outputs/run_001/02_rewards/reward_spec.yaml

metadata:
  generated_by: inspect_model_architecture.py
  all_available_layers:      # All layer types in the model
    - layer_name_1
    - layer_name_2
```

### Key Differences Between Architectures

**GPT-2 Style (DialoGPT)**
- **LoRA Targets**: `c_attn`, `c_proj`
- **Attention**: Combined QKV projection (`c_attn`)
- **Memory**: Lower requirements
- **Speed**: Faster training

**Llama Style (Llama, TinyLlama, Mistral)**
- **LoRA Targets**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Attention**: Separate Q, K, V projections
- **Memory**: Higher requirements (especially Llama-3-8B)
- **Quality**: Often better for complex reasoning

### Adjusting for Your Hardware

**If you have limited GPU memory**:
```yaml
training:
  batch_size: 1              # Reduce batch size
  gradient_accumulation_steps: 16  # Increase grad accum
  fp16: true                 # Use mixed precision
```

**If you have plenty of GPU memory**:
```yaml
training:
  batch_size: 8              # Increase batch size
  gradient_accumulation_steps: 2  # Reduce grad accum
lora:
  r: 32                      # Increase LoRA rank for better quality
```

### Model Selection Tips

1. **Start with DialoGPT** for fast iteration and testing
2. **Use TinyLlama** to test Llama-specific code without huge downloads
3. **Use Llama-3-8B** for final research results
4. **Adjust batch_size × gradient_accumulation_steps** to match your GPU memory
5. **Effective batch size** = batch_size × gradient_accumulation_steps × num_gpus
6. Keep **effective batch size around 16-32** for stable training

### Algorithm Configs

Algorithm configurations contain only RL method-specific hyperparameters.

**Available Algorithms:**

**1. PPO (Proximal Policy Optimization)**
- **File**: `configs/algorithms/ppo.yaml`
- **Key params**: clip_range=0.2, kl_coef=0.05, ppo_epochs=4
- **Use case**: General-purpose RL with value function
- **Usage**:
  ```bash
  python 05_train_ppo.py \
    --config configs/models/config_microsoft_DialoGPT_medium.yaml \
    --algo_config configs/algorithms/ppo.yaml
  ```

**2. GRPO (Group Relative Policy Optimization)**
- **File**: `configs/algorithms/grpo.yaml`
- **Key params**: group_size=4, use_batch_bonus=true
- **Use case**: Critic-free, group-based optimization
- **Usage**:
  ```bash
  python 06_train_grpo.py \
    --config configs/models/config_TinyLlama_1.1B_Chat.yaml \
    --algo_config configs/algorithms/grpo.yaml
  ```

**3. RLOO (REINFORCE Leave-One-Out)**
- **File**: `configs/algorithms/rloo.yaml`
- **Key params**: num_samples=4, baseline_type="loo"
- **Use case**: Variance-reduced REINFORCE
- **Usage**:
  ```bash
  python 07_train_rloo.py \
    --config configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml \
    --algo_config configs/algorithms/rloo.yaml
  ```

**4. DPO (Direct Preference Optimization)**
- **File**: `configs/algorithms/dpo.yaml`
- **Key params**: beta=0.1, loss_type="sigmoid"
- **Use case**: Preference-based training
- **Usage**:
  ```bash
  python 09_train_dpo.py \
    --config configs/models/config_microsoft_DialoGPT_medium.yaml \
    --algo_config configs/algorithms/dpo.yaml
  ```

### Config Combination Examples

**Fast Testing** (DialoGPT + PPO):
```bash
python 05_train_ppo.py \
  --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --algo_config configs/algorithms/ppo.yaml \
  --quick_test
```

**Production Run** (Llama-3-8B + GRPO):
```bash
python 06_train_grpo.py \
  --config configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml \
  --algo_config configs/algorithms/grpo.yaml \
  --output_dir outputs/run_002/06_grpo
```

**Override Algorithm Params**:
```bash
python 05_train_ppo.py \
  --config configs/models/config_TinyLlama_1.1B_Chat.yaml \
  --algo_config configs/algorithms/ppo.yaml \
  --clip_range 0.3 \
  --kl_coef 0.1
```

## 🎯 Reward Components

The reward system evaluates three key aspects:

- **Outcome (exact)**: exact-match or numeric-with-tolerance correctness
- **Program validity**: executable and correct intermediate steps (when available)
- **Format compliance**: JSON parses and contains all required fields

Weights are configurable in `reward_spec.yaml` and logged in each `manifest.json`.

## ✅ Acceptance Criteria

Before connecting modules, ensure each passes these criteria:

1. **00**: dataset summary matches expectations; examples parse
2. **01**: standardized JSONL ready; schema contract validated
3. **02**: unit tests pass; component rewards behave on edge cases
4. **03**: ≥90% valid JSON generations on val; losses stable
5. **04**: K candidates per item generated; parse rate ≥70%
6. **05/06/07**: reward increases on val; KL/variance controlled; fewer format errors than SFT
7. **08/09**: pairs have positive margins; DPO improves preference accuracy and EM vs SFT
8. **10**: test metrics reproducible; spot-checked against 02's scorer
9. **11**: report summarizes seeds/methods; conclusions supported by numbers

## 🛠 Development Workflow

### Daily Usage
```bash
# Activate environment
source activate.sh

# Run individual modules
python 00_check_data.py --data_root datasets/finqa/

# Or use the driver for full pipeline
python 12_driver.py --method ppo --run_id run_001
```

### Adding Dependencies
```bash
# Install new packages
pip install <package>

# Update requirements
pip freeze > requirements.txt
```

### Code Quality
```bash
# Format code
black *.py utils/ && isort *.py utils/

# Run linting
flake8 *.py utils/

# Run tests
python -m pytest tests/
```

## 📊 Expected Outcomes

This research aims to:

1. **Compare RL Methods**: Evaluate PPO, GRPO, RLOO, and DPO on FinQA mathematical reasoning
2. **Improve Accuracy**: Demonstrate measurable improvements over supervised fine-tuning baselines
3. **Analyze Stability**: Assess variance and robustness across different seeds and hyperparameters
4. **Efficiency Analysis**: Compare computational costs vs. performance gains

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Generation Error During Validation**

**Problem**: Error "probability tensor contains inf/nan" during validation after training.

**Symptoms**:
```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

**Workaround**: Use `--skip_validation` flag to skip validation during training:
```bash
python 03_sft_train.py --config configs/models/config_microsoft_DialoGPT_medium.yaml \
  --skip_validation
```

**Note**: The validation code exists and works for some configurations, but encounters CUDA errors with certain model/sampling combinations. The training itself works fine; only the validation sampling is affected.

**2. Out of Memory (OOM) Errors**

**Problem**: GPU runs out of memory during training.

**Solutions**:
- Reduce batch size: `--batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 16`
- Use smaller model (DialoGPT instead of Llama-3-8B)
- Enable mixed precision (should be enabled by default in configs)

**3. HuggingFace Authentication Required**

**Problem**: Cannot download Llama-3-8B model.

**Solution**:
1. Create HuggingFace account
2. Request access to Llama models
3. Create access token: https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login`

**4. Import Errors After Setup**

**Problem**: Cannot import modules like `transformers`, `torch`, etc.

**Solution**:
```bash
# Make sure you activated the virtual environment
source activate.sh

# Verify activation (should see (venv) in prompt)
which python  # Should point to venv/bin/python

# Reinstall if needed
./setup.sh --restart
```

**5. Dataset Download Fails**

**Problem**: `download.sh` fails or dataset incomplete.

**Solution**:
```bash
# Manual download
cd datasets/
git clone https://github.com/czyssrs/FinQA.git finqa

# Verify files exist
ls finqa/dataset/  # Should show train.json, dev.json, test.json
```

### Performance Tips

**For Faster Training**:
- Use DialoGPT-medium for development/testing
- Increase batch size if GPU memory allows
- Use `--quick_test` for fast sanity checks
- Enable mixed precision (FP16) in configs

**For Better Quality**:
- Use Llama-3-8B for final research results
- Increase LoRA rank (r: 32 instead of 16)
- Train for more epochs
- Use larger effective batch size

**For Stability**:
- Fix random seeds for reproducibility
- Use smaller learning rates for RL training
- Monitor KL divergence in RL methods
- Save checkpoints frequently

## 🤝 Contributing

1. Follow the module-first approach - each file should be self-contained
2. Maintain the JSON manifest system for reproducibility
3. Include comprehensive verification steps for each module
4. Use the provided configuration system for hyperparameters

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a research implementation. Each module should be thoroughly tested before connecting to the full pipeline.