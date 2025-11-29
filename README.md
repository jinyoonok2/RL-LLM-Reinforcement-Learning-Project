# RL-LLM Reinforcement Learning Project

**RL Pipeline for LLMs — FinQA → SFT → RL → Evaluation**

This project implements and compares multiple reinforcement learning methods for training Large Language Models on the **FinQA** dataset. The codebase uses a clean, modular design where each component is a single Python file with clear inputs/outputs and responsibilities.

> **Research Goal**: Compare RL methods (PPO, GRPO, RLOO, DPO) for improving mathematical reasoning in financial question-answering tasks.

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Project Structure](#-project-structure)
3. [Module Specifications](#-module-specifications)
4. [Model Configuration](#-model-configuration)
5. [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU with 24GB+ VRAM (for 8B models)
- Linux/WSL environment

### Setup Instructions

1. **Clone and Setup Environment**:
   ```bash
   git clone https://github.com/jinyoonok2/RL-LLM-Reinforcement-Learning-Project.git
   cd RL-LLM-Reinforcement-Learning-Project
   
   # Create virtual environment and install dependencies
   ./setup.sh
   ```

2. **Activate Environment**:
   ```bash
   source activate.sh           # Use 'source' to stay in the environment
   # You should see (venv) in your prompt
   ```

3. **Download Data & Models**:
   ```bash
   # Default: FinQA dataset + Llama-3.2-1B model (~2.5GB, fast testing)
   bash download.sh
   
   # For production 8B model:
   bash download.sh --model meta-llama/Meta-Llama-3-8B-Instruct
   ```

4. **Run the Pipeline**:
   ```bash
   # Step 1: Check data
   python 00_check_data.py
   
   # Step 2: Prepare dataset
   python 01_prepare_dataset.py
   
   # Step 3: Build rewards
   python 02_build_rewards.py
   
   # Step 4: SFT training (uses Llama-3.2-1B by default)
   python 03_sft_train.py
   
   # Step 5: Generate candidates
   python 04_generate_candidates.py --policy_ckpt outputs/run_001/03_sft
   
   # For 8B model training:
   python 03_sft_train.py --config configs/models/llama-3-8b.yaml
   ```

### Model Selection

**Ultra-Fast (DialoGPT)**: 863MB, ~20-30 min training
- Use: `--config configs/models/dialogpt-medium.yaml`
- Best for: Rapid prototyping and algorithm development

**Fast (TinyLlama)**: 2.2GB, ~40-60 min training  
- Use: `--config configs/models/tinyllama-1.1b.yaml`
- Best for: Testing Llama-style architectures quickly

**Balanced (Llama-3.2-1B)**: 2.5GB, ~60-90 min training
- Use: `--config configs/models/llama-3.2-1b.yaml` (default)
- Best for: Good balance of speed and quality

**Ultra-Fast Mode**: Any model with rank-8 LoRA
- Use: `--config configs/models/llama-3.2-1b-ultrafast.yaml`
- Best for: Lightning-fast iterations

## 📁 Project Structure

```
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── setup.sh                       # Environment setup
├── activate.sh                    # Environment activation
├── download.sh                    # Data & model download
│
├── configs/
│   ├── models/                    # Model configurations
│   │   ├── llama-3.2-1b.yaml      # Default: 1B fast model
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

## 🔬 Research Methodology

### Global Conventions

- **Dataset**: `datasets/finqa/` (original JSON)
- **Preprocessed**: `datasets/finqa_processed/` (JSONL format)
- **Outputs**: `outputs/run_XXX/YY_module/` structure
- **Models**: Default 1B for testing, 8B for production
- **Configs**: Model-specific YAML files in `configs/models/`
- **Determinism**: All modules accept `--seed` parameter
- **Manifests**: Each module writes `manifest.json` with parameters

### Pipeline Overview

1. **Data Processing** (00-01): Validate and prepare FinQA dataset
2. **Reward Setup** (02): Define reward functions for RL training
3. **SFT Training** (03): Supervised fine-tuning base policy
4. **Candidate Generation** (04): Sample multiple responses per prompt
5. **RL Training** (05-09): Train with PPO/GRPO/RLOO/DPO methods
6. **Evaluation** (10-11): Compare and analyze results

## 📋 Module Specifications

### 00_check_data.py — Data Validation ✅

Validates FinQA dataset integrity and structure.

```bash
python 00_check_data.py
```

**Outputs**: `outputs/run_001/00_data_validation/`
- Dataset statistics summary
- Sample examples
- Manifest with validation results

---

### 01_prepare_dataset.py — Dataset Preparation ✅

Converts FinQA to JSONL format with unified prompt/target structure.

```bash
python 01_prepare_dataset.py
```

**Outputs**: `datasets/finqa_processed/`
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- Fields: `id`, `input_text`, `target_answer`, `target_program`, `question`

---

### 02_build_rewards.py — Reward Functions ✅

Implements FinQA reward calculator (answer + program correctness).

```bash
python 02_build_rewards.py
```

**Outputs**: `outputs/run_001/02_rewards/`
- `reward_spec.yaml`: Reward configuration
- Test results showing reward components

**Reward Components**:
- Exact match bonus: +1.0
- Numeric match (5% tolerance): +0.8  
- Partial program match: +0.3
- Format penalty: -0.2

---

### 03_sft_train.py — Supervised Fine-Tuning ✅

Fine-tunes LLM to generate JSON responses with LoRA.

```bash
# Default: Llama-3.2-1B
python 03_sft_train.py

# Production: Llama-3-8B
python 03_sft_train.py --config configs/models/llama-3-8b.yaml
```

**Key Features**:
- LoRA training (0.34% trainable params)
- Batch size 1 + gradient accumulation 4
- Learning rate: 5e-6 (stable for LoRA)
- Validation every 500 steps

**Outputs**: `outputs/run_001/03_sft/`
- `ckpt_sft/`: Model checkpoints
- `valid_samples/`: Generated samples with rewards
- Training logs with loss curves

---

### 04_generate_candidates.py — Candidate Generation ✅

Generates K diverse candidate responses for RL training.

```bash
python 04_generate_candidates.py --policy_ckpt outputs/run_001/03_sft
```

**Parameters**:
- `--num_candidates`: Number of samples per prompt (default: 4)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Nucleus sampling (default: 0.9)

**Outputs**: `outputs/run_001/04_candidates/`
- `candidates.jsonl`: K responses per prompt
- `scores.jsonl`: Reward scores for each candidate

---

### 05_train_ppo.py — PPO Training 🚧

**Status**: In progress

Proximal Policy Optimization with KL penalty.

```bash
python 05_train_ppo.py --policy_ckpt outputs/run_001/03_sft
```

---

### 06-09: Additional RL Methods 🚧

**Status**: Planned

- **06_train_grpo.py**: Group Relative Policy Optimization
- **07_train_rloo.py**: REINFORCE Leave-One-Out
- **08_build_prefs.py**: Preference pair construction
- **09_train_dpo.py**: Direct Preference Optimization

---

### 10-11: Evaluation 🚧

**Status**: Planned

- **10_evaluate.py**: Unified evaluation framework
- **11_compare_runs.py**: Cross-method comparison

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
# Authenticate with HuggingFace
huggingface-cli login

# Accept Llama license at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

**3. Training Loss is NaN**
- Already fixed with learning_rate=5e-6
- If still occurs, lower to 1e-6

**4. Parse Rate Low (<50%)**
- SFT needs more epochs or better prompt format
- Check validation samples in `outputs/run_001/03_sft/valid_samples/`

**5. Zero Rewards**
- Verify reward function: `python 02_build_rewards.py`
- Check JSON format in generated samples

### Getting Help

- Check module outputs in `outputs/run_001/`
- Review `manifest.json` files for configuration
- Examine validation samples for model behavior

---

**Last Updated**: November 2025  
**Status**: Modules 00-04 complete, 05-11 in development
- `--policy_ckpt`
- `--reward_spec`
- `--num_samples K` (per prompt), `--seed`

**Saves (in `outputs/finqa_rl/run_001/07_rloo/`)**
- `ckpt_rloo/`
- `logs/` (per-sample reward, LOO baselines, gradient norm)
- `val_generations/`
- `manifest.json`

**Verify**
- Unbiased estimator sanity checks pass (expected sign vs advantage)
- Reward variance and gradient variance lower than plain REINFORCE
- EM/format metrics rise vs SFT baseline

---

### 08_build_prefs.py — Preference Pairs for DPO

**Goal**: Construct (preferred, rejected) pairs automatically from candidate generations + reward rules.

**Inputs**
- `--candidates outputs/finqa_rl/run_001/04_candidates/candidates.jsonl`
- `--reward_spec outputs/finqa_rl/02_rewards/reward_spec.yaml`
- Pairing rules: e.g., prefer exact-match & valid JSON over non-match or invalid

**Saves (in `outputs/finqa_rl/run_001/08_prefs/`)**
- `pairs.jsonl` (fields: `id`, `prompt`, `chosen`, `rejected`, optional reward diffs)
- `stats.txt` (pair counts, margin histograms)
- `manifest.json`

**Verify**
- All pairs pass JSON parse checks; chosen has ≥ rejected reward
- Pair margins (reward(chosen) − reward(rejected)) show healthy spread
- At least N≥10k pairs for robust DPO (or log smaller scale)

---

### 09_train_dpo.py — Direct Preference Optimization

**Goal**: Train with DPO objective to increase probability of preferred outputs under KL regularization.

**Inputs**
- `--base_or_sft_ckpt`
- `--pairs outputs/finqa_rl/run_001/08_prefs/pairs.jsonl`
- `--beta` (temperature/regularization), `--seed`

**Saves (in `outputs/finqa_rl/run_001/09_dpo/`)**
- `ckpt_dpo/`
- `logs/` (DPO loss, KL to reference)
- `val_generations/`
- `manifest.json`

**Verify**
- Preference accuracy on a held-out preference set increases
- JSON validity remains high; EM improves vs SFT
- KL stays within intended range (no drift)

---

### 10_evaluate.py — Metrics on Held-Out Test

**Goal**: Compute end-to-end metrics on `test.jsonl` for any checkpoint.

**Inputs**
- `--policy_ckpt <path>` (e.g., `05_ppo/ckpt_ppo/` or `06_grpo/ckpt_grpo/` etc.)
- `--test_jsonl outputs/finqa_rl/01_prepared/test.jsonl`
- `--reward_spec outputs/finqa_rl/02_rewards/reward_spec.yaml`

**Saves (in `outputs/finqa_rl/run_001/10_eval/`)**
- `metrics.json`: exact match, numeric correctness, program validity, format compliance
- `breakdown.csv`: per-item metrics and failure reasons
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