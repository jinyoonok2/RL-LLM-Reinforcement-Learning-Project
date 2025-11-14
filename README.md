# RL-LLM Reinforcement Learning Project

**RL Pipeline (Module-First Plan) — FinQA → SFT → RL → Eval**

This project implements reinforcement learning methods for training Large Language Models (LLMs) on the **FinQA** dataset. The approach uses a **module-first** design where each component is a single Python file with clear responsibilities, inputs/outputs, and verification steps.

> **Research Goal**: Compare multiple RL methods (PPO, GRPO, RLOO, DPO) for improving mathematical reasoning in financial question-answering tasks.

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Project Structure](#-project-structure)
3. [Research Methodology](#-research-methodology)
4. [Module Specifications](#-module-specifications)
5. [Output Directory Structure](#-output-directory-structure)
6. [Model Configuration](#-model-configuration)
7. [Module Dependencies](#-module-dependencies)
8. [Development Workflow](#-development-workflow)
9. [Troubleshooting](#-troubleshooting)
10. [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Linux/WSL environment
- CUDA-capable GPU (recommended)

### Setup Instructions

1. **Clone and Setup Environment**:
   ```bash
   git clone https://github.com/jinyoonok2/RL-LLM-Reinforcement-Learning-Project.git
   cd RL-LLM-Reinforcement-Learning-Project
   
   # Create virtual environment and install dependencies
   ./setup.sh
   ```

2. **Activate Environment** (for daily development):
   ```bash
   source activate.sh           # Use 'source' to stay in the environment
   # You should see (venv) in your prompt
   ```

3. **Configure Settings**:
   ```bash
   cp .env.template .env
   # Edit .env with your HuggingFace tokens, WANDB settings, etc.
   ```

4. **Download Data & Models**:
   ```bash
   ./download.sh                # Dataset + DialoGPT-medium (863MB, for testing)
   ./download.sh --full-model   # Dataset + Llama-3-8B (15GB, for research)
   ```

5. **Test Installation**:
   ```bash
   python test_setup.py
   ```

### Script Options
- **`./setup.sh --restart`**: Remove existing environment and recreate from scratch
- **`./setup.sh --help`**: Show help message
- **`./activate.sh`**: Simple activation (use this for daily development)

**Note**: `setup.sh` only handles Python virtual environment and package installation. 
Make sure you have basic development tools installed on your system:
```bash
# Ubuntu/Debian/WSL:
sudo apt install python3 python3-venv python3-pip build-essential git
```

**Dataset Download**: The FinQA dataset is downloaded directly from the official GitHub repository using git clone for reliability.

## 📁 Project Structure

```
├── README.md                      # This comprehensive documentation
├── requirements.txt               # All Python dependencies (core + dev)
├── setup.sh                       # Create venv & install dependencies
├── activate.sh                    # Simple environment activation
├── .env.template                  # Environment variables template
├── .gitignore                     # Git ignore patterns
│
├── configs/                       # Configuration files
│   ├── schema.json                # FinQA output schema (defines expected JSON format)
│   ├── models/                    # Model-specific configs (architecture + training)
│   │   ├── config_microsoft_DialoGPT_medium.yaml
│   │   ├── config_TinyLlama_1.1B_Chat.yaml
│   │   └── config_meta_llama_Llama_3_8B_Instruct.yaml
│   └── algorithms/                # RL algorithm hyperparameters
│       ├── ppo.yaml               # PPO-specific settings
│       ├── grpo.yaml              # GRPO-specific settings
│       ├── rloo.yaml              # RLOO-specific settings
│       └── dpo.yaml               # DPO-specific settings
│
├── 00_check_data.py               # Data validation
├── 01_prepare_dataset.py          # Dataset preparation
├── 02_build_rewards.py            # Reward function implementation
├── 03_sft_train.py                # Supervised fine-tuning
├── 04_sample_candidates.py        # Candidate generation (TBD)
├── 05_train_ppo.py                # PPO training (TBD)
├── 06_train_grpo.py               # GRPO training (TBD)
├── 07_train_rloo.py               # RLOO training (TBD)
├── 08_build_prefs.py              # Preference pair construction (TBD)
├── 09_train_dpo.py                # DPO training (TBD)
├── 10_evaluate.py                 # Evaluation framework (TBD)
├── 11_compare_runs.py             # Results comparison (TBD)
├── 12_driver.py                   # Full pipeline orchestration (TBD)
├── inspect_model_architecture.py  # Model architecture inspection tool
│
├── utils/                         # Shared utilities
│   ├── evaluation.py              # ModelEvaluator for validation/testing
│   └── trainer.py                 # SFTTrainer for training loops
│
├── datasets/finqa/                # FinQA dataset files
├── outputs/                       # Experiment outputs (see Output Structure section)
│   ├── run_001/                   # First experimental run
│   ├── run_002/                   # Second experimental run
│   └── ...
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks for analysis
├── download.sh                    # Reliable download script (git clone + HF models)
└── test_setup.py                  # Setup validation script
```

## 🔬 Research Methodology

### Global Conventions

- **Dataset**: `datasets/finqa/` (original FinQA JSON files)
- **Preprocessed data**: `datasets/finqa_processed/` (prepared for training)
- **Dataset**: `datasets/finqa/` (original FinQA JSON files)
- **Preprocessed data**: `datasets/finqa_processed/` (prepared for training)
- **Experiment root**: `outputs/`
- **Run naming**: `run_001/`, `run_002/`, … (each full SFT+RL+Eval cycle)
- **Module outputs**: `outputs/run_XXX/YY_module_name/` (unified structure)
  - Example: `outputs/run_001/02_rewards/`, `outputs/run_001/03_sft/`
- **Seeds & determinism**: every module accepts `--seed`; all produced artifacts include seed/version info
- **JSON manifest**: every module writes `manifest.json` with params, seeds, git commit/hash, and version strings
- **Configs**: Two-tier system:
  - **Model configs** (`configs/models/*.yaml`): Architecture, LoRA, training params, paths
  - **Algorithm configs** (`configs/algorithms/*.yaml`): RL method hyperparameters only
  - Both optional; modules also accept CLI flags
- **Schema contract**: prompts and outputs follow a fixed **JSON schema** (`configs/schema.json`) for evaluation consistency

### Pipeline Overview

The research pipeline consists of 12 modules that can be run independently or orchestrated together:

## 📋 Module Specifications

### 00_check_data.py — Validate FinQA Inputs

**Goal**: Confirm FinQA is present and parsable; summarize splits and fields.

**Inputs**
- `--data_root datasets/finqa/`

**Saves (in `outputs/run_001/00_data_validation/`)**
- `summary.txt`: sample counts (train/val/test), missing fields, basic stats on answer types
- `examples/` (5–10 pretty-printed examples): `train_000.json`, `val_000.json`
- `manifest.json`: paths, timestamp, seed

**Verify**
- `summary.txt` shows expected split sizes (~8k total samples)
- Examples contain question, context reference(s), answer, and (if available) program steps
- Any malformed record count is reported (and < 1% of data)

---

### 01_prepare_dataset.py — Standardize Prompt/Output Items

**Goal**: Produce a normalized **prompt → expected JSON output** dataset the model will use for SFT and evaluation.

**Inputs**
- `--data_root datasets/finqa/`
- `--schema configs/schema.json` (defines required JSON fields; e.g., `{ "answer": <str/num>, "program": [ ... ] }`)
- `--split_proportions 0.8 0.1 0.1` (or use provided FinQA splits)

**Saves (in `outputs/finqa_rl/01_prepared/`)**
- `train.jsonl`, `val.jsonl`, `test.jsonl` (one example per line; fields: `id`, `prompt`, `target_json`)
- `schema_used.json`
- `manifest.json`

**Verify**
- Head/tail preview shows clean, self-contained prompts (no leakage of labels)
- Random 20-sample validation: `target_json` parses and matches schema
- Counts align with `00_data_check/summary.txt`

---

### 02_build_rewards.py — Programmatic Reward Functions

**Goal**: Implement scoring functions the RL methods will call.

**Inputs**
- `--schema configs/schema.json`
- `--weights "exact=1.0,program=0.3,format=0.2"` (example; keep weights configurable)

**Saves (in `outputs/finqa_rl/02_rewards/`)**
- `reward_spec.yaml`: exact-match, numeric tolerance, program execution validity rules, JSON format compliance
- `unit_tests/` (gold cases): input/output pairs with expected reward components
- `manifest.json`

**Verify**
- Run built-in unit tests: each test prints component scores and total reward
- JSON format violations yield non-crashing, zeroed (or penalized) rewards
- Numeric tolerance logic behaves as intended on provided corner cases

---

### 03_sft_train.py — Supervised Fine-Tuning (Base Policy)

**Goal**: Fine-tune a base LLM to emit **valid JSON** in the required schema.

**Inputs**
- `--train_jsonl outputs/finqa_rl/01_prepared/train.jsonl`
- `--val_jsonl outputs/finqa_rl/01_prepared/val.jsonl`
- `--base_model <hf_model_or_path>` (e.g., `meta-llama/Meta-Llama-3-8B-Instruct` or your local model)
- `--epochs 1-3`, `--lr`, `--batch_size`, `--max_len`, `--seed`

**Saves (in `outputs/finqa_rl/run_001/03_sft/`)**
- `ckpt_sft/` (tokenizer + model checkpoint or LoRA adapters)
- `logs/` (loss curves, validation metrics)
- `valid_samples/` (N model generations with parse status and schema diff)
- `manifest.json`

**Verify**
- At least 90% of `valid_samples` parse as JSON without errors
- Validation loss decreases; simple exact-match and format metrics improve over training
- Spot-check generations: structure matches schema; fields present and typed correctly

---

### 04_sample_candidates.py — Candidate Generation (for GRPO/RLOO/DPO)

**Goal**: Generate **K** candidate answers per prompt from the SFT (or current) policy for downstream RL or preference building.

**Inputs**
- `--policy_ckpt outputs/finqa_rl/run_001/03_sft/ckpt_sft/`
- `--split val` (or a subset list of IDs)
- `--num_candidates K` (e.g., 4–8), `--temperature`, `--top_p`, `--seed`

**Saves (in `outputs/finqa_rl/run_001/04_candidates/`)**
- `candidates.jsonl` (fields: `id`, `prompt`, `cands`: [ {`text`, `parsed_json`, `parse_ok`} x K ])
- `scores.jsonl` (optional: preliminary reward components per candidate via 02's reward functions)
- `manifest.json`

**Verify**
- Parse rate per candidate > 70% on SFT; invalid parses flagged but kept for analysis
- Score histogram is non-degenerate (variation across candidates)
- K is respected for all prompts (or failures recorded)

---

### 05_train_ppo.py — RL: Proximal Policy Optimization

**Goal**: Optimize policy with **clipped PPO** and **KL anchoring** to a reference (SFT).

**Inputs**
- `--policy_ckpt outputs/finqa_rl/run_001/03_sft/ckpt_sft/`
- `--reward_spec outputs/finqa_rl/02_rewards/reward_spec.yaml`
- `--train_jsonl outputs/finqa_rl/01_prepared/train.jsonl`
- PPO hyperparams: `--kl_coef`, `--clip_range`, `--rollout_batch`, `--mini_epochs`, `--seed`

**Saves (in `outputs/finqa_rl/run_001/05_ppo/`)**
- `ckpt_ppo/` (policy after RL)
- `logs/` (reward components, KL, clip frac, value stats if using critic)
- `val_generations/` (periodic eval generations with reward breakdown)
- `manifest.json`

**Verify**
- Mean total reward rises; KL stays within target band
- Format violation rate drops vs SFT
- Exact-match improves on a held-out val subset without mode collapse

---

### 06_train_grpo.py — RL: Group Relative Policy Optimization

**Goal**: **Critic-free** policy optimization by comparing groups of sampled outputs and pushing toward above-average samples.

**Inputs**
- `--policy_ckpt` (start from SFT or PPO)
- `--reward_spec` (same as 05)
- `--group_size G` (e.g., 4–8), `--batch_bonus on|off`, `--seed`
- Optionally reuse `04_candidates/candidates.jsonl` to warm-start

**Saves (in `outputs/finqa_rl/run_001/06_grpo/`)**
- `ckpt_grpo/`
- `logs/` (group mean vs winners, variance across seeds)
- `val_generations/`
- `manifest.json`

**Verify**
- Group winners' rewards consistently exceed group means
- With/without batch bonus ablation shows predictable differences
- Stability across seeds improves relative to PPO (variance narrows)

---

### 07_train_rloo.py — RL: REINFORCE Leave-One-Out

**Goal**: **Variance-reduced** critic-free updates using a leave-one-out baseline per candidate.

**Inputs**
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