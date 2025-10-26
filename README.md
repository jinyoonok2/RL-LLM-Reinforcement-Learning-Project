# RL-LLM Reinforcement Learning Project

**RL Pipeline (Module-First Plan) — FinQA → SFT → RL → Eval**

This project implements reinforcement learning methods for training Large Language Models (LLMs) on the **FinQA** dataset. The approach uses a **module-first** design where each component is a single Python file with clear responsibilities, inputs/outputs, and verification steps.

> **Research Goal**: Compare multiple RL methods (PPO, GRPO, RLOO, DPO) for improving mathematical reasoning in financial question-answering tasks.

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
├── README.md                  # This file
├── requirements.txt           # All Python dependencies (core + dev)
├── setup.sh                   # Create venv & install dependencies
├── activate.sh               # Simple environment activation
├── .env.template             # Environment variables template
├── .gitignore                # Git ignore patterns
│
├── configs/                  # Configuration files
│   ├── base.yaml            # Base configuration
│   ├── sft.yaml             # SFT training config
│   ├── ppo.yaml             # PPO training config
│   └── schema.json          # FinQA output schema
│
├── src/                     # Source code modules (to be implemented)
│   ├── 00_check_data.py     # Data validation
│   ├── 01_prepare_dataset.py # Dataset preparation
│   ├── 02_build_rewards.py  # Reward function implementation
│   ├── 03_sft_train.py      # Supervised fine-tuning
│   ├── 04_sample_candidates.py # Candidate generation
│   ├── 05_train_ppo.py      # PPO training
│   ├── 06_train_grpo.py     # GRPO training
│   ├── 07_train_rloo.py     # RLOO training
│   ├── 08_build_prefs.py    # Preference pair construction
│   ├── 09_train_dpo.py      # DPO training
│   ├── 10_evaluate.py       # Evaluation framework
│   ├── 11_compare_runs.py   # Results comparison
│   └── 12_driver.py         # Full pipeline orchestration
│
├── datasets/finqa/          # FinQA dataset files
├── outputs/finqa_rl/        # Experiment outputs
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for analysis
├── docs/                    # Documentation
├── download.sh              # Reliable download script (git clone + HF models)
└── test_setup.py            # Setup validation script
```

## 🔬 Research Methodology

### Global Conventions

- **Dataset**: `datasets/finqa/` (original FinQA JSON files)
- **Experiment root**: `outputs/finqa_rl/`
- **Run naming**: `run_001/`, `run_002/`, … (each full SFT+RL+Eval cycle)
- **Seeds & determinism**: every module accepts `--seed`; all produced artifacts include seed/version info
- **JSON manifest**: every module writes `<module>/manifest.json` with params, seeds, git commit/hash, and version strings
- **Configs**: optional YAMLs under `configs/`, but each module must also accept CLI flags
- **Schema contract**: prompts and outputs follow a fixed **JSON schema** for evaluation consistency

### Pipeline Overview

The research pipeline consists of 12 modules that can be run independently or orchestrated together:

## 📋 Module Specifications

### 00_check_data.py — Validate FinQA Inputs

**Goal**: Confirm FinQA is present and parsable; summarize splits and fields.

**Inputs**
- `--data_root datasets/finqa/`

**Saves (in `outputs/finqa_rl/00_data_check/`)**
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
python src/00_check_data.py --data_root datasets/finqa/

# Or use the driver for full pipeline
python src/12_driver.py --method ppo --run_id run_001
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
black src/ && isort src/

# Run linting
flake8 src/

# Run tests
python -m pytest tests/
```

## 📊 Expected Outcomes

This research aims to:

1. **Compare RL Methods**: Evaluate PPO, GRPO, RLOO, and DPO on FinQA mathematical reasoning
2. **Improve Accuracy**: Demonstrate measurable improvements over supervised fine-tuning baselines
3. **Analyze Stability**: Assess variance and robustness across different seeds and hyperparameters
4. **Efficiency Analysis**: Compare computational costs vs. performance gains

## 🤝 Contributing

1. Follow the module-first approach - each file should be self-contained
2. Maintain the JSON manifest system for reproducibility
3. Include comprehensive verification steps for each module
4. Use the provided configuration system for hyperparameters

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a research implementation. Each module should be thoroughly tested before connecting to the full pipeline.