#!/bin/bash
# Complete pipeline runner: Dataset preparation → SFT → PPO
# Runs steps 00-05 in sequence for full RL-LLM training

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  RL-LLM Complete Training Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Step 0: Check data
echo -e "${GREEN}[Step 0/5]${NC} Checking FinQA dataset..."
python 00_check_data.py
echo ""

# Step 1: Prepare dataset
echo -e "${GREEN}[Step 1/5]${NC} Preparing dataset..."
python 01_prepare_dataset.py
echo ""

# Step 2: Generate candidates
echo -e "${GREEN}[Step 2/5]${NC} Generating candidates (8 per question)..."
python 02_generate_candidates.py
echo ""

# Step 3: Build rewards
echo -e "${GREEN}[Step 3/5]${NC} Building rewards for candidates..."
python 03_build_rewards.py
echo ""

# Step 4: SFT training
echo -e "${GREEN}[Step 4/5]${NC} Starting SFT training..."
echo -e "${YELLOW}This will take several hours on 4x RTX 4090...${NC}"
python 04_sft_train.py --config configs/models/llama-3.2-3b.yaml
echo ""

# Step 5: PPO training
echo -e "${GREEN}[Step 5/5]${NC} Starting PPO training..."
echo -e "${YELLOW}This will take several hours...${NC}"
python 05_train_ppo.py --policy_ckpt outputs/run_001/04_sft_llama3b/best_model
echo ""

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ Complete Pipeline Finished!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Final models saved to:"
echo "  - SFT: outputs/run_001/04_sft_llama3b/"
echo "  - PPO: outputs/run_001/05_ppo/"
