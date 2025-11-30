#!/bin/bash
# Fair Model Comparison Script
# Trains all lightweight models with identical settings for comparison

set -e

echo "🏁 Starting Fair Model Comparison Experiment"
echo "============================================"

# Define models to compare
MODELS=(
    "dialogpt-medium:outputs/run_001/03_sft_dialogpt:outputs/run_001/05_ppo_dialogpt"
    "tinyllama-1.1b:outputs/run_001/03_sft_tinyllama:outputs/run_001/05_ppo_tinyllama" 
    "llama-3.2-1b:outputs/run_001/03_sft:outputs/run_001/05_ppo_llama32"
)

# Shared training parameters for fair comparison
SHARED_PARAMS="--seed 42 --max_train_samples 1000"  # Same seed and data size

echo "📊 Training Parameters:"
echo "  - Seed: 42 (for reproducibility)"
echo "  - Max samples: 1000 (for fair comparison)"
echo "  - Same FinQA dataset and reward function"
echo ""

# Train each model
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r config sft_output ppo_output <<< "$model_info"
    
    echo "🚀 Training: $config"
    echo "----------------------------------------"
    
    # SFT Training
    echo "📚 Step 1: Supervised Fine-Tuning"
    python 03_sft_train.py \
        --config "configs/models/${config}.yaml" \
        $SHARED_PARAMS
    
    # Generate Candidates  
    echo "🎯 Step 2: Generate Candidates"
    python 04_generate_candidates.py \
        --policy_ckpt "$sft_output" \
        $SHARED_PARAMS
    
    # PPO Training
    echo "🎪 Step 3: PPO Training"
    python 05_train_ppo.py \
        --policy_ckpt "$sft_output" \
        --config "configs/models/${config}.yaml" \
        --output_dir "$ppo_output" \
        $SHARED_PARAMS
    
    # Evaluate
    echo "📈 Step 4: Evaluation"
    python check_validation.py --model_path "$ppo_output"
    
    echo "✅ Completed: $config"
    echo ""
done

echo "🎉 All models trained! Results ready for comparison."
echo ""
echo "📊 Compare results with:"
echo "python 11_compare_runs.py \\"
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r config sft_output ppo_output <<< "$model_info"
    echo "    $ppo_output \\"
done
echo ""