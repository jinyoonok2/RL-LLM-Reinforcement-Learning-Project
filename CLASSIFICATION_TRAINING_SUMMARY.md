# Updated Classification Pipeline - Summary

## ✅ What Was Done

### 1. Created New `04_sft_train.py` (Classification-Based)
**Replaced the old generation-based training with candidate ranking/selection.**

**Key Changes:**
- Uses `AutoModel` instead of `AutoModelForCausalLM` (encoder only, not generation)
- Added `CandidateRankingModel` class with scoring head
- Trains to **select best candidate** from pool of 8
- Loss: Cross-entropy on candidate selection
- Metrics: Accuracy + Reward Ratio

**Architecture:**
```
Question + 8 Candidates
    ↓
LLM Encoder (LoRA fine-tuned)
    ↓
Pooling (last token)
    ↓
Score Head (2-layer MLP)
    ↓
Scores [8 candidates]
    ↓
Select best (argmax)
```

**Training Loop:**
- Input: Batch of questions, each with 8 candidates
- Forward: Score all candidates
- Label: Index of highest-reward candidate
- Loss: CrossEntropyLoss(scores, best_idx)
- Optimize: Select candidates with high rewards

### 2. Deprecated Old Training Script
**Old generation-based script → `04_sft_train_generation.py.old`**

This preserves the original code for reference but is no longer the main approach.

### 3. Updated Documentation
**PIPELINE_CLASSIFICATION.md** now includes:
- Complete explanation of new classification training
- Architecture diagram
- Example training flow
- Benefits comparison

## 📊 Performance Expectations

### VRAM Usage:
- **Old (Generation)**: ~20-30GB training, ~60-70GB RL
- **New (Classification)**: ~15-20GB training, ~25-35GB RL
- **Savings**: 40-50% reduction

### Speed:
- **Old**: Token-by-token generation (slow)
- **New**: Single forward pass per candidate
- **Speedup**: 10-100x faster per step

### Stability:
- **Old**: Large action space (50k vocabulary), unstable gradients
- **New**: Small action space (8 choices), stable gradients
- **RL**: Much easier to train with PPO

## 🚀 Running the New Pipeline

### Full Commands:

```bash
# Step 00: Check data
python 00_check_data.py

# Step 01: Prepare dataset (1 answer per question)
python 01_prepare_dataset.py

# Step 02: Generate candidate pools (8 candidates per question)
python 02_generate_candidates.py \
  --input_dir datasets/finqa_processed \
  --output_dir datasets/finqa_candidates \
  --num_candidates 8 \
  --corruption_rate 0.5

# Step 03: Calculate rewards for all candidates
python 03_build_rewards.py \
  --input_dir datasets/finqa_candidates \
  --output_dir datasets/finqa_with_rewards

# Step 04: Train classification model (NEW!)
python 04_sft_train.py --config configs/models/llama-3-8b.yaml

# Quick test:
python 04_sft_train.py --config configs/models/llama-3-8b.yaml --quick_test

# Step 05: RL training (TODO - needs update for classification)
python 05_train_ppo.py --config configs/models/llama-3-8b.yaml
```

## 🔍 What's Different in Training

### Input Format:
**Old (Generation):**
```
Input: "Question: What is revenue? Answer:"
Output: Generate tokens → "{"answer": "1000", "program": "..."}"
```

**New (Classification):**
```
Input: [
  "Question: What is revenue?\nAnswer: 1000",
  "Question: What is revenue?\nAnswer: 1050",
  "Question: What is revenue?\nAnswer: Error",
  ... (8 total)
]
Output: Scores [2.1, 0.5, -1.2, ...] → Select index 0
```

### Loss Function:
**Old:**
```python
loss = CrossEntropyLoss(logits, next_token_ids)  # Per token
```

**New:**
```python
loss = CrossEntropyLoss(candidate_scores, best_candidate_idx)  # Per question
```

### Metrics:
**Old:**
- Perplexity
- Parse rate (% valid JSON)
- Reward (of generated answer)

**New:**
- **Accuracy**: % times model picks best candidate
- **Reward Ratio**: Actual reward / Max possible reward
- **Avg Reward**: Average reward of selected candidates

## 📝 Model Output Files

After training, you'll get:

```
outputs/run_001/04_sft/
├── best_model/
│   ├── adapter_config.json       # LoRA config
│   ├── adapter_model.bin         # LoRA weights
│   ├── score_head.pt            # Scoring head weights
│   └── tokenizer files
├── final_model/
│   └── (same structure)
└── training_manifest.json        # Training stats
```

## 🎯 Expected Results

With the new classification approach on 8B model:

**After Step 04 (SFT):**
- Accuracy: 60-80% (picks best candidate)
- Reward Ratio: 0.75-0.85 (gets 75-85% of max reward)
- Much faster training: ~30 min vs 2-3 hours

**After Step 05 (PPO - when implemented):**
- Accuracy: 80-90%
- Reward Ratio: 0.85-0.95
- Learns to explore and find better selections

## 🔄 Reverting to Old Approach

If you need to go back to generation-based training:

```bash
mv 04_sft_train.py 04_sft_train_classification.py
mv 04_sft_train_generation.py.old 04_sft_train.py
```

## 📚 Next Steps

1. ✅ Test `04_sft_train.py` on small dataset
2. ⏳ Run full training on 8B model
3. ⏳ Update `05_train_ppo.py` for classification RL
4. ⏳ Evaluate end-to-end performance

## 🐛 Troubleshooting

**Issue: "Data not found in datasets/finqa_with_rewards"**
- Solution: Run steps 02 and 03 first to generate candidates and rewards

**Issue: CUDA OOM**
- Solution: Reduce `batch_size` or `max_length` in config
- Classification uses less VRAM, so this should be rare

**Issue: Accuracy stuck at ~12.5% (random guessing)**
- Solution: Check that rewards are calculated correctly
- Ensure candidates have diverse rewards (not all 0)

**Issue: Model always picks candidate 0**
- Solution: Shuffle candidates during data loading
- Check label distribution in dataset
